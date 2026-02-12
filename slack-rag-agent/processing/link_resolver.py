from __future__ import annotations

import os
import re
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 30
USER_AGENT = os.environ.get(
    "PAPER_FETCH_USER_AGENT",
    "Mozilla/5.0 (compatible; SlackRAGBot/1.0; +https://example.local)",
)


def _requests_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def is_probably_full_text(text: str, min_words: int = 900) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return False
    words = normalized.split()
    if len(words) < min_words:
        return False

    # A lightweight paper-like heuristic.
    markers = ("abstract", "introduction", "methods", "results", "conclusion", "references")
    lowered = normalized.lower()
    marker_hits = sum(1 for m in markers if m in lowered)
    return marker_hits >= 2


def _extract_candidate_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(base_url, href)
        l = full.lower()
        if ".pdf" in l or "/pdf/" in l or "download" in l:
            candidates.append(full)
    return candidates


def fetch_url_payload(url: str) -> dict:
    resp = requests.get(url, headers=_requests_headers(), timeout=DEFAULT_TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    final_url = resp.url
    ctype = resp.headers.get("Content-Type", "").lower()

    if "application/pdf" in ctype or final_url.lower().endswith(".pdf"):
        return {
            "kind": "pdf",
            "resolved_url": final_url,
            "content_bytes": resp.content,
        }

    html = resp.text
    candidates = _extract_candidate_links(html=html, base_url=final_url)
    for candidate in candidates[:10]:
        try:
            pdf_resp = requests.get(candidate, headers=_requests_headers(), timeout=DEFAULT_TIMEOUT, allow_redirects=True)
            pdf_resp.raise_for_status()
            pdf_ctype = pdf_resp.headers.get("Content-Type", "").lower()
            if "application/pdf" in pdf_ctype or pdf_resp.url.lower().endswith(".pdf"):
                return {
                    "kind": "pdf",
                    "resolved_url": pdf_resp.url,
                    "content_bytes": pdf_resp.content,
                }
        except Exception:
            continue

    return {
        "kind": "html",
        "resolved_url": final_url,
        "html": html,
    }


def _extract_doi(text: str) -> str | None:
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", unquote(text))
    if not m:
        return None
    return m.group(1).rstrip(").,;")


def _extract_arxiv_id(text: str) -> str | None:
    m = re.search(r"arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", text, re.IGNORECASE)
    if m:
        return m.group(2)
    m2 = re.search(r"\b([0-9]{4}\.[0-9]{4,5})(v\d+)?\b", text)
    return m2.group(1) if m2 else None


def discover_fallback_sources(url: str, context_text: str = "") -> list[str]:
    candidates: list[str] = []

    arxiv_id = _extract_arxiv_id(url + " " + context_text)
    if arxiv_id:
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        candidates.append(f"https://arxiv.org/abs/{arxiv_id}")

    doi = _extract_doi(url + " " + context_text)
    if doi:
        candidates.append(f"https://doi.org/{doi}")

        # Optional OA fallback if user configures an email for Unpaywall.
        email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
        if email:
            try:
                api = f"https://api.unpaywall.org/v2/{doi}?email={email}"
                resp = requests.get(api, headers=_requests_headers(), timeout=DEFAULT_TIMEOUT)
                if resp.ok:
                    payload = resp.json()
                    best = payload.get("best_oa_location") or {}
                    if best.get("url_for_pdf"):
                        candidates.append(best["url_for_pdf"])
                    if best.get("url"):
                        candidates.append(best["url"])
                    for loc in payload.get("oa_locations", []):
                        if loc.get("url_for_pdf"):
                            candidates.append(loc["url_for_pdf"])
                        if loc.get("url"):
                            candidates.append(loc["url"])
            except Exception:
                pass

    # de-duplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
