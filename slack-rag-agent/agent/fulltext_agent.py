from __future__ import annotations

import hashlib
import logging
from typing import Any

import requests

from processing.link_resolver import (
    discover_fallback_sources,
    fetch_url_payload,
    is_probably_full_text,
)
from processing.pdf_parser import parse_pdf_bytes
from processing.text_extractor import extract_text_from_html

logger = logging.getLogger(__name__)


async def ensure_full_text_for_paper(url: str, context_text: str = "") -> dict[str, Any]:
    """
    Resolve a paper URL to usable full text.
    Strategy:
    1) direct URL
    2) candidate PDF/full-text links from page
    3) fallback sources (arXiv/DOI/Open-access sources)
    """
    trace: list[str] = []
    candidates = [url]
    candidates.extend(discover_fallback_sources(url=url, context_text=context_text))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            payload = fetch_url_payload(candidate)
        except Exception as exc:  # noqa: BLE001
            trace.append(f"fetch_error:{candidate}:{exc}")
            continue

        trace.append(f"fetched:{candidate}:{payload.get('kind')}")
        kind = payload.get("kind")
        if kind == "pdf":
            try:
                text = parse_pdf_bytes(payload["content_bytes"])
            except Exception as exc:  # noqa: BLE001
                trace.append(f"pdf_parse_error:{candidate}:{exc}")
                continue
            if is_probably_full_text(text):
                return {
                    "ok": True,
                    "source_url": payload.get("resolved_url", candidate),
                    "source_kind": "pdf",
                    "full_text": text,
                    "content_hash": hashlib.sha256(payload["content_bytes"]).hexdigest(),
                    "trace": trace,
                }
            trace.append(f"low_text_quality:{candidate}")
            continue

        if kind == "html":
            html = payload.get("html", "")
            extracted = extract_text_from_html(html)
            if is_probably_full_text(extracted):
                return {
                    "ok": True,
                    "source_url": payload.get("resolved_url", candidate),
                    "source_kind": "html",
                    "full_text": extracted,
                    "content_hash": hashlib.sha256(extracted.encode("utf-8")).hexdigest(),
                    "trace": trace,
                }
            trace.append(f"html_not_full_text:{candidate}")

    logger.warning("Could not resolve full text for url=%s", url)
    return {
        "ok": False,
        "reason": "full_text_not_found",
        "source_url": url,
        "trace": trace,
    }
