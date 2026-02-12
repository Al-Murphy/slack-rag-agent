from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from processing.link_resolver import (
    discover_fallback_sources,
    expand_source_url_candidates,
    fetch_url_payload,
    is_probably_full_text,
)
from processing.pdf_parser import parse_pdf_bytes
from processing.text_extractor import extract_text_from_html

logger = logging.getLogger(__name__)

_CACHE_MAX = int(os.environ.get("URL_RESOLUTION_CACHE_SIZE", "512"))
_URL_RESOLUTION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _url_cache_key(url: str) -> str:
    normalized = url.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cache_get(url_key: str) -> dict[str, Any] | None:
    item = _URL_RESOLUTION_CACHE.get(url_key)
    if item is None:
        return None
    _URL_RESOLUTION_CACHE.move_to_end(url_key)
    return deepcopy(item)


def _cache_set(url_key: str, value: dict[str, Any]) -> None:
    _URL_RESOLUTION_CACHE[url_key] = deepcopy(value)
    _URL_RESOLUTION_CACHE.move_to_end(url_key)
    while len(_URL_RESOLUTION_CACHE) > _CACHE_MAX:
        _URL_RESOLUTION_CACHE.popitem(last=False)


async def ensure_full_text_for_paper(url: str, context_text: str = "") -> dict[str, Any]:
    """
    Resolve a paper URL to usable full text.
    Strategy:
    1) known URL expansions for source domain (e.g. biorxiv .full/.full.pdf)
    2) direct URL and discovered PDF/full-text links from page
    3) fallback sources (arXiv/DOI/Open-access sources)
    """
    url_key = _url_cache_key(url)
    cached = _cache_get(url_key)
    if cached is not None:
        cached_trace = cached.get("trace", [])
        cached["trace"] = [f"cache_hit:{url_key[:12]}"] + cached_trace
        return cached

    trace: list[str] = []
    candidates = []
    candidates.extend(expand_source_url_candidates(url))
    candidates.extend(discover_fallback_sources(url=url, context_text=context_text))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            payload = await asyncio.to_thread(fetch_url_payload, candidate)
        except Exception as exc:  # noqa: BLE001
            trace.append(f"fetch_error:{candidate}:{exc}")
            continue

        trace.append(f"fetched:{candidate}:{payload.get('kind')}")
        kind = payload.get("kind")
        if kind == "pdf":
            try:
                text = await asyncio.to_thread(parse_pdf_bytes, payload["content_bytes"])
            except Exception as exc:  # noqa: BLE001
                trace.append(f"pdf_parse_error:{candidate}:{exc}")
                continue
            if is_probably_full_text(text):
                result = {
                    "ok": True,
                    "source_url": payload.get("resolved_url", candidate),
                    "source_kind": "pdf",
                    "full_text": text,
                    "content_hash": hashlib.sha256(payload["content_bytes"]).hexdigest(),
                    "trace": trace,
                }
                _cache_set(url_key, result)
                return result
            trace.append(f"low_text_quality:{candidate}")
            continue

        if kind == "html":
            html = payload.get("html", "")
            extracted = await asyncio.to_thread(extract_text_from_html, html)
            if is_probably_full_text(extracted):
                result = {
                    "ok": True,
                    "source_url": payload.get("resolved_url", candidate),
                    "source_kind": "html",
                    "full_text": extracted,
                    "content_hash": hashlib.sha256(extracted.encode("utf-8")).hexdigest(),
                    "trace": trace,
                }
                _cache_set(url_key, result)
                return result
            trace.append(f"html_not_full_text:{candidate}")

    logger.warning("Could not resolve full text for url=%s", url)
    miss = {
        "ok": False,
        "reason": "full_text_not_found",
        "source_url": url,
        "trace": trace,
    }
    _cache_set(url_key, miss)
    return miss
