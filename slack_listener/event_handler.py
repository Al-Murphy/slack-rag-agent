from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any

from agent.fulltext_agent import ensure_full_text_for_paper
from agent.review_agent import review_ingested_document
from database.vector_store import (
    apply_review_updates,
    document_exists_by_hash,
    get_documents_by_doc_ids,
    get_related_documents_for_doc_ids,
    insert_paper_into_db,
)
from processing.link_resolver import _extract_arxiv_id, _extract_doi, paper_signal_score
from processing.pdf_parser import parse_pdf_bytes
from processing.structurer import extract_structured_sections
from slack_listener.downloader import download_slack_file
from slack_listener.slack_client import fetch_file_info, slack_auth_headers

logger = logging.getLogger(__name__)


def _canonical_paper_url_from_text(raw_text: str, file_info: dict[str, Any]) -> str:
    text_blob = "\n".join(
        [
            raw_text[:5000],
            str(file_info.get("title", "")),
            str(file_info.get("name", "")),
            str(file_info.get("initial_comment", {}).get("comment", "")),
        ]
    )

    doi = _extract_doi(text_blob)
    if doi:
        return f"https://doi.org/{doi}"

    arxiv_id = _extract_arxiv_id(text_blob)
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"

    return ""


def _looks_like_paper_text(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if not normalized:
        return False
    words = normalized.lower().split()
    if len(words) < 700:
        return False

    lowered = " ".join(words)
    markers = ("abstract", "introduction", "methods", "results", "discussion", "conclusion", "references")
    hits = sum(1 for m in markers if m in lowered)

    # Exclude obvious non-paper prose pages/docs.
    non_paper_signals = ("methods to watch", "news", "collection", "editorial", "podcast")
    if any(s in lowered for s in non_paper_signals) and hits < 3:
        return False

    return hits >= 2


async def _ingest_structured_document(
    *,
    doc_hash: str,
    doc_id: str,
    structured: dict[str, Any],
    source: str,
    source_ref: str,
    source_url: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if document_exists_by_hash(doc_hash):
        return {
            "processed": True,
            "doc_hash": doc_hash,
            "chunks_inserted": 0,
            "duplicate": True,
            **(extra or {}),
        }

    inserted = insert_paper_into_db(
        structured,
        doc_id=doc_id,
        doc_hash=doc_hash,
        source=source,
        source_ref=source_ref,
        source_url=source_url,
    )

    review_updates = {"documents": 0, "relations": 0}
    review_result: dict[str, Any] = {"review_confidence": 0.0, "issues": []}

    run_post_ingest_review = os.environ.get("ENABLE_POST_INGEST_REVIEW", "true").lower() in {"1", "true", "yes"}
    enable_review_db_updates = os.environ.get("ENABLE_REVIEW_DB_UPDATES", "true").lower() in {"1", "true", "yes"}
    if inserted > 0 and run_post_ingest_review:
        try:
            doc_lookup = get_documents_by_doc_ids([doc_id])
            related_map = get_related_documents_for_doc_ids([doc_id], per_doc_limit=12)
            related_docs = related_map.get(doc_id, [])
            doc_payload = doc_lookup.get(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": structured.get("title", ""),
                    "authors": structured.get("authors", []),
                    "source_url": source_url,
                    "source_ref": source_ref,
                    "source": source,
                    "tldr": structured.get("tldr", ""),
                    "paper_type": "",
                },
            )
            review_result = await review_ingested_document(
                doc=doc_payload,
                structured=structured,
                related_documents=related_docs,
            )
            review_payload = {
                "document_updates": review_result.get("document_updates", []),
                "connection_updates": review_result.get("connection_updates", []),
            }
            if enable_review_db_updates:
                review_updates = apply_review_updates(review_payload)
        except Exception:
            logger.exception("Post-ingest review failed doc_id=%s", doc_id)

    return {
        "processed": True,
        "doc_hash": doc_hash,
        "chunks_inserted": inserted,
        "duplicate": False,
        "post_ingest_review": {
            "enabled": run_post_ingest_review,
            "db_writeback_enabled": enable_review_db_updates,
            "review_confidence": float(review_result.get("review_confidence", 0.0) or 0.0),
            "issues": review_result.get("issues", []) if isinstance(review_result.get("issues", []), list) else [],
            "db_updates": {
                "documents": int(review_updates.get("documents", 0) or 0),
                "relations": int(review_updates.get("relations", 0) or 0),
            },
        },
        **(extra or {}),
    }


async def process_slack_file_id(file_id: str, source_ref: str | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    t_fetch_meta = t_download = t_parse_pdf = t_structure = t_insert = 0.0

    t0 = time.perf_counter()
    file_info = await asyncio.to_thread(fetch_file_info, file_id)
    t_fetch_meta += (time.perf_counter() - t0) * 1000

    logger.info("Processing Slack file_shared event file_id=%s", file_id)
    download_url = file_info.get("url_private_download") or file_info.get("url_private")
    if not download_url:
        return {"processed": False, "reason": "missing_download_url", "file_id": file_id}

    t1 = time.perf_counter()
    file_bytes = await asyncio.to_thread(download_slack_file, download_url, slack_auth_headers())
    t_download += (time.perf_counter() - t1) * 1000

    if file_info.get("mimetype") != "application/pdf":
        logger.warning("Skipping unsupported mimetype file_id=%s mimetype=%s", file_id, file_info.get("mimetype"))
        return {
            "processed": False,
            "reason": "unsupported_mimetype",
            "mimetype": file_info.get("mimetype"),
            "file_id": file_id,
        }

    # Early dedup check avoids parse + structuring for repeated files.
    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    if document_exists_by_hash(doc_hash):
        return {
            "processed": True,
            "file_id": file_id,
            "doc_hash": doc_hash,
            "chunks_inserted": 0,
            "duplicate": True,
            "timing_ms": {
                "total": round((time.perf_counter() - t_start) * 1000, 2),
                "fetch_metadata": round(t_fetch_meta, 2),
                "download": round(t_download, 2),
                "parse_pdf": 0.0,
                "structure": 0.0,
                "insert": 0.0,
            },
        }

    t2 = time.perf_counter()
    raw_text = await asyncio.to_thread(parse_pdf_bytes, file_bytes)
    t_parse_pdf += (time.perf_counter() - t2) * 1000

    if not _looks_like_paper_text(raw_text):
        return {
            "processed": False,
            "reason": "non_paper_file",
            "file_id": file_id,
            "timing_ms": {
                "total": round((time.perf_counter() - t_start) * 1000, 2),
                "fetch_metadata": round(t_fetch_meta, 2),
                "download": round(t_download, 2),
                "parse_pdf": round(t_parse_pdf, 2),
                "structure": 0.0,
                "insert": 0.0,
            },
        }

    t3 = time.perf_counter()
    structured = await extract_structured_sections(raw_text)
    t_structure += (time.perf_counter() - t3) * 1000

    doc_id = f"slack:{file_id}:{doc_hash[:12]}"
    t4 = time.perf_counter()
    canonical_source_url = _canonical_paper_url_from_text(raw_text, file_info)

    result = await _ingest_structured_document(
        doc_hash=doc_hash,
        doc_id=doc_id,
        structured=structured,
        source="slack",
        source_ref=source_ref or file_info.get("permalink", file_id),
        source_url=canonical_source_url,
        extra={"file_id": file_id, "source_url_inferred": bool(canonical_source_url)},
    )
    t_insert += (time.perf_counter() - t4) * 1000

    result["timing_ms"] = {
        "total": round((time.perf_counter() - t_start) * 1000, 2),
        "fetch_metadata": round(t_fetch_meta, 2),
        "download": round(t_download, 2),
        "parse_pdf": round(t_parse_pdf, 2),
        "structure": round(t_structure, 2),
        "insert": round(t_insert, 2),
    }

    logger.info(
        "Completed Slack PDF ingestion file_id=%s doc_id=%s chunks=%s duplicate=%s",
        file_id,
        doc_id,
        result.get("chunks_inserted"),
        result.get("duplicate"),
    )
    return result


async def process_slack_paper_url(url: str, source_ref: str, context_text: str = "") -> dict[str, Any]:
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    resolution = await ensure_full_text_for_paper(url=url, context_text=context_text)
    resolve_call_ms = (time.perf_counter() - t0) * 1000

    resolution_timing = resolution.get("timing_ms", {})
    if not resolution.get("ok"):
        return {
            "processed": False,
            "reason": resolution.get("reason", "full_text_not_found"),
            "url": url,
            "trace": resolution.get("trace", []),
            "timing_ms": {
                "total": round((time.perf_counter() - t_start) * 1000, 2),
                "resolve_total": round(resolve_call_ms, 2),
                "resolve_fetch": round(float(resolution_timing.get("fetch", 0.0)), 2),
                "resolve_parse_pdf": round(float(resolution_timing.get("parse_pdf", 0.0)), 2),
                "resolve_extract_html": round(float(resolution_timing.get("extract_html", 0.0)), 2),
                "structure": 0.0,
                "insert": 0.0,
            },
        }

    full_text = resolution["full_text"]
    paper_check = paper_signal_score(resolution.get("source_url", url), full_text)
    if not bool(paper_check.get("is_paper")):
        return {
            "processed": False,
            "reason": "non_paper_link",
            "url": url,
            "resolved_source_url": resolution.get("source_url", url),
            "resolved_source_kind": resolution.get("source_kind", "unknown"),
            "trace": resolution.get("trace", []),
            "paper_check": paper_check,
            "timing_ms": {
                "total": round((time.perf_counter() - t_start) * 1000, 2),
                "resolve_total": round(resolve_call_ms, 2),
                "resolve_fetch": round(float(resolution_timing.get("fetch", 0.0)), 2),
                "resolve_parse_pdf": round(float(resolution_timing.get("parse_pdf", 0.0)), 2),
                "resolve_extract_html": round(float(resolution_timing.get("extract_html", 0.0)), 2),
                "structure": 0.0,
                "insert": 0.0,
            },
        }

    doc_hash = resolution["content_hash"]
    doc_id = f"slack:url:{doc_hash[:12]}"

    # Early dedup check avoids expensive LLM structuring when content already indexed.
    if document_exists_by_hash(doc_hash):
        return {
            "processed": True,
            "doc_hash": doc_hash,
            "chunks_inserted": 0,
            "duplicate": True,
            "url": url,
            "resolved_source_url": resolution.get("source_url", url),
            "resolved_source_kind": resolution.get("source_kind", "unknown"),
            "trace": resolution.get("trace", []),
            "cache_hit": resolution.get("cache_hit", False),
            "timing_ms": {
                "total": round((time.perf_counter() - t_start) * 1000, 2),
                "resolve_total": round(resolve_call_ms, 2),
                "resolve_fetch": round(float(resolution_timing.get("fetch", 0.0)), 2),
                "resolve_parse_pdf": round(float(resolution_timing.get("parse_pdf", 0.0)), 2),
                "resolve_extract_html": round(float(resolution_timing.get("extract_html", 0.0)), 2),
                "structure": 0.0,
                "insert": 0.0,
            },
        }

    t1 = time.perf_counter()
    structured = await extract_structured_sections(full_text)
    structure_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    result = await _ingest_structured_document(
        doc_hash=doc_hash,
        doc_id=doc_id,
        structured=structured,
        source="slack_link",
        source_ref=source_ref,
        source_url=resolution.get("source_url", url),
        extra={
            "url": url,
            "resolved_source_url": resolution.get("source_url", url),
            "resolved_source_kind": resolution.get("source_kind", "unknown"),
            "trace": resolution.get("trace", []),
            "cache_hit": resolution.get("cache_hit", False),
        },
    )
    insert_ms = (time.perf_counter() - t2) * 1000

    result["timing_ms"] = {
        "total": round((time.perf_counter() - t_start) * 1000, 2),
        "resolve_total": round(resolve_call_ms, 2),
        "resolve_fetch": round(float(resolution_timing.get("fetch", 0.0)), 2),
        "resolve_parse_pdf": round(float(resolution_timing.get("parse_pdf", 0.0)), 2),
        "resolve_extract_html": round(float(resolution_timing.get("extract_html", 0.0)), 2),
        "structure": round(structure_ms, 2),
        "insert": round(insert_ms, 2),
    }

    logger.info(
        "Completed Slack URL ingestion url=%s resolved=%s chunks=%s duplicate=%s",
        url,
        result.get("resolved_source_url"),
        result.get("chunks_inserted"),
        result.get("duplicate"),
    )
    return result


async def handle_slack_event(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Handle Slack event payloads and process file_shared events.
    """
    event = payload.get("event", {})
    if event.get("type") != "file_shared":
        return {"processed": False, "reason": "unsupported_event"}

    file_id = event.get("file_id") or event.get("file", {}).get("id")
    if not file_id:
        return {"processed": False, "reason": "missing_file_id"}

    return await process_slack_file_id(file_id=file_id)
