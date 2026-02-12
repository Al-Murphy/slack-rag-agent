from __future__ import annotations

import hashlib
import logging
from typing import Any

from agent.fulltext_agent import ensure_full_text_for_paper
from database.vector_store import document_exists_by_hash, insert_paper_into_db
from processing.pdf_parser import parse_pdf_bytes
from processing.structurer import extract_structured_sections
from slack_listener.downloader import download_slack_file
from slack_listener.slack_client import fetch_file_info, slack_auth_headers

logger = logging.getLogger(__name__)


async def _ingest_structured_document(
    *,
    doc_hash: str,
    doc_id: str,
    structured: dict[str, Any],
    source: str,
    source_ref: str,
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
    )
    return {
        "processed": True,
        "doc_hash": doc_hash,
        "chunks_inserted": inserted,
        "duplicate": False,
        **(extra or {}),
    }


async def process_slack_file_id(file_id: str, source_ref: str | None = None) -> dict[str, Any]:
    file_info = fetch_file_info(file_id)
    logger.info("Processing Slack file_shared event file_id=%s", file_id)
    download_url = file_info.get("url_private_download") or file_info.get("url_private")
    if not download_url:
        return {"processed": False, "reason": "missing_download_url", "file_id": file_id}

    file_bytes = download_slack_file(download_url, slack_auth_headers())

    if file_info.get("mimetype") != "application/pdf":
        logger.warning("Skipping unsupported mimetype file_id=%s mimetype=%s", file_id, file_info.get("mimetype"))
        return {
            "processed": False,
            "reason": "unsupported_mimetype",
            "mimetype": file_info.get("mimetype"),
            "file_id": file_id,
        }

    raw_text = parse_pdf_bytes(file_bytes)
    structured = await extract_structured_sections(raw_text)

    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    doc_id = f"slack:{file_id}:{doc_hash[:12]}"
    result = await _ingest_structured_document(
        doc_hash=doc_hash,
        doc_id=doc_id,
        structured=structured,
        source="slack",
        source_ref=source_ref or file_info.get("permalink", file_id),
        extra={"file_id": file_id},
    )
    logger.info(
        "Completed Slack PDF ingestion file_id=%s doc_id=%s chunks=%s duplicate=%s",
        file_id,
        doc_id,
        result.get("chunks_inserted"),
        result.get("duplicate"),
    )
    return result


async def process_slack_paper_url(url: str, source_ref: str, context_text: str = "") -> dict[str, Any]:
    resolution = await ensure_full_text_for_paper(url=url, context_text=context_text)
    if not resolution.get("ok"):
        return {
            "processed": False,
            "reason": resolution.get("reason", "full_text_not_found"),
            "url": url,
            "trace": resolution.get("trace", []),
        }

    full_text = resolution["full_text"]
    doc_hash = resolution["content_hash"]
    doc_id = f"slack:url:{doc_hash[:12]}"
    structured = await extract_structured_sections(full_text)

    result = await _ingest_structured_document(
        doc_hash=doc_hash,
        doc_id=doc_id,
        structured=structured,
        source="slack_link",
        source_ref=source_ref,
        extra={
            "url": url,
            "resolved_source_url": resolution.get("source_url", url),
            "resolved_source_kind": resolution.get("source_kind", "unknown"),
            "trace": resolution.get("trace", []),
        },
    )
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
