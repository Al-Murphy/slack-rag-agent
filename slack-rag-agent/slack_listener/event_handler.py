from __future__ import annotations

import hashlib
import logging
from typing import Any

from database.vector_store import document_exists_by_hash, insert_paper_into_db
from processing.pdf_parser import parse_pdf_bytes
from processing.structurer import extract_structured_sections
from slack_listener.downloader import download_slack_file
from slack_listener.slack_client import fetch_file_info, slack_auth_headers

logger = logging.getLogger(__name__)


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
    if document_exists_by_hash(doc_hash):
        logger.info("Duplicate Slack document skipped file_id=%s hash=%s", file_id, doc_hash)
        return {
            "processed": True,
            "file_id": file_id,
            "doc_hash": doc_hash,
            "chunks_inserted": 0,
            "duplicate": True,
        }

    inserted = insert_paper_into_db(
        structured,
        doc_id=doc_id,
        doc_hash=doc_hash,
        source="slack",
        source_ref=file_info.get("permalink", file_id),
    )
    logger.info("Completed Slack ingestion file_id=%s doc_id=%s chunks=%d", file_id, doc_id, inserted)

    return {
        "processed": True,
        "file_id": file_id,
        "doc_hash": doc_hash,
        "chunks_inserted": inserted,
        "duplicate": False,
    }
