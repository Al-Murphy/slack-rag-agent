from __future__ import annotations

import hashlib
from typing import Any

from processing.pdf_parser import parse_pdf_bytes
from processing.structurer import extract_structured_sections
from database.vector_store import insert_paper_into_db
from slack_listener.downloader import download_slack_file
from slack_listener.slack_client import fetch_file_info, slack_auth_headers


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
    download_url = file_info.get("url_private_download") or file_info.get("url_private")
    if not download_url:
        return {"processed": False, "reason": "missing_download_url", "file_id": file_id}

    file_bytes = download_slack_file(download_url, slack_auth_headers())

    if file_info.get("mimetype") != "application/pdf":
        return {
            "processed": False,
            "reason": "unsupported_mimetype",
            "mimetype": file_info.get("mimetype"),
            "file_id": file_id,
        }

    raw_text = parse_pdf_bytes(file_bytes)
    structured = await extract_structured_sections(raw_text)

    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    inserted = insert_paper_into_db(structured, doc_id=f"slack:{file_id}:{doc_hash[:12]}")

    return {
        "processed": True,
        "file_id": file_id,
        "doc_hash": doc_hash,
        "chunks_inserted": inserted,
    }
