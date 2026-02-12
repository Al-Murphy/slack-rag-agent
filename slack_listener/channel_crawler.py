from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def parse_channel_targets(channel_ids: list[str] | None = None) -> list[str]:
    if channel_ids:
        return [c.strip() for c in channel_ids if c.strip()]
    configured = os.environ.get("SLACK_CHANNEL_IDS", "")
    return [c.strip() for c in configured.split(",") if c.strip()]


def extract_pdf_file_ids(messages: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for message in messages:
        files = message.get("files", [])
        for f in files:
            file_id = f.get("id")
            mimetype = f.get("mimetype", "")
            if not file_id:
                continue
            if mimetype == "application/pdf" or f.get("filetype") == "pdf":
                if file_id not in seen:
                    seen.add(file_id)
                    output.append(file_id)
    return output


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    # Slack format: <https://example.com|label>
    for raw in re.findall(r"<(https?://[^>|]+)(?:\|[^>]+)?>", text):
        urls.append(raw.strip())
    scrubbed = re.sub(r"<https?://[^>]+>", " ", text)
    # plain links
    for raw in re.findall(r"(https?://[^\s>]+)", scrubbed):
        clean = raw.rstrip("),.;")
        urls.append(clean)
    return urls


def _is_candidate_paper_url(url: str) -> bool:
    u = url.lower()
    if ".pdf" in u:
        return True
    domains = (
        "arxiv.org",
        "doi.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "springer.com",
        "frontiersin.org",
        "acm.org",
        "ieee.org",
        "jamanetwork.com",
        "thelancet.com",
        "biorxiv.org",
        "medrxiv.org",
        "plos.org",
        "cell.com",
    )
    return any(d in u for d in domains)


def extract_paper_urls(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for message in messages:
        raw_text = " ".join(
            [
                message.get("text", ""),
                message.get("blocks", [{}])[0].get("text", {}).get("text", ""),
            ]
        ).strip()
        for url in _extract_urls(raw_text):
            if not _is_candidate_paper_url(url):
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append({"url": url, "message_ts": message.get("ts", ""), "message_text": raw_text[:3000]})
    return out


def _list_all_accessible_channels(limit_per_page: int = 200) -> list[str]:
    from slack_listener.slack_client import list_channels

    channel_ids: list[str] = []
    cursor: str | None = None

    while True:
        resp = list_channels(limit=limit_per_page, cursor=cursor)
        channels = resp.get("channels", [])
        channel_ids.extend(ch.get("id") for ch in channels if ch.get("id"))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channel_ids


def _channel_history_pages(channel_id: str, oldest_ts: str | None, limit: int = 200, page_cap: int = 10) -> list[dict[str, Any]]:
    from slack_listener.slack_client import get_channel_history

    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(page_cap):
        resp = get_channel_history(channel_id=channel_id, oldest_ts=oldest_ts, limit=limit, cursor=cursor)
        pages.append(resp)
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return pages


async def ingest_channels(
    channel_ids: list[str] | None = None,
    scan_all_accessible: bool = False,
    days_back: int = 30,
    per_channel_page_cap: int = 10,
    top_k_files_per_channel: int = 50,
    include_links: bool = True,
    top_k_links_per_channel: int = 50,
) -> dict[str, Any]:
    from slack_listener.event_handler import process_slack_file_id, process_slack_paper_url

    started = time.time()
    targets = parse_channel_targets(channel_ids)
    if scan_all_accessible:
        targets = _list_all_accessible_channels()
    if not targets:
        return {"ok": False, "reason": "no_channels_configured", "processed": 0}

    oldest_ts = str(time.time() - (days_back * 24 * 60 * 60))
    results: list[dict[str, Any]] = []
    by_channel: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "pdf_discovered": 0,
            "link_discovered": 0,
            "ingested": 0,
            "duplicates": 0,
            "errors": 0,
        }
    )

    for channel_id in targets:
        try:
            pages = _channel_history_pages(
                channel_id=channel_id,
                oldest_ts=oldest_ts,
                page_cap=per_channel_page_cap,
            )
            messages = []
            for page in pages:
                messages.extend(page.get("messages", []))
            file_ids = extract_pdf_file_ids(messages)[:top_k_files_per_channel]
            by_channel[channel_id]["pdf_discovered"] = len(file_ids)
            paper_links = extract_paper_urls(messages)[:top_k_links_per_channel] if include_links else []
            by_channel[channel_id]["link_discovered"] = len(paper_links)

            for file_id in file_ids:
                try:
                    r = await process_slack_file_id(file_id=file_id, source_ref=f"channel:{channel_id}")
                    results.append(r)
                    if r.get("duplicate"):
                        by_channel[channel_id]["duplicates"] += 1
                    elif r.get("processed"):
                        by_channel[channel_id]["ingested"] += 1
                    else:
                        by_channel[channel_id]["errors"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to process file_id=%s in channel=%s", file_id, channel_id)
                    by_channel[channel_id]["errors"] += 1
                    results.append({"processed": False, "file_id": file_id, "error": str(exc), "channel_id": channel_id})
                    await asyncio.sleep(0.1)

            for link in paper_links:
                url = link["url"]
                try:
                    r = await process_slack_paper_url(
                        url=url,
                        source_ref=f"channel:{channel_id}:{link.get('message_ts', '')}",
                        context_text=link.get("message_text", ""),
                    )
                    results.append(r)
                    if r.get("duplicate"):
                        by_channel[channel_id]["duplicates"] += 1
                    elif r.get("processed"):
                        by_channel[channel_id]["ingested"] += 1
                    else:
                        by_channel[channel_id]["errors"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to process paper URL=%s in channel=%s", url, channel_id)
                    by_channel[channel_id]["errors"] += 1
                    results.append({"processed": False, "url": url, "error": str(exc), "channel_id": channel_id})
                    await asyncio.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed channel scan channel_id=%s", channel_id)
            by_channel[channel_id]["errors"] += 1
            results.append({"processed": False, "channel_id": channel_id, "error": str(exc)})

    duration = round((time.time() - started) * 1000, 2)
    total_ingested = sum(v["ingested"] for v in by_channel.values())
    total_duplicates = sum(v["duplicates"] for v in by_channel.values())
    total_errors = sum(v["errors"] for v in by_channel.values())

    return {
        "ok": True,
        "scan_all_accessible": scan_all_accessible,
        "channels_scanned": len(targets),
        "metrics": {
            "ingested": total_ingested,
            "duplicates": total_duplicates,
            "errors": total_errors,
            "duration_ms": duration,
        },
        "by_channel": by_channel,
        "results": results,
    }
