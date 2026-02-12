from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Load .env before importing modules that read environment variables.
load_dotenv()

from agent.rag_agent import query_rag
from database.vector_store import get_state_value, init_db, set_state_value
from slack_listener.channel_crawler import ingest_channels
from slack_listener.event_handler import handle_slack_event

app = FastAPI(title="Slack RAG Agent MVP")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

INCREMENTAL_CRAWL_STATE_KEY = "last_incremental_crawl_ts"


class ChannelIngestRequest(BaseModel):
    channel_ids: list[str] = Field(default_factory=list)
    scan_all_accessible: bool = False
    days_back: int = Field(default=30, ge=1, le=365)
    per_channel_page_cap: int = Field(default=5, ge=1, le=50)
    top_k_files_per_channel: int = Field(default=50, ge=1, le=500)
    include_links: bool = True
    top_k_links_per_channel: int = Field(default=50, ge=1, le=500)
    link_concurrency_limit: int = Field(default=3, ge=1, le=20)


class IncrementalCrawlRequest(BaseModel):
    channel_ids: list[str] = Field(default_factory=list)
    include_links: bool = True
    per_channel_page_cap: int = Field(default=5, ge=1, le=50)
    top_k_files_per_channel: int = Field(default=50, ge=1, le=500)
    top_k_links_per_channel: int = Field(default=50, ge=1, le=500)
    link_concurrency_limit: int = Field(default=3, ge=1, le=20)
    initial_days_back: int = Field(default=1, ge=1, le=365)


@app.on_event("startup")
def startup() -> None:
    logger.info("Initializing database schema")
    init_db()
    logger.info("Startup complete")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return "<h1>Slack RAG Chat</h1><p>UI not found. Expected ui/index.html</p>"


def _verify_slack_signature(raw_body: bytes, headers: dict[str, str]) -> bool:
    secret = os.environ.get("SLACK_SIGNING_SECRET")
    if not secret:
        return True

    timestamp = headers.get("x-slack-request-timestamp", "")
    signature = headers.get("x-slack-signature", "")

    if not timestamp or not signature:
        return False

    # Reject very old requests to reduce replay risk.
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    sig_basestring = f"v0:{timestamp}:{raw_body.decode('utf-8')}"
    computed = "v0=" + hmac.new(secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


@app.post("/slack/events")
async def slack_events(req: Request) -> dict[str, Any]:
    raw_body = await req.body()
    header_map = {k.lower(): v for k, v in req.headers.items()}

    if not _verify_slack_signature(raw_body, header_map):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    data = await req.json()

    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge")}

    result = await handle_slack_event(data)
    logger.info("Slack event processed result=%s", result)
    return {"ok": True, "result": result}


@app.get("/query")
async def search(q: str, top_k: int = 5) -> dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    return await query_rag(q, top_k=top_k)


@app.post("/slack/ingest/channels")
async def ingest_from_channels(body: ChannelIngestRequest) -> dict[str, Any]:
    result = await ingest_channels(
        channel_ids=body.channel_ids,
        scan_all_accessible=body.scan_all_accessible,
        days_back=body.days_back,
        per_channel_page_cap=body.per_channel_page_cap,
        top_k_files_per_channel=body.top_k_files_per_channel,
        include_links=body.include_links,
        top_k_links_per_channel=body.top_k_links_per_channel,
        link_concurrency_limit=body.link_concurrency_limit,
    )
    logger.info("Channel crawler completed result_summary=%s", result.get("metrics"))
    return result


@app.get("/crawl/incremental/status")
def crawl_incremental_status() -> dict[str, Any]:
    last_run_ts = get_state_value(INCREMENTAL_CRAWL_STATE_KEY)
    return {"ok": True, "last_run_ts": last_run_ts}


@app.post("/crawl/incremental")
async def crawl_incremental(body: IncrementalCrawlRequest) -> dict[str, Any]:
    previous_run_ts = get_state_value(INCREMENTAL_CRAWL_STATE_KEY)
    oldest_ts_used = previous_run_ts or str(time.time() - (body.initial_days_back * 24 * 60 * 60))

    result = await ingest_channels(
        channel_ids=body.channel_ids,
        scan_all_accessible=False,
        days_back=body.initial_days_back,
        per_channel_page_cap=body.per_channel_page_cap,
        top_k_files_per_channel=body.top_k_files_per_channel,
        include_links=body.include_links,
        top_k_links_per_channel=body.top_k_links_per_channel,
        link_concurrency_limit=body.link_concurrency_limit,
        oldest_ts=oldest_ts_used,
    )

    new_run_ts = str(time.time())
    set_state_value(INCREMENTAL_CRAWL_STATE_KEY, new_run_ts)

    logger.info(
        "Incremental crawl complete previous_run_ts=%s new_run_ts=%s metrics=%s",
        previous_run_ts,
        new_run_ts,
        result.get("metrics"),
    )

    return {
        "ok": True,
        "previous_run_ts": previous_run_ts,
        "new_run_ts": new_run_ts,
        "oldest_ts_used": oldest_ts_used,
        "result": result,
    }
