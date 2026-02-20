from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env before importing modules that read environment variables.
load_dotenv()

from agent.rag_agent import query_rag
from agent.prompts import DEFAULT_PERSONA_PROFILE, refine_persona_profile
from agent.testing_agent import run_retrieval_eval
from database.vector_store import clear_database, get_database_stats, get_paper_graph_data, get_state_value, init_db, set_state_value
from slack_listener.channel_crawler import ingest_channels
from slack_listener.event_handler import handle_slack_event

app = FastAPI(title="Slack RAG Agent MVP")
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "ui"), name="ui")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

INCREMENTAL_CRAWL_STATE_KEY = "last_incremental_crawl_ts"
PERSONA_PROFILE_STATE_KEY = "chat_persona_profile"
PERSONA_ENABLED_STATE_KEY = "chat_persona_enabled"
CRAWL_JOBS: dict[str, dict[str, Any]] = {}


class ChannelIngestRequest(BaseModel):
    channel_ids: list[str] = Field(default_factory=list)
    scan_all_accessible: bool = False
    days_back: int = Field(default=30, ge=1, le=3650)
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
    initial_days_back: int = Field(default=1, ge=1, le=3650)


class ClearDbRequest(BaseModel):
    confirm_phrase: str


class AsyncChannelIngestRequest(BaseModel):
    channel_ids: list[str] = Field(default_factory=list)
    scan_all_accessible: bool = False
    days_back: int = Field(default=30, ge=1, le=3650)
    per_channel_page_cap: int = Field(default=5, ge=1, le=50)
    top_k_files_per_channel: int = Field(default=50, ge=1, le=500)
    include_links: bool = True
    top_k_links_per_channel: int = Field(default=50, ge=1, le=500)
    link_concurrency_limit: int = Field(default=3, ge=1, le=20)


class RetrievalEvalRequest(BaseModel):
    channel_id: str | None = None
    days_back: int = Field(default=30, ge=1, le=3650)
    paraphrase_count: int = Field(default=5, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=20)


class PersonaUpdateRequest(BaseModel):
    profile: str = Field(default="")
    enabled: bool = True


class PersonaRefineRequest(BaseModel):
    instructions: str = Field(min_length=1)
    enabled: bool | None = None


@app.on_event("startup")
def startup() -> None:
    logger.info("Initializing database schema")
    init_db()

    if get_state_value(PERSONA_PROFILE_STATE_KEY) is None:
        set_state_value(PERSONA_PROFILE_STATE_KEY, DEFAULT_PERSONA_PROFILE)
    if get_state_value(PERSONA_ENABLED_STATE_KEY) is None:
        set_state_value(PERSONA_ENABLED_STATE_KEY, "true")

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


async def _run_channel_ingest_job(job_id: str, body: AsyncChannelIngestRequest) -> None:
    job = CRAWL_JOBS[job_id]
    job["status"] = "running"
    job["started_at"] = time.time()

    def _on_progress(update: dict[str, Any]) -> None:
        processed = int(update.get("processed_channels", 0) or 0)
        total = int(update.get("total_channels", 0) or 0)
        pct = 0.0 if total <= 0 else round((processed / total) * 100, 2)
        job["processed_channels"] = processed
        job["total_channels"] = total
        job["percent"] = pct
        job["current_channel_id"] = update.get("current_channel_id")

    try:
        result = await ingest_channels(
            channel_ids=body.channel_ids,
            scan_all_accessible=body.scan_all_accessible,
            days_back=body.days_back,
            per_channel_page_cap=body.per_channel_page_cap,
            top_k_files_per_channel=body.top_k_files_per_channel,
            include_links=body.include_links,
            top_k_links_per_channel=body.top_k_links_per_channel,
            link_concurrency_limit=body.link_concurrency_limit,
            progress_callback=_on_progress,
        )
        job["result"] = result
        job["status"] = "completed"
        job["percent"] = 100.0
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)
        logger.exception("Async ingest job failed job_id=%s", job_id)
    finally:
        job["finished_at"] = time.time()


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

    persona_profile = get_state_value(PERSONA_PROFILE_STATE_KEY) or DEFAULT_PERSONA_PROFILE
    persona_enabled_raw = (get_state_value(PERSONA_ENABLED_STATE_KEY) or "true").strip().lower()
    persona_enabled = persona_enabled_raw in {"1", "true", "yes", "on"}

    return await query_rag(
        q,
        top_k=top_k,
        persona_profile=persona_profile,
        persona_enabled=persona_enabled,
    )


@app.post("/eval/retrieval")
async def eval_retrieval(body: RetrievalEvalRequest) -> dict[str, Any]:
    channel_id = (body.channel_id or "").strip() or None
    return await run_retrieval_eval(
        channel_id=channel_id,
        days_back=body.days_back,
        paraphrase_count=body.paraphrase_count,
        top_k=body.top_k,
    )


@app.get("/db/stats")
def db_stats() -> dict[str, Any]:
    return {"ok": True, "stats": get_database_stats()}


@app.get("/papers/graph")
def papers_graph(max_docs: int = 300, min_score: float = 0.0) -> dict[str, Any]:
    if max_docs < 20 or max_docs > 2000:
        raise HTTPException(status_code=400, detail="max_docs must be between 20 and 2000")
    if min_score < 0.0 or min_score > 1.0:
        raise HTTPException(status_code=400, detail="min_score must be between 0.0 and 1.0")

    graph = get_paper_graph_data(max_docs=max_docs, min_score=min_score)
    return {"ok": True, "graph": graph}


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



@app.post("/slack/ingest/channels/async")
async def ingest_from_channels_async(body: AsyncChannelIngestRequest) -> dict[str, Any]:
    job_id = str(uuid4())
    CRAWL_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "processed_channels": 0,
        "total_channels": 0,
        "percent": 0.0,
        "current_channel_id": None,
        "request": body.model_dump() if hasattr(body, "model_dump") else body.dict(),
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_channel_ingest_job(job_id, body))
    return {"ok": True, "job_id": job_id}


@app.get("/slack/ingest/channels/async/{job_id}")
def ingest_from_channels_async_status(job_id: str) -> dict[str, Any]:
    job = CRAWL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {"ok": True, "job": job}


@app.post("/admin/clear-db")
def admin_clear_db(body: ClearDbRequest) -> dict[str, Any]:
    if os.environ.get("ALLOW_DB_CLEAR", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(status_code=403, detail="DB clear is disabled. Set ALLOW_DB_CLEAR=true in .env and restart the server.")

    required_phrase = os.environ.get("DB_CLEAR_CONFIRM_PHRASE", "CLEAR DB")
    if body.confirm_phrase.strip() != required_phrase:
        raise HTTPException(status_code=400, detail="Invalid confirmation phrase")

    deleted = clear_database()
    logger.warning("Database cleared via admin endpoint deleted=%s", deleted)
    return {"ok": True, "deleted": deleted}


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


@app.get("/agent/persona")
def get_agent_persona() -> dict[str, Any]:
    profile = get_state_value(PERSONA_PROFILE_STATE_KEY) or DEFAULT_PERSONA_PROFILE
    enabled_raw = (get_state_value(PERSONA_ENABLED_STATE_KEY) or "true").strip().lower()
    enabled = enabled_raw in {"1", "true", "yes", "on"}
    return {"ok": True, "persona": {"profile": profile, "enabled": enabled}}


@app.put("/agent/persona")
def update_agent_persona(body: PersonaUpdateRequest) -> dict[str, Any]:
    profile = (body.profile or "").strip() or DEFAULT_PERSONA_PROFILE
    set_state_value(PERSONA_PROFILE_STATE_KEY, profile)
    set_state_value(PERSONA_ENABLED_STATE_KEY, "true" if body.enabled else "false")
    return {"ok": True, "persona": {"profile": profile, "enabled": bool(body.enabled)}}


@app.post("/agent/persona/refine")
async def refine_agent_persona(body: PersonaRefineRequest) -> dict[str, Any]:
    current = get_state_value(PERSONA_PROFILE_STATE_KEY) or DEFAULT_PERSONA_PROFILE
    refined = await refine_persona_profile(current, body.instructions)
    enabled = body.enabled
    if enabled is None:
        enabled_raw = (get_state_value(PERSONA_ENABLED_STATE_KEY) or "true").strip().lower()
        enabled = enabled_raw in {"1", "true", "yes", "on"}

    set_state_value(PERSONA_PROFILE_STATE_KEY, refined)
    set_state_value(PERSONA_ENABLED_STATE_KEY, "true" if enabled else "false")

    return {"ok": True, "persona": {"profile": refined, "enabled": bool(enabled)}}
