from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from agent.rag_agent import query_rag
from database.vector_store import init_db
from slack_listener.event_handler import handle_slack_event

load_dotenv()

app = FastAPI(title="Slack RAG Agent MVP")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    return {"ok": True, "result": result}


@app.get("/query")
async def search(q: str, top_k: int = 5) -> dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    return await query_rag(q, top_k=top_k)
