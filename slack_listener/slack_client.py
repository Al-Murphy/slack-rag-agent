from __future__ import annotations

import os
import ssl
from typing import Any

import certifi
from slack_sdk import WebClient


def _slack_client() -> WebClient:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    return WebClient(token=os.environ.get("SLACK_BOT_TOKEN", ""), ssl=ssl_context)


def fetch_file_info(file_id: str) -> dict:
    """Fetch metadata for a file stored in Slack."""
    result = _slack_client().files_info(file=file_id)
    return result.get("file", {})


def slack_auth_headers() -> dict:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    return {"Authorization": f"Bearer {token}"}


def list_channels(limit: int = 200, cursor: str | None = None) -> dict[str, Any]:
    """List Slack channels visible to the bot."""
    return _slack_client().conversations_list(
        limit=limit,
        cursor=cursor,
        types="public_channel,private_channel",
        exclude_archived=True,
    )


def get_channel_history(
    channel_id: str,
    oldest_ts: str | None = None,
    limit: int = 200,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Fetch message history for a channel."""
    kwargs: dict[str, Any] = {
        "channel": channel_id,
        "limit": limit,
        "cursor": cursor,
        "inclusive": True,
    }
    if oldest_ts:
        kwargs["oldest"] = oldest_ts
    return _slack_client().conversations_history(**kwargs)
