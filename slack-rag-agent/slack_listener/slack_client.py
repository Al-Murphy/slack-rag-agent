import os

from slack_sdk import WebClient

slack = WebClient(token=os.environ.get("SLACK_BOT_TOKEN", ""))


def fetch_file_info(file_id: str) -> dict:
    """Fetch metadata for a file stored in Slack."""
    result = slack.files_info(file=file_id)
    return result.get("file", {})


def slack_auth_headers() -> dict:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    return {"Authorization": f"Bearer {token}"}
