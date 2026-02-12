import requests


def download_slack_file(url: str, headers: dict) -> bytes:
    """Download file content from Slack with auth headers."""
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content
