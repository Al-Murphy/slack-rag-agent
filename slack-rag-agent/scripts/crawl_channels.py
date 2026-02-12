#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from slack_listener.channel_crawler import ingest_channels


async def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Slack channels and ingest PDF files into RAG DB.")
    parser.add_argument("--channel-id", action="append", default=[], help="Channel ID to scan (repeatable).")
    parser.add_argument("--scan-all-accessible", action="store_true", help="Scan all channels visible to the bot.")
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--per-channel-page-cap", type=int, default=5)
    parser.add_argument("--top-k-files-per-channel", type=int, default=50)
    parser.add_argument("--output", default="", help="Optional JSON output file.")
    args = parser.parse_args()

    load_dotenv()
    result = await ingest_channels(
        channel_ids=args.channel_id,
        scan_all_accessible=args.scan_all_accessible,
        days_back=args.days_back,
        per_channel_page_cap=args.per_channel_page_cap,
        top_k_files_per_channel=args.top_k_files_per_channel,
    )
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    asyncio.run(main())
