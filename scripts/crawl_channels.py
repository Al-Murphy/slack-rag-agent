#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from slack_listener.channel_crawler import ingest_channels


def _fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):,.0f} ms"
    except (TypeError, ValueError):
        return "-"


def _summarize(result: dict[str, Any], max_channels: int = 10) -> str:
    metrics = result.get("metrics", {})
    by_channel = result.get("by_channel", {})

    channels_scanned = int(result.get("channels_scanned", 0))
    ingested = int(metrics.get("ingested", 0))
    duplicates = int(metrics.get("duplicates", 0))
    errors = int(metrics.get("errors", 0))

    channel_rows = []
    for cid, stats in by_channel.items():
        pdf_discovered = int(stats.get("pdf_discovered", 0))
        link_discovered = int(stats.get("link_discovered", 0))
        total_discovered = pdf_discovered + link_discovered
        channel_rows.append(
            {
                "channel_id": cid,
                "discovered": total_discovered,
                "pdf_discovered": pdf_discovered,
                "link_discovered": link_discovered,
                "ingested": int(stats.get("ingested", 0)),
                "duplicates": int(stats.get("duplicates", 0)),
                "errors": int(stats.get("errors", 0)),
                "timing_total": float((stats.get("timing_ms") or {}).get("total", 0.0)),
            }
        )

    active = [r for r in channel_rows if (r["discovered"] > 0 or r["ingested"] > 0 or r["errors"] > 0 or r["duplicates"] > 0)]
    idle = [r for r in channel_rows if r not in active]
    ingested_channels = [r["channel_id"] for r in channel_rows if r["ingested"] > 0]
    duplicate_only_channels = [r["channel_id"] for r in channel_rows if r["duplicates"] > 0 and r["ingested"] == 0 and r["errors"] == 0]
    error_channels = [r["channel_id"] for r in channel_rows if r["errors"] > 0]

    active_sorted = sorted(
        active,
        key=lambda r: (r["ingested"], r["discovered"], r["duplicates"], -r["errors"], r["timing_total"]),
        reverse=True,
    )

    timing = metrics.get("timing_ms", {})
    lines = [
        "Run Summary",
        f"- Channels scanned: {channels_scanned}",
        f"- Active channels: {len(active)} | Idle channels: {len(idle)}",
        f"- Papers ingested: {ingested} | Duplicates: {duplicates} | Errors: {errors}",
        f"- Total duration: {_fmt_ms(metrics.get('duration_ms'))}",
        "- Timing bottlenecks:",
        f"  - Structure: {_fmt_ms(timing.get('structure'))}",
        f"  - Resolve fetch: {_fmt_ms(timing.get('resolve_fetch'))}",
        f"  - Insert: {_fmt_ms(timing.get('insert'))}",
    ]

    if ingested_channels:
        lines.append(f"- Ingested channels ({len(ingested_channels)}): {', '.join(ingested_channels[:max_channels])}")
    if duplicate_only_channels:
        lines.append(f"- Duplicate-only channels ({len(duplicate_only_channels)}): {', '.join(duplicate_only_channels[:max_channels])}")
    if error_channels:
        lines.append(f"- Error channels ({len(error_channels)}): {', '.join(error_channels[:max_channels])}")

    if active_sorted:
        lines.append("\nTop Active Channels")
        for row in active_sorted[:max_channels]:
            lines.append(
                f"- {row['channel_id']}: ingested={row['ingested']}, discovered={row['discovered']} "
                f"(pdf={row['pdf_discovered']}, links={row['link_discovered']}), "
                f"duplicates={row['duplicates']}, errors={row['errors']}, time={_fmt_ms(row['timing_total'])}"
            )

    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Slack channels and ingest PDF files into RAG DB.")
    parser.add_argument("--channel-id", action="append", default=[], help="Channel ID to scan (repeatable).")
    parser.add_argument("--scan-all-accessible", action="store_true", help="Scan all channels visible to the bot.")
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--per-channel-page-cap", type=int, default=5)
    parser.add_argument("--top-k-files-per-channel", type=int, default=50)
    parser.add_argument("--include-links", dest="include_links", action="store_true", help="Also ingest paper links found in messages.")
    parser.add_argument("--no-include-links", dest="include_links", action="store_false", help="Disable paper link ingestion.")
    parser.set_defaults(include_links=True)
    parser.add_argument("--top-k-links-per-channel", type=int, default=50)
    parser.add_argument("--link-concurrency-limit", type=int, default=3)
    parser.add_argument("--output", default="", help="Optional JSON output file.")
    parser.add_argument("--full-json", action="store_true", help="Print full JSON to stdout instead of concise summary.")
    parser.add_argument("--max-channels", type=int, default=10, help="Max channels to display in summary sections.")
    args = parser.parse_args()

    load_dotenv()
    result = await ingest_channels(
        channel_ids=args.channel_id,
        scan_all_accessible=args.scan_all_accessible,
        days_back=args.days_back,
        per_channel_page_cap=args.per_channel_page_cap,
        top_k_files_per_channel=args.top_k_files_per_channel,
        include_links=args.include_links,
        top_k_links_per_channel=args.top_k_links_per_channel,
        link_concurrency_limit=args.link_concurrency_limit,
    )

    text = json.dumps(result, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote: {args.output}")

    if args.full_json:
        print(text)
    else:
        print(_summarize(result, max_channels=args.max_channels))


if __name__ == "__main__":
    asyncio.run(main())
