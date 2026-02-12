#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from agent.rag_agent import query_rag
from slack_listener.event_handler import handle_slack_event


@dataclass
class QueryEvaluation:
    query: str
    mode: str
    latency_ms: float
    matches_returned: int
    lexical_overlap_score: float
    section_coverage_count: int
    answer_preview: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _keywords(query: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "what",
        "is",
        "are",
        "how",
        "why",
        "for",
        "to",
        "and",
        "or",
        "of",
        "in",
        "on",
        "with",
        "about",
        "from",
        "explain",
        "summarize",
    }
    return {t for t in _tokenize(query) if t not in stop}


def _lexical_overlap(query: str, matches: list[dict[str, Any]]) -> float:
    keys = _keywords(query)
    if not keys or not matches:
        return 0.0

    matched = 0
    for m in matches:
        content_tokens = set(_tokenize(m.get("content", "")))
        if keys & content_tokens:
            matched += 1
    return matched / len(matches)


async def run_ingest(file_ids: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file_id in file_ids:
        payload = {"event": {"type": "file_shared", "file_id": file_id}}
        t0 = time.perf_counter()
        result = await handle_slack_event(payload)
        t1 = time.perf_counter()
        result["latency_ms"] = round((t1 - t0) * 1000, 2)
        results.append(result)
    return results


async def run_queries(query_specs: list[dict[str, str]], top_k: int) -> list[QueryEvaluation]:
    evaluations: list[QueryEvaluation] = []
    for spec in query_specs:
        query = spec["query"]
        mode = spec.get("mode", "unspecified")

        rag_out = await query_rag(query=query, top_k=top_k)
        matches = rag_out.get("matches", [])
        metrics = rag_out.get("metrics", {})
        latency = metrics.get("latency_ms", {}).get("total", 0.0)
        section_coverage = metrics.get("retrieval", {}).get("section_coverage_count", 0)

        evaluations.append(
            QueryEvaluation(
                query=query,
                mode=mode,
                latency_ms=float(latency),
                matches_returned=len(matches),
                lexical_overlap_score=round(_lexical_overlap(query, matches), 3),
                section_coverage_count=int(section_coverage),
                answer_preview=(rag_out.get("answer", "")[:220]).replace("\n", " "),
            )
        )
    return evaluations


def _default_queries() -> list[dict[str, str]]:
    return [
        {"mode": "exact_phrase", "query": "What are the key findings of this paper?"},
        {"mode": "conceptual", "query": "How did the authors evaluate their method and what outcomes mattered most?"},
        {"mode": "edge_case", "query": "What limitations are explicitly stated, if any?"},
    ]


def _summarize(eval_rows: list[QueryEvaluation]) -> dict[str, Any]:
    if not eval_rows:
        return {"count": 0}

    latency_values = [r.latency_ms for r in eval_rows]
    overlap_values = [r.lexical_overlap_score for r in eval_rows]
    coverage_values = [r.section_coverage_count for r in eval_rows]

    return {
        "count": len(eval_rows),
        "latency_ms": {
            "avg": round(statistics.mean(latency_values), 2),
            "p95": round(sorted(latency_values)[max(0, int(len(latency_values) * 0.95) - 1)], 2),
            "max": round(max(latency_values), 2),
        },
        "retrieval_relevance_proxy": {
            "avg_lexical_overlap_score": round(statistics.mean(overlap_values), 3),
            "min_lexical_overlap_score": round(min(overlap_values), 3),
        },
        "coverage_proxy": {
            "avg_section_coverage_count": round(statistics.mean(coverage_values), 2),
            "max_section_coverage_count": max(coverage_values),
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP validation for Slack ingest and RAG querying.")
    parser.add_argument("--file-id", action="append", default=[], help="Slack file ID to ingest (repeatable).")
    parser.add_argument("--queries-json", default="", help="JSON file with queries, each item: {\"mode\":..., \"query\":...}.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks to retrieve per query.")
    parser.add_argument("--report", default="validation_report.json", help="Output JSON report path.")
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required in environment/.env")
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required in environment/.env")

    query_specs = _default_queries()
    if args.queries_json:
        with open(args.queries_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("queries JSON must be a list of objects")
            query_specs = loaded

    ingest_results = []
    if args.file_id:
        ingest_results = await run_ingest(args.file_id)

    query_results = await run_queries(query_specs, top_k=args.top_k)
    query_summary = _summarize(query_results)

    report = {
        "ingest_results": ingest_results,
        "query_results": [r.__dict__ for r in query_results],
        "query_summary": query_summary,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report: {args.report}")
    print(json.dumps(query_summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
