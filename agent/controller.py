from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Plan:
    query_type: str
    min_required_matches: int
    confidence_threshold: float


def _keywords(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "is",
        "are",
        "for",
        "in",
        "on",
        "with",
        "what",
        "how",
        "why",
        "where",
        "when",
        "which",
    }
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if t not in stop and len(t) > 2}


def plan_query(query: str) -> Plan:
    q = query.lower()
    if any(k in q for k in ("compare", "difference", "versus", "vs")):
        return Plan(query_type="comparative", min_required_matches=3, confidence_threshold=0.45)
    if any(k in q for k in ("result", "finding", "conclusion", "evidence")):
        return Plan(query_type="evidence", min_required_matches=2, confidence_threshold=0.4)
    return Plan(query_type="general", min_required_matches=2, confidence_threshold=0.35)


def rerank_chunks(query: str, chunks: list[Any]) -> list[tuple[Any, float]]:
    query_keys = _keywords(query)
    if not chunks:
        return []

    scored: list[tuple[Any, float]] = []
    for idx, chunk in enumerate(chunks):
        content = (chunk.content or "").lower()
        content_keys = _keywords(content)
        lexical = len(query_keys & content_keys) / max(1, len(query_keys))

        section_bonus = {
            "results": 0.2,
            "conclusion": 0.15,
            "key_findings": 0.25,
            "abstract": 0.1,
            "methods": 0.05,
            "title": 0.03,
        }.get((chunk.section or "").lower(), 0.0)

        # small rank prior to preserve semantic search order
        rank_prior = max(0.0, 0.08 - (idx * 0.01))
        score = lexical + section_bonus + rank_prior
        scored.append((chunk, round(score, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def confidence_score(plan: Plan, ranked_chunks: list[tuple[Any, float]]) -> float:
    if not ranked_chunks:
        return 0.0
    top_scores = [score for _, score in ranked_chunks[:5]]
    avg_top = sum(top_scores) / max(1, len(top_scores))
    volume_factor = min(1.0, len(ranked_chunks) / max(1, plan.min_required_matches))
    return round(min(1.0, avg_top * math.sqrt(volume_factor)), 4)


def answer_support_score(answer_text: str, ranked_chunks: list[tuple[Any, float]]) -> float:
    if not answer_text.strip() or not ranked_chunks:
        return 0.0
    answer_keys = _keywords(answer_text)
    if not answer_keys:
        return 0.0

    context_text = " ".join(chunk.content for chunk, _ in ranked_chunks[:5]).lower()
    context_keys = _keywords(context_text)
    overlap = len(answer_keys & context_keys) / max(1, len(answer_keys))
    return round(overlap, 4)
