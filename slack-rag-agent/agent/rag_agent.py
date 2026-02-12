from __future__ import annotations

import logging
import os
import time
from collections import Counter
from typing import Any

from openai import AsyncOpenAI

from agent.controller import answer_support_score, confidence_score, plan_query, rerank_chunks
from agent.prompts import RAG_SYSTEM_PROMPT
from agent.tools import build_context
from database.vector_store import search_similar_chunks
from processing.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _fallback_response(reason: str, confidence: float) -> str:
    return (
        "I am not confident the retrieved context is sufficient to answer this reliably. "
        f"Reason: {reason}. Confidence={confidence:.2f}. "
        "Please provide additional documents or a more specific question."
    )


async def generate_answer(query: str, relevant_chunks: list[Any]) -> str:
    client = AsyncOpenAI()
    context_text = build_context(relevant_chunks)
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": RAG_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Context:\n{context_text}\n\nQuestion:\n{query}",
                    }
                ],
            },
        ],
    )
    return resp.output_text or "No answer generated."


async def query_rag(query: str, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    plan = plan_query(query)
    q_vec = get_embeddings(query)
    t1 = time.perf_counter()
    chunks = search_similar_chunks(q_vec, top_k=max(top_k * 2, top_k))
    t2 = time.perf_counter()

    ranked = rerank_chunks(query, chunks)
    selected_ranked = ranked[:top_k]
    selected_chunks = [c for c, _ in selected_ranked]
    conf = confidence_score(plan, selected_ranked)
    needs_fallback = len(selected_chunks) < plan.min_required_matches or conf < plan.confidence_threshold

    if needs_fallback:
        reason = "insufficient retrieval confidence"
        answer = _fallback_response(reason, conf)
        validator_score = 0.0
    else:
        answer = await generate_answer(query, selected_chunks)
        validator_score = answer_support_score(answer, selected_ranked)
        if validator_score < 0.2:
            answer = _fallback_response("generated answer not strongly supported by retrieved context", conf)

    t3 = time.perf_counter()

    section_counts = Counter(c.section for c in selected_chunks)
    logger.info(
        "RAG query processed top_k=%d retrieved=%d selected=%d confidence=%.3f validator=%.3f",
        top_k,
        len(chunks),
        len(selected_chunks),
        conf,
        validator_score,
    )

    return {
        "answer": answer,
        "matches": [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "section": c.section,
                "content": c.content,
                "rerank_score": score,
            }
            for c, score in selected_ranked
        ],
        "citations": [
            {
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "section": c.section,
            }
            for c in selected_chunks[:3]
        ],
        "metrics": {
            "latency_ms": {
                "embedding": round((t1 - t0) * 1000, 2),
                "retrieval": round((t2 - t1) * 1000, 2),
                "generation": round((t3 - t2) * 1000, 2),
                "total": round((t3 - t0) * 1000, 2),
            },
            "retrieval": {
                "top_k": top_k,
                "matches_returned": len(selected_chunks),
                "section_coverage_count": len(section_counts.keys()),
                "sections": dict(section_counts),
                "planner": {
                    "query_type": plan.query_type,
                    "min_required_matches": plan.min_required_matches,
                    "confidence_threshold": plan.confidence_threshold,
                    "confidence_score": conf,
                    "validator_support_score": validator_score,
                },
            },
        },
    }
