from __future__ import annotations

import os
import time
from collections import Counter

from openai import AsyncOpenAI

from agent.prompts import RAG_SYSTEM_PROMPT
from agent.tools import build_context
from database.vector_store import search_similar_chunks
from processing.embeddings import get_embeddings


async def generate_answer(query: str, relevant_chunks: list) -> str:
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
    q_vec = get_embeddings(query)
    t1 = time.perf_counter()
    chunks = search_similar_chunks(q_vec, top_k=top_k)
    t2 = time.perf_counter()
    answer = await generate_answer(query, chunks)
    t3 = time.perf_counter()

    section_counts = Counter(c.section for c in chunks)
    return {
        "answer": answer,
        "matches": [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "section": c.section,
                "content": c.content,
            }
            for c in chunks
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
                "matches_returned": len(chunks),
                "section_coverage_count": len(section_counts.keys()),
                "sections": dict(section_counts),
            },
        },
    }
