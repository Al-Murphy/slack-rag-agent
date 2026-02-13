from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from agent.controller import rerank_chunks
from database.vector_store import get_eval_seed_finding, search_hybrid_chunks
from processing.embeddings import get_embeddings


async def _generate_paraphrases(text: str, n: int) -> list[str]:
    n = max(1, min(100, int(n)))
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")
    client = AsyncOpenAI()

    schema = {
        "name": "paraphrase_set",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "paraphrases": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["paraphrases"],
            "additionalProperties": False,
        },
    }

    prompt = (
        f"Create {n} meaning-preserving query paraphrases for retrieval evaluation. "
        "Vary wording and structure while keeping scientific meaning intact. "
        "Do not add facts. Keep each as a standalone query sentence."
    )

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "You generate high-quality retrieval test queries."}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"{prompt}\n\nSource finding:\n{text[:1600]}"}],
            },
        ],
        text={"format": {"type": "json_schema", "name": schema["name"], "schema": schema["schema"], "strict": True}},
    )

    try:
        payload = json.loads(resp.output_text or "{}")
        out = [" ".join(str(x).split()) for x in payload.get("paraphrases", []) if str(x).strip()]
    except Exception:
        out = []

    out = out[:n]
    if len(out) < n:
        out.extend([text] * (n - len(out)))
    return out


async def run_retrieval_eval(
    channel_id: str | None = None,
    days_back: int = 30,
    paraphrase_count: int = 5,
    top_k: int = 5,
) -> dict[str, Any]:
    seed = get_eval_seed_finding(channel_id=channel_id, days_back=days_back)
    if not seed:
        return {
            "ok": False,
            "reason": "no_seed_finding",
            "message": "No recent finding-like chunk found for evaluation. Try a wider window or check ingestion.",
        }

    finding = seed["finding_text"]
    doc_id = seed["doc_id"]

    paraphrases = await _generate_paraphrases(finding, paraphrase_count)
    queries = [finding] + paraphrases

    eval_rows: list[dict[str, Any]] = []
    hits = 0
    reciprocal_ranks: list[float] = []

    for q in queries:
        q_vec = get_embeddings(q)
        candidates = search_hybrid_chunks(query=q, query_embedding=q_vec, top_k=max(60, top_k * 10), vector_k=80, sparse_k=80)
        ranked = rerank_chunks(q, candidates)
        top = ranked[:top_k]

        rank = None
        for i, (chunk, _) in enumerate(top, start=1):
            if chunk.doc_id == doc_id:
                rank = i
                break

        hit = rank is not None
        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / float(rank))

        eval_rows.append(
            {
                "query": q,
                "hit": hit,
                "rank": rank,
                "top_doc_ids": [c.doc_id for c, _ in top],
            }
        )

    total = len(eval_rows)
    hit_rate = (hits / total) if total else 0.0
    mrr = (sum(reciprocal_ranks) / total) if total else 0.0

    return {
        "ok": True,
        "seed": {
            "doc_id": doc_id,
            "title": seed.get("title", ""),
            "source_url": seed.get("source_url", ""),
            "section": seed.get("section", ""),
            "finding_text": finding,
            "tldr": seed.get("tldr", ""),
        },
        "config": {
            "channel_id": channel_id,
            "days_back": days_back,
            "paraphrase_count": paraphrase_count,
            "top_k": top_k,
            "queries_tested": total,
        },
        "metrics": {
            "hit_rate_at_k": round(hit_rate, 4),
            "mrr_at_k": round(mrr, 4),
            "hits": hits,
            "total": total,
        },
        "queries": eval_rows,
    }
