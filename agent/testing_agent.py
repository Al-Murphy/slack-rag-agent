from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from agent.controller import rerank_chunks
from database.vector_store import get_eval_seed_bundle, search_hybrid_chunks
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
    seed = get_eval_seed_bundle(channel_id=channel_id, days_back=days_back)
    if not seed:
        return {
            "ok": False,
            "reason": "no_seed_finding",
            "message": "No recent finding-like chunk found for evaluation. Try a wider window or check ingestion.",
        }

    doc_id = seed["doc_id"]
    seed_items = seed.get("seeds", [])

    eval_rows: list[dict[str, Any]] = []
    total_hits = 0
    total_rr_sum = 0.0
    section_metrics: dict[str, dict[str, Any]] = {}

    for item in seed_items:
        section = str(item.get("section", "") or "unknown")
        finding = str(item.get("finding_text", "") or "").strip()
        if not finding:
            continue

        paraphrases = await _generate_paraphrases(finding, paraphrase_count)
        queries = [finding] + paraphrases

        section_hits = 0
        section_rr_sum = 0.0
        section_total = 0

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
                section_hits += 1
                rr = 1.0 / float(rank)
                section_rr_sum += rr
                total_rr_sum += rr
                total_hits += 1

            section_total += 1
            eval_rows.append(
                {
                    "section": section,
                    "query": q,
                    "hit": hit,
                    "rank": rank,
                    "top_doc_ids": [c.doc_id for c, _ in top],
                }
            )

        hit_rate = (section_hits / section_total) if section_total else 0.0
        mrr = (section_rr_sum / section_total) if section_total else 0.0
        section_metrics[section] = {
            "hit_rate_at_k": round(hit_rate, 4),
            "mrr_at_k": round(mrr, 4),
            "hits": section_hits,
            "total": section_total,
        }

    total = len(eval_rows)
    hit_rate = (total_hits / total) if total else 0.0
    mrr = (total_rr_sum / total) if total else 0.0

    primary = seed_items[0] if seed_items else {"section": "", "finding_text": ""}

    return {
        "ok": True,
        "seed": {
            "doc_id": doc_id,
            "title": seed.get("title", ""),
            "source_url": seed.get("source_url", ""),
            "source_ref": seed.get("source_ref", ""),
            "section": primary.get("section", ""),
            "finding_text": primary.get("finding_text", ""),
            "tldr": seed.get("tldr", ""),
            "sections_tested": [str(x.get("section", "")) for x in seed_items if x.get("section")],
            "seed_findings": seed_items,
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
            "hits": total_hits,
            "total": total,
        },
        "section_metrics": section_metrics,
        "queries": eval_rows,
    }
