from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Any

from openai import AsyncOpenAI

from agent.controller import answer_support_score, confidence_score, plan_query, rerank_chunks
from agent.prompts import compose_system_prompt
from agent.tools import build_context
from database.vector_store import (
    get_documents_by_doc_ids,
    get_related_documents_for_doc_ids,
    search_hybrid_chunks,
)
from processing.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _expand_query_for_retrieval(query: str) -> str:
    q = query.strip()
    low = q.lower()
    extras: list[str] = []

    if "glm" in low or "glms" in low or "genomic language model" in low:
        extras.extend([
            "genomic language model",
            "genomic language models",
            "DNA language model",
            "DNA language models",
            "DNALM",
            "regulatory DNA language model",
        ])

    if "kundaje" in low or "kundaje group" in low or "kundaje lab" in low:
        extras.extend(["Anshul Kundaje", "Kundaje lab", "Stanford genomics"])

    if "seq2func" in low or "sequence to function" in low:
        extras.extend(["sequence-to-function", "regulatory genomics model", "functional genomics"])

    if not extras:
        return q

    uniq: list[str] = []
    seen = set()
    for e in extras:
        el = e.strip()
        if not el:
            continue
        lk = el.lower()
        if lk in seen:
            continue
        seen.add(lk)
        uniq.append(el)

    return f"{q} | aliases: {'; '.join(uniq)}"


def _metadata_relevance_bonus(query: str, chunk: Any, doc_lookup: dict[str, dict]) -> float:
    doc = doc_lookup.get(chunk.doc_id, {}) if doc_lookup else {}
    if not doc:
        return 0.0

    q_tokens = {t for t in query.lower().split() if len(t) > 2}
    title = str(doc.get("title", "") or "").lower()
    authors = " ".join(str(a) for a in (doc.get("authors", []) or [])).lower()
    tldr = str(doc.get("tldr", "") or "").lower()
    meta = f"{title} {authors} {tldr}"

    if not meta.strip() or not q_tokens:
        return 0.0

    overlap = sum(1 for t in q_tokens if t in meta)
    base = min(0.22, 0.06 * overlap)

    extra = 0.0
    ql = query.lower()
    if "group" in ql or "lab" in ql:
        if any(t in authors for t in q_tokens):
            extra += 0.08
    if "glm" in ql or "glms" in ql:
        if any(k in meta for k in ["genomic language model", "dna language model", "dnalm"]):
            extra += 0.06

    return min(0.30, base + extra)


def _fallback_response(reason: str, confidence: float) -> str:
    return (
        "I am not confident the retrieved context is sufficient to answer this reliably. "
        f"Reason: {reason}. Confidence={confidence:.2f}. "
        "Please provide additional documents or a more specific question."
    )


async def generate_answer(
    query: str,
    relevant_chunks: list[Any],
    doc_lookup: dict[str, dict] | None = None,
    persona_profile: str = "",
    persona_enabled: bool = True,
) -> str:
    client = AsyncOpenAI()
    context_text = build_context(relevant_chunks, doc_lookup=doc_lookup)
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    system_prompt = compose_system_prompt(persona_profile=persona_profile, persona_enabled=persona_enabled)

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
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


async def _llm_rerank(query: str, ranked: list[tuple[Any, float]], max_candidates: int = 20) -> list[tuple[Any, float]]:
    if not ranked:
        return ranked

    use_llm_rerank = os.environ.get("ENABLE_LLM_RERANK", "true").lower() in {"1", "true", "yes"}
    if not use_llm_rerank:
        return ranked

    candidates = ranked[:max_candidates]
    candidate_payload = [
        {
            "id": c.id,
            "section": c.section,
            "doc_id": c.doc_id,
            "text": (c.content or "")[:1200],
        }
        for c, _ in candidates
    ]

    client = AsyncOpenAI()
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    schema = {
        "name": "rerank_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ordered_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            },
            "required": ["ordered_ids"],
            "additionalProperties": False,
        },
    }

    try:
        resp = await client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Rank candidates by how directly they answer the query. Prefer evidence-rich chunks.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Query:\n{query}\n\n"
                                f"Candidates JSON:\n{json.dumps(candidate_payload)}\n\n"
                                "Return ordered_ids from most relevant to least relevant."
                            ),
                        }
                    ],
                },
            ],
            text={"format": {"type": "json_schema", "name": schema["name"], "schema": schema["schema"], "strict": True}},
        )
        payload = json.loads(resp.output_text or "{}")
        ordered_ids = payload.get("ordered_ids", [])
        if not isinstance(ordered_ids, list):
            return ranked

        by_id = {c.id: (c, s) for c, s in candidates}
        ordered: list[tuple[Any, float]] = []
        seen: set[int] = set()

        for cid in ordered_ids:
            try:
                i = int(cid)
            except Exception:
                continue
            if i in by_id and i not in seen:
                ordered.append(by_id[i])
                seen.add(i)

        for c, s in candidates:
            if c.id not in seen:
                ordered.append((c, s))

        tail = ranked[max_candidates:]
        return ordered + tail
    except Exception:
        logger.exception("LLM rerank failed; using heuristic ranking")
        return ranked


async def query_rag(
    query: str,
    top_k: int = 5,
    persona_profile: str = "",
    persona_enabled: bool = True,
) -> dict:
    t0 = time.perf_counter()
    plan = plan_query(query)
    retrieval_query = _expand_query_for_retrieval(query)
    q_vec = get_embeddings(retrieval_query)
    t1 = time.perf_counter()

    retrieval_k = min(120, max(top_k * 12, top_k + 20))
    chunks = search_hybrid_chunks(query=retrieval_query, query_embedding=q_vec, top_k=retrieval_k, vector_k=retrieval_k, sparse_k=retrieval_k)
    t2 = time.perf_counter()

    ranked = rerank_chunks(query, chunks)

    # Metadata-aware boosting before LLM rerank helps author/lab queries (e.g., "kundaje group glm").
    candidate_doc_ids = [c.doc_id for c, _ in ranked[: min(120, len(ranked))]]
    candidate_doc_lookup = get_documents_by_doc_ids(candidate_doc_ids)
    boosted: list[tuple[Any, float]] = []
    for chunk, score in ranked:
        boosted.append((chunk, round(float(score) + _metadata_relevance_bonus(query, chunk, candidate_doc_lookup), 4)))
    boosted.sort(key=lambda x: x[1], reverse=True)

    ranked = await _llm_rerank(query, boosted, max_candidates=min(30, len(boosted)))

    selected_ranked = ranked[:top_k]
    selected_chunks = [c for c, _ in selected_ranked]
    selected_doc_ids = [c.doc_id for c in selected_chunks]
    doc_lookup = get_documents_by_doc_ids(selected_doc_ids)
    related_map = get_related_documents_for_doc_ids(selected_doc_ids, per_doc_limit=5)

    conf = confidence_score(plan, selected_ranked)
    needs_fallback = len(selected_chunks) < plan.min_required_matches or conf < plan.confidence_threshold

    if needs_fallback:
        reason = "insufficient retrieval confidence"
        answer = _fallback_response(reason, conf)
        validator_score = 0.0
    else:
        answer = await generate_answer(
            query,
            selected_chunks,
            doc_lookup=doc_lookup,
            persona_profile=persona_profile,
            persona_enabled=persona_enabled,
        )
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
                "paper": doc_lookup.get(c.doc_id, {}),
                "related_papers": related_map.get(c.doc_id, []),
            }
            for c, score in selected_ranked
        ],
        "papers": [
            {
                **doc,
                "related_papers": related_map.get(doc_id, []),
            }
            for doc_id, doc in doc_lookup.items()
        ],
        "citations": [
            {
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "section": c.section,
                "paper": doc_lookup.get(c.doc_id, {}),
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
                "retrieved_candidates": len(chunks),
                "section_coverage_count": len(section_counts.keys()),
                "sections": dict(section_counts),
                "planner": {
                    "query_type": plan.query_type,
                    "min_required_matches": plan.min_required_matches,
                    "confidence_threshold": plan.confidence_threshold,
                    "confidence_score": conf,
                    "validator_support_score": validator_score,
                    "llm_rerank_enabled": os.environ.get("ENABLE_LLM_RERANK", "true"),
                },
            },
        },
    }
