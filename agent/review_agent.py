from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI


async def review_answer(
    *,
    query: str,
    draft_answer: str,
    context_text: str,
    document_catalog: list[dict[str, Any]] | None = None,
    connection_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Review and, if needed, correct an answer against retrieved context and domain taxonomy.

    Returns: {
      approved: bool,
      corrected_answer: str,
      issues: list[str],
      review_confidence: float,
      document_updates: list[...],
      connection_updates: list[...],
    }
    """
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")
    reviewer_model = os.environ.get("OPENAI_MODEL_REVIEW", model)
    client = AsyncOpenAI()

    schema = {
        "name": "answer_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "corrected_answer": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "review_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "document_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string"},
                            "paper_type": {"type": "string", "enum": ["genomic_language_model", "seq2func", "other", "unknown"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "note": {"type": "string"},
                        },
                        "required": ["doc_id", "paper_type", "confidence", "note"],
                        "additionalProperties": False,
                    },
                },
                "connection_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_doc_id": {"type": "string"},
                            "target_doc_id": {"type": "string"},
                            "action": {"type": "string", "enum": ["keep", "suppress", "activate"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["source_doc_id", "target_doc_id", "action", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["approved", "corrected_answer", "issues", "review_confidence", "document_updates", "connection_updates"],
            "additionalProperties": False,
        },
    }

    system_text = (
        "You are a strict scientific answer reviewer for a genomics RAG assistant. "
        "Your job is to catch taxonomy errors, unsupported claims, and label mismatches. "
        "Key rule: if query asks about genomic language models (gLMs), do not present sequence-to-function "
        "predictors as gLMs unless context explicitly classifies them as such. "
        "Use only provided context; do not add external facts."
    )

    doc_json = json.dumps(document_catalog or [], ensure_ascii=True)
    conn_json = json.dumps(connection_catalog or [], ensure_ascii=True)

    user_text = (
        f"Question:\n{query}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"Document catalog (candidate docs):\n{doc_json}\n\n"
        f"Connection catalog (existing doc links):\n{conn_json}\n\n"
        "Tasks:\n"
        "1) Determine if draft is accurate and properly scoped to the question.\n"
        "2) Identify issues (if any), especially model-classification errors.\n"
        "3) Provide corrected_answer that is concise, grounded, and preserves citation tags when possible.\n"
        "4) Emit document_updates only when confident: label docs as genomic_language_model / seq2func / other / unknown.\n"
        "5) Emit connection_updates only if needed to suppress clearly wrong links or activate clearly justified links.\n"
        "If draft is fine, set approved=true and corrected_answer to draft answer.\n"
        "Be conservative: prefer keep over suppress when uncertain."
    )

    try:
        resp = await client.responses.create(
            model=reviewer_model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            text={"format": {"type": "json_schema", "name": schema["name"], "schema": schema["schema"], "strict": True}},
        )
        payload = json.loads(resp.output_text or "{}")
        approved = bool(payload.get("approved", False))
        corrected = str(payload.get("corrected_answer", "") or "").strip() or draft_answer
        issues_raw = payload.get("issues", [])
        issues = [str(x).strip() for x in issues_raw if str(x).strip()] if isinstance(issues_raw, list) else []
        review_confidence = float(payload.get("review_confidence", 0.0) or 0.0)
        review_confidence = max(0.0, min(1.0, review_confidence))
        doc_updates = payload.get("document_updates", [])
        conn_updates = payload.get("connection_updates", [])
        if not isinstance(doc_updates, list):
            doc_updates = []
        if not isinstance(conn_updates, list):
            conn_updates = []
        return {
            "approved": approved,
            "corrected_answer": corrected,
            "issues": issues,
            "review_confidence": review_confidence,
            "document_updates": doc_updates,
            "connection_updates": conn_updates,
        }
    except Exception:
        return {
            "approved": True,
            "corrected_answer": draft_answer,
            "issues": ["review_failed"],
            "review_confidence": 0.0,
            "document_updates": [],
            "connection_updates": [],
        }
