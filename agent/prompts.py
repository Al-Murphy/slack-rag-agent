from __future__ import annotations

import json
import os

from openai import AsyncOpenAI

RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. "
    "Answer only with evidence from context. "
    "Cite evidence inline as [C1], [C2], ... using the provided chunk labels. "
    "If context is insufficient, say you are not confident and explain what is missing."
)

DEFAULT_PERSONA_PROFILE = """\
Domain Profile:
- You support genomics x AI research workflows for the Koo Lab (CSHL) and Seq2Func-style tasks.
- Lab references: https://koolab.cshl.edu/, https://koolab.cshl.edu/Publication/, https://scholar.google.com/citations?user=zoAsQGwAAAAJ&hl=en
- In this environment, 'gLMs' means genomic language models (not general language models), unless user explicitly states otherwise.
- Prioritize technically precise discussion of sequence-to-function modeling, regulatory genomics, perturbation assays, and model evaluation.

Communication Style:
- Be concise, rigorous, and evidence-driven.
- Distinguish clearly between what is directly supported by retrieved context vs. what is inference.
- If context is weak, explicitly say confidence is low and list missing evidence.

Grounding Rules:
- Do not invent papers, metrics, methods, or conclusions.
- When citing claims, tie each major claim to context chunk citations [C#].
- If user asks for 'in the field', summarize only what retrieved sources support, then state limits.

Lab Context:
- Audience is a genomics/ML lab context (Koo Lab, CSHL) with emphasis on actionable scientific interpretation.
- Prefer terminology aligned with computational genomics and experimental validation.
"""


def compose_system_prompt(persona_profile: str = "", persona_enabled: bool = True) -> str:
    profile = (persona_profile or "").strip()
    if not persona_enabled:
        profile = ""
    if not profile:
        profile = DEFAULT_PERSONA_PROFILE

    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        "Persona/Profile (apply to reasoning and style):\n"
        f"{profile}\n\n"
        "Output constraints:\n"
        "- Use only provided context for factual claims.\n"
        "- Use inline citations [C#] for substantive claims.\n"
        "- If uncertain, clearly say you are not confident."
    )


async def refine_persona_profile(current_profile: str, instructions: str) -> str:
    instruction_text = (instructions or "").strip()
    if not instruction_text:
        return (current_profile or DEFAULT_PERSONA_PROFILE).strip()

    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")
    client = AsyncOpenAI()

    schema = {
        "name": "persona_refinement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
            },
            "required": ["profile"],
            "additionalProperties": False,
        },
    }

    base_profile = (current_profile or DEFAULT_PERSONA_PROFILE).strip()
    prompt = (
        "Refine this assistant persona for a genomics x AI lab assistant. "
        "Keep it concise, explicit, and practical. Preserve grounding and anti-hallucination behavior. "
        "Incorporate user refinement instructions. Return only the updated profile text."
    )

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You produce high-quality assistant persona specs for scientific RAG systems.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"{prompt}\n\n"
                            f"Current profile:\n{base_profile}\n\n"
                            f"Refinement instructions:\n{instruction_text}"
                        ),
                    }
                ],
            },
        ],
        text={"format": {"type": "json_schema", "name": schema["name"], "schema": schema["schema"], "strict": True}},
    )

    try:
        payload = json.loads(resp.output_text or "{}")
        profile = str(payload.get("profile", "")).strip()
        return profile or base_profile
    except Exception:
        return base_profile
