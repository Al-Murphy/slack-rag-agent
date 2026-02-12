import json
import os
import re

from openai import AsyncOpenAI

_SCHEMA = {
    "name": "paper_structure",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "abstract": {"type": "string"},
            "methods": {"type": "string"},
            "results": {"type": "string"},
            "conclusion": {"type": "string"},
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title", "abstract", "methods", "results", "conclusion", "key_findings"],
        "additionalProperties": False,
    },
}


def _extract_by_header(text: str, header: str) -> str:
    pattern = rf"(?is){header}\s*[:\n]\s*(.*?)(?=\n[A-Z][A-Za-z ]{{2,40}}\s*[:\n]|$)"
    m = re.search(pattern, text)
    return (m.group(1).strip() if m else "")[:6000]


def heuristic_structure(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0][:300] if lines else "Untitled"
    return {
        "title": title,
        "abstract": _extract_by_header(text, "abstract"),
        "methods": _extract_by_header(text, "methods?|methodology"),
        "results": _extract_by_header(text, "results?|findings"),
        "conclusion": _extract_by_header(text, "conclusion|discussion"),
        "key_findings": [],
    }


async def extract_structured_sections(text: str) -> dict:
    """Extract structured fields from raw research text."""
    client = AsyncOpenAI()
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    heuristics = heuristic_structure(text)
    prompt = (
        "Extract this paper into strict JSON with fields: "
        "title, abstract, methods, results, conclusion, key_findings. "
        "Use the heuristic draft as a hint, but correct it when evidence suggests better sections. "
        "Be faithful to source text and avoid invented claims."
    )

    response = await client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You convert raw paper text into a strict JSON schema.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"{prompt}\n\nHeuristic draft:\n{json.dumps(heuristics)}\n\n"
                            f"Paper text:\n{text[:120000]}"
                        ),
                    }
                ],
            },
        ],
        text={"format": {"type": "json_schema", "name": _SCHEMA["name"], "schema": _SCHEMA["schema"], "strict": True}},
    )

    if not getattr(response, "output_text", None):
        raise ValueError("Model did not return structured JSON output")

    try:
        data = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON returned from structuring model") from exc

    for key in ("title", "abstract", "methods", "results", "conclusion", "key_findings"):
        if key not in data:
            raise ValueError(f"Missing required field in structured output: {key}")

    if not isinstance(data["key_findings"], list):
        raise ValueError("key_findings must be a list")

    # Hybrid fallback if model misses sections on noisy PDFs.
    if not data["abstract"].strip():
        data["abstract"] = heuristics["abstract"]
    if not data["methods"].strip():
        data["methods"] = heuristics["methods"]
    if not data["results"].strip():
        data["results"] = heuristics["results"]
    if not data["conclusion"].strip():
        data["conclusion"] = heuristics["conclusion"]

    return data
