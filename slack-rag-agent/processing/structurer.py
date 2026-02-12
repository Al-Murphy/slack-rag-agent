import json
import os

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


async def extract_structured_sections(text: str) -> dict:
    """Extract structured fields from raw research text."""
    client = AsyncOpenAI()
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    prompt = (
        "Extract this paper into strict JSON with fields: "
        "title, abstract, methods, results, conclusion, key_findings. "
        "Return concise strings and an array of findings."
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
                "content": [{"type": "input_text", "text": f"{prompt}\n\nPaper text:\n{text[:120000]}"}],
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

    return data
