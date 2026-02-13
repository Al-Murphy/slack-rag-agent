import json
import os
import re

_SCHEMA = {
    "name": "paper_structure",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "tldr": {"type": "string"},
            "abstract": {"type": "string"},
            "methods": {"type": "string"},
            "results": {"type": "string"},
            "conclusion": {"type": "string"},
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title", "authors", "tldr", "abstract", "methods", "results", "conclusion", "key_findings"],
        "additionalProperties": False,
    },
}


def _extract_by_header(text: str, header: str) -> str:
    pattern = rf"(?is)(?:{header})\s*[:\n]\s*(.*?)(?=\n[A-Z][A-Za-z ]{{2,40}}\s*[:\n]|$)"
    m = re.search(pattern, text)
    if not m:
        return ""
    section = m.group(1) or ""
    return section.strip()[:6000]


def _extract_authors_heuristic(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[1:8]:
        lower = line.lower()
        if any(k in lower for k in ("abstract", "introduction", "doi", "journal", "keywords")):
            continue
        if "," in line and len(line) < 300 and "." not in line:
            candidates = [a.strip() for a in line.split(",") if a.strip()]
            if 1 <= len(candidates) <= 20:
                return candidates[:20]
    return []


def _fallback_tldr(data: dict) -> str:
    parts = []
    if data.get("abstract"):
        parts.append(data["abstract"].strip())
    if data.get("results"):
        parts.append("Results: " + data["results"].strip())
    if data.get("conclusion"):
        parts.append("Conclusion: " + data["conclusion"].strip())
    if data.get("key_findings"):
        parts.append("Key findings: " + "; ".join(data.get("key_findings", [])[:4]))
    raw = " ".join(parts)
    return " ".join(raw.split())[:900]


def heuristic_structure(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0][:300] if lines else "Untitled"
    data = {
        "title": title,
        "authors": _extract_authors_heuristic(text),
        "tldr": "",
        "abstract": _extract_by_header(text, "abstract"),
        "methods": _extract_by_header(text, "methods?|methodology"),
        "results": _extract_by_header(text, "results?|findings"),
        "conclusion": _extract_by_header(text, "conclusion|discussion"),
        "key_findings": [],
    }
    data["tldr"] = _fallback_tldr(data)
    return data


async def extract_structured_sections(text: str) -> dict:
    """Extract structured fields from raw research text, including an informative TLDR."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    heuristics = heuristic_structure(text)
    prompt = (
        "Extract this paper into strict JSON with fields: "
        "title, authors, tldr, abstract, methods, results, conclusion, key_findings. "
        "The TLDR must be accurate, informative, and grounded in the paper: 4-7 sentences covering "
        "the problem, approach, main quantitative/qualitative findings, and limitations if stated. "
        "Do not invent claims. "
        "Use the heuristic draft as a hint, but correct it when evidence suggests better sections. "
        "Authors should be a list of author names."
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

    for key in ("title", "authors", "tldr", "abstract", "methods", "results", "conclusion", "key_findings"):
        if key not in data:
            raise ValueError(f"Missing required field in structured output: {key}")

    if not isinstance(data["key_findings"], list):
        raise ValueError("key_findings must be a list")
    if not isinstance(data["authors"], list):
        data["authors"] = []

    if not data["authors"]:
        data["authors"] = heuristics["authors"]
    if not data["abstract"].strip():
        data["abstract"] = heuristics["abstract"]
    if not data["methods"].strip():
        data["methods"] = heuristics["methods"]
    if not data["results"].strip():
        data["results"] = heuristics["results"]
    if not data["conclusion"].strip():
        data["conclusion"] = heuristics["conclusion"]

    data["tldr"] = " ".join((data.get("tldr") or "").split())
    if len(data["tldr"]) < 80:
        data["tldr"] = _fallback_tldr(data)

    return data
