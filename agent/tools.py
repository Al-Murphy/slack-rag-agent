from __future__ import annotations

from database.models import Chunk


def build_context(chunks: list[Chunk], doc_lookup: dict[str, dict] | None = None) -> str:
    lines: list[str] = []
    doc_lookup = doc_lookup or {}
    for i, chunk in enumerate(chunks, start=1):
        doc = doc_lookup.get(chunk.doc_id, {})
        title = doc.get("title", "")
        authors = doc.get("authors", [])
        author_text = ", ".join(authors[:6]) if authors else ""
        prefix = f"[C{i}] id={chunk.id} section={chunk.section} doc={chunk.doc_id}"
        if title:
            prefix += f" title={title}"
        if author_text:
            prefix += f" authors={author_text}"
        lines.append(f"{prefix}\n{chunk.content}")
    return "\n\n".join(lines)
