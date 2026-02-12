from __future__ import annotations

from database.models import Chunk


def build_context(chunks: list[Chunk]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({chunk.section}) {chunk.content}")
    return "\n\n".join(lines)
