from __future__ import annotations

from typing import Iterable


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for embeddings."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        start += step

    return chunks


def sections_to_chunks(sections: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Convert named sections to chunk tuples: (section_name, chunk_text)."""
    output: list[tuple[str, str]] = []
    for section_name, text in sections:
        for chunk in split_text(text):
            output.append((section_name, chunk))
    return output
