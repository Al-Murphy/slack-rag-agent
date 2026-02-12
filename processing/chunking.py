from __future__ import annotations

import re
from typing import Iterable


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def is_low_information(text: str, min_words: int = 40) -> bool:
    """Heuristic low-information filter for noisy extraction output."""
    normalized = " ".join(text.split())
    if not normalized:
        return True

    words = _word_tokens(normalized)
    if len(words) < min_words:
        return True

    alnum_chars = sum(ch.isalnum() for ch in normalized)
    if alnum_chars / max(1, len(normalized)) < 0.45:
        return True

    unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
    if unique_ratio < 0.08:
        return True

    return False


def split_text(
    text: str,
    max_tokens: int = 900,
    min_tokens: int = 500,
    overlap_tokens: int = 90,
) -> list[str]:
    """
    Split text into semantically coherent chunks (~500-1000 tokens).
    Uses paragraph boundaries first, then token overlap for continuity.
    """
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    if min_tokens > max_tokens:
        raise ValueError("min_tokens must be <= max_tokens")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    raw_chunks: list[list[str]] = []
    current_tokens: list[str] = []

    for paragraph in paragraphs:
        p_tokens = _word_tokens(paragraph)
        if not p_tokens:
            continue

        if len(p_tokens) > max_tokens:
            if current_tokens:
                raw_chunks.append(current_tokens)
                current_tokens = []

            start = 0
            step = max_tokens - overlap_tokens
            while start < len(p_tokens):
                window = p_tokens[start : start + max_tokens]
                raw_chunks.append(window)
                if start + max_tokens >= len(p_tokens):
                    break
                start += step
            continue

        if len(current_tokens) + len(p_tokens) <= max_tokens:
            current_tokens.extend(p_tokens)
            continue

        if current_tokens:
            raw_chunks.append(current_tokens)
        current_tokens = p_tokens[:]

    if current_tokens:
        raw_chunks.append(current_tokens)

    merged: list[list[str]] = []
    for token_chunk in raw_chunks:
        if merged and len(token_chunk) < min_tokens:
            merged[-1].extend(token_chunk)
        else:
            merged.append(token_chunk)

    return [" ".join(tokens).strip() for tokens in merged if tokens]


def sections_to_chunks(sections: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Convert named sections to chunk tuples: (section_name, chunk_text)."""
    output: list[tuple[str, str]] = []
    for section_name, text in sections:
        if is_low_information(text):
            continue
        for chunk in split_text(text):
            if is_low_information(chunk, min_words=25):
                continue
            output.append((section_name, chunk))
    return output
