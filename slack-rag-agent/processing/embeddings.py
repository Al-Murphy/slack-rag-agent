from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI()


def embedding_dimension() -> int:
    return int(os.environ.get("EMBEDDING_DIM", "3072"))


def get_embeddings(text: str) -> list[float]:
    """Generate an embedding vector for a block of text."""
    model = os.environ.get("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding
