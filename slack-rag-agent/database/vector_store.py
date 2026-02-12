from __future__ import annotations

import json
import os
from collections.abc import Sequence

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from database.models import Base, Chunk
from processing.chunking import sections_to_chunks
from processing.embeddings import get_embeddings

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:pass@localhost:5432/ragdb")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def insert_paper_into_db(structured_json: dict, doc_id: str) -> int:
    """
    Split key sections into chunks, embed each, and persist to vector store.
    Returns number of inserted chunks.
    """
    sections = [
        ("title", structured_json.get("title", "")),
        ("abstract", structured_json.get("abstract", "")),
        ("methods", structured_json.get("methods", "")),
        ("results", structured_json.get("results", "")),
        ("conclusion", structured_json.get("conclusion", "")),
        ("key_findings", "\n".join(structured_json.get("key_findings", []))),
    ]

    named_chunks = sections_to_chunks(sections)
    rows = []
    for idx, (section_name, chunk_text) in enumerate(named_chunks):
        embedding = get_embeddings(chunk_text)
        rows.append(
            Chunk(
                doc_id=doc_id,
                section=section_name,
                chunk_index=idx,
                content=chunk_text,
                meta_json=json.dumps({"section": section_name}),
                vector=embedding,
            )
        )

    if not rows:
        return 0

    with SessionLocal() as session:
        session.add_all(rows)
        session.commit()

    return len(rows)


def search_similar_chunks(query_embedding: Sequence[float], top_k: int = 5) -> list[Chunk]:
    """Search by cosine distance using pgvector ordering."""
    with SessionLocal() as session:
        stmt = select(Chunk).order_by(Chunk.vector.cosine_distance(query_embedding)).limit(top_k)
        return list(session.execute(stmt).scalars().all())
