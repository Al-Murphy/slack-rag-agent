from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from functools import lru_cache

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from database.models import Base, Chunk, CrawlState, Document
from processing.chunking import sections_to_chunks
from processing.embeddings import get_embeddings

DEFAULT_DATABASE_URL = "postgresql://user:pass@localhost:5432/ragdb"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _engine():
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return create_engine(database_url, pool_pre_ping=True)


@lru_cache(maxsize=1)
def _session_factory():
    return sessionmaker(bind=_engine(), autoflush=False, autocommit=False)


def init_db() -> None:
    with _engine().begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=_engine())


def document_exists_by_hash(doc_hash: str) -> bool:
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        stmt = select(Document.id).where(Document.doc_hash == doc_hash).limit(1)
        return session.execute(stmt).scalar_one_or_none() is not None


def insert_paper_into_db(
    structured_json: dict,
    doc_id: str,
    doc_hash: str,
    source: str = "slack",
    source_ref: str = "",
) -> int:
    """
    Split key sections into chunks, embed each, and persist to vector store.
    Returns number of inserted chunks.
    """
    if document_exists_by_hash(doc_hash):
        logger.info("Skipping duplicate document hash=%s doc_id=%s", doc_hash, doc_id)
        return 0

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
        logger.warning("No chunks generated for doc_id=%s", doc_id)
        return 0

    SessionLocal = _session_factory()
    with SessionLocal() as session:
        session.add(
            Document(
                doc_id=doc_id,
                doc_hash=doc_hash,
                source=source,
                source_ref=source_ref,
                title=structured_json.get("title", "")[:500],
            )
        )
        session.add_all(rows)
        session.commit()

    logger.info("Inserted document doc_id=%s chunks=%d", doc_id, len(rows))
    return len(rows)


def search_similar_chunks(query_embedding: Sequence[float], top_k: int = 5) -> list[Chunk]:
    """Search by cosine distance using pgvector ordering."""
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        stmt = select(Chunk).order_by(Chunk.vector.cosine_distance(query_embedding)).limit(top_k)
        return list(session.execute(stmt).scalars().all())


def get_state_value(key: str) -> str | None:
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        row = session.get(CrawlState, key)
        return row.value if row else None


def set_state_value(key: str, value: str) -> None:
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        row = session.get(CrawlState, key)
        if row:
            row.value = value
        else:
            session.add(CrawlState(key=key, value=value))
        session.commit()
