from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from database.models import Base, Chunk, CrawlState, Document, PaperRelation
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

    # Lightweight schema migration path for existing DBs.
    with _engine().begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS authors_json TEXT DEFAULT '[]'"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_url TEXT DEFAULT ''"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_text TEXT DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_vector vector({int(os.environ.get('EMBEDDING_DIM', '3072'))})"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS paper_relations (id SERIAL PRIMARY KEY, source_doc_id VARCHAR(255), target_doc_id VARCHAR(255), score FLOAT, reason VARCHAR(100), created_at TIMESTAMP DEFAULT NOW())"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_rel_pair ON paper_relations (source_doc_id, target_doc_id)"))


def document_exists_by_hash(doc_hash: str) -> bool:
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        stmt = select(Document.id).where(Document.doc_hash == doc_hash).limit(1)
        return session.execute(stmt).scalar_one_or_none() is not None


def _build_summary_text(structured_json: dict) -> str:
    title = structured_json.get("title", "")
    abstract = structured_json.get("abstract", "")
    results = structured_json.get("results", "")
    conclusion = structured_json.get("conclusion", "")
    key_findings = "\n".join(structured_json.get("key_findings", []))
    return "\n\n".join([part for part in [title, abstract, results, conclusion, key_findings] if part]).strip()[:20000]


def _get_existing_doc_pairs(session, source_doc_id: str, candidate_ids: list[str]) -> set[tuple[str, str]]:
    if not candidate_ids:
        return set()
    stmt = select(PaperRelation.source_doc_id, PaperRelation.target_doc_id).where(
        PaperRelation.source_doc_id.in_([source_doc_id] + candidate_ids),
        PaperRelation.target_doc_id.in_([source_doc_id] + candidate_ids),
    )
    return set(session.execute(stmt).all())


def _link_related_documents(session, new_doc: Document, top_k: int = 5, max_distance: float = 0.45) -> int:
    if not new_doc.summary_vector:
        return 0

    stmt = (
        select(Document)
        .where(Document.doc_id != new_doc.doc_id)
        .where(Document.summary_vector.is_not(None))
        .order_by(Document.summary_vector.cosine_distance(new_doc.summary_vector))
        .limit(top_k)
    )
    candidates = list(session.execute(stmt).scalars().all())
    if not candidates:
        return 0

    existing_pairs = _get_existing_doc_pairs(session, new_doc.doc_id, [c.doc_id for c in candidates])
    created = 0

    for c in candidates:
        try:
            dist_stmt = select(Document.summary_vector.cosine_distance(new_doc.summary_vector)).where(Document.doc_id == c.doc_id)
            distance = float(session.execute(dist_stmt).scalar_one())
        except Exception:
            continue

        if distance > max_distance:
            continue

        score = round(max(0.0, 1.0 - distance), 4)
        pair1 = (new_doc.doc_id, c.doc_id)
        pair2 = (c.doc_id, new_doc.doc_id)
        if pair1 not in existing_pairs:
            session.add(PaperRelation(source_doc_id=new_doc.doc_id, target_doc_id=c.doc_id, score=score, reason="semantic_similarity"))
            existing_pairs.add(pair1)
            created += 1
        if pair2 not in existing_pairs:
            session.add(PaperRelation(source_doc_id=c.doc_id, target_doc_id=new_doc.doc_id, score=score, reason="semantic_similarity"))
            existing_pairs.add(pair2)
            created += 1

    return created


def insert_paper_into_db(
    structured_json: dict,
    doc_id: str,
    doc_hash: str,
    source: str = "slack",
    source_ref: str = "",
    source_url: str = "",
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

    authors = structured_json.get("authors", [])
    if not isinstance(authors, list):
        authors = []

    summary_text = _build_summary_text(structured_json)
    summary_vector = get_embeddings(summary_text) if summary_text else None

    SessionLocal = _session_factory()
    with SessionLocal() as session:
        new_doc = Document(
            doc_id=doc_id,
            doc_hash=doc_hash,
            source=source,
            source_ref=source_ref,
            title=structured_json.get("title", "")[:500],
            authors_json=json.dumps(authors),
            source_url=source_url or "",
            summary_text=summary_text,
            summary_vector=summary_vector,
        )
        session.add(new_doc)
        session.add_all(rows)
        linked = _link_related_documents(session, new_doc)
        session.commit()

    logger.info("Inserted document doc_id=%s chunks=%d related_links=%d", doc_id, len(rows), linked)
    return len(rows)


def search_similar_chunks(query_embedding: Sequence[float], top_k: int = 5) -> list[Chunk]:
    """Search by cosine distance using pgvector ordering."""
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        stmt = select(Chunk).order_by(Chunk.vector.cosine_distance(query_embedding)).limit(top_k)
        return list(session.execute(stmt).scalars().all())


def get_documents_by_doc_ids(doc_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not doc_ids:
        return {}
    SessionLocal = _session_factory()
    with SessionLocal() as session:
        stmt = select(Document).where(Document.doc_id.in_(list(set(doc_ids))))
        docs = list(session.execute(stmt).scalars().all())

    output: dict[str, dict[str, Any]] = {}
    for d in docs:
        try:
            authors = json.loads(d.authors_json or "[]")
            if not isinstance(authors, list):
                authors = []
        except Exception:
            authors = []
        output[d.doc_id] = {
            "doc_id": d.doc_id,
            "title": d.title,
            "authors": authors,
            "source_url": d.source_url,
            "source_ref": d.source_ref,
            "source": d.source,
        }
    return output


def get_related_documents_for_doc_ids(doc_ids: Sequence[str], per_doc_limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    if not doc_ids:
        return {}

    SessionLocal = _session_factory()
    with SessionLocal() as session:
        rel_stmt = (
            select(PaperRelation)
            .where(PaperRelation.source_doc_id.in_(list(set(doc_ids))))
            .order_by(PaperRelation.score.desc())
        )
        relations = list(session.execute(rel_stmt).scalars().all())
        target_ids = [r.target_doc_id for r in relations]
        docs = get_documents_by_doc_ids(target_ids)

    out: dict[str, list[dict[str, Any]]] = {d: [] for d in doc_ids}
    for r in relations:
        if len(out.get(r.source_doc_id, [])) >= per_doc_limit:
            continue
        target = docs.get(r.target_doc_id)
        if not target:
            continue
        out.setdefault(r.source_doc_id, []).append(
            {
                "doc_id": r.target_doc_id,
                "score": r.score,
                "reason": r.reason,
                "title": target.get("title", ""),
                "authors": target.get("authors", []),
                "source_url": target.get("source_url", ""),
            }
        )
    return out




def clear_database() -> dict[str, int]:
    """Delete indexed data for development reset workflows."""
    with _engine().begin() as conn:
        pr = conn.execute(text("DELETE FROM paper_relations"))
        ch = conn.execute(text("DELETE FROM chunks"))
        dc = conn.execute(text("DELETE FROM documents"))
        cs = conn.execute(text("DELETE FROM crawl_state"))

    return {
        "paper_relations_deleted": int(pr.rowcount or 0),
        "chunks_deleted": int(ch.rowcount or 0),
        "documents_deleted": int(dc.rowcount or 0),
        "crawl_state_deleted": int(cs.rowcount or 0),
    }

def get_database_stats() -> dict[str, Any]:
    """Return high-level DB stats for UI diagnostics."""
    with _engine().begin() as conn:
        documents_count = int(conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0)
        chunks_count = int(conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0)
        relations_count = int(conn.execute(text("SELECT COUNT(*) FROM paper_relations")).scalar() or 0)

        size_bytes = None
        try:
            size_bytes = int(conn.execute(text("SELECT pg_database_size(current_database())")).scalar() or 0)
        except Exception:
            # Non-Postgres fallback.
            size_bytes = None

    return {
        "documents_count": documents_count,
        "chunks_count": chunks_count,
        "relations_count": relations_count,
        "database_size_bytes": size_bytes,
    }


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
