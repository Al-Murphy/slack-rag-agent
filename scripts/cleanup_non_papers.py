#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


NON_PAPER_URL_PATTERNS = (
    r"/articles/?\?type=",
    r"/subjects/",
    r"/collections/",
    r"/news(?:/|$)",
    r"/category/",
    r"/latest(?:/|$)",
    r"/journal(?:/|$)",
    r"/toc(?:/|$)",
)

NON_PAPER_TITLE_TERMS = (
    "methods to watch",
    "news",
    "collection",
    "editorial",
    "podcast",
    "perspective",
    "career",
)


@dataclass
class Candidate:
    doc_id: str
    source_url: str
    title: str
    reason: str


def _engine():
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url, pool_pre_ping=True)


def _is_obvious_non_paper_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u:
        return False

    allow_signals = ("/abs/", "/pdf/", "/article/", "/articles/", "/content/", "doi.org/", "arxiv.org/")
    if any(s in u for s in allow_signals) and "type=" not in u:
        return False

    return any(re.search(pattern, u) for pattern in NON_PAPER_URL_PATTERNS)


def _looks_non_paper_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    return any(term in t for term in NON_PAPER_TITLE_TERMS)


def _find_candidates(limit: int | None = None) -> list[Candidate]:
    query = (
        "SELECT doc_id, COALESCE(source_url, ''), COALESCE(title, '') "
        "FROM documents ORDER BY created_at DESC"
    )
    params: dict[str, Any] = {}
    if limit is not None and limit > 0:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    with _engine().begin() as conn:
        rows = conn.execute(text(query), params).all()

    out: list[Candidate] = []
    for row in rows:
        doc_id = str(row[0])
        source_url = str(row[1] or "")
        title = str(row[2] or "")

        if source_url and _is_obvious_non_paper_url(source_url):
            out.append(Candidate(doc_id=doc_id, source_url=source_url, title=title, reason="non_paper_url_pattern"))
            continue

        if _looks_non_paper_title(title):
            out.append(Candidate(doc_id=doc_id, source_url=source_url, title=title, reason="non_paper_title_pattern"))

    return out


def _as_in_list(doc_ids: list[str]) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {}
    placeholders: list[str] = []
    for i, doc_id in enumerate(doc_ids):
        key = f"id_{i}"
        params[key] = doc_id
        placeholders.append(f":{key}")
    return "(" + ",".join(placeholders) + ")", params


def _apply_suppress(candidates: list[Candidate]) -> dict[str, int]:
    if not candidates:
        return {"documents_updated": 0, "chunks_deleted": 0, "relations_suppressed": 0}

    doc_ids = [c.doc_id for c in candidates]
    in_sql, in_params = _as_in_list(doc_ids)
    now = datetime.utcnow().isoformat(timespec="seconds")

    with _engine().begin() as conn:
        for c in candidates:
            note = f"[cleanup_non_papers:{now}] {c.reason} source_url={c.source_url}"
            conn.execute(
                text(
                    "UPDATE documents "
                    "SET paper_type='non_paper', review_notes=COALESCE(review_notes, '') || :suffix "
                    "WHERE doc_id=:doc_id"
                ),
                {"doc_id": c.doc_id, "suffix": ("\n" + note)},
            )

        chunks_deleted = int(conn.execute(text(f"DELETE FROM chunks WHERE doc_id IN {in_sql}"), in_params).rowcount or 0)
        relations_suppressed = int(
            conn.execute(
                text(
                    f"UPDATE paper_relations SET status='suppressed' "
                    f"WHERE source_doc_id IN {in_sql} OR target_doc_id IN {in_sql}"
                ),
                in_params,
            ).rowcount
            or 0
        )

    return {
        "documents_updated": len(doc_ids),
        "chunks_deleted": chunks_deleted,
        "relations_suppressed": relations_suppressed,
    }


def _apply_delete(candidates: list[Candidate]) -> dict[str, int]:
    if not candidates:
        return {"documents_deleted": 0, "chunks_deleted": 0, "relations_deleted": 0}

    doc_ids = [c.doc_id for c in candidates]
    in_sql, in_params = _as_in_list(doc_ids)

    with _engine().begin() as conn:
        chunks_deleted = int(conn.execute(text(f"DELETE FROM chunks WHERE doc_id IN {in_sql}"), in_params).rowcount or 0)
        relations_deleted = int(
            conn.execute(
                text(
                    f"DELETE FROM paper_relations "
                    f"WHERE source_doc_id IN {in_sql} OR target_doc_id IN {in_sql}"
                ),
                in_params,
            ).rowcount
            or 0
        )
        documents_deleted = int(conn.execute(text(f"DELETE FROM documents WHERE doc_id IN {in_sql}"), in_params).rowcount or 0)

    return {
        "documents_deleted": documents_deleted,
        "chunks_deleted": chunks_deleted,
        "relations_deleted": relations_deleted,
    }


def _serialize_candidates(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": c.doc_id,
            "source_url": c.source_url,
            "title": c.title,
            "reason": c.reason,
        }
        for c in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time cleanup for non-paper document entries")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument(
        "--mode",
        choices=["suppress", "delete"],
        default="suppress",
        help="suppress=keep metadata but remove chunks/suppress relations; delete=hard delete documents",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit scanned documents (0 = no limit)")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    load_dotenv()

    candidates = _find_candidates(limit=args.limit if args.limit > 0 else None)
    payload: dict[str, Any] = {
        "ok": True,
        "mode": args.mode,
        "apply": bool(args.apply),
        "candidate_count": len(candidates),
        "candidates": _serialize_candidates(candidates),
    }

    if args.apply:
        if args.mode == "delete":
            payload["changes"] = _apply_delete(candidates)
        else:
            payload["changes"] = _apply_suppress(candidates)

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"mode={args.mode} apply={args.apply} candidates={len(candidates)}")
    for c in candidates[:25]:
        print(f"- {c.doc_id} [{c.reason}] {c.source_url or '(no source_url)'} :: {c.title}")
    if len(candidates) > 25:
        print(f"... and {len(candidates)-25} more")
    if args.apply:
        print(f"changes={payload.get('changes', {})}")
    else:
        print("Dry-run only. Re-run with --apply to execute changes.")


if __name__ == "__main__":
    main()
