from __future__ import annotations

import os
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    section: Mapped[str] = mapped_column(String(100), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    meta_json: Mapped[str] = mapped_column(Text, default="{}")
    vector: Mapped[list[float]] = mapped_column(Vector(int(os.environ.get("EMBEDDING_DIM", "3072"))))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
