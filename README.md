# Slack RAG Agent

![Slack RAG Agent Logo](ui/logo-v3.svg)

Slack-to-RAG system for genomics paper workflows.

## What It Does

- Crawls Slack channels for PDFs and paper links.
- Resolves full text (with fallback source discovery).
- Extracts structured fields (`title`, `authors`, `tldr`, `abstract`, `methods`, `results`, `conclusion`, `key_findings`).
- Chunks + embeds documents into Postgres/pgvector.
- Links related papers by semantic similarity.
- Answers questions with grounded retrieval, reranking, confidence checks, citations, and source links.

## Agentic Loop

1. Ingest: discover paper in Slack.
2. Resolve: fetch best available full text.
3. Structure: parse and extract scientific sections.
4. Index: chunk, embed, store, deduplicate, link related work.
5. Retrieve: hybrid search (vector + sparse + metadata-aware rerank).
6. Respond: generate grounded answer with fallback when confidence is low.

## UI

- Chat interface with saved history.
- `Update DB` incremental ingestion.
- `Scrape Window` async backfill with progress.
- `Run Retrieval Test` quality eval.
- `Persona` editor/refiner for lab-specific prompt behavior.
- `Paper Graph` interactive network of related papers (click nodes for TLDR and metadata).

## Core Endpoints

- `GET /` UI
- `GET /query`
- `POST /crawl/incremental`
- `POST /slack/ingest/channels/async`
- `GET /slack/ingest/channels/async/{job_id}`
- `GET /papers/graph`
- `GET /db/stats`

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn server:app --reload
```

Required env vars: `OPENAI_API_KEY`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `DATABASE_URL`.
