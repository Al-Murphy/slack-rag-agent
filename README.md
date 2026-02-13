# Slack RAG Agent

Slack-to-RAG system for ingesting papers from Slack, structuring content, storing vectors in Postgres/pgvector, and answering grounded research questions with citations.

## What It Does

- Ingests from Slack file uploads and paper links across channels.
- Resolves full text with fallback source discovery (publisher pages, PDF links, DOI/arXiv paths).
- Extracts structure (`title`, `authors`, `abstract`, `methods`, `results`, `conclusion`, `key_findings`).
- Chunks and embeds content into pgvector.
- Links related papers by semantic similarity.
- Answers queries with retrieval, reranking, confidence checks, and citations.

## Agentic System Structure

The system is split into specialized components:

- `slack_listener/*`: Slack event handling + channel crawler.
- `agent/fulltext_agent.py`: Full-text resolver for paper URLs.
- `processing/*`: parsing, structuring, chunking, embeddings.
- `database/*`: schema, vector search, related-paper graph, state.
- `agent/*`: planner/controller/retriever/generator/validator loop.
- `server.py`: FastAPI API + UI serving.
- `ui/index.html`: chat UI with history + ingest controls.

Core loop:

1. Discover paper (Slack PDF/link)
2. Resolve full text
3. Structure and chunk
4. Embed and persist
5. Link related papers
6. Query: plan -> retrieve -> rerank -> answer -> validate

## Data Model (High Level)

- `documents`: per-paper metadata (`title`, `authors_json`, canonical `source_url`, `summary_vector`, hash).
- `chunks`: chunk text + section + embedding vector.
- `paper_relations`: semantic links between related papers.
- `crawl_state`: state keys (e.g., incremental crawl timestamp).

## API Endpoints

Core:

- `GET /health`
- `GET /` (chat UI)
- `POST /slack/events`
- `GET /query?q=...&top_k=5`

Ingestion:

- `POST /slack/ingest/channels` (sync crawl)
- `POST /crawl/incremental` (since last run)
- `GET /crawl/incremental/status`

Async backfill (used by UI progress bar):

- `POST /slack/ingest/channels/async`
- `GET /slack/ingest/channels/async/{job_id}`

DB tooling (dev):

- `GET /db/stats`
- `POST /admin/clear-db` (requires env safety flag)

## UI Features

- ChatGPT-style conversation history (left sidebar, localStorage).
- `Update DB` (incremental crawl since last run).
- `Scrape Window` (X days/weeks/months/years) with real channel-based progress.
- `Clear DB` with double-confirm safety flow.
- Source links + related papers panel in responses.
- Top header DB stats (size, docs, chunks).

## Use Cases

- Lab paper channels: auto-index new papers from Slack and query by concept.
- Team literature review: retrieve related work linked to each result.
- Daily triage: incremental ingestion of newly posted papers.
- RAG QA over internal paper-sharing workflows with grounded citations.

## Quick Start

1. Create and activate a venv.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Configure env:

```bash
cp .env.example .env
```

Required:

- `OPENAI_API_KEY`
- `SLACK_BOT_TOKEN`
- `SLACK_SIGNING_SECRET`
- `DATABASE_URL`

Typical model config:

- `OPENAI_MODEL_CHAT=gpt-4o-mini`
- `OPENAI_MODEL_EMBEDDING=text-embedding-3-large`
- `EMBEDDING_DIM=3072`

4. Ensure Postgres has `pgvector` enabled.
5. Run:

```bash
uvicorn server:app --reload
```

## Slack Crawler CLI

Configure target channels:

```env
SLACK_CHANNEL_IDS=C01234567,C07654321
```

Run:

```bash
python scripts/crawl_channels.py --days-back 30 --per-channel-page-cap 5 --top-k-files-per-channel 50
```

Scan all channels visible to the bot:

```bash
python scripts/crawl_channels.py --scan-all-accessible --days-back 7
```

## Security and Guardrails

- Signature verification for Slack events (when signing secret is set).
- Dedup by content hash before expensive structuring.
- Confidence fallback when retrieval support is weak.
- `Clear DB` is disabled by default.

To enable dev DB clear:

```env
ALLOW_DB_CLEAR=true
DB_CLEAR_CONFIRM_PHRASE=CLEAR DB
```

Then restart server.

## Notes

- This is an MVP architecture focused on correctness and iteration speed.
- For production: add auth, rate limits, retries, background worker durability, and evaluation dashboards.
