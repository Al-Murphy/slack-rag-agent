# Slack RAG Agent (MVP)

Minimal Slack-to-RAG starter for:
- Slack file ingestion (`file_shared`)
- PDF parsing + structured extraction
- Embedding + pgvector storage
- Query endpoint with retrieval + answer generation
- Planner/controller orchestration with confidence fallback
- Deduplication by document hash and low-information filtering
- Grounded answers with citations and validation scoring

## Agentic AI approach

This system uses specialized agents in a single pipeline:
- **Ingestion agent**: scans Slack channels, ingests PDFs and paper links, resolves full text, deduplicates, and stores chunks.
- **Retrieval agent**: embeds query + retrieves candidate chunks from pgvector.
- **Controller agent**: plans, reranks, checks confidence, and decides answer vs fallback.
- **Generation agent**: produces grounded answers with chunk citations.

Core loop: **ingest -> structure -> chunk -> embed -> retrieve -> rerank -> answer -> validate**.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env file and configure secrets:

```bash
cp .env.example .env
```

4. Ensure PostgreSQL has `pgvector` extension installed.
5. Run the API server:

```bash
uvicorn server:app --reload
```

## Docker local staging

```bash
docker compose up --build
```

## Endpoints

- `GET /health`
- `POST /slack/events`
- `GET /query?q=...&top_k=5`
- `POST /slack/ingest/channels`

## End-to-end validation

1. Start server:

```bash
uvicorn server:app --reload
```

2. Trigger real Slack ingestion:
- Upload a PDF to a channel where your bot is present.
- Capture the file ID from the event payload/logs.
- Optional direct test using validator:

```bash
python scripts/validate_mvp.py --file-id FXXXXXXXX --report validation_report.json
```

3. Validate RAG search quality and latency:

```bash
python scripts/validate_mvp.py --top-k 5 --report validation_report.json
```

Or run with your own query set:

```bash
python scripts/validate_mvp.py --queries-json tests/validation_queries.json --top-k 5 --report validation_report.json
```

## Automated channel crawling (agentic ingest)

Set channel targets in env:

```env
SLACK_CHANNEL_IDS=C01234567,C07654321
```

Run crawler via CLI (PDFs + paper links by default):

```bash
python scripts/crawl_channels.py --days-back 30 --per-channel-page-cap 5 --top-k-files-per-channel 50
```

Scan all channels visible to the bot:

```bash
python scripts/crawl_channels.py --scan-all-accessible --days-back 7
```

Disable link ingestion if needed:

```bash
python scripts/crawl_channels.py --channel-id C01234567 --no-include-links
```

Trigger via API:

```bash
curl -X POST http://127.0.0.1:8000/slack/ingest/channels \\
  -H \"Content-Type: application/json\" \\
  -d '{
    "channel_ids": ["C01234567"],
    "days_back": 30,
    "per_channel_page_cap": 5,
    "top_k_files_per_channel": 50,
    "include_links": true,
    "top_k_links_per_channel": 50
  }'
```

Required Slack scopes for crawler:
- `channels:history` and/or `groups:history`
- `channels:read` and/or `groups:read`
- `files:read`

Optional for stronger full-text fallback via Unpaywall:
- set `UNPAYWALL_EMAIL` in `.env`

### Metrics included in `validation_report.json`

- Retrieval relevance proxy: lexical overlap between query keywords and returned chunks.
- Coverage proxy: count of distinct extracted sections represented in top-k matches.
- Latency: embedding, retrieval, generation, and total query timings.

## Retrieval and safety behavior

- Ingestion deduplicates on SHA-256 file hash via the `documents` table.
- Link ingestion uses a full-text agent:
  - tries direct URL
  - tries discovered PDF links in page HTML
  - tries fallback sources (arXiv/DOI, optional Unpaywall OA links)
- Chunking targets semantically coherent chunks in the ~500-1000 token range.
- Low-information/noisy chunks are filtered before embedding.
- Query flow runs: plan -> retrieve -> rerank -> generate -> validate.
- If confidence is low, the agent returns an explicit \"not confident\" fallback.
- Answers include citations metadata (`citations`) for verification.

## Notes

- The Slack event endpoint includes URL verification support and optional signature checks.
- This is an MVP baseline. Add retries, deduplication, auth hardening, and better chunking before production.
