# Slack RAG Agent (MVP)

Minimal Slack-to-RAG starter for:
- Slack file ingestion (`file_shared`)
- PDF parsing + structured extraction
- Embedding + pgvector storage
- Query endpoint with retrieval + answer generation

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

## Endpoints

- `GET /health`
- `POST /slack/events`
- `GET /query?q=...&top_k=5`

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

### Metrics included in `validation_report.json`

- Retrieval relevance proxy: lexical overlap between query keywords and returned chunks.
- Coverage proxy: count of distinct extracted sections represented in top-k matches.
- Latency: embedding, retrieval, generation, and total query timings.

## Notes

- The Slack event endpoint includes URL verification support and optional signature checks.
- This is an MVP baseline. Add retries, deduplication, auth hardening, and better chunking before production.
