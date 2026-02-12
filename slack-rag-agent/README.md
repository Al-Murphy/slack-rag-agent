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

## Notes

- The Slack event endpoint includes URL verification support and optional signature checks.
- This is an MVP baseline. Add retries, deduplication, auth hardening, and better chunking before production.
