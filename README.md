# Slack RAG Agent

![Slack RAG Agent Logo](ui/logo-v3.svg)

Agentic Slack-to-RAG system for genomics x AI paper workflows.

## Core Features

- Ingests Slack PDFs and paper links across channels
- Resolves full text (PDF/HTML + fallback resolution paths)
- Extracts structured sections and generates high-quality TLDRs
- Stores chunks, embeddings, metadata, and related-paper graph in Postgres + pgvector
- Grounded retrieval with citations, related papers, and confidence reporting
- Review agent validates answers before return
- **Review-to-DB writeback:** review agent can update document labels/notes/confidence and paper link status (`active` / `suppressed`)

## Agentic Pipeline

1. Slack crawler discovers files/links
2. Full-text resolver fetches best available paper source
3. Structurer extracts title/authors/sections/TLDR
4. Vector store inserts chunks + embeddings + document metadata
5. Retrieval agent finds evidence and drafts response
6. Review agent checks grounding/fit and can curate DB entries/connections

## Required `.env`

```env
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBEDDING=text-embedding-3-large
EMBEDDING_DIM=3072

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...

# Database (Supabase/Postgres)
DATABASE_URL=postgresql://postgres:<password>@<host>:5432/postgres?sslmode=require

# App flags
ENABLE_REVIEW_DB_UPDATES=true
ALLOW_DB_CLEAR=false

# Optional crawler defaults
CRAWL_CHANNEL_IDS=C12345,C67890
CRAWL_DAYS_BACK_DEFAULT=1
```

## Run

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

## Notes

- Python 3.11+ recommended
- Enable `vector` extension (`pgvector`) on your Postgres database
- Slack bot needs channel history/file scopes and must be in the target channels
