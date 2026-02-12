import os

from openai import AsyncOpenAI

from agent.prompts import RAG_SYSTEM_PROMPT
from agent.tools import build_context
from database.vector_store import search_similar_chunks
from processing.embeddings import get_embeddings

client = AsyncOpenAI()


async def generate_answer(query: str, relevant_chunks: list) -> str:
    context_text = build_context(relevant_chunks)
    model = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": RAG_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Context:\n{context_text}\n\nQuestion:\n{query}",
                    }
                ],
            },
        ],
    )
    return resp.output_text or "No answer generated."


async def query_rag(query: str, top_k: int = 5) -> dict:
    q_vec = get_embeddings(query)
    chunks = search_similar_chunks(q_vec, top_k=top_k)
    answer = await generate_answer(query, chunks)
    return {
        "answer": answer,
        "matches": [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "section": c.section,
                "content": c.content,
            }
            for c in chunks
        ],
    }
