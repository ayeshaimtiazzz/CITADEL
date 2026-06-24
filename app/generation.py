"""
app/generation.py

Generation step: builds a grounded prompt from reranked chunks and calls
a locally running Ollama model (default: llama3.1:8b). Fully local, no
API key, no rate limits, no network dependency at all once the model is
pulled.
"""

import ollama

from app.config import OLLAMA_HOST, OLLAMA_MODEL

SYSTEM_PROMPT = """You are CITADEL, a careful research assistant that answers \
questions ONLY using the provided source excerpts.

Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say exactly: \
"I cannot answer this from the provided sources." Do not guess.
3. Cite sources inline using the bracketed numbers, e.g. [1], [2], matching the \
excerpt numbers given to you.
4. Be concise and factual. Do not pad the answer with filler.
"""

_client = ollama.Client(host=OLLAMA_HOST)


def build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] (Source: {chunk['source_title']})\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[dict], model: str = None) -> dict:
    """
    Returns {"answer": str, "model": str, "sources": list[dict]}.
    `chunks` should already be the final reranked top-N chunks.
    """
    context_block = build_context_block(chunks)
    model_name = model or OLLAMA_MODEL

    user_prompt = f"""Context excerpts:

{context_block}

Question: {query}

Answer the question using only the excerpts above, with inline citations like [1]."""

    response = _client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.1,
            "num_predict": 600,  # Ollama's equivalent of max_tokens
        },
    )

    answer = response["message"]["content"]

    sources = [
        {"index": i + 1, "title": c["source_title"], "url": c.get("source_url", "")}
        for i, c in enumerate(chunks)
    ]

    return {
        "answer": answer,
        "model": model_name,
        "sources": sources,
    }
