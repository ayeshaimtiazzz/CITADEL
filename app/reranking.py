"""
app/reranking.py

Reranking layer between hybrid retrieval and generation.

Primary: Cohere Rerank API (free tier: 1000 calls/month — plenty for a
30-50 question eval set run a few times).

Fallback: local cross-encoder (sentence-transformers) if no Cohere key
is set, so the whole pipeline still runs 100% free/offline if needed.
"""

from functools import lru_cache

import cohere

from app.config import COHERE_API_KEY, RERANK_TOP_K


@lru_cache(maxsize=1)
def _get_cohere_client():
    return cohere.Client(COHERE_API_KEY)


@lru_cache(maxsize=1)
def _get_local_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K, use_cohere: bool = True):
    """
    candidates: list of dicts each containing at least "text".
    Returns the top_k candidates, each annotated with "rerank_score",
    sorted descending by that score.
    """
    if not candidates:
        return []

    if use_cohere and COHERE_API_KEY:
        return _rerank_cohere(query, candidates, top_k)
    return _rerank_local(query, candidates, top_k)


def _rerank_cohere(query: str, candidates: list[dict], top_k: int):
    client = _get_cohere_client()
    docs = [c["text"] for c in candidates]

    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=min(top_k, len(docs)),
    )

    reranked = []
    for result in response.results:
        entry = dict(candidates[result.index])
        entry["rerank_score"] = float(result.relevance_score)
        reranked.append(entry)
    return reranked


def _rerank_local(query: str, candidates: list[dict], top_k: int):
    model = _get_local_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for candidate, score in scored[:top_k]:
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        reranked.append(entry)
    return reranked
