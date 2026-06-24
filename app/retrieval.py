"""
app/retrieval.py

Hybrid retrieval: BM25 (keyword) + Qdrant (vector), fused with
Reciprocal Rank Fusion (RRF). This is the "hybrid search" requirement.

RRF is used (rather than naive score averaging) because BM25 scores and
cosine similarity scores live on different, incomparable scales. RRF only
needs each system's *rank ordering*, which makes fusion robust without
any score-normalization heuristics.
"""

import pickle
from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import BM25_INDEX_PATH, HYBRID_TOP_K, QDRANT_COLLECTION, QDRANT_URL, RRF_K
from app.embeddings import embed_query


@lru_cache(maxsize=1)
def _load_bm25():
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_lookup"]


@lru_cache(maxsize=1)
def _get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def _tokenize(text: str):
    import re
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def bm25_search(query: str, top_k: int = HYBRID_TOP_K):
    bm25, chunk_lookup = _load_bm25()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        chunk = chunk_lookup[idx]
        results.append({**chunk, "bm25_score": float(score)})
    return results


def vector_search(query: str, top_k: int = HYBRID_TOP_K):
    client = _get_qdrant_client()
    query_vector = embed_query(query)

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )
    results = []
    for hit in response.points:
        results.append({
            "chunk_id": hit.payload["chunk_id"],
            "text": hit.payload["text"],
            "source_title": hit.payload["source_title"],
            "source_url": hit.payload["source_url"],
            "vector_score": float(hit.score),
        })
    return results


def reciprocal_rank_fusion(bm25_results: list[dict], vector_results: list[dict], k: int = RRF_K):
    """
    RRF score for a document = sum over each ranking list of 1 / (k + rank),
    where rank is 1-indexed. Documents appearing in both lists accumulate
    score from both, naturally boosting consensus results.
    """
    rrf_scores = {}
    chunk_data = {}

    for rank, doc in enumerate(bm25_results, start=1):
        cid = doc["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_data[cid] = doc

    for rank, doc in enumerate(vector_results, start=1):
        cid = doc["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_data[cid] = doc  # overwrites with vector copy; text/source same either way

    fused = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(chunk_data[cid])
        entry["rrf_score"] = score
        fused.append(entry)

    return fused


def hybrid_retrieve(query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
    """
    Runs BM25 + vector search in parallel (logically), fuses with RRF,
    and returns the top_k fused candidates ready for reranking.
    """
    bm25_results = bm25_search(query, top_k=top_k)
    vector_results = vector_search(query, top_k=top_k)
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused[:top_k]
