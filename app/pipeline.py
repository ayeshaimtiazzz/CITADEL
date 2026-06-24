"""
app/pipeline.py

Orchestrates the full CITADEL pipeline:
  query -> hybrid retrieval (BM25 + vector, RRF-fused) -> rerank -> generation

Returns timing breakdown per stage and (optionally) logs a full trace to
Langfuse for observability.
"""

import time
from functools import lru_cache

from app.config import HYBRID_TOP_K, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, RERANK_TOP_K
from app.generation import generate_answer
from app.reranking import rerank
from app.retrieval import hybrid_retrieve


@lru_cache(maxsize=1)
def _get_langfuse_client():
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return None
    from langfuse import Langfuse
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )


def run_query(
    query: str,
    use_reranking: bool = True,
    hybrid_top_k: int = HYBRID_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    trace_name: str = "citadel-query",
):
    """
    Runs the full pipeline for a single query.

    use_reranking=False is the A/B toggle: skips the rerank step and feeds
    the top `rerank_top_k` RRF-fused candidates straight to generation,
    so you can measure reranking's effect on RAGAS scores.

    Returns a dict with the answer, sources, retrieved/reranked chunks,
    and a timing breakdown for each stage (in milliseconds).

    Langfuse tracing (v2.x SDK, matching the self-hosted langfuse/langfuse:2
    server) is fully optional: if no keys are configured, or if the Langfuse
    server is unreachable, tracing silently no-ops and the pipeline still
    returns a complete result. The v2 SDK uses trace()/span(), each
    returning an object with its own .update() method to set output.
    """
    langfuse = _get_langfuse_client()
    timings = {}

    if langfuse:
        try:
            trace = langfuse.trace(name=trace_name, input={"query": query})
            result = _run_pipeline_stages(
                query, use_reranking, hybrid_top_k, rerank_top_k, timings, trace
            )
            trace.update(output={"answer": result["answer"]})
        except Exception as e:
            print(f"[langfuse] tracing failed, continuing without it: {e}")
            result = _run_pipeline_stages(
                query, use_reranking, hybrid_top_k, rerank_top_k, timings, None
            )
    else:
        result = _run_pipeline_stages(
            query, use_reranking, hybrid_top_k, rerank_top_k, timings, None
        )

    timings["total_ms"] = round(sum(timings.values()), 1)
    result["timings"] = timings
    return result


def _run_pipeline_stages(query, use_reranking, hybrid_top_k, rerank_top_k, timings, trace):
    # --- Stage 1: hybrid retrieval ---
    t0 = time.perf_counter()
    if trace:
        span = trace.span(name="hybrid_retrieval", input={"query": query, "top_k": hybrid_top_k})
        candidates = hybrid_retrieve(query, top_k=hybrid_top_k)
        span.update(output={
            "n_candidates": len(candidates),
            "chunk_ids": [c["chunk_id"] for c in candidates],
        })
    else:
        candidates = hybrid_retrieve(query, top_k=hybrid_top_k)
    timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # --- Stage 2: reranking (or pass-through for A/B comparison) ---
    t0 = time.perf_counter()
    if trace:
        span = trace.span(
            name="reranking", input={"used_reranking": use_reranking, "top_k": rerank_top_k}
        )
        if use_reranking:
            final_chunks = rerank(query, candidates, top_k=rerank_top_k)
        else:
            final_chunks = candidates[:rerank_top_k]
        span.update(output={"chunk_ids": [c["chunk_id"] for c in final_chunks]})
    else:
        if use_reranking:
            final_chunks = rerank(query, candidates, top_k=rerank_top_k)
        else:
            final_chunks = candidates[:rerank_top_k]
    timings["reranking_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # --- Stage 3: generation ---
    t0 = time.perf_counter()
    if trace:
        generation = trace.generation(name="generation")
        result = generate_answer(query, final_chunks)
        generation.update(output={"answer": result["answer"]}, model=result.get("model"))
    else:
        result = generate_answer(query, final_chunks)
    timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "used_reranking": use_reranking,
        "retrieved_chunks": candidates,
        "final_chunks": final_chunks,
    }