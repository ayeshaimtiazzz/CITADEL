"""
scripts/05_run_evaluation.py

Runs the golden Q&A set through the full pipeline TWICE — once with
reranking enabled, once disabled — and scores both runs with RAGAS on:
  - faithfulness (is the answer grounded in retrieved context?)
  - answer_relevancy (does the answer address the question?)
  - context_precision (are retrieved chunks relevant?)
  - context_recall (did retrieval find what's needed to answer?)

Saves a before/after comparison report — this is the project's key
portfolio artifact, proving reranking's measured impact.

Usage:
    python scripts/05_run_evaluation.py                 # full 42-question set
    python scripts/05_run_evaluation.py --limit 15       # fast subset (stratified)
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from tqdm import tqdm

from app.config import EVAL_RESULTS_DIR, EVAL_SET_PATH, OLLAMA_HOST, OLLAMA_MODEL
from app.embeddings import embed_texts
from app.pipeline import run_query


class CitadelRagasEmbeddings(BaseRagasEmbeddings):
    """
    Thin RAGAS-compatible wrapper around our own local sentence-transformers
    model (app/embeddings.py). Avoids langchain_community entirely, which has
    broken transitive imports (e.g. ChatVertexAI) in some installed versions.
    """

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text], is_query=True)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, is_query=False)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


# RAGAS defaults to OpenAI for its internal "judge" LLM and for embeddings.
# We swap both for free, fully local alternatives: Ollama as judge LLM, and
# the same local sentence-transformers model we already use for retrieval.
# RAGAS 0.4.x requires the LLM to satisfy BaseRagasLLM, so we wrap the raw
# LangChain ChatOllama instance with LangchainLLMWrapper.
RAGAS_JUDGE_LLM = LangchainLLMWrapper(
    ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL, temperature=0)
)
RAGAS_EMBEDDINGS = CitadelRagasEmbeddings()


def load_golden_set(limit: int | None = None):
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        full_set = json.load(f)

    if limit is None or limit >= len(full_set):
        return full_set

    # Stratified subset: preserve the ratio of unanswerable (hallucination-test)
    # questions rather than just slicing the first N, which would silently drop
    # the hallucination-resistance test if all unanswerable items are at the end.
    answerable = [q for q in full_set if "UNANSWERABLE" not in q["ground_truth"]]
    unanswerable = [q for q in full_set if "UNANSWERABLE" in q["ground_truth"]]

    frac_unanswerable = len(unanswerable) / len(full_set)
    n_unanswerable = max(1, round(limit * frac_unanswerable))
    n_answerable = limit - n_unanswerable

    subset = answerable[:n_answerable] + unanswerable[:n_unanswerable]
    return subset


def run_pipeline_for_eval(golden_set: list[dict], use_reranking: bool) -> Dataset:
    # RAGAS 0.4.x metrics expect this column schema (confirmed via each
    # metric's _required_columns): user_input, response, retrieved_contexts,
    # and reference for recall/precision against ground truth.
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    label = "WITH reranking" if use_reranking else "WITHOUT reranking"
    for item in tqdm(golden_set, desc=f"Running pipeline ({label})"):
        result = run_query(
            item["question"],
            use_reranking=use_reranking,
            trace_name=f"eval-{'rerank' if use_reranking else 'norerank'}",
        )
        rows["user_input"].append(item["question"])
        rows["response"].append(result["answer"])
        rows["retrieved_contexts"].append([c["text"] for c in result["final_chunks"]])
        rows["reference"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run on a stratified subset of N questions instead of the full golden set "
             "(preserves the ratio of unanswerable hallucination-test questions). "
             "Useful for a fast end-to-end run with local/CPU LLMs.",
    )
    args = parser.parse_args()

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    golden_set = load_golden_set(limit=args.limit)

    n_unanswerable = sum(1 for q in golden_set if "UNANSWERABLE" in q["ground_truth"])
    print(f"Loaded {len(golden_set)} golden Q&A pairs "
          f"({n_unanswerable} unanswerable / hallucination-test questions)"
          + (f" [subset of full set, --limit {args.limit}]" if args.limit else ""))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    summary_rows = []

    for use_reranking in [True, False]:
        dataset = run_pipeline_for_eval(golden_set, use_reranking=use_reranking)
        scores = evaluate(
            dataset,
            metrics=metrics,
            llm=RAGAS_JUDGE_LLM,
            embeddings=RAGAS_EMBEDDINGS,
        )
        df = scores.to_pandas()

        tag = "with_reranking" if use_reranking else "without_reranking"
        df.to_csv(f"{EVAL_RESULTS_DIR}/{tag}_detailed.csv", index=False)

        row = {"condition": tag}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            row[metric] = df[metric].mean()
        summary_rows.append(row)

        print(f"\n=== {tag} ===")
        print(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{EVAL_RESULTS_DIR}/comparison_summary.csv", index=False)

    print("\n=== Reranking A/B Comparison ===")
    print(summary_df.to_string(index=False))
    print(f"\nFull report saved to {EVAL_RESULTS_DIR}/comparison_summary.csv")
    if args.limit:
        print(f"NOTE: this run used a {len(golden_set)}-question subset (--limit {args.limit}), "
              f"not the full {EVAL_SET_PATH} set. Mention this in your README results table.")


if __name__ == "__main__":
    main()