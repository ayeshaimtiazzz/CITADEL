"""
scripts/06_smoke_test.py

Quick sanity check that every component is wired up correctly before
running the full evaluation. Run this after steps 1-4 (corpus fetch,
chunking, vector index, BM25 index) and after setting your .env keys.

Usage:
    python scripts/06_smoke_test.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import run_query


def main():
    test_query = "What is the Apollo program?"
    print(f"Running test query: '{test_query}'\n")

    result = run_query(test_query, use_reranking=True)

    print("=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for src in result["sources"]:
        print(f"  [{src['index']}] {src['title']}")
    print("\nTIMINGS:")
    for stage, ms in result["timings"].items():
        print(f"  {stage}: {ms} ms")
    print("=" * 60)
    print("\nSmoke test passed — pipeline is fully wired.")


if __name__ == "__main__":
    main()
