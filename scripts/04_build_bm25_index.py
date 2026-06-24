"""
scripts/04_build_bm25_index.py

Builds a BM25 keyword index over the same chunks used for vector search,
keyed by the same chunk_id strings so results can be fused later.
Saves a pickled index + chunk lookup table.
"""

import json
import os
import pickle
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi

from app.config import BM25_INDEX_PATH, CHUNKS_PATH


def simple_tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def main():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    tokenized_corpus = [simple_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # chunk_lookup maps the index position in the BM25 corpus -> chunk metadata,
    # so a BM25 score at position i corresponds to chunks[i].
    chunk_lookup = [
        {
            "chunk_id": c["id"],
            "text": c["text"],
            "source_title": c["source_title"],
            "source_url": c["source_url"],
        }
        for c in chunks
    ]

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_lookup": chunk_lookup}, f)

    print(f"Saved BM25 index to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
