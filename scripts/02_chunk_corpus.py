"""
scripts/02_chunk_corpus.py

Reads data/raw/corpus.jsonl, chunks every document, and writes
data/processed/chunks.jsonl — the canonical chunk set used by both
the BM25 index and the vector index, keyed by the same chunk IDs.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunking import chunk_document
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, CORPUS_PATH


def main():
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents from {CORPUS_PATH}")

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    print(f"Produced {len(all_chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens)")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved chunks to {CHUNKS_PATH}")

    avg_tokens = sum(c["n_tokens"] for c in all_chunks) / len(all_chunks)
    print(f"Average chunk size: {avg_tokens:.0f} tokens")


if __name__ == "__main__":
    main()
