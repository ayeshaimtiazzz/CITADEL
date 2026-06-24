"""
scripts/03_build_vector_index.py

Embeds every chunk (local model, free) and upserts into Qdrant.
Requires Qdrant running locally:
    docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

from app.config import CHUNKS_PATH, EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL
from app.embeddings import embed_texts


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    client = QdrantClient(url=QDRANT_URL)

    # Recreate collection fresh each run for reproducibility
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
    )
    print(f"Created collection '{QDRANT_COLLECTION}' in Qdrant at {QDRANT_URL}")

    batch_size = 32
    point_id = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + upserting"):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        vectors = embed_texts(texts, is_query=False)

        points = []
        for chunk, vector in zip(batch, vectors):
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "source_title": chunk["source_title"],
                        "source_url": chunk["source_url"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )
            point_id += 1

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    count = client.count(collection_name=QDRANT_COLLECTION).count
    print(f"Done. Qdrant collection '{QDRANT_COLLECTION}' now has {count} points.")


if __name__ == "__main__":
    main()
