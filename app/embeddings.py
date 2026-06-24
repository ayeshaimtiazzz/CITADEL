"""
app/embeddings.py

Local, free embedding model via sentence-transformers.
BAAI/bge-small-en-v1.5 — 384 dims, ~130MB, strong MTEB scores for its size,
runs comfortably on CPU.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 32, is_query: bool = False):
    """
    BGE models recommend a query instruction prefix for queries
    (not for documents) to improve retrieval quality.
    """
    model = get_embedding_model()
    if is_query:
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    return embeddings.tolist()


def embed_query(query: str):
    return embed_texts([query], is_query=True)[0]
