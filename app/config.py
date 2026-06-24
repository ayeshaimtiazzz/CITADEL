"""
app/config.py

Centralized configuration loaded from environment variables (.env).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Ollama (LLM, fully local, free) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")

# --- Cohere (Reranker) ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "citadel_chunks")

# --- Langfuse ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# --- Embedding model (local, free) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, fast, strong for its size
EMBEDDING_DIM = 384

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Retrieval ---
HYBRID_TOP_K = 20       # candidates pulled from fused BM25 + vector search
RERANK_TOP_K = 5        # final chunks sent to the LLM after reranking
RRF_K = 60               # Reciprocal Rank Fusion constant (standard default)

# --- Paths ---
CORPUS_PATH = "data/raw/corpus.jsonl"
CHUNKS_PATH = "data/processed/chunks.jsonl"
BM25_INDEX_PATH = "data/processed/bm25_index.pkl"
EVAL_SET_PATH = "eval/golden_qa.json"
EVAL_RESULTS_DIR = "eval/results"
