# 🛰️ CITADEL

**C**ontext-**I**ndexed **T**rustworthy **A**nswering with **D**ual-**E**ncoder **L**ayered retrieval

A production-grade RAG system with a real evaluation harness — not a notebook demo. CITADEL combines hybrid retrieval (BM25 + vector search, fused via Reciprocal Rank Fusion), a reranking layer, and quantitative RAGAS scoring to measure — not assume — retrieval quality and answer faithfulness. Built entirely on free-tier tools and local models: **zero paid API usage.**

## Why this exists

Most "RAG chatbot" portfolio projects stop at "it answers questions." CITADEL exists to answer a harder question: **how do you know your RAG system is actually good, and does each architectural choice (hybrid search, reranking) measurably help?** The centerpiece is a before/after evaluation report quantifying reranking's effect on faithfulness, relevancy, and context precision/recall.

## Architecture

```
                         ┌─────────────────┐
                         │   User Query     │
                         └────────┬─────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                     ▼
        ┌───────────────┐                   ┌──────────────────┐
        │   BM25 Index   │                   │  Qdrant (vector)  │
        │  (keyword)     │                   │  bge-small-en-v1.5│
        └───────┬───────┘                   └─────────┬─────────┘
                │            top 20 each                │
                └──────────────┬───────────────────────┘
                                ▼
                  ┌─────────────────────────┐
                  │ Reciprocal Rank Fusion   │
                  │   (RRF, k=60)            │
                  └────────────┬─────────────┘
                                ▼
                  ┌─────────────────────────┐
                  │  Reranker                │
                  │  Cohere rerank-v3        │
                  │  (or local cross-encoder)│
                  └────────────┬─────────────┘
                          top 5 chunks
                                ▼
                  ┌─────────────────────────┐
                  │  Ollama (Llama 3.1 8B)   │
                  │  Local, no API key       │
                  │  Grounded generation     │
                  │  + inline citations      │
                  └────────────┬─────────────┘
                                ▼
                     ┌─────────────────┐
                     │  Answer + Sources │
                     └─────────────────┘

   Every stage traced end-to-end in Langfuse (self-hosted, free).
```

## Stack (100% free)

| Component | Tool | Why |
|---|---|---|
| Corpus | Wikipedia API (space exploration cluster, ~50 articles) | Clean text, no scraping pain, no rate limits |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, sentence-transformers) | Free, runs on CPU, strong MTEB score for its size |
| Vector DB | Qdrant (self-hosted, Docker) | Free, fast, simple API |
| Keyword search | `rank_bm25` | Free, no infra |
| Fusion | Reciprocal Rank Fusion (RRF) | Combines two incomparable score scales using only rank order |
| Reranker | Cohere Rerank v3 (free tier, 1000 calls/mo) — local cross-encoder fallback | Cohere quality with zero-cost fallback if no key |
| LLM | Ollama (Llama 3.1 8B, fully local) | Zero API key, no rate limits, runs entirely on your machine |
| Evaluation | RAGAS (judge LLM = Ollama, not OpenAI) | Faithfulness, relevancy, context precision/recall |
| Observability | Langfuse (self-hosted, Docker) | Full request tracing, free |
| API | FastAPI | Standard, fast to stand up |
| UI | Streamlit | Fast iteration, eval dashboard sidebar |

## Project structure

```
citadel-rag/
├── app/
│   ├── config.py          # central settings, loaded from .env
│   ├── chunking.py         # token-aware chunking with overlap
│   ├── embeddings.py        # local embedding model wrapper
│   ├── retrieval.py         # BM25 + vector search + RRF fusion
│   ├── reranking.py         # Cohere / local cross-encoder reranking
│   ├── generation.py        # Ollama-based grounded generation
│   └── pipeline.py          # orchestrates retrieval -> rerank -> generation
├── api/
│   └── main.py              # FastAPI app (POST /query)
├── scripts/
│   ├── 01_fetch_corpus.py        # pull Wikipedia corpus
│   ├── 02_chunk_corpus.py        # chunk into data/processed/chunks.jsonl
│   ├── 03_build_vector_index.py  # embed + upsert into Qdrant
│   ├── 04_build_bm25_index.py    # build BM25 index over same chunks
│   ├── 05_run_evaluation.py      # RAGAS eval, with/without reranking A/B
│   └── 06_smoke_test.py          # quick end-to-end sanity check
├── eval/
│   ├── golden_qa.json        # 42 hand-written Q&A pairs (incl. 7 unanswerable)
│   └── results/              # CSV reports land here after evaluation
├── streamlit_app.py          # UI with eval dashboard
├── docker-compose.yml         # Qdrant + Langfuse
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url> citadel-rag
cd citadel-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model (fully local, no signup)

Download Ollama from [ollama.com](https://ollama.com), install it, then pull the model:

```bash
ollama pull llama3.1:8b
```

Optionally, get a free Cohere key for reranking: [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) — 1000 rerank calls/month free. (Optional — the pipeline falls back to a local cross-encoder if you skip this.)

```bash
cp .env.example .env
# Defaults already point at Ollama running locally — just fill in COHERE_API_KEY if you have one
```

### 3. Start infrastructure

```bash
docker compose up -d
```

This starts Qdrant (`localhost:6333`) and Langfuse (`localhost:3000`).

On first run, visit `http://localhost:3000`, sign up locally (it's self-hosted, your data stays on your machine), create a project, and copy the generated public/secret keys into `.env`.

### 4. Build the corpus and indexes

```bash
python scripts/01_fetch_corpus.py        # ~50 Wikipedia articles, ~1-2 min
python scripts/02_chunk_corpus.py        # chunk into 512-token pieces
python scripts/03_build_vector_index.py  # embed + upsert into Qdrant, ~2-5 min on CPU
python scripts/04_build_bm25_index.py    # build keyword index
```

### 5. Smoke test

```bash
python scripts/06_smoke_test.py
```

You should see a grounded answer with citations and a timing breakdown.

### 6. Run the evaluation (the key portfolio artifact)

```bash
python scripts/05_run_evaluation.py
```

This runs all 42 golden questions through the pipeline **twice** — once with reranking, once without — scores both with RAGAS, and writes:

- `eval/results/with_reranking_detailed.csv`
- `eval/results/without_reranking_detailed.csv`
- `eval/results/comparison_summary.csv` ← the before/after numbers

### 7. Launch the API and UI

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run streamlit_app.py
```

Query via API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Artemis program?", "use_reranking": true}'
```

## Evaluation methodology

- **Golden set**: 42 hand-written Q&A pairs drawn directly from the corpus, covering missions, agencies, technology, and history. 7 are deliberately unanswerable from the corpus (e.g. "What's the ISS wifi password?") to test hallucination resistance — a faithful system should say it cannot answer rather than fabricate.
- **Metrics** (via RAGAS, judged by a local Ollama model rather than OpenAI):
  - **Faithfulness** — is every claim in the answer supported by the retrieved context?
  - **Answer relevancy** — does the answer actually address the question asked?
  - **Context precision** — of the retrieved chunks, how many were actually relevant?
  - **Context recall** — did retrieval surface everything needed to fully answer?
- **A/B comparison**: the entire golden set runs twice — hybrid retrieval alone vs. hybrid retrieval + reranking — isolating reranking's measured contribution rather than asserting it helps.

*(Run `scripts/05_run_evaluation.py` and paste your `comparison_summary.csv` numbers here once you have them — this table is what recruiters will actually read.)*

| Metric | Without Reranking | With Reranking | Δ |
condition,faithfulness,answer_relevancy,context_precision,context_recall
with_reranking,,0.3508301775242504,,0.8809523809523808
without_reranking,,0.4357906546125016,,0.9166666666666666


## Design notes

- **Why RRF over score normalization**: BM25 and cosine similarity scores aren't on comparable scales, and naive min-max normalization is sensitive to outliers. RRF only uses rank position, making fusion robust without tuning a weighting scheme.
- **Why a local embedding model**: keeps embedding cost and latency at zero, and demonstrates the system doesn't require a paid API to function — only generation and (optionally) reranking call external services.
- **Why Ollama for both generation and RAGAS judging**: removes any external API dependency entirely — no keys, no rate limits, fully reproducible offline. Llama 3.1 8B is strong enough for grounded QA and as a competent judge model for RAGAS's LLM-based metrics, though it will be slower than a hosted API on CPU-only machines.
- **Hallucination test design**: unanswerable questions are the simplest, highest-signal check that the system is grounded rather than fluent-but-wrong.

## Possible extensions

- Swap the Wikipedia corpus for a domain you care about (legal, financial, internal docs) — the pipeline is corpus-agnostic.
- Add a query-rewriting step before retrieval (e.g. HyDE) and measure its effect the same way reranking is measured here.
- Deploy the FastAPI backend to Railway or Fly.io free tier for a live demo link.
