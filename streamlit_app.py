"""
streamlit_app.py

Minimal Streamlit UI for CITADEL:
  - Main panel: ask a question, see the answer with highlighted citations
  - A/B toggle: reranking on/off
  - Sidebar: eval dashboard showing the latest RAGAS comparison results,
    if scripts/05_run_evaluation.py has been run

Run with:
    streamlit run streamlit_app.py
"""

import os

import pandas as pd
import streamlit as st

from app.config import EVAL_RESULTS_DIR
from app.pipeline import run_query

st.set_page_config(page_title="CITADEL", page_icon="🛰️", layout="wide")

st.title("🛰️ CITADEL")
st.caption("Context-Indexed Trustworthy Answering with Dual-Encoder Layered retrieval")

# --- Sidebar: eval dashboard ---
with st.sidebar:
    st.header("📊 Evaluation Dashboard")
    summary_path = f"{EVAL_RESULTS_DIR}/comparison_summary.csv"
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df_display = df.set_index("condition").T
        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
        st.dataframe(df_display.style.format("{:.3f}"))
        st.caption("Faithfulness, relevancy, precision, recall — reranking ON vs OFF")
    else:
        st.info("No eval results yet. Run `python scripts/05_run_evaluation.py` first.")

    st.divider()
    st.markdown(
        "**Stack:** Qdrant + BM25 (RRF fusion) → Cohere/cross-encoder rerank "
        "→ Ollama (Llama 3.1, local) generation, scored with RAGAS, traced with Langfuse."
    )

# --- Main panel ---
use_reranking = st.toggle("Use reranking", value=True, help="Toggle off to compare raw hybrid retrieval vs reranked results")

question = st.text_input("Ask a question about space exploration:", placeholder="What is the Artemis program?")

if st.button("Ask CITADEL", type="primary") and question:
    with st.spinner("Retrieving, reranking, and generating..."):
        result = run_query(question, use_reranking=use_reranking)

    st.subheader("Answer")
    st.write(result["answer"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retrieval", f"{result['timings']['retrieval_ms']} ms")
    col2.metric("Reranking", f"{result['timings']['reranking_ms']} ms")
    col3.metric("Generation", f"{result['timings']['generation_ms']} ms")
    col4.metric("Total", f"{result['timings']['total_ms']} ms")

    st.subheader("Sources")
    for src in result["sources"]:
        st.markdown(f"**[{src['index']}]** [{src['title']}]({src['url']})")

    with st.expander("🔍 Retrieved chunks (before reranking)"):
        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            st.markdown(f"**{i}. {chunk['source_title']}** (RRF score: {chunk.get('rrf_score', 0):.4f})")
            st.text(chunk["text"][:300] + "...")

    with st.expander("✅ Final chunks sent to LLM (after reranking)"):
        for i, chunk in enumerate(result["final_chunks"], start=1):
            score_label = "rerank_score" if use_reranking else "rrf_score"
            st.markdown(f"**{i}. {chunk['source_title']}** ({score_label}: {chunk.get(score_label, 0):.4f})")
            st.text(chunk["text"][:300] + "...")
