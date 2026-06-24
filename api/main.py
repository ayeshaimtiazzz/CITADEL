"""
api/main.py

FastAPI app exposing the CITADEL RAG pipeline.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pipeline import run_query

app = FastAPI(
    title="CITADEL",
    description="Context-Indexed Trustworthy Answering with Dual-Encoder Layered retrieval",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    use_reranking: bool = Field(True, description="Toggle reranking on/off (for A/B comparison)")


class SourceItem(BaseModel):
    index: int
    title: str
    url: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    used_reranking: bool
    timings: dict


@app.get("/health")
def health():
    return {"status": "ok", "service": "CITADEL"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    try:
        result = run_query(req.question, use_reranking=req.use_reranking)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=result["sources"],
        used_reranking=result["used_reranking"],
        timings=result["timings"],
    )
