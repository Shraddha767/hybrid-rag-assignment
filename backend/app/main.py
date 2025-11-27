from dotenv import load_dotenv
load_dotenv()
# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from rag.pipeline import HybridRAGPipeline

app = FastAPI(
    title="Hybrid RAG System",
    description="BM25 + Dense + Hybrid Fusion + Cross-Encoder Reranker + LLM",
    version="1.0.0",
)

# Build pipeline at startup
pipeline = HybridRAGPipeline()


class AskRequest(BaseModel):
    query: str


class ScoreEntry(BaseModel):
    chunk_id: int
    dense_score: float
    sparse_score: float
    final_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    scores: List[ScoreEntry]
    latency_ms: int


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Example input:
    {
      "query": "What are the main principles discussed in the document?"
    }
    """
    result = pipeline.answer_question(req.query)
    return result
