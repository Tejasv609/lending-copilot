"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank: bool | None = None


class Citation(BaseModel):
    doc_id: str
    title: str
    chunk_index: int
    score: float


class RetrievedChunk(BaseModel):
    citation: Citation
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]
    model: str
    used_llm: bool


class IngestResponse(BaseModel):
    docs_added: int
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    docs_indexed: int
    chunks_indexed: int
    llm_configured: bool
    llm_model: str
