"""Configuration loading: config.yaml + LENDING_* env var overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.1"
    temperature: float = 0.1
    max_tokens: int = 512


class RetrievalConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10
    score_threshold: float = 4.0  # min top BM25 score; below -> "no relevant context"
    dense_threshold: float = 0.3  # min top dense cosine; backstop for tiny corpora/paraphrases
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_backend: str = "sentence_transformers"  # or "hash"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    sample_docs: str = "data/sample_docs"
    uploads: str = "data/uploads"
    index_dir: str = "data/index"


class AppConfig(BaseModel):
    app: dict = Field(default_factory=lambda: {"host": "0.0.0.0", "port": 8000})
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


_ENV_MAP = {
    # section -> {FIELD_SUFFIX: (attr, cast)}
    "llm": {
        "BASE_URL": ("base_url", str),
        "API_KEY": ("api_key", str),
        "MODEL": ("model", str),
        "TEMPERATURE": ("temperature", float),
        "MAX_TOKENS": ("max_tokens", int),
    },
    "retrieval": {
        "CHUNK_SIZE": ("chunk_size", int),
        "CHUNK_OVERLAP": ("chunk_overlap", int),
        "TOP_K": ("top_k", int),
        "RRF_K": ("rrf_k", int),
        "DENSE_WEIGHT": ("dense_weight", float),
        "SPARSE_WEIGHT": ("sparse_weight", float),
        "RERANK": ("rerank", lambda v: v.lower() in ("1", "true", "yes")),
        "RERANK_MODEL": ("rerank_model", str),
        "RERANK_TOP_N": ("rerank_top_n", int),
        "SCORE_THRESHOLD": ("score_threshold", float),
    "DENSE_THRESHOLD": ("dense_threshold", float),
        "EMBED_MODEL": ("embed_model", str),
        "EMBED_BACKEND": ("embed_backend", str),
    },
    "app": {
        "HOST": ("host", str),
        "PORT": ("port", int),
    },
    "paths": {
        "DATA_DIR": ("data_dir", str),
        "SAMPLE_DOCS": ("sample_docs", str),
        "UPLOADS": ("uploads", str),
        "INDEX_DIR": ("index_dir", str),
    },
}


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    for section, fields in _ENV_MAP.items():
        target = getattr(cfg, section)
        updates: dict[str, Any] = {}
        for suffix, (attr, cast) in fields.items():
            env_key = f"LENDING_{section.upper()}_{suffix}"
            raw = os.environ.get(env_key)
            if raw is not None and raw != "":
                try:
                    updates[attr] = cast(raw)
                except (ValueError, TypeError):
                    continue
        if updates:
            if isinstance(target, BaseModel):
                setattr(cfg, section, target.model_copy(update=updates))
            else:  # plain dict (app)
                target.update(updates)
    return cfg


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load config.yaml (if present) and apply LENDING_* env overrides."""
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    cfg = AppConfig.model_validate(data)
    return _apply_env_overrides(cfg)
