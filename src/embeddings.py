"""Embedding backends: sentence-transformers (default) or a fast deterministic hash embedder (tests)."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbeddingBackend:
    """Deterministic, dependency-light embeddings: hashed token counts, L2-normalized.

    Good enough for tests and offline demos; lexical overlap drives similarity.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _token_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
            # signed hashing reduces collisions
            if (h >> 7) & 1:
                vec[h % self.dim] *= -1.0
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.stack([self._token_vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


class SentenceTransformerBackend:
    """Dense embeddings via sentence-transformers. Model loads lazily."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return self._dim  # type: ignore[return-value]

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def make_backend(name: str, model_name: str) -> EmbeddingBackend:
    if name == "hash":
        return HashEmbeddingBackend()
    if name == "sentence_transformers":
        return SentenceTransformerBackend(model_name)
    raise ValueError(f"Unknown embedding backend: {name!r}")
