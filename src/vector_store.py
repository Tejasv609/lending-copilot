"""FAISS dense vector store with on-disk persistence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissStore:
    def __init__(self, dim: int) -> None:
        import faiss
        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        self.metas: list[dict] = []

    def add(self, embeddings: np.ndarray, metas: list[dict]) -> None:
        if embeddings.shape[0] == 0:
            return
        assert embeddings.shape[1] == self.dim, "embedding dim mismatch"
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.index.add((embeddings / norms).astype(np.float32))
        self.metas.extend(metas)

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_position, cosine_score)] best-first."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = query_emb.astype(np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) or 1.0)
        scores, idxs = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def clear(self) -> None:
        self.index.reset()
        self.metas = []

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "chunks.jsonl").write_text(
            "\n".join(json.dumps(m) for m in self.metas), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, dim: int) -> "FaissStore":
        import faiss
        directory = Path(directory)
        store = cls(dim)
        idx_path = directory / "vectors.faiss"
        meta_path = directory / "chunks.jsonl"
        if idx_path.exists():
            store.index = faiss.read_index(str(idx_path))
        if meta_path.exists():
            store.metas = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return store
