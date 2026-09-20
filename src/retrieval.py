"""Sparse retrieval (BM25), hybrid RRF fusion, and optional cross-encoder reranking."""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    position: int       # index into the chunk list
    score: float
    source: str         # "dense" | "sparse" | "fused" | "rerank"


@dataclass
class RetrievalResult:
    hits: list[ScoredChunk]
    top_sparse_score: float  # raw BM25 of the best sparse hit; 0.0 when sparse found nothing.
    top_dense_score: float   # cosine similarity of the best dense hit; 0.0 on empty index.
    # ^ honesty signals. BM25's idf weighting means a low sparse value indicates the
    # corpus shares no substantive vocabulary with the question (only stopword overlap).
    # The dense score backstops degenerate corpora (e.g. a single document) where BM25
    # idf is meaningless, and paraphrases with no lexical overlap.


class BM25Retriever:
    def __init__(self, texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi
        self.texts = texts
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(position=i, score=float(scores[i]), source="sparse")
                for i in order if scores[i] > 0]


def rrf_fuse(dense: list[ScoredChunk], sparse: list[ScoredChunk],
             rrf_k: int = 60, dense_weight: float = 0.6,
             sparse_weight: float = 0.4) -> list[ScoredChunk]:
    """Weighted reciprocal-rank fusion over dense and sparse ranked lists."""
    fused: dict[int, float] = {}
    for rank, hit in enumerate(dense, start=1):
        fused[hit.position] = fused.get(hit.position, 0.0) + dense_weight / (rrf_k + rank)
    for rank, hit in enumerate(sparse, start=1):
        fused[hit.position] = fused.get(hit.position, 0.0) + sparse_weight / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredChunk(position=pos, score=score, source="fused") for pos, score in ordered]


class CrossEncoderReranker:
    """Optional reranker. Loads lazily; disabled entirely if the model can't load."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = False

    def rerank(self, query: str, candidates: list[tuple[int, str]],
               top_n: int) -> list[ScoredChunk]:
        if self._failed:
            return [ScoredChunk(position=p, score=0.0, source="rerank") for p, _ in candidates[:top_n]]
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            pairs = [(query, text) for _, text in candidates]
            scores = self._model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip([p for p, _ in candidates], scores),
                            key=lambda x: x[1], reverse=True)[:top_n]
            return [ScoredChunk(position=p, score=float(s), source="rerank") for p, s in ranked]
        except Exception:
            self._failed = True
            return [ScoredChunk(position=p, score=0.0, source="rerank") for p, _ in candidates[:top_n]]
