"""End-to-end RAG pipeline: ingest -> hybrid retrieve -> rerank -> answer with citations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_document
from .config import AppConfig
from .embeddings import EmbeddingBackend, make_backend
from .llm import LLMClient
from .retrieval import BM25Retriever, CrossEncoderReranker, RetrievalResult, ScoredChunk, rrf_fuse
from .schemas import AskResponse, Citation, RetrievedChunk
from .vector_store import FaissStore

NO_CONTEXT_ANSWER = (
    "I couldn't find relevant information in the indexed documents to answer that question. "
    "Try rephrasing, or ingest the relevant policy document first."
)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


class RAGPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        r = config.retrieval
        self.backend: EmbeddingBackend = make_backend(r.embed_backend, r.embed_model)
        self.store = FaissStore(self.backend.dim)
        self.chunks: list[Chunk] = []
        self.bm25 = BM25Retriever([])
        self.reranker = CrossEncoderReranker(r.rerank_model)
        self.llm = LLMClient(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        self.load()

    # ---- persistence ----------------------------------------------------
    @property
    def _index_dir(self) -> Path:
        return Path(self.config.paths.index_dir)

    def load(self) -> None:
        store = FaissStore.load(self._index_dir, self.backend.dim)
        # rebuild chunk list from stored metas
        self.chunks = [
            Chunk(doc_id=m["doc_id"], title=m["title"], chunk_index=m["chunk_index"], text=m["text"])
            for m in store.metas
        ]
        self.store = store
        self.bm25 = BM25Retriever([c.text for c in self.chunks])

    def _persist(self) -> None:
        self.store.save(self._index_dir)

    # ---- ingest ----------------------------------------------------------
    def ingest_documents(self, docs: list[Document], clear: bool = False) -> tuple[int, int]:
        r = self.config.retrieval
        if clear:
            self.store.clear()
            self.chunks = []
        existing = {c.doc_id for c in self.chunks}
        new_chunks: list[Chunk] = []
        for doc in docs:
            if doc.doc_id in existing:
                continue  # idempotent: skip already-indexed docs
            new_chunks.extend(
                chunk_document(doc.doc_id, doc.title, doc.text, r.chunk_size, r.chunk_overlap)
            )
        if new_chunks:
            embs = self.backend.encode([c.text for c in new_chunks])
            metas = [
                {"doc_id": c.doc_id, "title": c.title, "chunk_index": c.chunk_index, "text": c.text}
                for c in new_chunks
            ]
            self.store.add(np.asarray(embs, dtype=np.float32), metas)
            self.chunks.extend(new_chunks)
            self.bm25 = BM25Retriever([c.text for c in self.chunks])
            self._persist()
        return len({c.doc_id for c in new_chunks}), len(new_chunks)

    @property
    def doc_count(self) -> int:
        return len({c.doc_id for c in self.chunks})

    # ---- retrieval -------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None,
                 rerank: bool | None = None) -> RetrievalResult:
        r = self.config.retrieval
        top_k = top_k or r.top_k
        use_rerank = r.rerank if rerank is None else rerank
        if not self.chunks:
            return RetrievalResult(hits=[], top_sparse_score=0.0, top_dense_score=0.0)

        q_emb = self.backend.encode([query])[0]
        dense = [ScoredChunk(position=p, score=s, source="dense")
                 for p, s in self.store.search(q_emb, max(top_k * 4, r.rerank_top_n))]
        sparse = self.bm25.search(query, max(top_k * 4, r.rerank_top_n))
        top_sparse_score = sparse[0].score if sparse else 0.0
        top_dense_score = dense[0].score if dense else 0.0
        fused = rrf_fuse(dense, sparse, rrf_k=r.rrf_k,
                         dense_weight=r.dense_weight, sparse_weight=r.sparse_weight)[: max(top_k, r.rerank_top_n)]

        if use_rerank and fused:
            candidates = [(h.position, self.chunks[h.position].text) for h in fused]
            hits = self.reranker.rerank(query, candidates, top_n=top_k)
        else:
            hits = fused[:top_k]
        return RetrievalResult(hits=hits, top_sparse_score=top_sparse_score,
                               top_dense_score=top_dense_score)

    # ---- question answering ----------------------------------------------
    def answer(self, question: str, top_k: int | None = None,
               rerank: bool | None = None) -> AskResponse:
        r = self.config.retrieval
        result = self.retrieve(question, top_k=top_k, rerank=rerank)
        hits = result.hits

        # Honesty gate: answer only when the corpus shows substantive overlap with the
        # question. The sparse (BM25) check catches the common case — idf weighting
        # downranks pure stopword overlap. The dense check backstops degenerate corpora
        # (e.g. a single document, where BM25 idf is meaningless) and paraphrases with
        # no lexical overlap. Both must be weak to trigger the fallback.
        if not hits or (result.top_sparse_score < r.score_threshold
                        and result.top_dense_score < r.dense_threshold):
            return AskResponse(answer=NO_CONTEXT_ANSWER, citations=[], chunks=[],
                               model=self.llm.model if self.llm.configured else "extractive",
                               used_llm=False)

        context_blocks, citations, chunk_models = [], [], []
        for rank, hit in enumerate(hits, start=1):
            c = self.chunks[hit.position]
            context_blocks.append(f"[{rank}] {c.title}\n{c.text}")
            citations.append(Citation(doc_id=c.doc_id, title=c.title,
                                      chunk_index=c.chunk_index, score=round(hit.score, 4)))
            chunk_models.append(RetrievedChunk(citation=citations[-1], text=c.text))

        llm_answer = self.llm.chat(question, context_blocks)
        return AskResponse(answer=llm_answer.text, citations=citations, chunks=chunk_models,
                           model=llm_answer.model, used_llm=llm_answer.used_llm)
