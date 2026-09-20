"""Shared pytest fixtures: an isolated RAG pipeline with the fast hash embedder."""
from __future__ import annotations

import os

# src.main builds its module-level app at import time; keep that import light
# (no sentence-transformers/torch) regardless of test execution order.
os.environ.setdefault("LENDING_RETRIEVAL_EMBED_BACKEND", "hash")

import pytest

from src.config import AppConfig
from src.pipeline import Document, RAGPipeline


@pytest.fixture()
def test_config(tmp_path):
    cfg = AppConfig()
    cfg.retrieval.embed_backend = "hash"
    cfg.retrieval.rerank = False
    cfg.retrieval.chunk_size = 200
    cfg.retrieval.chunk_overlap = 40
    # BM25 scores scale with corpus size; tiny test corpora need a lower honesty gate.
    cfg.retrieval.score_threshold = 0.5
    cfg.paths.index_dir = str(tmp_path / "index")
    cfg.paths.uploads = str(tmp_path / "uploads")
    cfg.paths.sample_docs = str(tmp_path / "sample_docs")
    return cfg


@pytest.fixture()
def pipeline(test_config):
    return RAGPipeline(test_config)


@pytest.fixture()
def seeded_pipeline(pipeline):
    docs = [
        Document(doc_id="rates", title="Interest Rate Card",
                 text="Personal loans are priced by CIBIL band. Borrowers with CIBIL 750 and above "
                      "get 11.5% to 13.0% per annum on a reducing balance. The processing fee is 2% "
                      "of the loan amount plus GST, capped at INR 25,000."),
        Document(doc_id="kyc", title="KYC Checklist",
                 text="Every borrower must complete KYC before disbursement. PAN card is mandatory "
                      "as identity proof. Address proof can be Aadhaar or a utility bill not older "
                      "than 2 months. High-risk customers need a KYC refresh every 2 years."),
        Document(doc_id="collections", title="Collections Playbook",
                 text="Accounts are bucketed by days past due. Field visits start at 31 DPD for "
                      "ticket sizes above INR 2,00,000. Accounts at 180+ DPD with no payment activity "
                      "for 90 days are recommended for write-off."),
    ]
    pipeline.ingest_documents(docs)
    return pipeline
