"""Retrieval quality tests on a small seeded corpus (hash embedder, no reranker)."""
from src.pipeline import NO_CONTEXT_ANSWER


def test_query_returns_right_document(seeded_pipeline):
    result = seeded_pipeline.retrieve("What interest rate do I get with CIBIL 760?", top_k=3)
    assert result.hits
    top = seeded_pipeline.chunks[result.hits[0].position]
    assert top.doc_id == "rates"


def test_kyc_query_returns_kyc_doc(seeded_pipeline):
    result = seeded_pipeline.retrieve("Which identity proof is mandatory for KYC?", top_k=3)
    top = seeded_pipeline.chunks[result.hits[0].position]
    assert top.doc_id == "kyc"


def test_answer_includes_citations(seeded_pipeline):
    resp = seeded_pipeline.answer("When do field visits start for delinquent accounts?")
    assert resp.citations, "answer must carry citations"
    assert any(c.doc_id == "collections" for c in resp.citations)
    assert "31" in resp.answer or "field visit" in resp.answer.lower()


def test_no_context_fallback(pipeline):
    # empty index -> honest "don't know" instead of hallucination
    resp = pipeline.answer("What is the capital of France?")
    assert resp.answer == NO_CONTEXT_ANSWER
    assert resp.citations == []


def test_irrelevant_question_fallback(seeded_pipeline):
    resp = seeded_pipeline.answer("Explain quantum entanglement in detail with equations")
    assert resp.answer == NO_CONTEXT_ANSWER
