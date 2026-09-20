from src.chunking import chunk_document, split_text


def test_short_text_is_single_chunk():
    assert split_text("hello world", chunk_size=500, chunk_overlap=100) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   ", chunk_size=100, chunk_overlap=20) == []


def test_long_text_chunks_with_overlap():
    text = " ".join(f"sentence number {i} about lending policy." for i in range(60))
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # boundary snapping may exceed slightly
    # overlap: consecutive chunks should share some content
    assert any(word in chunks[1] for word in chunks[0].split()[-8:])


def test_chunk_document_carries_metadata():
    chunks = chunk_document("doc1", "Policy", "alpha beta gamma " * 100,
                            chunk_size=120, chunk_overlap=30)
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].title == "Policy"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.text for c in chunks)
