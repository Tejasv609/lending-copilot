"""Text chunking with configurable size and overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    title: str
    chunk_index: int
    text: str


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping character chunks, preferring paragraph/sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # try to break on a sentence or word boundary near the end
            window = text[max(start, end - 120): end]
            for sep in (". ", "\n", "; ", " "):
                idx = window.rfind(sep)
                if idx != -1:
                    end = max(start, end - 120) + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # overlap, but snap the next start to a word boundary so chunks
        # never begin mid-word (nicer citations, same retrieval behavior)
        start = max(end - chunk_overlap, start + 1)
        while start < end and not text[start].isspace():
            start += 1
    return chunks


def chunk_document(doc_id: str, title: str, text: str,
                   chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, title=title, chunk_index=i, text=t)
        for i, t in enumerate(split_text(text, chunk_size, chunk_overlap))
    ]
