#!/usr/bin/env python
"""Ingest documents into the Lending Copilot index.

Ingests every .md/.txt/.pdf in the sample-docs dir and the uploads dir.
Re-running is idempotent: already-indexed doc_ids are skipped (use --clear to rebuild).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline import Document, RAGPipeline  # noqa: E402


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into the Lending Copilot index.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--clear", action="store_true", help="wipe the index before ingesting")
    ap.add_argument("--extra", action="append", default=[], help="extra file or dir to ingest")
    args = ap.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    dirs = [Path(config.paths.sample_docs), Path(config.paths.uploads)]
    for extra in args.extra:
        dirs.append(Path(extra))

    docs: list[Document] = []
    for d in dirs:
        if not d.exists():
            continue
        files = [d] if d.is_file() else sorted(p for p in d.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"})
        for f in files:
            text = read_file(f)
            if text.strip():
                doc_id = f.stem
                docs.append(Document(doc_id=doc_id,
                                     title=doc_id.replace("_", " ").replace("-", " ").title(),
                                     text=text))

    docs_added, chunks_added = pipeline.ingest_documents(docs, clear=args.clear)
    print(f"docs added: {docs_added} | chunks added: {chunks_added} | "
          f"total docs: {pipeline.doc_count} | total chunks: {len(pipeline.chunks)}")


if __name__ == "__main__":
    main()
