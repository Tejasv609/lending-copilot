"""FastAPI app: /health, /ingest, /ask + static chat UI."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .pipeline import Document, RAGPipeline
from .schemas import AskRequest, AskResponse, HealthResponse, IngestResponse

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def create_app(config_path: str = "config.yaml") -> FastAPI:
    config = load_config(config_path)
    pipeline = RAGPipeline(config)
    uploads_dir = Path(config.paths.uploads)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Lending Copilot", version="1.0.0",
                  description="RAG question-answering over lending and credit-policy documents.")

    def read_upload(upload: UploadFile) -> Document:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type: {suffix}. Use md, txt or pdf.")
        raw = upload.file.read()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                raise HTTPException(400, "PDF support needs the 'pypdf' package: pip install pypdf")
            try:
                import io
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join((p.extract_text() or "") for p in reader.pages)
            except Exception as exc:
                raise HTTPException(400, f"Could not parse PDF: {exc}")
        else:
            text = raw.decode("utf-8", errors="replace")
        if not text.strip():
            raise HTTPException(400, "Uploaded file is empty.")
        doc_id = Path(upload.filename or "upload").stem
        return Document(doc_id=doc_id, title=doc_id.replace("_", " ").replace("-", " ").title(), text=text)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            docs_indexed=pipeline.doc_count,
            chunks_indexed=len(pipeline.chunks),
            llm_configured=pipeline.llm.configured,
            llm_model=pipeline.llm.model if pipeline.llm.configured else "extractive",
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
        docs = [read_upload(f) for f in files]
        for doc, f in zip(docs, files):
            dest = uploads_dir / f"{doc.doc_id}{Path(f.filename or '').suffix.lower()}"
            f.file.seek(0)
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
        docs_added, chunks_added = pipeline.ingest_documents(docs)
        return IngestResponse(docs_added=docs_added, chunks_added=chunks_added,
                              total_chunks=len(pipeline.chunks))

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        return pipeline.answer(req.question, top_k=req.top_k, rerank=req.rerank)

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(frontend_dir / "index.html"))

    return app


app = create_app()
