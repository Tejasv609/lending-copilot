"""API tests via FastAPI TestClient."""
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.main as main_mod


def make_client(test_config):
    with patch.object(main_mod, "load_config", return_value=test_config):
        app = main_mod.create_app()
    return TestClient(app)


def test_health(test_config):
    r = make_client(test_config).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["docs_indexed"] == 0
    assert body["llm_model"] == "extractive"  # no API key in tests


def test_ingest_and_ask_flow(test_config):
    client = make_client(test_config)

    files = [("files", ("policy.md", "# Policy\nPersonal loans need CIBIL 650 minimum.", "text/markdown"))]
    r = client.post("/ingest", files=files)
    assert r.status_code == 200
    assert r.json()["docs_added"] == 1

    r = client.post("/ask", json={"question": "What is the minimum CIBIL for a personal loan?"})
    assert r.status_code == 200
    body = r.json()
    assert "650" in body["answer"]
    assert len(body["citations"]) > 0

    assert client.get("/health").json()["docs_indexed"] == 1


def test_ingest_is_idempotent(test_config):
    client = make_client(test_config)
    files = [("files", ("policy.md", "# Policy\nPersonal loans need CIBIL 650 minimum.", "text/markdown"))]
    client.post("/ingest", files=files)
    r = client.post("/ingest", files=files)
    assert r.json()["docs_added"] == 0  # same doc_id skipped


def test_ingest_rejects_unsupported_type(test_config):
    client = make_client(test_config)
    files = [("files", ("evil.exe", "MZ...", "application/octet-stream"))]
    assert client.post("/ingest", files=files).status_code == 400


def test_ask_validates_question(test_config):
    client = make_client(test_config)
    assert client.post("/ask", json={"question": ""}).status_code == 422
