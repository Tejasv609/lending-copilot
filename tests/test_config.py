import os

from src.config import load_config


def test_defaults_load_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("nonexistent.yaml")
    assert cfg.retrieval.top_k == 5
    assert cfg.llm.model == "llama3.1"
    assert cfg.paths.index_dir == "data/index"


def test_env_overrides_win(monkeypatch):
    monkeypatch.setenv("LENDING_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LENDING_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LENDING_RETRIEVAL_TOP_K", "8")
    monkeypatch.setenv("LENDING_RETRIEVAL_RERANK", "false")
    cfg = load_config("nonexistent.yaml")
    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.llm.api_key == "sk-test"
    assert cfg.retrieval.top_k == 8
    assert cfg.retrieval.rerank is False


def test_yaml_values_load(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("retrieval:\n  top_k: 3\nllm:\n  model: mistral\n")
    cfg = load_config(p)
    assert cfg.retrieval.top_k == 3
    assert cfg.llm.model == "mistral"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("llm:\n  model: mistral\n")
    monkeypatch.setenv("LENDING_LLM_MODEL", "gpt-4o-mini")
    cfg = load_config(p)
    assert cfg.llm.model == "gpt-4o-mini"
