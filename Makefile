.PHONY: setup ingest serve eval test docker clean

PY ?= python
VENV ?= .venv

setup: ## Create venv and install dependencies
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

ingest: ## Ingest sample docs (+ anything in data/uploads/) into the index
	$(VENV)/bin/python scripts/ingest.py

serve: ## Run the API + chat UI
	$(VENV)/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

eval: ## Run retrieval + answer evals
	$(VENV)/bin/python evals/run_eval.py

test: ## Run the pytest suite
	$(VENV)/bin/python -m pytest -q

docker: ## Build and run the container
	docker build -t lending-copilot .
	docker run --rm -p 8000:8000 --env-file .env lending-copilot

clean: ## Remove caches and built indexes
	rm -rf data/index __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
