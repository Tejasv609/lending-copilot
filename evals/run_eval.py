#!/usr/bin/env python
"""Run retrieval + answer evals for Lending Copilot.

Metrics:
  - hit_rate@k: fraction of questions where expected_doc is in top-k retrieved chunks
  - MRR: mean reciprocal rank of the first chunk from expected_doc
  - keyword_coverage: fraction of expected keywords present in the answer text
  - citation_rate: fraction of answers that include >=1 citation

Writes evals/results.json and evals/results.md (committed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(config_path: str, k: int) -> dict:
    config = load_config(config_path)
    pipeline = RAGPipeline(config)
    if pipeline.doc_count == 0:
        raise SystemExit("Index is empty. Run `make ingest` first.")

    items = [json.loads(l) for l in (HERE / "eval_set.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for item in items:
        result = pipeline.retrieve(item["question"], top_k=k, rerank=False)
        hits = result.hits
        ranks = [i for i, h in enumerate(hits, start=1)
                 if pipeline.chunks[h.position].doc_id == item["expected_doc"]]
        hit = bool(ranks)
        rr = 1.0 / ranks[0] if ranks else 0.0

        ans = pipeline.answer(item["question"], top_k=k, rerank=False)
        ans_lower = ans.answer.lower()
        kw_hits = [kw for kw in item["keywords"] if kw.lower() in ans_lower]
        rows.append({
            "question": item["question"],
            "expected_doc": item["expected_doc"],
            "hit@k": hit,
            "rr": round(rr, 3),
            "keyword_coverage": round(len(kw_hits) / len(item["keywords"]), 3),
            "citations": len(ans.citations),
            "used_llm": ans.used_llm,
        })

    n = len(rows)
    summary = {
        "n": n,
        "k": k,
        "hit_rate@k": round(sum(r["hit@k"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
        "avg_keyword_coverage": round(sum(r["keyword_coverage"] for r in rows) / n, 3),
        "citation_rate": round(sum(1 for r in rows if r["citations"] > 0) / n, 3),
        "llm_mode": "llm" if any(r["used_llm"] for r in rows) else "extractive",
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    results = run(args.config, args.k)
    s = results["summary"]
    (HERE / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Lending Copilot — Eval Results",
        "",
        f"Evaluated {s['n']} questions against the sample corpus (top-k={s['k']}, answer mode: {s['llm_mode']}).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit rate@{s['k']} | {s['hit_rate@k']} |",
        f"| MRR | {s['mrr']} |",
        f"| Avg keyword coverage | {s['avg_keyword_coverage']} |",
        f"| Citation rate | {s['citation_rate']} |",
        "",
        "## Per-question breakdown",
        "",
        "| # | Question | Expected doc | Hit | RR | Keyword cov. | Citations |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results["rows"], start=1):
        lines.append(f"| {i} | {r['question'][:60]} | {r['expected_doc']} | "
                     f"{'yes' if r['hit@k'] else 'no'} | {r['rr']} | "
                     f"{r['keyword_coverage']} | {r['citations']} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
