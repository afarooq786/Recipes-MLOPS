#!/usr/bin/env python3
"""
monitoring/production_validation.py — Row 20: Production Validation.

Sends the isolated, never-used-for-training clean test batch
(monitoring/drift_data/clean_candidates.csv, built by
monitoring/drift_simulation.py) through the deployed /predict
endpoint, and compares the API's live predictions against the
offline validation-set performance already recorded in
models/champion_config.json.

This is intentionally an HTTP client hitting a real running service
-- it validates the full deployed path (FastAPI -> MLflow model
registry -> inference), not just the model object in isolation.

Prerequisites (see README "Quick Start with Docker Compose"):
    docker compose up -d --build
    curl http://localhost:8000/health   # should report model_loaded: true

Usage:
    python -m monitoring.production_validation
    python -m monitoring.production_validation --api-url http://localhost:8000 --batch-size 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("production_validation")


def _check_health(api_url: str) -> dict:
    resp = requests.get(f"{api_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _score_batch(api_url: str, candidates: list[dict], timeout: int = 30) -> list[dict]:
    payload = {
        "candidates": [
            {
                "recipe_id": c["recipe_id"],
                "title": c.get("title"),
                "ingredients_parsed": c["ingredients_parsed"],
            }
            for c in candidates
        ],
        # No constraints here -- production validation scores every candidate
        # so predictions can be compared 1:1 against ground truth labels.
    }
    start = time.time()
    resp = requests.post(f"{api_url}/predict", json=payload, timeout=timeout)
    elapsed_ms = (time.time() - start) * 1000
    resp.raise_for_status()
    body = resp.json()
    for r in body["results"]:
        r["_latency_ms_for_batch"] = elapsed_ms
    return body["results"]


def run_validation(api_url: str, candidates_csv: Path, batch_size: int, champion_config_path: Path) -> dict:
    logger.info("Checking API health at %s ...", api_url)
    health = _check_health(api_url)
    if not health.get("model_loaded"):
        raise RuntimeError(f"API reports model_loaded=False: {health}")
    logger.info("API healthy: %s", health)

    df = pd.read_csv(candidates_csv)
    if "label" not in df.columns:
        raise ValueError(f"{candidates_csv} has no 'label' column -- can't compute production ROC-AUC.")

    records = df.to_dict(orient="records")
    all_scores: dict[str, float] = {}
    latencies = []

    logger.info("Scoring %d candidates in batches of %d ...", len(records), batch_size)
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        results = _score_batch(api_url, batch)
        for r in results:
            all_scores[r["recipe_id"]] = r["score"]
        if results:
            latencies.append(results[0]["_latency_ms_for_batch"])

    df["predicted_score"] = df["recipe_id"].map(all_scores)
    scored = df.dropna(subset=["predicted_score"])
    missing_count = len(df) - len(scored)
    if missing_count:
        logger.warning("%d candidate(s) got no prediction back from the API.", missing_count)

    production_auc = None
    if scored["label"].nunique() > 1:
        production_auc = float(roc_auc_score(scored["label"], scored["predicted_score"]))

    offline_val_auc = None
    if champion_config_path.exists():
        offline_val_auc = json.loads(champion_config_path.read_text()).get("validation_roc_auc")

    report = {
        "api_url": api_url,
        "candidates_sent": len(df),
        "candidates_scored": len(scored),
        "candidates_missing_prediction": missing_count,
        "production_roc_auc": production_auc,
        "offline_validation_roc_auc": offline_val_auc,
        "auc_delta": (production_auc - offline_val_auc)
        if production_auc is not None and offline_val_auc is not None
        else None,
        "mean_batch_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "max_batch_latency_ms": max(latencies) if latencies else None,
        "passed": production_auc is not None and missing_count == 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the deployed API against the isolated test set.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--candidates-csv", default="monitoring/drift_data/clean_candidates.csv")
    parser.add_argument("--champion-config", default="models/champion_config.json")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--out", default="monitoring/reports/production_validation_report.json")
    args = parser.parse_args()

    report = run_validation(
        api_url=args.api_url,
        candidates_csv=Path(args.candidates_csv),
        batch_size=args.batch_size,
        champion_config_path=Path(args.champion_config),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Production validation report:")
    logger.info(json.dumps(report, indent=2))
    logger.info("Wrote report to %s", out_path)


if __name__ == "__main__":
    main()
