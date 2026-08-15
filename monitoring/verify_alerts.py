#!/usr/bin/env python3
"""
monitoring/verify_alerts.py — Row 24: Anomaly Verification.

Runs BOTH monitoring layers (Evidently statistical drift +
rule-based data quality/schema-contract checks) against the clean
baseline and every corrupted variant produced by
monitoring/drift_simulation.py, then writes one consolidated
pass/fail evidence report.

This is the script to run and screenshot for the presentation's
"Drift Analysis" section -- it directly answers the guideline's
question: "Document and verify how your monitoring dashboard
catches and alerts you to this system anomaly."

Usage:
    python -m monitoring.drift_simulation        # if not already run
    python -m monitoring.verify_alerts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from monitoring.data_quality_checks import run_checks
from monitoring.evidently_report import run_drift_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_alerts")

DRIFT_DATA_DIR = Path("monitoring/drift_data")
REPORTS_DIR = Path("monitoring/reports")

# name -> (file, should this variant actually be flagged as anomalous?)
SCENARIOS = {
    "clean_baseline": ("clean_candidates.csv", False),
    "missing_values": ("drift_missing_values.csv", True),
    "out_of_range": ("drift_out_of_range.csv", True),
    "schema_swap": ("drift_schema_swap.csv", True),
    "schema_change": ("drift_schema_change.csv", True),
}


def main() -> None:
    reference_csv = DRIFT_DATA_DIR / "clean_candidates.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(
            f"{reference_csv} not found -- run `python -m monitoring.drift_simulation` first."
        )

    results = []
    for name, (filename, expected_anomaly) in SCENARIOS.items():
        current_csv = DRIFT_DATA_DIR / filename
        if not current_csv.exists():
            logger.warning("Skipping %s -- %s not found.", name, current_csv)
            continue

        logger.info("Checking scenario: %s", name)

        quality_result = run_checks(current_csv, reference_csv)

        drift_summary = None
        try:
            drift_summary = run_drift_report(reference_csv, current_csv, REPORTS_DIR, name)
        except Exception as exc:  # noqa: BLE001
            # schema_change intentionally breaks the expected columns -- the
            # quality-contract check is what's supposed to catch that case,
            # so a drift-report failure here is itself informative, not fatal.
            logger.info("Evidently drift report skipped for %s (%s) -- schema too different to compare.", name, exc)

        detected = quality_result["anomaly_detected"] or bool(
            drift_summary and drift_summary["dataset_drift_detected"]
        )

        results.append(
            {
                "scenario": name,
                "expected_anomaly": expected_anomaly,
                "detected_anomaly": detected,
                "correctly_classified": detected == expected_anomaly,
                "quality_check_issues": quality_result["issue_count"],
                "evidently_dataset_drift_detected": bool(
                    drift_summary and drift_summary["dataset_drift_detected"]
                ),
            }
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "verify_alerts_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 70)
    logger.info("%-18s %-10s %-10s %-10s", "Scenario", "Expected", "Detected", "Correct")
    for r in results:
        logger.info(
            "%-18s %-10s %-10s %-10s",
            r["scenario"],
            r["expected_anomaly"],
            r["detected_anomaly"],
            r["correctly_classified"],
        )
    logger.info("=" * 70)
    n_correct = sum(r["correctly_classified"] for r in results)
    logger.info("%d / %d scenarios correctly classified.", n_correct, len(results))
    logger.info("Full evidence written to %s", out_path)


if __name__ == "__main__":
    main()
