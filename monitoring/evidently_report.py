#!/usr/bin/env python3
"""
monitoring/evidently_report.py — Row 21: Model/data monitoring with
Evidently AI.

Builds an Evidently data-drift report comparing a reference batch
(the clean, isolated test-split candidates) against a current batch
(either the same clean data, for a "no drift" baseline check, or one
of the corrupted variants produced by monitoring/drift_simulation.py).

Writes both:
  - an HTML report (visual, for the presentation / live demo)
  - a small JSON summary (machine-readable: which columns drifted,
    whether the overall dataset is flagged as drifted) for automated
    pass/fail checks in monitoring/verify_alerts.py

Usage:
    # Baseline check: clean vs clean (should show no drift)
    python -m monitoring.evidently_report \
        --reference monitoring/drift_data/clean_candidates.csv \
        --current monitoring/drift_data/clean_candidates.csv \
        --name baseline

    # Drift check: clean vs a corrupted variant (should show drift)
    python -m monitoring.evidently_report \
        --reference monitoring/drift_data/clean_candidates.csv \
        --current monitoring/drift_data/drift_out_of_range.csv \
        --name out_of_range
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evidently_report")

# Only compare the numeric/text feature columns the API actually consumes --
# recipe_id/title are identifiers, not features, and shouldn't be scored for drift.
DRIFT_COLUMNS = ["cook_time_minutes", "calories"]


def _load_for_drift(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    present = [c for c in DRIFT_COLUMNS if c in df.columns]
    missing = [c for c in DRIFT_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            "%s is missing expected column(s) %s -- this itself is a schema-drift signal.",
            csv_path,
            missing,
        )
    return df[present] if present else df


def run_drift_report(reference_csv: Path, current_csv: Path, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_df = _load_for_drift(reference_csv)
    current_df = _load_for_drift(current_csv)

    # Declare these as numerical explicitly -- without it, Evidently can infer a
    # sparse/low-cardinality numeric column as categorical and fall back to a
    # chi-square test that is much less sensitive to the kind of drift we're
    # simulating (negative values, extreme outliers) than a numerical test is.
    numerical_present = [c for c in DRIFT_COLUMNS if c in reference_df.columns]
    data_definition = DataDefinition(numerical_columns=numerical_present)

    reference_ds = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_ds = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference_ds, current_data=current_ds)

    html_path = out_dir / f"drift_report_{name}.html"
    result.save_html(str(html_path))
    logger.info("Wrote HTML report to %s", html_path)

    result_dict = result.dict()
    summary = _summarize(result_dict, name)

    json_path = out_dir / f"drift_summary_{name}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote drift summary to %s: drifted=%s", json_path, summary["dataset_drift_detected"])

    return summary


def _summarize(result_dict: dict, name: str) -> dict:
    """Pull the pass/fail drift signal + per-column p-values out of the raw Evidently result."""
    per_column = {}
    dataset_drift_detected = False
    drifted_share = None

    for metric in result_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        if metric_name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            drifted_share = value.get("share")
            # Flag the whole batch as drifted if more than half the checked
            # columns individually show significant drift.
            dataset_drift_detected = bool(drifted_share is not None and drifted_share > 0.5)
        elif metric_name.startswith("ValueDrift(column="):
            col = metric_name.split("column=")[1].split(",")[0]
            p_value = metric.get("value")
            per_column[col] = {
                "p_value": p_value,
                "drifted": bool(p_value is not None and p_value < 0.05),
            }

    return {
        "report_name": name,
        "dataset_drift_detected": dataset_drift_detected,
        "drifted_column_share": drifted_share,
        "per_column": per_column,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Evidently data-drift report.")
    parser.add_argument("--reference", required=True, help="Path to reference (baseline) CSV.")
    parser.add_argument("--current", required=True, help="Path to current (possibly drifted) CSV.")
    parser.add_argument("--out-dir", default="monitoring/reports")
    parser.add_argument("--name", default="run", help="Short label used in output filenames.")
    args = parser.parse_args()

    run_drift_report(Path(args.reference), Path(args.current), Path(args.out_dir), args.name)


if __name__ == "__main__":
    main()
