#!/usr/bin/env python3
"""
monitoring/data_quality_checks.py — Row 21/24 support: rule-based data
quality + schema-contract checks.

Evidently's DataDriftPreset (monitoring/evidently_report.py) is a
*statistical* check: it flags when a column's overall distribution has
shifted enough to be significant. That's the right tool for detecting
gradual drift, but it is intentionally insensitive to a *minority* of
rows having implausible or missing values, or to a renamed/dropped
column -- those are data-quality and schema-contract problems, not
distribution problems, and need a different kind of check.

This module is that second, complementary layer:
  - REQUIRED_COLUMNS:  the API's input contract. A missing/renamed
    column is flagged immediately, independent of any statistics.
  - RANGE_CHECKS:      physically-plausible bounds per column (e.g. a
    recipe can't have -20 minutes of cook time or 100,000 calories).
  - NULL_RATE_CHECKS:  flags when a column's null rate is well above
    what's normal in the reference data, even if not every null is
    itself out of range.

Usage:
    python -m monitoring.data_quality_checks \
        --current monitoring/drift_data/drift_out_of_range.csv \
        --name out_of_range
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_quality_checks")

# The API's actual input contract (see api/schemas.py RecipeCandidate).
REQUIRED_COLUMNS = ["recipe_id", "ingredients_parsed", "cook_time_minutes", "calories"]

# Physically-plausible bounds, set generously around what's observed in the
# real clean test split (cook_time max ~660 min, calories max ~3700) so
# legitimate recipes never trip these, but corrupted values reliably do.
RANGE_CHECKS = {
    "cook_time_minutes": (0, 1440),  # 0 to 24 hours
    "calories": (0, 5000),  # per serving
}

# Null-rate spike threshold: if a column's null rate in the current batch
# exceeds the reference rate by more than this many percentage points, flag it.
NULL_RATE_SPIKE_THRESHOLD = 0.15


def check_schema_contract(df: pd.DataFrame) -> list[dict]:
    """Flag any required column that is missing or renamed."""
    issues = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(
                {
                    "check": "schema_contract",
                    "column": col,
                    "issue": "required column is missing from the current batch",
                }
            )
    return issues


def check_ranges(df: pd.DataFrame) -> list[dict]:
    """Flag rows with physically-implausible values in range-checked columns."""
    issues = []
    for col, (low, high) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out_of_bounds = df[(df[col].notna()) & ((df[col] < low) | (df[col] > high))]
        if len(out_of_bounds) > 0:
            issues.append(
                {
                    "check": "range",
                    "column": col,
                    "issue": f"{len(out_of_bounds)} row(s) outside plausible bounds [{low}, {high}]",
                    "affected_row_count": int(len(out_of_bounds)),
                    "example_values": out_of_bounds[col].head(5).tolist(),
                }
            )
    return issues


def check_null_rate_spike(current_df: pd.DataFrame, reference_df: pd.DataFrame | None) -> list[dict]:
    """Flag columns whose null rate has spiked well above the reference rate."""
    issues = []
    if reference_df is None:
        return issues
    for col in RANGE_CHECKS:
        if col not in current_df.columns or col not in reference_df.columns:
            continue
        ref_rate = reference_df[col].isna().mean()
        cur_rate = current_df[col].isna().mean()
        if cur_rate - ref_rate > NULL_RATE_SPIKE_THRESHOLD:
            issues.append(
                {
                    "check": "null_rate_spike",
                    "column": col,
                    "issue": f"null rate rose from {ref_rate:.1%} (reference) to {cur_rate:.1%} (current)",
                }
            )
    return issues


def run_checks(current_csv: Path, reference_csv: Path | None) -> dict:
    current_df = pd.read_csv(current_csv)
    reference_df = pd.read_csv(reference_csv) if reference_csv is not None else None

    issues = []
    issues += check_schema_contract(current_df)
    issues += check_ranges(current_df)
    issues += check_null_rate_spike(current_df, reference_df)

    return {
        "current_file": str(current_csv),
        "reference_file": str(reference_csv) if reference_csv else None,
        "anomaly_detected": len(issues) > 0,
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based data quality + schema-contract checks.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--reference", default="monitoring/drift_data/clean_candidates.csv")
    parser.add_argument("--out-dir", default="monitoring/reports")
    parser.add_argument("--name", default="run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference) if args.reference else None
    result = run_checks(Path(args.current), reference_path)

    out_path = out_dir / f"quality_checks_{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "Wrote quality check result to %s: anomaly_detected=%s (%d issue(s))",
        out_path,
        result["anomaly_detected"],
        result["issue_count"],
    )


if __name__ == "__main__":
    main()
