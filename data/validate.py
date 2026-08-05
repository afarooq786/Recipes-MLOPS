#!/usr/bin/env python3
"""
data/validate.py — Step 2: Schema validation

Validates the raw recipe data against a defined schema (required columns,
dtypes, allowed ranges, null rules) and writes a validation report.

Two files are checked:
  - recipes.csv       -> the primary training dataset. This GATES the
                          pipeline: any critical failure here causes this
                          script to exit non-zero, which should stop
                          downstream steps (split, DVC, training) from
                          running on bad data.
  - test_recipes.csv   -> the separate production/holdout candidate set
                          used later for API validation and drift
                          simulation. It's checked and reported on, but
                          does not gate this pipeline on its own — it isn't
                          part of the train/val/test split.

Usage:
    python data/validate.py
    python data/validate.py --raw-dir data/raw --report-dir data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

# --------------------------------------------------------------------------
# Schema: recipes.csv (primary training data)
# --------------------------------------------------------------------------
# Null rules below reflect what's normal for this dataset: prep/cook/total
# time and yield are frequently missing in the source data (scraped recipe
# pages don't always list them), so they're nullable. Everything that
# actually feeds the model or the recommendation task must be present.
RECIPES_SCHEMA = DataFrameSchema(
    {
        "recipe_name": Column(str, nullable=False),
        "prep_time": Column(str, nullable=True),
        "cook_time": Column(str, nullable=True),
        "total_time": Column(str, nullable=True),
        "servings": Column(int, Check.gt(0), nullable=False),
        "yield": Column(str, nullable=True),
        "ingredients": Column(str, Check.str_length(min_value=1), nullable=False),
        "directions": Column(str, Check.str_length(min_value=1), nullable=False),
        "rating": Column(float, Check.in_range(0, 5), nullable=False),
        "url": Column(str, Check.str_startswith("http"), nullable=False),
        "cuisine_path": Column(str, nullable=False),
        "nutrition": Column(str, nullable=False),
        "timing": Column(str, nullable=False),
        "img_src": Column(str, nullable=True),
    },
    strict=False,  # allow extra columns (e.g. the leading unnamed index col) without failing
    coerce=False,
)

# --------------------------------------------------------------------------
# Schema: test_recipes.csv (separate production/holdout candidate set —
# different shape: capitalized headers, no rating, ingredients as
# structured records rather than a flat string).
# --------------------------------------------------------------------------
TEST_RECIPES_SCHEMA = DataFrameSchema(
    {
        "Name": Column(str, nullable=False),
        "Prep Time": Column(str, nullable=True),
        "Cook Time": Column(str, nullable=True),
        "Total Time": Column(str, nullable=True),
        "Servings": Column(nullable=True),  # observed mixed/blank values in source data
        "Ingredients": Column(str, Check.str_length(min_value=1), nullable=False),
        "Directions": Column(str, Check.str_length(min_value=1), nullable=False),
        "url": Column(str, Check.str_startswith("http"), nullable=False),
    },
    strict=False,
    coerce=False,
)

# Null-rate columns we expect to be sparse; a warning is logged if the rate
# jumps well past what's been historically normal, but it does not fail
# the run.
EXPECTED_NULLABLE_WARN_THRESHOLD = 0.60  # warn if > 60% of a nullable column is missing


def _lazy_validate(df: pd.DataFrame, schema: DataFrameSchema) -> tuple[bool, list[dict]]:
    """Run schema validation collecting ALL failures (not just the first)."""
    try:
        schema.validate(df, lazy=True)
        return True, []
    except SchemaErrors as err:
        failure_cases = err.failure_cases
        failures = failure_cases.to_dict(orient="records")
        return False, failures


def _null_rate_warnings(df: pd.DataFrame, nullable_cols: list[str]) -> list[str]:
    warnings = []
    for col in nullable_cols:
        if col not in df.columns:
            continue
        rate = df[col].isna().mean()
        if rate > EXPECTED_NULLABLE_WARN_THRESHOLD:
            warnings.append(f"Column '{col}' is {rate:.0%} null — well above the expected threshold.")
    return warnings


def _duplicate_warning(df: pd.DataFrame, key_cols: list[str]) -> list[str]:
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return []
    dupes = df.duplicated(subset=key_cols).sum()
    if dupes:
        return [f"{dupes} duplicate row(s) found based on {key_cols}."]
    return []


def validate_recipes(path: Path) -> dict:
    result = {"file": str(path), "exists": path.exists()}
    if not path.exists():
        result["critical_failures"] = [f"File not found: {path}"]
        result["warnings"] = []
        result["passed"] = False
        return result

    df = pd.read_csv(path)
    ok, failures = _lazy_validate(df, RECIPES_SCHEMA)

    warnings = _null_rate_warnings(df, ["prep_time", "cook_time", "total_time", "yield", "img_src"])
    warnings += _duplicate_warning(df, ["recipe_name", "url"])

    result.update(
        {
            "row_count": len(df),
            "column_count": len(df.columns),
            "passed": ok,
            "critical_failures": failures,
            "warnings": warnings,
        }
    )
    return result


def validate_test_recipes(path: Path) -> dict:
    result = {"file": str(path), "exists": path.exists()}
    if not path.exists():
        result["critical_failures"] = []
        result["warnings"] = [f"File not found: {path} (non-gating)"]
        result["passed"] = True  # not gating — absence is a warning only
        return result

    df = pd.read_csv(path)
    ok, failures = _lazy_validate(df, TEST_RECIPES_SCHEMA)
    warnings = _null_rate_warnings(df, ["Prep Time", "Cook Time", "Total Time"])
    warnings += _duplicate_warning(df, ["Name", "url"])

    result.update(
        {
            "row_count": len(df),
            "column_count": len(df.columns),
            "passed": ok,
            # Non-gating file: report failures as warnings, don't fail the run on them.
            "critical_failures": [],
            "warnings": warnings + ([f"{len(failures)} schema failure(s) — see failures_detail"] if failures else []),
            "failures_detail": failures,
        }
    )
    return result


def write_reports(report: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "validation_report.json"
    md_path = report_dir / "validation_report.md"

    json_path.write_text(json.dumps(report, indent=2, default=str))

    lines = ["# Data Validation Report", ""]
    for section_name, section in report.items():
        lines.append(f"## {section_name}")
        lines.append(f"- File: `{section.get('file')}`")
        lines.append(f"- Rows: {section.get('row_count', 'n/a')}, Columns: {section.get('column_count', 'n/a')}")
        lines.append(f"- Passed: **{section.get('passed')}**")
        crit = section.get("critical_failures") or []
        lines.append(f"- Critical failures: {len(crit)}")
        for f in crit[:20]:
            lines.append(f"  - `{f}`")
        warn = section.get("warnings") or []
        lines.append(f"- Warnings: {len(warn)}")
        for w in warn:
            lines.append(f"  - {w}")
        lines.append("")
    md_path.write_text("\n".join(lines))

    logger.info("Wrote %s and %s", json_path, md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw recipe data against the defined schema.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing the raw CSV files.")
    parser.add_argument("--report-dir", default="data", help="Directory to write the validation report into.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)

    recipes_result = validate_recipes(raw_dir / "recipes.csv")
    test_result = validate_test_recipes(raw_dir / "test_recipes.csv")

    report = {"recipes.csv": recipes_result, "test_recipes.csv": test_result}
    write_reports(report, report_dir)

    if recipes_result["passed"]:
        logger.info(
            "recipes.csv: PASSED (%d rows, %d warning(s))",
            recipes_result["row_count"],
            len(recipes_result["warnings"]),
        )
    else:
        logger.error(
            "recipes.csv: FAILED — %d critical failure(s). See %s",
            len(recipes_result["critical_failures"]),
            report_dir / "validation_report.json",
        )

    if not test_result["passed"]:
        logger.warning("test_recipes.csv: schema issues found (non-gating) — see report for details.")
    elif test_result.get("warnings"):
        logger.info("test_recipes.csv: passed with %d warning(s).", len(test_result["warnings"]))
    else:
        logger.info("test_recipes.csv: PASSED (%d rows)", test_result["row_count"])

    if not recipes_result["passed"]:
        logger.error("Validation failed on recipes.csv. Stopping pipeline.")
        sys.exit(1)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
