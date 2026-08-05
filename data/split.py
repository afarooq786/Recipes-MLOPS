#!/usr/bin/env python3
"""
data/split.py — Step 3: Train / validation / test split

Reads the validated recipes.csv, builds the binary classification label
(rating >= 4), deduplicates recipes that appear more than once in the raw
data, and produces a reproducible, seeded, stratified train/val/test split.
The test set is written separately and is meant to stay untouched by
anyone doing model selection or hyperparameter tuning — it only gets used
for production validation later in the pipeline.

Why dedup first: the raw recipes.csv contains recipes that appear as
multiple fully-identical rows (same name, ingredients, rating, etc. —
only the source row index differs; see data/validation_report.json,
"duplicate row(s) found"). Splitting before deduping would let the same
recipe land in both train and test, which is a data leakage bug. We dedupe
on (recipe_name, url) — the natural identity of a recipe — keeping the
first occurrence, before splitting.

Usage:
    python data/split.py
    python data/split.py --seed 42 --train-frac 0.7 --val-frac 0.15
    python data/split.py --skip-validation-check   # not recommended
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("split")

RATING_POSITIVE_THRESHOLD = 4.0


def _check_validation_passed(report_path: Path) -> None:
    if not report_path.exists():
        logger.error(
            "No validation report found at %s. Run data/validate.py first — "
            "the split step refuses to run on unvalidated data.",
            report_path,
        )
        sys.exit(1)

    report = json.loads(report_path.read_text())
    recipes_result = report.get("recipes.csv", {})
    if not recipes_result.get("passed", False):
        logger.error(
            "recipes.csv failed schema validation (%d critical failure(s)). "
            "Fix the data before splitting — see %s.",
            len(recipes_result.get("critical_failures", [])),
            report_path,
        )
        sys.exit(1)

    logger.info("Validation report confirms recipes.csv passed — proceeding with split.")


def dedupe(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    before = len(df)
    deduped = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    removed = before - len(deduped)
    if removed:
        logger.info("Removed %d duplicate recipe row(s) based on %s (kept first occurrence).", removed, key_cols)
    return deduped


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (df["rating"] >= RATING_POSITIVE_THRESHOLD).astype(int)
    return df


def split_data(
    df: pd.DataFrame, seed: int, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(f"train_frac + val_frac must be < 1.0 (got test_frac={test_frac:.3f})")

    # First carve off the isolated test set.
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=seed,
        stratify=df["label"],
    )
    # Then split the remainder into train / validation.
    relative_val_frac = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_frac,
        random_state=seed,
        stratify=train_val_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _class_balance(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "rows": total,
        "positive": int(counts.get(1, 0)),
        "negative": int(counts.get(0, 0)),
        "positive_rate": round(counts.get(1, 0) / total, 4) if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible train/val/test split of recipes.csv.")
    parser.add_argument("--raw-path", default="data/raw/recipes.csv", help="Path to the validated recipes.csv.")
    parser.add_argument("--report-path", default="data/validation_report.json", help="Path to the validation report.")
    parser.add_argument("--out-dir", default="data/processed/splits", help="Directory to write split CSVs into.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Fraction of data for training.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of data for validation.")
    parser.add_argument(
        "--skip-validation-check",
        action="store_true",
        help="Skip requiring a passing validation report (not recommended).",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    out_dir = Path(args.out_dir)

    if not args.skip_validation_check:
        _check_validation_passed(Path(args.report_path))

    if not raw_path.exists():
        logger.error("Raw file not found: %s. Run data/ingest.py first.", raw_path)
        sys.exit(1)

    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows from %s", len(df), raw_path)

    df = dedupe(df, key_cols=["recipe_name", "url"])
    df = build_label(df)

    train_df, val_df, test_df = split_data(df, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        logger.info("Wrote %s: %d rows", path, len(split_df))

    manifest = {
        "seed": args.seed,
        "source_file": str(raw_path),
        "rating_positive_threshold": RATING_POSITIVE_THRESHOLD,
        "dedup_key": ["recipe_name", "url"],
        "rows_after_dedup": len(df),
        "target_fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": round(1.0 - args.train_frac - args.val_frac, 4),
        },
        "splits": {name: _class_balance(split_df) for name, split_df in splits.items()},
        "note": (
            "The test split is isolated for production validation only — it must not be used "
            "for model selection or hyperparameter tuning."
        ),
    }
    manifest_path = out_dir / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", manifest_path)

    logger.info("Split complete: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))


if __name__ == "__main__":
    main()
