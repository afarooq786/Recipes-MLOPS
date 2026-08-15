#!/usr/bin/env python3
"""
monitoring/drift_simulation.py — Rows 20 & 23: production baseline data +
drift/corruption simulation.

This script has two jobs:

1. Prepare a "clean" batch of API-request-shaped candidates from the
   isolated test split (data/processed/clean/test.csv). This is the
   dataset that gets sent to the deployed /predict endpoint for baseline
   production validation (row 20) -- it was never used for model
   selection or tuning, per the README's data-role separation.

2. Generate several corrupted/drifted variants of that same batch, each
   simulating a distinct real-world failure mode:
     - missing_values:  required fields randomly nulled out
     - out_of_range:    physically implausible values (negative time,
                         absurd calories)
     - schema_swap:     two columns' values swapped with each other
                         (simulates an upstream column-mapping bug)
     - schema_change:   a column renamed, and another dropped entirely
                         (simulates an upstream schema change)

Every corruption is seeded and reproducible, and a manifest records
exactly which rows/columns were touched by which corruption, so the
"was this anomaly actually caught?" check in row 24 has clean ground
truth to compare monitoring output against.

Usage:
    python -m monitoring.drift_simulation
    python -m monitoring.drift_simulation --n-rows 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drift_simulation")

# Rough macro -> calorie conversion (kcal/g). The source `nutrition` text
# blob does not include a literal Calories field, only macro grams, so this
# is a documented estimate, not a scraped ground-truth calorie count.
_FAT_KCAL_PER_G = 9
_CARB_KCAL_PER_G = 4
_PROTEIN_KCAL_PER_G = 4

_GRAMS_RE = {
    "fat": re.compile(r"Total Fat (\d+(?:\.\d+)?)g"),
    "carb": re.compile(r"Total Carbohydrate (\d+(?:\.\d+)?)g"),
    "protein": re.compile(r"Protein (\d+(?:\.\d+)?)g"),
}


def _estimate_calories(nutrition_text: str) -> float | None:
    """Estimate calories/serving from macro grams in the nutrition text blob."""
    if not isinstance(nutrition_text, str) or not nutrition_text.strip():
        return None
    grams = {}
    for key, pattern in _GRAMS_RE.items():
        match = pattern.search(nutrition_text)
        grams[key] = float(match.group(1)) if match else 0.0
    calories = (
        grams["fat"] * _FAT_KCAL_PER_G
        + grams["carb"] * _CARB_KCAL_PER_G
        + grams["protein"] * _PROTEIN_KCAL_PER_G
    )
    return round(calories, 1) if calories > 0 else None


def build_clean_candidates(test_csv: Path, n_rows: int | None, seed: int) -> pd.DataFrame:
    """Build the clean, API-request-shaped batch from the isolated test split."""
    df = pd.read_csv(test_csv)
    if n_rows is not None and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

    candidates = pd.DataFrame(
        {
            "recipe_id": [f"test_{i}" for i in range(len(df))],
            "title": df.get("recipe_name"),
            "ingredients_parsed": df.get("ingredients_parsed"),
            "cook_time_minutes": df.get("cook_time_minutes"),
            "calories": df["nutrition"].apply(_estimate_calories) if "nutrition" in df.columns else None,
            "label": df.get("label"),  # kept for offline comparison only, not sent to the API
        }
    )
    # Rows missing the one truly required model input aren't valid candidates.
    candidates = candidates.dropna(subset=["ingredients_parsed"]).reset_index(drop=True)
    return candidates


def corrupt_missing_values(df: pd.DataFrame, seed: int, frac: float = 0.25) -> tuple[pd.DataFrame, dict]:
    """Randomly null out cook_time_minutes and calories for a fraction of rows."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n_affected = max(1, int(len(out) * frac))
    affected_idx = rng.choice(out.index, size=n_affected, replace=False)
    out.loc[affected_idx, "cook_time_minutes"] = np.nan
    out.loc[affected_idx, "calories"] = np.nan
    manifest = {
        "corruption_type": "missing_values",
        "description": "cook_time_minutes and calories set to null for a subset of rows.",
        "affected_row_count": int(n_affected),
        "affected_recipe_ids": out.loc[affected_idx, "recipe_id"].tolist(),
    }
    return out, manifest


def corrupt_out_of_range(df: pd.DataFrame, seed: int, frac: float = 0.20) -> tuple[pd.DataFrame, dict]:
    """Inject physically implausible values: negative cook time, extreme calories."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n_affected = max(1, int(len(out) * frac))
    affected_idx = rng.choice(out.index, size=n_affected, replace=False)
    half = len(affected_idx) // 2
    negative_time_idx = affected_idx[:half]
    extreme_cal_idx = affected_idx[half:]
    out.loc[negative_time_idx, "cook_time_minutes"] = -rng.integers(5, 60, size=len(negative_time_idx))
    out.loc[extreme_cal_idx, "calories"] = rng.integers(50_000, 200_000, size=len(extreme_cal_idx))
    manifest = {
        "corruption_type": "out_of_range",
        "description": "cook_time_minutes set negative for half the affected rows; "
        "calories set to an implausible 50k-200k range for the other half.",
        "affected_row_count": int(n_affected),
        "affected_recipe_ids": out.loc[affected_idx, "recipe_id"].tolist(),
    }
    return out, manifest


def corrupt_schema_swap(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Swap the values of two columns (simulates an upstream column-mapping bug)."""
    out = df.copy()
    out["cook_time_minutes"], out["calories"] = df["calories"], df["cook_time_minutes"]
    manifest = {
        "corruption_type": "schema_swap",
        "description": "cook_time_minutes and calories column VALUES swapped for all rows "
        "(column names unchanged) -- simulates an upstream field-mapping bug.",
        "affected_row_count": int(len(out)),
        "affected_recipe_ids": out["recipe_id"].tolist(),
    }
    return out, manifest


def corrupt_schema_change(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename one column and drop another (simulates an upstream schema change)."""
    out = df.copy()
    out = out.rename(columns={"cook_time_minutes": "cook_minutes"})  # renamed, API won't recognize it
    out = out.drop(columns=["calories"])  # dropped entirely
    manifest = {
        "corruption_type": "schema_change",
        "description": "cook_time_minutes renamed to cook_minutes; calories column dropped entirely "
        "-- simulates an upstream schema/contract change.",
        "affected_row_count": int(len(out)),
        "affected_recipe_ids": out["recipe_id"].tolist(),
    }
    return out, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean + corrupted drift-simulation datasets.")
    parser.add_argument("--test-csv", default="data/processed/clean/test.csv")
    parser.add_argument("--out-dir", default="monitoring/drift_data")
    parser.add_argument("--n-rows", type=int, default=None, help="Subsample size (default: use all test rows).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building clean candidate batch from %s", args.test_csv)
    clean = build_clean_candidates(Path(args.test_csv), args.n_rows, args.seed)
    clean.to_csv(out_dir / "clean_candidates.csv", index=False)
    logger.info("Wrote %d clean candidates to %s", len(clean), out_dir / "clean_candidates.csv")

    manifests = []

    missing, m1 = corrupt_missing_values(clean, args.seed)
    missing.to_csv(out_dir / "drift_missing_values.csv", index=False)
    manifests.append(m1)

    oor, m2 = corrupt_out_of_range(clean, args.seed)
    oor.to_csv(out_dir / "drift_out_of_range.csv", index=False)
    manifests.append(m2)

    swapped, m3 = corrupt_schema_swap(clean, args.seed)
    swapped.to_csv(out_dir / "drift_schema_swap.csv", index=False)
    manifests.append(m3)

    changed, m4 = corrupt_schema_change(clean)
    changed.to_csv(out_dir / "drift_schema_change.csv", index=False)
    manifests.append(m4)

    manifest_path = out_dir / "drift_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"seed": args.seed, "clean_row_count": len(clean), "corruptions": manifests}, f, indent=2)
    logger.info("Wrote drift manifest to %s", manifest_path)
    logger.info("Done. 4 corrupted variants + 1 clean batch written to %s", out_dir)


if __name__ == "__main__":
    main()
