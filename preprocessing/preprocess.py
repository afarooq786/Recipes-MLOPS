#!/usr/bin/env python3
"""
preprocessing/preprocess.py — Step 5: Preprocessing

Reusable preprocessing module that normalizes recipe and ingredient fields
coming out of data/processed/splits/{train,val,test}.csv. This does NOT
re-split or re-dedupe data (data/split.py already handled that) — it cleans
and standardizes fields so downstream feature engineering (features/build_features.py)
and modeling can rely on consistent, well-typed columns.

What it does:
  - Cleans/normalizes `ingredients` into a parsed list of individual
    ingredient strings (lowercased, punctuation-stripped, whitespace-collapsed).
  - Standardizes free-text fields (recipe_name, directions, cuisine_path)
    via whitespace/casing cleanup.
  - Parses `prep_time` / `cook_time` / `total_time` (strings like "15 mins",
    "1 hr 30 mins") into numeric minutes, nullable.
  - Handles missing/inconsistent values: fills missing numeric time fields
    with NaN (not 0 — 0 would imply "instant"), fills missing `yield`/`img_src`
    with None, and drops rows missing required identity fields
    (recipe_name, ingredients, directions) since those can't be recovered.
  - Removes any residual exact-duplicate rows (belt-and-suspenders after
    split.py's dedupe) on (recipe_name, url).

This module is meant to be imported (`from preprocessing.preprocess import
preprocess_dataframe`) by features/build_features.py and elsewhere, and can
also be run standalone as a pipeline stage.

Usage:
    python preprocessing/preprocess.py
    python preprocessing/preprocess.py --split-dir data/processed/splits --out-dir data/processed/clean
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

REQUIRED_FIELDS = ["recipe_name", "ingredients", "directions"]
TEXT_FIELDS = ["recipe_name", "directions", "cuisine_path"]
TIME_FIELDS = ["prep_time", "cook_time", "total_time"]

_WHITESPACE_RE = re.compile(r"\s+")
_TIME_HR_RE = re.compile(r"(\d+)\s*hr")
_TIME_MIN_RE = re.compile(r"(\d+)\s*min")
# Ingredient lines look like "1 cup all-purpose flour" or "2 tbsp. olive oil, chopped".
# We split on newlines/semicolons (source scraping uses one ingredient per line).
_INGREDIENT_SPLIT_RE = re.compile(r"[\n;]+")
_PUNCT_RE = re.compile(r"[^\w\s./%-]")


def clean_text(value: object) -> str | None:
    """Lowercase-preserving whitespace/punctuation cleanup for free text fields."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text if text else None


def parse_time_to_minutes(value: object) -> float | None:
    """Parse strings like '1 hr 30 mins', '45 mins', '2 hrs' into total minutes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    hours = _TIME_HR_RE.search(text)
    minutes = _TIME_MIN_RE.search(text)

    if not hours and not minutes:
        # Fallback: a bare number means minutes (e.g. "30").
        if text.isdigit():
            return float(text)
        return None

    total = 0.0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return total


def normalize_ingredient(raw: str) -> str:
    """Lowercase an individual ingredient line, strip quantities/punctuation noise."""
    text = raw.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def parse_ingredients(value: object) -> list[str]:
    """Split the raw `ingredients` blob into a cleaned list of ingredient strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    raw_lines = _INGREDIENT_SPLIT_RE.split(str(value))
    cleaned = [normalize_ingredient(line) for line in raw_lines]
    return [c for c in cleaned if c]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning/normalization steps and return a new, cleaned DataFrame."""
    df = df.copy()
    before = len(df)

    # Drop rows missing anything we can't reasonably impute.
    missing_required = df[REQUIRED_FIELDS].isna().any(axis=1)
    if missing_required.any():
        logger.info("Dropping %d row(s) missing required field(s) %s.", missing_required.sum(), REQUIRED_FIELDS)
    df = df[~missing_required].reset_index(drop=True)

    # Standardize free text fields.
    for col in TEXT_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Parse time fields into numeric minutes (nullable — missing stays missing, not 0).
    for col in TIME_FIELDS:
        if col in df.columns:
            df[f"{col}_minutes"] = df[col].apply(parse_time_to_minutes)

    # Parse ingredients into a structured list + a count feature seed.
    if "ingredients" in df.columns:
        df["ingredients_parsed"] = df["ingredients"].apply(parse_ingredients)
        df["ingredient_count"] = df["ingredients_parsed"].apply(len)

    # Fill remaining soft-missing text fields with None rather than NaN float artifacts.
    for col in ["yield", "img_src"]:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)

    # Belt-and-suspenders dedupe (split.py already dedupes, but this module
    # may be called on data from other sources too).
    dedupe_keys = [c for c in ["recipe_name", "url"] if c in df.columns]
    if dedupe_keys:
        before_dedupe = len(df)
        df = df.drop_duplicates(subset=dedupe_keys, keep="first").reset_index(drop=True)
        removed = before_dedupe - len(df)
        if removed:
            logger.info("Removed %d residual duplicate row(s) on %s.", removed, dedupe_keys)

    logger.info("Preprocessed %d -> %d rows.", before, len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and normalize recipe/ingredient fields.")
    parser.add_argument("--split-dir", default="data/processed/splits", help="Directory containing train/val/test.csv.")
    parser.add_argument("--out-dir", default="data/processed/clean", help="Directory to write cleaned CSVs into.")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], help="Which split files to process."
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        in_path = split_dir / f"{split_name}.csv"
        if not in_path.exists():
            logger.warning("Skipping %s — file not found: %s", split_name, in_path)
            continue

        df = pd.read_csv(in_path)
        logger.info("Loaded %s: %d rows from %s", split_name, len(df), in_path)

        cleaned = preprocess_dataframe(df)

        # ingredients_parsed is a Python list — store as a serialized string for CSV round-tripping.
        if "ingredients_parsed" in cleaned.columns:
            cleaned["ingredients_parsed"] = cleaned["ingredients_parsed"].apply(lambda lst: "|".join(lst))

        out_path = out_dir / f"{split_name}.csv"
        cleaned.to_csv(out_path, index=False)
        logger.info("Wrote %s: %d rows", out_path, len(cleaned))


if __name__ == "__main__":
    main()
