#!/usr/bin/env python3
"""
features/build_features.py — Step 6: Feature Engineering

Builds model-ready features from the preprocessed splits written by
preprocessing/preprocess.py. Reads data/processed/clean/{train,val,test}.csv
and writes data/processed/features/{train,val,test}.csv plus a
feature_manifest.json documenting what was built and any train-fit
statistics used (e.g. popularity percentiles), so val/test are transformed
consistently with train and don't leak information.

Feature groups:
  1. Nutritional      — parses the `nutrition` JSON blob into numeric columns.
  2. Ingredient        — counts, diversity, and keyword-flag features
                          (protein type, common allergens) from ingredients_parsed.
  3. Cooking-time       — buckets + ratios from the *_minutes columns preprocessing
                          already produced.
  4. Dietary-preference — heuristic vegetarian/vegan/gluten-free flags based on
                          ingredient keyword absence. These are heuristics, not
                          verified dietary claims — see NOTE in the docstring below.
  5. Recipe-popularity  — this dataset has one aggregate rating per recipe and no
                          view/save/click counts (see README "Known Data Notes"),
                          so popularity here means rating-derived signal: a
                          percentile rank of `rating` within the recipe's cuisine,
                          fit on train and applied to val/test.
  6. User-interaction    — NOTE: there is no real per-user interaction data in this
                          dataset (single aggregate rating per recipe). Real
                          user-interaction features (e.g. per-user click/save
                          history) cannot be built until a per-user interaction
                          log exists upstream. This module can optionally attach
                          SYNTHETIC placeholder user-interaction features
                          (--synthetic-user-features) purely so the downstream
                          pipeline (training, serving schema) can be exercised
                          end-to-end — every such column is prefixed `synth_` and
                          documented in the manifest so it's never mistaken for
                          real user behavior.

Usage:
    python features/build_features.py
    python features/build_features.py --synthetic-user-features
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_features")

TIME_BUCKET_EDGES = [0, 15, 30, 60, 120, float("inf")]
TIME_BUCKET_LABELS = ["under_15m", "15_30m", "30_60m", "1_2h", "over_2h"]

PROTEIN_KEYWORDS = {
    "chicken": ["chicken"],
    "beef": ["beef", "steak", "ground beef"],
    "pork": ["pork", "bacon", "ham", "sausage"],
    "fish_seafood": ["salmon", "shrimp", "tuna", "fish", "crab", "cod"],
    "tofu_plant": ["tofu", "tempeh", "seitan", "lentil", "chickpea", "bean"],
    "egg": ["egg"],
    "dairy_cheese": ["cheese", "cream", "milk", "butter", "yogurt"],
}

# Keyword lists used for the heuristic dietary flags. Presence of ANY keyword
# from the relevant list means the recipe is assumed NOT to satisfy that
# dietary preference. This is a coarse heuristic based on ingredient text —
# it is not a verified/authoritative dietary classification (e.g. it can't
# detect hidden animal-derived additives), and should be labeled as such
# anywhere it's surfaced to end users.
NON_VEGETARIAN_KEYWORDS = ["chicken", "beef", "pork", "bacon", "ham", "sausage", "salmon", "shrimp", "tuna", "fish", "crab", "cod", "gelatin"]
NON_VEGAN_EXTRA_KEYWORDS = ["cheese", "cream", "milk", "butter", "yogurt", "egg", "honey"]
GLUTEN_KEYWORDS = ["flour", "wheat", "pasta", "bread", "barley", "rye", "soy sauce"]


def _parse_ingredients_list(value: object) -> list[str]:
    """ingredients_parsed comes from preprocessing as a '|'-joined string on disk."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [v for v in str(value).split("|") if v]


def _safe_json_loads(value: object) -> dict:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}


def _coerce_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def build_nutritional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the `nutrition` JSON blob into numeric nutrition_<key> columns."""
    if "nutrition" not in df.columns:
        return df
    parsed = df["nutrition"].apply(_safe_json_loads)
    all_keys = sorted({k for d in parsed for k in d.keys()})
    for key in all_keys:
        col = f"nutrition_{key.strip().lower().replace(' ', '_')}"
        df[col] = parsed.apply(lambda d, k=key: _coerce_numeric(d.get(k)))
    logger.info("Built %d nutritional feature column(s) from keys: %s", len(all_keys), all_keys)
    return df


def build_ingredient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingredient count/diversity and protein-type keyword flags."""
    ingredients_lists = df["ingredients_parsed"].apply(_parse_ingredients_list) if "ingredients_parsed" in df.columns else None
    if ingredients_lists is None:
        return df

    if "ingredient_count" not in df.columns:
        df["ingredient_count"] = ingredients_lists.apply(len)

    df["ingredient_unique_word_count"] = ingredients_lists.apply(
        lambda lst: len({w for line in lst for w in line.split()})
    )
    # Diversity: unique words per ingredient line — higher means more varied phrasing/detail.
    df["ingredient_diversity_ratio"] = (
        df["ingredient_unique_word_count"] / df["ingredient_count"].replace(0, np.nan)
    ).fillna(0.0)

    for feature_name, keywords in PROTEIN_KEYWORDS.items():
        df[f"has_{feature_name}"] = ingredients_lists.apply(
            lambda lst, kws=keywords: int(any(any(kw in line for kw in kws) for line in lst))
        )

    logger.info("Built ingredient count/diversity + %d protein-type flag(s).", len(PROTEIN_KEYWORDS))
    return df


def build_cooking_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buckets and ratios from the *_minutes columns produced by preprocessing."""
    minute_cols = [c for c in ["prep_time_minutes", "cook_time_minutes", "total_time_minutes"] if c in df.columns]
    if not minute_cols:
        return df

    if "total_time_minutes" in df.columns:
        df["total_time_bucket"] = pd.cut(
            df["total_time_minutes"], bins=TIME_BUCKET_EDGES, labels=TIME_BUCKET_LABELS, right=False
        )

    if "prep_time_minutes" in df.columns and "cook_time_minutes" in df.columns:
        denom = df["cook_time_minutes"].replace(0, np.nan)
        df["prep_to_cook_ratio"] = (df["prep_time_minutes"] / denom).replace([np.inf, -np.inf], np.nan)

    logger.info("Built cooking-time bucket/ratio feature(s) from %s.", minute_cols)
    return df


def build_dietary_preference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic vegetarian/vegan/gluten-free flags from ingredient keywords.

    NOTE: heuristic only — keyword absence is not proof of a dietary claim.
    """
    if "ingredients_parsed" not in df.columns:
        return df
    ingredients_lists = df["ingredients_parsed"].apply(_parse_ingredients_list)

    def _none_present(lst: list[str], keywords: list[str]) -> int:
        return int(not any(any(kw in line for kw in keywords) for line in lst))

    df["is_likely_vegetarian"] = ingredients_lists.apply(lambda lst: _none_present(lst, NON_VEGETARIAN_KEYWORDS))
    df["is_likely_vegan"] = ingredients_lists.apply(
        lambda lst: _none_present(lst, NON_VEGETARIAN_KEYWORDS + NON_VEGAN_EXTRA_KEYWORDS)
    )
    df["is_likely_gluten_free"] = ingredients_lists.apply(lambda lst: _none_present(lst, GLUTEN_KEYWORDS))

    logger.info("Built heuristic dietary-preference flags (vegetarian/vegan/gluten-free).")
    return df


def fit_popularity_stats(df: pd.DataFrame) -> dict:
    """Fit cuisine-level rating percentile stats on TRAIN only, to avoid leakage."""
    if "rating" not in df.columns or "cuisine_path" not in df.columns:
        return {}
    stats = df.groupby("cuisine_path")["rating"].agg(["mean", "std", "count"]).to_dict(orient="index")
    return stats


def apply_popularity_features(df: pd.DataFrame, popularity_stats: dict) -> pd.DataFrame:
    """Apply train-fit cuisine rating stats to compute a popularity percentile-ish score."""
    if "rating" not in df.columns or "cuisine_path" not in df.columns or not popularity_stats:
        return df

    global_mean = np.mean([s["mean"] for s in popularity_stats.values()]) if popularity_stats else None

    def _score(row):
        stats = popularity_stats.get(row["cuisine_path"])
        if not stats or not stats.get("std") or pd.isna(stats.get("std")):
            baseline = stats["mean"] if stats else global_mean
            return row["rating"] - baseline if baseline is not None else 0.0
        return (row["rating"] - stats["mean"]) / stats["std"]

    df["popularity_zscore_within_cuisine"] = df.apply(_score, axis=1)
    df["popularity_rank_pct_within_cuisine"] = df.groupby("cuisine_path")["rating"].rank(pct=True)
    logger.info("Applied cuisine-relative popularity features.")
    return df


def build_synthetic_user_interaction_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Attach clearly-labeled SYNTHETIC user-interaction features.

    There is no real per-user interaction log in this dataset (see module
    docstring). These synth_* columns exist only so the rest of the pipeline
    (training, serving schema, monitoring) can be built and tested end-to-end
    before real interaction data exists. They must never be presented as, or
    trained into a model shipped as using, real user behavior.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    df["synth_user_view_count"] = rng.poisson(lam=50, size=n)
    df["synth_user_save_rate"] = rng.beta(a=2, b=5, size=n)
    df["synth_user_avg_session_seconds"] = rng.normal(loc=90, scale=30, size=n).clip(min=0)
    logger.warning(
        "Attached %d SYNTHETIC user-interaction column(s) (synth_ prefix). "
        "These are placeholder data, not real user behavior.",
        3,
    )
    return df


def build_features(df: pd.DataFrame, popularity_stats: dict, synthetic_user_features: bool, seed: int) -> pd.DataFrame:
    df = df.copy()
    df = build_nutritional_features(df)
    df = build_ingredient_features(df)
    df = build_cooking_time_features(df)
    df = build_dietary_preference_features(df)
    df = apply_popularity_features(df, popularity_stats)
    if synthetic_user_features:
        df = build_synthetic_user_interaction_features(df, seed)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model features from preprocessed recipe splits.")
    parser.add_argument("--clean-dir", default="data/processed/clean", help="Directory with preprocessed split CSVs.")
    parser.add_argument("--out-dir", default="data/processed/features", help="Directory to write feature CSVs into.")
    parser.add_argument(
        "--synthetic-user-features",
        action="store_true",
        help="Attach clearly-labeled synthetic (synth_*) user-interaction placeholder features.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic feature generation.")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = clean_dir / "train.csv"
    if not train_path.exists():
        logger.error("Train split not found at %s. Run preprocessing/preprocess.py first.", train_path)
        raise SystemExit(1)

    train_df = pd.read_csv(train_path)
    popularity_stats = fit_popularity_stats(train_df)

    manifest = {
        "synthetic_user_features": args.synthetic_user_features,
        "popularity_stats_fit_on": "train",
        "popularity_stats_cuisine_count": len(popularity_stats),
        "dietary_flags_are_heuristic": True,
        "seed": args.seed,
    }

    for split_name in ["train", "val", "test"]:
        in_path = clean_dir / f"{split_name}.csv"
        if not in_path.exists():
            logger.warning("Skipping %s — file not found: %s", split_name, in_path)
            continue
        df = pd.read_csv(in_path)
        logger.info("Loaded %s: %d rows from %s", split_name, len(df), in_path)

        featured = build_features(df, popularity_stats, args.synthetic_user_features, args.seed)

        out_path = out_dir / f"{split_name}.csv"
        featured.to_csv(out_path, index=False)
        logger.info("Wrote %s: %d rows, %d columns", out_path, len(featured), len(featured.columns))

    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
