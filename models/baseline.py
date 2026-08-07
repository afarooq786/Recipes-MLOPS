#!/usr/bin/env python3
"""
models/baseline.py — Step 7: Baseline Model

The required non-ML baseline: a popularity recommender. It ranks recipes
purely by training-set popularity — no features, no learned parameters —
so every real model in this project has a floor to beat.

"Popularity" here means the recipe's cuisine-relative mean rating on TRAIN
(see features/build_features.py's popularity features for the same idea
applied as a model input). At inference/eval time, every recipe gets scored
by looking up its cuisine's train-fit mean rating — recipes are never scored
using their own held-out rating, which would leak the label.

Cold-start handling: a cuisine_path seen in val/test but never seen in train
falls back to the train-wide global mean rating.

Usage:
    python models/baseline.py
    python models/baseline.py --features-dir data/processed/features --out-dir evaluation/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python models/baseline.py` to import sibling top-level packages
# (evaluation/) regardless of invocation style, since this script's own
# directory (not the repo root) is what Python puts on sys.path by default.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline")


class PopularityBaseline:
    """Non-ML baseline: score = training-set mean rating for the recipe's cuisine."""

    def __init__(self) -> None:
        self.cuisine_mean_rating_: dict[str, float] = {}
        self.global_mean_rating_: float | None = None
        self.is_fit_ = False

    def fit(self, df: pd.DataFrame, cuisine_col: str = "cuisine_path", rating_col: str = "rating") -> "PopularityBaseline":
        if cuisine_col not in df.columns or rating_col not in df.columns:
            raise ValueError(f"fit() requires columns '{cuisine_col}' and '{rating_col}' in the training data.")
        self.cuisine_mean_rating_ = df.groupby(cuisine_col)[rating_col].mean().to_dict()
        self.global_mean_rating_ = float(df[rating_col].mean())
        self.is_fit_ = True
        logger.info(
            "Fit popularity baseline on %d training rows across %d cuisine(s). Global mean rating: %.3f",
            len(df),
            len(self.cuisine_mean_rating_),
            self.global_mean_rating_,
        )
        return self

    def score(self, df: pd.DataFrame, cuisine_col: str = "cuisine_path") -> np.ndarray:
        if not self.is_fit_:
            raise RuntimeError("Call fit() before score().")
        return df[cuisine_col].map(self.cuisine_mean_rating_).fillna(self.global_mean_rating_).to_numpy()

    def to_dict(self) -> dict:
        return {
            "cuisine_mean_rating": self.cuisine_mean_rating_,
            "global_mean_rating": self.global_mean_rating_,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the popularity baseline recommender.")
    parser.add_argument("--features-dir", default="data/processed/features", help="Directory with train/val/test feature CSVs.")
    parser.add_argument("--out-dir", default="evaluation/results", help="Directory to write baseline artifacts + metrics.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--cuisine-col", default="cuisine_path")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"], help="Which split to evaluate on.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = features_dir / "train.csv"
    eval_path = features_dir / f"{args.eval_split}.csv"
    if not train_path.exists() or not eval_path.exists():
        logger.error("Missing feature file(s). Run features/build_features.py first. Looked for %s and %s.", train_path, eval_path)
        raise SystemExit(1)

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    model = PopularityBaseline().fit(train_df, cuisine_col=args.cuisine_col, rating_col=args.rating_col)
    scores = model.score(eval_df, cuisine_col=args.cuisine_col)

    # Persist the fitted lookup table so evaluation/inference can be re-run without refitting.
    model_path = out_dir / "baseline_popularity_model.json"
    model_path.write_text(json.dumps(model.to_dict(), indent=2))
    logger.info("Wrote fitted baseline model to %s", model_path)

    # Write predictions in the shape evaluation/metrics.py expects (label, score, group col).
    predictions = pd.DataFrame(
        {
            args.label_col: eval_df[args.label_col],
            "score": scores,
            args.cuisine_col: eval_df[args.cuisine_col],
        }
    )
    predictions_path = out_dir / f"baseline_{args.eval_split}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    logger.info("Wrote %s: %d rows", predictions_path, len(predictions))

    # Evaluate using the shared evaluation module.
    from evaluation.metrics import evaluate, plot_precision_at_k_bar, plot_roc_curve, save_results

    results = evaluate(
        predictions[args.label_col].to_numpy(),
        predictions["score"].to_numpy(),
        groups=predictions[args.cuisine_col].to_numpy(),
        k=args.k,
    )
    save_results(results, out_dir, model_name=f"baseline_{args.eval_split}")
    plot_roc_curve(
        predictions[args.label_col].to_numpy(),
        predictions["score"].to_numpy(),
        out_dir / f"baseline_{args.eval_split}_roc.png",
        model_name="baseline",
    )
    plot_precision_at_k_bar(
        results["grouped_precision_at_k"],
        out_dir / f"baseline_{args.eval_split}_precision_at_k.png",
        model_name="baseline",
    )

    logger.info(
        "Baseline on %s split — ROC-AUC: %s | Precision@%d (overall): %s | Precision@%d (macro-avg by %s): %s",
        args.eval_split,
        results["roc_auc"],
        args.k,
        results["precision_at_k_overall"],
        args.k,
        args.cuisine_col,
        results["grouped_precision_at_k"]["macro_avg_precision_at_k"],
    )


if __name__ == "__main__":
    main()
