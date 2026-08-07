#!/usr/bin/env python3
"""
evaluation/metrics.py — Step 8: Evaluation

Reusable evaluation functions for classification/ranking models in this
pipeline. Computes ROC-AUC and Precision@5 (plus a few supporting metrics),
and saves results in both machine-readable (JSON) and visual (PNG plot)
formats. Designed to be imported by any model's evaluation step (baseline,
Logistic Regression, XGBoost, etc.) so every model reports metrics the same
way.

Precision@5 here is computed per-group: within each group (default:
`cuisine_path`, since there's no per-user data — see README "Known Data
Notes"), rank recipes by predicted score and check what fraction of the
top 5 are truly positive (rating >= 4 / label == 1). This mirrors how the
metric will behave later if grouped by user once real interaction data
exists — the grouping key is swappable via --group-col.

Usage (as a library):
    from evaluation.metrics import evaluate, save_results

    results = evaluate(y_true, y_score, groups=df["cuisine_path"])
    save_results(results, out_dir="evaluation/results", model_name="baseline")

Usage (standalone, from a predictions CSV with columns: label, score, and
a group column):
    python evaluation/metrics.py --predictions preds.csv --group-col cuisine_path
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("metrics")

DEFAULT_K = 5


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = DEFAULT_K) -> float | None:
    """Precision@k over a single ranked list: of the top-k by score, what fraction are positive."""
    if len(y_true) == 0:
        return None
    k_eff = min(k, len(y_true))
    order = np.argsort(-y_score)[:k_eff]
    top_k_labels = np.asarray(y_true)[order]
    return float(top_k_labels.sum() / k_eff)


def grouped_precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = DEFAULT_K
) -> dict:
    """Precision@k computed within each group, plus the macro-average across groups.

    Groups with fewer than k items still contribute (k is clipped to the
    group size), so every group is represented rather than silently dropped.
    """
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score, "group": groups})
    per_group = {}
    for group_value, group_df in df.groupby("group"):
        score = precision_at_k(group_df["y_true"].to_numpy(), group_df["y_score"].to_numpy(), k=k)
        per_group[str(group_value)] = {"precision_at_k": score, "group_size": len(group_df)}

    valid_scores = [v["precision_at_k"] for v in per_group.values() if v["precision_at_k"] is not None]
    macro_avg = float(np.mean(valid_scores)) if valid_scores else None

    return {
        "k": k,
        "macro_avg_precision_at_k": macro_avg,
        "num_groups": len(per_group),
        "per_group": per_group,
    }


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """ROC-AUC is undefined with only one class present — return None instead of raising."""
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true — ROC-AUC is undefined, returning None.")
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray | None = None,
    k: int = DEFAULT_K,
) -> dict:
    """Compute the full metrics bundle: overall ROC-AUC, overall Precision@k,
    and (if groups given) grouped Precision@k.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    results = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "roc_auc": safe_roc_auc(y_true, y_score),
        "precision_at_k_overall": precision_at_k(y_true, y_score, k=k),
        "k": k,
    }

    if groups is not None:
        results["grouped_precision_at_k"] = grouped_precision_at_k(y_true, y_score, np.asarray(groups), k=k)

    return results


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, model_name: str) -> bool:
    """Save an ROC curve plot. Returns False (and logs, doesn't raise) if plotting is unavailable
    or ROC-AUC is undefined for this data.
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Skipping ROC plot — only one class present in y_true.")
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation. `pip install matplotlib` to enable.")
        return False

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote ROC curve plot to %s", out_path)
    return True


def plot_precision_at_k_bar(grouped_results: dict, out_path: Path, model_name: str, top_n: int = 20) -> bool:
    """Bar chart of per-group Precision@k for the largest groups (capped at top_n for readability)."""
    per_group = grouped_results.get("per_group", {})
    if not per_group:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation.")
        return False

    items = sorted(per_group.items(), key=lambda kv: kv[1]["group_size"], reverse=True)[:top_n]
    labels = [k for k, _ in items]
    values = [v["precision_at_k"] if v["precision_at_k"] is not None else 0 for _, v in items]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 5))
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel(f"Precision@{grouped_results.get('k', DEFAULT_K)}")
    ax.set_title(f"Precision@{grouped_results.get('k', DEFAULT_K)} by group — {model_name}")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=60, ha="right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote Precision@k bar plot to %s", out_path)
    return True


def save_results(results: dict, out_dir: str | Path, model_name: str) -> Path:
    """Write results to <out_dir>/<model_name>_metrics.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote metrics to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions with ROC-AUC and Precision@k.")
    parser.add_argument("--predictions", required=True, help="CSV with columns: label, score[, group-col].")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--group-col", default=None, help="Optional column to group Precision@k by.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    groups = df[args.group_col] if args.group_col and args.group_col in df.columns else None

    results = evaluate(df[args.label_col], df[args.score_col], groups=groups, k=args.k)
    save_results(results, args.out_dir, args.model_name)

    plot_roc_curve(df[args.label_col].to_numpy(), df[args.score_col].to_numpy(), Path(args.out_dir) / f"{args.model_name}_roc.png", args.model_name)
    if groups is not None:
        plot_precision_at_k_bar(results["grouped_precision_at_k"], Path(args.out_dir) / f"{args.model_name}_precision_at_k.png", args.model_name)

    logger.info("ROC-AUC: %s | Precision@%d (overall): %s", results["roc_auc"], args.k, results["precision_at_k_overall"])


if __name__ == "__main__":
    main()
