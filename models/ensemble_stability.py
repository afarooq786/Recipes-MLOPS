#!/usr/bin/env python3
"""
Repeated fold-by-fold stability comparison of individual text models
and ensembles for the original rating >= 4 target.

Every candidate is evaluated on the exact same 10 x 5 repeated
stratified CV splits. Test and validation sets are never read.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("ensemble_stability")

TRAIN_PATH = Path("data/processed/features/train.csv")
OUTPUT_DIR = Path("evaluation/results/ensemble_stability")

RANDOM_STATE = 42


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)

    units = [
        "cup", "cups",
        "tablespoon", "tablespoons",
        "teaspoon", "teaspoons",
        "ounce", "ounces", "oz",
        "pound", "pounds", "lb", "lbs",
        "gram", "grams", "kg",
        "ml", "liter", "liters",
        "package", "packages",
        "can", "cans",
        "slice", "slices",
    ]

    pattern = r"\b(?:" + "|".join(units) + r")\b"
    text = re.sub(pattern, " ", text)

    return re.sub(r"\s+", " ", text).strip()


def char_logistic():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=3.0,
                    penalty="l1",
                    class_weight=None,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def word_logistic():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=3,
                    sublinear_tf=False,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=10.0,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def char_svm():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=2,
                    sublinear_tf=False,
                ),
            ),
            (
                "model",
                LinearSVC(
                    C=0.03,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


def word_svm():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LinearSVC(
                    C=10.0,
                    class_weight=None,
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def rank_normalize(scores):
    return (
        pd.Series(scores)
        .rank(method="average", pct=True)
        .to_numpy()
    )


def summarize(values):
    values = np.asarray(values)

    return {
        "mean_auc": float(np.mean(values)),
        "std_auc": float(np.std(values)),
        "median_auc": float(np.median(values)),
        "min_auc": float(np.min(values)),
        "max_auc": float(np.max(values)),
        "n_folds": int(len(values)),
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(TRAIN_PATH)

    X = (
        df["ingredients_parsed"]
        .fillna("")
        .apply(clean_text)
        .to_numpy()
    )

    y = df["label"].to_numpy()

    print("Training rows:", len(df))
    print("\nTarget distribution:")
    print(pd.Series(y).value_counts().sort_index())

    models = {
        "char_logistic": char_logistic(),
        "word_logistic": word_logistic(),
        "char_svm": char_svm(),
        "word_svm": word_svm(),
    }

    candidates = [
        "char_logistic",
        "word_logistic",
        "char_svm",
        "word_svm",
        "logistic_blend_25char_75word",
        "logistic_blend_50char_50word",
        "logistic_blend_75char_25word",
        "svm_blend_50char_50word",
        "char_logistic_plus_word_svm",
        "char_svm_plus_word_svm_plus_char_logistic",
    ]

    results = {
        name: []
        for name in candidates
    }

    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    total_folds = (
        splitter.get_n_splits(X, y)
    )

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y),
        start=1,
    ):

        if fold == 1 or fold % 10 == 0:
            logger.info(
                "Running fold %d/%d",
                fold,
                total_folds,
            )

        X_train = X[train_idx]
        X_fold = X[val_idx]

        y_train = y[train_idx]
        y_fold = y[val_idx]

        fold_scores = {}

        for name, model in models.items():

            fitted = clone(model)

            fitted.fit(
                X_train,
                y_train,
            )

            fold_scores[name] = get_scores(
                fitted,
                X_fold,
            )

            results[name].append(
                roc_auc_score(
                    y_fold,
                    fold_scores[name],
                )
            )

        # --------------------------------------------
        # Logistic probability ensembles
        # --------------------------------------------

        char_prob = fold_scores[
            "char_logistic"
        ]

        word_prob = fold_scores[
            "word_logistic"
        ]

        logistic_blends = {
            "logistic_blend_25char_75word":
                0.25 * char_prob
                + 0.75 * word_prob,

            "logistic_blend_50char_50word":
                0.50 * char_prob
                + 0.50 * word_prob,

            "logistic_blend_75char_25word":
                0.75 * char_prob
                + 0.25 * word_prob,
        }

        for name, scores in logistic_blends.items():

            results[name].append(
                roc_auc_score(
                    y_fold,
                    scores,
                )
            )

        # --------------------------------------------
        # Rank-based cross-model ensembles
        # --------------------------------------------

        char_log_rank = rank_normalize(
            fold_scores["char_logistic"]
        )

        char_svm_rank = rank_normalize(
            fold_scores["char_svm"]
        )

        word_svm_rank = rank_normalize(
            fold_scores["word_svm"]
        )

        svm_blend = (
            0.50 * char_svm_rank
            + 0.50 * word_svm_rank
        )

        results[
            "svm_blend_50char_50word"
        ].append(
            roc_auc_score(
                y_fold,
                svm_blend,
            )
        )

        char_log_word_svm = (
            0.50 * char_log_rank
            + 0.50 * word_svm_rank
        )

        results[
            "char_logistic_plus_word_svm"
        ].append(
            roc_auc_score(
                y_fold,
                char_log_word_svm,
            )
        )

        three_model = (
            0.40 * char_log_rank
            + 0.30 * char_svm_rank
            + 0.30 * word_svm_rank
        )

        results[
            "char_svm_plus_word_svm_plus_char_logistic"
        ].append(
            roc_auc_score(
                y_fold,
                three_model,
            )
        )

    # --------------------------------------------
    # Summaries
    # --------------------------------------------

    summary = {
        name: summarize(values)
        for name, values in results.items()
    }

    summary_df = pd.DataFrame(
        [
            {
                "model": name,
                **metrics,
            }
            for name, metrics in summary.items()
        ]
    ).sort_values(
        "mean_auc",
        ascending=False,
    )

    print("\n" + "=" * 100)
    print("PROPER REPEATED-CV ENSEMBLE RESULTS")
    print("=" * 100)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("=" * 100)

    winner = summary_df.iloc[0]

    print(
        f"\nWinner: {winner['model']}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{winner['mean_auc']:.4f}"
    )

    print(
        f"Std ROC-AUC: "
        f"{winner['std_auc']:.4f}"
    )

    print(
        f"Median ROC-AUC: "
        f"{winner['median_auc']:.4f}"
    )

    print("\nTEST SET USED: NO")

    summary_df.to_csv(
        OUTPUT_DIR /
        "ensemble_stability_results.csv",
        index=False,
    )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "target": "rating >= 4",
                "cv": "RepeatedStratifiedKFold: 5 folds x 10 repeats",
                "test_set_used": False,
                "results": summary,
                "winner": winner["model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()