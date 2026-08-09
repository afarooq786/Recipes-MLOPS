#!/usr/bin/env python3
"""
Final challenger experiments for the rating >= 4 target.

Compares:
1. Optimized character TF-IDF + Logistic Regression
2. Optimized word TF-IDF + Logistic Regression
3. Character TF-IDF + Linear SVM
4. Word TF-IDF + Linear SVM
5. Probability/ranking ensembles of the strongest models

Model comparison is based on repeated stratified cross-validation using
TRAIN ONLY. Validation is evaluated after the CV comparison.

The isolated test set is never read.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from evaluation.metrics import evaluate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_challengers")


TRAIN_PATH = Path("data/processed/features/train.csv")
VAL_PATH = Path("data/processed/features/val.csv")
OUTPUT_DIR = Path("evaluation/results/challengers")

TEXT_COL = "ingredients_parsed"
TARGET_COL = "label"
RANDOM_STATE = 42


def clean_text(text: object) -> str:
    """Normalize ingredient text while retaining useful lexical information."""
    text = str(text).lower()

    # Remove standalone numbers.
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

    unit_pattern = r"\b(?:" + "|".join(units) + r")\b"
    text = re.sub(unit_pattern, " ", text)

    return re.sub(r"\s+", " ", text).strip()


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    train["text"] = train[TEXT_COL].fillna("").apply(clean_text)
    val["text"] = val[TEXT_COL].fillna("").apply(clean_text)

    logger.info("Train: %s", train.shape)
    logger.info("Validation: %s", val.shape)

    return train, val


def make_char_logistic():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_word_logistic():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_char_svm():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                ),
            ),
            (
                "model",
                LinearSVC(
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_word_svm():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                ),
            ),
            (
                "model",
                LinearSVC(
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_model(name, pipeline, param_grid, X, y):
    logger.info("=" * 70)
    logger.info("Tuning %s", name)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X, y)

    logger.info(
        "%s best CV AUC: %.4f",
        name,
        search.best_score_,
    )
    logger.info(
        "%s best params: %s",
        name,
        search.best_params_,
    )

    return search.best_estimator_, search.best_params_, search.best_score_


def repeated_cv_predictions(model, X, y):
    """
    Evaluate one model across 10 repeats x 5 folds.

    Returns the AUC for each individual held-out fold plus an averaged
    out-of-fold score for every training observation.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    fold_aucs = []
    prediction_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=float)

    X = np.asarray(X)
    y = np.asarray(y)

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        fitted = clone(model)
        fitted.fit(X[train_idx], y[train_idx])

        if hasattr(fitted, "predict_proba"):
            scores = fitted.predict_proba(X[test_idx])[:, 1]
        else:
            scores = fitted.decision_function(X[test_idx])

        auc = roc_auc_score(y[test_idx], scores)
        fold_aucs.append(auc)

        prediction_sum[test_idx] += scores
        prediction_count[test_idx] += 1

    averaged_scores = prediction_sum / prediction_count

    return np.asarray(fold_aucs), averaged_scores


def normalize_scores(scores):
    """
    Rank-normalize scores to 0..1.

    Useful for combining Logistic Regression probabilities with SVM
    decision scores without pretending they share the same scale.
    """
    scores = pd.Series(scores)
    return scores.rank(method="average", pct=True).to_numpy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val = load_data()

    X_train = train["text"].to_numpy()
    y_train = train[TARGET_COL].to_numpy()

    X_val = val["text"].to_numpy()
    y_val = val[TARGET_COL].to_numpy()

    # ---------------------------------------------------------
    # 1. Tune four serious individual challengers
    # ---------------------------------------------------------

    char_logistic, char_log_params, _ = tune_model(
        "char_logistic",
        make_char_logistic(),
        {
            "tfidf__ngram_range": [
                (2, 4),
                (3, 5),
                (4, 6),
                (5, 7),
            ],
            "tfidf__min_df": [1, 2, 3],
            "tfidf__sublinear_tf": [False, True],
            "model__C": [
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
            ],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
    )

    word_logistic, word_log_params, _ = tune_model(
        "word_logistic",
        make_word_logistic(),
        {
            "tfidf__ngram_range": [
                (1, 1),
                (1, 2),
                (1, 3),
            ],
            "tfidf__min_df": [1, 2, 3],
            "tfidf__sublinear_tf": [False, True],
            "model__C": [
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
            ],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
    )

    char_svm, char_svm_params, _ = tune_model(
        "char_svm",
        make_char_svm(),
        {
            "tfidf__ngram_range": [
                (2, 4),
                (3, 5),
                (4, 6),
                (5, 7),
            ],
            "tfidf__min_df": [1, 2, 3],
            "tfidf__sublinear_tf": [False, True],
            "model__C": [
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
            ],
            "model__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
    )

    word_svm, word_svm_params, _ = tune_model(
        "word_svm",
        make_word_svm(),
        {
            "tfidf__ngram_range": [
                (1, 1),
                (1, 2),
                (1, 3),
            ],
            "tfidf__min_df": [1, 2, 3],
            "tfidf__sublinear_tf": [False, True],
            "model__C": [
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
            ],
            "model__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
    )

    models = {
        "char_logistic": char_logistic,
        "word_logistic": word_logistic,
        "char_svm": char_svm,
        "word_svm": word_svm,
    }

    params = {
        "char_logistic": char_log_params,
        "word_logistic": word_log_params,
        "char_svm": char_svm_params,
        "word_svm": word_svm_params,
    }

    # ---------------------------------------------------------
    # 2. Repeated CV
    # ---------------------------------------------------------

    cv_scores = {}
    oof_predictions = {}

    for name, model in models.items():
        logger.info("Repeated CV: %s", name)

        scores, predictions = repeated_cv_predictions(
            model,
            X_train,
            y_train,
        )

        cv_scores[name] = scores
        oof_predictions[name] = predictions

        logger.info(
            "%s | mean=%.4f std=%.4f median=%.4f",
            name,
            scores.mean(),
            scores.std(),
            np.median(scores),
        )

    # ---------------------------------------------------------
    # 3. Ensemble experiments
    # ---------------------------------------------------------

    ensemble_candidates = {}

    # Probability blend of the two Logistic Regression models.
    for weight in [
        0.25,
        0.50,
        0.75,
    ]:
        name = f"logistic_blend_char_{weight:.2f}"

        ensemble_candidates[name] = (
            weight * oof_predictions["char_logistic"]
            + (1 - weight) * oof_predictions["word_logistic"]
        )

    # Rank-based blends allow us to safely combine SVM decision scores
    # with Logistic Regression probabilities.
    char_log_rank = normalize_scores(
        oof_predictions["char_logistic"]
    )
    word_log_rank = normalize_scores(
        oof_predictions["word_logistic"]
    )
    char_svm_rank = normalize_scores(
        oof_predictions["char_svm"]
    )
    word_svm_rank = normalize_scores(
        oof_predictions["word_svm"]
    )

    ensemble_candidates[
        "rank_char_logistic_plus_char_svm"
    ] = (
        0.5 * char_log_rank
        + 0.5 * char_svm_rank
    )

    ensemble_candidates[
        "rank_all_four"
    ] = (
        0.4 * char_log_rank
        + 0.2 * word_log_rank
        + 0.3 * char_svm_rank
        + 0.1 * word_svm_rank
    )

    # Evaluate OOF ensemble ranking.
    ensemble_auc = {}

    for name, scores in ensemble_candidates.items():
        auc = roc_auc_score(
            y_train,
            scores,
        )
        ensemble_auc[name] = auc

    # ---------------------------------------------------------
    # 4. Print CV comparison
    # ---------------------------------------------------------

    rows = []

    for name, scores in cv_scores.items():
        rows.append(
            {
                "model": name,
                "mean_fold_auc": scores.mean(),
                "std_fold_auc": scores.std(),
                "median_fold_auc": np.median(scores),
                "oof_auc": roc_auc_score(
                    y_train,
                    oof_predictions[name],
                ),
            }
        )

    for name, auc in ensemble_auc.items():
        rows.append(
            {
                "model": name,
                "mean_fold_auc": np.nan,
                "std_fold_auc": np.nan,
                "median_fold_auc": np.nan,
                "oof_auc": auc,
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        "oof_auc",
        ascending=False,
    )

    print("\n" + "=" * 85)
    print("FINAL CHALLENGER CV RESULTS")
    print("=" * 85)
    print(comparison.to_string(index=False))
    print("=" * 85)

    comparison.to_csv(
        OUTPUT_DIR / "challenger_cv_results.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Validation results
    #
    # We evaluate all four individuals and the simple Logistic
    # blends. This is diagnostic; CV remains our primary basis
    # for deciding whether we genuinely improved.
    # ---------------------------------------------------------

    val_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_val)[:, 1]
        else:
            scores = model.decision_function(X_val)

        val_scores[name] = scores

    for weight in [0.25, 0.50, 0.75]:
        name = f"logistic_blend_char_{weight:.2f}"

        val_scores[name] = (
            weight * val_scores["char_logistic"]
            + (1 - weight) * val_scores["word_logistic"]
        )

    # Rank ensembles on validation.
    val_scores[
        "rank_char_logistic_plus_char_svm"
    ] = (
        0.5 * normalize_scores(val_scores["char_logistic"])
        + 0.5 * normalize_scores(val_scores["char_svm"])
    )

    val_scores[
        "rank_all_four"
    ] = (
        0.4 * normalize_scores(val_scores["char_logistic"])
        + 0.2 * normalize_scores(val_scores["word_logistic"])
        + 0.3 * normalize_scores(val_scores["char_svm"])
        + 0.1 * normalize_scores(val_scores["word_svm"])
    )

    validation_rows = []

    for name, scores in val_scores.items():
        results = evaluate(
            y_true=y_val,
            y_score=scores,
            groups=val["cuisine_path"].to_numpy(),
            k=5,
        )

        validation_rows.append(
            {
                "model": name,
                "validation_roc_auc": results["roc_auc"],
                "validation_precision_at_5": results[
                    "precision_at_k_overall"
                ],
                "validation_grouped_precision_at_5": results[
                    "grouped_precision_at_k"
                ]["macro_avg_precision_at_k"],
            }
        )

    validation_df = pd.DataFrame(
        validation_rows
    ).sort_values(
        "validation_roc_auc",
        ascending=False,
    )

    print("\n" + "=" * 85)
    print("VALIDATION RESULTS")
    print("=" * 85)
    print(validation_df.to_string(index=False))
    print("=" * 85)

    validation_df.to_csv(
        OUTPUT_DIR / "challenger_validation_results.csv",
        index=False,
    )

    # Save the four fitted component models.
    for name, model in models.items():
        joblib.dump(
            model,
            OUTPUT_DIR / f"{name}.joblib",
        )

    summary = {
        "target": "rating >= 4",
        "test_set_used": False,
        "individual_model_params": params,
        "cv_results": comparison.to_dict(orient="records"),
        "validation_results": validation_df.to_dict(orient="records"),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    print("\nTEST SET USED: NO")


if __name__ == "__main__":
    main()