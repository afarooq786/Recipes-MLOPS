#!/usr/bin/env python3
"""
models/train_final_candidates.py — Step 11 final candidate tracking

Trains and logs the strongest final text-model candidates:

1. Character TF-IDF + Logistic Regression
2. Word TF-IDF + Linear SVM
3. Rank-based ensemble:
       50% Character Logistic rank
       50% Word SVM rank

The ensemble is packaged as a single MLflow pyfunc model so downstream
serving can treat it as one deployable artifact.

Important:
- Target remains rating >= 4.
- Validation is used for final candidate evaluation.
- Test data is never read.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from evaluation.metrics import (
    evaluate,
    plot_roc_curve,
    save_results,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("train_final_candidates")


# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------
import os
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
MLFLOW_EXPERIMENT = "recipe-recommender-training"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_PATH = Path(
    "data/processed/features/train.csv"
)

VAL_PATH = Path(
    "data/processed/features/val.csv"
)

OUTPUT_DIR = Path(
    "evaluation/results/final_candidates"
)

TEXT_COL = "ingredients_parsed"
TARGET_COL = "label"
GROUP_COL = "cuisine_path"


# ---------------------------------------------------------------------
# Frozen CV results from the proper repeated-CV comparison
# ---------------------------------------------------------------------

CHAR_LOGISTIC_CV_MEAN = 0.6276982564679416
CHAR_LOGISTIC_CV_STD = 0.09957458592954689

WORD_SVM_CV_MEAN = 0.6476743532058492
WORD_SVM_CV_STD = 0.09359382254909848

ENSEMBLE_CV_MEAN = 0.681039
ENSEMBLE_CV_STD = 0.089698
ENSEMBLE_CV_MEDIAN = 0.668799


# ---------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------

def clean_text(text: object) -> str:
    """Normalize ingredient text."""

    text = str(text).lower()

    text = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ",
        text,
    )

    units = [
        "cup",
        "cups",
        "tablespoon",
        "tablespoons",
        "teaspoon",
        "teaspoons",
        "ounce",
        "ounces",
        "oz",
        "pound",
        "pounds",
        "lb",
        "lbs",
        "gram",
        "grams",
        "kg",
        "ml",
        "liter",
        "liters",
        "package",
        "packages",
        "can",
        "cans",
        "slice",
        "slices",
    ]

    unit_pattern = (
        r"\b(?:"
        + "|".join(units)
        + r")\b"
    )

    text = re.sub(
        unit_pattern,
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def prepare_text(
    df: pd.DataFrame,
) -> np.ndarray:
    """Extract and clean ingredient text."""

    return (
        df[TEXT_COL]
        .fillna("")
        .apply(clean_text)
        .to_numpy()
    )


# ---------------------------------------------------------------------
# Final tuned models
# ---------------------------------------------------------------------

def build_char_logistic() -> Pipeline:
    """
    Tuned character TF-IDF Logistic Regression.

    Parameters come from train_challengers.py.
    """

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
                    l1_ratio=1,
                    class_weight=None,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_word_svm() -> Pipeline:
    """
    Tuned word TF-IDF Linear SVM.

    Parameters come from train_challengers.py.
    """

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


# ---------------------------------------------------------------------
# Rank utilities
# ---------------------------------------------------------------------

def rank_normalize(
    scores: np.ndarray,
) -> np.ndarray:
    """
    Convert arbitrary model scores to percentile ranks.

    This allows Logistic probabilities and SVM decision scores
    to be combined without assuming they share a numeric scale.
    """

    return (
        pd.Series(scores)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )


def ensemble_scores(
    char_scores: np.ndarray,
    word_svm_scores: np.ndarray,
) -> np.ndarray:
    """50/50 rank ensemble."""

    char_rank = rank_normalize(
        char_scores
    )

    svm_rank = rank_normalize(
        word_svm_scores
    )

    return (
        0.50 * char_rank
        + 0.50 * svm_rank
    )


# ---------------------------------------------------------------------
# MLflow custom ensemble
# ---------------------------------------------------------------------

class RecipeRankingEnsemble(
    mlflow.pyfunc.PythonModel
):
    """
    MLflow wrapper around the two fitted component models.

    Important:
    Rank normalization is performed across the input batch, so this model
    is designed to rank a candidate set of recipes rather than score one
    isolated recipe independently.
    """

    def load_context(
        self,
        context,
    ):

        self.char_model = joblib.load(
            context.artifacts[
                "char_model"
            ]
        )

        self.word_svm_model = joblib.load(
            context.artifacts[
                "word_svm_model"
            ]
        )

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):

        if isinstance(
            model_input,
            pd.DataFrame,
        ):
            if (
                "ingredients_parsed"
                in model_input.columns
            ):
                texts = (
                    model_input[
                        "ingredients_parsed"
                    ]
                    .fillna("")
                    .apply(clean_text)
                    .to_numpy()
                )

            elif "text" in model_input.columns:
                texts = (
                    model_input[
                        "text"
                    ]
                    .fillna("")
                    .apply(clean_text)
                    .to_numpy()
                )

            else:
                raise ValueError(
                    "Expected a DataFrame "
                    "containing either "
                    "'ingredients_parsed' "
                    "or 'text'."
                )

        else:
            texts = np.asarray(
                model_input
            )

        char_scores = (
            self.char_model
            .predict_proba(
                texts
            )[:, 1]
        )

        svm_scores = (
            self.word_svm_model
            .decision_function(
                texts
            )
        )

        scores = ensemble_scores(
            char_scores,
            svm_scores,
        )

        return pd.DataFrame(
            {
                "score": scores,
            }
        )


# ---------------------------------------------------------------------
# Individual MLflow model logging
# ---------------------------------------------------------------------

def log_char_logistic(
    model,
    metrics,
    metrics_path,
    roc_path,
):

    with mlflow.start_run(
        run_name=(
            "char_logistic_text_candidate"
        )
    ):

        mlflow.set_tag(
            "candidate_status",
            "final_candidate",
        )

        mlflow.set_tag(
            "stage",
            "validation",
        )

        mlflow.log_param(
            "model_type",
            "logistic_regression",
        )

        mlflow.log_param(
            "feature_set",
            "char_tfidf",
        )

        mlflow.log_param(
            "target",
            "rating>=4",
        )

        mlflow.log_param(
            "test_set_used",
            False,
        )

        mlflow.log_param(
            "tfidf_analyzer",
            "char_wb",
        )

        mlflow.log_param(
            "tfidf_ngram_range",
            "3-5",
        )

        mlflow.log_param(
            "tfidf_min_df",
            3,
        )

        mlflow.log_param(
            "tfidf_sublinear_tf",
            True,
        )

        mlflow.log_param(
            "C",
            3.0,
        )

        mlflow.log_param(
            "penalty",
            "l1",
        )

        mlflow.log_param(
            "class_weight",
            "None",
        )

        mlflow.log_metric(
            "repeated_cv_mean_roc_auc",
            CHAR_LOGISTIC_CV_MEAN,
        )

        mlflow.log_metric(
            "repeated_cv_std_roc_auc",
            CHAR_LOGISTIC_CV_STD,
        )

        mlflow.log_metric(
            "validation_roc_auc",
            metrics["roc_auc"],
        )

        mlflow.log_metric(
            "validation_precision_at_5",
            metrics[
                "precision_at_k_overall"
            ],
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(roc_path),
            artifact_path="evaluation",
        )


def log_word_svm(
    model,
    metrics,
    metrics_path,
    roc_path,
):

    with mlflow.start_run(
        run_name=(
            "word_svm_text_candidate"
        )
    ):

        mlflow.set_tag(
            "candidate_status",
            "final_candidate",
        )

        mlflow.set_tag(
            "stage",
            "validation",
        )

        mlflow.log_param(
            "model_type",
            "linear_svm",
        )

        mlflow.log_param(
            "feature_set",
            "word_tfidf",
        )

        mlflow.log_param(
            "target",
            "rating>=4",
        )

        mlflow.log_param(
            "test_set_used",
            False,
        )

        mlflow.log_param(
            "tfidf_analyzer",
            "word",
        )

        mlflow.log_param(
            "tfidf_ngram_range",
            "1-3",
        )

        mlflow.log_param(
            "tfidf_min_df",
            3,
        )

        mlflow.log_param(
            "tfidf_sublinear_tf",
            True,
        )

        mlflow.log_param(
            "C",
            10.0,
        )

        mlflow.log_param(
            "class_weight",
            "None",
        )

        mlflow.log_metric(
            "repeated_cv_mean_roc_auc",
            WORD_SVM_CV_MEAN,
        )

        mlflow.log_metric(
            "repeated_cv_std_roc_auc",
            WORD_SVM_CV_STD,
        )

        mlflow.log_metric(
            "validation_roc_auc",
            metrics["roc_auc"],
        )

        mlflow.log_metric(
            "validation_precision_at_5",
            metrics[
                "precision_at_k_overall"
            ],
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(roc_path),
            artifact_path="evaluation",
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

    logger.info(
        "Loaded train shape: %s",
        train_df.shape,
    )

    logger.info(
        "Loaded validation shape: %s",
        val_df.shape,
    )

    X_train = prepare_text(
        train_df
    )

    X_val = prepare_text(
        val_df
    )

    y_train = (
        train_df[
            TARGET_COL
        ].to_numpy()
    )

    y_val = (
        val_df[
            TARGET_COL
        ].to_numpy()
    )

    groups = (
        val_df[
            GROUP_COL
        ].to_numpy()
    )

    # -----------------------------------------------------------------
    # Character Logistic
    # -----------------------------------------------------------------

    logger.info(
        "Training final character "
        "Logistic Regression..."
    )

    char_model = (
        build_char_logistic()
    )

    char_model.fit(
        X_train,
        y_train,
    )

    char_val_scores = (
        char_model
        .predict_proba(
            X_val
        )[:, 1]
    )

    char_metrics = evaluate(
        y_true=y_val,
        y_score=char_val_scores,
        groups=groups,
        k=5,
    )

    char_metrics_path = (
        save_results(
            char_metrics,
            OUTPUT_DIR,
            "char_logistic",
        )
    )

    char_roc_path = (
        OUTPUT_DIR
        / "char_logistic_roc.png"
    )

    plot_roc_curve(
        y_val,
        char_val_scores,
        char_roc_path,
        "Character Logistic Regression",
    )

    char_joblib_path = (
        OUTPUT_DIR
        / "char_logistic.joblib"
    )

    joblib.dump(
        char_model,
        char_joblib_path,
    )

    log_char_logistic(
        char_model,
        char_metrics,
        char_metrics_path,
        char_roc_path,
    )

    # -----------------------------------------------------------------
    # Word SVM
    # -----------------------------------------------------------------

    logger.info(
        "Training final word SVM..."
    )

    word_svm_model = (
        build_word_svm()
    )

    word_svm_model.fit(
        X_train,
        y_train,
    )

    word_svm_val_scores = (
        word_svm_model
        .decision_function(
            X_val
        )
    )

    word_svm_metrics = evaluate(
        y_true=y_val,
        y_score=word_svm_val_scores,
        groups=groups,
        k=5,
    )

    word_svm_metrics_path = (
        save_results(
            word_svm_metrics,
            OUTPUT_DIR,
            "word_svm",
        )
    )

    word_svm_roc_path = (
        OUTPUT_DIR
        / "word_svm_roc.png"
    )

    plot_roc_curve(
        y_val,
        word_svm_val_scores,
        word_svm_roc_path,
        "Word Linear SVM",
    )

    word_svm_joblib_path = (
        OUTPUT_DIR
        / "word_svm.joblib"
    )

    joblib.dump(
        word_svm_model,
        word_svm_joblib_path,
    )

    log_word_svm(
        word_svm_model,
        word_svm_metrics,
        word_svm_metrics_path,
        word_svm_roc_path,
    )

    # -----------------------------------------------------------------
    # Ensemble
    # -----------------------------------------------------------------

    logger.info(
        "Evaluating final ensemble..."
    )

    ensemble_val_scores = (
        ensemble_scores(
            char_val_scores,
            word_svm_val_scores,
        )
    )

    ensemble_metrics = evaluate(
        y_true=y_val,
        y_score=ensemble_val_scores,
        groups=groups,
        k=5,
    )

    ensemble_metrics_path = (
        save_results(
            ensemble_metrics,
            OUTPUT_DIR,
            "ensemble_text",
        )
    )

    ensemble_roc_path = (
        OUTPUT_DIR
        / "ensemble_text_roc.png"
    )

    plot_roc_curve(
        y_val,
        ensemble_val_scores,
        ensemble_roc_path,
        "Char Logistic + Word SVM Ensemble",
    )

    ensemble_predictions_path = (
        OUTPUT_DIR
        / "ensemble_predictions.csv"
    )

    pd.DataFrame(
        {
            "label":
                y_val,
            "score":
                ensemble_val_scores,
            GROUP_COL:
                groups,
        }
    ).to_csv(
        ensemble_predictions_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Log ensemble as one MLflow pyfunc model
    # -----------------------------------------------------------------

    with mlflow.start_run(
        run_name=(
            "ensemble_text_champion"
        )
    ):

        mlflow.set_tag(
            "candidate_status",
            "overall_champion_candidate",
        )

        mlflow.set_tag(
            "stage",
            "validation",
        )

        mlflow.log_param(
            "model_type",
            "rank_ensemble",
        )

        mlflow.log_param(
            "feature_set",
            "ingredient_text",
        )

        mlflow.log_param(
            "target",
            "rating>=4",
        )

        mlflow.log_param(
            "test_set_used",
            False,
        )

        mlflow.log_param(
            "component_1",
            "char_logistic",
        )

        mlflow.log_param(
            "component_2",
            "word_svm",
        )

        mlflow.log_param(
            "char_logistic_weight",
            0.5,
        )

        mlflow.log_param(
            "word_svm_weight",
            0.5,
        )

        mlflow.log_param(
            "ensemble_strategy",
            "rank_normalized_average",
        )

        mlflow.log_param(
            "cv_protocol",
            "5-fold x 10 repeats",
        )

        mlflow.log_metric(
            "repeated_cv_mean_roc_auc",
            ENSEMBLE_CV_MEAN,
        )

        mlflow.log_metric(
            "repeated_cv_std_roc_auc",
            ENSEMBLE_CV_STD,
        )

        mlflow.log_metric(
            "repeated_cv_median_roc_auc",
            ENSEMBLE_CV_MEDIAN,
        )

        mlflow.log_metric(
            "validation_roc_auc",
            ensemble_metrics[
                "roc_auc"
            ],
        )

        mlflow.log_metric(
            "validation_precision_at_5",
            ensemble_metrics[
                "precision_at_k_overall"
            ],
        )

        # Custom PythonModel with two fitted component artifacts.
        mlflow.pyfunc.log_model(
            name="model",
            python_model=(
                RecipeRankingEnsemble()
            ),
            artifacts={
                "char_model":
                    char_joblib_path.as_posix(),
                "word_svm_model":
                    word_svm_joblib_path.as_posix(),
            },
            pip_requirements=[
                "mlflow",
                "pandas",
                "numpy",
                "scikit-learn",
                "joblib",
            ],
        )

        mlflow.log_artifact(
            str(
                ensemble_metrics_path
            ),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(
                ensemble_roc_path
            ),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(
                ensemble_predictions_path
            ),
            artifact_path="evaluation",
        )

    # -----------------------------------------------------------------
    # Save comparison summary
    # -----------------------------------------------------------------

    comparison = {
        "target":
            "rating >= 4",

        "test_set_used":
            False,

        "char_logistic": {
            "repeated_cv_mean_auc":
                CHAR_LOGISTIC_CV_MEAN,
            "validation_auc":
                char_metrics[
                    "roc_auc"
                ],
        },

        "word_svm": {
            "repeated_cv_mean_auc":
                WORD_SVM_CV_MEAN,
            "validation_auc":
                word_svm_metrics[
                    "roc_auc"
                ],
        },

        "ensemble": {
            "repeated_cv_mean_auc":
                ENSEMBLE_CV_MEAN,
            "repeated_cv_std_auc":
                ENSEMBLE_CV_STD,
            "validation_auc":
                ensemble_metrics[
                    "roc_auc"
                ],
        },
    }

    comparison_path = (
        OUTPUT_DIR
        / "final_candidate_summary.json"
    )

    comparison_path.write_text(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    logger.info(
        "Final candidate comparison:"
    )

    logger.info(
        "Char Logistic | CV=%.4f | VAL=%.4f",
        CHAR_LOGISTIC_CV_MEAN,
        char_metrics["roc_auc"],
    )

    logger.info(
        "Word SVM | CV=%.4f | VAL=%.4f",
        WORD_SVM_CV_MEAN,
        word_svm_metrics["roc_auc"],
    )

    logger.info(
        "Ensemble | CV=%.4f | VAL=%.4f",
        ENSEMBLE_CV_MEAN,
        ensemble_metrics["roc_auc"],
    )

    logger.info(
        "TEST SET USED: NO"
    )


if __name__ == "__main__":
    main()