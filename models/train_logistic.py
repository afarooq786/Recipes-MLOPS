#!/usr/bin/env python3
"""
models/train_logistic.py — Step 9 + Step 11

Trains Logistic Regression candidate models on the processed feature set,
evaluates them on the validation split using the shared evaluation module,
logs every candidate to MLflow, and logs the selected Logistic Regression
champion with its model and evaluation artifacts.

Important:
- The test split is intentionally not used here.
- rating-derived popularity features are excluded to avoid target leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.metrics import (
    evaluate,
    plot_precision_at_k_bar,
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

logger = logging.getLogger("train_logistic")


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

# We log the final selected model manually.
mlflow.sklearn.autolog(
    log_models=False,
    silent=True,
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
)


# ---------------------------------------------------------------------
# Paths / columns
# ---------------------------------------------------------------------

TRAIN_PATH = Path("data/processed/features/train.csv")
VAL_PATH = Path("data/processed/features/val.csv")

OUTPUT_DIR = Path(
    "evaluation/results/logistic_regression"
)

LABEL_COL = "label"
GROUP_COL = "cuisine_path"


# ---------------------------------------------------------------------
# Leakage-safe feature list
# ---------------------------------------------------------------------

NUMERIC_FEATURES = [
    "servings",
    "prep_time_minutes",
    "cook_time_minutes",
    "total_time_minutes",
    "ingredient_count",
    "ingredient_unique_word_count",
    "ingredient_diversity_ratio",
    "has_chicken",
    "has_beef",
    "has_pork",
    "has_fish_seafood",
    "has_tofu_plant",
    "has_egg",
    "has_dairy_cheese",
    "prep_to_cook_ratio",
    "is_likely_vegetarian",
    "is_likely_vegan",
    "is_likely_gluten_free",
]

CATEGORICAL_FEATURES = [
    "total_time_bucket",
    "cuisine_path",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation feature datasets."""

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    logger.info(
        "Loaded train shape: %s",
        train_df.shape,
    )

    logger.info(
        "Loaded validation shape: %s",
        val_df.shape,
    )

    return train_df, val_df


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def build_pipeline(
    c: float,
    class_weight: str | None,
) -> Pipeline:
    """Create preprocessing + Logistic Regression pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LogisticRegression(
        C=c,
        class_weight=class_weight,
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------

def main() -> None:

    train_df, val_df = load_data()

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL_COL]

    X_val = val_df[FEATURES]
    y_val = val_df[LABEL_COL]

    groups = val_df[GROUP_COL]

    candidate_configs = [
        {
            "C": 0.01,
            "class_weight": None,
        },
        {
            "C": 0.1,
            "class_weight": None,
        },
        {
            "C": 1.0,
            "class_weight": None,
        },
        {
            "C": 10.0,
            "class_weight": None,
        },
        {
            "C": 0.01,
            "class_weight": "balanced",
        },
        {
            "C": 0.1,
            "class_weight": "balanced",
        },
        {
            "C": 1.0,
            "class_weight": "balanced",
        },
        {
            "C": 10.0,
            "class_weight": "balanced",
        },
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    best_result = None
    best_pipeline = None
    best_scores = None

    # -----------------------------------------------------------------
    # Candidate runs
    # -----------------------------------------------------------------

    for config in candidate_configs:

        logger.info(
            "Training Logistic Regression: "
            "C=%s, class_weight=%s",
            config["C"],
            config["class_weight"],
        )

        run_name = (
            f"logistic_structured_"
            f"C{config['C']}_"
            f"cw_{config['class_weight']}"
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            pipeline = build_pipeline(
                c=config["C"],
                class_weight=config[
                    "class_weight"
                ],
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            val_scores = (
                pipeline.predict_proba(
                    X_val
                )[:, 1]
            )

            metrics = evaluate(
                y_true=y_val.to_numpy(),
                y_score=val_scores,
                groups=groups.to_numpy(),
                k=5,
            )

            result = {
                "model":
                    "logistic_regression",
                "C":
                    config["C"],
                "class_weight":
                    config["class_weight"],
                "roc_auc":
                    metrics["roc_auc"],
                "precision_at_5":
                    metrics[
                        "precision_at_k_overall"
                    ],
                "grouped_precision_at_5":
                    metrics[
                        "grouped_precision_at_k"
                    ][
                        "macro_avg_precision_at_k"
                    ],
            }

            # Project-specific metadata.
            mlflow.log_param(
                "model_type",
                "logistic_regression",
            )

            mlflow.log_param(
                "feature_set",
                "structured",
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
                "C_selected",
                config["C"],
            )

            mlflow.log_param(
                "class_weight_selected",
                str(
                    config[
                        "class_weight"
                    ]
                ),
            )

            # Project evaluation metrics.
            mlflow.log_metric(
                "validation_roc_auc",
                result["roc_auc"],
            )

            mlflow.log_metric(
                "validation_precision_at_5",
                result[
                    "precision_at_5"
                ],
            )

            mlflow.log_metric(
                "validation_grouped_precision_at_5",
                result[
                    "grouped_precision_at_5"
                ],
            )

            all_results.append(
                result
            )

            logger.info(
                "Validation ROC-AUC=%.4f | "
                "Precision@5=%.4f | "
                "Grouped Precision@5=%.4f",
                result["roc_auc"],
                result[
                    "precision_at_5"
                ],
                result[
                    "grouped_precision_at_5"
                ],
            )

            if (
                best_result is None
                or result["roc_auc"]
                > best_result["roc_auc"]
            ):
                best_result = result
                best_pipeline = pipeline
                best_scores = val_scores

    # -----------------------------------------------------------------
    # Save candidate results
    # -----------------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "candidate_results.json"
    )

    results_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        )
    )

    logger.info(
        "Best Logistic Regression config: %s",
        best_result,
    )

    # -----------------------------------------------------------------
    # Save selected local model
    # -----------------------------------------------------------------

    model_path = (
        OUTPUT_DIR
        / "logistic_regression_best.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path,
    )

    logger.info(
        "Saved best Logistic Regression "
        "pipeline to %s",
        model_path,
    )

    # -----------------------------------------------------------------
    # Evaluate selected model
    # -----------------------------------------------------------------

    best_metrics = evaluate(
        y_true=y_val.to_numpy(),
        y_score=best_scores,
        groups=groups.to_numpy(),
        k=5,
    )

    metrics_path = save_results(
        best_metrics,
        out_dir=OUTPUT_DIR,
        model_name=(
            "logistic_regression_best"
        ),
    )

    roc_path = (
        OUTPUT_DIR
        / "logistic_regression_best_roc.png"
    )

    plot_roc_curve(
        y_val.to_numpy(),
        best_scores,
        roc_path,
        "Logistic Regression",
    )

    precision_path = (
        OUTPUT_DIR
        / (
            "logistic_regression_best_"
            "precision_at_5.png"
        )
    )

    plot_precision_at_k_bar(
        best_metrics[
            "grouped_precision_at_k"
        ],
        precision_path,
        "Logistic Regression",
    )

    predictions_path = (
        OUTPUT_DIR
        / (
            "logistic_regression_best_"
            "predictions.csv"
        )
    )

    predictions_df = pd.DataFrame(
        {
            "label":
                y_val,
            "score":
                best_scores,
            GROUP_COL:
                groups,
        }
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Dedicated MLflow champion run
    # -----------------------------------------------------------------

    with mlflow.start_run(
        run_name=(
            "logistic_structured_champion"
        )
    ):

        mlflow.set_tag(
            "candidate_status",
            "logistic_champion",
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
            "structured",
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
            "C_selected",
            best_result["C"],
        )

        mlflow.log_param(
            "class_weight_selected",
            str(
                best_result[
                    "class_weight"
                ]
            ),
        )

        mlflow.log_param(
            "n_train_rows",
            len(train_df),
        )

        mlflow.log_param(
            "n_validation_rows",
            len(val_df),
        )

        mlflow.log_metric(
            "validation_roc_auc",
            best_result["roc_auc"],
        )

        mlflow.log_metric(
            "validation_precision_at_5",
            best_result[
                "precision_at_5"
            ],
        )

        mlflow.log_metric(
            "validation_grouped_precision_at_5",
            best_result[
                "grouped_precision_at_5"
            ],
        )

        # Log model.
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # Log evaluation artifacts.
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(roc_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(precision_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(predictions_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(results_path),
            artifact_path="evaluation",
        )

    logger.info(
        "Finished Logistic Regression "
        "MLflow tracking."
    )


if __name__ == "__main__":
    main()