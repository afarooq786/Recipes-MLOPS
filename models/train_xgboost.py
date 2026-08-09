#!/usr/bin/env python3
"""
models/train_xgboost.py — Step 10 + Step 11

Tunes an XGBoost classifier using stratified cross-validation on the training
split, evaluates the best configuration once on the validation split, logs
GridSearchCV experiment information to MLflow, and logs the selected XGBoost
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

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

logger = logging.getLogger("train_xgboost")


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

# Let MLflow capture GridSearchCV / sklearn metadata.
# The selected champion model is logged manually afterward.
mlflow.sklearn.autolog(
    log_models=False,
    silent=True,
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
)


# ---------------------------------------------------------------------
# Paths / columns
# ---------------------------------------------------------------------

TRAIN_PATH = Path(
    "data/processed/features/train.csv"
)

VAL_PATH = Path(
    "data/processed/features/val.csv"
)

OUTPUT_DIR = Path(
    "evaluation/results/xgboost"
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

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load train and validation feature datasets."""

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

    return train_df, val_df


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Create preprocessing + XGBoost pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=1,
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
# Main
# ---------------------------------------------------------------------

def main() -> None:

    train_df, val_df = load_data()

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL_COL]

    X_val = val_df[FEATURES]
    y_val = val_df[LABEL_COL]

    groups = val_df[GROUP_COL]

    pipeline = build_pipeline()

    param_grid = {
        "model__n_estimators": [
            100,
            200,
            400,
        ],
        "model__max_depth": [
            2,
            3,
            4,
        ],
        "model__learning_rate": [
            0.03,
            0.1,
        ],
        "model__min_child_weight": [
            1,
            3,
        ],
        "model__subsample": [
            0.8,
            1.0,
        ],
        "model__colsample_bytree": [
            0.8,
            1.0,
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Grid-search MLflow run
    # -----------------------------------------------------------------

    with mlflow.start_run(
        run_name="xgboost_structured_grid_search"
    ):

        mlflow.set_tag(
            "stage",
            "model_selection",
        )

        mlflow.log_param(
            "model_type",
            "xgboost",
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
            "cv_folds",
            5,
        )

        mlflow.log_param(
            "scoring_metric",
            "roc_auc",
        )

        logger.info(
            "Starting XGBoost GridSearchCV..."
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Best CV ROC-AUC: %.4f",
            grid_search.best_score_,
        )

        logger.info(
            "Best parameters: %s",
            grid_search.best_params_,
        )

        # Explicit project-level GridSearch metrics.
        mlflow.log_metric(
            "best_cv_roc_auc",
            float(
                grid_search.best_score_
            ),
        )

        mlflow.log_metric(
            "n_parameter_combinations",
            len(
                grid_search.cv_results_[
                    "params"
                ]
            ),
        )

    # -----------------------------------------------------------------
    # Selected estimator
    # -----------------------------------------------------------------

    best_pipeline = (
        grid_search.best_estimator_
    )

    val_scores = (
        best_pipeline.predict_proba(
            X_val
        )[:, 1]
    )

    metrics = evaluate(
        y_true=y_val.to_numpy(),
        y_score=val_scores,
        groups=groups.to_numpy(),
        k=5,
    )

    # -----------------------------------------------------------------
    # Save evaluation artifacts
    # -----------------------------------------------------------------

    metrics_path = save_results(
        metrics,
        out_dir=OUTPUT_DIR,
        model_name="xgboost_best",
    )

    roc_path = (
        OUTPUT_DIR
        / "xgboost_best_roc.png"
    )

    plot_roc_curve(
        y_val.to_numpy(),
        val_scores,
        roc_path,
        "XGBoost",
    )

    precision_path = (
        OUTPUT_DIR
        / "xgboost_best_precision_at_5.png"
    )

    plot_precision_at_k_bar(
        metrics[
            "grouped_precision_at_k"
        ],
        precision_path,
        "XGBoost",
    )

    predictions_path = (
        OUTPUT_DIR
        / "xgboost_best_predictions.csv"
    )

    predictions_df = pd.DataFrame(
        {
            "label": y_val,
            "score": val_scores,
            GROUP_COL: groups,
        }
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Save local fitted pipeline
    # -----------------------------------------------------------------

    model_path = (
        OUTPUT_DIR
        / "xgboost_best.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path,
    )

    logger.info(
        "Saved best XGBoost pipeline to %s",
        model_path,
    )

    # -----------------------------------------------------------------
    # Save full grid search results
    # -----------------------------------------------------------------

    grid_results_path = (
        OUTPUT_DIR
        / "grid_search_results.csv"
    )

    cv_results = pd.DataFrame(
        grid_search.cv_results_
    )

    cv_results.to_csv(
        grid_results_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Save summary JSON
    # -----------------------------------------------------------------

    summary = {
        "best_cv_roc_auc":
            float(
                grid_search.best_score_
            ),
        "best_params":
            grid_search.best_params_,
        "validation_roc_auc":
            metrics["roc_auc"],
        "validation_precision_at_5":
            metrics[
                "precision_at_k_overall"
            ],
        "validation_grouped_precision_at_5":
            metrics[
                "grouped_precision_at_k"
            ][
                "macro_avg_precision_at_k"
            ],
        "n_parameter_combinations":
            len(
                grid_search.cv_results_[
                    "params"
                ]
            ),
        "test_set_used":
            False,
    }

    summary_path = (
        OUTPUT_DIR
        / "xgboost_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    logger.info(
        "Validation ROC-AUC=%.4f | "
        "Precision@5=%.4f | "
        "Grouped Precision@5=%.4f",
        metrics["roc_auc"],
        metrics[
            "precision_at_k_overall"
        ],
        metrics[
            "grouped_precision_at_k"
        ][
            "macro_avg_precision_at_k"
        ],
    )

    # -----------------------------------------------------------------
    # Dedicated MLflow champion run
    # -----------------------------------------------------------------

    with mlflow.start_run(
        run_name="xgboost_structured_champion"
    ):

        mlflow.set_tag(
            "candidate_status",
            "xgboost_champion",
        )

        mlflow.set_tag(
            "stage",
            "validation",
        )

        mlflow.log_param(
            "model_type",
            "xgboost",
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
            "n_train_rows",
            len(train_df),
        )

        mlflow.log_param(
            "n_validation_rows",
            len(val_df),
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        # Log winning hyperparameters cleanly.
        for (
            param_name,
            param_value,
        ) in grid_search.best_params_.items():

            clean_name = (
                param_name
                .replace(
                    "model__",
                    ""
                )
            )

            mlflow.log_param(
                clean_name,
                param_value,
            )

        # CV performance.
        mlflow.log_metric(
            "cv_roc_auc",
            float(
                grid_search.best_score_
            ),
        )

        # Validation performance.
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

        mlflow.log_metric(
            "validation_grouped_precision_at_5",
            metrics[
                "grouped_precision_at_k"
            ][
                "macro_avg_precision_at_k"
            ],
        )

        # Log complete preprocessing + XGBoost pipeline.
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        # Evaluation artifacts.
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
            str(grid_results_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(summary_path),
            artifact_path="evaluation",
        )

    logger.info(
        "Finished XGBoost MLflow tracking."
    )


if __name__ == "__main__":
    main()