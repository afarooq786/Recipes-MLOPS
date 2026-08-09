#!/usr/bin/env python3
"""
models/register_best_model.py — Step 12: Model Registry

Finds the latest successful ensemble_text_champion MLflow run,
registers its logged model in the MLflow Model Registry,
adds model/version metadata, and assigns the `champion` alias.

The registered model can later be loaded with:

    models:/recipe-recommender@champion

A future improved model can be registered as a new version and the
`champion` alias can be moved without changing serving code.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("register_best_model")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
import os 

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT = "recipe-recommender-training"

REGISTERED_MODEL_NAME = "recipe-recommender"

SOURCE_RUN_NAME = "ensemble_text_champion"

SEMANTIC_VERSION = "1.0.0"

CHAMPION_ALIAS = "champion"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # -------------------------------------------------------------
    # Find experiment
    # -------------------------------------------------------------

    experiment = (
        mlflow.get_experiment_by_name(
            MLFLOW_EXPERIMENT
        )
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment "
            f"'{MLFLOW_EXPERIMENT}' "
            f"was not found."
        )

    logger.info(
        "Found experiment: %s",
        MLFLOW_EXPERIMENT,
    )

    # -------------------------------------------------------------
    # Find latest successful ensemble champion run
    # -------------------------------------------------------------

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            f"tags.mlflow.runName = "
            f"'{SOURCE_RUN_NAME}' "
            f"and attributes.status = "
            f"'FINISHED'"
        ),
        order_by=[
            "attributes.start_time DESC"
        ],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            f"No successful MLflow run named "
            f"'{SOURCE_RUN_NAME}' was found."
        )

    run_id = runs.iloc[0]["run_id"]

    logger.info(
        "Found champion run: %s",
        run_id,
    )

    # The ensemble was logged under name="model".
    model_uri = (
        f"runs:/{run_id}/model"
    )

    # -------------------------------------------------------------
    # Register model
    # -------------------------------------------------------------

    logger.info(
        "Registering model from %s",
        model_uri,
    )

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = str(
        model_version.version
    )

    logger.info(
        "Registered %s version %s",
        REGISTERED_MODEL_NAME,
        version,
    )

    # -------------------------------------------------------------
    # Registered-model metadata
    # -------------------------------------------------------------

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "task",
        "recipe_recommendation_ranking",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "target",
        "rating>=4",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "owner",
        "Recipes-MLOPS",
    )

    # -------------------------------------------------------------
    # Version-specific metadata
    # -------------------------------------------------------------

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "semantic_version",
        SEMANTIC_VERSION,
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "validation_status",
        "approved",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "model_type",
        "char_logistic_plus_word_svm_ensemble",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "feature_set",
        "ingredient_text",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "cv_protocol",
        "RepeatedStratifiedKFold_5x10",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "repeated_cv_mean_roc_auc",
        "0.681039",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "repeated_cv_std_roc_auc",
        "0.089698",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "validation_roc_auc",
        "0.6903",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "test_set_used_for_selection",
        "false",
    )

    # -------------------------------------------------------------
    # Champion alias
    # -------------------------------------------------------------

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        CHAMPION_ALIAS,
        version,
    )

    logger.info(
        "Assigned alias '%s' to "
        "%s version %s",
        CHAMPION_ALIAS,
        REGISTERED_MODEL_NAME,
        version,
    )

    # -------------------------------------------------------------
    # Verify
    # -------------------------------------------------------------

    champion = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
        )
    )

    logger.info(
        "Verified champion alias -> "
        "version %s",
        champion.version,
    )

    print("\n" + "=" * 70)
    print("MODEL REGISTRATION COMPLETE")
    print("=" * 70)

    print(
        f"Registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"MLflow registry version: "
        f"{version}"
    )

    print(
        f"Semantic version: "
        f"{SEMANTIC_VERSION}"
    )

    print(
        f"Alias: "
        f"{CHAMPION_ALIAS}"
    )

    print(
        "Serving URI: "
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )

    print(
        f"Source run: "
        f"{SOURCE_RUN_NAME}"
    )

    print(
        "Test set used for model "
        "selection: NO"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()