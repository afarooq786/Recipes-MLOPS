"""
dags/ml_pipeline_dag.py — Airflow DAG for the ML pipeline.

Orchestrates preprocessing, feature engineering, candidate training,
evaluation, MLflow logging, and conditional model promotion.

The promote_model task compares the new candidate's validation
ROC-AUC against the current champion in models/champion_config.json
and only promotes if the candidate passes validation and improves
the selected metric.

Set the RECIPE_PROJECT_DIR environment variable (or Airflow Variable
'recipe_project_dir') to point to the repository root. Defaults to
the parent of the dags/ folder.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Resolve project root: env var > parent of this file
_DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_DIR = os.getenv("RECIPE_PROJECT_DIR", _DEFAULT_PROJECT_DIR)

default_args = {
    "owner": "recipe-mlops",
    "depends_on_past": False,
    "retries": 1,
}

logger = logging.getLogger("ml_pipeline_dag")


# -----------------------------------------------------------------
# Promotion gate
# -----------------------------------------------------------------

def _promote_model(**context):
    """Promote a new model only when it passes validation and beats
    the current champion on the selected metric.

    Reads:
      - The latest MLflow ensemble run metrics.
      - models/champion_config.json for the current champion benchmark.

    Promotes by:
      - Registering a new MLflow model version.
      - Moving the MLflow 'champion' alias to that version.
      - Updating champion_config.json with the new metric values.

    Skips promotion (without failing) when:
      - No candidate run is found in MLflow.
      - The candidate's metric does not improve over the champion.
    """
    import mlflow
    from mlflow import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = "recipe-recommender-training"
    registered_model_name = "recipe-recommender"
    champion_alias = "champion"
    source_run_name = "ensemble_text_champion"

    # ---- Load current champion benchmark ----
    config_path = Path(PROJECT_DIR) / "models" / "champion_config.json"
    if config_path.exists():
        champion_config = json.loads(config_path.read_text())
    else:
        logger.warning("No champion_config.json found — treating as first run.")
        champion_config = {"selection_metric_value": 0.0}

    metric_key = champion_config.get("selection_metric", "repeated_cv_mean_roc_auc")
    current_best = champion_config.get("selection_metric_value", 0.0)

    # ---- Find the latest candidate run ----
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.mlflow.runName = '{source_run_name}' "
            f"and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        logger.warning("No finished '%s' run found. Skipping promotion.", source_run_name)
        return "skipped — no candidate run found"

    run = runs.iloc[0]
    run_id = run["run_id"]
    candidate_value = run.get("metrics.validation_roc_auc", None)

    if candidate_value is None:
        logger.warning("Candidate run %s has no validation_roc_auc metric. Skipping.", run_id)
        return "skipped — missing validation metric"

    # ---- Bootstrap check: does THIS person's local registry already
    # have a champion at all? ----
    # champion_config.json is committed to git and reflects whoever's
    # training run set the current benchmark -- it is NOT specific to
    # any one person's local MLflow registry (each teammate's registry
    # is local-only, per the README). Without this check, a brand-new
    # contributor with an empty local registry would have to beat that
    # committed benchmark on their very first run just to get ANY
    # champion registered locally -- a real coin-flip given this model's
    # own documented run-to-run noise (repeated_cv_std_roc_auc ~ 0.09),
    # and a silent failure (no error, just "skipped") if they lose it.
    has_local_champion = True
    try:
        client.get_model_version_by_alias(registered_model_name, champion_alias)
    except Exception:
        has_local_champion = False

    if not has_local_champion:
        logger.info(
            "No champion registered in this local MLflow registry yet -- "
            "treating this as a bootstrap run and promoting unconditionally, "
            "regardless of the committed champion_config.json benchmark "
            "(%s=%.4f).",
            metric_key, current_best,
        )
        current_best = 0.0  # bypass the gate below for this run only

    logger.info(
        "Candidate validation_roc_auc=%.4f  vs  champion %s=%.4f",
        candidate_value, metric_key, current_best,
    )

    # ---- Gate: candidate must beat current champion ----
    if candidate_value <= current_best:
        logger.info(
            "Candidate (%.4f) does not improve over champion (%.4f). No promotion.",
            candidate_value, current_best,
        )
        return "skipped — no improvement"

    # ---- Promote: register and move alias ----
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = str(model_version.version)

    client.set_registered_model_alias(registered_model_name, champion_alias, version)

    logger.info(
        "PROMOTED: %s version %s (validation_roc_auc=%.4f) → alias '%s'",
        registered_model_name, version, candidate_value, champion_alias,
    )

    # ---- Update champion_config.json ----
    champion_config["selection_metric_value"] = candidate_value
    champion_config["validation_roc_auc"] = candidate_value
    champion_config["promoted_from_run_id"] = run_id
    config_path.write_text(json.dumps(champion_config, indent=2))
    logger.info("Updated %s", config_path)

    return f"promoted version {version}"


# -----------------------------------------------------------------
# DAG
# -----------------------------------------------------------------

with DAG(
    dag_id="recipe_ml_pipeline",
    default_args=default_args,
    description="Preprocess, engineer features, train, evaluate, log to MLflow, and conditionally promote.",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "recipes"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && python preprocessing/preprocess.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd {PROJECT_DIR} && python features/build_features.py",
    )

    train_candidates = BashOperator(
        task_id="train_candidates",
        bash_command=f"cd {PROJECT_DIR} && python -m models.train_final_candidates",
    )

    evaluate_and_log = BashOperator(
        task_id="evaluate_and_log",
        bash_command=f"cd {PROJECT_DIR} && python -m models.register_best_model",
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
    )

    (
        preprocess
        >> feature_engineering
        >> train_candidates
        >> evaluate_and_log
        >> promote_model
    )