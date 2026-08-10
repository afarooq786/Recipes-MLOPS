"""
dags/data_pipeline_dag.py — Airflow DAG for the data pipeline.

Orchestrates data ingestion, schema validation, DVC versioning,
and train/val/test splitting as four sequential Airflow tasks.
Each task calls the existing project script via BashOperator.

Set the RECIPE_PROJECT_DIR environment variable (or Airflow Variable
'recipe_project_dir') to point to the repository root. Defaults to
the parent of the dags/ folder.
"""

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

# Resolve project root: env var > Airflow Variable > parent of this file
_DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_DIR = os.getenv("RECIPE_PROJECT_DIR", _DEFAULT_PROJECT_DIR)

default_args = {
    "owner": "recipe-mlops",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="recipe_data_pipeline",
    default_args=default_args,
    description="Download, validate, version, and split recipe data.",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["data", "recipes"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command=f"cd {PROJECT_DIR} && python data/ingest.py",
    )

    validate_schema = BashOperator(
        task_id="validate_schema",
        bash_command=f"cd {PROJECT_DIR} && python data/validate.py",
    )

    version_files = BashOperator(
        task_id="version_files",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "dvc add data/raw && "
            "dvc push"
        ),
    )

    create_splits = BashOperator(
        task_id="create_splits",
        bash_command=f"cd {PROJECT_DIR} && python data/split.py",
    )

    download_data >> validate_schema >> version_files >> create_splits
