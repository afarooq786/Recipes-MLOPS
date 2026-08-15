# Recipe Recommender --- MLOps Pipeline

An open-source recipe recommendation system that predicts whether a
recipe will be positively rated (rating ≥ 4), built as a full MLOps
pipeline: data ingestion → validation → versioning → feature engineering
→ training → experiment tracking → model registry → deployment →
monitoring.

This README documents the full pipeline currently implemented in the
repository: data ingestion, validation, splitting, versioning,
preprocessing, feature engineering, baseline, evaluation,
model-training, experiment-tracking, model-registry, Airflow
orchestration, containerized deployment (FastAPI + Docker Compose),
and production monitoring (drift simulation, Evidently + rule-based
data quality checks, and a custom `/metrics` endpoint). Remaining work
is a single recorded end-to-end run and the final presentation -- see
the Status section near the bottom.

## Project Structure

``` text
recipe-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── filters.py
│   └── metrics_middleware.py
├── dags/
│   ├── data_pipeline_dag.py
│   └── ml_pipeline_dag.py
├── data/
│   ├── ingest.py
│   ├── validate.py
│   ├── split.py
│   ├── raw/
│   ├── processed/splits/
│   ├── processed/clean/
│   ├── processed/features/
│   ├── validation_report.json
│   └── validation_report.md
├── preprocessing/
│   └── preprocess.py
├── features/
│   └── build_features.py
├── models/
│   ├── baseline.py
│   ├── champion_config.json
│   ├── train_logistic.py
│   ├── train_xgboost.py
│   ├── train_challengers.py
│   ├── ensemble_stability.py
│   ├── train_final_candidates.py
│   └── register_best_model.py
├── evaluation/
│   ├── metrics.py
│   └── results/
├── monitoring/
│   ├── drift_simulation.py
│   ├── evidently_report.py
│   ├── data_quality_checks.py
│   ├── verify_alerts.py
│   ├── production_validation.py
│   ├── drift_data/         (generated -- gitignored)
│   └── reports/            (generated -- gitignored)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_build_features.py
│   ├── test_metrics.py
│   ├── test_validate.py
│   ├── test_filters.py
│   ├── test_schemas.py
│   ├── test_api_docker.py
│   ├── test_api_integration.py
│   └── test_metrics_endpoint.py
├── Dockerfile.api
├── Dockerfile.airflow
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── .dvc/config
├── requirements.txt
├── requirements-dev.txt
├── requirements-ci.txt
└── README.md
```

## Setup

1.  Clone the repo and create a virtual environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2.  Kaggle credentials are only needed to run `data/ingest.py` directly.
    They are not needed if pulling already-versioned data with DVC.

3.  GCP access is needed for `dvc pull` / `dvc push`.

Ask the project owner for access to the shared GCP project and DVC
remote, then authenticate with the Google Cloud CLI.

See `docs/Recipe_MLOps_Data_Infrastructure_Handoff.pdf` for the full
walkthrough if present in your checkout.

## Reproducing the Data Pipeline Locally

Once setup is complete, run:

``` bash
dvc repro
```

The data pipeline performs:

  --------------------------------------------------------------------------------
  Step              Stage             Script               What it does
  ----------------- ----------------- -------------------- -----------------------
  1                 Ingestion         `data/ingest.py`     Downloads the source
                                                           recipe data.

  2                 Validation        `data/validate.py`   Validates required
                                                           schema, data types,
                                                           null rules, and allowed
                                                           ranges and writes
                                                           validation reports.

  3                 Split             `data/split.py`      Deduplicates recipes,
                                                           creates the
                                                           `rating >= 4` target,
                                                           and creates seeded
                                                           stratified
                                                           train/validation/test
                                                           splits.

  4                 Versioning        `dvc.yaml` /         Tracks pipeline
                                      `dvc.lock`           dependencies and
                                                           generated data by hash
                                                           for reproducibility.
  --------------------------------------------------------------------------------

To pull the exact versioned data:

``` bash
dvc pull
```

To push a newly generated data version:

``` bash
dvc push
```

## Preprocessing, Features, Baseline & Evaluation

After the split data is available:

``` bash
python preprocessing/preprocess.py
python features/build_features.py
python models/baseline.py --eval-split val
```

  -------------------------------------------------------------------------------------
  Step              Stage             Script                          What it does
  ----------------- ----------------- ------------------------------- -----------------
  5                 Preprocessing     `preprocessing/preprocess.py`   Normalizes text,
                                                                      parses
                                                                      ingredients,
                                                                      converts cooking
                                                                      times, removes
                                                                      unusable rows,
                                                                      and performs
                                                                      residual
                                                                      deduplication.

  6                 Feature           `features/build_features.py`    Creates
                    Engineering                                       ingredient,
                                                                      cooking-time,
                                                                      dietary,
                                                                      nutrition, and
                                                                      related
                                                                      engineered
                                                                      features.

  7                 Baseline          `models/baseline.py`            Provides the
                                                                      required non-ML
                                                                      popularity
                                                                      baseline using
                                                                      train-set cuisine
                                                                      ratings.

  8                 Evaluation        `evaluation/metrics.py`         Provides shared
                                                                      ROC-AUC,
                                                                      Precision@5,
                                                                      grouped
                                                                      Precision@5, and
                                                                      plotting
                                                                      utilities.
  -------------------------------------------------------------------------------------

## Model Training and Experimentation

### Target and evaluation strategy

The primary target is:

``` text
label = 1 when rating >= 4
label = 0 otherwise
```

After deduplication, this target is highly imbalanced: roughly 94--95%
of recipes are positive. For that reason, accuracy is not used as the
primary model-selection metric. ROC-AUC is the primary discrimination
metric, with Precision@5 and grouped Precision@5 retained as
ranking-oriented secondary metrics.

The project maintains three separate data roles:

-   Training split: model fitting, hyperparameter search, and repeated
    cross-validation.
-   Validation split: held-out comparison after training-side model
    selection.
-   Test split: intentionally isolated from model selection and tuning
    for later production validation.

The isolated test set was not used during the modeling experiments
described below.

### Structured Logistic Regression

Run:

``` bash
python -m models.train_logistic
```

`models/train_logistic.py` trains structured Logistic Regression
candidates using engineered numeric and categorical features. It
compares regularization strengths and class-weighting options, evaluates
candidates consistently through `evaluation.metrics`, writes local
artifacts, and logs training runs to MLflow.

This model provides a simple and interpretable ML benchmark above the
non-ML popularity baseline.

### XGBoost

Run:

``` bash
python -m models.train_xgboost
```

`models/train_xgboost.py` trains and tunes an XGBoost classifier over
the engineered structured features using stratified cross-validation.
The search covers tree depth, number of estimators, learning rate,
minimum child weight, row subsampling, and column subsampling.

In experimentation, structured XGBoost did not provide a meaningful
improvement over chance-level ranking performance, which motivated
testing whether the raw ingredient text contained stronger predictive
signal than the manually engineered structured features.

### Text challengers

Run:

``` bash
python -m models.train_challengers
```

`models/train_challengers.py` evaluates serious text-based challengers
using cleaned `ingredients_parsed` text. The implemented model families
include:

-   Character-level TF-IDF + Logistic Regression
-   Word-level TF-IDF + Logistic Regression
-   Character-level TF-IDF + Linear SVM
-   Word-level TF-IDF + Linear SVM
-   Blends/rank ensembles of complementary text models

Hyperparameters are tuned on training data, followed by repeated
stratified cross-validation. The validation set is evaluated only after
the training-side comparison.

The experiments showed that ingredient text carries substantially more
useful signal than the structured feature set.

### Ensemble stability

Run:

``` bash
python -m models.ensemble_stability
```

`models/ensemble_stability.py` evaluates the strongest individual text
models and ensemble combinations on the exact same 10 × 5
RepeatedStratifiedKFold splits.

This is important because the dataset is small and contains very few
negative examples. A single validation split can therefore produce a
deceptively high or low ROC-AUC. Repeated CV gives a more credible
estimate of whether an apparent improvement is persistent.

The strongest repeated-CV candidate was a rank ensemble combining
character TF-IDF Logistic Regression and word TF-IDF Linear SVM:

``` text
Mean ROC-AUC:   0.6810
Std ROC-AUC:    0.0897
Median ROC-AUC: 0.6688
Folds:          50
```

This materially exceeds random ranking (ROC-AUC = 0.50), while the
fold-to-fold variation also documents the uncertainty caused by the
small and highly imbalanced dataset.

### Final candidates

Run:

``` bash
python -m models.train_final_candidates
```

`models/train_final_candidates.py` packages the final serious candidates
into a reproducible comparison and logs them through MLflow.

The final comparison produced:

``` text
Char Logistic | repeated-CV ROC-AUC ≈ 0.6277 | validation ROC-AUC ≈ 0.6976
Word SVM      | repeated-CV ROC-AUC ≈ 0.6477 | validation ROC-AUC ≈ 0.6682
Ensemble      | repeated-CV ROC-AUC ≈ 0.6810 | validation ROC-AUC ≈ 0.6903
```

The ensemble is preferred because model selection is based primarily on
the more robust repeated-CV evidence rather than choosing whichever
candidate happened to score highest on one validation split.

## Additional Modeling Findings

Several additional experiments were performed during model development
to understand the problem before settling on the reproducible final
candidates.

Key findings included:

-   Structured-only Logistic Regression and XGBoost contained relatively
    weak predictive signal.
-   TF-IDF representations of ingredient text improved discrimination
    substantially.
-   Character n-grams were particularly effective for Logistic
    Regression.
-   A dedicated experiment changing the positive target from
    `rating >= 4` to `rating >= 4.5` created a more balanced target but
    did not improve predictive performance enough to justify changing
    the agreed problem definition.
-   Repeated CV was preferred over trusting a single small validation
    split.
-   Ensembles were accepted only when they improved the training-side
    repeated-CV evidence rather than merely producing a lucky validation
    result.

Exploratory scripts and intermediate experiment outputs may be kept
outside the committed production path. The committed modeling scripts
represent the reproducible path needed by the team.

## MLflow Experiment Tracking

MLflow is included in `requirements.txt` and is used to track
model-training experiments.

Start a local MLflow server from the repository environment:

``` bash
mlflow server --host 127.0.0.1 --port 5000
```

Then, in a separate terminal with the same environment active, run the
training scripts.

The current training code uses:

``` text
Tracking URI: http://127.0.0.1:5000
Experiment:   recipe-recommender-training
```

MLflow records model configuration, metrics, and artifacts so the
training process is auditable instead of relying on manually copied
terminal output.

Because this MLflow server is local, the MLflow database itself is not
automatically shared by Git. What is shared through the repository is
the code required to reproduce the experiment runs. A teammate can
clone/pull the repository, install the dependencies, obtain the
versioned data, start MLflow locally, and rerun the scripts to generate
equivalent tracked experiments.

## Model Registry

The selected model is registered in MLflow under:

``` text
recipe-recommender
```

The current selected version is assigned the:

``` text
champion
```

alias.

Run:

``` bash
python -m models.register_best_model
```

`models/register_best_model.py` registers the finalized model in the
MLflow Model Registry and assigns the champion alias. Downstream
deployment code should load the model through the registry alias rather
than hard-coding a local training-run path.

This gives the project a clean handoff between experimentation and
deployment:

``` text
training → MLflow experiment → final candidate → registered model → champion alias → deployment
```

When a genuinely better model is selected later, a new registered
version can be created and the `champion` alias moved to that version
without changing the deployment interface.

## Reproducing the Modeling Workflow

A teammate reproducing the current modeling work should:

``` bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Obtain the versioned data
dvc pull

# 3. Generate preprocessing/features if needed
python preprocessing/preprocess.py
python features/build_features.py

# 4. Start MLflow in a separate terminal
mlflow server --host 127.0.0.1 --port 5000

# 5. Train tracked structured benchmarks
python -m models.train_logistic
python -m models.train_xgboost

# 6. Run final text-model comparison/stability work
python -m models.train_challengers
python -m models.ensemble_stability
python -m models.train_final_candidates

# 7. Register the selected model
python -m models.register_best_model
```

The test split should remain untouched during these steps.

### Champion Benchmark and Promotion

Because the project currently uses a local MLflow server, each developer's MLflow
tracking database and Model Registry exist only on their own machine. The registered
`champion` alias therefore is not automatically shared when another team member clones
the GitHub repository.

To make the selected champion reproducible across environments, the repository includes:

`models/champion_config.json`

This file is the Git-tracked source of truth for the currently approved model and its
benchmark performance. The current champion is the character-TF-IDF Logistic Regression
+ word-TF-IDF Linear SVM rank ensemble, selected using 5-fold cross-validation repeated
10 times (50 held-out folds).

Current champion benchmark:

- Mean repeated-CV ROC-AUC: 0.6810
- Median repeated-CV ROC-AUC: 0.6688
- Validation ROC-AUC: approximately 0.690
- Test set used for model selection: No

The manifest also records the component model hyperparameters and ensemble configuration,
allowing the champion to be reconstructed from the versioned training data.

New training runs should be treated as challengers rather than automatically replacing
the champion. A challenger should only be considered for promotion if it improves on the
current champion under the same evaluation methodology and passes the project's validation
checks. If promoted, the new model should be registered as a new MLflow model version, the
`champion` alias should be moved to that version, and `champion_config.json` should be
updated to record the new approved benchmark.

This design allows the team to use local MLflow instances during development while still
maintaining a shared, version-controlled definition of the currently approved model.

## Airflow Orchestration

### How it was done

Two Airflow DAGs were created in `dags/` to orchestrate the full
pipeline end-to-end. Each DAG calls the project's existing Python
scripts through `BashOperator` tasks, so no existing code was
modified. A `PythonOperator` handles the conditional model promotion
gate, which is the only new logic.

Two separate DAGs were chosen so the data pipeline and ML pipeline
can be triggered independently.

### DAG 1: `recipe_data_pipeline`

Handles data ingestion through splitting.

| # | Task | Operator | What it runs |
|---|------|----------|--------------|
| 1 | `download_data` | BashOperator | `python data/ingest.py` — downloads raw data from Kaggle |
| 2 | `validate_schema` | BashOperator | `python data/validate.py` — validates schemas with Pandera |
| 3 | `version_files` | BashOperator | `dvc add data/raw && dvc push` — versions raw files in DVC |
| 4 | `create_splits` | BashOperator | `python data/split.py` — creates train/val/test splits |

``` text
download_data → validate_schema → version_files → create_splits
```

### DAG 2: `recipe_ml_pipeline`

Handles preprocessing through model promotion.

| # | Task | Operator | What it runs |
|---|------|----------|--------------|
| 1 | `preprocess` | BashOperator | `python preprocessing/preprocess.py` — cleans and normalizes data |
| 2 | `feature_engineering` | BashOperator | `python features/build_features.py` — builds model features |
| 3 | `train_candidates` | BashOperator | `python -m models.train_final_candidates` — trains and evaluates candidates, logs to MLflow |
| 4 | `evaluate_and_log` | BashOperator | `python -m models.register_best_model` — registers best model in MLflow |
| 5 | `promote_model` | PythonOperator | Conditional promotion gate (see below) |

``` text
preprocess → feature_engineering → train_candidates → evaluate_and_log → promote_model
```

### Model promotion gate

The `promote_model` task implements a conditional promotion check
that prevents regressions from being deployed:

1. Reads the current champion benchmark from `models/champion_config.json`.
2. Finds the latest finished `ensemble_text_champion` run in MLflow.
3. Compares the candidate's `validation_roc_auc` against the champion's
   `selection_metric_value`.
4. **Promotes only if the candidate improves the metric.**
5. On promotion: registers a new MLflow model version, moves the `champion`
   alias, and updates `champion_config.json`.
6. On skip: logs the decision and exits cleanly — no changes are made.

### Running the Airflow DAGs

**Prerequisites:** Make sure MLflow is running before triggering the ML
pipeline (`mlflow server --host 127.0.0.1 --port 5000`).

**Step 1 — Install Airflow:**

``` bash
pip install "apache-airflow>=2.9"
```

**Step 2 — Set environment variables:**

``` bash
# Tell Airflow where its home directory is
export AIRFLOW_HOME=~/airflow

# Tell the DAGs where the project repo lives
# (defaults to the parent of dags/ if not set)
export RECIPE_PROJECT_DIR=$(pwd)
```

**Step 3 — Initialize the Airflow database:**

``` bash
airflow db migrate
```

**Step 4 — Make the DAGs visible to Airflow:**

Symlink the `dags/` directory into Airflow's expected location:

``` bash
ln -s "$RECIPE_PROJECT_DIR/dags" "$AIRFLOW_HOME/dags"
```

Alternatively, edit `$AIRFLOW_HOME/airflow.cfg` and set
`dags_folder` to point directly to the repository's `dags/`
directory.

**Step 5 — Verify both DAGs are discovered:**

``` bash
airflow dags list
```

You should see `recipe_data_pipeline` and `recipe_ml_pipeline`.

**Step 6 — Start the Airflow scheduler and webserver:**

``` bash
airflow scheduler &
airflow webserver --port 8080 &
```

Open `http://localhost:8080` to access the Airflow web UI.

**Step 7 — Trigger the pipelines:**

``` bash
# Run the data pipeline first
airflow dags trigger recipe_data_pipeline

# After data is ready, run the ML pipeline
airflow dags trigger recipe_ml_pipeline
```

You can also trigger DAGs from the web UI by clicking the play
button on each DAG's page.

## Monitoring, Drift Simulation, and Production Validation

The repository implements the full monitoring stack described in the
project guidelines, using a lightweight, custom-built layer in place
of a full Prometheus/Grafana deployment. All scripts below are
reproducible and were verified against the actual isolated test split
(`data/processed/clean/test.csv`) before being committed.

### Layer 1: Data quality and system metrics (row 22)

`api/metrics_middleware.py` records every request's latency, path, and
status code in-process. `GET /metrics` on the running API returns the
same signals a Prometheus + Grafana dashboard would surface (request
volume, error rate, p50/p95 latency), without the operational overhead
of standing up a separate metrics stack for a class project timeline.

```bash
curl http://localhost:8000/metrics
```

### Layer 2: Drift simulation (row 23)

`monitoring/drift_simulation.py` builds a clean, API-request-shaped
batch from the isolated test split, plus four corrupted variants
simulating distinct real-world failure modes:

| Variant | What's corrupted |
| --- | --- |
| `drift_missing_values.csv` | `cook_time_minutes` / `calories` nulled out for a subset of rows |
| `drift_out_of_range.csv` | Negative cook times; calorie values in the tens of thousands |
| `drift_schema_swap.csv` | `cook_time_minutes` and `calories` column *values* swapped |
| `drift_schema_change.csv` | A column renamed, another dropped entirely |

```bash
python -m monitoring.drift_simulation
```

Outputs land in `monitoring/drift_data/`, along with a
`drift_manifest.json` recording exactly which rows/columns were
touched by each corruption -- the ground truth used to check whether
monitoring actually caught each one.

### Layer 3: Evidently statistical drift + rule-based quality checks (row 21)

Two complementary checks run against every batch:

- `monitoring/evidently_report.py` -- an Evidently `DataDriftPreset`
  report comparing the current batch's distribution against the
  reference (clean) batch. Good at catching gradual, distribution-level
  shift (e.g. the column-swap scenario).
- `monitoring/data_quality_checks.py` -- rule-based schema-contract,
  physically-plausible-range, and null-rate-spike checks. Good at
  catching row-level anomalies and structural breaks that a pure
  distribution test can miss (missing values, out-of-range values,
  renamed/dropped columns).

```bash
python -m monitoring.evidently_report \
    --reference monitoring/drift_data/clean_candidates.csv \
    --current monitoring/drift_data/drift_out_of_range.csv \
    --name out_of_range

python -m monitoring.data_quality_checks \
    --current monitoring/drift_data/drift_out_of_range.csv \
    --name out_of_range
```

### Layer 4: Anomaly verification (row 24)

`monitoring/verify_alerts.py` runs both layers across the clean
baseline and all four corrupted variants in one pass and writes a
consolidated pass/fail evidence report to
`monitoring/reports/verify_alerts_summary.json`, plus HTML drift
reports per scenario for the presentation demo.

```bash
python -m monitoring.verify_alerts
```

Verified result: **5/5 scenarios correctly classified** (the clean
baseline correctly shows no anomaly; all four corrupted variants are
correctly flagged), with zero false positives.

### Production baseline validation (row 20)

`monitoring/production_validation.py` sends the clean candidate batch
through the *deployed* `/predict` endpoint (not just the model object
in isolation) and compares the live API's ROC-AUC against the offline
validation ROC-AUC recorded in `models/champion_config.json`.

```bash
# 1. Start MLflow on the host, then the rest of the stack
#    (see "Quick Start with Docker Compose" below)
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --allowed-hosts '*' --cors-allowed-origins '*'   # separate terminal, leave running
docker compose up -d --build
curl http://localhost:8000/health   # confirm model_loaded: true

# 2. Build the clean validation batch if not already built
python -m monitoring.drift_simulation

# 3. Run production validation against the live API
python -m monitoring.production_validation --api-url http://localhost:8000
```

---

## Testing

Run the full test suite:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

- `tests/test_preprocessing.py`, `tests/test_build_features.py`,
  `tests/test_metrics.py`, `tests/test_validate.py` -- unit tests for
  the core data pipeline functions and the Pandera validation schema.
- `tests/test_filters.py`, `tests/test_schemas.py`,
  `tests/test_api_docker.py` -- unit tests for the API's filtering
  logic and Pydantic request/response contracts.
- `tests/test_api_integration.py` -- integration tests exercising the
  full request -> filter -> score -> rank -> response path against a
  fake model, so CI doesn't need a live MLflow server.
- `tests/test_metrics_endpoint.py` -- verifies the `/metrics`
  monitoring endpoint records and summarizes real request traffic.

All 96 tests pass in ~2 seconds on a clean install of
`requirements-ci.txt`.

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **test** -- installs `requirements-ci.txt` and runs the full pytest suite.
2. **lint** -- runs `ruff` (currently non-blocking; a baseline config still needs to be agreed on).
3. **docker-build** -- confirms both `Dockerfile.api` and `Dockerfile.airflow` build successfully.

---

## FastAPI Inference Service & Nutritional Filtering Layer

The repository provides a production-grade inference service in `api/`:

- `api/main.py`: FastAPI application loading the champion model from MLflow (`models:/recipe-recommender@champion`).
- `api/schemas.py`: Pydantic request/response contracts for recipe candidates, dietary tags, filter constraints, and predictions.
- `api/filters.py`: Deterministic pre-inference filter layer processing calorie limits, cooking times, required dietary tags, and excluded ingredients *before* candidate recipes are sent to the model.

### API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Service status, model loaded flag, model name, and alias |
| `/predict` | `POST` | Scores and ranks recipe candidates after applying constraint filters |
| `/reload-model` | `POST` | Hot-reloads the champion model from the MLflow registry |

---

## Containerization and Docker Compose Deployment

The Inference API, Airflow Webserver/Scheduler, and PostgreSQL are containerized and orchestrated via Docker Compose. **MLflow currently runs directly on the host machine, not in a container** -- see the note below.

### ⚠️ MLflow: run on the host, not in Docker

The `mlflow` service in `docker-compose.yml` is present but **disabled by default** (`profiles: ["disabled"]`). On the machine this was last verified on, that exact container/version combo (MLflow 3.15.1) bound only to `127.0.0.1` inside the container despite `--host 0.0.0.0` being set -- confirmed at the kernel socket level, not just from log messages. Root cause wasn't identified; the same command run directly (outside Docker) works correctly. See the comment block above the `mlflow:` service in `docker-compose.yml` for full details.

**Workaround (what to actually run):** start MLflow directly on your host, pointing at your existing local `mlflow.db` / `mlartifacts/`:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --allowed-hosts '*' --cors-allowed-origins '*'
```

Leave that running in its own terminal. The `api`, `airflow-webserver`, and `airflow-scheduler` services are all configured to reach it via `http://host.docker.internal:5000`, Docker's built-in hostname for "the machine Docker is running on" -- no other setup needed on their end.

If/when the containerized MLflow bind issue gets root-caused, remove the `profiles` line from the `mlflow` service and switch the other services' `MLFLOW_TRACKING_URI` back to `http://mlflow:5000`.

### Container Architecture

| Service | Container Name | Port | Description |
| --- | --- | --- | --- |
| **FastAPI Service** | `recipe-api` | `8000` | Inference API built from `Dockerfile.api`. Requires the Python 3.12-based image (see note below) and a running host MLflow. |
| **MLflow Server** | *(runs on host, not containerized)* | `5000` | Tracking server & model registry. See "MLflow: run on the host" above. |
| **Airflow Webserver** | `recipe-airflow-webserver` | `8080` | Airflow Web UI built from `Dockerfile.airflow` |
| **Airflow Scheduler** | `recipe-airflow-scheduler` | — | Background task scheduler running DAGs |
| **Airflow Init** | `recipe-airflow-init` | — | Migration task creating Airflow DB schema & admin user |
| **PostgreSQL** | `recipe-postgres` | `5432` | Backing database for Airflow metadata |

### ⚠️ Python version must match the training environment

`Dockerfile.api` builds on `python:3.12-slim` to match the Python version models are actually trained/pickled under (check with `python --version` in your training environment). If your champion model was trained under a different Python minor version, update the `FROM` line in `Dockerfile.api` to match -- a mismatch here fails at model-load time with an error like `code expected at most N arguments, got M`, which is a Python bytecode/pickle incompatibility across minor versions, not an MLflow or app bug.

### Quick Start with Docker Compose

1. **Start MLflow on your host first** (see "MLflow: run on the host" above) and leave it running in its own terminal.

2. **Build and start the rest of the stack in detached mode** (the disabled `mlflow` service is skipped automatically):

```bash
docker compose up -d --build
```

3. **Verify container health:**

```bash
docker compose ps
```

4. **Access Service Interfaces:**
   - **FastAPI Documentation & Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **MLflow Tracking & Model Registry** (on host): [http://localhost:5000](http://localhost:5000)
   - **Airflow Web UI**: [http://localhost:8080](http://localhost:8080) (Credentials: `admin` / `admin`)

5. **Verify FastAPI Endpoint Health:**

```bash
curl http://localhost:8000/health
```

Expect `"model_loaded": true`. If `false`, check `docker compose logs api` -- the two most common causes are the host MLflow not running/reachable, or the Python-version mismatch described above.

6. **Send a Test Prediction Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "candidates": [
         {
           "recipe_id": "r101",
           "title": "Healthy Summer Salad",
           "ingredients_parsed": "fresh spinach, cherry tomatoes, olive oil, lemon juice",
           "calories": 220,
           "cook_time_minutes": 10,
           "dietary_tags": ["vegetarian", "vegan", "gluten_free"]
         },
         {
           "recipe_id": "r102",
           "title": "High Calorie Bacon Burger",
           "ingredients_parsed": "beef patty, bacon, cheddar cheese, bun",
           "calories": 950,
           "cook_time_minutes": 25
         }
       ],
       "constraints": {
         "max_calories": 500,
         "required_dietary_tags": ["vegetarian"]
       },
       "top_k": 5
     }'
```

6. **Tear down the stack:**

```bash
# Stop containers keeping volumes intact
docker compose down

# Stop containers and wipe persistent data volumes
docker compose down -v
```

### Running Automated API Integration Tests

To run the Pytest integration suite for the API logic:

```bash
python -m pytest tests/test_api_docker.py -v
```

## Known Data Notes

-   `recipes.csv` is the labeled source used for the
    train/validation/test split.
-   The classification problem is recipe-level prediction of whether
    aggregate rating is at least 4, not a personalized per-user
    recommender, because the source data does not contain real
    user-level interaction history.
-   Duplicate raw recipes are removed before splitting to prevent
    identical recipes from leaking across train, validation, and test
    sets.
-   The target is extremely imbalanced, with roughly 94--95% positive
    examples after deduplication.
-   Accuracy is therefore misleading as a primary metric.
-   The small number of negative examples makes ROC-AUC estimates noisy,
    which is why repeated stratified CV is used for final model
    comparison.
-   `test_recipes.csv` is a separate differently-shaped
    unlabeled/holdout-style dataset and is intended for later deployment
    validation and drift simulation rather than model tuning.
-   Any synthetic user-interaction features produced by feature
    engineering are placeholders and should not be represented as real
    user behavior.

## What's Next

Remaining work is verification on your own machine and presentation
assembly -- see "Final Steps" below.

-   [x] Run `docker compose up -d --build` (with MLflow started on the host first) and confirm the full stack comes up healthy -- **done**, see the Docker Compose section above.
-   Run `python -m monitoring.production_validation` against the live API.
-   Run the end-to-end pipeline (ingestion -> ... -> monitoring) once, recorded.
-   Capture MLflow, registry, API, container, and monitoring evidence
    for the final presentation.

## Status

Completed:

-   [x] Dataset ingestion
-   [x] Schema/data validation
-   [x] Reproducible train/validation/test split
-   [x] DVC data/pipeline versioning foundation
-   [x] Preprocessing
-   [x] Feature engineering
-   [x] Non-ML popularity baseline
-   [x] Shared ROC-AUC / Precision@k evaluation utilities
-   [x] Structured Logistic Regression training
-   [x] XGBoost tuning
-   [x] Text-feature model experimentation
-   [x] Character and word TF-IDF challengers
-   [x] Logistic Regression and Linear SVM challengers
-   [x] Repeated stratified cross-validation stability analysis
-   [x] Ensemble comparison
-   [x] Final reproducible candidate comparison
-   [x] MLflow experiment tracking
-   [x] MLflow model logging
-   [x] MLflow Model Registry integration
-   [x] `recipe-recommender` registered model
-   [x] `champion` model alias

Remaining / downstream:

-   [x] Full workflow orchestration (Airflow DAGs)
-   [x] Containerized inference API (FastAPI)
-   [x] Docker Compose deployment stack (FastAPI + MLflow + Airflow + Postgres)
-   [x] Production validation script (`monitoring/production_validation.py` -- ready to run once the stack is up)
-   [x] Monitoring dashboard/framework (Evidently + custom rule-based checks + custom `/metrics` endpoint)
-   [x] Drift/stress-test simulation (`monitoring/drift_simulation.py`, 4 corruption types)
-   [x] Anomaly verification (`monitoring/verify_alerts.py` -- 5/5 scenarios correctly classified, 0 false positives)
-   [x] Unit test suite (96 tests, `tests/`)
-   [x] Integration test suite (`tests/test_api_integration.py`)
-   [x] CI/CD (`.github/workflows/ci.yml`)
-   [ ] Live production validation run against the deployed Docker stack (script is ready; needs to be run locally/CI where Docker is available)
-   [ ] Final presentation/demo artifacts

## Modeling Summary

The modeling results illustrate why the MLOps workflow matters. The
dataset is small and highly imbalanced, so individual validation results
can vary considerably. Structured recipe metadata provided limited
discrimination, while ingredient text produced substantially stronger
signal.

Rather than selecting the highest single validation score, the final
candidate was chosen using repeated stratified cross-validation across
50 held-out folds. The character-Logistic + word-SVM ensemble achieved
an average ROC-AUC of approximately 0.681, compared with 0.50 for random
ranking, and approximately 0.690 ROC-AUC on the held-out validation
split.

The resulting workflow provides reproducible training, explicit
experiment comparison, tracked metrics/artifacts, and a registered
champion model ready for the deployment stage.
