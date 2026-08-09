# Recipe Recommender --- MLOps Pipeline

An open-source recipe recommendation system that predicts whether a
recipe will be positively rated (rating ≥ 4), built as a full MLOps
pipeline: data ingestion → validation → versioning → feature engineering
→ training → experiment tracking → model registry → deployment →
monitoring.

This README documents the data ingestion, validation, splitting,
versioning, preprocessing, feature engineering, baseline, evaluation,
model-training, experiment-tracking, and model-registry work currently
implemented in the repository. Deployment, orchestration, and production
monitoring remain downstream project stages.

## Project Structure

``` text
recipe-mlops/
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
│   ├── train_logistic.py
│   ├── train_xgboost.py
│   ├── train_challengers.py
│   ├── ensemble_stability.py
│   ├── train_final_candidates.py
│   └── register_best_model.py
├── evaluation/
│   ├── metrics.py
│   └── results/
├── dvc.yaml
├── dvc.lock
├── .dvc/config
├── requirements.txt
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

The remaining project work builds on the data and registered model:

-   Wire remaining preprocessing/training steps into the automated
    orchestrator/DVC workflow as needed.
-   Complete the Airflow or equivalent orchestration layer.
-   Package the registered champion model behind a FastAPI/Flask/BentoML
    inference service.
-   Containerize the service with Docker.
-   Pass clean test inputs through the deployed service for baseline
    production validation.
-   Implement monitoring with EvidentlyAI, Prometheus/Grafana, or the
    team's selected framework.
-   Create corrupted/drifted test scenarios.
-   Verify that monitoring detects the simulated anomaly/drift.
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

-   [ ] Full workflow orchestration
-   [ ] Containerized inference API
-   [ ] Production baseline validation
-   [ ] Monitoring dashboard/framework
-   [ ] Drift/stress-test simulation
-   [ ] Anomaly verification
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
