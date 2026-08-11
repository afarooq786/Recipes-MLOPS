"""
FastAPI inference service for the recipe recommender.

Loads the registered champion model from the MLflow Model Registry
(model name `recipe-recommender`, alias `champion` -- see the README's
"Model Registry" section and `models/champion_config.json`) and serves
scored/ranked recipe predictions, with an optional nutritional
filtering step applied first (api/filters.py).

Run locally:
    # 1. Make sure MLflow is running and the champion alias is set
    mlflow server --host 127.0.0.1 --port 5000

    # 2. Start the API
    uvicorn api.main:app --reload --port 8000

Environment variables:
    MLFLOW_TRACKING_URI   default: http://127.0.0.1:5000
    MODEL_NAME             default: recipe-recommender
    MODEL_ALIAS             default: champion
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.filters import apply_filters
from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RecipeScore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recipe_api")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "recipe-recommender")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

_model_state: dict = {"model": None}


def _load_champion_model():
    """Load the current champion model from the MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _model_state["model"] = _load_champion_model()
    except Exception as exc:  # noqa: BLE001
        # Don't crash the process on startup -- surface the failure via
        # /health and /predict instead, so the container can still come
        # up (useful behind an orchestrator that retries/backoffs).
        logger.error("Failed to load champion model at startup: %s", exc)
        _model_state["model"] = None
    yield
    _model_state["model"] = None


app = FastAPI(
    title="Recipe Recommender Inference API",
    description="Scores and ranks candidate recipes using the registered champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model():
    model = _model_state.get("model")
    if model is None:
        try:
            model = _load_champion_model()
            _model_state["model"] = model
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"Champion model is unavailable: {exc}",
            ) from exc
    return model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model_state.get("model") is not None else "degraded",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_loaded=_model_state.get("model") is not None,
    )


@app.post("/reload-model", response_model=HealthResponse)
def reload_model() -> HealthResponse:
    """Force a re-fetch of the champion model, e.g. after a promotion event."""
    _model_state["model"] = _load_champion_model()
    return health()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = _get_model()

    surviving, filtered_out_count = apply_filters(request.candidates, request.constraints)

    if not surviving:
        return PredictResponse(
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
            results=[],
            filtered_out_count=filtered_out_count,
            total_candidates=len(request.candidates),
        )

    inference_df = pd.DataFrame(
        {
            "ingredients_parsed": [c.ingredients_parsed for c in surviving],
        }
    )

    try:
        raw_scores = model.predict(inference_df)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    scores = _extract_positive_class_scores(raw_scores)

    scored = list(zip(surviving, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    if request.top_k is not None:
        scored = scored[: request.top_k]

    results = [
        RecipeScore(recipe_id=recipe.recipe_id, title=recipe.title, score=float(score), rank=i + 1)
        for i, (recipe, score) in enumerate(scored)
    ]

    return PredictResponse(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        results=results,
        filtered_out_count=filtered_out_count,
        total_candidates=len(request.candidates),
    )


def _extract_positive_class_scores(raw_scores) -> list:
    """
    Normalize whatever the pyfunc model returns into a flat list of
    positive-class probabilities.

    Handles the common shapes: a 1-D array/list of probabilities, a
    2-D array of [P(neg), P(pos)] pairs, or a DataFrame with a
    'probability'/'score' column.
    """
    if isinstance(raw_scores, pd.DataFrame):
        for col in ("probability", "score", "prediction"):
            if col in raw_scores.columns:
                return raw_scores[col].tolist()
        return raw_scores.iloc[:, -1].tolist()

    raw_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
    if raw_list and isinstance(raw_list[0], (list, tuple)):
        return [row[-1] for row in raw_list]
    return raw_list