"""
tests/test_api_integration.py — Row 26: Integration tests.

Exercises the complete path: HTTP request -> Pydantic validation ->
nutritional filtering -> model scoring -> ranking -> HTTP response,
using a fake pyfunc-compatible model in place of a live MLflow
registry connection. This lets the full request/response contract be
tested in CI without requiring a running MLflow server or trained
model artifact (see tests/test_api_docker.py for the lighter unit-level
schema/filter tests that don't touch model state at all).

IMPORTANT: `api.main`'s lifespan calls `_load_champion_model()` on
startup, which makes a real network call to the configured MLflow
tracking server. Every fixture below monkeypatches that function
BEFORE the TestClient context is entered, so tests never attempt a
real network connection and never hang waiting on one.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


class _FakeChampionModel:
    """Deterministic stand-in for the MLflow pyfunc champion model.

    Scores every candidate by a simple, deterministic rule (ingredient
    text length) so ranking order is predictable and assertable, without
    depending on the real trained ensemble or a live MLflow server.
    """

    def predict(self, inference_df: pd.DataFrame):
        return [
            min(0.99, 0.1 + 0.05 * len(text.split()))
            for text in inference_df["ingredients_parsed"]
        ]


@pytest.fixture
def client_with_fake_model(monkeypatch):
    """A TestClient where the champion model loader is patched to return a
    fake model, so lifespan startup never touches the network and /predict
    exercises the real scoring/ranking code path deterministically."""
    monkeypatch.setattr(api_main, "_load_champion_model", lambda: _FakeChampionModel())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_unavailable_model(monkeypatch):
    """A TestClient where the champion model loader always fails, simulating
    an unreachable MLflow registry -- without ever making a real network call."""

    def _raise():
        raise RuntimeError("MLflow registry unreachable")

    monkeypatch.setattr(api_main, "_load_champion_model", _raise)
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_reports_model_loaded_true_when_model_present(self, client_with_fake_model):
        response = client_with_fake_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"

    def test_reports_degraded_when_model_unavailable(self, client_with_unavailable_model):
        response = client_with_unavailable_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False
        assert data["status"] == "degraded"


class TestPredictEndToEnd:
    def test_scores_and_ranks_multiple_candidates(self, client_with_fake_model):
        payload = {
            "candidates": [
                {"recipe_id": "r1", "title": "Short", "ingredients_parsed": "salt"},
                {
                    "recipe_id": "r2",
                    "title": "Long",
                    "ingredients_parsed": "chicken breast diced onion garlic olive oil basil",
                },
            ]
        }
        response = client_with_fake_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_candidates"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["recipe_id"] == "r2"
        assert data["results"][0]["rank"] == 1
        assert data["results"][0]["score"] >= data["results"][1]["score"]

    def test_filtering_runs_before_scoring(self, client_with_fake_model):
        payload = {
            "candidates": [
                {"recipe_id": "r1", "ingredients_parsed": "cheap salad", "calories": 200},
                {"recipe_id": "r2", "ingredients_parsed": "huge burger feast", "calories": 1500},
            ],
            "constraints": {"max_calories": 500},
        }
        response = client_with_fake_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["filtered_out_count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["recipe_id"] == "r1"

    def test_top_k_truncates_results(self, client_with_fake_model):
        payload = {
            "candidates": [
                {"recipe_id": f"r{i}", "ingredients_parsed": "ingredient " * i}
                for i in range(1, 6)
            ],
            "top_k": 2,
        }
        response = client_with_fake_model.post("/predict", json=payload)
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2

    def test_all_filtered_out_returns_empty_results_not_error(self, client_with_fake_model):
        payload = {
            "candidates": [{"recipe_id": "r1", "ingredients_parsed": "burger", "calories": 2000}],
            "constraints": {"max_calories": 100},
        }
        response = client_with_fake_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["filtered_out_count"] == 1


class TestPredictWithoutModel:
    def test_returns_503_when_model_unavailable(self, client_with_unavailable_model):
        """If the champion model can't be loaded, /predict should fail loudly
        with a 503, not silently return wrong scores."""
        response = client_with_unavailable_model.post(
            "/predict",
            json={"candidates": [{"recipe_id": "r1", "ingredients_parsed": "salt"}]},
        )
        assert response.status_code == 503
