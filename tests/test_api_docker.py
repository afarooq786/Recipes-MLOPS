"""
tests/test_api_docker.py

Unit and integration tests for the FastAPI recipe recommender inference service.
Tests health endpoints, request validation, pre-inference nutritional filtering,
and response schemas.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.filters import apply_filters, passes_filters
from api.main import app
from api.schemas import (
    DietaryTag,
    FilterConstraints,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RecipeCandidate,
)


@pytest.fixture
def client(monkeypatch):
    """
    NOTE: this used to be a bare module-level `client = TestClient(app)`.
    Without entering the ASGI lifespan context, the app's startup hook
    (which calls the real `_load_champion_model()` against whatever
    MLFLOW_TRACKING_URI is configured) could still fire on first request
    and block indefinitely with no MLflow server reachable -- every test
    in this file, even ones that never touch the model, would hang.

    Patching the loader before entering the `with` block keeps these as
    true unit/schema tests with no real network dependency.
    """
    monkeypatch.setattr(api_main, "_load_champion_model", lambda: None)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------
# Schema & Validation Tests
# ---------------------------------------------------------------------

def test_recipe_candidate_schema():
    candidate = RecipeCandidate(
        recipe_id="r101",
        title="Chicken Salad",
        ingredients_parsed="chicken breast, lettuce, olive oil",
        calories=350.0,
        cook_time_minutes=15.0,
        dietary_tags=[DietaryTag.GLUTEN_FREE, DietaryTag.DAIRY_FREE],
        excluded_ingredients_present=["chicken", "lettuce"],
    )
    assert candidate.recipe_id == "r101"
    assert DietaryTag.GLUTEN_FREE in candidate.dietary_tags


def test_filter_constraints_lowercasing():
    constraints = FilterConstraints(
        max_calories=500.0,
        excluded_ingredients=[" PEANUTS ", "Shellfish "],
    )
    assert constraints.excluded_ingredients == ["peanuts", "shellfish"]


# ---------------------------------------------------------------------
# Pre-Inference Nutritional Filtering Unit Tests
# ---------------------------------------------------------------------

def test_filtering_max_calories():
    c1 = RecipeCandidate(
        recipe_id="1",
        ingredients_parsed="apple",
        calories=150.0,
    )
    c2 = RecipeCandidate(
        recipe_id="2",
        ingredients_parsed="burger",
        calories=800.0,
    )
    constraints = FilterConstraints(max_calories=500.0)

    surviving, count = apply_filters([c1, c2], constraints)
    assert len(surviving) == 1
    assert surviving[0].recipe_id == "1"
    assert count == 1


def test_filtering_dietary_tags():
    c1 = RecipeCandidate(
        recipe_id="1",
        ingredients_parsed="salad",
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.GLUTEN_FREE],
    )
    c2 = RecipeCandidate(
        recipe_id="2",
        ingredients_parsed="steak",
        dietary_tags=[DietaryTag.KETO],
    )
    constraints = FilterConstraints(required_dietary_tags=[DietaryTag.VEGAN])

    surviving, count = apply_filters([c1, c2], constraints)
    assert len(surviving) == 1
    assert surviving[0].recipe_id == "1"


def test_filtering_excluded_ingredients():
    c1 = RecipeCandidate(
        recipe_id="1",
        ingredients_parsed="tofu, soy sauce, rice",
        excluded_ingredients_present=["soy sauce", "tofu"],
    )
    c2 = RecipeCandidate(
        recipe_id="2",
        ingredients_parsed="shrimp pasta",
        excluded_ingredients_present=["shrimp"],
    )
    constraints = FilterConstraints(excluded_ingredients=["shrimp"])

    surviving, count = apply_filters([c1, c2], constraints)
    assert len(surviving) == 1
    assert surviving[0].recipe_id == "1"
    assert count == 1


# ---------------------------------------------------------------------
# API Endpoint Integration Tests
# ---------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_name" in data
    assert data["model_name"] == "recipe-recommender"


def test_predict_endpoint_empty_candidates_fails(client):
    payload = {"candidates": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable entity due to min_length=1


def test_predict_endpoint_with_all_candidates_filtered(client):
    payload = {
        "candidates": [
            {
                "recipe_id": "r1",
                "title": "High Calorie Feast",
                "ingredients_parsed": "cheese, butter, pork",
                "calories": 1200.0,
            }
        ],
        "constraints": {
            "max_calories": 500.0,
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["filtered_out_count"] == 1
    assert data["total_candidates"] == 1
