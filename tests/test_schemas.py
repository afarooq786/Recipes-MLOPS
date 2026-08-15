"""
Unit tests for api/schemas.py (Pydantic request/response contracts).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    DietaryTag,
    FilterConstraints,
    PredictRequest,
    RecipeCandidate,
)


class TestRecipeCandidate:
    def test_valid_minimal_candidate(self):
        c = RecipeCandidate(recipe_id="r1", ingredients_parsed="flour, sugar")
        assert c.recipe_id == "r1"
        assert c.dietary_tags == []

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            RecipeCandidate(recipe_id="r1")  # missing ingredients_parsed

    def test_negative_calories_rejected(self):
        with pytest.raises(ValidationError):
            RecipeCandidate(recipe_id="r1", ingredients_parsed="x", calories=-5)

    def test_negative_cook_time_rejected(self):
        with pytest.raises(ValidationError):
            RecipeCandidate(recipe_id="r1", ingredients_parsed="x", cook_time_minutes=-1)

    def test_invalid_dietary_tag_rejected(self):
        with pytest.raises(ValidationError):
            RecipeCandidate(recipe_id="r1", ingredients_parsed="x", dietary_tags=["not_a_real_tag"])


class TestFilterConstraints:
    def test_excluded_ingredients_lowercased_and_trimmed(self):
        c = FilterConstraints(excluded_ingredients=[" Peanuts ", "SHELLFISH"])
        assert c.excluded_ingredients == ["peanuts", "shellfish"]

    def test_blank_excluded_ingredients_dropped(self):
        c = FilterConstraints(excluded_ingredients=["  ", "garlic"])
        assert c.excluded_ingredients == ["garlic"]


class TestPredictRequest:
    def test_requires_at_least_one_candidate(self):
        with pytest.raises(ValidationError):
            PredictRequest(candidates=[])

    def test_top_k_must_be_positive(self):
        with pytest.raises(ValidationError):
            PredictRequest(
                candidates=[RecipeCandidate(recipe_id="r1", ingredients_parsed="x")],
                top_k=0,
            )

    def test_valid_request_with_constraints(self):
        req = PredictRequest(
            candidates=[RecipeCandidate(recipe_id="r1", ingredients_parsed="x")],
            constraints=FilterConstraints(max_calories=500, required_dietary_tags=[DietaryTag.VEGAN]),
            top_k=5,
        )
        assert req.top_k == 5
        assert req.constraints.required_dietary_tags == [DietaryTag.VEGAN]
