"""
Unit tests for api/filters.py (the deterministic nutritional/dietary
filtering layer applied before candidates reach the model).
"""

from __future__ import annotations

from api.filters import apply_filters, passes_filters
from api.schemas import DietaryTag, FilterConstraints, RecipeCandidate


def make_candidate(**overrides) -> RecipeCandidate:
    defaults = dict(
        recipe_id="r1",
        title="Test Recipe",
        ingredients_parsed="chicken, rice, broccoli",
        cook_time_minutes=30,
        calories=400,
        dietary_tags=[],
        excluded_ingredients_present=[],
    )
    defaults.update(overrides)
    return RecipeCandidate(**defaults)


class TestCalorieLimit:
    def test_passes_under_limit(self):
        recipe = make_candidate(calories=300)
        constraints = FilterConstraints(max_calories=500)
        assert passes_filters(recipe, constraints) is True

    def test_fails_over_limit(self):
        recipe = make_candidate(calories=900)
        constraints = FilterConstraints(max_calories=500)
        assert passes_filters(recipe, constraints) is False

    def test_unknown_calories_excluded_when_limit_requested(self):
        recipe = make_candidate(calories=None)
        constraints = FilterConstraints(max_calories=500)
        assert passes_filters(recipe, constraints) is False

    def test_no_constraint_always_passes(self):
        recipe = make_candidate(calories=None)
        constraints = FilterConstraints()
        assert passes_filters(recipe, constraints) is True


class TestCookTimeLimit:
    def test_passes_under_limit(self):
        recipe = make_candidate(cook_time_minutes=15)
        constraints = FilterConstraints(max_cook_time_minutes=20)
        assert passes_filters(recipe, constraints) is True

    def test_fails_over_limit(self):
        recipe = make_candidate(cook_time_minutes=45)
        constraints = FilterConstraints(max_cook_time_minutes=20)
        assert passes_filters(recipe, constraints) is False


class TestDietaryTags:
    def test_passes_when_all_required_tags_present(self):
        recipe = make_candidate(dietary_tags=[DietaryTag.VEGETARIAN, DietaryTag.GLUTEN_FREE])
        constraints = FilterConstraints(required_dietary_tags=[DietaryTag.VEGETARIAN])
        assert passes_filters(recipe, constraints) is True

    def test_fails_when_missing_a_required_tag(self):
        recipe = make_candidate(dietary_tags=[DietaryTag.GLUTEN_FREE])
        constraints = FilterConstraints(required_dietary_tags=[DietaryTag.VEGAN])
        assert passes_filters(recipe, constraints) is False


class TestExcludedIngredients:
    def test_fails_when_excluded_ingredient_in_preextracted_list(self):
        recipe = make_candidate(excluded_ingredients_present=["peanuts"])
        constraints = FilterConstraints(excluded_ingredients=["peanuts"])
        assert passes_filters(recipe, constraints) is False

    def test_fails_via_substring_fallback(self):
        recipe = make_candidate(ingredients_parsed="pasta, shellfish stock, garlic")
        constraints = FilterConstraints(excluded_ingredients=["shellfish"])
        assert passes_filters(recipe, constraints) is False

    def test_passes_when_excluded_ingredient_absent(self):
        recipe = make_candidate(ingredients_parsed="pasta, tomato, garlic")
        constraints = FilterConstraints(excluded_ingredients=["shellfish"])
        assert passes_filters(recipe, constraints) is True


class TestApplyFilters:
    def test_no_constraints_returns_all_candidates(self):
        candidates = [make_candidate(recipe_id="r1"), make_candidate(recipe_id="r2")]
        surviving, dropped = apply_filters(candidates, None)
        assert len(surviving) == 2
        assert dropped == 0

    def test_filters_out_correct_count(self):
        candidates = [
            make_candidate(recipe_id="r1", calories=200),
            make_candidate(recipe_id="r2", calories=900),
        ]
        constraints = FilterConstraints(max_calories=500)
        surviving, dropped = apply_filters(candidates, constraints)
        assert [c.recipe_id for c in surviving] == ["r1"]
        assert dropped == 1

    def test_excluded_ingredients_are_lowercased_on_input(self):
        constraints = FilterConstraints(excluded_ingredients=["  PEANUTS  "])
        assert constraints.excluded_ingredients == ["peanuts"]
