"""
Unit tests for data/validate.py -- the Pandera schema rules that gate the
data pipeline (Step 2).
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.validate import RECIPES_SCHEMA, _duplicate_warning, _null_rate_warnings


def valid_row(**overrides) -> dict:
    row = dict(
        recipe_name="Test Recipe",
        prep_time="10 mins",
        cook_time="20 mins",
        total_time="30 mins",
        servings=4,
        yield_="4 servings",
        ingredients="flour, sugar, eggs",
        directions="Mix and bake.",
        rating=4.5,
        url="https://example.com/recipe/1",
        cuisine_path="/Desserts/",
        nutrition="{}",
        timing="Prep Time: 10 mins",
        img_src="https://example.com/img.jpg",
    )
    row.update(overrides)
    row["yield"] = row.pop("yield_")
    return row


class TestRecipesSchema:
    def test_valid_row_passes(self):
        df = pd.DataFrame([valid_row()])
        RECIPES_SCHEMA.validate(df)  # should not raise

    def test_rating_out_of_range_fails(self):
        df = pd.DataFrame([valid_row(rating=6.0)])
        with pytest.raises(Exception):
            RECIPES_SCHEMA.validate(df, lazy=True)

    def test_negative_servings_fails(self):
        df = pd.DataFrame([valid_row(servings=-1)])
        with pytest.raises(Exception):
            RECIPES_SCHEMA.validate(df, lazy=True)

    def test_empty_ingredients_fails(self):
        df = pd.DataFrame([valid_row(ingredients="")])
        with pytest.raises(Exception):
            RECIPES_SCHEMA.validate(df, lazy=True)

    def test_url_not_starting_with_http_fails(self):
        df = pd.DataFrame([valid_row(url="www.example.com/recipe/1")])
        with pytest.raises(Exception):
            RECIPES_SCHEMA.validate(df, lazy=True)

    def test_null_recipe_name_fails(self):
        df = pd.DataFrame([valid_row(recipe_name=None)])
        with pytest.raises(Exception):
            RECIPES_SCHEMA.validate(df, lazy=True)

    def test_extra_columns_allowed(self):
        row = valid_row()
        row["some_unexpected_column"] = "ok"
        df = pd.DataFrame([row])
        RECIPES_SCHEMA.validate(df)  # strict=False -> should not raise


class TestNullRateWarnings:
    def test_flags_high_null_rate_column(self):
        df = pd.DataFrame({"prep_time": [None] * 8 + ["10 mins"] * 2})
        warnings = _null_rate_warnings(df, ["prep_time"])
        assert any("prep_time" in w for w in warnings)

    def test_no_warning_for_low_null_rate(self):
        df = pd.DataFrame({"prep_time": [None] + ["10 mins"] * 9})
        warnings = _null_rate_warnings(df, ["prep_time"])
        assert warnings == []


class TestDuplicateWarning:
    def test_flags_duplicate_rows(self):
        df = pd.DataFrame({"url": ["https://a.com", "https://a.com", "https://b.com"]})
        warnings = _duplicate_warning(df, ["url"])
        assert len(warnings) == 1

    def test_no_warning_when_unique(self):
        df = pd.DataFrame({"url": ["https://a.com", "https://b.com"]})
        warnings = _duplicate_warning(df, ["url"])
        assert warnings == []
