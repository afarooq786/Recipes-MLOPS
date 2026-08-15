"""
Unit tests for preprocessing/preprocess.py.

Covers the pure text/time-parsing helper functions used to clean raw
recipe fields before feature engineering.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from preprocessing.preprocess import (
    clean_text,
    normalize_ingredient,
    parse_ingredients,
    parse_time_to_minutes,
    preprocess_dataframe,
)


class TestCleanText:
    def test_strips_and_collapses_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_none_returns_none(self):
        assert clean_text(None) is None

    def test_nan_returns_none(self):
        assert clean_text(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert clean_text("   ") is None

    def test_preserves_case(self):
        assert clean_text("Cherry Pie") == "Cherry Pie"


class TestParseTimeToMinutes:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("45 mins", 45.0),
            ("1 hr 30 mins", 90.0),
            ("2 hrs", 120.0),
            ("30", 30.0),
            ("1 hrs 23 mins", 83.0),
        ],
    )
    def test_parses_common_formats(self, raw, expected):
        assert parse_time_to_minutes(raw) == expected

    def test_none_returns_none(self):
        assert parse_time_to_minutes(None) is None

    def test_nan_returns_none(self):
        assert parse_time_to_minutes(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert parse_time_to_minutes("") is None

    def test_garbage_text_returns_none(self):
        assert parse_time_to_minutes("overnight, ideally") is None


class TestNormalizeIngredient:
    def test_lowercases_and_strips_punctuation(self):
        result = normalize_ingredient("2 CUPS Flour!!")
        assert result == result.lower()
        assert "!" not in result

    def test_collapses_internal_whitespace(self):
        assert normalize_ingredient("olive   oil") == "olive oil"


class TestParseIngredients:
    def test_splits_on_semicolon_and_newline(self):
        result = parse_ingredients("flour; sugar\nbutter")
        assert result == ["flour", "sugar", "butter"]

    def test_none_returns_empty_list(self):
        assert parse_ingredients(None) == []

    def test_nan_returns_empty_list(self):
        assert parse_ingredients(float("nan")) == []

    def test_drops_empty_segments(self):
        result = parse_ingredients("flour;; sugar;")
        assert "" not in result


class TestPreprocessDataframe:
    def test_returns_new_dataframe_not_mutated_in_place(self):
        df = pd.DataFrame(
            {
                "ingredients": ["flour; sugar"],
                "directions": ["Mix well."],
                "recipe_name": ["Test Recipe"],
                "rating": [4.5],
            }
        )
        original_id = id(df)
        result = preprocess_dataframe(df)
        assert id(result) != original_id

    def test_output_is_dataframe(self):
        df = pd.DataFrame(
            {
                "ingredients": ["flour; sugar"],
                "directions": ["Mix well."],
                "recipe_name": ["Test Recipe"],
                "rating": [4.5],
            }
        )
        result = preprocess_dataframe(df)
        assert isinstance(result, pd.DataFrame)
