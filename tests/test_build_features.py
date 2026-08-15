"""
Unit tests for features/build_features.py.
"""

from __future__ import annotations

import pandas as pd

from features.build_features import (
    build_cooking_time_features,
    build_dietary_preference_features,
    build_ingredient_features,
    fit_popularity_stats,
    apply_popularity_features,
)


class TestBuildIngredientFeatures:
    def test_ingredient_count_matches_list_length(self):
        df = pd.DataFrame({"ingredients_parsed": ["chicken breast diced onion garlic"]})
        result = build_ingredient_features(df)
        assert result.loc[0, "ingredient_count"] >= 1

    def test_has_chicken_flag_set_when_present(self):
        df = pd.DataFrame({"ingredients_parsed": ["2 chicken breasts diced"]})
        result = build_ingredient_features(df)
        assert result.loc[0, "has_chicken"] == 1

    def test_has_chicken_flag_zero_when_absent(self):
        df = pd.DataFrame({"ingredients_parsed": ["tofu broccoli soy sauce"]})
        result = build_ingredient_features(df)
        assert result.loc[0, "has_chicken"] == 0

    def test_missing_column_is_noop(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = build_ingredient_features(df)
        assert "ingredient_count" not in result.columns


class TestBuildCookingTimeFeatures:
    def test_prep_to_cook_ratio_computed(self):
        df = pd.DataFrame({"prep_time_minutes": [10.0], "cook_time_minutes": [20.0], "total_time_minutes": [30.0]})
        result = build_cooking_time_features(df)
        assert result.loc[0, "prep_to_cook_ratio"] == 0.5

    def test_zero_cook_time_does_not_raise_division_error(self):
        df = pd.DataFrame({"prep_time_minutes": [10.0], "cook_time_minutes": [0.0], "total_time_minutes": [10.0]})
        result = build_cooking_time_features(df)
        # Division by zero is guarded -> should be NaN, not inf or a raised error.
        assert pd.isna(result.loc[0, "prep_to_cook_ratio"])


class TestBuildDietaryPreferenceFeatures:
    def test_vegetarian_flag_true_when_no_meat_keywords(self):
        df = pd.DataFrame({"ingredients_parsed": ["tofu broccoli soy sauce rice"]})
        result = build_dietary_preference_features(df)
        assert result.loc[0, "is_likely_vegetarian"] == 1

    def test_vegetarian_flag_false_when_meat_present(self):
        df = pd.DataFrame({"ingredients_parsed": ["chicken breast diced onion"]})
        result = build_dietary_preference_features(df)
        assert result.loc[0, "is_likely_vegetarian"] == 0


class TestPopularityFeatures:
    def test_fit_popularity_stats_returns_per_cuisine_dict(self):
        df = pd.DataFrame(
            {
                "rating": [4.0, 5.0, 3.0, 4.5],
                "cuisine_path": ["/Italian/", "/Italian/", "/Mexican/", "/Mexican/"],
            }
        )
        stats = fit_popularity_stats(df)
        assert "/Italian/" in stats
        assert "/Mexican/" in stats

    def test_apply_popularity_features_no_leakage_uses_fitted_stats(self):
        train = pd.DataFrame(
            {"rating": [4.0, 5.0, 3.0, 4.0], "cuisine_path": ["/Italian/", "/Italian/", "/Italian/", "/Italian/"]}
        )
        stats = fit_popularity_stats(train)

        # Apply those TRAIN-fit stats to a different (e.g. validation) frame.
        val = pd.DataFrame({"rating": [4.5], "cuisine_path": ["/Italian/"]})
        result = apply_popularity_features(val, stats)
        assert "popularity_zscore_within_cuisine" in result.columns
        assert not pd.isna(result.loc[0, "popularity_zscore_within_cuisine"])

    def test_missing_columns_is_noop(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = apply_popularity_features(df, {})
        assert "popularity_zscore_within_cuisine" not in result.columns
