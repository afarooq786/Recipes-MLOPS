"""
tests/test_webapp.py — Unit and integration tests for the webapp pipeline adapter.
"""

import json
import pytest
from webapp.pipeline_adapter import PipelineAdapter


@pytest.fixture
def adapter():
    return PipelineAdapter(api_url="http://localhost:8000")


def test_validate_recipe_valid(adapter):
    valid_recipe = {
        "title": "Lemon Garlic Pasta",
        "ingredients": "pasta, fresh garlic, olive oil, lemon juice, parmesan cheese",
        "calories": 450,
        "cook_time_minutes": 20,
    }
    is_valid, errors = adapter.validate_recipe(valid_recipe)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_recipe_missing_fields(adapter):
    invalid_recipe = {
        "title": "",
        "ingredients": "",
    }
    is_valid, errors = adapter.validate_recipe(invalid_recipe)
    assert is_valid is False
    assert any("title" in e.lower() for e in errors)
    assert any("ingredients" in e.lower() for e in errors)


def test_validate_recipe_invalid_ranges(adapter):
    invalid_recipe = {
        "title": "Extreme Recipe",
        "ingredients": "water, salt",
        "calories": -50,
        "cook_time_minutes": -10,
    }
    is_valid, errors = adapter.validate_recipe(invalid_recipe)
    assert is_valid is False
    assert any("calories cannot be negative" in e.lower() for e in errors)
    assert any("cook time cannot be negative" in e.lower() for e in errors)


def test_preprocess_recipe(adapter):
    raw_recipe = {
        "recipe_name": "  Classic Guacamole!  ",
        "ingredients": "2 ripe avocados;\n1 tbsp lime juice;\n1/4 cup chopped cilantro",
        "cook_time": "15 mins",
        "calories": "220",
    }
    pre = adapter.preprocess_recipe(raw_recipe)
    assert pre["title"] == "Classic Guacamole!"
    assert "avocados" in pre["ingredients_parsed"]
    assert "lime juice" in pre["ingredients_parsed"]
    assert pre["cook_time_minutes"] == 15.0
    assert pre["calories"] == 220.0


def test_derive_features_vegetarian_vegan(adapter):
    vegan_recipe = {
        "title": "Fresh Fruit Bowl",
        "ingredients_parsed": "fresh apples, bananas, strawberries, blueberries, chia seeds",
        "cook_time_minutes": 5.0,
        "normalized_token_list": ["fresh apples", "bananas", "strawberries", "blueberries", "chia seeds"],
    }
    feats = adapter.derive_features(vegan_recipe)
    assert "vegetarian" in feats["dietary_tags"]
    assert "vegan" in feats["dietary_tags"]
    assert "gluten_free" in feats["dietary_tags"]
    assert feats["time_bucket"] == "quick_under_15m"


def test_derive_features_meat_non_vegetarian(adapter):
    meat_recipe = {
        "title": "Grilled Chicken Salad",
        "ingredients_parsed": "grilled chicken breast, romaine lettuce, parmesan cheese, caesar dressing",
        "cook_time_minutes": 25.0,
        "normalized_token_list": ["chicken", "romaine", "parmesan", "dressing"],
    }
    feats = adapter.derive_features(meat_recipe)
    assert "vegetarian" not in feats["dietary_tags"]
    assert "vegan" not in feats["dietary_tags"]
    assert "chicken" in feats["detected_proteins"]
    assert feats["time_bucket"] == "moderate_15_30m"


def test_apply_constraints_calories(adapter):
    pre = {"calories": 600, "cook_time_minutes": 20, "ingredients_parsed": "rice, beef, peppers"}
    tags = ["gluten_free"]

    # Exceeds max calories
    passed, reason = adapter.apply_constraints(pre, tags, {"max_calories": 500})
    assert passed is False
    assert "calories" in reason.lower()

    # Under max calories
    passed, reason = adapter.apply_constraints(pre, tags, {"max_calories": 700})
    assert passed is True


def test_apply_constraints_allergens_and_tags(adapter):
    pre = {"calories": 300, "cook_time_minutes": 15, "ingredients_parsed": "peanut butter, oat flour, honey"}
    tags = ["vegetarian"]

    # Excluded allergen present
    passed, reason = adapter.apply_constraints(pre, tags, {"excluded_ingredients": ["peanut"]})
    assert passed is False
    assert "peanut" in reason.lower()

    # Missing required tag
    passed, reason = adapter.apply_constraints(pre, tags, {"required_dietary_tags": ["vegan"]})
    assert passed is False
    assert "missing required dietary tag" in reason.lower()


def test_single_recipe_end_to_end(adapter):
    recipe = {
        "title": "Tomato Basil Bruschetta",
        "ingredients": "crusty baguette, ripe roma tomatoes, extra virgin olive oil, fresh basil, garlic, balsamic glaze",
        "cook_time_minutes": 15,
        "calories": 280,
    }
    result = adapter.run_single_recipe(recipe)
    assert result.is_valid is True
    assert result.is_filtered_out is False
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert len(result.steps) == 5
    assert all(s.status == "success" for s in result.steps)


def test_parse_uploaded_csv(adapter):
    csv_data = (
        "recipe_name,ingredients,calories,cook_time\n"
        "Apple Crisp,\"apples, oats, brown sugar, cinnamon\",350,45 mins\n"
        "Avocado Toast,\"bread, avocado, olive oil, salt\",250,5 mins\n"
    ).encode("utf-8")

    parsed = adapter.parse_uploaded_file(csv_data, "test_recipes.csv")
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Apple Crisp"
    assert "apples" in parsed[0]["ingredients"]
    assert parsed[1]["title"] == "Avocado Toast"


def test_parse_uploaded_json(adapter):
    json_data = json.dumps([
        {
            "recipe_id": "r1",
            "title": "Greek Salad",
            "ingredients": "cucumbers, tomatoes, feta, olives, oregano, olive oil",
            "calories": 300,
            "cook_time_minutes": 10,
        }
    ]).encode("utf-8")

    parsed = adapter.parse_uploaded_file(json_data, "recipes.json")
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Greek Salad"


def test_batch_scoring_and_ranking(adapter):
    candidates = [
        {"recipe_id": "c1", "title": "Garlic Bread", "ingredients_parsed": "baguette, butter, garlic, parsley", "calories": 200, "cook_time_minutes": 10, "dietary_tags": ["vegetarian"]},
        {"recipe_id": "c2", "title": "Bacon Steak", "ingredients_parsed": "beef steak, bacon fat, butter", "calories": 900, "cook_time_minutes": 30, "dietary_tags": []},
    ]

    # Test with calorie constraint
    out = adapter.score_candidates(candidates, constraints={"max_calories": 500}, top_k=5)
    assert out["total_candidates"] == 2
    assert out["filtered_out_count"] == 1
    assert len(out["results"]) == 1
    assert out["results"][0]["recipe_id"] == "c1"
    assert out["results"][0]["rank"] == 1
