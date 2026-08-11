"""
Request/response schemas for the recipe recommender inference API.

These schemas define the contract between clients and the FastAPI
service in `api/main.py`. Keep them in sync with the feature set the
champion model (see models/champion_config.json) expects at inference
time -- currently the cleaned `ingredients_parsed` text field, plus
metadata used only by the nutritional filtering layer (api/filters.py),
not by the model itself.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class DietaryTag(str, Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    LOW_CARB = "low_carb"
    KETO = "keto"
    PALEO = "paleo"


class RecipeCandidate(BaseModel):
    """A single recipe to be scored and/or filtered."""

    recipe_id: str = Field(..., description="Stable unique identifier for the recipe.")
    title: Optional[str] = Field(None, description="Recipe title, for display only.")
    ingredients_parsed: str = Field(
        ..., description="Cleaned ingredient text, matching the preprocessing pipeline output."
    )
    cook_time_minutes: Optional[float] = Field(
        None, ge=0, description="Total cook/prep time in minutes, if known."
    )
    calories: Optional[float] = Field(
        None, ge=0, description="Calories per serving, if known."
    )
    dietary_tags: List[DietaryTag] = Field(
        default_factory=list,
        description="Dietary properties this recipe satisfies (e.g. vegetarian, gluten_free).",
    )
    excluded_ingredients_present: List[str] = Field(
        default_factory=list,
        description=(
            "Optional pre-extracted list of notable ingredients in this recipe, used for "
            "fast exclusion filtering (e.g. ['peanuts', 'shellfish']). If omitted, the "
            "filtering layer falls back to substring matching against ingredients_parsed."
        ),
    )


class FilterConstraints(BaseModel):
    """Constraints applied by the nutritional filtering layer before ranking."""

    max_calories: Optional[float] = Field(None, ge=0)
    max_cook_time_minutes: Optional[float] = Field(None, ge=0)
    required_dietary_tags: List[DietaryTag] = Field(
        default_factory=list,
        description="Recipe must satisfy ALL of these tags to pass filtering.",
    )
    excluded_ingredients: List[str] = Field(
        default_factory=list,
        description="Recipe is dropped if any of these ingredients are present.",
    )

    @field_validator("excluded_ingredients")
    @classmethod
    def _lowercase_excluded(cls, v: List[str]) -> List[str]:
        return [item.strip().lower() for item in v if item.strip()]


class PredictRequest(BaseModel):
    candidates: List[RecipeCandidate] = Field(..., min_length=1)
    constraints: Optional[FilterConstraints] = Field(
        None, description="If omitted, no filtering is applied and all candidates are scored."
    )
    top_k: Optional[int] = Field(
        None, ge=1, description="If set, only the top_k highest-scoring recipes are returned."
    )


class RecipeScore(BaseModel):
    recipe_id: str
    title: Optional[str] = None
    score: float = Field(..., description="Model-predicted probability of rating >= 4.")
    rank: int = Field(..., description="1-indexed rank among returned results, best first.")


class PredictResponse(BaseModel):
    model_name: str
    model_alias: str
    results: List[RecipeScore]
    filtered_out_count: int = Field(
        0, description="Number of candidates removed by the nutritional filtering layer."
    )
    total_candidates: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_alias: str
    model_loaded: bool