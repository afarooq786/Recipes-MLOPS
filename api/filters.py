"""
Nutritional filtering layer.

Filters candidate recipes on calorie limits, dietary restrictions,
cooking-time constraints, and excluded ingredients *before* the
candidates are passed to the champion model for ranking. This keeps
the model's job strictly to scoring/ranking, and keeps hard business
constraints (allergies, diet type, time budget) deterministic and
easy to unit test independently of the model.

Usage:
    from api.filters import apply_filters

    passed, dropped = apply_filters(candidates, constraints)
"""

from __future__ import annotations

from typing import List, Tuple

from api.schemas import FilterConstraints, RecipeCandidate


def _fails_calorie_limit(recipe: RecipeCandidate, constraints: FilterConstraints) -> bool:
    if constraints.max_calories is None:
        return False
    # Recipes with unknown calories are conservatively excluded once a
    # calorie constraint is actually requested, rather than silently
    # passing them through.
    if recipe.calories is None:
        return True
    return recipe.calories > constraints.max_calories


def _fails_cook_time_limit(recipe: RecipeCandidate, constraints: FilterConstraints) -> bool:
    if constraints.max_cook_time_minutes is None:
        return False
    if recipe.cook_time_minutes is None:
        return True
    return recipe.cook_time_minutes > constraints.max_cook_time_minutes


def _fails_dietary_requirements(recipe: RecipeCandidate, constraints: FilterConstraints) -> bool:
    if not constraints.required_dietary_tags:
        return False
    recipe_tags = set(recipe.dietary_tags)
    required_tags = set(constraints.required_dietary_tags)
    return not required_tags.issubset(recipe_tags)


def _contains_excluded_ingredient(recipe: RecipeCandidate, constraints: FilterConstraints) -> bool:
    if not constraints.excluded_ingredients:
        return False

    # Prefer the pre-extracted ingredient list when available (fast, exact).
    if recipe.excluded_ingredients_present:
        present = {i.strip().lower() for i in recipe.excluded_ingredients_present}
        return any(excl in present for excl in constraints.excluded_ingredients)

    # Fall back to substring matching against the parsed ingredient text.
    text = recipe.ingredients_parsed.lower()
    return any(excl in text for excl in constraints.excluded_ingredients)


def passes_filters(recipe: RecipeCandidate, constraints: FilterConstraints) -> bool:
    """Return True if `recipe` satisfies every constraint in `constraints`."""
    if _fails_calorie_limit(recipe, constraints):
        return False
    if _fails_cook_time_limit(recipe, constraints):
        return False
    if _fails_dietary_requirements(recipe, constraints):
        return False
    if _contains_excluded_ingredient(recipe, constraints):
        return False
    return True


def apply_filters(
    candidates: List[RecipeCandidate],
    constraints: FilterConstraints | None,
) -> Tuple[List[RecipeCandidate], int]:
    """
    Apply nutritional/dietary/time filtering to a list of candidates.

    Returns:
        (surviving_candidates, filtered_out_count)
    """
    if constraints is None:
        return candidates, 0

    surviving = [c for c in candidates if passes_filters(c, constraints)]
    filtered_out_count = len(candidates) - len(surviving)
    return surviving, filtered_out_count