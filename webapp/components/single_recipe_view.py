"""
webapp/components/single_recipe_view.py — Consumer-focused Recipe Evaluator & Quality Predictor.
"""

import textwrap
from typing import Dict, Any, List
import streamlit as st
from webapp.pipeline_adapter import PipelineAdapter

PRESETS: Dict[str, Dict[str, Any]] = {
    "— 📖 Select a Chef-Curated Recipe to Try —": {
        "title": "",
        "ingredients": "",
        "cook_time_minutes": 20,
        "calories": 400,
        "dietary_tags": [],
    },
    "🥗 Mediterranean Herb Chickpea Salad (Fresh & Vibrant)": {
        "title": "Mediterranean Herb Chickpea Salad",
        "ingredients": "chickpeas, cucumber, cherry tomatoes, kalamata olives, red onion, fresh parsley, extra virgin olive oil, lemon juice, feta cheese, garlic, sea salt, oregano",
        "cook_time_minutes": 15,
        "calories": 320,
        "dietary_tags": ["vegetarian", "gluten_free"],
    },
    "🍝 Creamy Tuscan Garlic Butter Pasta (Comfort Classic)": {
        "title": "Creamy Tuscan Garlic Butter Pasta",
        "ingredients": "fettuccine pasta, heavy cream, parmesan cheese, garlic cloves, butter, sun-dried tomatoes, baby spinach, fresh basil, cracked black pepper",
        "cook_time_minutes": 25,
        "calories": 580,
        "dietary_tags": ["vegetarian"],
    },
    "🥩 Rosemary Garlic Butter Seared Ribeye (Gourmet Keto)": {
        "title": "Rosemary Garlic Butter Seared Ribeye",
        "ingredients": "prime ribeye steak, unsalted butter, fresh rosemary sprigs, fresh thyme, crushed garlic cloves, coarse sea salt, black peppercorns",
        "cook_time_minutes": 20,
        "calories": 650,
        "dietary_tags": ["gluten_free", "keto"],
    },
    "🍲 Spicy Sesame Tofu & Veggie Stir Fry (Plant-Based)": {
        "title": "Spicy Sesame Tofu & Veggie Stir Fry",
        "ingredients": "crispy extra firm tofu, broccoli florets, red bell peppers, snap peas, toasted sesame oil, tamari soy sauce, fresh grated ginger, minced garlic, sriracha, scallions",
        "cook_time_minutes": 25,
        "calories": 360,
        "dietary_tags": ["vegan", "vegetarian", "gluten_free"],
    },
    "🍫 Decadent Warm Molten Chocolate Cake (Sweet Indulgence)": {
        "title": "Decadent Warm Molten Chocolate Cake",
        "ingredients": "dark bittersweet chocolate, unsalted butter, powdered sugar, farm fresh eggs, vanilla extract, pinch of sea salt, cocoa powder",
        "cook_time_minutes": 20,
        "calories": 490,
        "dietary_tags": ["vegetarian", "gluten_free"],
    },
}


def render_single_recipe_view(adapter: PipelineAdapter):
    """Renders the consumer recipe assessment studio."""
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-family: 'Playfair Display', serif; font-size: 1.6rem; margin-bottom: 0.25rem;">
                ✨ Recipe Quality & Flavor Evaluator
            </h2>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
                Type your recipe or pick a curated favorite to get an instant AI culinary rating, dietary badges, and nutrition insights.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Preset Selector with food icons
    preset_choice = st.selectbox(
        "Explore Chef's Favorites",
        options=list(PRESETS.keys()),
        index=0,
        help="Quickly populate fields with balanced sample recipes.",
    )
    preset_data = PRESETS[preset_choice]

    # Input Form Layout
    col1, col2 = st.columns([1.25, 0.75])

    with col1:
        title = st.text_input(
            "🍽️ Recipe Name",
            value=preset_data["title"],
            placeholder="e.g. Lemon Herb Roasted Chicken with Asparagus",
        )
        ingredients = st.text_area(
            "🛒 Ingredients (list items separated by commas or lines)",
            value=preset_data["ingredients"],
            height=145,
            placeholder="e.g.\nchicken breast\nfresh garlic\nrosemary\nextra virgin olive oil\nlemon slices",
        )

    with col2:
        cook_time = st.number_input(
            "⏱️ Cook & Prep Time (mins)",
            min_value=1,
            max_value=360,
            value=int(preset_data.get("cook_time_minutes") or 20),
            step=5,
        )
        calories = st.number_input(
            "🔥 Calories per Serving (kcal)",
            min_value=10,
            max_value=3000,
            value=int(preset_data.get("calories") or 400),
            step=25,
        )
        dietary_choices = ["vegetarian", "vegan", "gluten_free", "dairy_free", "nut_free", "keto", "paleo"]
        user_diet = st.multiselect(
            "🏷️ Dietary Preferences / Tags",
            options=dietary_choices,
            default=[t for t in preset_data.get("dietary_tags", []) if t in dietary_choices],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    # Preferences / Allergen Filter Dropdown
    with st.expander("⚙️ Dietary Constraints & Allergen Exclusions", expanded=False):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            max_cal_in = st.number_input("Max Calorie Cap (0 = No limit)", min_value=0, max_value=2500, value=0, step=50)
            max_cal = float(max_cal_in) if max_cal_in > 0 else None
        with f_col2:
            max_time_in = st.number_input("Max Cook Time (0 = No limit)", min_value=0, max_value=240, value=0, step=5)
            max_time = float(max_time_in) if max_time_in > 0 else None

        allergen_input = st.text_input("🚫 Exclude Allergens / Disliked Ingredients", placeholder="e.g. peanuts, shellfish, dairy, cilantro")
        excluded = [x.strip() for x in allergen_input.split(",") if x.strip()]

    constraints = None
    if max_cal or max_time or excluded:
        constraints = {
            "max_calories": max_cal,
            "max_cook_time_minutes": max_time,
            "required_dietary_tags": [],
            "excluded_ingredients": excluded,
        }

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    evaluate_clicked = st.button("✨ Evaluate & Score Recipe", type="primary", use_container_width=True)

    if evaluate_clicked:
        if not title.strip() or not ingredients.strip():
            st.warning("Please provide both a recipe name and a list of ingredients.")
            return

        with st.spinner("Analyzing flavor profile, nutrition, and predicted rating..."):
            raw_input = {
                "recipe_id": "user_recipe",
                "title": title,
                "ingredients": ingredients,
                "cook_time_minutes": cook_time,
                "calories": calories,
                "dietary_tags": user_diet,
            }
            result = adapter.run_single_recipe(raw_input, constraints=constraints)

        if not result.is_valid:
            st.error("Please double check your recipe inputs (all fields must have realistic values).")
            return

        if result.is_filtered_out:
            st.warning(f"⚠️ **Recipe Filtered Out:** {result.filter_reason}")
            return

        # Success: Render Clean Thematic Score Card
        score_val = result.score or 0.85
        score_pct = score_val * 100.0

        if score_val >= 0.85:
            stars = "★ ★ ★ ★ ★"
            verdict = "Exceptional Dish · Highly Recommended!"
            verdict_text = "This recipe features a harmonious balance of fresh aromatic ingredients that statistically correlates with rave 5-star reviews."
            circle_color = "#10b981"
        elif score_val >= 0.70:
            stars = "★ ★ ★ ★ ☆"
            verdict = "Delightful Quality Recipe"
            verdict_text = "Well-balanced flavors and solid ingredient synergy. Expected to be warmly received."
            circle_color = "#f59e0b"
        else:
            stars = "★ ★ ★ ☆ ☆"
            verdict = "Good Foundation · Room for Flair"
            verdict_text = "Consider elevating the flavor profile with fresh herbs, citrus zest, or aromatic spices for higher rating potential."
            circle_color = "#64748b"

        # Build dietary badges
        diet_badges_html = ""
        for tag in result.dietary_tags:
            tag_name = tag.replace("_", " ").title()
            icon = "🌱" if tag == "vegetarian" else ("🌿" if tag == "vegan" else ("🌾" if tag == "gluten_free" else "🏷️"))
            diet_badges_html += f'<span class="diet-pill diet-pill-green">{icon} {tag_name}</span>'

        if not diet_badges_html:
            diet_badges_html = '<span class="diet-pill">Standard Balanced</span>'

        prep_m = int(result.cook_time_minutes or 20)
        cal_m = int(result.calories or 350)
        tier_m = result.confidence_tier.split(' ')[0] if result.confidence_tier else "High"

        # Construct unindented single-block HTML to avoid Markdown indentation code-block parsing
        hero_html = (
            f'<div class="score-hero-card">'
            f'<div class="stars-rating">{stars}</div>'
            f'<div class="score-badge-circle" style="border-color: {circle_color};">'
            f'<div class="score-number">{score_pct:.0f}%</div>'
            f'<div class="score-unit">Quality Index</div>'
            f'</div>'
            f'<div class="verdict-title">{verdict}</div>'
            f'<div class="verdict-desc">{verdict_text}</div>'
            f'<div style="margin-top: 1rem; margin-bottom: 1rem;">{diet_badges_html}</div>'
            f'<div class="dish-specs">'
            f'<div class="spec-item"><div class="spec-label">Prep Time</div><div class="spec-value">⏱️ {prep_m}m</div></div>'
            f'<div class="spec-item"><div class="spec-label">Calorie Level</div><div class="spec-value">🔥 {cal_m} kcal</div></div>'
            f'<div class="spec-item"><div class="spec-label">Culinary Tier</div><div class="spec-value" style="color: {circle_color};">{tier_m}</div></div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(hero_html, unsafe_allow_html=True)

        # Flavor Insights & Chef's Notes
        flavor_html = (
            f'<div class="recipe-card">'
            f'<h3 style="font-family: \'Playfair Display\', serif; font-size: 1.25rem; margin-top: 0; margin-bottom: 0.75rem;">🌿 Flavor Profile & Key Ingredients</h3>'
            f'<p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 0;"><b>Parsed Ingredients:</b> {result.preprocessed_ingredients}</p>'
            f'</div>'
        )
        st.markdown(flavor_html, unsafe_allow_html=True)
