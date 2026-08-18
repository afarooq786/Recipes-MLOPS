"""
webapp/components/batch_upload_view.py — Built-in In-App Recipe Builder & Menu Ranker.
"""

from pathlib import Path
from typing import Dict, Any, List
import io
import pandas as pd
import streamlit as st
from webapp.pipeline_adapter import PipelineAdapter

DEFAULT_TEMPLATE_ROWS = [
    {
        "Dish Name": "Lemon Herb Grilled Salmon",
        "Ingredients": "wild salmon fillet, fresh garlic, olive oil, lemon juice, rosemary, sea salt, cracked black pepper",
        "Cook Time (mins)": 20,
        "Calories (kcal)": 480,
        "Dietary Tags": "gluten_free, keto",
    },
    {
        "Dish Name": "Mediterranean Chickpea Salad",
        "Ingredients": "chickpeas, cucumber, cherry tomatoes, kalamata olives, red onion, parsley, extra virgin olive oil, feta cheese",
        "Cook Time (mins)": 15,
        "Calories (kcal)": 320,
        "Dietary Tags": "vegetarian, gluten_free",
    },
    {
        "Dish Name": "Creamy Tuscan Garlic Pasta",
        "Ingredients": "fettuccine, heavy cream, parmesan cheese, minced garlic, butter, baby spinach, sun-dried tomatoes",
        "Cook Time (mins)": 25,
        "Calories (kcal)": 580,
        "Dietary Tags": "vegetarian",
    },
    {
        "Dish Name": "Triple Bacon Cheddar Burger",
        "Ingredients": "ground beef patty, smoked bacon, cheddar cheese, brioche bun, mayonnaise, barbecue sauce",
        "Cook Time (mins)": 30,
        "Calories (kcal)": 1050,
        "Dietary Tags": "",
    },
    {
        "Dish Name": "Spicy Sesame Tofu Stir Fry",
        "Ingredients": "tofu, broccoli florets, bell peppers, snap peas, sesame oil, tamari soy sauce, ginger, garlic, scallions",
        "Cook Time (mins)": 25,
        "Calories (kcal)": 360,
        "Dietary Tags": "vegan, vegetarian, gluten_free",
    },
    {
        "Dish Name": "Rosemary Garlic Ribeye Steak",
        "Ingredients": "prime ribeye steak, unsalted butter, rosemary, thyme, crushed garlic, sea salt, black pepper",
        "Cook Time (mins)": 20,
        "Calories (kcal)": 640,
        "Dietary Tags": "gluten_free, keto",
    },
]


def render_batch_upload_view(adapter: PipelineAdapter):
    """Renders built-in interactive template editor and file upload menu ranker."""
    header_html = (
        '<div style="margin-bottom: 1.25rem;">'
        '<h2 style="font-family: \'Playfair Display\', serif; font-size: 1.6rem; margin-bottom: 0.25rem;">📋 Menu & Recipe Collection Ranker</h2>'
        '<p style="color: #64748b; font-size: 0.95rem; margin: 0;">Use the built-in interactive menu editor or upload a recipe file to score, filter, and curate your top culinary creations.</p>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Mode Selector: Built-in Editor vs File Upload
    mode = st.radio(
        "Select Creation Mode",
        options=["✍️ Built-in Interactive Menu Template", "📂 Upload File (CSV / JSON)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    records: List[Dict[str, Any]] = []

    if mode == "✍️ Built-in Interactive Menu Template":
        st.markdown(
            '<div style="margin-top: 0.5rem; margin-bottom: 0.5rem;"><h3 style="font-family: \'Playfair Display\', serif; font-size: 1.2rem;">📝 Built-in Recipe Template & Editor</h3><p style="color: #64748b; font-size: 0.88rem; margin: 0;">Edit dishes below directly, add new rows by clicking the bottom row, or customize ingredients:</p></div>',
            unsafe_allow_html=True,
        )

        # Initialize session state for template data if not present
        if "template_df" not in st.session_state:
            st.session_state["template_df"] = pd.DataFrame(DEFAULT_TEMPLATE_ROWS)

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            if st.button("🔄 Reset to Chef's Starter Template", use_container_width=True):
                st.session_state["template_df"] = pd.DataFrame(DEFAULT_TEMPLATE_ROWS)
                st.rerun()
        with col_t2:
            if st.button("➕ Start with Blank Rows", use_container_width=True):
                st.session_state["template_df"] = pd.DataFrame([
                    {"Dish Name": "", "Ingredients": "", "Cook Time (mins)": 20, "Calories (kcal)": 400, "Dietary Tags": ""},
                    {"Dish Name": "", "Ingredients": "", "Cook Time (mins)": 15, "Calories (kcal)": 350, "Dietary Tags": ""},
                ])
                st.rerun()

        # Editable table built right into the app
        edited_df = st.data_editor(
            st.session_state["template_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Dish Name": st.column_config.TextColumn("🍽️ Dish Name", required=True, width="medium"),
                "Ingredients": st.column_config.TextColumn("🛒 Ingredients", required=True, width="large"),
                "Cook Time (mins)": st.column_config.NumberColumn("⏱️ Time (min)", min_value=1, max_value=360, default=20),
                "Calories (kcal)": st.column_config.NumberColumn("🔥 Calories", min_value=10, max_value=3000, default=400),
                "Dietary Tags": st.column_config.TextColumn("🏷️ Dietary Tags", width="medium", help="e.g. vegetarian, vegan, gluten_free, keto"),
            },
            hide_index=True,
        )
        st.session_state["template_df"] = edited_df

        # Convert edited table to records
        for idx, row in edited_df.iterrows():
            name = str(row.get("Dish Name", "")).strip()
            ings = str(row.get("Ingredients", "")).strip()
            if name or ings:
                tags_raw = str(row.get("Dietary Tags", ""))
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                records.append({
                    "recipe_id": f"item_{idx+1}",
                    "title": name or f"Dish #{idx+1}",
                    "ingredients": ings,
                    "cook_time_minutes": row.get("Cook Time (mins)", 20),
                    "calories": row.get("Calories (kcal)", 400),
                    "dietary_tags": tags,
                })

    else:
        # File Upload Mode
        col_up, col_actions = st.columns([1.3, 0.7])
        with col_up:
            uploaded_file = st.file_uploader(
                "📂 Upload Recipes File (CSV or JSON)",
                type=["csv", "json"],
                help="Upload a CSV with recipe names, ingredients, and optional calories/cook times.",
            )
        with col_actions:
            st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
            load_curated = st.button("🍴 Load Curated Menu (20 Dishes)", use_container_width=True)

        if uploaded_file is not None:
            try:
                records = adapter.parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
                st.success(f"Loaded {len(records)} dishes from '{uploaded_file.name}'.")
            except Exception as exc:
                st.error(f"Could not parse file: {exc}")
                return
        elif load_curated:
            sample_path = Path("data/processed/clean/test.csv")
            if sample_path.exists():
                df = pd.read_csv(sample_path).head(20)
                for idx, row in df.iterrows():
                    records.append({
                        "recipe_id": f"rec_{idx+1}",
                        "title": str(row.get("recipe_name", f"Curated Dish #{idx+1}")),
                        "ingredients": str(row.get("ingredients_parsed") or row.get("ingredients", "")),
                        "calories": row.get("calories", 380),
                        "cook_time_minutes": row.get("cook_time_minutes", 25),
                        "dietary_tags": [],
                    })
                st.info(f"Loaded {len(records)} curated tasting menu dishes.")

    if not records:
        st.info("Enter dishes into the built-in template above to evaluate your menu.")
        return

    # User Dining & Dietary Preferences
    pref_header_html = (
        '<div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">'
        '<h3 style="font-family: \'Playfair Display\', serif; font-size: 1.25rem; margin-bottom: 0.2rem;">🎯 Dietary & Calorie Preferences</h3>'
        '</div>'
    )
    st.markdown(pref_header_html, unsafe_allow_html=True)

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        b_max_cal = st.number_input("Max Calories (0 = Any)", min_value=0, max_value=2500, value=0, step=50, key="bm_cal")
    with b_col2:
        b_max_time = st.number_input("Max Time (0 = Any)", min_value=0, max_value=240, value=0, step=5, key="bm_time")
    with b_col3:
        b_req_tags = st.multiselect(
            "Require Diet",
            options=["vegetarian", "vegan", "gluten_free", "keto"],
            default=[],
            format_func=lambda x: x.replace("_", " ").title(),
            key="bm_tags",
        )
    with b_col4:
        top_k = st.number_input("Top Recommendations", min_value=1, max_value=max(1, len(records)), value=min(10, max(1, len(records))))

    b_excluded = st.text_input("🚫 Exclude Allergens (e.g. peanuts, shellfish, dairy)", placeholder="peanuts, shellfish", key="bm_ex")
    excluded_list = [x.strip() for x in b_excluded.split(",") if x.strip()]

    constraints = None
    if (b_max_cal > 0) or (b_max_time > 0) or b_req_tags or excluded_list:
        constraints = {
            "max_calories": float(b_max_cal) if b_max_cal > 0 else None,
            "max_cook_time_minutes": float(b_max_time) if b_max_time > 0 else None,
            "required_dietary_tags": b_req_tags,
            "excluded_ingredients": excluded_list,
        }

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    rank_button = st.button("🏆 Rank Menu by Recipe Quality", type="primary", use_container_width=True)

    if rank_button:
        with st.spinner(f"Evaluating and ranking {len(records)} dishes..."):
            processed_candidates = []
            for r in records:
                is_valid, _ = adapter.validate_recipe(r)
                if not is_valid:
                    continue
                pre = adapter.preprocess_recipe(r)
                feats = adapter.derive_features(pre, r.get("dietary_tags"))
                processed_candidates.append({
                    "recipe_id": r.get("recipe_id"),
                    "title": pre["title"],
                    "ingredients_parsed": pre["ingredients_parsed"],
                    "calories": pre["calories"],
                    "cook_time_minutes": pre["cook_time_minutes"],
                    "dietary_tags": feats["dietary_tags"],
                })

            score_out = adapter.score_candidates(
                processed_candidates,
                constraints=constraints,
                top_k=int(top_k),
            )

        results = score_out.get("results", [])
        total = score_out.get("total_candidates", len(records))

        # Overview metric cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="recipe-card" style="text-align: center;"><div class="spec-label">Total Dishes Analyzed</div><div class="spec-value" style="font-size: 1.75rem;">🍽️ {total}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="recipe-card" style="text-align: center;"><div class="spec-label">Matching Preferences</div><div class="spec-value" style="font-size: 1.75rem; color: #10b981;">✅ {len(results)}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            top_name = results[0]['title'] if results else "None"
            st.markdown(
                f'<div class="recipe-card" style="text-align: center;"><div class="spec-label">#1 Top Ranked Dish</div><div class="spec-value" style="font-size: 1.15rem; color: #f59e0b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">🥇 {top_name}</div></div>',
                unsafe_allow_html=True,
            )

        if not results:
            st.warning("No dishes met the selected dietary and calorie constraints.")
            return

        # Render Top Picks
        st.markdown(
            '<div style="margin-top: 1rem; margin-bottom: 0.5rem;"><h3 style="font-family: \'Playfair Display\', serif; font-size: 1.35rem;">🌟 Curated Top Recommendations</h3></div>',
            unsafe_allow_html=True,
        )

        rows = []
        for item in results:
            rank = item.get("rank", 1)
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
            score_pct = item.get("score", 0.0) * 100.0

            rows.append({
                "Rank": f"{medal} #{rank}",
                "Dish Name": item.get("title"),
                "Quality Index": f"{score_pct:.1f}%",
                "Rating Prediction": "★★★★★ 5-Star" if score_pct >= 85 else "★★★★☆ 4-Star",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Download CSV
        export_df = pd.DataFrame(rows)
        csv_io = io.StringIO()
        export_df.to_csv(csv_io, index=False)
        st.download_button(
            label="📥 Download Curated Menu (CSV)",
            data=csv_io.getvalue(),
            file_name="curated_recipe_menu.csv",
            mime="text/csv",
            type="primary",
        )
