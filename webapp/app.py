"""
webapp/app.py — Main Entry Point for SavorAI Culinary Studio.

A consumer-focused, minimalist recipe quality evaluator and menu creator.
Base theme: White & Grey with culinary micro-animations.

Run locally:
    streamlit run webapp/app.py
"""

from __future__ import annotations

import os
import streamlit as st

from webapp.styles import apply_custom_styles
from webapp.pipeline_adapter import PipelineAdapter
from webapp.components.header import render_header
from webapp.components.single_recipe_view import render_single_recipe_view
from webapp.components.batch_upload_view import render_batch_upload_view

# Configure Streamlit page
st.set_page_config(
    page_title="SavorAI · Recipe Studio",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom white & grey theme with CSS animations
apply_custom_styles()

# API Configuration (seamlessly defaults to localhost or standalone)
api_url_env = os.environ.get("API_URL", "http://localhost:8000")
adapter = PipelineAdapter(api_url=api_url_env)

# Sidebar: Consumer culinary helper
with st.sidebar:
    sidebar_brand = (
        '<div style="padding: 0.5rem 0 1rem 0;">'
        '<div style="font-size: 1.5rem; font-weight: 800; color: #0f172a;">🍳 SavorAI</div>'
        '<div style="font-size: 0.85rem; color: #64748b;">Smart Culinary Quality Assistant</div>'
        '</div>'
    )
    st.markdown(sidebar_brand, unsafe_allow_html=True)

    st.markdown(
        """### 👨‍🍳 How Quality is Evaluated
- **Flavor Synergy:** Evaluates ingredient aromatics, fresh herbs, citrus, and seasoning harmony.
- **Dietary Accuracy:** Automatically categorizes vegetarian, vegan, and gluten-free traits.
- **Rating Prediction:** Estimates likelihood of receiving **4.5+ star** reviews based on recipe attributes.
"""
    )

    st.markdown("---")
    st.markdown(
        """### 💡 Chef's Tips
- **Aromatics:** Garlic, shallots, and fresh herbs boost rating likelihood.
- **Balanced Acid:** A splash of lemon or vinegar elevates heavy dishes.
- **Freshness:** Fresh produce scores higher in diner satisfaction.
"""
    )

# Render main branded header
render_header()

# Main Consumer Tabs
tab1, tab2 = st.tabs([
    "✨ Recipe Evaluator",
    "📋 Menu & Collection Ranker",
])

with tab1:
    render_single_recipe_view(adapter)

with tab2:
    render_batch_upload_view(adapter)
