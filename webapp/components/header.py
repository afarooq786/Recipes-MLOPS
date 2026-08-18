"""
webapp/components/header.py — Thematic White & Grey culinary header with micro-animations.
"""

import streamlit as st


def render_header():
    """Renders the top culinary brand header bar with food animations."""
    header_html = (
        '<div class="culinary-header">'
        '<div>'
        '<div style="display: flex; align-items: center;">'
        '<span class="floating-food-icon">🍳</span>'
        '<div>'
        '<h1 class="brand-title">SavorAI Studio</h1>'
        '<div class="brand-subtitle">Smart Recipe Evaluator, Culinary Quality Predictor & Menu Creator</div>'
        '</div>'
        '</div>'
        '</div>'
        '<div style="display: flex; gap: 0.5rem; align-items: center;">'
        '<span class="diet-pill diet-pill-green">✨ AI Quality Assured</span>'
        '<span class="diet-pill diet-pill-gold">🥗 Dietary Smart</span>'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
