"""
webapp/components/pipeline_inspector_view.py — Interactive visual MLOps pipeline architecture inspector.
"""

import streamlit as st
from webapp.pipeline_adapter import PipelineAdapter


def render_pipeline_inspector_view(adapter: PipelineAdapter):
    """Renders the end-to-end pipeline architecture visualizer."""
    st.subheader("MLOps Pipeline Architecture Inspector")
    st.caption("Inspect the data transformations and contracts enforced at each stage of the recipe lifecycle.")

    stages = [
        {
            "num": "01",
            "name": "Data Ingestion & Versioning",
            "tool": "KaggleHub + DVC + GCP",
            "desc": "Pulls raw recipe archives, records content hashes into dvc.yaml/dvc.lock for 100% deterministic reproducibility.",
            "inputs": "Raw Kaggle dataset (recipes.csv)",
            "outputs": "Versioned raw storage in data/raw/",
        },
        {
            "num": "02",
            "name": "Schema & Range Validation",
            "tool": "Pandera DataFrameSchema",
            "desc": "Validates required columns, types, allowed ratings (0–5), and positive servings/cooking times before downstream stages.",
            "inputs": "data/raw/recipes.csv",
            "outputs": "data/validation_report.json (Non-zero exit if critical failure)",
        },
        {
            "num": "03",
            "name": "Stratified Splitting & Deduplication",
            "tool": "Scikit-Learn StratifiedKFold",
            "desc": "Deduplicates recipe signatures to prevent train/val/test data leakage. Binarizes target: label = 1 if rating >= 4.",
            "inputs": "data/raw/recipes.csv",
            "outputs": "data/processed/splits/{train, val, test}.csv",
        },
        {
            "num": "04",
            "name": "Preprocessing & Text Cleaning",
            "tool": "Regex + String Tokenizer",
            "desc": "Lowercases ingredient lines, strips noise/punctuation, parses prep/cook durations into clean numeric minutes.",
            "inputs": "data/processed/splits/*.csv",
            "outputs": "data/processed/clean/{train, val, test}.csv",
        },
        {
            "num": "05",
            "name": "Feature Engineering & Heuristics",
            "tool": "Nutritional & Dietary Extractors",
            "desc": "Extracts dietary tags (vegetarian, vegan, gluten_free), allergen markers, protein categories, and time buckets.",
            "inputs": "data/processed/clean/*.csv",
            "outputs": "data/processed/features/*.csv + feature_manifest.json",
        },
        {
            "num": "06",
            "name": "Constraint Filtering Layer",
            "tool": "FastAPI / api.filters",
            "desc": "Deterministic pre-inference gate eliminating recipes violating user calorie caps, cook times, or allergen restrictions.",
            "inputs": "CandidateRecipe objects + FilterConstraints",
            "outputs": "Surviving candidates to model scoring",
        },
        {
            "num": "07",
            "name": "Champion Model Scoring",
            "tool": "Char-Logistic + Word-SVM Rank Ensemble",
            "desc": "Combines character n-gram Logistic Regression and word n-gram Linear SVM. Selected via 50-fold repeated CV.",
            "inputs": "Clean ingredients_parsed string",
            "outputs": "Ranked predictions (Probability rating >= 4)",
        },
        {
            "num": "08",
            "name": "Production Monitoring & Drift",
            "tool": "Evidently + Rule-based Data Quality",
            "desc": "Monitors in-flight API latency, schema integrity, and statistical distribution drift against baseline distributions.",
            "inputs": "Live API traffic & synthetic drift batches",
            "outputs": "monitoring/reports/verify_alerts_summary.json",
        },
    ]

    for stage in stages:
        st.markdown(
            f"""
            <div class="pipeline-node">
                <div class="pipeline-node-header">
                    <span><span style="color: #6b7280; margin-right: 0.5rem;">{stage['num']}</span> {stage['name']}</span>
                    <span class="badge-pill badge-neutral">{stage['tool']}</span>
                </div>
                <div style="font-size: 0.85rem; margin-bottom: 0.4rem;">{stage['desc']}</div>
                <div class="pipeline-node-detail">
                    <b>Input:</b> <code>{stage['inputs']}</code> &nbsp;·&nbsp; <b>Output:</b> <code>{stage['outputs']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
