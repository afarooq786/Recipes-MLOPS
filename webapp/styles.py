"""
webapp/styles.py — White & Grey Base Thematic Styling with Culinary Animations.
"""

THEMATIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

/* Force Light Theme Colors Regardless of Browser Mode */
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-card: #ffffff;
    --border-subtle: #e2e8f0;
    --border-focus: #94a3b8;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --accent-gold: #d97706;
    --accent-emerald: #059669;
    --accent-rose: #e11d48;
    --accent-indigo: #4338ca;
    --shadow-soft: 0 4px 20px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
    --shadow-hover: 0 10px 25px -3px rgba(15, 23, 42, 0.1), 0 4px 10px -2px rgba(15, 23, 42, 0.04);
}

/* Base Canvas & Global Text Contrast */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Force All Headings, Paragraphs, and Labels to High Contrast Dark Text */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 700 !important;
}

p, span, div, li, label, strong, b {
    color: #0f172a !important;
}

.stMarkdown p, .stMarkdown span {
    color: #334155 !important;
}

/* Sidebar Explicit High Contrast */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: #334155 !important;
}

/* Streamlit Input Labels & Input Fields */
label[data-testid="stWidgetLabel"] p,
.stTextInput label, .stTextArea label, .stSelectbox label,
.stMultiSelect label, .stNumberInput label, .stRadio label {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

input, textarea, select {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

input:focus, textarea:focus {
    border-color: #64748b !important;
    box-shadow: 0 0 0 2px rgba(100, 116, 139, 0.15) !important;
}

/* Selectbox & Multiselect Options */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] * {
    color: #0f172a !important;
}

div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}

li[role="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}

span[data-baseweb="tag"] {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
}

span[data-baseweb="tag"] * {
    color: #1e293b !important;
}

/* Expanders */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] * {
    color: #0f172a !important;
}

/* Keyframe Animations */
@keyframes floatAnimation {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-6px) rotate(2deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.25); }
    70% { box-shadow: 0 0 0 12px rgba(5, 150, 105, 0); }
    100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
}

@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Floating Culinary Icon */
.floating-food-icon {
    display: inline-block;
    animation: floatAnimation 3.5s ease-in-out infinite;
    font-size: 2.25rem;
    margin-right: 0.75rem;
    vertical-align: middle;
}

/* Culinary Header */
.culinary-header {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.75rem 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: var(--shadow-soft) !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    position: relative !important;
    overflow: hidden !important;
}

.culinary-header::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #d97706, #059669, #4f46e5);
}

.brand-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.01em !important;
}

.brand-subtitle {
    font-size: 0.95rem !important;
    color: #475569 !important;
    margin-top: 0.35rem !important;
    font-weight: 400 !important;
}

/* Polished Recipe Card */
.recipe-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.75rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: var(--shadow-soft) !important;
    animation: slideUpFade 0.4s ease-out;
}

/* Result Hero Score Card */
.score-hero-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: var(--shadow-hover) !important;
    text-align: center !important;
    position: relative !important;
    animation: slideUpFade 0.5s ease-out;
}

.score-badge-circle {
    display: inline-flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    width: 140px !important;
    height: 140px !important;
    border-radius: 50% !important;
    background: #f8fafc !important;
    border: 3px solid #059669 !important;
    margin: 0.5rem auto 1rem auto !important;
    animation: pulseGlow 2.5s infinite;
}

.score-number {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    line-height: 1 !important;
}

.score-unit {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-top: 0.2rem !important;
}

.stars-rating {
    font-size: 1.4rem !important;
    color: #d97706 !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: 2px !important;
}

.verdict-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 0.35rem !important;
}

.verdict-desc {
    font-size: 0.95rem !important;
    color: #334155 !important;
    max-width: 550px !important;
    margin: 0 auto 1.25rem auto !important;
    line-height: 1.5 !important;
}

/* Feature & Metric Pills */
.diet-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.35rem !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: 9999px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    margin-right: 0.45rem !important;
    margin-bottom: 0.45rem !important;
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

.diet-pill-green {
    background: #ecfdf5 !important;
    color: #065f46 !important;
    border-color: #a7f3d0 !important;
}

.diet-pill-gold {
    background: #fffbeb !important;
    color: #92400e !important;
    border-color: #fde68a !important;
}

/* Dish Summary Attribute Grid */
.dish-specs {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)) !important;
    gap: 1rem !important;
    margin-top: 1.5rem !important;
    padding-top: 1.25rem !important;
    border-top: 1px solid #e2e8f0 !important;
}

.spec-item {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 0.85rem !important;
    text-align: center !important;
}

.spec-label {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    color: #475569 !important;
    letter-spacing: 0.05em !important;
}

.spec-value {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin-top: 0.2rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    color: #0f172a !important;
}

.stButton > button[kind="primary"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
}

.stButton > button[kind="primary"] * {
    color: #ffffff !important;
}

/* Clean Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem !important;
    background-color: transparent !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding-bottom: 0.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #64748b !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
}

.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    background-color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}

/* Dataframe / Data Editor Text Contrast */
.stDataFrame, [data-testid="stDataFrame"], [data-testid="stTable"] {
    background-color: #ffffff !important;
}

[data-testid="stDataFrame"] * {
    color: #0f172a !important;
}

/* Hide Streamlit default clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


def apply_custom_styles():
    """Apply white/grey culinary thematic styling with animations."""
    import streamlit as st
    st.markdown(THEMATIC_CSS, unsafe_allow_html=True)
