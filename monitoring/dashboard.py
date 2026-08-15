"""
Live monitoring dashboard for the Recipe Recommender MLOps pipeline.

Combines:
  - Live API health/metrics    (GET /health, GET /metrics on the FastAPI service)
  - Offline champion benchmark (models/champion_config.json)
  - Production validation      (monitoring/reports/production_validation.json, if present)
  - Drift / data-quality       (monitoring/reports/verify_alerts_summary.json)
  - Evidently per-scenario HTML reports (monitoring/reports/*.html)

Run from the repo root with the API + (optionally) the monitoring stack already up:

    pip install streamlit requests pandas
    streamlit run monitoring/dashboard.py

This intentionally stays outside docker-compose -- it's a demo/inspection tool,
not a production service, so running it locally against the already-running
stack keeps things simple.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config -- adjust if your paths/ports differ
# ---------------------------------------------------------------------------
# Defaults to localhost for running `streamlit run` on your machine.
# When running as the `dashboard` service in docker-compose, set
# API_URL=http://<api-service-name>:8000 as an environment variable so it
# reaches the API over the Docker network instead of localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000")
CHAMPION_CONFIG_PATH = Path("models/champion_config.json")
VERIFY_ALERTS_PATH = Path("monitoring/reports/verify_alerts_summary.json")
PRODUCTION_VALIDATION_PATH = Path("monitoring/reports/production_validation.json")
EVIDENTLY_REPORTS_DIR = Path("monitoring/reports")

st.set_page_config(page_title="Recipe Recommender Monitoring", layout="wide")
st.title("Recipe Recommender -- Monitoring Dashboard")
st.caption(f"API target: {API_URL}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        st.warning(f"Could not parse {path}: {e}")
        return None


def get_api(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=3)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def run_module(module: str, extra_args=None):
    """Run `python -m <module>` and stream result back to the UI."""
    cmd = ["python", "-m", module] + (extra_args or [])
    with st.spinner(f"Running `{' '.join(cmd)}` ..."):
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        st.success(f"{module} finished successfully.")
    else:
        st.error(f"{module} exited with code {result.returncode}")
    with st.expander(f"Output: {module}"):
        st.code(result.stdout + "\n" + result.stderr or "(no output)")


# ---------------------------------------------------------------------------
# Sidebar -- run controls
# ---------------------------------------------------------------------------
st.sidebar.header("Run monitoring steps")
st.sidebar.caption("Requires the API + MLflow to already be running.")

if st.sidebar.button("1. Build drift batches"):
    run_module("monitoring.drift_simulation")

if st.sidebar.button("2. Verify alerts (drift + quality, 5 scenarios)"):
    run_module("monitoring.verify_alerts")

if st.sidebar.button("3. Production validation (live API vs offline)"):
    run_module(
        "monitoring.production_validation",
        ["--api-url", API_URL, "--output", str(PRODUCTION_VALIDATION_PATH)],
    )

auto_refresh = st.sidebar.checkbox("Auto-refresh system metrics (5s)", value=False)
if st.sidebar.button("Refresh now") or auto_refresh:
    st.rerun()


# ---------------------------------------------------------------------------
# Panel 1: System health (live)
# ---------------------------------------------------------------------------
st.header("1. System health")

health, health_err = get_api("/health")
metrics, metrics_err = get_api("/metrics")

col1, col2, col3, col4 = st.columns(4)

if health:
    col1.metric("Model loaded", "Yes" if health.get("model_loaded") else "No")
    col2.metric("Model", health.get("model_name", "-"))
    col3.metric("Alias", health.get("alias", "-"))
else:
    st.error(f"Could not reach {API_URL}/health -- is the API running? ({health_err})")

if metrics:
    # Adjust these keys to match your actual /metrics response shape.
    col4.metric("Requests served", metrics.get("request_count", "-"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Error rate", f"{metrics.get('error_rate', 0):.2%}"
              if isinstance(metrics.get("error_rate"), (int, float)) else "-")
    m2.metric("p50 latency (ms)", metrics.get("p50_latency_ms", "-"))
    m3.metric("p95 latency (ms)", metrics.get("p95_latency_ms", "-"))
else:
    st.warning(f"Could not reach {API_URL}/metrics ({metrics_err})")


# ---------------------------------------------------------------------------
# Panel 2: Model performance -- offline champion vs live production validation
# ---------------------------------------------------------------------------
st.header("2. Model performance")

champion = load_json(CHAMPION_CONFIG_PATH)
prod_val = load_json(PRODUCTION_VALIDATION_PATH)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Offline benchmark (champion_config.json)")
    if champion:
        st.metric("Mean repeated-CV ROC-AUC", champion.get("mean_repeated_cv_roc_auc", "-"))
        st.metric("Median repeated-CV ROC-AUC", champion.get("median_repeated_cv_roc_auc", "-"))
        st.metric("Validation ROC-AUC", champion.get("validation_roc_auc", "-"))
    else:
        st.info("champion_config.json not found.")

with c2:
    st.subheader("Live production validation")
    if prod_val:
        st.metric("Live API ROC-AUC", prod_val.get("live_roc_auc", "-"))
        offline = champion.get("validation_roc_auc") if champion else None
        live = prod_val.get("live_roc_auc")
        if offline is not None and live is not None:
            delta = live - offline
            st.metric("Delta vs offline", f"{delta:+.4f}")
    else:
        st.info("No production validation run yet -- use the sidebar button.")


# ---------------------------------------------------------------------------
# Panel 3: Drift & data quality
# ---------------------------------------------------------------------------
st.header("3. Drift & data quality (verify_alerts)")

alerts = load_json(VERIFY_ALERTS_PATH)
if alerts:
    scenarios = alerts.get("scenarios", alerts)  # tolerate either shape
    if isinstance(scenarios, dict):
        rows = [{"scenario": k, **v} for k, v in scenarios.items()]
    else:
        rows = scenarios
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if EVIDENTLY_REPORTS_DIR.exists():
        html_reports = sorted(EVIDENTLY_REPORTS_DIR.glob("*.html"))
        if html_reports:
            st.caption("Evidently HTML reports:")
            for report in html_reports:
                st.markdown(f"- `{report.name}`")
else:
    st.info("verify_alerts_summary.json not found -- run step 2 in the sidebar.")


if auto_refresh:
    time.sleep(5)
    st.rerun()