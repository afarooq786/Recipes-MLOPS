"""
Live monitoring dashboard for the Recipe Recommender MLOps pipeline.

Combines:
  - Live API health/metrics    (GET /health, GET /metrics on the FastAPI service)
  - Offline champion benchmark (models/champion_config.json)
  - Production validation      (monitoring/reports/production_validation_report.json, if present)
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
# NOTE: monitoring/production_validation.py writes to
# "production_validation_report.json" (see its --output default / log line
# "Wrote report to monitoring/reports/production_validation_report.json").
# This used to point at "production_validation.json" (no "_report"), which
# never existed, so this whole panel silently showed nothing.
PRODUCTION_VALIDATION_PATH = Path("monitoring/reports/production_validation_report.json")
EVIDENTLY_REPORTS_DIR = Path("monitoring/reports")

st.set_page_config(page_title="Recipe Recommender Monitoring", layout="wide", page_icon="🍳")

# ---------------------------------------------------------------------------
# Light styling -- status pills, card-like metric containers, tighter header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e6e8eb;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-pass { background: #e6f4ea; color: #1e7e34; }
    .status-fail { background: #fbe7e9; color: #c0392b; }
    .status-warn { background: #fff4e5; color: #b8720b; }
    .section-caption { color: #6b7280; font-size: 0.9rem; margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍳 Recipe Recommender -- Monitoring Dashboard")
st.caption(f"API target: `{API_URL}`")


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


def pill(text: str, kind: str) -> str:
    """kind: 'pass' | 'fail' | 'warn'"""
    return f'<span class="status-pill status-{kind}">{text}</span>'


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
        st.code((result.stdout or "") + "\n" + (result.stderr or "") or "(no output)")


# ---------------------------------------------------------------------------
# Sidebar -- run controls
# ---------------------------------------------------------------------------
st.sidebar.header("Run monitoring steps")
st.sidebar.caption("Requires the API + MLflow to already be running.")

if st.sidebar.button("1. Build drift batches", use_container_width=True):
    run_module("monitoring.drift_simulation")

if st.sidebar.button("2. Verify alerts (drift + quality, 5 scenarios)", use_container_width=True):
    run_module("monitoring.verify_alerts")

if st.sidebar.button("3. Production validation (live API vs offline)", use_container_width=True):
    run_module(
        "monitoring.production_validation",
        ["--api-url", API_URL, "--output", str(PRODUCTION_VALIDATION_PATH)],
    )

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-refresh system metrics (5s)", value=False)
if st.sidebar.button("Refresh now", use_container_width=True) or auto_refresh:
    st.rerun()


# ---------------------------------------------------------------------------
# Panel 1: System health (live)
# ---------------------------------------------------------------------------
st.header("1. System health")

health, health_err = get_api("/health")
metrics, metrics_err = get_api("/metrics")

col1, col2, col3, col4 = st.columns(4)

if health:
    loaded = bool(health.get("model_loaded"))
    col1.metric("Model loaded", "Yes" if loaded else "No")
    col2.metric("Model", health.get("model_name", "-"))
    # Fixed: API returns "model_alias", not "alias".
    col3.metric("Alias", health.get("model_alias", "-"))
    col4.markdown(
        pill("Healthy", "pass") if loaded else pill("Model not loaded", "fail"),
        unsafe_allow_html=True,
    )
else:
    st.markdown(pill(f"Unreachable -- {health_err}", "fail"), unsafe_allow_html=True)

if metrics:
    st.markdown('<p class="section-caption">Request-level metrics from /metrics</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Requests served", metrics.get("request_count", "-"))
    err = metrics.get("error_rate")
    m2.metric("Error rate", f"{err:.2%}" if isinstance(err, (int, float)) else "-")
    m3.metric("p95 latency (ms)", metrics.get("p95_latency_ms", "-"))
    with st.expander("Raw /metrics response"):
        st.json(metrics)
else:
    st.info(f"Could not reach {API_URL}/metrics ({metrics_err})")


# ---------------------------------------------------------------------------
# Panel 2: Model performance -- offline champion vs live production validation
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Model performance")

champion = load_json(CHAMPION_CONFIG_PATH)
prod_val = load_json(PRODUCTION_VALIDATION_PATH)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Offline benchmark")
    st.caption("models/champion_config.json")
    if champion:
        st.metric("Mean repeated-CV ROC-AUC", champion.get("mean_repeated_cv_roc_auc", "-"))
        st.metric("Median repeated-CV ROC-AUC", champion.get("median_repeated_cv_roc_auc", "-"))
        st.metric("Validation ROC-AUC", champion.get("validation_roc_auc", "-"))
    else:
        st.info("champion_config.json not found.")

with c2:
    st.subheader("Live production validation")
    st.caption("monitoring/reports/production_validation_report.json")
    if prod_val:
        # Fixed: script's actual output key is "production_roc_auc", not
        # "live_roc_auc" -- and it already computes the delta for us as
        # "auc_delta", so no need to recompute it here.
        live = prod_val.get("production_roc_auc")
        offline = prod_val.get("offline_validation_roc_auc")
        delta = prod_val.get("auc_delta")
        passed = prod_val.get("passed")

        st.metric("Live API ROC-AUC", f"{live:.4f}" if isinstance(live, (int, float)) else "-")
        if delta is not None:
            st.metric("Delta vs offline", f"{delta:+.4f}", delta=f"{delta:+.4f}")
        if passed is not None:
            st.markdown(pill("Passed", "pass") if passed else pill("Failed", "fail"), unsafe_allow_html=True)

        lat1, lat2 = st.columns(2)
        lat1.metric("Mean batch latency (ms)", round(prod_val.get("mean_batch_latency_ms", 0), 1))
        lat2.metric("Max batch latency (ms)", round(prod_val.get("max_batch_latency_ms", 0), 1))
    else:
        st.info("No production validation run yet -- use the sidebar button.")


# ---------------------------------------------------------------------------
# Panel 3: Drift & data quality
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Drift & data quality (verify_alerts)")

alerts = load_json(VERIFY_ALERTS_PATH)
if alerts:
    # Fixed: verify_alerts.py writes a JSON *list* of per-scenario dicts
    # (json.dump(results, ...) where results = [...]), not a dict with a
    # "scenarios" key -- calling .get() on that list raised AttributeError.
    rows = alerts if isinstance(alerts, list) else alerts.get("scenarios", [])

    if rows:
        df = pd.DataFrame(rows)

        n_correct = int(df["correctly_classified"].sum())
        n_total = len(df)
        st.markdown(
            pill(f"{n_correct} / {n_total} scenarios correctly classified", "pass" if n_correct == n_total else "warn"),
            unsafe_allow_html=True,
        )
        st.write("")

        display_df = df.rename(
            columns={
                "scenario": "Scenario",
                "expected_anomaly": "Expected anomaly",
                "detected_anomaly": "Detected anomaly",
                "correctly_classified": "Correct",
                "quality_check_issues": "Quality issues",
                "evidently_dataset_drift_detected": "Evidently drift",
            }
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Correct": st.column_config.CheckboxColumn(),
                "Expected anomaly": st.column_config.CheckboxColumn(),
                "Detected anomaly": st.column_config.CheckboxColumn(),
                "Evidently drift": st.column_config.CheckboxColumn(),
            },
        )

        if EVIDENTLY_REPORTS_DIR.exists():
            html_reports = sorted(EVIDENTLY_REPORTS_DIR.glob("*.html"))
            if html_reports:
                st.subheader("Evidently drift reports")
                report_names = [r.stem.replace("drift_report_", "") for r in html_reports]
                choice = st.selectbox("View a scenario's full Evidently report:", report_names)
                selected_path = EVIDENTLY_REPORTS_DIR / f"drift_report_{choice}.html"
                with st.expander(f"drift_report_{choice}.html", expanded=True):
                    st.components.v1.html(selected_path.read_text(), height=600, scrolling=True)
    else:
        st.info("verify_alerts_summary.json was empty.")
else:
    st.info("verify_alerts_summary.json not found -- run step 2 in the sidebar.")


if auto_refresh:
    time.sleep(5)
    st.rerun()