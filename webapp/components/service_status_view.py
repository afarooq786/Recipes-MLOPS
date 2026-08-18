"""
webapp/components/service_status_view.py — Live API Health & Telemetry Inspector.
"""

import requests
import streamlit as st
from webapp.pipeline_adapter import PipelineAdapter


def render_service_status_view(adapter: PipelineAdapter):
    """Renders the live service health, latency percentiles, and hot-reload controls."""
    st.subheader("Service Health & Real-time Telemetry")
    st.caption("Inspect live status and monitoring metrics from the FastAPI inference container.")

    health = adapter.check_api_health()
    metrics = adapter.get_api_metrics()

    c1, c2, c3, c4 = st.columns(4)

    status_val = health.get("status", "unavailable").upper()
    model_loaded = health.get("model_loaded", False)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">API Service Status</div>
                <div class="metric-value">{status_val}</div>
                <div style="font-size: 0.8rem; color: {'#22c55e' if status_val=='OK' else '#ef4444'}; margin-top: 0.25rem;">
                    {'Active & Responding' if status_val=='OK' else 'Offline / Standalone'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Registry Alias</div>
                <div class="metric-value">{health.get("model_alias", "champion")}</div>
                <div style="font-size: 0.8rem; color: #888888; margin-top: 0.25rem;">
                    Target: {health.get("model_name", "recipe-recommender")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        req_count = metrics.get("request_count", 0) if metrics else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Requests Handled</div>
                <div class="metric-value">{req_count}</div>
                <div style="font-size: 0.8rem; color: #888888; margin-top: 0.25rem;">Live API Traffic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        p95 = metrics.get("latency_p95_ms", 0.0) if metrics else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">P95 Inference Latency</div>
                <div class="metric-value">{p95:.1f} ms</div>
                <div style="font-size: 0.8rem; color: #888888; margin-top: 0.25rem;">p50: {metrics.get('latency_p50_ms', 0.0):.1f} ms if metrics else 0.0 ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Service Management & Actions")
    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        if st.button("Hot-Reload Champion Model from Registry", use_container_width=True):
            try:
                resp = requests.post(f"{adapter.api_url}/reload-model", timeout=5.0)
                if resp.status_code == 200:
                    st.success("Successfully triggered hot-reload of champion model from MLflow.")
                else:
                    st.warning(f"Reload responded with HTTP {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Could not connect to API for model reload: {exc}")

    with m_col2:
        if st.button("Refresh Telemetry Data", use_container_width=True):
            st.rerun()

    if metrics:
        st.markdown("### Raw Metrics Telemetry Payload (`/metrics`)")
        st.json(metrics)
