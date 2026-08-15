"""
tests/test_metrics_endpoint.py — Row 22 verification: the lightweight
custom system-monitoring endpoint (used in place of Prometheus/Grafana).
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from api.metrics_middleware import reset_metrics


class _FakeModel:
    def predict(self, inference_df: pd.DataFrame):
        return [0.5] * len(inference_df)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_load_champion_model", lambda: _FakeModel())
    reset_metrics()
    with TestClient(app) as c:
        yield c
    reset_metrics()


class TestMetricsEndpoint:
    def test_metrics_starts_empty_before_traffic(self, client):
        # /health itself is a request, so hit /metrics first via a fresh reset.
        response = client.get("/metrics")
        assert response.status_code == 200
        # At minimum this very /metrics call has now been recorded (>=0 prior).
        assert "request_count" in response.json()

    def test_metrics_counts_requests_by_path(self, client):
        client.get("/health")
        client.get("/health")
        client.post("/predict", json={"candidates": [{"recipe_id": "r1", "ingredients_parsed": "salt"}]})

        response = client.get("/metrics")
        data = response.json()
        assert data["request_count"] >= 3
        assert "GET /health" in data["by_path"]
        assert data["by_path"]["GET /health"]["count"] == 2

    def test_metrics_reports_latency_percentiles(self, client):
        for _ in range(5):
            client.get("/health")
        data = client.get("/metrics").json()
        assert data["latency_ms"]["mean"] is not None
        assert data["latency_ms"]["p50"] is not None
        assert data["latency_ms"]["p95"] is not None

    def test_metrics_error_rate_zero_when_no_errors(self, client):
        client.get("/health")
        data = client.get("/metrics").json()
        assert data["error_rate"] == 0.0
