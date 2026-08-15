"""
api/metrics_middleware.py — Row 22: System Monitoring (lightweight custom
implementation, used in place of a full Prometheus/Grafana stack).

Standing up Prometheus + Grafana as separate containers is a lot of
infrastructure for what this project needs to demonstrate: that
latency, request volume, error rate, and health are being tracked and
are visible somewhere. The MLOps guidelines explicitly list
"Prometheus/Grafana... or a custom dashboard" as interchangeable
options, so this implements the same signal with a single in-process
FastAPI middleware.

- Every request's method, path, status code, and latency is recorded
  in a small in-memory ring buffer (bounded, so it can't leak memory
  in a long-running container).
- GET /metrics returns a JSON summary: request counts by path/status,
  error rate, and p50/p95/mean latency -- the same signals a
  Prometheus + Grafana panel would show, just served directly.

This is intentionally simple: it is not a replacement for Prometheus
in a real production deployment (no persistence across restarts, no
multi-instance aggregation), but it satisfies the project's monitoring
requirement without the operational overhead a 10-day class project
can't productively spend time on.
"""

from __future__ import annotations

import time
from collections import deque
from statistics import mean
from typing import Deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_MAX_RECORDS = 2000
_records: Deque[dict] = deque(maxlen=_MAX_RECORDS)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _records.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": elapsed_ms,
                    "timestamp": time.time(),
                }
            )
        return response


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


def metrics_summary() -> dict:
    """Build the same summary signals a Prometheus/Grafana panel would show."""
    records = list(_records)
    if not records:
        return {
            "request_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "latency_ms": {"mean": None, "p50": None, "p95": None},
            "by_path": {},
        }

    latencies = [r["latency_ms"] for r in records]
    errors = [r for r in records if r["status_code"] >= 500]

    by_path: dict[str, dict] = {}
    for r in records:
        key = f"{r['method']} {r['path']}"
        entry = by_path.setdefault(key, {"count": 0, "error_count": 0, "latencies": []})
        entry["count"] += 1
        entry["latencies"].append(r["latency_ms"])
        if r["status_code"] >= 500:
            entry["error_count"] += 1

    for entry in by_path.values():
        entry["mean_latency_ms"] = round(mean(entry["latencies"]), 2)
        entry["p95_latency_ms"] = round(_percentile(entry["latencies"], 0.95), 2)
        del entry["latencies"]

    return {
        "request_count": len(records),
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(records), 4),
        "latency_ms": {
            "mean": round(mean(latencies), 2),
            "p50": round(_percentile(latencies, 0.5), 2),
            "p95": round(_percentile(latencies, 0.95), 2),
        },
        "by_path": by_path,
    }


def metrics_summary_response() -> JSONResponse:
    return JSONResponse(metrics_summary())


def reset_metrics() -> None:
    """Test-only helper to clear recorded metrics between test runs."""
    _records.clear()
