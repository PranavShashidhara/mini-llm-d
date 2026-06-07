"""
router/metrics.py

Prometheus metric definitions and sliding-window TTFT/TPOT tracking
for the mini-llmd router.

All metrics are exported at GET /metrics in Prometheus text format.
"""

from __future__ import annotations

import time
from collections import deque

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Prometheus metric definitions
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

# Router-level counters
requests_total = Counter(
    "router_requests_total",
    "Total requests received by the router",
    ["routing_mode", "cache_hit"],
    registry=REGISTRY,
)

requests_failed = Counter(
    "router_requests_failed_total",
    "Total failed / circuit-broken requests",
    registry=REGISTRY,
)

# Latency histograms (seconds)
ttft_histogram = Histogram(
    "router_ttft_seconds",
    "Time to first token, as measured by the router",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
    registry=REGISTRY,
)

tpot_histogram = Histogram(
    "router_tpot_seconds",
    "Time per output token, as measured by the router",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    registry=REGISTRY,
)

e2e_latency_histogram = Histogram(
    "router_e2e_latency_seconds",
    "End-to-end request latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

# Per-pod gauges
pod_queue_depth = Gauge(
    "router_pod_queue_depth",
    "Queue depth observed on each pod",
    ["pod"],
    registry=REGISTRY,
)

pod_tokens_per_sec = Gauge(
    "router_pod_tokens_per_sec",
    "Tokens/sec on each pod (from last /metrics scrape)",
    ["pod"],
    registry=REGISTRY,
)

pod_kv_cache_util = Gauge(
    "router_pod_kv_cache_util",
    "KV cache utilization fraction on each pod (0–1)",
    ["pod"],
    registry=REGISTRY,
)

pod_circuit_broken = Gauge(
    "router_pod_circuit_broken",
    "1 if the pod is currently circuit-broken, 0 otherwise",
    ["pod"],
    registry=REGISTRY,
)

# Aggregate gauges
prefix_cache_hit_rate = Gauge(
    "router_prefix_cache_hit_rate",
    "Rolling fraction of requests that hit a warm prefix cache",
    registry=REGISTRY,
)

active_pods = Gauge(
    "router_active_pods",
    "Number of healthy (non-circuit-broken) pods",
    registry=REGISTRY,
)


def prometheus_output() -> bytes:
    return generate_latest(REGISTRY)


# ---------------------------------------------------------------------------
# Rolling TTFT / TPOT window (for summary stats independent of Prometheus)
# ---------------------------------------------------------------------------

_WINDOW = 10.0  # seconds


class RollingWindow:
    """Append-only deque with time-based pruning and percentile queries."""

    def __init__(self, window_secs: float = _WINDOW):
        self._data: deque[tuple[float, float]] = deque()
        self._window = window_secs

    def record(self, value: float) -> None:
        self._data.append((time.monotonic(), value))

    def _prune(self) -> list[float]:
        cutoff = time.monotonic() - self._window
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()
        return [v for _, v in self._data]

    def percentile(self, p: float) -> float:
        vals = sorted(self._prune())
        if not vals:
            return 0.0
        idx = max(0, int(len(vals) * p / 100) - 1)
        return vals[idx]

    def mean(self) -> float:
        vals = self._prune()
        return sum(vals) / len(vals) if vals else 0.0

    def count(self) -> int:
        self._prune()
        return len(self._data)


# Shared instances used by router.py
ttft_window = RollingWindow()
tpot_window = RollingWindow()


# ---------------------------------------------------------------------------
# Cache hit rate tracker
# ---------------------------------------------------------------------------


class CacheHitTracker:
    def __init__(self, window_secs: float = 60.0):
        self._hits: deque[tuple[float, bool]] = deque()
        self._window = window_secs

    def record(self, hit: bool) -> None:
        self._hits.append((time.monotonic(), hit))

    def rate(self) -> float:
        cutoff = time.monotonic() - self._window
        while self._hits and self._hits[0][0] < cutoff:
            self._hits.popleft()
        if not self._hits:
            return 0.0
        total = len(self._hits)
        hits = sum(1 for _, h in self._hits if h)
        return hits / total


cache_hit_tracker = CacheHitTracker()
