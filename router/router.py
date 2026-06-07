"""
router/router.py

Prefix-cache aware request router for mini-llmd.

Routing algorithm (priority order):
  1. Hash first PREFIX_HASH_TOKENS tokens of the prompt
  2. If a healthy pod has a matching KV cache entry AND is not overloaded → route there
  3. On cache miss (or overloaded cache-hit pod) → route to shortest queue depth pod
  4. Break ties by highest tokens/sec
  5. Circuit-break pods that haven't responded within CIRCUIT_BREAK_TIMEOUT_MS

Exports:
  GET /metrics  – Prometheus text
  GET /health   – Router liveness
  POST /generate – Proxied to a selected inference pod (streaming SSE)

Environment variables:
  POD_ENDPOINTS       Comma-separated list of pod base URLs
  ROUTING_MODE        round_robin | least_queue | prefix_cache  (default: prefix_cache)
  CIRCUIT_BREAK_MS    Timeout before marking a pod circuit-broken (default: 100)
  POLL_INTERVAL_MS    How often to scrape pod /metrics (default: 500)
  OTEL_ENDPOINT       gRPC OTLP endpoint (default: http://jaeger:4317)
  PREFIX_HASH_TOKENS  Tokens to hash for prefix matching (default: 256)
  OVERLOAD_THRESHOLD  Queue depth multiplier to consider a pod overloaded (default: 2.0)
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from metrics import (
    active_pods,
    cache_hit_tracker,
    e2e_latency_histogram,
    pod_circuit_broken,
    pod_kv_cache_util,
    pod_queue_depth,
    pod_tokens_per_sec,
    prefix_cache_hit_rate,
    prometheus_output,
    requests_failed,
    requests_total,
    tpot_histogram,
    tpot_window,
    ttft_histogram,
    ttft_window,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prefix_cache import PrefixCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_ENDPOINTS = os.environ.get("POD_ENDPOINTS", "http://inference-0:8000,http://inference-1:8000")
POD_ENDPOINTS: list[str] = [e.strip() for e in RAW_ENDPOINTS.split(",") if e.strip()]
ROUTING_MODE: str = os.environ.get("ROUTING_MODE", "prefix_cache")
CIRCUIT_BREAK_MS: float = float(os.environ.get("CIRCUIT_BREAK_MS", "100"))
POLL_INTERVAL_S: float = float(os.environ.get("POLL_INTERVAL_MS", "500")) / 1000
OTEL_ENDPOINT: str = os.environ.get("OTEL_ENDPOINT", "http://jaeger:4317")
PREFIX_HASH_TOKENS: int = int(os.environ.get("PREFIX_HASH_TOKENS", "256"))
OVERLOAD_THRESHOLD: float = float(os.environ.get("OVERLOAD_THRESHOLD", "2.0"))

# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------

resource = Resource.create({"service.name": "mini-llmd-router"})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Pod state
# ---------------------------------------------------------------------------


@dataclass
class PodState:
    endpoint: str
    queue_depth: int = 0
    tokens_per_sec: float = 0.0
    kv_cache_util: float = 0.0
    active_requests: int = 0
    circuit_broken: bool = False
    last_seen: float = field(default_factory=time.monotonic)
    consecutive_failures: int = 0

    @property
    def healthy(self) -> bool:
        return not self.circuit_broken

    def mark_failure(self, threshold: int = 3) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= threshold:
            self.circuit_broken = True
            log.warning(f"Circuit-breaking pod {self.endpoint}")

    def mark_success(self) -> None:
        self.consecutive_failures = 0
        if self.circuit_broken:
            log.info(f"Restoring pod {self.endpoint}")
        self.circuit_broken = False
        self.last_seen = time.monotonic()


_pods: dict[str, PodState] = {ep: PodState(endpoint=ep) for ep in POD_ENDPOINTS}
_prefix_cache = PrefixCache(max_age_secs=300)
_rr_counter = itertools.cycle(POD_ENDPOINTS)

# ---------------------------------------------------------------------------
# Background scraper
# ---------------------------------------------------------------------------


async def _scrape_pod(client: httpx.AsyncClient, pod: PodState) -> None:
    try:
        resp = await asyncio.wait_for(
            client.get(f"{pod.endpoint}/metrics"),
            timeout=CIRCUIT_BREAK_MS / 1000,
        )
        resp.raise_for_status()
        data = resp.json()
        pod.queue_depth = data.get("queue_depth", 0)
        pod.tokens_per_sec = data.get("tokens_per_sec", 0.0)
        pod.kv_cache_util = data.get("kv_cache_util", 0.0)
        pod.active_requests = data.get("active_requests", 0)
        pod.mark_success()

        # Update Prometheus gauges
        pod_queue_depth.labels(pod=pod.endpoint).set(pod.queue_depth)
        pod_tokens_per_sec.labels(pod=pod.endpoint).set(pod.tokens_per_sec)
        pod_kv_cache_util.labels(pod=pod.endpoint).set(pod.kv_cache_util)
        pod_circuit_broken.labels(pod=pod.endpoint).set(0)

    except Exception as exc:
        log.debug(f"Scrape failed for {pod.endpoint}: {exc}")
        pod.mark_failure()
        pod_circuit_broken.labels(pod=pod.endpoint).set(1)


async def _background_scraper() -> None:
    async with httpx.AsyncClient(timeout=1.0) as client:
        while True:
            tasks = [_scrape_pod(client, pod) for pod in _pods.values()]
            await asyncio.gather(*tasks, return_exceptions=True)

            healthy = sum(1 for p in _pods.values() if p.healthy)
            active_pods.set(healthy)
            prefix_cache_hit_rate.set(cache_hit_tracker.rate())

            await asyncio.sleep(POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def _healthy_pods() -> list[PodState]:
    return [p for p in _pods.values() if p.healthy]


def _avg_queue_depth() -> float:
    hp = _healthy_pods()
    if not hp:
        return 0.0
    return sum(p.queue_depth for p in hp) / len(hp)


def _select_round_robin() -> Optional[PodState]:
    for _ in range(len(POD_ENDPOINTS)):
        ep = next(_rr_counter)
        pod = _pods.get(ep)
        if pod and pod.healthy:
            return pod
    return None


def _select_least_queue() -> Optional[PodState]:
    hp = _healthy_pods()
    if not hp:
        return None
    # Primary: lowest queue depth. Tie-break: highest tokens/sec
    return min(hp, key=lambda p: (p.queue_depth, -p.tokens_per_sec))


async def _select_prefix_cache(prefix_hash: str) -> tuple[Optional[PodState], bool]:
    """
    Returns (selected_pod, cache_hit).
    Falls back to least-queue if the cache-hit pod is overloaded.
    """
    cached_ep = await _prefix_cache.lookup(prefix_hash)
    avg_q = _avg_queue_depth()

    if cached_ep:
        pod = _pods.get(cached_ep)
        if pod and pod.healthy:
            overload_limit = max(2, avg_q * OVERLOAD_THRESHOLD)
            if pod.queue_depth <= overload_limit:
                return pod, True
            log.debug(
                f"Cache-hit pod {cached_ep} overloaded (q={pod.queue_depth}), "
                f"falling back to least-queue"
            )

    # Cache miss or overloaded cache-hit pod
    return _select_least_queue(), False


async def _select_pod(
    routing_mode: str, prefix_hash: Optional[str]
) -> tuple[Optional[PodState], bool]:
    """Returns (pod, cache_hit)."""
    if routing_mode == "round_robin":
        return _select_round_robin(), False
    if routing_mode == "least_queue":
        return _select_least_queue(), False
    # prefix_cache (default)
    if prefix_hash:
        return await _select_prefix_cache(prefix_hash)
    return _select_least_queue(), False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="mini-llmd-router")

# Strong references to background tasks so they are not garbage-collected.
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    scraper = asyncio.create_task(_background_scraper())
    evictor = asyncio.create_task(_eviction_loop())
    _background_tasks.update({scraper, evictor})
    try:
        yield
    finally:
        for t in _background_tasks:
            t.cancel()
        _background_tasks.clear()


app.router.lifespan_context = lifespan


async def _eviction_loop():
    while True:
        await asyncio.sleep(60)
        evicted = await _prefix_cache.evict_stale()
        if evicted:
            log.info(f"Evicted {evicted} stale prefix cache entries")


@app.get("/health")
async def health():
    hp = len(_healthy_pods())
    return {"status": "ok" if hp > 0 else "degraded", "healthy_pods": hp}


@app.get("/metrics")
async def metrics():
    return Response(content=prometheus_output(), media_type="text/plain; charset=utf-8")


@app.get("/cache-stats")
async def cache_stats():
    return await _prefix_cache.stats()


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt: str = body.get("prompt", "")
    routing_mode = ROUTING_MODE

    request_start = time.perf_counter()

    with tracer.start_as_current_span("router.handle_request") as span:
        span.set_attribute("routing_mode", routing_mode)
        span.set_attribute("prompt_length_chars", len(prompt))

        # Fetch prefix hash from pod (or compute locally as fallback)
        prefix_hash: Optional[str] = None
        if routing_mode == "prefix_cache" and prompt:
            try:
                hp = _healthy_pods()
                if hp:
                    async with httpx.AsyncClient(timeout=0.5) as client:
                        r = await client.get(
                            f"{hp[0].endpoint}/prefix-hash",
                            params={"prompt": prompt[:2000]},
                        )
                        if r.status_code == 200:
                            prefix_hash = r.json().get("prefix_hash")
            except Exception:
                pass  # continue without prefix hash

        # Select pod
        with tracer.start_as_current_span("router.select_pod") as sel_span:
            pod, cache_hit = await _select_pod(routing_mode, prefix_hash)
            sel_span.set_attribute("cache_hit", cache_hit)
            if pod:
                sel_span.set_attribute("selected_pod", pod.endpoint)

        if pod is None:
            requests_failed.inc()
            span.set_attribute("error", "no_healthy_pods")
            return JSONResponse({"error": "no healthy pods available"}, status_code=503)

        # Track metrics
        requests_total.labels(routing_mode=routing_mode, cache_hit=str(cache_hit)).inc()
        cache_hit_tracker.record(cache_hit)
        if prefix_hash:
            await _prefix_cache.record(prefix_hash, pod.endpoint)

        span.set_attribute("cache_hit", cache_hit)
        span.set_attribute("pod_endpoint", pod.endpoint)

        # Proxy the request
        first_token_seen = False
        ttft_ms: Optional[float] = None
        prev_token_time: Optional[float] = None

        async def proxy_stream():
            nonlocal first_token_seen, ttft_ms, prev_token_time

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{pod.endpoint}/generate",
                        json=body,
                        timeout=None,
                    ) as resp:
                        pod.mark_success()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            now = time.perf_counter()

                            if not first_token_seen:
                                ttft_ms = (now - request_start) * 1000
                                ttft_histogram.observe(ttft_ms / 1000)
                                ttft_window.record(ttft_ms)
                                first_token_seen = True

                            if prev_token_time is not None:
                                tpot_s = now - prev_token_time
                                tpot_histogram.observe(tpot_s)
                                tpot_window.record(tpot_s * 1000)

                            prev_token_time = now
                            # Re-emit with proper SSE framing (data line + blank line)
                            yield line + "\n\n"

                e2e_latency_histogram.observe(time.perf_counter() - request_start)

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                pod.mark_failure()
                log.warning(f"Pod {pod.endpoint} failed during streaming: {exc}")
                requests_failed.inc()
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(proxy_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        "router:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 9000)),
        log_level="info",
    )
