"""
mini-llm-d request router.

Sits in front of the inference pod fleet and routes incoming /generate
requests using a weighted-least-queue algorithm.  Exports Prometheus
metrics consumed by the custom HPA adapter.

Routing algorithm (priority order):
  1. Route to the healthy pod with the shortest queue depth.
  2. Break ties by highest recent tokens/sec throughput.
  3. Circuit-break pods that miss the /metrics poll timeout.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from metrics import MetricsRegistry

# ---------------------------------------------------------------------------
# Config (environment-driven so k8s ConfigMap can override)
# ---------------------------------------------------------------------------

POD_ENDPOINTS: list[str] = [
    ep.strip()
    for ep in os.getenv(
        "POD_ENDPOINTS",
        "http://inference-0.inference:8000,http://inference-1.inference:8000",
    ).split(",")
    if ep.strip()
]

POLL_INTERVAL_SECS: float = float(os.getenv("POLL_INTERVAL_SECS", "1.0"))
POD_TIMEOUT_SECS: float = float(os.getenv("POD_TIMEOUT_SECS", "0.1"))   # 100ms
CIRCUIT_COOLDOWN_SECS: float = float(os.getenv("CIRCUIT_COOLDOWN_SECS", "30.0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Registry (global singleton — safe because we're single-process async)
# ---------------------------------------------------------------------------

registry = MetricsRegistry()

# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------

async def _poll_pod(client: httpx.AsyncClient, endpoint: str) -> None:
    url = f"{endpoint}/metrics"
    try:
        resp = await client.get(url, timeout=POD_TIMEOUT_SECS)
        resp.raise_for_status()
        data = resp.json()
        await registry.update(endpoint, data)
    except Exception:
        log.warning("poll timeout/error for %s — opening circuit", endpoint)
        await registry.record_timeout(endpoint)


async def background_poller() -> None:
    """Continuously poll all pod /metrics endpoints at POLL_INTERVAL_SECS."""
    async with httpx.AsyncClient() as client:
        while True:
            tasks = [_poll_pod(client, ep) for ep in POD_ENDPOINTS]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(POLL_INTERVAL_SECS)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

async def select_pod() -> str:
    """
    Choose the best pod for the next request.
    Raises HTTPException 503 if no healthy pods are available.
    """
    candidates = await registry.healthy_pods()
    if not candidates:
        raise HTTPException(status_code=503, detail="No healthy inference pods available")

    # Sort by (queue_depth ASC, tokens_per_sec DESC)
    candidates.sort(key=lambda m: (m.queue_depth, -m.tokens_per_sec))
    chosen = candidates[0]
    await registry.record_routed(chosen.endpoint)
    return chosen.endpoint


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register all known pod endpoints
    for ep in POD_ENDPOINTS:
        await registry.register(ep)
    # Start background poller
    poller_task = asyncio.create_task(background_poller())
    log.info("Router started, polling %d pods every %.1fs", len(POD_ENDPOINTS), POLL_INTERVAL_SECS)
    yield
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="mini-llm-d router", lifespan=lifespan)


@app.get("/health")
async def health():
    pods = await registry.healthy_pods()
    return {"status": "ok", "healthy_pods": len(pods)}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus text exposition — consumed by the custom HPA metric adapter."""
    return await registry.prometheus_text()


@app.post("/generate")
async def generate(request: Request):
    """
    Proxy a /generate request to the best inference pod.
    Supports both streaming (SSE) and non-streaming responses.
    """
    endpoint = await select_pod()
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    target_url = f"{endpoint}/generate"
    start = time.monotonic()

    async def _stream_proxy() -> bytes:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", target_url, content=body, headers=headers, timeout=120.0
            ) as upstream:
                if upstream.status_code >= 400:
                    raise HTTPException(status_code=upstream.status_code)
                async for chunk in upstream.aiter_bytes():
                    yield chunk
        elapsed_ms = (time.monotonic() - start) * 1000
        log.debug("Proxied to %s in %.1fms", endpoint, elapsed_ms)

    # Detect streaming intent from query param or JSON body
    params = dict(request.query_params)
    stream = params.get("stream", "true").lower() != "false"

    # Try to peek at the JSON body for stream flag
    try:
        import json
        body_json = json.loads(body)
        stream = body_json.get("stream", stream)
    except Exception:
        pass

    if stream:
        return StreamingResponse(_stream_proxy(), media_type="text/event-stream")

    # Non-streaming: collect and forward
    async with httpx.AsyncClient() as client:
        resp = await client.post(target_url, content=body, headers=headers, timeout=120.0)
    elapsed_ms = (time.monotonic() - start) * 1000
    log.debug("Proxied to %s in %.1fms", endpoint, elapsed_ms)
    return resp.json()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("router:app", host="0.0.0.0", port=9000, log_level="info")
