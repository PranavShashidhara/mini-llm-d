"""
inference/server.py

FastAPI wrapper around vLLM AsyncLLMEngine with:
  - SSE token streaming
  - TTFT / TPOT tracking (10-second sliding window)
  - Prefix-hash endpoint for cache-aware routing
  - OpenTelemetry span emission per request
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvicorn
import xxhash
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL_NAME: str = CFG["model"]["name"]
MODEL_PATH: str = CFG["model"]["path"]
GPU_MEMORY_UTILIZATION: float = CFG["model"].get("gpu_memory_utilization", 0.90)
MAX_MODEL_LEN: int = CFG["model"].get("max_model_len", 4096)
POD_ROLE: str = os.environ.get("POD_ROLE", "combined")  # prefill | decode | combined
POD_NAME: str = os.environ.get("POD_NAME", "pod-0")
OTEL_ENDPOINT: str = os.environ.get("OTEL_ENDPOINT", "http://jaeger:4317")
PREFIX_HASH_TOKENS: int = CFG.get("routing", {}).get("prefix_hash_tokens", 256)

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------

resource = Resource.create({"service.name": f"mini-llmd-inference-{POD_NAME}"})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Metrics state (sliding 10-second window)
# ---------------------------------------------------------------------------

_WINDOW_SECS = 10.0

# Each entry: (timestamp, ttft_ms)
_ttft_window: deque[tuple[float, float]] = deque()
# Each entry: (timestamp, tpot_ms)
_tpot_window: deque[tuple[float, float]] = deque()
# Each entry: (timestamp, tokens_generated)
_throughput_window: deque[tuple[float, int]] = deque()

_active_requests: int = 0
_queue_depth: int = 0
_lock = asyncio.Lock()


def _prune(window: deque, now: float) -> None:
    while window and now - window[0][0] > _WINDOW_SECS:
        window.popleft()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return sorted_v[idx]


async def record_ttft(ttft_ms: float) -> None:
    async with _lock:
        _ttft_window.append((time.monotonic(), ttft_ms))


async def record_tpot(tpot_ms: float) -> None:
    async with _lock:
        _tpot_window.append((time.monotonic(), tpot_ms))


async def record_tokens(n: int) -> None:
    async with _lock:
        _throughput_window.append((time.monotonic(), n))


async def get_metrics_snapshot() -> dict:
    async with _lock:
        now = time.monotonic()
        for w in (_ttft_window, _tpot_window, _throughput_window):
            _prune(w, now)

        ttft_vals = [v for _, v in _ttft_window]
        tpot_vals = [v for _, v in _tpot_window]
        tokens_total = sum(v for _, v in _throughput_window)

        return {
            "pod_name": POD_NAME,
            "pod_role": POD_ROLE,
            "active_requests": _active_requests,
            "queue_depth": _queue_depth,
            "tokens_per_sec": round(tokens_total / _WINDOW_SECS, 2),
            "ttft_p50": round(_percentile(ttft_vals, 50), 2),
            "ttft_p95": round(_percentile(ttft_vals, 95), 2),
            "ttft_p99": round(_percentile(ttft_vals, 99), 2),
            "tpot_p50": round(_percentile(tpot_vals, 50), 2),
            "tpot_p95": round(_percentile(tpot_vals, 95), 2),
            "tpot_p99": round(_percentile(tpot_vals, 99), 2),
        }


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

engine: AsyncLLMEngine | None = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tokenizer
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = get_tokenizer(MODEL_PATH, trust_remote_code=True)
    yield
    # Graceful shutdown
    if engine:
        engine.shutdown_background_loop()


app = FastAPI(title="mini-llmd-inference", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Kubernetes liveness probe."""
    return {"status": "ok", "pod": POD_NAME, "role": POD_ROLE}


@app.get("/metrics")
async def metrics():
    """Return sliding-window performance metrics for the router."""
    return JSONResponse(await get_metrics_snapshot())


@app.get("/prefix-hash")
async def prefix_hash(prompt: str = Query(..., description="Raw prompt text")):
    """
    Return the xxhash of the first PREFIX_HASH_TOKENS tokens.
    Used by the router to check KV cache affinity before routing.
    """
    token_ids = tokenizer.encode(prompt)[:PREFIX_HASH_TOKENS]
    token_bytes = json.dumps(token_ids).encode()
    h = xxhash.xxh64(token_bytes).hexdigest()
    return {"prefix_hash": h, "prefix_tokens": len(token_ids)}


@app.post("/generate")
async def generate(request: Request):
    """
    SSE streaming generation endpoint.

    Request body:
        prompt (str): The input prompt.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling p.
        stream (bool): If false, return full response as JSON.
    """
    global _active_requests, _queue_depth

    body = await request.json()
    prompt: str = body["prompt"]
    max_tokens: int = body.get("max_tokens", 512)
    temperature: float = body.get("temperature", 0.7)
    top_p: float = body.get("top_p", 0.95)
    do_stream: bool = body.get("stream", True)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    request_id = hashlib.md5(f"{prompt}{time.time()}".encode(), usedforsecurity=False).hexdigest()[
        :12
    ]

    with tracer.start_as_current_span("inference.generate") as span:
        span.set_attribute("pod.name", POD_NAME)
        span.set_attribute("pod.role", POD_ROLE)
        span.set_attribute("prompt.length_chars", len(prompt))
        span.set_attribute("max_tokens", max_tokens)

        async with _lock:
            _queue_depth += 1

        async def token_stream() -> AsyncGenerator[str, None]:
            global _active_requests, _queue_depth

            async with _lock:
                _queue_depth -= 1
                _active_requests += 1

            first_token_time: Optional[float] = None
            prev_token_time: Optional[float] = None
            output_tokens = 0
            prev_text_len = 0
            request_start = time.perf_counter()

            try:
                async for output in engine.generate(prompt, sampling_params, request_id):
                    now = time.perf_counter()
                    completion = output.outputs[0]
                    # vLLM returns cumulative text/token_ids on each step;
                    # emit only the newly generated delta to the client.
                    cumulative_text = completion.text
                    token_text = cumulative_text[prev_text_len:]
                    prev_text_len = len(cumulative_text)
                    new_token_count = len(completion.token_ids)

                    if first_token_time is None:
                        first_token_time = now
                        ttft_ms = (first_token_time - request_start) * 1000
                        await record_ttft(ttft_ms)
                        span.set_attribute("ttft_ms", round(ttft_ms, 2))

                    if prev_token_time is not None and new_token_count > output_tokens:
                        # Average inter-token time across tokens produced this step
                        steps = new_token_count - output_tokens
                        tpot_ms = ((now - prev_token_time) * 1000) / steps
                        await record_tpot(tpot_ms)

                    prev_token_time = now
                    output_tokens = new_token_count

                    chunk = json.dumps(
                        {
                            "token": token_text,
                            "finished": output.finished,
                            "usage": {
                                "prompt_tokens": len(output.prompt_token_ids),
                                "completion_tokens": output_tokens,
                            }
                            if output.finished
                            else None,
                        }
                    )
                    yield f"data: {chunk}\n\n"

                    if output.finished:
                        await record_tokens(output_tokens)
                        span.set_attribute("output_tokens", output_tokens)
                        break

            finally:
                async with _lock:
                    _active_requests -= 1

        if do_stream:
            return StreamingResponse(token_stream(), media_type="text/event-stream")

        # Non-streaming: collect all tokens
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        async for chunk_str in token_stream():
            data_str = chunk_str.removeprefix("data: ").strip()
            if data_str:
                data = json.loads(data_str)
                full_text += data.get("token", "")
                if data.get("usage"):
                    prompt_tokens = data["usage"]["prompt_tokens"]
                    completion_tokens = data["usage"]["completion_tokens"]

        return JSONResponse(
            {
                "text": full_text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
        )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
