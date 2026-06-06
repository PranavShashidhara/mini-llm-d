"""
mini-llm-d inference server
FastAPI wrapper around vLLM AsyncLLMEngine serving Llama 3.2 3B.
Exposes /generate (SSE streaming), /metrics, and /health endpoints.
"""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open("/app/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

MODEL_NAME = CONFIG["model"]["name"]
MAX_MODEL_LEN = CONFIG["model"].get("max_model_len", 4096)
DTYPE = CONFIG["model"].get("dtype", "auto")
METRICS_WINDOW_SECS = CONFIG.get("metrics_window_secs", 10)

# ---------------------------------------------------------------------------
# Token counter — sliding-window tokens/sec
# ---------------------------------------------------------------------------

class TokenCounter:
    """Thread-safe sliding-window token throughput tracker."""

    def __init__(self, window_secs: float = METRICS_WINDOW_SECS):
        self.window_secs = window_secs
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        self._active_requests: int = 0
        self._queue_depth: int = 0

    async def record(self, token_count: int) -> None:
        async with self._lock:
            now = time.monotonic()
            self._events.append((now, token_count))
            # prune expired events
            cutoff = now - self.window_secs
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    async def tokens_per_sec(self) -> float:
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_secs
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()
            total = sum(count for _, count in self._events)
            return total / self.window_secs

    def inc_active(self) -> None:
        self._active_requests += 1
        self._queue_depth = self._active_requests

    def dec_active(self) -> None:
        self._active_requests = max(0, self._active_requests - 1)
        self._queue_depth = self._active_requests

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def queue_depth(self) -> int:
        return self._queue_depth


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

engine: AsyncLLMEngine
counter: TokenCounter


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, counter
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        trust_remote_code=True,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    counter = TokenCounter()
    yield
    await engine.shutdown_background_tasks()


app = FastAPI(title="mini-llm-d inference", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Kubernetes liveness and readiness probe."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Return pod-level metrics consumed by the router and HPA adapter."""
    tps = await counter.tokens_per_sec()
    return {
        "queue_depth": counter.queue_depth,
        "active_requests": counter.active_requests,
        "tokens_per_sec": round(tps, 2),
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Accept a prompt, stream tokens back via SSE.
    Tracks active request count and token throughput.
    """
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    request_id = f"req-{time.time_ns()}"
    counter.inc_active()

    async def token_stream() -> AsyncGenerator[str, None]:
        generated_tokens = 0
        try:
            async for output in engine.generate(req.prompt, sampling_params, request_id):
                if output.outputs:
                    delta = output.outputs[0].text
                    token_count = len(output.outputs[0].token_ids)
                    # yield incremental delta for streaming
                    yield f"data: {delta}\n\n"
                    generated_tokens = token_count
        finally:
            counter.dec_active()
            await counter.record(generated_tokens)

    if req.stream:
        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # Non-streaming: collect all output and return
    full_text = ""
    total_tokens = 0
    try:
        async for output in engine.generate(req.prompt, sampling_params, request_id):
            if output.outputs:
                full_text = output.outputs[0].text
                total_tokens = len(output.outputs[0].token_ids)
    finally:
        counter.dec_active()
        await counter.record(total_tokens)

    return {"text": full_text, "tokens_generated": total_tokens}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
