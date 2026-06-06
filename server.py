#!/usr/bin/env python3
"""
server.py
---------
vLLM + FastAPI inference server for a single Llama 3.2 3B pod.

--no-gpu / MINI_LLM_NO_GPU=1:
  - No vLLM, no GPU, no model download required.
  - Requests are served by a stub that returns fake tokens with realistic
    simulated latency (TTFT + TPOT calibrated to CPU speeds).
  - OTEL spans (rdma_write_ms, nccl_barrier_ms) are emitted identically.
  - Prometheus /metrics endpoint exposes the same metric names as real vLLM.

This lets you start the full server → router → benchmark pipeline on a
laptop to validate plumbing before touching any hardware.

Usage
-----
    # Laptop
    python server.py --no-gpu --port 8000 --pod-id prefill-0 --role prefill

    # Real vLLM (GPU pod)
    python server.py --port 8000 --pod-id prefill-0 --role prefill \
        --model meta-llama/Llama-3.2-3B-Instruct --tp 1
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import logging
import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from rdma.kv_transfer import RDMAKVTransfer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def _no_gpu_from_env() -> bool:
    return os.environ.get("MINI_LLM_NO_GPU", "0").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Metrics store (Prometheus-compatible counters)
# ---------------------------------------------------------------------------

class _Metrics:
    def __init__(self, pod_id: str, role: str) -> None:
        self.pod_id = pod_id
        self.role   = role
        self._lock  = threading.Lock()
        self.data: dict[str, float] = {
            "vllm_requests_total":         0,
            "vllm_tokens_generated_total": 0,
            "vllm_ttft_seconds_sum":       0,
            "vllm_ttft_seconds_count":     0,
            "vllm_tpot_seconds_sum":       0,
            "vllm_tpot_seconds_count":     0,
            "vllm_gpu_cache_usage":        0,  # 0.0 in sim
            "vllm_rdma_kv_bytes_total":    0,
        }

    def record(self, ttft_s: float, tpot_s: float, out_tokens: int, rdma_bytes: int = 0) -> None:
        with self._lock:
            self.data["vllm_requests_total"]         += 1
            self.data["vllm_tokens_generated_total"] += out_tokens
            self.data["vllm_ttft_seconds_sum"]       += ttft_s
            self.data["vllm_ttft_seconds_count"]     += 1
            self.data["vllm_tpot_seconds_sum"]       += tpot_s * out_tokens
            self.data["vllm_tpot_seconds_count"]     += out_tokens
            self.data["vllm_rdma_kv_bytes_total"]    += rdma_bytes

    def text(self) -> str:
        pod = self.pod_id
        role = self.role
        lines: list[str] = []
        for k, v in self.data.items():
            lines.append(f'{k}{{pod="{pod}",role="{role}"}} {v}')
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# OTEL stub
# ---------------------------------------------------------------------------

class _Span:
    def __init__(self, name: str) -> None:
        self.name  = name
        self._t0   = time.perf_counter()
        self._attrs: dict = {}

    def set_attribute(self, k: str, v) -> None:
        self._attrs[k] = v

    def end(self) -> None:
        ms = (time.perf_counter() - self._t0) * 1000
        logger.debug("[OTEL] %s %.2fms %s", self.name, ms, self._attrs)

    def __enter__(self): return self
    def __exit__(self, *_): self.end()


def _span(name: str) -> _Span:
    try:
        from opentelemetry import trace   # type: ignore
        return trace.get_tracer("mini-llm-d.server").start_span(name)   # type: ignore
    except ImportError:
        return _Span(name)


# ---------------------------------------------------------------------------
# Simulated inference engine (no-gpu mode)
# ---------------------------------------------------------------------------

class _SimEngine:
    """
    Simulates a single vLLM GPU pod with realistic CPU-calibrated latency.

    Prefill role: runs the attention over the full prompt → returns KV cache.
    Decode  role: auto-regressively generates *max_new_tokens* tokens.
    """

    # CPU-sim timings (wall-clock; much slower than real GPU)
    MS_PER_INPUT_TOKEN  = 0.5    # prefill
    MS_PER_OUTPUT_TOKEN = 2.0    # decode
    GPU_UTIL_SIM        = 0.0    # no GPU

    def __init__(self, pod_id: str, role: str) -> None:
        self.pod_id = pod_id
        self.role   = role
        self._xfer  = RDMAKVTransfer(no_gpu=True)
        self._queue_depth = 0
        self._lock  = threading.Lock()

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def prefill(self, prompt: str) -> dict:
        """Run prefill on *prompt*, returning KV cache metadata."""
        tokens = max(1, len(prompt.split()))
        with self._lock:
            self._queue_depth += 1
        try:
            with _span("engine.prefill") as sp:
                t0 = time.perf_counter()
                time.sleep(tokens * self.MS_PER_INPUT_TOKEN / 1000)
                ttft_s = time.perf_counter() - t0
                sp.set_attribute("input_tokens", tokens)
                sp.set_attribute("ttft_ms", ttft_s * 1000)

            # Register RDMA buffer for KV cache
            kv_bytes = tokens * 57344
            mr = self._xfer.register_buffer(kv_bytes, role="prefill")

            with _span("rdma_write") as sp:
                rdma_result = self._xfer.simulate_kv_handoff(kv_bytes)
                sp.set_attribute("rdma_write_ms", rdma_result.latency_ms)
                sp.set_attribute("rdma_bytes", rdma_result.bytes_transferred)

            return {
                "input_tokens":    tokens,
                "ttft_s":          ttft_s,
                "kv_bytes":        kv_bytes,
                "rdma_latency_ms": rdma_result.latency_ms,
                "rdma_bytes":      rdma_result.bytes_transferred,
                "rkey":            mr.rkey,
                "remote_addr":     mr.remote_addr,
            }
        finally:
            with self._lock:
                self._queue_depth -= 1

    def decode(self, kv_meta: dict, max_new_tokens: int = 64) -> dict:
        """Run decode given prefill KV metadata."""
        with self._lock:
            self._queue_depth += 1
        try:
            with _span("engine.decode") as sp:
                t0 = time.perf_counter()
                time.sleep(max_new_tokens * self.MS_PER_OUTPUT_TOKEN / 1000)
                decode_s = time.perf_counter() - t0
                sp.set_attribute("output_tokens", max_new_tokens)
                sp.set_attribute("tpot_ms", decode_s * 1000 / max(1, max_new_tokens))

            fake_text = " ".join([f"<tok_{i}>" for i in range(max_new_tokens)])
            return {
                "text":         fake_text,
                "output_tokens": max_new_tokens,
                "tpot_s":       decode_s,
                "tpot_per_token_ms": decode_s * 1000 / max(1, max_new_tokens),
            }
        finally:
            with self._lock:
                self._queue_depth -= 1

    def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        """Combined prefill + decode (for single-pod TP=1 mode)."""
        prefill_meta = self.prefill(prompt)
        decode_meta  = self.decode(prefill_meta, max_new_tokens)
        return {**prefill_meta, **decode_meta}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_engine: Optional[_SimEngine] = None
_metrics: Optional[_Metrics]  = None


class _Handler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:                      # noqa: N802
        length  = int(self.headers.get("Content-Length", 0))
        body    = json.loads(self.rfile.read(length) or b"{}")
        path    = self.path

        assert _engine and _metrics

        if path == "/prefill":
            prompt = body.get("prompt", "")
            result = _engine.prefill(prompt)
            _metrics.record(
                ttft_s=result["ttft_s"],
                tpot_s=0,
                out_tokens=0,
                rdma_bytes=result["rdma_bytes"],
            )
        elif path == "/decode":
            max_tokens = int(body.get("max_new_tokens", 64))
            kv_meta    = body.get("kv_meta", {})
            result     = _engine.decode(kv_meta, max_tokens)
            _metrics.record(
                ttft_s=0,
                tpot_s=result["tpot_s"],
                out_tokens=result["output_tokens"],
            )
        elif path == "/generate":
            prompt     = body.get("prompt", "")
            max_tokens = int(body.get("max_tokens", 64))
            result     = _engine.generate(prompt, max_tokens)
            _metrics.record(
                ttft_s=result["ttft_s"],
                tpot_s=result["tpot_s"],
                out_tokens=result["output_tokens"],
                rdma_bytes=result.get("rdma_bytes", 0),
            )
        else:
            self.send_response(404); self.end_headers(); return

        resp = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self) -> None:                       # noqa: N802
        assert _metrics
        if self.path == "/metrics":
            body = _metrics.text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        logger.info("server | " + fmt, *args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mini-llm-d vLLM inference pod (real or simulation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode (no vLLM / GPU required)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--pod-id", default="prefill-0",
                   help="Pod identifier shown in metrics labels")
    p.add_argument("--role", choices=["prefill", "decode", "combined"],
                   default="combined",
                   help="Pod role (prefill | decode | combined)")
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="[real mode] HuggingFace model ID")
    p.add_argument("--tp", type=int, default=1,
                   help="[real mode] Tensor parallel degree")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    global _engine, _metrics

    args   = _parse_args()
    no_gpu = args.no_gpu or _no_gpu_from_env()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    _engine  = _SimEngine(pod_id=args.pod_id, role=args.role)
    _metrics = _Metrics(pod_id=args.pod_id, role=args.role)

    if not no_gpu:
        logger.warning(
            "Real vLLM path not wired — falling back to simulation. "
            "Install vllm + GPU drivers and wire AsyncLLMEngine here."
        )

    print(f"\n{'='*60}")
    print(f"  mini-llm-d Inference Server")
    print(f"  Mode    : {'CPU-simulation (--no-gpu)' if no_gpu else 'Real vLLM'}")
    print(f"  Pod ID  : {args.pod_id}")
    print(f"  Role    : {args.role}")
    print(f"  Port    : {args.port}")
    print(f"  POST /generate  POST /prefill  POST /decode")
    print(f"  GET  /metrics   GET  /health")
    print(f"{'='*60}\n")

    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
