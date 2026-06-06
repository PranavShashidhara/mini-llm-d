#!/usr/bin/env python3
"""
router.py
----------
Prefix-cache aware request router for mini-llm-d.

Responsibilities (§2 Layer 2):
  - Prefix-hash based routing to maximise KV-cache reuse on Prefill pods
  - Queue-depth aware load balancing
  - Circuit breaking on pod failure
  - OTEL span attributes: rdma_write_latency_ms, rdma_bytes, nccl_barrier_ms

--no-gpu / MINI_LLM_NO_GPU=1:
  - Pod backends are replaced by in-process stub servers (no vLLM / GPU needed)
  - RDMA KV-cache hand-off uses the CPU-simulation path in kv_transfer.py
  - NCCL all-reduce uses the simulation path in nccl_init.py
  - All OTEL spans are emitted exactly as in production (to Jaeger or stdout)

Usage
-----
    # Laptop (no GPU, no vLLM)
    python router.py --no-gpu --port 8080

    # Production (GPU pods running)
    python router.py --backends http://prefill-0:8000 http://prefill-1:8000 \
                                http://decode-0:8000  --port 8080
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import random
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from rdma.kv_transfer import RDMAKVTransfer
from rdma.nccl_init import NCCLComm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def _no_gpu_from_env() -> bool:
    return os.environ.get("MINI_LLM_NO_GPU", "0").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# OTEL stub (real opentelemetry-sdk is optional)
# ---------------------------------------------------------------------------

class _Span:
    """Minimal OTEL-compatible span that writes to logger when SDK absent."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._attrs: dict = {}
        self._start = time.perf_counter()

    def set_attribute(self, key: str, value) -> None:
        self._attrs[key] = value

    def end(self) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        logger.debug(
            "[OTEL] span=%s duration=%.2fms attrs=%s",
            self.name, elapsed_ms, self._attrs,
        )

    def __enter__(self): return self
    def __exit__(self, *_): self.end()


def _new_span(name: str) -> _Span:
    try:
        from opentelemetry import trace  # type: ignore
        tracer = trace.get_tracer("mini-llm-d.router")
        return tracer.start_span(name)   # type: ignore
    except ImportError:
        return _Span(name)


# ---------------------------------------------------------------------------
# Simulated Pod backend (no-gpu mode)
# ---------------------------------------------------------------------------

class _SimPod:
    """
    In-process simulated vLLM pod. Accepts a prompt, returns a fake
    completion after sleeping to simulate TTFT + decode time.
    """
    TTFT_MS_PER_TOKEN = 0.5    # simulated prefill speed
    TPOT_MS_PER_TOKEN = 2.0    # simulated decode speed (CPU; GPU would be ~5ms)

    def __init__(self, pod_id: str, role: str = "decode") -> None:
        self.pod_id   = pod_id
        self.role     = role
        self.queue_depth = 0
        self._lock    = threading.Lock()

    def infer(self, prompt: str, max_tokens: int = 64) -> dict:
        input_tokens = max(1, len(prompt.split()))
        with self._lock:
            self.queue_depth += 1
        try:
            # Simulate TTFT (prefill)
            ttft_s = input_tokens * self.TTFT_MS_PER_TOKEN / 1000
            time.sleep(ttft_s)
            # Simulate decode
            tpot_s = max_tokens * self.TPOT_MS_PER_TOKEN / 1000
            time.sleep(tpot_s)
            fake_text = f"[sim:{self.pod_id}] " + " ".join(
                ["token"] * max_tokens
            )
            return {
                "text": fake_text,
                "ttft_ms": round(ttft_s * 1000, 2),
                "tpot_ms": round(tpot_s * 1000 / max(1, max_tokens), 2),
                "input_tokens": input_tokens,
                "output_tokens": max_tokens,
                "pod_id": self.pod_id,
            }
        finally:
            with self._lock:
                self.queue_depth -= 1


# ---------------------------------------------------------------------------
# Prefix-cache aware router core
# ---------------------------------------------------------------------------

class Router:
    """
    Routes inference requests to the best Prefill/Decode pod pair.

    Routing strategy:
      1. Hash the prompt prefix (first 64 tokens) → select Prefill pod that
         likely already holds the KV cache for this prefix.
      2. Among Decode pods with the lowest queue depth, select one.
      3. On circuit-break (3 consecutive failures), skip that pod for 30 s.

    Parameters
    ----------
    prefill_pods : list
        List of _SimPod (no-gpu) or backend URL strings (real).
    decode_pods : list
        Same as above for Decode pods.
    no_gpu : bool
        When True, uses in-process simulation for pods and RDMA transfer.
    """

    CIRCUIT_BREAK_THRESHOLD = 3
    CIRCUIT_BREAK_WINDOW_S  = 30.0

    def __init__(
        self,
        prefill_pods: list,
        decode_pods:  list,
        no_gpu: bool = False,
    ) -> None:
        self.prefill_pods = prefill_pods
        self.decode_pods  = decode_pods
        self.no_gpu = no_gpu

        self._xfer = RDMAKVTransfer(no_gpu=no_gpu)
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._circuit_open_until: dict[str, float] = {}
        self._lock = threading.Lock()

        # Prometheus-style counters (simplified)
        self.metrics: dict[str, float] = {
            "requests_total": 0,
            "rdma_kv_bytes_total": 0,
            "errors_total": 0,
        }

    # ------------------------------------------------------------------
    # Pod selection
    # ------------------------------------------------------------------

    def _prefix_hash(self, prompt: str, n_tokens: int = 64) -> int:
        prefix = " ".join(prompt.split()[:n_tokens])
        return int(hashlib.sha256(prefix.encode()).hexdigest(), 16)

    def _select_prefill(self, prompt: str) -> Optional[_SimPod]:
        available = [p for p in self.prefill_pods if not self._is_open(str(p.pod_id if hasattr(p, 'pod_id') else p))]
        if not available:
            return None
        h = self._prefix_hash(prompt)
        return available[h % len(available)]

    def _select_decode(self) -> Optional[_SimPod]:
        available = [p for p in self.decode_pods if not self._is_open(str(p.pod_id if hasattr(p, 'pod_id') else p))]
        if not available:
            return None
        return min(available, key=lambda p: p.queue_depth)

    def _is_open(self, pod_id: str) -> bool:
        deadline = self._circuit_open_until.get(pod_id, 0)
        if time.time() < deadline:
            return True
        return False

    def _record_failure(self, pod_id: str) -> None:
        with self._lock:
            self._failure_counts[pod_id] += 1
            if self._failure_counts[pod_id] >= self.CIRCUIT_BREAK_THRESHOLD:
                self._circuit_open_until[pod_id] = time.time() + self.CIRCUIT_BREAK_WINDOW_S
                logger.warning("Circuit open for pod %s for %.0fs", pod_id, self.CIRCUIT_BREAK_WINDOW_S)

    def _record_success(self, pod_id: str) -> None:
        with self._lock:
            self._failure_counts[pod_id] = 0
            self._circuit_open_until.pop(pod_id, None)

    # ------------------------------------------------------------------
    # Main routing function
    # ------------------------------------------------------------------

    def route(self, prompt: str, max_tokens: int = 64) -> dict:
        """Route a request through Prefill → (RDMA KV transfer) → Decode."""
        t_start = time.perf_counter()
        self.metrics["requests_total"] += 1

        with _new_span("router.select_pod") as span:
            prefill = self._select_prefill(prompt)
            decode  = self._select_decode()
            span.set_attribute("prefill_pod", str(getattr(prefill, "pod_id", "none")))
            span.set_attribute("decode_pod",  str(getattr(decode,  "pod_id", "none")))

        if prefill is None or decode is None:
            self.metrics["errors_total"] += 1
            return {"error": "no available pods", "status": 503}

        # 1. Prefill
        prefill_result: dict = {}
        with _new_span("prefill") as span:
            try:
                prefill_result = prefill.infer(prompt, max_tokens=0)   # type: ignore
                self._record_success(prefill.pod_id)
                span.set_attribute("ttft_ms", prefill_result.get("ttft_ms", 0))
            except Exception as exc:
                logger.exception("Prefill pod %s failed: %s", prefill.pod_id, exc)
                self._record_failure(prefill.pod_id)
                self.metrics["errors_total"] += 1
                return {"error": str(exc), "status": 500}

        # 2. RDMA KV-cache hand-off (Prefill → Decode)
        kv_bytes = prefill_result.get("input_tokens", 1) * 57344   # ~56 KB/token
        rdma_result = None
        with _new_span("rdma_kv_write") as span:
            rdma_result = self._xfer.simulate_kv_handoff(kv_bytes)
            span.set_attribute("rdma_write_latency_ms", rdma_result.latency_ms)
            span.set_attribute("rdma_bytes", rdma_result.bytes_transferred)
            self.metrics["rdma_kv_bytes_total"] += rdma_result.bytes_transferred

        # 3. Decode
        decode_result: dict = {}
        with _new_span("decode") as span:
            try:
                decode_result = decode.infer(prompt, max_tokens=max_tokens)   # type: ignore
                self._record_success(decode.pod_id)
                span.set_attribute("tpot_ms", decode_result.get("tpot_ms", 0))
            except Exception as exc:
                logger.exception("Decode pod %s failed: %s", decode.pod_id, exc)
                self._record_failure(decode.pod_id)
                self.metrics["errors_total"] += 1
                return {"error": str(exc), "status": 500}

        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "text":          decode_result.get("text", ""),
            "ttft_ms":       prefill_result.get("ttft_ms", 0),
            "tpot_ms":       decode_result.get("tpot_ms", 0),
            "total_ms":      round(total_ms, 2),
            "rdma_ms":       rdma_result.latency_ms if rdma_result else 0,
            "rdma_bytes":    rdma_result.bytes_transferred if rdma_result else 0,
            "prefill_pod":   prefill.pod_id,
            "decode_pod":    decode.pod_id,
            "input_tokens":  prefill_result.get("input_tokens", 0),
            "output_tokens": decode_result.get("output_tokens", max_tokens),
            "status":        200,
        }

    def metrics_text(self) -> str:
        """Prometheus-format /metrics output."""
        lines = []
        for k, v in self.metrics.items():
            lines.append(f"router_{k} {v}")
        # Per-pod queue depth
        for p in self.decode_pods:
            lines.append(f'router_pod_queue_depth{{pod="{p.pod_id}"}} {p.queue_depth}')
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTTP server wrapping the router
# ---------------------------------------------------------------------------

_router: Optional[Router] = None


class _RouterHandler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:                      # noqa: N802
        if self.path != "/generate":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length) or b"{}")
        prompt     = body.get("prompt", "Hello")
        max_tokens = int(body.get("max_tokens", 64))

        assert _router is not None
        result = _router.route(prompt, max_tokens)

        resp = json.dumps(result).encode()
        self.send_response(result.get("status", 200))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self) -> None:                       # noqa: N802
        assert _router is not None
        if self.path == "/metrics":
            body = _router.metrics_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200); self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        logger.info(fmt, *args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mini-llm-d prefix-cache aware router",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode (stub pods, simulated RDMA/NCCL)")
    p.add_argument("--port", type=int, default=8080,
                   help="Router HTTP listen port")
    p.add_argument("--num-prefill", type=int, default=2,
                   help="[no-gpu] Number of simulated Prefill pods")
    p.add_argument("--num-decode",  type=int, default=2,
                   help="[no-gpu] Number of simulated Decode pods")
    p.add_argument("--backends", nargs="*", default=[],
                   metavar="URL",
                   help="[real] Backend URLs: first half Prefill, second half Decode")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    global _router

    args = _parse_args()
    no_gpu = args.no_gpu or _no_gpu_from_env()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if no_gpu:
        prefill_pods = [_SimPod(f"prefill-{i}", role="prefill") for i in range(args.num_prefill)]
        decode_pods  = [_SimPod(f"decode-{i}",  role="decode")  for i in range(args.num_decode)]
        logger.info(
            "Router: CPU-simulation mode — %d prefill pods, %d decode pods",
            len(prefill_pods), len(decode_pods),
        )
    else:
        urls  = args.backends
        mid   = len(urls) // 2
        # For real path, you'd wrap URLs in an HttpPod class — left as exercise
        raise NotImplementedError(
            "Real backend URL routing not yet wired. "
            "Add an HttpPod wrapper and pass --backends ... or use --no-gpu."
        )

    _router = Router(prefill_pods=prefill_pods, decode_pods=decode_pods, no_gpu=no_gpu)

    print(f"\n{'='*60}")
    print(f"  mini-llm-d Router")
    print(f"  Mode    : {'CPU-simulation (--no-gpu)' if no_gpu else 'Real backends'}")
    print(f"  Port    : {args.port}")
    print(f"  Prefill : {len(prefill_pods)} pod(s)")
    print(f"  Decode  : {len(decode_pods)} pod(s)")
    print(f"  POST /generate  {{\"prompt\": \"...\", \"max_tokens\": 64}}")
    print(f"  GET  /metrics")
    print(f"  GET  /health")
    print(f"{'='*60}\n")

    server = HTTPServer(("0.0.0.0", args.port), _RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Router stopped.")


if __name__ == "__main__":
    main()
