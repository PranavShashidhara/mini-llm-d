#!/usr/bin/env python3
"""
rdma/rdma_exporter/exporter.py
-------------------------------
Lightweight RDMA counter exporter — Prometheus metrics for queue-pair state,
port rx/tx data, and error counters.

In production this is a Go sidecar that reads from:
    /sys/class/infiniband/<dev>/ports/1/counters/

On a laptop (--no-gpu / MINI_LLM_NO_GPU=1):
  - There is no /sys/class/infiniband; the sysfs path does not exist.
  - The exporter runs in simulation mode: it generates realistic synthetic
    RDMA counter deltas (based on the kv_transfer simulation bandwidth),
    exposing them on the same Prometheus /metrics endpoint.
  - Grafana dashboards and alert rules require zero changes.

Metrics exposed
---------------
    rdma_port_rcv_data_total{pod, device}         gauge (bytes)
    rdma_port_xmit_data_total{pod, device}        gauge (bytes)
    rdma_port_rcv_errors_total{pod, device}       gauge
    rdma_port_xmit_discards_total{pod, device}    gauge
    rdma_qp_state{pod, qp_num, state}             gauge (1 = active)
    rdma_exporter_mode{mode}                      info gauge

Usage
-----
    # Laptop
    python rdma/rdma_exporter/exporter.py --no-gpu --port 9101

    # Real ibverbs hardware
    python rdma/rdma_exporter/exporter.py --port 9101

    # Verify
    curl http://localhost:9101/metrics
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import socket
import logging
import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def _no_gpu_from_env() -> bool:
    return os.environ.get("MINI_LLM_NO_GPU", "0").strip() in ("1", "true", "yes")


POD_NAME   = os.environ.get("POD_NAME", socket.gethostname())
DEVICE     = os.environ.get("RDMA_DEVICE", "rxe0")   # RoCEv2 soft-RDMA device
SYSFS_BASE = Path(f"/sys/class/infiniband/{DEVICE}/ports/1/counters")

# ---------------------------------------------------------------------------
# Sysfs counter reader (real path)
# ---------------------------------------------------------------------------

SYSFS_COUNTERS = {
    "rdma_port_rcv_data_total":        "port_rcv_data",
    "rdma_port_xmit_data_total":       "port_xmit_data",
    "rdma_port_rcv_errors_total":      "port_rcv_errors",
    "rdma_port_xmit_discards_total":   "port_xmit_discards",
}


def _read_sysfs_counter(name: str) -> Optional[int]:
    path = SYSFS_BASE / name
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _read_qp_states() -> dict[str, str]:
    """Return {qp_num: state_str} from sysfs (real InfiniBand / RoCEv2 device)."""
    states: dict[str, str] = {}
    qp_dir = Path(f"/sys/class/infiniband/{DEVICE}/")
    if not qp_dir.exists():
        return states
    for entry in qp_dir.glob("*/qp_state"):
        qp_num = entry.parent.name
        try:
            state = entry.read_text().strip()
            states[qp_num] = state
        except IOError:
            pass
    return states


# ---------------------------------------------------------------------------
# Simulation counter generator
# ---------------------------------------------------------------------------

class _SimCounters:
    """
    Generates monotonically increasing synthetic RDMA counters.
    Simulates KV-cache transfer traffic at ~10 Gb/s loopback.
    """
    BW_BYTES_PER_SEC = 1.25e9   # 10 Gb/s

    def __init__(self) -> None:
        self._start = time.time()
        self._error_injected = False
        self._qp_nums = ["0x1a2b", "0x3c4d"]

    def snapshot(self) -> dict:
        elapsed = time.time() - self._start
        # Simulate cumulative bytes transferred
        rcv_data  = int(elapsed * self.BW_BYTES_PER_SEC * 0.5)
        xmit_data = int(elapsed * self.BW_BYTES_PER_SEC * 0.5)
        # Small random jitter
        rcv_data  += random.randint(0, 4096)
        xmit_data += random.randint(0, 4096)
        # Zero errors (spec target: 0 errors during benchmark)
        rcv_errors     = 0
        xmit_discards  = 0

        qp_states = {qp: "RTS" for qp in self._qp_nums}   # Ready-To-Send

        return {
            "rdma_port_rcv_data_total":      rcv_data,
            "rdma_port_xmit_data_total":     xmit_data,
            "rdma_port_rcv_errors_total":    rcv_errors,
            "rdma_port_xmit_discards_total": xmit_discards,
            "qp_states": qp_states,
        }


# ---------------------------------------------------------------------------
# Prometheus text formatter
# ---------------------------------------------------------------------------

def _format_metrics(snapshot: dict, mode: str) -> str:
    lines: list[str] = []
    pod  = POD_NAME
    dev  = DEVICE

    def gauge(name: str, labels: dict, value: float, help_text: str = "") -> None:
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{{{label_str}}} {value}")

    gauge("rdma_port_rcv_data_total",
          {"pod": pod, "device": dev},
          snapshot["rdma_port_rcv_data_total"],
          "Cumulative bytes received on RDMA port")

    gauge("rdma_port_xmit_data_total",
          {"pod": pod, "device": dev},
          snapshot["rdma_port_xmit_data_total"],
          "Cumulative bytes transmitted on RDMA port")

    gauge("rdma_port_rcv_errors_total",
          {"pod": pod, "device": dev},
          snapshot["rdma_port_rcv_errors_total"],
          "Cumulative receive errors on RDMA port")

    gauge("rdma_port_xmit_discards_total",
          {"pod": pod, "device": dev},
          snapshot["rdma_port_xmit_discards_total"],
          "Cumulative transmit discards on RDMA port")

    for qp_num, state in snapshot.get("qp_states", {}).items():
        gauge("rdma_qp_state",
              {"pod": pod, "qp_num": qp_num, "state": state},
              1,
              "RDMA Queue Pair state (1 = active in this state)")

    gauge("rdma_exporter_mode",
          {"mode": mode},
          1,
          "Exporter operating mode: 'sim' or 'real'")

    lines.append("")   # trailing newline
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    sim: Optional[_SimCounters] = None
    no_gpu: bool = True

    def do_GET(self) -> None:                         # noqa: N802
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return

        if self.__class__.no_gpu or self.__class__.sim is not None:
            snap = (self.__class__.sim or _SimCounters()).snapshot()
            mode = "sim"
        else:
            # Real sysfs path
            snap = {
                k: (_read_sysfs_counter(v) or 0)
                for k, v in SYSFS_COUNTERS.items()
            }
            snap["qp_states"] = _read_qp_states()
            mode = "real"

        body = _format_metrics(snap, mode).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # suppress access log spam
        logger.debug(fmt, *args)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(port: int, no_gpu: bool) -> None:
    mode = "CPU-simulation" if no_gpu else "real sysfs"
    sim = _SimCounters() if no_gpu else None
    _Handler.sim = sim
    _Handler.no_gpu = no_gpu

    server = HTTPServer(("0.0.0.0", port), _Handler)
    logger.info(
        "rdma-exporter running on http://0.0.0.0:%d/metrics  [mode=%s  pod=%s  device=%s]",
        port, mode, POD_NAME, DEVICE,
    )
    print(f"  rdma-exporter listening on :{port}/metrics  ({mode})")
    print(f"  curl http://localhost:{port}/metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down exporter.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RDMA Prometheus exporter (sysfs or simulation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode (no RDMA hardware required)")
    p.add_argument("--port", type=int, default=9101,
                   help="HTTP port to expose /metrics on")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    run_server(port=args.port, no_gpu=args.no_gpu or _no_gpu_from_env())
