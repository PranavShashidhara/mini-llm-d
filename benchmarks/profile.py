"""
benchmarks/profile.py

GPU utilization capture and scaling_curves.png generation.

Runs Locust at increasing concurrency levels, captures:
  - Aggregate throughput (tokens/sec)
  - TTFT and TPOT percentiles
  - GPU utilization per pod (via nvidia-smi over kubectl exec)
  - HPA scale events (from Kubernetes events)

Outputs:
  results/scaling_curves.png   — throughput + latency vs concurrency
  results/gpu_utilization.png  — per-pod GPU util timeline

Usage:
    ROUTER_URL=http://$(minikube ip):30900 python profile.py [--scenario C]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:30900")
NAMESPACE = "mini-llmd"

# Concurrency levels to sweep
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64]
RUN_TIME_PER_LEVEL = os.environ.get("BENCH_RUN_TIME", "60s")


# ---------------------------------------------------------------------------
# GPU scraping
# ---------------------------------------------------------------------------


@dataclass
class GpuSample:
    timestamp: float
    pod: str
    utilization_pct: float
    memory_used_mib: float
    memory_total_mib: float


_gpu_samples: list[GpuSample] = []
_stop_gpu_scraping = threading.Event()


def _scrape_gpu_loop(interval: float = 5.0):
    """Background thread: scrape nvidia-smi from every inference pod."""
    while not _stop_gpu_scraping.is_set():
        pods = _get_inference_pods()
        for pod in pods:
            try:
                result = subprocess.run(
                    [
                        "kubectl",
                        "-n",
                        NAMESPACE,
                        "exec",
                        pod,
                        "--",
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) == 3:
                        _gpu_samples.append(
                            GpuSample(
                                timestamp=time.time(),
                                pod=pod,
                                utilization_pct=float(parts[0].strip()),
                                memory_used_mib=float(parts[1].strip()),
                                memory_total_mib=float(parts[2].strip()),
                            )
                        )
            except Exception:
                pass
        time.sleep(interval)


def _get_inference_pods() -> list[str]:
    # Inference pods carry a `role` label (combined|prefill|decode); the router
    # shares app=mini-llmd but has `component=router` and no `role`, so we use a
    # label-existence selector to exclude it.
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "pods",
            "-l",
            "app=mini-llmd,role",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip().split()
    return []


# ---------------------------------------------------------------------------
# HPA event scraping
# ---------------------------------------------------------------------------


@dataclass
class HpaEvent:
    timestamp: float
    replicas: int
    message: str


def _get_hpa_events() -> list[HpaEvent]:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "events",
            "--field-selector=involvedObject.name=inference-hpa",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    events = []
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            for item in data.get("items", []):
                msg = item.get("message", "")
                t = item.get("lastTimestamp") or item.get("eventTime", "")
                if t:
                    ts = time.time()  # approximate
                    events.append(HpaEvent(timestamp=ts, replicas=0, message=msg))
        except json.JSONDecodeError:
            pass
    return events


# ---------------------------------------------------------------------------
# Locust helper
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    users: int
    throughput_tps: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tpot_p50: float
    tpot_p95: float
    tpot_p99: float
    pod_count: int


def run_level(users: int, scenario: str) -> LevelResult:
    import pickle

    metrics_path = f"/tmp/prof_metrics_{users}.pkl"
    env = os.environ.copy()
    env["SCENARIO"] = scenario
    env["METRICS_OUT"] = metrics_path

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(Path(__file__).parent / "load_test.py"),
        "--headless",
        "--host",
        ROUTER_URL,
        "--users",
        str(users),
        "--spawn-rate",
        str(max(1, users // 4)),
        "--run-time",
        RUN_TIME_PER_LEVEL,
    ]
    subprocess.run(cmd, env=env, capture_output=True)

    try:
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
    except FileNotFoundError:
        return LevelResult(users, 0, 0, 0, 0, 0, 0, 0, 0)

    successful = [m for m in metrics if m.success and m.ttft is not None]
    if not successful:
        return LevelResult(users, 0, 0, 0, 0, 0, 0, 0, 0)

    ttfts = [m.ttft for m in successful]
    tpots = []
    total_tokens = 0
    total_time = RUN_TIME_PER_LEVEL.replace("s", "")
    try:
        run_secs = float(total_time)
    except ValueError:
        run_secs = 60.0

    for m in successful:
        total_tokens += m.total_tokens
        if len(m.token_times) > 1:
            tpots.extend(m.token_times[1:])

    throughput = total_tokens / run_secs

    def pct(data, p):
        return float(np.percentile(data, p)) if data else 0.0

    pod_count = len(_get_inference_pods())

    return LevelResult(
        users=users,
        throughput_tps=round(throughput, 1),
        ttft_p50=round(pct(ttfts, 50), 1),
        ttft_p95=round(pct(ttfts, 95), 1),
        ttft_p99=round(pct(ttfts, 99), 1),
        tpot_p50=round(pct(tpots, 50), 1),
        tpot_p95=round(pct(tpots, 95), 1),
        tpot_p99=round(pct(tpots, 99), 1),
        pod_count=pod_count,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scaling_curves(results: list[LevelResult], scenario: str):
    users = [r.users for r in results]
    tps = [r.throughput_tps for r in results]
    ttft_p99 = [r.ttft_p99 for r in results]
    tpot_p99 = [r.tpot_p99 for r in results]
    pods = [r.pod_count for r in results]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # --- Throughput ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_r = ax1.twinx()
    ax1.plot(users, tps, "o-", color="#2ecc71", linewidth=2, label="tokens/sec")
    ax1_r.step(users, pods, "--", color="#3498db", linewidth=1.5, where="post", label="pod count")
    ax1.set_xlabel("Concurrent users")
    ax1.set_ylabel("Aggregate tokens/sec", color="#2ecc71")
    ax1_r.set_ylabel("Pod count", color="#3498db")
    ax1.set_title(f"Throughput — Scenario {scenario}")
    ax1.grid(True, alpha=0.3)

    # --- TTFT ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(users, [r.ttft_p50 for r in results], "o-", label="p50", color="#3498db")
    ax2.plot(users, [r.ttft_p95 for r in results], "s-", label="p95", color="#e67e22")
    ax2.plot(users, ttft_p99, "^-", label="p99", color="#e74c3c")
    ax2.axhline(100, color="red", linestyle="--", linewidth=1, alpha=0.5, label="100ms target")
    ax2.set_xlabel("Concurrent users")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title(f"Time to First Token — Scenario {scenario}")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- TPOT ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(users, [r.tpot_p50 for r in results], "o-", label="p50", color="#3498db")
    ax3.plot(users, [r.tpot_p95 for r in results], "s-", label="p95", color="#e67e22")
    ax3.plot(users, tpot_p99, "^-", label="p99", color="#e74c3c")
    ax3.set_xlabel("Concurrent users")
    ax3.set_ylabel("TPOT (ms)")
    ax3.set_title(f"Time per Output Token — Scenario {scenario}")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- GPU utilization timeline ---
    ax4 = fig.add_subplot(gs[1, 1])
    if _gpu_samples:
        pod_names = sorted(set(s.pod for s in _gpu_samples))
        colors = plt.cm.tab10(np.linspace(0, 1, len(pod_names)))
        t0 = _gpu_samples[0].timestamp
        for pod, color in zip(pod_names, colors):
            pts = [(s.timestamp - t0, s.utilization_pct) for s in _gpu_samples if s.pod == pod]
            if pts:
                ts, utils = zip(*pts)
                ax4.plot(ts, utils, label=pod, color=color, linewidth=1.5)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("GPU utilization (%)")
        ax4.set_title("Per-pod GPU Utilization")
        ax4.set_ylim(0, 105)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "No GPU data\n(nvidia-smi not available)",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            color="gray",
        )
        ax4.set_title("Per-pod GPU Utilization")

    fig.suptitle(
        f"mini-llmd Scaling Curves — Scenario {scenario}",
        fontsize=14,
        fontweight="bold",
    )

    out = RESULTS_DIR / "scaling_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="C", choices=["A", "B", "C"])
    args = parser.parse_args()

    print(f"mini-llmd profiler — Scenario {args.scenario}")
    print(f"Router: {ROUTER_URL}")
    print(f"Concurrency sweep: {CONCURRENCY_LEVELS}")
    print()

    # Start GPU background scraper
    gpu_thread = threading.Thread(target=_scrape_gpu_loop, daemon=True)
    gpu_thread.start()

    results = []
    for users in CONCURRENCY_LEVELS:
        print(f"→ Running {users} concurrent users ...")
        r = run_level(users, args.scenario)
        results.append(r)
        print(
            f"   tps={r.throughput_tps} "
            f"TTFT p99={r.ttft_p99}ms "
            f"TPOT p99={r.tpot_p99}ms "
            f"pods={r.pod_count}"
        )

    _stop_gpu_scraping.set()

    plot_scaling_curves(results, args.scenario)

    # Print results table
    print("\nScaling Results Table:")
    header = (
        f"{'Users':>6} {'tps':>8} {'TTFT p50':>10} {'TTFT p95':>10} "
        f"{'TTFT p99':>10} {'TPOT p99':>10} {'Pods':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.users:>6} {r.throughput_tps:>8.1f} "
            f"{r.ttft_p50:>10.1f} {r.ttft_p95:>10.1f} {r.ttft_p99:>10.1f} "
            f"{r.tpot_p99:>10.1f} {r.pod_count:>6}"
        )


if __name__ == "__main__":
    main()
