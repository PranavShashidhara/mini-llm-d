"""
benchmarks/routing_comparison.py

Side-by-side benchmark of three routing modes:
  1. round_robin
  2. least_queue
  3. prefix_cache

For each mode, runs a Locust load test against scenarios A, B, C
and captures TTFT CDF curves. Outputs:
  results/routing_comparison.png

Usage:
    ROUTER_URL=http://$(minikube ip):30900 python routing_comparison.py

The script patches the router's ROUTING_MODE via its /admin endpoint
(or directly via kubectl env patch) between runs.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:30900")
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ROUTING_MODES = ["round_robin", "least_queue", "prefix_cache"]
SCENARIOS = ["A", "B", "C"]

# Locust settings per run
USERS = int(os.environ.get("BENCH_USERS", "16"))
SPAWN_RATE = int(os.environ.get("BENCH_SPAWN_RATE", "2"))
RUN_TIME = os.environ.get("BENCH_RUN_TIME", "90s")

NAMESPACE = "mini-llmd"
ROUTER_DEPLOYMENT = "router"


def patch_routing_mode(mode: str) -> None:
    """Patch the router Deployment environment variable via kubectl."""
    cmd = [
        "kubectl",
        "-n",
        NAMESPACE,
        "set",
        "env",
        f"deployment/{ROUTER_DEPLOYMENT}",
        f"ROUTING_MODE={mode}",
    ]
    print(f"  → patching router to ROUTING_MODE={mode}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    WARNING: kubectl patch failed: {result.stderr.strip()}")
    # Wait for rollout
    subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "rollout",
            "status",
            f"deployment/{ROUTER_DEPLOYMENT}",
            "--timeout=60s",
        ],
        capture_output=True,
    )
    time.sleep(5)  # let router settle


def run_locust(scenario: str, mode: str) -> list:
    """Run a Locust headless benchmark and return raw RequestMetrics list."""
    metrics_path = f"/tmp/bench_metrics_{scenario}_{mode}.pkl"
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
        str(USERS),
        "--spawn-rate",
        str(SPAWN_RATE),
        "--run-time",
        RUN_TIME,
    ]
    print(f"  → locust scenario={scenario} mode={mode} users={USERS} time={RUN_TIME}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    WARNING: locust exited {result.returncode}")
        print(result.stderr[-500:])

    # Load metrics
    try:
        with open(metrics_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"    WARNING: metrics file not found: {metrics_path}")
        return []


def collect_ttfts(metrics: list) -> list[float]:
    return [m.ttft for m in metrics if m.success and m.ttft is not None]


def cdf(data: list[float], percentiles=None):
    """Return (x, y) for a CDF curve."""
    if not data:
        return np.array([0]), np.array([0])
    if percentiles is None:
        percentiles = np.linspace(0, 100, 500)
    x = np.percentile(sorted(data), percentiles)
    y = percentiles / 100
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Structure: results[mode][scenario] = list[RequestMetrics]
    results: dict[str, dict[str, list]] = {m: {} for m in ROUTING_MODES}

    for mode in ROUTING_MODES:
        print(f"\n{'='*50}")
        print(f"ROUTING MODE: {mode}")
        print(f"{'='*50}")
        patch_routing_mode(mode)

        for scenario in SCENARIOS:
            metrics = run_locust(scenario, mode)
            results[mode][scenario] = metrics
            ttfts = collect_ttfts(metrics)
            if ttfts:
                print(
                    f"    Scenario {scenario}: n={len(ttfts)} "
                    f"p50={np.percentile(ttfts, 50):.1f}ms "
                    f"p95={np.percentile(ttfts, 95):.1f}ms "
                    f"p99={np.percentile(ttfts, 99):.1f}ms"
                )

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "mini-llmd: TTFT CDF — Routing Mode Comparison",
        fontsize=14,
        fontweight="bold",
    )

    colors = {
        "round_robin": "#e74c3c",
        "least_queue": "#3498db",
        "prefix_cache": "#2ecc71",
    }
    labels = {
        "round_robin": "Round Robin (baseline)",
        "least_queue": "Least Queue",
        "prefix_cache": "Prefix Cache (enhanced)",
    }
    scenario_titles = {
        "A": "Scenario A — Short prompts (<128 tok)",
        "B": "Scenario B — Long prompts (512–2048 tok)",
        "C": "Scenario C — ShareGPT mixed",
    }

    for ax, scenario in zip(axes, SCENARIOS):
        ax.set_title(scenario_titles[scenario], fontsize=11)
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("CDF" if scenario == "A" else "")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.99, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(0, 0.955, "p95", fontsize=8, color="gray")
        ax.text(0, 0.995, "p99", fontsize=8, color="gray")
        ax.axvline(100, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.text(102, 0.05, "100ms\ntarget", fontsize=7, color="red", alpha=0.7)

        for mode in ROUTING_MODES:
            ttfts = collect_ttfts(results[mode].get(scenario, []))
            if not ttfts:
                continue
            x, y = cdf(ttfts)
            ax.plot(x, y, color=colors[mode], label=labels[mode], linewidth=2)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "routing_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {out_path}")

    # Print summary table
    print("\nSummary (TTFT p95, ms)")
    print(f"{'Mode':<20} {'Scenario A':>12} {'Scenario B':>12} {'Scenario C':>12}")
    print("-" * 58)
    for mode in ROUTING_MODES:
        row = [labels[mode][:19]]
        for scenario in SCENARIOS:
            ttfts = collect_ttfts(results[mode].get(scenario, []))
            v = f"{np.percentile(ttfts, 95):.1f}" if ttfts else "N/A"
            row.append(f"{v:>12}")
        print("  ".join(row))


if __name__ == "__main__":
    main()
