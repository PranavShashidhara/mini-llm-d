"""
mini-llm-d profiling script.

Two modes:
  1. gpu_profile  — polls nvidia-smi on each inference pod at 1s intervals
                    and writes per-pod GPU stats to results/gpu_profile.csv
  2. plot_curves  — reads Locust CSV output + GPU profile and produces
                    results/scaling_curves.png

Usage:
    # Start GPU profiling (run during the Locust benchmark):
    python profile.py gpu_profile --pods inference-0 inference-1

    # Generate the scaling curve chart after the benchmark:
    python profile.py plot_curves \
        --locust-csv results/benchmark_stats.csv \
        --gpu-csv results/gpu_profile.csv \
        --events results/hpa_events.json \
        --output results/scaling_curves.png
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU profiling via nvidia-smi (runs outside cluster; uses kubectl exec)
# ---------------------------------------------------------------------------

GPU_FIELDS = [
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "clocks_throttle_reasons.hw_slowdown",
]
GPU_QUERY = ",".join(GPU_FIELDS)


def poll_gpu(pod_name: str, namespace: str = "mini-llm-d") -> dict:
    """Run nvidia-smi inside the given pod and return parsed metrics."""
    cmd = [
        "kubectl", "exec", pod_name, "-n", namespace, "--",
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, timeout=5, text=True).strip()
        values = [v.strip() for v in out.split(",")]
        return {
            "pod": pod_name,
            "timestamp": datetime.utcnow().isoformat(),
            "gpu_util_pct": float(values[0]),
            "mem_util_pct": float(values[1]),
            "mem_used_mb": float(values[2]),
            "mem_total_mb": float(values[3]),
            "hw_slowdown": values[4],
        }
    except Exception as exc:
        return {"pod": pod_name, "timestamp": datetime.utcnow().isoformat(), "error": str(exc)}


def run_gpu_profile(pods: list[str], output: Path, interval: float = 1.0):
    """Poll GPU stats for all pods at `interval` seconds and write to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Profiling {pods}, writing to {output}. Press Ctrl-C to stop.")
    fieldnames = ["pod", "timestamp", "gpu_util_pct", "mem_util_pct",
                  "mem_used_mb", "mem_total_mb", "hw_slowdown", "error"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        try:
            while True:
                for pod in pods:
                    row = poll_gpu(pod)
                    writer.writerow(row)
                    f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("GPU profiling stopped.")


# ---------------------------------------------------------------------------
# Scaling curve plotting
# ---------------------------------------------------------------------------

def plot_curves(locust_csv: Path, gpu_csv: Path, events_json: Path, output: Path):
    """
    Produce scaling_curves.png: throughput & latency vs concurrency,
    annotated with HPA scale events and overlaid GPU utilization.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import pandas as pd
    except ImportError:
        print("ERROR: pip install matplotlib pandas")
        sys.exit(1)

    # --- Load Locust stats CSV ---
    df = pd.read_csv(locust_csv)
    # Filter to aggregate row only
    agg = df[df["Name"] == "Aggregated"].copy()
    # Locust stats_history CSV has a Timestamp column
    # If using stats_history CSV use that; otherwise fall back to stat rows
    if "Timestamp" in agg.columns:
        agg["Timestamp"] = pd.to_datetime(agg["Timestamp"], unit="s")

    # --- Load GPU profile ---
    gpu_df = None
    if gpu_csv.exists():
        gpu_df = pd.read_csv(gpu_csv)
        gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"])

    # --- Load HPA events ---
    events = []
    if events_json.exists():
        with events_json.open() as f:
            events = json.load(f)  # list of {"time": ISO, "replicas": N, "reason": str}

    # --- Plot ---
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("mini-llm-d — Throughput / Latency / GPU Scaling", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    ax_tput = fig.add_subplot(gs[0])
    ax_lat  = fig.add_subplot(gs[1])
    ax_gpu  = fig.add_subplot(gs[2])

    # Throughput
    if "Requests/s" in agg.columns and "Timestamp" in agg.columns:
        ax_tput.plot(agg["Timestamp"], agg["Requests/s"], color="#2196F3", label="req/s")
    ax_tput.set_ylabel("Requests / sec")
    ax_tput.set_title("Throughput")
    ax_tput.legend()

    # Latency percentiles
    for col, color, label in [
        ("50%", "#4CAF50", "p50"),
        ("95%", "#FF9800", "p95"),
        ("99%", "#F44336", "p99"),
    ]:
        if col in agg.columns and "Timestamp" in agg.columns:
            ax_lat.plot(agg["Timestamp"], agg[col], color=color, label=label)
    ax_lat.axhline(100, color="red", linestyle="--", linewidth=0.8, label="100ms target")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Latency Percentiles")
    ax_lat.legend()

    # GPU utilization
    if gpu_df is not None:
        for pod_name, pod_df in gpu_df.groupby("pod"):
            ax_gpu.plot(pod_df["timestamp"], pod_df["gpu_util_pct"], label=pod_name)
    ax_gpu.set_ylabel("GPU Util %")
    ax_gpu.set_title("GPU Utilization per Pod")
    ax_gpu.set_ylim(0, 105)
    ax_gpu.legend()

    # Annotate HPA events on all axes
    for ev in events:
        t = pd.to_datetime(ev["time"])
        for ax in [ax_tput, ax_lat, ax_gpu]:
            ax.axvline(t, color="purple", linestyle=":", linewidth=1.0)
        ax_tput.text(t, ax_tput.get_ylim()[1] * 0.9,
                     f"→{ev['replicas']}p", fontsize=7, color="purple", rotation=90)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="mini-llm-d profiling utilities")
    sub = parser.add_subparsers(dest="cmd")

    # gpu_profile subcommand
    gp = sub.add_parser("gpu_profile", help="Poll nvidia-smi on inference pods")
    gp.add_argument("--pods", nargs="+", default=["inference-0", "inference-1"])
    gp.add_argument("--output", type=Path, default=Path("results/gpu_profile.csv"))
    gp.add_argument("--interval", type=float, default=1.0)

    # plot_curves subcommand
    pc = sub.add_parser("plot_curves", help="Generate scaling_curves.png")
    pc.add_argument("--locust-csv",  type=Path, default=Path("results/benchmark_stats_history.csv"))
    pc.add_argument("--gpu-csv",     type=Path, default=Path("results/gpu_profile.csv"))
    pc.add_argument("--events",      type=Path, default=Path("results/hpa_events.json"))
    pc.add_argument("--output",      type=Path, default=Path("results/scaling_curves.png"))

    args = parser.parse_args()

    if args.cmd == "gpu_profile":
        run_gpu_profile(args.pods, args.output, args.interval)
    elif args.cmd == "plot_curves":
        plot_curves(args.locust_csv, args.gpu_csv, args.events, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
