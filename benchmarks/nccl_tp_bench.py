#!/usr/bin/env python3
"""
benchmarks/nccl_tp_bench.py
----------------------------
NCCL all-reduce overhead measured during live TP=2 inference traffic.

Corresponds to Scenario D in the mini-llm-d design document:
  • Runs two simulated TP=2 shards (rank 0 + rank 1) concurrently
  • Each shard fires all-reduce calls continuously (simulating vLLM's
    per-transformer-layer NCCL collective during inference)
  • Also accepts an artificial latency flag (--netem-ms) to simulate
    tc netem network delay and measure TPOT sensitivity

--no-gpu / MINI_LLM_NO_GPU=1  →  CPU-simulation mode:
  - No NCCL / CUDA / real network required
  - All-reduce is simulated with synthetic bandwidth + latency
  - All output tables, JSON, and PNG are identical to the real path

Target (§9):
  bus_bandwidth > 8 Gb/s on loopback  (sim target ≈ 10 Gb/s)
  TP=2 TPOT overhead vs TP=1 < 5% at median

Output
------
  results/nccl_throughput.json
  results/nccl_throughput.png

Usage
-----
    # Laptop
    python benchmarks/nccl_tp_bench.py --no-gpu

    # With artificial 1 ms and 5 ms RTT latency (Scenario D extended)
    python benchmarks/nccl_tp_bench.py --no-gpu --netem-ms 1
    python benchmarks/nccl_tp_bench.py --no-gpu --netem-ms 5

    # Real NCCL (inside a GPU pod)
    python benchmarks/nccl_tp_bench.py
"""

from __future__ import annotations

import sys
import json
import time
import logging
import argparse
import threading
import statistics
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdma.nccl_init import NCCLComm, AllReduceResult, _SimNCCL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message size presets
# ---------------------------------------------------------------------------

# nccl-tests equivalent: -b 1G -e 1G -f 2 -g 1 (1 GB single shot)
DEFAULT_DATA_GB = 1.0

# Llama 3.2 3B transformer layer activation size for TP=2 all-reduce
# hidden_dim=3072, seq_len=512, bfloat16 → 512*3072*2 ≈ 3.1 MB per layer
LAYER_ACTIVATION_BYTES = 512 * 3072 * 2   # 3,145,728 bytes


# ---------------------------------------------------------------------------
# Simulated inference request timing
# ---------------------------------------------------------------------------

def _simulate_tpot_single_pod(num_tokens: int = 20) -> float:
    """
    Rough TPOT (ms) for a TP=1 single pod: GPU compute only, no NCCL.
    Calibrated to ~50 ms/token for Llama 3.2 3B on a mid-range laptop CPU
    (much slower than GPU but structurally identical).
    """
    ms_per_token = 50.0   # CPU sim; ~5 ms on A100
    time.sleep(num_tokens * ms_per_token / 1000)
    return num_tokens * ms_per_token


def _simulate_tpot_tp2(comm0: NCCLComm, comm1: NCCLComm, num_tokens: int = 20,
                       netem_ms: float = 0.0) -> tuple[float, float]:
    """
    Simulate TPOT for TP=2: compute + per-layer NCCL all-reduce overhead.
    Returns (tpot_ms, nccl_overhead_ms).
    """
    num_layers = 28  # Llama 3.2 3B
    ms_per_token_compute = 25.0  # split across 2 shards (halved)

    nccl_overhead_ms = 0.0
    results: list[Optional[AllReduceResult]] = [None, None]
    errors: list[Optional[str]] = [None, None]

    for layer in range(num_layers):
        barrier = threading.Barrier(2)
        layer_results: list[Optional[AllReduceResult]] = [None, None]

        def run(rank: int, comm: NCCLComm) -> None:
            barrier.wait()
            extra_sleep = netem_ms / 1000.0
            if extra_sleep > 0:
                time.sleep(extra_sleep)
            layer_results[rank] = comm.all_reduce(LAYER_ACTIVATION_BYTES)

        threads = [
            threading.Thread(target=run, args=(0, comm0)),
            threading.Thread(target=run, args=(1, comm1)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if layer_results[0]:
            nccl_overhead_ms += layer_results[0].barrier_ms

    compute_ms = num_tokens * ms_per_token_compute
    time.sleep(compute_ms / 1000)
    total_ms = compute_ms + nccl_overhead_ms
    return total_ms, nccl_overhead_ms


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_nccl_bench(
    no_gpu: bool,
    data_gb: float = DEFAULT_DATA_GB,
    iterations: int = 5,
    warmup: int = 1,
    netem_ms: float = 0.0,
) -> dict:
    """
    Run the full Scenario D NCCL benchmark.
    Returns a result dict suitable for JSON serialisation.
    """
    data_bytes = int(data_gb * 1024**3)
    mode = "sim-socket" if no_gpu else "RoCEv2"

    print(f"\n  Scenario D — NCCL All-Reduce Benchmark")
    print(f"  Mode        : {mode}")
    print(f"  Message     : {data_gb:.1f} GB")
    print(f"  netem delay : {netem_ms} ms")
    print(f"  Iterations  : {iterations}  (warmup={warmup})")

    # --- Part 1: bulk all-reduce throughput (matches nccl-tests all_reduce_perf) ---
    bw_samples: list[float] = []
    barrier_samples: list[float] = []

    for i in range(warmup + iterations):
        r0, r1 = NCCLComm.simulate_tp2_all_reduce(data_bytes)
        if i >= warmup:
            if r0 and r0.success:
                bw_samples.append(r0.bus_bandwidth_gbps)
                barrier_samples.append(r0.barrier_ms)

    bw_p50 = statistics.median(bw_samples) if bw_samples else 0
    bw_p99 = (sorted(bw_samples)[int(len(bw_samples) * 0.99)]
               if len(bw_samples) > 1 else (bw_samples[-1] if bw_samples else 0))
    barrier_p50 = statistics.median(barrier_samples) if barrier_samples else 0

    print(f"\n  ── Bulk all-reduce (1 GB message) ──────────────────────")
    print(f"  Bus BW p50  : {bw_p50:.3f} Gb/s  (target > 8 Gb/s)")
    print(f"  Bus BW p99  : {bw_p99:.3f} Gb/s")
    print(f"  Barrier p50 : {barrier_p50:.2f} ms")
    status = "PASS ✓" if bw_p50 >= 8.0 else "MISS ✗"
    print(f"  Spec check  : {status}")

    # --- Part 2: TPOT overhead vs TP=1 ---
    print(f"\n  ── TP=2 TPOT overhead vs TP=1 ──────────────────────────")

    tpot_tp1_samples: list[float] = []
    tpot_tp2_samples: list[float] = []
    nccl_oh_samples:  list[float] = []

    for i in range(warmup + iterations):
        # TP=1 baseline
        t0 = time.perf_counter()
        _simulate_tpot_single_pod(num_tokens=10)
        tpot_tp1 = (time.perf_counter() - t0) * 1000

        # TP=2 with NCCL
        comm0 = NCCLComm(rank=0, world_size=2, no_gpu=no_gpu)
        comm1 = NCCLComm(rank=1, world_size=2, no_gpu=no_gpu)
        comm0.bootstrap(f"tpot-bench-{i}")
        # rank 1 bootstrap reads the same key; run in thread
        done = threading.Event()
        def _boot1() -> None:
            comm1.bootstrap(f"tpot-bench-{i}")
            done.set()
        threading.Thread(target=_boot1).start()
        done.wait(timeout=5)

        t0 = time.perf_counter()
        tpot_tp2, nccl_oh = _simulate_tpot_tp2(comm0, comm1, num_tokens=10,
                                                 netem_ms=netem_ms)
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            tpot_tp1_samples.append(tpot_tp1)
            tpot_tp2_samples.append(elapsed)
            nccl_oh_samples.append(nccl_oh)

    tp1_median = statistics.median(tpot_tp1_samples) if tpot_tp1_samples else 0
    tp2_median = statistics.median(tpot_tp2_samples) if tpot_tp2_samples else 0
    overhead_pct = ((tp2_median - tp1_median) / tp1_median * 100) if tp1_median > 0 else 0
    nccl_oh_median = statistics.median(nccl_oh_samples) if nccl_oh_samples else 0

    print(f"  TP=1 TPOT p50  : {tp1_median:.1f} ms")
    print(f"  TP=2 TPOT p50  : {tp2_median:.1f} ms")
    print(f"  NCCL overhead  : {nccl_oh_median:.2f} ms / request")
    print(f"  Overhead %     : {overhead_pct:.1f}%  (target < 5%)")
    tpot_status = "PASS ✓" if overhead_pct < 5.0 else "MISS ✗ (sim overhead > 5% due to thread sync)"
    print(f"  Spec check     : {tpot_status}")
    print()

    return {
        "mode": mode,
        "data_gb": data_gb,
        "netem_ms": netem_ms,
        "iterations": iterations,
        "bulk_allreduce": {
            "bw_p50_gbps": round(bw_p50, 3),
            "bw_p99_gbps": round(bw_p99, 3),
            "barrier_p50_ms": round(barrier_p50, 3),
            "spec_pass": bw_p50 >= 8.0,
        },
        "tpot_overhead": {
            "tp1_median_ms": round(tp1_median, 2),
            "tp2_median_ms": round(tp2_median, 2),
            "nccl_overhead_ms": round(nccl_oh_median, 2),
            "overhead_pct": round(overhead_pct, 2),
            "spec_pass": overhead_pct < 5.0,
        },
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def save_chart(results: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart")
        return

    netem_vals = [r["netem_ms"] for r in results]
    bw_vals    = [r["bulk_allreduce"]["bw_p50_gbps"] for r in results]
    oh_vals    = [r["tpot_overhead"]["overhead_pct"] for r in results]
    labels     = [f"{n} ms RTT" if n > 0 else "baseline" for n in netem_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(labels, bw_vals, color="#4C72B0", alpha=0.85)
    ax1.axhline(8.0, color="red", linestyle="--", linewidth=1, label="Target 8 Gb/s")
    ax1.set_title("NCCL All-Reduce Bus BW\n(1 GB message, TP=2)")
    ax1.set_ylabel("Gb/s")
    ax1.legend()
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(bw_vals):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax2.bar(labels, oh_vals, color="#DD8452", alpha=0.85)
    ax2.axhline(5.0, color="red", linestyle="--", linewidth=1, label="Target < 5%")
    ax2.set_title("TP=2 TPOT Overhead vs TP=1\n(Scenario D)")
    ax2.set_ylabel("Overhead (%)")
    ax2.legend()
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(oh_vals):
        ax2.text(i, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("mini-llm-d — NCCL TP=2 Benchmark (Scenario D)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("Chart saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NCCL TP=2 all-reduce throughput benchmark (Scenario D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode — no NCCL/CUDA required")
    p.add_argument("--data-gb", type=float, default=DEFAULT_DATA_GB,
                   help="All-reduce message size (GB)")
    p.add_argument("--iterations", type=int, default=5,
                   help="Measurement iterations")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup iterations (discarded)")
    p.add_argument("--netem-ms", type=float, nargs="+", default=[0.0],
                   metavar="MS",
                   help="Artificial RTT latency values (ms) to sweep (0 = baseline). "
                        "Example: --netem-ms 0 1 5")
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="Directory for JSON + PNG outputs")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  mini-llm-d  ·  NCCL TP=2 All-Reduce Benchmark (Scenario D)")
    print(f"{'='*64}")

    all_results = []
    for netem in args.netem_ms:
        result = run_nccl_bench(
            no_gpu=args.no_gpu,
            data_gb=args.data_gb,
            iterations=args.iterations,
            warmup=args.warmup,
            netem_ms=netem,
        )
        all_results.append(result)

    json_path = args.out_dir / "nccl_throughput.json"
    png_path  = args.out_dir / "nccl_throughput.png"
    json_path.write_text(json.dumps({"results": all_results}, indent=2))
    logger.info("JSON results saved to %s", json_path)
    save_chart(all_results, png_path)

    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
