#!/usr/bin/env python3
"""
benchmarks/rdma_kv_bench.py
----------------------------
RDMA write latency vs KV cache size (sequence-length sweep).

Measures wall-clock time of: MR registration + ibv_post_send + CQ poll
across sequence lengths that map to realistic KV cache sizes for Llama 3.2 3B.

--no-gpu flag (or MINI_LLM_NO_GPU=1):
  Uses the CPU-simulation path in kv_transfer.py. All output rows, chart,
  and JSON results are produced identically — only the latency numbers will
  reflect simulated loopback (~10 Gb/s ceiling) rather than real RoCEv2.

Output
------
  results/rdma_kv_latency.json   — raw per-sample data
  results/rdma_kv_latency.png   — p50 / p99 vs sequence length (matplotlib)
  Stdout table matching the spec:

  Seq Len │ KV Size (MB) │ p50 (ms) │ p99 (ms) │ Mode
  ────────┼──────────────┼──────────┼──────────┼──────────────
     512  │    24.0      │   xx.x   │   xx.x   │ sim-loopback
    1024  │    48.0      │   xx.x   │   xx.x   │ sim-loopback
    2048  │    96.0      │   xx.x   │   xx.x   │ sim-loopback

Usage
-----
    # Laptop (no GPU)
    python benchmarks/rdma_kv_bench.py --no-gpu

    # Real RoCEv2 hardware
    python benchmarks/rdma_kv_bench.py

    # Custom sweep
    python benchmarks/rdma_kv_bench.py --no-gpu --seq-lens 256 512 1024 4096 \
        --iterations 20 --warmup 3
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import statistics
from pathlib import Path
from typing import Optional

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdma.kv_transfer import RDMAKVTransfer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Llama 3.2 3B KV cache size calculation
# ---------------------------------------------------------------------------
# Architecture: 28 layers, 8 KV heads, head_dim=64, bfloat16 (2 bytes)
# KV size per token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
#                   = 2 * 28 * 8 * 64 * 2 = 57,344 bytes ≈ 56 KB per token

KV_BYTES_PER_TOKEN = 2 * 28 * 8 * 64 * 2   # 57,344 bytes


def seq_len_to_kv_bytes(seq_len: int) -> int:
    return seq_len * KV_BYTES_PER_TOKEN


# Spec table reference values (approximate, rounded):
# 512 tokens  → ~24 MB
# 1024 tokens → ~48 MB
# 2048 tokens → ~96 MB


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    xfer: RDMAKVTransfer,
    seq_lens: list[int],
    iterations: int = 10,
    warmup: int = 2,
) -> list[dict]:
    """
    For each sequence length, transfer the corresponding KV cache size
    *iterations* times and collect latency samples.

    Returns a list of row dicts (one per seq_len).
    """
    rows = []
    mode = "sim-loopback" if xfer.no_gpu else "RoCEv2"

    for seq_len in seq_lens:
        kv_bytes = seq_len_to_kv_bytes(seq_len)
        samples: list[float] = []

        logger.info(
            "Benchmarking seq_len=%d  kv_bytes=%.1f MB  mode=%s",
            seq_len, kv_bytes / 1e6, mode,
        )

        for i in range(warmup + iterations):
            result = xfer.simulate_kv_handoff(kv_bytes)
            if not result.success:
                logger.warning("Transfer failed for seq_len=%d iter=%d: %s",
                               seq_len, i, result.error)
                continue
            if i >= warmup:                 # skip warmup samples
                samples.append(result.latency_ms)

        if not samples:
            logger.error("No successful samples for seq_len=%d", seq_len)
            continue

        samples.sort()
        p50 = statistics.median(samples)
        p99 = samples[int(len(samples) * 0.99)] if len(samples) > 1 else samples[-1]
        p_min = samples[0]
        p_max = samples[-1]
        mean = statistics.mean(samples)

        rows.append({
            "seq_len": seq_len,
            "kv_bytes": kv_bytes,
            "kv_mb": round(kv_bytes / 1e6, 1),
            "mode": mode,
            "samples": len(samples),
            "p50_ms": round(p50, 3),
            "p99_ms": round(p99, 3),
            "min_ms": round(p_min, 3),
            "max_ms": round(p_max, 3),
            "mean_ms": round(mean, 3),
            "raw": samples,
        })

    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    """Print a human-readable benchmark table to stdout."""
    header = (
        f"{'Seq Len':>8} │ {'KV Size (MB)':>12} │ "
        f"{'p50 (ms)':>10} │ {'p99 (ms)':>10} │ {'Mode'}"
    )
    sep = "─" * 8 + "┼" + "─" * 14 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 16
    print()
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['seq_len']:>8} │ {r['kv_mb']:>12.1f} │ "
            f"{r['p50_ms']:>10.2f} │ {r['p99_ms']:>10.2f} │ {r['mode']}"
        )
    print()


def save_json(rows: list[dict], path: Path) -> None:
    # Strip raw samples before saving to keep JSON readable, save separately
    summary = [{k: v for k, v in r.items() if k != "raw"} for r in rows]
    path.write_text(json.dumps({"results": summary, "kv_bytes_per_token": KV_BYTES_PER_TOKEN}, indent=2))
    logger.info("JSON results saved to %s", path)


def save_chart(rows: list[dict], path: Path) -> None:
    """Generate a matplotlib bar+line chart of p50/p99 vs seq_len."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart (pip install matplotlib)")
        return

    seq_lens = [r["seq_len"] for r in rows]
    p50s = [r["p50_ms"] for r in rows]
    p99s = [r["p99_ms"] for r in rows]
    labels = [f"{sl}" for sl in seq_lens]
    x = range(len(seq_lens))
    bar_w = 0.35
    mode = rows[0]["mode"] if rows else "unknown"

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_p50 = ax.bar([i - bar_w / 2 for i in x], p50s, bar_w,
                      label="p50", color="#4C72B0", alpha=0.85)
    bars_p99 = ax.bar([i + bar_w / 2 for i in x], p99s, bar_w,
                      label="p99", color="#DD8452", alpha=0.85)

    # Spec reference lines (loopback targets from §9)
    spec_p50 = {512: 20, 1024: None, 2048: None}
    for idx, sl in enumerate(seq_lens):
        target = spec_p50.get(sl)
        if target is not None:
            ax.axhline(y=target, color="red", linestyle="--", linewidth=1.0,
                       label=f"Target p50 ≤{target} ms ({sl} tok)")

    # Value labels on bars
    for bar in [*bars_p50, *bars_p99]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s} tok\n{r['kv_mb']} MB" for s, r in zip(seq_lens, rows)])
    ax.set_xlabel("Sequence Length (tokens) / KV Cache Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        f"RDMA KV-Cache Transfer Latency — mini-llm-d\n"
        f"Mode: {mode}  |  Llama 3.2 3B"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RDMA KV-cache transfer latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode — no ibverbs hardware required")
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[512, 1024, 2048],
                   metavar="N",
                   help="Sequence lengths (tokens) to benchmark")
    p.add_argument("--iterations", type=int, default=10,
                   help="Measurement iterations per seq length (after warmup)")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup iterations (discarded)")
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
    print(f"  mini-llm-d  ·  RDMA KV-Cache Latency Benchmark")
    print(f"{'='*64}")
    print(f"  Mode       : {'CPU-simulation (--no-gpu)' if args.no_gpu else 'Real ibverbs / RoCEv2'}")
    print(f"  Seq lens   : {args.seq_lens}")
    print(f"  Iterations : {args.iterations}  (warmup={args.warmup})")
    print(f"  Output dir : {args.out_dir}")
    print(f"{'='*64}")

    xfer = RDMAKVTransfer(no_gpu=args.no_gpu)
    rows = run_benchmark(xfer, args.seq_lens, args.iterations, args.warmup)

    print_table(rows)

    json_path = args.out_dir / "rdma_kv_latency.json"
    png_path  = args.out_dir / "rdma_kv_latency.png"
    save_json(rows, json_path)
    save_chart(rows, png_path)

    # Quick pass/fail against spec targets
    print("  Spec check (§9 targets):")
    for r in rows:
        if r["seq_len"] == 512:
            target = 20.0
            status = "PASS ✓" if r["p50_ms"] <= target else f"MISS ✗ (target ≤{target} ms)"
            print(f"    512-token p50 = {r['p50_ms']:.2f} ms  →  {status}")
    print()


if __name__ == "__main__":
    main()
