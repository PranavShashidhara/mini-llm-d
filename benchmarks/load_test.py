"""
benchmarks/load_test.py

Locust load test for mini-llmd.

Simulates three prompt scenarios (A / B / C) using a ShareGPT-style
prompt-length distribution. TTFT and TPOT are tracked separately.

Usage:
    # Headless (CI / benchmark scripts)
    locust -f load_test.py \
        --headless \
        --host http://$(minikube ip):30900 \
        --users 16 \
        --spawn-rate 2 \
        --run-time 120s \
        --scenario C

    # Interactive UI
    locust -f load_test.py --host http://$(minikube ip):30900
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from locust import HttpUser, between, events, task
from locust.env import Environment

# ---------------------------------------------------------------------------
# Prompt corpus helpers
# ---------------------------------------------------------------------------

# Short system prefix used across all prompts to seed prefix-cache reuse
_COMMON_PREFIX = (
    "You are a helpful, concise AI assistant. "
    "Answer the user's question clearly and accurately. "
)

_SCENARIO = os.environ.get("SCENARIO", "C").upper()  # A | B | C

# Approximate token counts per word: 1 word ≈ 1.3 tokens
_WORDS_PER_TOKEN = 0.75


def _words(n_tokens: int) -> int:
    return max(1, int(n_tokens * _WORDS_PER_TOKEN))


def _lorem(n_words: int) -> str:
    """Generate a deterministic pseudo-Lorem prompt of ~n_words words."""
    base = (
        "the quick brown fox jumps over the lazy dog "
        "artificial intelligence language model inference cluster "
        "kubernetes autoscaling vllm throughput latency tokens "
    ).split()
    words = []
    for i in range(n_words):
        words.append(base[i % len(base)])
    return " ".join(words)


def _sample_prompt() -> tuple[str, int]:
    """
    Return (prompt_text, requested_max_tokens) sampled from the
    scenario distribution.

    Scenario A: short prompts  (<128 tokens)
    Scenario B: long prompts   (512–2048 tokens)
    Scenario C: ShareGPT mixed distribution
    """
    if _SCENARIO == "A":
        n_in = random.randint(32, 128)
        n_out = random.randint(64, 256)
    elif _SCENARIO == "B":
        n_in = random.randint(512, 2048)
        n_out = random.randint(128, 512)
    else:
        # ShareGPT-style: log-normal input length, exponential output
        n_in = int(np.clip(np.random.lognormal(5.5, 1.0), 32, 2048))
        n_out = int(np.clip(np.random.exponential(200), 32, 1024))

    prompt = _COMMON_PREFIX + _lorem(_words(n_in))
    return prompt, n_out


# ---------------------------------------------------------------------------
# Per-request TTFT / TPOT tracking
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    scenario: str
    prompt_tokens_approx: int
    start: float = field(default_factory=time.perf_counter)
    ttft: Optional[float] = None
    token_times: list[float] = field(default_factory=list)
    total_tokens: int = 0
    success: bool = False


_all_metrics: list[RequestMetrics] = []


# ---------------------------------------------------------------------------
# Locust user
# ---------------------------------------------------------------------------


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def generate(self):
        prompt, max_tokens = _sample_prompt()
        m = RequestMetrics(
            scenario=_SCENARIO,
            prompt_tokens_approx=len(prompt.split()),
        )

        try:
            with self.client.post(
                "/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": True,
                },
                stream=True,
                catch_response=True,
                name=f"/generate [scenario {_SCENARIO}]",
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                    return

                prev_time = m.start
                for line in resp.iter_lines():
                    if not line:
                        continue
                    raw = line
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    raw = raw.removeprefix("data: ").strip()
                    if not raw:
                        continue

                    now = time.perf_counter()

                    if m.ttft is None:
                        m.ttft = (now - m.start) * 1000  # ms

                    inter_token = (now - prev_time) * 1000
                    m.token_times.append(inter_token)
                    prev_time = now
                    m.total_tokens += 1

                    try:
                        data = json.loads(raw)
                        if data.get("finished"):
                            break
                    except json.JSONDecodeError:
                        pass

                m.success = True
                resp.success()

        except Exception as exc:
            self.environment.events.request.fire(
                request_type="POST",
                name=f"/generate [scenario {_SCENARIO}]",
                response_time=0,
                response_length=0,
                exception=exc,
            )
        finally:
            _all_metrics.append(m)


# ---------------------------------------------------------------------------
# Summary report on test stop
# ---------------------------------------------------------------------------


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    successful = [m for m in _all_metrics if m.success and m.ttft is not None]
    if not successful:
        print("No successful requests recorded.")
        return

    ttfts = [m.ttft for m in successful]
    tpots = []
    for m in successful:
        if len(m.token_times) > 1:
            # Skip first inter-token time (includes TTFT)
            tpots.extend(m.token_times[1:])

    def pct(data, p):
        return round(float(np.percentile(data, p)), 2) if data else 0

    print("\n" + "=" * 60)
    print(f"mini-llmd Benchmark Summary — Scenario {_SCENARIO}")
    print("=" * 60)
    print(f"  Total requests:    {len(_all_metrics)}")
    print(f"  Successful:        {len(successful)}")
    print(f"  TTFT (ms)  p50={pct(ttfts, 50)}  p95={pct(ttfts, 95)}  p99={pct(ttfts, 99)}")
    if tpots:
        print(f"  TPOT (ms)  p50={pct(tpots, 50)}  p95={pct(tpots, 95)}  p99={pct(tpots, 99)}")
    print("=" * 60 + "\n")

    # Save raw data for profile.py to pick up
    import pickle

    out = os.environ.get("METRICS_OUT", f"/tmp/bench_metrics_{_SCENARIO}.pkl")
    with open(out, "wb") as f:
        pickle.dump(_all_metrics, f)
    print(f"Raw metrics saved to {out}")
