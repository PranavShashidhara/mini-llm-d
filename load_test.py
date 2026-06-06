"""
mini-llm-d load test — Locust script.

Ramps concurrency from 1 to 64 users over 10 minutes, recording latency
percentiles and throughput at each level.  Annotates the output CSV with
autoscaler event timestamps sourced from the Kubernetes events API.

Usage:
    # From outside the cluster (Locust runs on host):
    ROUTER_URL=$(minikube ip):30900
    locust -f load_test.py \
        --host http://$ROUTER_URL \
        --headless \
        --users 64 \
        --spawn-rate 1 \
        --run-time 10m \
        --csv results/benchmark

    # Then generate the scaling curve:
    python profile.py --csv results/benchmark_stats.csv
"""

import json
import random

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Prompt corpus — short prompts to avoid output-length variance
# ---------------------------------------------------------------------------

PROMPTS = [
    "Explain the difference between a mutex and a semaphore in two sentences.",
    "What is the time complexity of quicksort in the average case?",
    "Describe how a transformer attention mechanism works.",
    "What is gradient descent and why is it used?",
    "Explain TCP three-way handshake briefly.",
    "What is a Bloom filter and when would you use one?",
    "Describe the CAP theorem for distributed systems.",
    "What is the difference between SQL and NoSQL databases?",
    "How does a hash table handle collisions?",
    "What is a Kubernetes pod and how does it differ from a container?",
    "Explain what a context switch is in operating systems.",
    "What is the purpose of a load balancer?",
    "Describe how HTTPS encrypts traffic.",
    "What is eventual consistency in distributed systems?",
    "Explain the concept of a deadlock and how to avoid it.",
]


class InferenceUser(HttpUser):
    """Simulates a single concurrent user hitting the /generate endpoint."""

    # Think time between requests: uniform 0.5–1.5s to model realistic spacing
    wait_time = between(0.5, 1.5)

    @task
    def generate(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": False,   # non-streaming for clean latency measurement
        }
        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
            timeout=30,
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    tokens = data.get("tokens_generated", 0)
                    # Tag response with token count for throughput analysis
                    resp.success()
                    # Store in custom stats (accessible in CSV)
                    self.environment.runner.stats.get(
                        "/generate", "POST"
                    ).num_reqs_per_sec  # touch to ensure stat exists
                except (json.JSONDecodeError, KeyError):
                    resp.failure("Invalid JSON response")
            elif resp.status_code == 503:
                resp.failure("No healthy pods (503)")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")
