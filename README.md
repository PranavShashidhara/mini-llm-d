# mini-llm-d

**Kubernetes-Native Auto-Scaling LLM Inference Cluster**  
*minikube + vLLM + Llama 3.2 3B*

A from-scratch re-implementation of the core ideas behind [llm-d](https://github.com/llm-d/llm-d) — the open-source Kubernetes-native LLM inference system. Every layer is built by hand, running entirely on a local minikube cluster with GPU passthrough.

---

## Key Features

- Horizontal pod scaling (1–4 replicas) driven by custom `tokens/sec` HPA metric
- **Prefix-cache aware routing** — routes requests to pods with warm KV caches
- **Prefill/Decode disaggregation** — splits P and D into separate pod pools
- **Workload-Variant Autoscaler (WVA)** — independently scales prefill and decode pools
- Full observability: Prometheus + Grafana dashboards + OpenTelemetry + Jaeger
- Realistic benchmarks using ShareGPT prompt distribution (Locust)
- Sub-100ms p99 latency target at peak concurrency

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        minikube                          │
│                                                          │
│   Locust ──▶ Router (prefix-cache aware)                 │
│                 │                                        │
│         ┌───────┴────────┐                               │
│         ▼                ▼                               │
│   Prefill Pod(s)    Decode Pod(s)                        │
│   (vLLM + FastAPI)  (vLLM + FastAPI)                     │
│         │                │                               │
│         └──── tmpfs ─────┘   (shared KV cache)          │
│                                                          │
│   Prometheus ◀── /metrics from router + pods            │
│   Grafana    ◀── Prometheus                              │
│   Jaeger     ◀── OpenTelemetry spans                     │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

> **Shortcut:** if you have `make`, the whole pipeline is one command — `make all`
> (runs setup → model download → cluster → build → deploy). Run `make` on its own
> to see all targets. The manual steps below are what `make all` wraps.

### Prerequisites

- Docker Desktop with NVIDIA Container Toolkit
- minikube ≥ 1.32
- kubectl, helm, python 3.11+
- NVIDIA GPU with ≥ 8GB VRAM (required for the inference pod)
- `uv` and `huggingface-cli` (installed/used by `setup.sh`)

### 1. Set up Python environments

`setup.sh` installs `uv` if needed and creates three isolated virtualenvs:
`inference/.venv` (vLLM — needs a GPU), `router/.venv` (CPU only), and
`benchmarks/.venv` (CPU only). It handles the CUDA-torch-before-vLLM install
ordering automatically.

```bash
./setup.sh              # full setup (inference env requires a GPU)
./setup.sh --no-gpu     # router + benchmarks only, skip the vLLM/inference env
./setup.sh --check      # verify the environment without installing anything
```

### 2. Download the model (first time only)

The inference pod reads the model from `models/llama-3.2-3b-instruct`
(see `inference/config.yaml`). Download it once:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir models/llama-3.2-3b-instruct
```

> Note: `meta-llama/Llama-3.2-3B-Instruct` is a gated model. Accept the license
> on Hugging Face and run `huggingface-cli login` first if you haven't already.

### 3. Start the cluster

```bash
./scripts/cluster-up.sh
```

### 4. Build and load images

```bash
./scripts/build-images.sh
```

### 5. Deploy the full stack

```bash
./scripts/deploy.sh
```

### 6. Run benchmarks

```bash
source benchmarks/.venv/bin/activate
export ROUTER_URL=http://$(minikube ip):30900
python benchmarks/routing_comparison.py
```

---

## Repository Structure

```
mini-llm-d/
├── inference/
│   ├── Dockerfile          # vLLM serving container
│   ├── server.py           # FastAPI + vLLM engine + OTEL tracing
│   └── config.yaml         # Model and runtime config
├── router/
│   ├── router.py           # Prefix-cache aware routing + circuit breaker
│   ├── metrics.py          # Queue depth, TTFT/TPOT tracking
│   └── prefix_cache.py     # Prefix hash computation + pod-cache map
├── k8s/
│   ├── deployment.yaml     # Inference pod StatefulSet
│   ├── service.yaml        # ClusterIP / NodePort / headless services
│   ├── hpa.yaml            # HPA on tokens_per_second_per_pod
│   ├── wva.yaml            # Workload-Variant Autoscaler
│   ├── custom-metrics.yaml # Prometheus Adapter rules
│   ├── jaeger.yaml         # Jaeger all-in-one
│   └── grafana-dashboards/ # Dashboard JSON provisioned via ConfigMap
├── benchmarks/
│   ├── load_test.py        # Locust + ShareGPT distribution
│   ├── profile.py          # GPU utilization + scaling_curves.png
│   └── routing_comparison.py  # Round-robin vs prefix-cache benchmark
├── results/                # Generated charts (gitignored binaries)
├── scripts/
│   ├── cluster-up.sh
│   ├── build-images.sh
│   └── deploy.sh
├── setup.sh                # Creates the three Python venvs (uv)
├── Makefile                # Convenience targets wrapping the scripts (make all, make bench, ...)
└── README.md
```

---

## Week-by-Week Plan

| Week | Focus | Done when |
|------|-------|-----------|
| 1 | Single-pod baseline | Stable TTFT/TPOT numbers across all 3 scenarios |
| 2 | Router + Observability + Tracing | Custom metric visible to API server; Grafana live |
| 3 | HPA + WVA + Full benchmark | scaling_curves.png + routing_comparison.png generated |
| 4 | P/D disaggregation + PR | TTFT improvement on long-prompt scenario documented |

---

## Benchmark Scenarios

| Scenario | Prompt Length | Primary Stress |
|----------|--------------|----------------|
| A | < 128 tokens | Decode throughput |
| B | 512–2048 tokens | Prefill; benefits most from P/D disaggregation + prefix cache |
| C | ShareGPT mixed | Realistic production traffic shape |

Expected result: prefix-cache routing shows **15–40% TTFT reduction** on scenarios B and C.

---

## Upstream Contribution

`results/routing_comparison.png` is the primary artifact for the planned PR to [llm-d](https://github.com/llm-d/llm-d) demonstrating prefix-cache routing improvements with real benchmark data.
