# mini-llm-d

Kubernetes-native auto-scaling LLM inference cluster — local minikube re-implementation of [llm-d](https://github.com/llm-d) core concepts.

Serves **Llama 3.2 3B** across dynamically scaled vLLM pods with a custom Python request router, token-throughput-driven HPA, and full observability under load.

---

## Quick Start

```bash
# 0. Prerequisites
#    - Docker with NVIDIA Container Toolkit
#    - minikube, kubectl, helm
#    - (optional) HF_TOKEN for auto-downloading weights

# 1. First-time full setup
./setup.sh all

# 2. Smoke test
./setup.sh smoke

# 3. Run the full benchmark (Week 3)
./setup.sh benchmark
```

### Step-by-step (manual)

```bash
./setup.sh check       # verify host deps
./setup.sh start       # minikube + NVIDIA device plugin
./setup.sh build       # build + load Docker images
./setup.sh monitoring  # Prometheus + custom metric adapter
./setup.sh deploy      # inference StatefulSet + router + HPA
./setup.sh smoke       # quick generation test
./setup.sh benchmark   # Locust load test + GPU profiling
```

---

## Architecture

```
Locust  →  Router (port 30900)  →  inference-0 (vLLM + FastAPI)
                                →  inference-1
                                →  inference-N  (HPA scales 1–4)

Router  →  /metrics (Prometheus)  →  Prometheus  →  Adapter  →  HPA
```

| Layer | Component | File |
|---|---|---|
| Inference | vLLM + FastAPI | `inference/server.py` |
| Router | Async Python | `router/router.py` |
| Metrics | Sliding window | `router/metrics.py` |
| Autoscaler | Kubernetes HPA | `k8s/hpa.yaml` |
| Benchmarking | Locust | `benchmarks/load_test.py` |
| Profiling | nvidia-smi + matplotlib | `benchmarks/profile.py` |

---

## Key Files

```
mini-llm-d/
├── inference/
│   ├── server.py        # FastAPI + vLLM AsyncLLMEngine
│   ├── config.yaml      # model + runtime config
│   ├── Dockerfile
│   └── requirements.txt
├── router/
│   ├── router.py        # weighted-least-queue routing + circuit breaker
│   ├── metrics.py       # per-pod metrics registry + Prometheus export
│   ├── Dockerfile
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml  # inference StatefulSet + ConfigMap
│   ├── service.yaml     # headless service + router Deployment + NodePort
│   ├── hpa.yaml         # HPA targeting tokens_per_second_per_pod
│   └── custom-metrics.yaml  # Prometheus Adapter rules
├── benchmarks/
│   ├── load_test.py     # Locust scenarios (1–64 users)
│   └── profile.py       # GPU polling + scaling_curves.png
├── results/             # benchmark output (gitignored)
└── setup.sh             # one-shot setup + per-stage commands
```

---

## HPA Metric

The HPA scales on `tokens_per_second_per_pod` — the router's rolling average throughput per healthy pod. Scale-up fires when this exceeds **80 tokens/sec/pod** (update after measuring your single-pod baseline in Week 1).

```yaml
# k8s/hpa.yaml excerpt
metrics:
  - type: External
    external:
      metric:
        name: tokens_per_second_per_pod
      target:
        type: AverageValue
        averageValue: "80"
```

---

## Routing Algorithm

1. Fetch `/metrics` from all pods asynchronously (100ms timeout)
2. Select pod with **shortest queue depth**
3. Break ties by **highest tokens/sec**
4. Circuit-break pods that miss the timeout (30s cooldown)

---

## Targets

| Metric | Target |
|---|---|
| p99 latency at 2-pod peak concurrency | < 100ms |
| Throughput scaling (1→2 pods) | ~linear |
| HPA scale-up lag | < 60s after threshold |

---

## Week-by-Week Plan

| Week | Focus |
|---|---|
| 1 | Single-pod baseline: tokens/sec, p99 latency, GPU util |
| 2 | Router + 2-pod deployment, Prometheus metrics pipeline |
| 3 | HPA + full Locust benchmark, scaling_curves.png |
| 4 | GPU profiling, write-up, upstream llm-d PR |
