#!/usr/bin/env bash
# mini-llm-d — local setup and run script
# Brings up the full stack on minikube step by step.
# Run each section manually the first time to verify each layer.

set -euo pipefail

NAMESPACE="mini-llm-d"
MINIKUBE_MEMORY="16g"
MINIKUBE_CPUS="6"

# ============================================================
# 0. Pre-flight checks
# ============================================================
check_deps() {
    echo "==> Checking dependencies..."
    for cmd in minikube kubectl helm docker; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: $cmd not found"; exit 1
        fi
    done
    # Check NVIDIA Container Toolkit
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo "WARNING: NVIDIA runtime not detected in Docker. GPU passthrough may fail."
    fi
    echo "    All dependencies present."
}

# ============================================================
# 1. Start minikube with GPU passthrough
# ============================================================
start_minikube() {
    echo "==> Starting minikube..."
    minikube start \
        --driver=docker \
        --gpus=all \
        --memory="${MINIKUBE_MEMORY}" \
        --cpus="${MINIKUBE_CPUS}"

    # Deploy NVIDIA device plugin
    kubectl apply -f \
        https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml

    echo "==> Verifying GPU visibility..."
    kubectl wait --for=condition=Ready node/minikube --timeout=120s
    kubectl get nodes -o json | python3 -c "
import json,sys
nodes=json.load(sys.stdin)
for n in nodes['items']:
    cap=n['status']['capacity']
    gpu=cap.get('nvidia.com/gpu','0')
    print(f\"  Node {n['metadata']['name']}: {gpu} GPU(s)\")
"
}

# ============================================================
# 2. Build and load images into minikube
# ============================================================
build_images() {
    echo "==> Building images inside minikube Docker daemon..."
    eval "$(minikube docker-env)"

    docker build -t mini-llmd-inference:latest inference/
    echo "    inference image built."

    docker build -t mini-llmd-router:latest router/
    echo "    router image built."
}

# ============================================================
# 3. Deploy Prometheus + Adapter
# ============================================================
deploy_monitoring() {
    echo "==> Deploying Prometheus + Adapter..."
    helm repo add prometheus-community \
        https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo update

    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

    # Prometheus
    helm upgrade --install prometheus prometheus-community/prometheus \
        -n monitoring \
        --set server.service.type=ClusterIP \
        --set alertmanager.enabled=false \
        --set pushgateway.enabled=false

    # Prometheus Adapter — points at our router /metrics for custom HPA metric
    helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
        -n monitoring \
        --set prometheus.url="http://prometheus-server.monitoring.svc.cluster.local" \
        --set prometheus.port="80"

    # Apply our custom metric rules ConfigMap
    kubectl apply -f k8s/custom-metrics.yaml

    echo "    Waiting for Prometheus to be ready..."
    kubectl rollout status deployment/prometheus-server -n monitoring --timeout=180s
}

# ============================================================
# 4. Deploy the inference stack
# ============================================================
deploy_inference() {
    echo "==> Deploying inference stack..."
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

    # Create HF token secret (no-op if already exists)
    if [ -n "${HF_TOKEN:-}" ]; then
        kubectl create secret generic hf-token \
            --from-literal=token="${HF_TOKEN}" \
            -n "${NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -f -
    else
        echo "    HF_TOKEN not set — model must be pre-cached at /tmp/mini-llmd-model-cache"
    fi

    kubectl apply -f k8s/service.yaml     # namespace + services + router deployment
    kubectl apply -f k8s/deployment.yaml  # inference StatefulSet + ConfigMap

    echo "    Waiting for inference-0 to be ready (may take 2-3 min for weight loading)..."
    kubectl rollout status statefulset/inference -n "${NAMESPACE}" --timeout=300s
}

# ============================================================
# 5. Apply HPA
# ============================================================
deploy_hpa() {
    echo "==> Applying HPA..."
    kubectl apply -f k8s/hpa.yaml
    echo "    HPA applied. Verify with: kubectl get hpa -n ${NAMESPACE}"
}

# ============================================================
# 6. Quick smoke test
# ============================================================
smoke_test() {
    echo "==> Running smoke test..."
    ROUTER_PORT=$(kubectl get svc router -n "${NAMESPACE}" \
        -o jsonpath='{.spec.ports[0].nodePort}')
    MINIKUBE_IP=$(minikube ip)
    ROUTER_URL="http://${MINIKUBE_IP}:${ROUTER_PORT}"

    echo "    Router at ${ROUTER_URL}"

    # Health check
    curl -sf "${ROUTER_URL}/health" | python3 -m json.tool

    # Metrics check
    echo "--- /metrics ---"
    curl -sf "${ROUTER_URL}/metrics" | head -20

    # Single generation
    echo "--- /generate ---"
    curl -sf -X POST "${ROUTER_URL}/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"Hello, world! Tell me something interesting.","max_tokens":64,"stream":false}' \
        | python3 -m json.tool
}

# ============================================================
# 7. Run benchmark (Week 3)
# ============================================================
run_benchmark() {
    echo "==> Starting benchmark..."
    ROUTER_PORT=$(kubectl get svc router -n "${NAMESPACE}" \
        -o jsonpath='{.spec.ports[0].nodePort}')
    MINIKUBE_IP=$(minikube ip)

    mkdir -p results

    # Start GPU profiler in background
    python3 benchmarks/profile.py gpu_profile \
        --pods inference-0 inference-1 inference-2 inference-3 \
        --output results/gpu_profile.csv &
    PROFILER_PID=$!

    # Run Locust
    pip install locust --quiet
    locust -f benchmarks/load_test.py \
        --host "http://${MINIKUBE_IP}:${ROUTER_PORT}" \
        --headless \
        --users 64 \
        --spawn-rate 1 \
        --run-time 10m \
        --csv results/benchmark \
        --html results/benchmark_report.html

    kill "${PROFILER_PID}" 2>/dev/null || true

    # Generate scaling curve
    python3 benchmarks/profile.py plot_curves \
        --locust-csv results/benchmark_stats_history.csv \
        --gpu-csv results/gpu_profile.csv \
        --output results/scaling_curves.png

    echo "==> Benchmark complete. See results/scaling_curves.png"
}

# ============================================================
# Main
# ============================================================
CMD="${1:-help}"

case "$CMD" in
    all)
        check_deps
        start_minikube
        build_images
        deploy_monitoring
        deploy_inference
        deploy_hpa
        smoke_test
        ;;
    start)       start_minikube ;;
    build)       build_images ;;
    monitoring)  deploy_monitoring ;;
    deploy)      deploy_inference && deploy_hpa ;;
    smoke)       smoke_test ;;
    benchmark)   run_benchmark ;;
    check)       check_deps ;;
    *)
        echo "Usage: $0 {all|start|build|monitoring|deploy|smoke|benchmark|check}"
        echo ""
        echo "  all         — full first-time setup"
        echo "  start       — start minikube with GPU passthrough"
        echo "  build       — build + load Docker images"
        echo "  monitoring  — deploy Prometheus + adapter"
        echo "  deploy      — deploy inference stack + HPA"
        echo "  smoke       — quick health + generation test"
        echo "  benchmark   — run Locust load test + GPU profiler"
        echo "  check       — verify host dependencies"
        ;;
esac
