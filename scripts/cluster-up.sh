#!/usr/bin/env bash
# scripts/cluster-up.sh
# Starts a minikube cluster with GPU passthrough and installs the
# monitoring stack (Prometheus, Grafana, Prometheus Adapter, Jaeger).

set -euo pipefail

echo "=== mini-llm-d: cluster-up ==="

# ---------------------------------------------------------------------------
# 1. Start minikube
# ---------------------------------------------------------------------------
echo "→ Starting minikube with GPU passthrough ..."
minikube start \
    --driver=docker \
    --gpus=all \
    --memory=16g \
    --cpus=6 \
    --disk-size=40g \
    --kubernetes-version=v1.29.3

# ---------------------------------------------------------------------------
# 2. NVIDIA device plugin
# ---------------------------------------------------------------------------
echo "→ Installing NVIDIA device plugin ..."
kubectl apply -f \
    https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml

# Wait for plugin
kubectl rollout status daemonset/nvidia-device-plugin-daemonset \
    -n kube-system --timeout=120s

# Verify GPU is visible
echo "→ GPU capacity:"
kubectl get nodes -o json | python3 -c \
    "import sys,json; n=json.load(sys.stdin)['items'][0]; print(json.dumps(n['status']['capacity'], indent=2))" \
    2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Namespaces
# ---------------------------------------------------------------------------
kubectl create namespace mini-llmd  --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace monitoring  --dry-run=client -o yaml | kubectl apply -f -

# ---------------------------------------------------------------------------
# 4. Helm repos
# ---------------------------------------------------------------------------
echo "→ Adding Helm repos ..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana               https://grafana.github.io/helm-charts
helm repo update

# ---------------------------------------------------------------------------
# 5. Prometheus
# ---------------------------------------------------------------------------
echo "→ Installing Prometheus ..."
helm upgrade --install prometheus prometheus-community/prometheus \
    -n monitoring \
    --set server.global.scrape_interval=15s \
    --set alertmanager.enabled=false \
    --set pushgateway.enabled=false \
    --wait --timeout=5m

# ---------------------------------------------------------------------------
# 6. Prometheus Adapter (custom metrics for HPA)
# ---------------------------------------------------------------------------
echo "→ Installing Prometheus Adapter ..."
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
    -n monitoring \
    --set prometheus.url=http://prometheus-server.monitoring \
    --set prometheus.port=80 \
    --wait --timeout=3m

# Apply custom metric rules
kubectl apply -f "$(dirname "$0")/../k8s/custom-metrics.yaml"

# ---------------------------------------------------------------------------
# 7. Grafana
# ---------------------------------------------------------------------------
echo "→ Creating Grafana dashboard ConfigMap ..."
kubectl create configmap grafana-dashboards \
    -n monitoring \
    --from-file="$(dirname "$0")/../k8s/grafana-dashboards/" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "→ Installing Grafana ..."
helm upgrade --install grafana grafana/grafana \
    -n monitoring \
    -f "$(dirname "$0")/../k8s/grafana-values.yaml" \
    --wait --timeout=3m

# ---------------------------------------------------------------------------
# 8. Jaeger
# ---------------------------------------------------------------------------
echo "→ Deploying Jaeger ..."
kubectl apply -f "$(dirname "$0")/../k8s/jaeger.yaml"
kubectl rollout status deployment/jaeger -n monitoring --timeout=60s

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "✓ Cluster ready!"
echo ""
echo "  Grafana UI:  http://$(minikube ip):30300  (admin / mini-llmd-admin)"
echo "  Jaeger UI:   http://$(minikube ip):30686"
echo ""
echo "Next: ./scripts/build-images.sh && ./scripts/deploy.sh"
