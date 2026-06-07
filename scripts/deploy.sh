#!/usr/bin/env bash
# scripts/deploy.sh
# Deploys the mini-llmd inference stack to minikube.
# Run after cluster-up.sh and build-images.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K8S="$REPO_ROOT/k8s"
NS="mini-llmd"

echo "=== mini-llm-d: deploy ==="

# ---------------------------------------------------------------------------
# Namespace + services
# ---------------------------------------------------------------------------
kubectl apply -f "$K8S/service.yaml"

# ---------------------------------------------------------------------------
# Inference pods + router
# ---------------------------------------------------------------------------
kubectl apply -f "$K8S/deployment.yaml"

echo "→ Waiting for inference pod(s) to be ready (this may take 2–3 min) ..."
kubectl rollout status statefulset/inference-combined -n "$NS" --timeout=300s

echo "→ Waiting for router ..."
kubectl rollout status deployment/router -n "$NS" --timeout=60s

# ---------------------------------------------------------------------------
# HPA
# ---------------------------------------------------------------------------
kubectl apply -f "$K8S/hpa.yaml"

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
echo ""
echo "✓ Deployment complete!"
echo ""
echo "Pods:"
kubectl get pods -n "$NS" -o wide

echo ""
echo "HPA:"
kubectl get hpa -n "$NS"

echo ""
MINIKUBE_IP="$(minikube ip)"
echo "  Router (NodePort): http://${MINIKUBE_IP}:30900"
echo "  Grafana:           http://${MINIKUBE_IP}:30300"
echo "  Jaeger:            http://${MINIKUBE_IP}:30686"
echo ""
echo "Test a request:"
echo "  curl -X POST http://${MINIKUBE_IP}:30900/generate \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"prompt\": \"Hello world\", \"max_tokens\": 64, \"stream\": false}'"
