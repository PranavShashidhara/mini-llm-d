#!/usr/bin/env bash
# scripts/build-images.sh
# Builds Docker images inside minikube's Docker daemon so they are
# immediately available to pods without an external registry.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== mini-llm-d: build-images ==="

# Point Docker CLI at minikube's daemon
eval "$(minikube docker-env)"

echo "→ Building inference image ..."
docker build \
    -t mini-llmd-inference:latest \
    -f "$REPO_ROOT/inference/Dockerfile" \
    "$REPO_ROOT/inference/"

echo "→ Building router image ..."
docker build \
    -t mini-llmd-router:latest \
    -f "$REPO_ROOT/router/Dockerfile" \
    "$REPO_ROOT/router/"

echo ""
echo "✓ Images built:"
docker images | grep mini-llmd
