# mini-llm-d — Makefile
#
# Thin convenience layer over the existing shell scripts and kubectl/helm.
# It does NOT replace setup.sh or scripts/*.sh — it just wraps them so the
# whole pipeline is reproducible with short commands.
#
#   make            # show this help
#   make all        # full pipeline: setup -> model -> cluster -> build -> deploy
#   make bench      # run the routing-comparison benchmark
#
# Override variables on the command line, e.g.:
#   make bench SCENARIO=B
#   make setup SETUP_FLAGS=--no-gpu

# ---------------------------------------------------------------------------
# Config (override on the command line)
# ---------------------------------------------------------------------------
SHELL          := /bin/bash
NS             ?= mini-llmd
MON_NS         ?= monitoring
MODEL          ?= meta-llama/Llama-3.2-3B-Instruct
MODEL_DIR      ?= models/llama-3.2-3b-instruct
SCENARIO       ?= C
SETUP_FLAGS    ?=
BENCH_PY       := benchmarks/.venv/bin/python

# minikube IP is resolved lazily so targets that don't need it still work
# even when the cluster is down.
MINIKUBE_IP     = $(shell minikube ip 2>/dev/null)
ROUTER_URL     ?= http://$(MINIKUBE_IP):30900

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@echo "mini-llm-d — make targets"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: NS=$(NS) MODEL_DIR=$(MODEL_DIR) SCENARIO=$(SCENARIO) ROUTER_URL=$(ROUTER_URL)"

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
.PHONY: all
all: setup model cluster build deploy ## Run the entire pipeline end-to-end
	@echo ""
	@echo "✓ Pipeline complete. Endpoints:"
	@echo "  Router:  $(ROUTER_URL)"
	@echo "  Grafana: http://$(MINIKUBE_IP):30300"
	@echo "  Jaeger:  http://$(MINIKUBE_IP):30686"
	@echo ""
	@echo "Next: make bench   (or)   make smoke"

# ---------------------------------------------------------------------------
# One-time-per-clone setup
# ---------------------------------------------------------------------------
.PHONY: setup
setup: ## Create the three Python venvs via setup.sh (use SETUP_FLAGS=--no-gpu to skip vLLM)
	chmod +x setup.sh scripts/*.sh
	./setup.sh $(SETUP_FLAGS)

.PHONY: check
check: ## Verify the environment (uv/GPU) without installing — wraps setup.sh --check
	./setup.sh --check

.PHONY: model
model: ## Download the model into $(MODEL_DIR) (skips if already present)
	@if [ -d "$(MODEL_DIR)" ] && [ -n "$$(ls -A '$(MODEL_DIR)' 2>/dev/null)" ]; then \
		echo "✓ Model already present at $(MODEL_DIR) — skipping download"; \
	else \
		echo "→ Downloading $(MODEL) (gated; run 'huggingface-cli login' first if needed)"; \
		huggingface-cli download $(MODEL) --local-dir $(MODEL_DIR); \
	fi

# ---------------------------------------------------------------------------
# Cluster / build / deploy (wrap scripts/)
# ---------------------------------------------------------------------------
.PHONY: cluster
cluster: ## Start minikube + monitoring stack (Prometheus/Grafana/Jaeger)
	./scripts/cluster-up.sh

.PHONY: build
build: ## Build router + inference images into minikube's Docker daemon
	./scripts/build-images.sh

.PHONY: deploy
deploy: ## Deploy inference pods, router, and HPA
	./scripts/deploy.sh

.PHONY: redeploy
redeploy: build ## Rebuild images and restart the workloads (fast iteration)
	kubectl -n $(NS) rollout restart statefulset/inference-combined deployment/router
	kubectl -n $(NS) rollout status  statefulset/inference-combined --timeout=300s
	kubectl -n $(NS) rollout status  deployment/router --timeout=60s

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
.PHONY: bench
bench: ## Routing comparison: round-robin vs least-queue vs prefix-cache (-> results/routing_comparison.png)
	ROUTER_URL=$(ROUTER_URL) $(BENCH_PY) benchmarks/routing_comparison.py

.PHONY: profile
profile: ## Concurrency sweep + GPU/scaling curves (-> results/scaling_curves.png). Use SCENARIO=A|B|C
	ROUTER_URL=$(ROUTER_URL) $(BENCH_PY) benchmarks/profile.py --scenario $(SCENARIO)

.PHONY: load
load: ## Interactive Locust UI against the router
	ROUTER_URL=$(ROUTER_URL) $(BENCH_PY) -m locust -f benchmarks/load_test.py --host $(ROUTER_URL)

# ---------------------------------------------------------------------------
# Operations / inspection
# ---------------------------------------------------------------------------
.PHONY: smoke
smoke: ## Send one test request through the router
	curl -sS -X POST $(ROUTER_URL)/generate \
		-H 'Content-Type: application/json' \
		-d '{"prompt": "Hello world", "max_tokens": 64, "stream": false}' | python3 -m json.tool

.PHONY: status
status: ## Show pods and HPA state
	kubectl -n $(NS) get pods -o wide
	@echo ""
	kubectl -n $(NS) get hpa

.PHONY: logs
logs: ## Tail router logs (override POD= for a specific pod)
	kubectl -n $(NS) logs -f deployment/router

.PHONY: urls
urls: ## Print all service URLs
	@echo "Router:  $(ROUTER_URL)"
	@echo "Grafana: http://$(MINIKUBE_IP):30300  (admin / mini-llmd-admin)"
	@echo "Jaeger:  http://$(MINIKUBE_IP):30686"

.PHONY: dashboards
dashboards: ## Re-apply Grafana dashboard ConfigMap after editing the JSON
	kubectl create configmap grafana-dashboards -n $(MON_NS) \
		--from-file=k8s/grafana-dashboards/ \
		--dry-run=client -o yaml | kubectl apply -f -

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
# ruff is run via `uvx` so it auto-installs on first use — no manual setup.
# Version is pinned to match dev-dependencies in pyproject.toml.
RUFF ?= uvx ruff@0.5.7

.PHONY: lint
lint: ## Ruff lint (auto-installs ruff via uvx; config in pyproject.toml)
	$(RUFF) check .

.PHONY: fmt
fmt: ## Ruff format + import sort (auto-installs ruff via uvx)
	$(RUFF) format .
	$(RUFF) check --select I --fix .

.PHONY: compile
compile: ## Byte-compile all Python to catch syntax errors
	python3 -m py_compile router/*.py inference/*.py benchmarks/*.py
	@echo "✓ all Python compiles"

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
.PHONY: undeploy
undeploy: ## Remove mini-llmd workloads (keeps the cluster + monitoring up)
	-kubectl delete -f k8s/hpa.yaml --ignore-not-found
	-kubectl delete -f k8s/deployment.yaml --ignore-not-found
	-kubectl delete -f k8s/service.yaml --ignore-not-found

.PHONY: clean
clean: ## Remove generated benchmark artifacts and temp pickles
	rm -f results/*.png results/*.csv /tmp/bench_*.pkl /tmp/prof_*.pkl

.PHONY: nuke
nuke: ## Delete the entire minikube cluster (destructive)
	minikube delete
