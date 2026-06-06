#!/usr/bin/env bash
# =============================================================================
#  setup.sh  —  mini-llm-d  ·  Master setup + run script
#  RDMA / NCCL Edition  ·  minikube + vLLM + Llama 3.2 3B
# =============================================================================
#
#  Usage
#  -----
#  Laptop (no GPU, no minikube required):
#      ./setup.sh --no-gpu
#
#  Full minikube run (GPU node):
#      ./setup.sh
#
#  Run only specific stages:
#      ./setup.sh --no-gpu --stage bench       # benchmarks only
#      ./setup.sh --no-gpu --stage router      # router only
#      ./setup.sh --no-gpu --stage exporter    # RDMA exporter only
#
#  Options
#  -------
#  --no-gpu          CPU-simulation mode (default: off)
#  --stage <name>    Run a single stage: install | bench | router | server |
#                    exporter | kv_bench | nccl_bench | all (default: all)
#  --port  <n>       Router HTTP port (default: 8080)
#  --iterations <n>  Benchmark iterations per data point (default: 10)
#  -v / --verbose    Extra logging
#  -h / --help       This message
#
#  What each stage does
#  --------------------
#  install     pip-install Python deps (matplotlib, prometheus_client, etc.)
#  server      Start 2 Prefill + 2 Decode simulated inference pods
#  router      Start the prefix-cache aware router on --port
#  exporter    Start the RDMA Prometheus exporter on :9101
#  kv_bench    Run rdma_kv_bench.py (seq-len sweep 512/1024/2048 tokens)
#  nccl_bench  Run nccl_tp_bench.py (Scenario D, netem sweep 0/1/5 ms)
#  bench       Run both benchmarks
#  all         install → server → router → exporter → bench (default)
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

NO_GPU=0
STAGE="all"
PORT=8080
ITERATIONS=10
VERBOSE=0
OUT_DIR="results"
PIDS=()           # child PIDs for cleanup

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'
BLU='\033[0;34m'; CYN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${BLU}[INFO]${NC}  $*"; }
success() { echo -e "${GRN}[OK]${NC}    $*"; }
warn()    { echo -e "${YEL}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }
section() { echo -e "\n${CYN}━━━  $*  ━━━${NC}"; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gpu)           NO_GPU=1;           shift ;;
    --stage)            STAGE="$2";         shift 2 ;;
    --port)             PORT="$2";          shift 2 ;;
    --iterations)       ITERATIONS="$2";    shift 2 ;;
    -v|--verbose)       VERBOSE=1;          shift ;;
    -h|--help)
      head -60 "$0" | grep "^#" | sed 's/^# \?//'
      exit 0 ;;
    *) error "Unknown argument: $1"; exit 1 ;;
  esac
done

# Export so child Python processes inherit it
if [[ $NO_GPU -eq 1 ]]; then
  export MINI_LLM_NO_GPU=1
fi

PYTHON="${PYTHON:-python3}"
VERBOSE_FLAG=""
[[ $VERBOSE -eq 1 ]] && VERBOSE_FLAG="--verbose"

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------

cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    info "Stopping background processes: ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Stage: install
# ---------------------------------------------------------------------------

stage_install() {
  section "Stage: install Python dependencies"

  REQUIRED_PKGS="matplotlib prometheus_client"

  if [[ $NO_GPU -eq 0 ]]; then
    info "GPU mode — also checking for opentelemetry-sdk, vllm, pyverbs"
    REQUIRED_PKGS="$REQUIRED_PKGS opentelemetry-sdk opentelemetry-exporter-otlp"
    warn "vllm and pyverbs must be installed separately (CUDA environment needed)"
  fi

  for pkg in $REQUIRED_PKGS; do
    if $PYTHON -c "import ${pkg//-/_}" 2>/dev/null; then
      success "$pkg already installed"
    else
      info "Installing $pkg ..."
      $PYTHON -m pip install --quiet "$pkg"
      success "$pkg installed"
    fi
  done

  # Create output directory
  mkdir -p "$OUT_DIR"
  success "Output directory: $OUT_DIR"
}

# ---------------------------------------------------------------------------
# Stage: server (simulated inference pods)
# ---------------------------------------------------------------------------

stage_server() {
  section "Stage: inference servers"

  if [[ $NO_GPU -eq 1 ]]; then
    info "Starting 2 simulated Prefill pods + 2 simulated Decode pods"

    for i in 0 1; do
      $PYTHON server.py \
        --no-gpu \
        --pod-id "prefill-$i" \
        --role prefill \
        --port "$((8100 + i))" \
        $VERBOSE_FLAG \
        &
      PIDS+=($!)
      info "  Prefill pod-$i → :$((8100 + i))"
    done

    for i in 0 1; do
      $PYTHON server.py \
        --no-gpu \
        --pod-id "decode-$i" \
        --role decode \
        --port "$((8200 + i))" \
        $VERBOSE_FLAG \
        &
      PIDS+=($!)
      info "  Decode  pod-$i → :$((8200 + i))"
    done

    sleep 1
    success "4 simulated inference pods running"
  else
    warn "GPU mode: expecting vLLM pods already running. Skipping server start."
    warn "Start pods manually: python server.py --pod-id prefill-0 --role prefill"
  fi
}

# ---------------------------------------------------------------------------
# Stage: router
# ---------------------------------------------------------------------------

stage_router() {
  section "Stage: prefix-cache aware router"

  NO_GPU_FLAG=""
  [[ $NO_GPU -eq 1 ]] && NO_GPU_FLAG="--no-gpu"

  $PYTHON router.py \
    $NO_GPU_FLAG \
    --port "$PORT" \
    $VERBOSE_FLAG \
    &
  PIDS+=($!)
  info "Router → http://localhost:$PORT"
  sleep 1

  # Quick health check
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    success "Router health check passed"
  else
    warn "Router health check failed (may still be starting up)"
  fi
}

# ---------------------------------------------------------------------------
# Stage: exporter (RDMA Prometheus sidecar)
# ---------------------------------------------------------------------------

stage_exporter() {
  section "Stage: RDMA Prometheus exporter"

  NO_GPU_FLAG=""
  [[ $NO_GPU -eq 1 ]] && NO_GPU_FLAG="--no-gpu"

  $PYTHON rdma/rdma_exporter/exporter.py \
    $NO_GPU_FLAG \
    --port 9101 \
    $VERBOSE_FLAG \
    &
  PIDS+=($!)
  sleep 1
  info "RDMA exporter → http://localhost:9101/metrics"

  if curl -sf "http://localhost:9101/metrics" > /dev/null 2>&1; then
    success "RDMA exporter responding"
  else
    warn "RDMA exporter not yet responding (may still be starting up)"
  fi
}

# ---------------------------------------------------------------------------
# Stage: kv_bench
# ---------------------------------------------------------------------------

stage_kv_bench() {
  section "Stage: RDMA KV-Cache Latency Benchmark"

  NO_GPU_FLAG=""
  [[ $NO_GPU -eq 1 ]] && NO_GPU_FLAG="--no-gpu"

  $PYTHON benchmarks/rdma_kv_bench.py \
    $NO_GPU_FLAG \
    --seq-lens 512 1024 2048 \
    --iterations "$ITERATIONS" \
    --warmup 2 \
    --out-dir "$OUT_DIR" \
    $VERBOSE_FLAG

  success "KV-cache benchmark complete → $OUT_DIR/rdma_kv_latency.{json,png}"
}

# ---------------------------------------------------------------------------
# Stage: nccl_bench
# ---------------------------------------------------------------------------

stage_nccl_bench() {
  section "Stage: NCCL TP=2 All-Reduce Benchmark (Scenario D)"

  NO_GPU_FLAG=""
  [[ $NO_GPU -eq 1 ]] && NO_GPU_FLAG="--no-gpu"

  $PYTHON benchmarks/nccl_tp_bench.py \
    $NO_GPU_FLAG \
    --data-gb 1.0 \
    --iterations "$ITERATIONS" \
    --warmup 1 \
    --netem-ms 0 1 5 \
    --out-dir "$OUT_DIR" \
    $VERBOSE_FLAG

  success "NCCL benchmark complete → $OUT_DIR/nccl_throughput.{json,png}"
}

# ---------------------------------------------------------------------------
# Stage: bench (both benchmarks)
# ---------------------------------------------------------------------------

stage_bench() {
  stage_kv_bench
  stage_nccl_bench
}

# ---------------------------------------------------------------------------
# Quick end-to-end smoke test (optional, runs after router is up)
# ---------------------------------------------------------------------------

smoke_test() {
  section "Smoke test: POST /generate"

  PAYLOAD='{"prompt":"Explain RDMA in one sentence.","max_tokens":16}'
  RESP=$(curl -sf -X POST \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "http://localhost:$PORT/generate" 2>&1) || true

  if echo "$RESP" | grep -q "text"; then
    success "Smoke test passed"
    [[ $VERBOSE -eq 1 ]] && echo "  Response: $RESP"
  else
    warn "Smoke test response unexpected: $RESP"
  fi
}

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

banner() {
  echo -e "${CYN}"
  cat << 'EOF'
  ╔══════════════════════════════════════════════════════╗
  ║   mini-llm-d  ·  RDMA / NCCL Edition                ║
  ║   Kubernetes-Native Auto-Scaling LLM Inference       ║
  ╚══════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"
  echo "  Mode       : $([ $NO_GPU -eq 1 ] && echo 'CPU-simulation (--no-gpu)' || echo 'Real GPU / RDMA')"
  echo "  Stage      : $STAGE"
  echo "  Router port: $PORT"
  echo "  Iterations : $ITERATIONS"
  echo "  Output dir : $OUT_DIR"
  echo
}

main() {
  banner

  case "$STAGE" in
    install)    stage_install ;;
    server)     stage_server ;;
    router)     stage_router; smoke_test ;;
    exporter)   stage_exporter ;;
    kv_bench)   stage_kv_bench ;;
    nccl_bench) stage_nccl_bench ;;
    bench)      stage_bench ;;
    all)
      stage_install
      stage_server
      stage_router
      stage_exporter
      sleep 1
      smoke_test
      stage_bench

      section "All stages complete"
      echo
      success "Results     : $OUT_DIR/"
      success "Router      : http://localhost:$PORT/generate"
      success "Metrics     : http://localhost:$PORT/metrics"
      success "RDMA export : http://localhost:9101/metrics"
      echo
      info "Background processes still running (Ctrl-C to stop):"
      for pid in "${PIDS[@]}"; do
        echo "  PID $pid"
      done
      echo
      wait   # keep script alive until Ctrl-C
      ;;
    *)
      error "Unknown stage: $STAGE"
      error "Valid stages: install server router exporter kv_bench nccl_bench bench all"
      exit 1
      ;;
  esac
}

main
