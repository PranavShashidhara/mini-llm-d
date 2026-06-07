#!/usr/bin/env bash
# setup.sh — mini-llm-d local development environment setup
#
# Sets up all three Python environments using uv:
#   1. inference/  — vLLM + FastAPI  (GPU/CUDA required for full install)
#   2. router/     — FastAPI + httpx (CPU only)
#   3. benchmarks/ — Locust + matplotlib (CPU only)
#
# Usage:
#   ./setup.sh              # full setup (inference env requires GPU)
#   ./setup.sh --no-gpu     # skip inference env (router + benchmarks only)
#   ./setup.sh --check      # just verify the environment, don't install
#
# Requirements:
#   - Python 3.11+ on PATH  (or pyenv / mise with 3.11 available)
#   - curl                  (to install uv if not present)
#   - NVIDIA GPU + CUDA 12.1+ drivers (for inference env only)
#   - nvidia-smi on PATH    (to detect GPU)

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}→${RESET} $*"; }
success() { echo -e "${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}⚠${RESET}  $*"; }
error()   { echo -e "${RED}✗${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
NO_GPU=false
CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --no-gpu)    NO_GPU=true ;;
        --check)     CHECK_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--no-gpu] [--check]"
            echo "  --no-gpu   Skip inference (vLLM) environment — no GPU needed"
            echo "  --check    Verify environment without installing"
            exit 0
            ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Detect GPU / CUDA
# ─────────────────────────────────────────────────────────────────────────────
header "=== GPU / CUDA detection ==="

HAS_GPU=false
CUDA_VERSION=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    CUDA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    CUDA_RUNTIME="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo 'unknown')"

    if [[ -n "$GPU_NAME" ]]; then
        HAS_GPU=true
        success "GPU detected: ${GPU_NAME}"
        info    "Driver version: ${CUDA_VERSION}"
        info    "CUDA runtime:   ${CUDA_RUNTIME}"

        # Warn if CUDA < 12.1 (vLLM minimum)
        CUDA_MAJOR="${CUDA_RUNTIME%%.*}"
        CUDA_MINOR="$(echo "$CUDA_RUNTIME" | cut -d. -f2)"
        if [[ "$CUDA_MAJOR" -lt 12 ]] || { [[ "$CUDA_MAJOR" -eq 12 ]] && [[ "$CUDA_MINOR" -lt 1 ]]; }; then
            warn "CUDA ${CUDA_RUNTIME} detected — vLLM requires CUDA >= 12.1"
            warn "The inference env install will likely fail. Use --no-gpu to skip it."
        fi
    fi
else
    warn "nvidia-smi not found — no GPU detected"
fi

if [[ "$NO_GPU" == true ]]; then
    warn "--no-gpu flag set: skipping inference environment"
    HAS_GPU=false
fi

if [[ "$CHECK_ONLY" == true ]]; then
    if [[ "$HAS_GPU" == true ]]; then
        success "GPU check passed"
    else
        warn "No GPU available (inference pod will not run locally)"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Install uv (if not already installed)
# ─────────────────────────────────────────────────────────────────────────────
header "=== uv installation ==="

if command -v uv &>/dev/null; then
    UV_VERSION="$(uv --version)"
    success "uv already installed: ${UV_VERSION}"
else
    if [[ "$CHECK_ONLY" == true ]]; then
        error "uv not installed. Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    info "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for this session (installer writes to ~/.cargo/bin or ~/.local/bin)
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        error "uv install appeared to succeed but 'uv' is not on PATH."
        error "Add one of the following to your shell profile and re-run:"
        error "  export PATH=\"\$HOME/.cargo/bin:\$PATH\""
        error "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
    success "uv installed: $(uv --version)"
fi

if [[ "$CHECK_ONLY" == true ]]; then
    success "uv check passed"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Python version check
# ─────────────────────────────────────────────────────────────────────────────
header "=== Python 3.11 ==="

# uv can download its own Python, but we prefer a system 3.11 when available
if uv python find 3.11 &>/dev/null; then
    PY="$(uv python find 3.11)"
    success "Python 3.11 found: ${PY}"
else
    info "Python 3.11 not found — asking uv to install it ..."
    uv python install 3.11
    success "Python 3.11 installed by uv"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Router environment
# ─────────────────────────────────────────────────────────────────────────────
header "=== Router environment (CPU only) ==="

info "Creating venv: router/.venv"
uv venv router/.venv --python 3.11 --quiet

info "Installing router dependencies ..."
uv pip install \
    --python router/.venv/bin/python \
    -r router/requirements.in \
    --quiet

success "Router environment ready"
echo    "  Activate: source router/.venv/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Benchmarks environment
# ─────────────────────────────────────────────────────────────────────────────
header "=== Benchmarks environment (CPU only) ==="

info "Creating venv: benchmarks/.venv"
uv venv benchmarks/.venv --python 3.11 --quiet

info "Installing benchmark dependencies ..."
uv pip install \
    --python benchmarks/.venv/bin/python \
    -r benchmarks/requirements.txt \
    --quiet

success "Benchmarks environment ready"
echo    "  Activate: source benchmarks/.venv/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Inference environment (GPU required)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$HAS_GPU" == true ]]; then
    header "=== Inference environment (GPU / CUDA) ==="

    info "Creating venv: inference/.venv"
    uv venv inference/.venv --python 3.11 --quiet

    # ── Step 1: Install PyTorch with CUDA wheels first ──────────────────────
    # vLLM declares torch as a dependency but PyPI only has CPU wheels.
    # We must install torch from the CUDA wheel index BEFORE vLLM so that
    # pip/uv does not overwrite it with the CPU variant.
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"

    info "Installing PyTorch (CUDA 12.1 wheels) ..."
    uv pip install \
        --python inference/.venv/bin/python \
        --index-url "$TORCH_INDEX" \
        "torch>=2.3.0,<2.4.0" \
        "torchvision" \
        "torchaudio" \
        --quiet

    # Verify CUDA is actually available in the installed torch
    TORCH_CUDA="$(inference/.venv/bin/python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'error')"
    if [[ "$TORCH_CUDA" != "True" ]]; then
        warn "torch.cuda.is_available() returned: ${TORCH_CUDA}"
        warn "PyTorch may have installed CPU-only wheels despite the index override."
        warn "This can happen if your CUDA driver doesn't match the wheel ABI."
        warn "Try: uv pip install --python inference/.venv/bin/python torch --index-url ${TORCH_INDEX}"
    else
        success "torch.cuda.is_available() = True"
    fi

    # ── Step 2: Install vLLM (it will reuse the already-installed torch) ────
    info "Installing vLLM and remaining inference dependencies ..."
    uv pip install \
        --python inference/.venv/bin/python \
        --extra-index-url "$TORCH_INDEX" \
        -r inference/requirements.in \
        --quiet

    # ── Step 3: Quick smoke test ─────────────────────────────────────────────
    info "Smoke-testing vLLM import ..."
    if inference/.venv/bin/python -c "import vllm; print(f'vLLM {vllm.__version__} OK')" 2>/dev/null; then
        success "Inference environment ready"
    else
        warn "vLLM import failed — see above. The Docker image uses pinned deps and"
        warn "may work even if the local venv doesn't (especially on driver mismatches)."
    fi

    echo "  Activate: source inference/.venv/bin/activate"

else
    header "=== Inference environment ==="
    warn "Skipping (no GPU / --no-gpu flag). The inference pod requires a GPU."
    warn "To install anyway (not recommended — torch will be CPU-only):"
    warn "  uv venv inference/.venv --python 3.11"
    warn "  uv pip install --python inference/.venv/bin/python -r inference/requirements.in"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. Print summary
# ─────────────────────────────────────────────────────────────────────────────
header "=== Setup complete ==="

echo ""
echo -e "  ${BOLD}Environments created:${RESET}"
[[ "$HAS_GPU" == true ]] && echo "    inference/.venv  — vLLM serving (GPU)"
echo "    router/.venv     — request router (CPU)"
echo "    benchmarks/.venv — Locust + matplotlib (CPU)"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo "    1. Download the model (first time only):"
echo "       huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \\"
echo "           --local-dir models/llama-3.2-3b-instruct"
echo ""
echo "    2. Start the cluster:"
echo "       ./scripts/cluster-up.sh"
echo ""
echo "    3. Build images and deploy:"
echo "       ./scripts/build-images.sh"
echo "       ./scripts/deploy.sh"
echo ""
echo "    4. Run benchmarks:"
echo "       source benchmarks/.venv/bin/activate"
echo "       export ROUTER_URL=http://\$(minikube ip):30900"
echo "       python benchmarks/routing_comparison.py"
echo ""
