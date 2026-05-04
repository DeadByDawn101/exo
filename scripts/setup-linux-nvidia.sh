#!/usr/bin/env bash
# ============================================================================
# setup-linux-nvidia.sh -- One-shot setup for exo on Linux with NVIDIA GPU
#
# Run from the repo root:
#   bash scripts/setup-linux-nvidia.sh
#
# Tested on: Pop!_OS 24.04 LTS (NVIDIA edition)
# Requirements: NVIDIA GPU with drivers installed (nvidia-smi must work)
# ============================================================================
set -euo pipefail

echo "============================================"
echo " Star Platinum -- Linux NVIDIA Setup"
echo " exo + tinygrad NV backend"
echo "============================================"
echo ""

# -- Preflight checks --
echo ">>> Preflight checks..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
echo "  [OK] NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "  [OK] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  [OK] VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  [OK] Python: $PYVER"

# -- System packages --
echo ""
echo ">>> Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3-venv python3-full git curl build-essential nodejs npm pkg-config libssl-dev 2>/dev/null

# -- Install uv --
echo ""
echo ">>> Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  [OK] uv: $(uv --version)"

# -- Install Rust --
echo ""
echo ">>> Installing Rust..."
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
rustup toolchain install nightly 2>/dev/null || true
echo "  [OK] Rust: $(rustc --version)"

# -- Create venv and install tinygrad --
echo ""
echo ">>> Setting up Python venv..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "  Installing tinygrad..."
pip install -q git+https://github.com/tinygrad/tinygrad.git numpy

echo "  Installing transformers + safetensors..."
pip install -q transformers safetensors huggingface_hub

# -- Verify tinygrad sees the GPU --
echo ""
echo ">>> Verifying tinygrad NV backend..."
DEVICE=$(python3 -c "from tinygrad import Device; print(Device.DEFAULT)" 2>&1)
echo "  tinygrad device: $DEVICE"

if [[ "$DEVICE" != "NV" && "$DEVICE" != "CUDA" && "$DEVICE" != "GPU" ]]; then
    echo "WARNING: tinygrad defaulted to $DEVICE instead of NV/CUDA"
else
    echo "  [OK] GPU detected!"
    python3 -c "
from tinygrad import Tensor, Device
x = Tensor.rand(512, 512)
y = Tensor.rand(512, 512)
z = (x @ y).sum()
print(f'  [OK] Matmul test: {z.item():.0f} (GPU compute working)')
"
fi

# -- Build dashboard --
echo ""
echo ">>> Building exo dashboard..."
cd "$REPO_ROOT/dashboard"
npm install --silent 2>/dev/null
npm run build --silent 2>/dev/null
cd "$REPO_ROOT"
echo "  [OK] Dashboard built"

# -- Summary --
echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo " To start exo:"
echo "   source .venv/bin/activate"
echo "   uv run python -m exo"
echo ""
echo " To join Mac cluster:"
echo "   uv run python -m exo --bootstrap-peers 192.168.1.248:52415,192.168.1.247:52415"
echo ""
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo " Backend: tinygrad ($DEVICE)"
echo " API: http://localhost:52415"
echo ""
