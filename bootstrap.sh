#!/bin/bash
# =============================================================================
# world-model-bench — Bootstrap (fetched via curl at pod start)
# =============================================================================
# Start Command in RunPod template:
#   bash -c "curl -sL https://raw.githubusercontent.com/TheCause/world-model-bench/main/bootstrap.sh | bash"
# =============================================================================

set -euo pipefail

echo "============================================"
echo "world-model-bench — Bootstrap"
echo "============================================"

# -----------------------------------------------
# 0. Python deps (skipped if already installed)
# -----------------------------------------------
echo "[0/4] Checking Python dependencies..."
python3 -c "import timm, transformers, einops, decord" 2>/dev/null || {
    echo "  Installing inference deps..."
    pip install --no-cache-dir timm transformers einops beartype decord \
        opencv-python scipy matplotlib pandas h5py pyyaml 2>&1 | tail -3
}
echo "  Dependencies OK"

# -----------------------------------------------
# 1. Directories
# -----------------------------------------------
echo "[1/4] Setting up workspace..."
mkdir -p /workspace/.cache/torch /workspace/.cache/huggingface \
         /workspace/models /workspace/benchmarks /workspace/data /workspace/results

# Env vars (persist across sessions)
grep -q TORCH_HOME ~/.bashrc 2>/dev/null || {
    echo 'export TORCH_HOME=/workspace/.cache/torch' >> ~/.bashrc
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
}
export TORCH_HOME=/workspace/.cache/torch
export HF_HOME=/workspace/.cache/huggingface

# -----------------------------------------------
# 2. V-JEPA 2 model code
# -----------------------------------------------
VJEPA2_DIR="/workspace/models/vjepa2"
if [ ! -d "$VJEPA2_DIR/.git" ]; then
    echo "[2/4] Cloning V-JEPA 2..."
    git clone --depth 1 https://github.com/facebookresearch/vjepa2 "$VJEPA2_DIR"
else
    echo "[2/4] V-JEPA 2 already on volume"
fi

# -----------------------------------------------
# 3. Pre-fetch V-JEPA 2 weights (cached on volume)
# -----------------------------------------------
echo "[3/4] Checking V-JEPA 2 weights..."
if [ -d "/workspace/.cache/torch/hub/facebookresearch_vjepa2_main" ]; then
    echo "  Weights already cached"
else
    echo "  Downloading V-JEPA 2 ViT-H (~9.7GB, first time only)..."
    python3 -c "
import torch
print('Fetching ViT-H...')
m = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge', trust_repo=True)
print('Cached OK')
del m; torch.cuda.empty_cache()
" || echo "  WARNING: Download failed. Will retry on first benchmark run."
fi

# -----------------------------------------------
# 4. Status server (background, port 8080)
# -----------------------------------------------
echo "[4/4] Starting status server on port 8080..."
mkdir -p /workspace/results
cd /workspace/results && python3 -m http.server 8080 --bind 0.0.0.0 &

echo ""
echo "============================================"
echo "Bootstrap complete."
echo ""
echo "  /workspace/models/vjepa2/    V-JEPA 2 code"
echo "  /workspace/benchmarks/       Benchmark scripts (deploy via scp)"
echo "  /workspace/data/             Datasets"
echo "  /workspace/results/          Logs + verdicts"
echo "  /workspace/.cache/           Model weights (persistent)"
echo ""
echo "Deploy scripts:  scp -P <port> scripts.py root@<ip>:/workspace/benchmarks/"
echo "Monitor:         curl https://<pod-id>-8080.proxy.runpod.net/"
echo "============================================"

# NOTE: This script must be launched with & in the start command:
#   bash -c "curl -sL .../bootstrap.sh | bash &"
# The & ensures RunPod's init process starts SSH normally.
