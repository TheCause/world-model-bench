#!/bin/bash
# =============================================================================
# world-model-bench — Bootstrap script
# =============================================================================
# Run once when pod starts. Sets up:
#   1. V-JEPA 2 model code (if not already on volume)
#   2. Benchmark scripts (rsync from M4 or git clone)
#   3. Status server for remote monitoring
#
# Usage: bash /opt/world-model-bench/bootstrap.sh [--no-status]
# =============================================================================

set -euo pipefail

echo "============================================"
echo "world-model-bench — Bootstrap"
echo "============================================"

# -----------------------------------------------
# 1. V-JEPA 2 model code
# -----------------------------------------------
VJEPA2_DIR="/workspace/models/vjepa2"
if [ ! -d "$VJEPA2_DIR/.git" ]; then
    echo "[1/4] Cloning V-JEPA 2..."
    git clone --depth 1 https://github.com/facebookresearch/vjepa2 "$VJEPA2_DIR"
else
    echo "[1/4] V-JEPA 2 already on volume, skipping"
fi

# -----------------------------------------------
# 2. Benchmark scripts (M4 first, GitHub fallback)
# -----------------------------------------------
BENCH_DIR="/workspace/benchmarks"
M4_BENCH="regis@192.168.1.60:/Users/regis/dev/RQZ_papers/_tools/rqz_toolkit/"
GITHUB_BENCH="https://github.com/TheCause/rqz-benchmark-scripts.git"

echo "[2/4] Deploying benchmark scripts..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes regis@192.168.1.60 "echo OK" 2>/dev/null; then
    echo "  M4 reachable — rsync from M4"
    rsync -az "$M4_BENCH" "$BENCH_DIR/rqz_toolkit/"
elif [ -d "$BENCH_DIR/rqz_toolkit" ]; then
    echo "  M4 unreachable but scripts already on volume"
else
    echo "  M4 unreachable — attempting GitHub fallback"
    if git clone --depth 1 "$GITHUB_BENCH" "$BENCH_DIR/rqz_toolkit" 2>/dev/null; then
        echo "  Cloned from GitHub"
    else
        echo "  WARNING: No benchmark scripts available. Deploy manually:"
        echo "    rsync -az user@host:/path/to/scripts/ $BENCH_DIR/rqz_toolkit/"
    fi
fi

# -----------------------------------------------
# 3. Pre-fetch V-JEPA 2 weights (optional, cached on volume)
# -----------------------------------------------
echo "[3/4] Checking V-JEPA 2 weights cache..."
if [ -d "/workspace/.cache/torch/hub/facebookresearch_vjepa2_main" ]; then
    echo "  Weights already cached"
else
    echo "  Pre-fetching V-JEPA 2 weights via torch.hub (first time only)..."
    python3 -c "
import torch
print('Downloading V-JEPA 2 ViT-H weights...')
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge', trust_repo=True)
print('Weights cached at $TORCH_HOME')
del model
torch.cuda.empty_cache()
" || echo "  WARNING: Weight download failed. Will download on first benchmark run."
fi

# -----------------------------------------------
# 4. Status server (background)
# -----------------------------------------------
if [[ "${1:-}" != "--no-status" ]]; then
    echo "[4/4] Starting status server on port 8080..."
    bash /opt/world-model-bench/status_server.sh &
else
    echo "[4/4] Status server disabled (--no-status)"
fi

echo ""
echo "============================================"
echo "Bootstrap complete."
echo ""
echo "Workspace layout:"
echo "  /workspace/models/vjepa2/    — V-JEPA 2 code"
echo "  /workspace/benchmarks/       — Benchmark scripts"
echo "  /workspace/data/             — Video datasets"
echo "  /workspace/results/          — Output (logs, verdicts)"
echo "  /workspace/.cache/           — Model weights (persistent)"
echo ""
echo "Run a benchmark:"
echo "  cd /workspace"
echo "  PYTHONPATH=/workspace/benchmarks python benchmarks/rqz_toolkit/tier3_phase1_runner.py --seeds 5"
echo ""
echo "Monitor remotely:"
echo "  curl https://\${RUNPOD_POD_ID}-8080.proxy.runpod.net/"
echo "============================================"
