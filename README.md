# world-model-bench

Generic GPU benchmark template for world models (V-JEPA 2, DreamerV3, TD-MPC2, etc.)

## Quick Start

### 1. Launch pod on RunPod

Use template `ghcr.io/thecause/world-model-bench:latest` with:
- GPU: A100 80GB (recommended) or A100 40GB
- Volume: 100GB+ at `/workspace`
- Ports: 8080/http (status server)

### 2. Bootstrap (first time)

```bash
bash /opt/world-model-bench/bootstrap.sh
```

This will:
- Clone V-JEPA 2 to `/workspace/models/vjepa2/`
- Deploy benchmark scripts to `/workspace/benchmarks/`
- Pre-fetch model weights (cached on volume)
- Start HTTP status server on port 8080

### 3. Run a benchmark

```bash
cd /workspace
PYTHONPATH=/workspace/benchmarks python benchmarks/rqz_toolkit/tier3_phase1_runner.py \
    --seeds 5 --variant vit_huge
```

### 4. Monitor remotely

```bash
# From any machine:
curl https://<pod-id>-8080.proxy.runpod.net/
# View specific log:
curl https://<pod-id>-8080.proxy.runpod.net/phase1/run_phase1_*.jsonl
```

## Workspace Layout

```
/workspace/
├── .cache/torch/          # Model weights (persistent, auto-downloaded)
├── .cache/huggingface/    # HF model cache
├── models/                # World model code
│   ├── vjepa2/            # V-JEPA 2 (auto-cloned)
│   ├── dreamerv3/         # Future: git clone + pip install
│   └── tdmpc2/            # Future: git clone + pip install
├── benchmarks/            # Benchmark scripts (rsync from M4 or git)
├── data/                  # Video datasets
└── results/               # JSONL logs, VERDICT.md, seed results
```

## Adding a New World Model

```bash
cd /workspace/models
git clone https://github.com/danijar/dreamerv3
cd dreamerv3 && pip install -r requirements.txt
# Add to PYTHONPATH:
export PYTHONPATH=/workspace/models/dreamerv3:$PYTHONPATH
```

No Docker rebuild needed.

## Pre-installed Dependencies

PyTorch 2.8 + CUDA 12.8 + cuDNN + Python 3.11

Inference: torch, torchvision, numpy, scipy, timm, transformers, einops,
decord, opencv-python, matplotlib, pandas, h5py

System: rsync, ffmpeg, SSH, JupyterLab (from RunPod base)

## Build & Push

```bash
# Via GitHub Actions (recommended):
git push  # triggers .github/workflows/build-template.yml

# Manual (from x86 machine):
docker build -t ghcr.io/thecause/world-model-bench:latest .
docker push ghcr.io/thecause/world-model-bench:latest
```
