# world-model-bench

Generic GPU benchmark template for world models (V-JEPA 2, DreamerV3, TD-MPC2, etc.)

## Quick Start

### 1. Launch pod on RunPod

Via MCP compute-m4 :
```
cloud_create_pod(name="bench", template_id="lofz87vsh0", gpu_type="standard", volume_gb=100)
```

Ou via la console RunPod : My Templates → `RQZ-Benchmark` → Deploy

- GPU recommande : A100 80GB PCIe ($1.19/h)
- Volume : 100GB minimum (poids V-JEPA 2 = ~10GB)

### 2. Bootstrap (automatique)

Le start command execute automatiquement `bootstrap.sh` qui :
- Installe les deps inference (timm, transformers, einops, decord, etc.)
- Clone V-JEPA 2 dans `/workspace/models/vjepa2/`
- Telecharge les poids ViT-H (~9.7GB, premier boot uniquement)
- Lance un serveur HTTP status sur le port 8080

Premier boot : ~5-7 minutes (download poids).
Boots suivants : ~30 secondes (poids caches sur volume).

### 3. Deployer les scripts de benchmark

Les scripts ne sont pas dans le template (generique). Les deployer via scp depuis M4 :
```bash
scp -P <port> -r rqz_toolkit/*.py root@<ip>:/workspace/benchmarks/rqz_toolkit/
```

### 4. Lancer un benchmark

```bash
cd /workspace
PYTHONPATH=/workspace/benchmarks python benchmarks/rqz_toolkit/tier3_phase1_runner.py \
    --seeds 5 --variant vit_huge
```

### 5. Monitorer a distance

```bash
curl https://<pod-id>-8080.proxy.runpod.net/
```

## Template

| Champ | Valeur |
|-------|--------|
| ID | `lofz87vsh0` |
| Nom | `RQZ-Benchmark` |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Start Command | `bash -c "curl -sL https://raw.githubusercontent.com/TheCause/world-model-bench/main/bootstrap.sh \| bash"` |
| Volume | 100GB a `/workspace` |
| Env vars | `TORCH_HOME=/workspace/.cache/torch`, `HF_HOME=/workspace/.cache/huggingface` |
| Statut | Valide (17 mars 2026) |

## Workspace Layout

```
/workspace/
├── .cache/torch/          # Poids modeles (persistant, auto-download)
├── .cache/huggingface/    # Cache HuggingFace
├── models/                # Code des world models
│   ├── vjepa2/            # V-JEPA 2 (auto-clone)
│   ├── dreamerv3/         # Futur : git clone + pip install
│   └── tdmpc2/            # Futur : git clone + pip install
├── benchmarks/            # Scripts de benchmark (deployes via scp)
├── data/                  # Datasets video
└── results/               # Logs JSONL, VERDICT.md, resultats seeds
```

## Ajouter un world model

```bash
cd /workspace/models
git clone https://github.com/danijar/dreamerv3
cd dreamerv3 && pip install -r requirements.txt
export PYTHONPATH=/workspace/models/dreamerv3:$PYTHONPATH
```

Pas besoin de reconstruire le template.

## Deps pre-installees

PyTorch 2.4 + CUDA 12.4 + cuDNN + Python 3.11

Inference : torch, torchvision, numpy, scipy, timm, transformers, einops,
decord, opencv-python, matplotlib, pandas, h5py, beartype, pyyaml

Systeme : ffmpeg, SSH, JupyterLab (base RunPod)
