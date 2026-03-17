# =============================================================================
# world-model-bench — Generic GPU benchmark template for world models
# =============================================================================
# Base: RunPod PyTorch (CUDA 12.8, Python 3.11, SSH + Jupyter pre-configured)
# Includes: V-JEPA 2 inference deps, video processing, scientific tools
# Usage: RunPod template → ghcr.io/thecause/world-model-bench:latest
# =============================================================================

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Persistent cache on /workspace volume (survives pod restarts)
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HOME=/workspace/.cache/huggingface
ENV PYTHONPATH=/workspace/models/vjepa2:$PYTHONPATH

# System deps
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        rsync \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — inference-only, minimal footprint
COPY requirements-inference.txt /tmp/requirements-inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements-inference.txt && \
    rm /tmp/requirements-inference.txt

# Bootstrap + status server scripts
COPY bootstrap.sh /opt/world-model-bench/bootstrap.sh
COPY status_server.sh /opt/world-model-bench/status_server.sh
RUN chmod +x /opt/world-model-bench/*.sh

# Workspace convention
RUN mkdir -p /workspace/models /workspace/benchmarks /workspace/data /workspace/results

# Expose HTTP status server port
EXPOSE 8080

# Default working directory
WORKDIR /workspace
