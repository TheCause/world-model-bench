#!/usr/bin/env python3
"""
Dry-run test for tier3_phase1_runner.

Tests the FULL pipeline with a mock V-JEPA 2 model:
- VJEPAModel.load() → mock encoder
- VJEPAModel.preprocess() → real transform on synthetic video
- VJEPAModel.encode_videos() → mock output with correct shape
- run_a5_seed() → DH-sigma noise vs drift
- run_a3_seed() → MIT drift detection
- validate_and_verdict() → VERDICT.md generation

No GPU needed. No real V-JEPA 2. Tests API contracts only.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Patch RESULTS_DIR before import
tmpdir = tempfile.mkdtemp(prefix="rqz_dryrun_")
print(f"Results dir: {tmpdir}")

import rqz_toolkit.tier3_phase1_runner as runner
runner.RESULTS_DIR = Path(tmpdir)

from rqz_toolkit.tier3_phase1_runner import (
    VJEPAModel, Phase1Config, load_video_clips,
    compute_temporal_errors, run_a5_seed, run_a3_seed,
    validate_and_verdict, save_seed_result, is_seed_done,
)
from rqz_toolkit.logger import RQZLogger


def test_preprocess():
    """Test preprocess on synthetic video."""
    print("\n=== Test preprocess ===")
    model = VJEPAModel(variant="vit_huge")
    # Synthetic video: 16 frames, 320x240, RGB
    video = np.random.randint(0, 255, (16, 320, 240, 3), dtype=np.uint8)
    result = model.preprocess(video)
    print(f"  Input:  {video.shape} uint8")
    print(f"  Output: {result.shape} {result.dtype}")
    assert result.shape[0] == 3, f"Expected C=3, got {result.shape[0]}"
    assert result.shape[1] == 16, f"Expected T=16, got {result.shape[1]}"
    assert result.shape[2] == 256, f"Expected H=256, got {result.shape[2]}"
    assert result.shape[3] == 256, f"Expected W=256, got {result.shape[3]}"
    print("  PASS")


def test_load_video_clips():
    """Test synthetic video fallback."""
    print("\n=== Test load_video_clips ===")
    clips = load_video_clips(2, 16, seed=42, video_dir="/nonexistent")
    print(f"  Clips: {len(clips)}")
    print(f"  Shape: {clips[0].shape}")
    assert len(clips) == 2
    assert clips[0].shape == (16, 224, 224, 3)
    print("  PASS")


def test_compute_temporal_errors():
    """Test temporal error computation."""
    print("\n=== Test compute_temporal_errors ===")
    # Mock V-JEPA 2 output: [1, n_patches, embed_dim]
    # With 16 frames, patch_size=16, tubelet_size=2:
    # spatial = (256/16)^2 = 256, temporal = 16/2 = 8
    # n_patches = 256 * 8 = 2048
    n_spatial = 256
    n_temporal = 8
    embed_dim = 1280
    features = torch.randn(1, n_spatial * n_temporal, embed_dim)
    errors = compute_temporal_errors(features, n_spatial_patches=n_spatial)
    print(f"  Features: {features.shape}")
    print(f"  Errors: {errors.shape}")
    assert len(errors) == n_temporal - 1, f"Expected {n_temporal-1} errors, got {len(errors)}"
    print("  PASS")


def test_a5_seed():
    """Test A5 (DH-sigma) with mock model."""
    print("\n=== Test A5 seed ===")
    model = VJEPAModel(variant="vit_huge")
    model.loaded = True

    # Mock encoder to return correct shape
    n_spatial, n_temporal, embed_dim = 256, 8, 1280
    mock_output = torch.randn(1, n_spatial * n_temporal, embed_dim)

    model.encoder = MagicMock()
    model.encoder.return_value = mock_output

    cfg = Phase1Config(n_seeds=1, n_video_frames=16)
    log = RQZLogger("test_a5", output_dir=tmpdir, also_print=False)

    result = run_a5_seed(model, seed=42, cfg=cfg, log=log)
    print(f"  rho_noise: {result['rho_noise_mean']:.3f}")
    print(f"  rho_drift: {result['rho_drift_mean']:.3f}")
    print(f"  gap: {result['gap']:.3f}")
    print(f"  pass: {result['pass']}")
    assert "gap" in result
    assert "rho_noise_mean" in result
    assert "rho_drift_mean" in result
    print("  PASS")


def test_a3_seed():
    """Test A3 (MIT drift) with mock model."""
    print("\n=== Test A3 seed ===")
    model = VJEPAModel(variant="vit_huge")
    model.loaded = True

    n_spatial, n_temporal, embed_dim = 256, 8, 1280
    mock_output = torch.randn(1, n_spatial * n_temporal, embed_dim)

    model.encoder = MagicMock()
    model.encoder.return_value = mock_output

    cfg = Phase1Config(n_seeds=1, n_video_frames=16)
    log = RQZLogger("test_a3", output_dir=tmpdir, also_print=False)

    result = run_a3_seed(model, seed=42, cfg=cfg, log=log)
    print(f"  latency: {result['detection_latency']}")
    print(f"  fp_rate: {result['fp_rate']:.3f}")
    print(f"  detected: {result['detected']}")
    print(f"  pass: {result['pass']}")
    assert "detection_latency" in result
    assert "fp_rate" in result
    print("  PASS")


def test_verdict():
    """Test verdict generation."""
    print("\n=== Test verdict ===")
    seeds = [42]
    cfg = Phase1Config(n_seeds=1)

    # Save mock results
    save_seed_result("A5", 42, {"seed": 42, "gap": 0.35, "rho_noise_mean": 0.48,
                                 "rho_drift_mean": 0.13, "pass": True})
    save_seed_result("A3", 42, {"seed": 42, "detection_latency": 12, "false_positives": 0,
                                 "fp_rate": 0.0, "detected": True, "pass": True})

    verdict = validate_and_verdict(seeds, cfg)
    print(f"  A5 verdict: {verdict['A5'].get('verdict')}")
    print(f"  A3 verdict: {verdict['A3'].get('verdict')}")
    print(f"  Overall: {verdict['overall']}")

    verdict_path = runner.RESULTS_DIR / "VERDICT.md"
    assert verdict_path.exists(), "VERDICT.md not created"
    print(f"  VERDICT.md: {verdict_path.read_text()[:200]}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("RQZ Tier 3 Phase 1 — DRY RUN")
    print("=" * 60)

    try:
        test_preprocess()
        test_load_video_clips()
        test_compute_temporal_errors()
        test_a5_seed()
        test_a3_seed()
        test_verdict()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
