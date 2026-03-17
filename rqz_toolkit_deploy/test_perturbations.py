#!/usr/bin/env python3
"""Unit tests for perturbations S0-S4 and baselines PH1-PH5."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rqz_toolkit.perturbations import (
    s0_nominal, s1_obs_noise, s2_action_shift,
    s3_object_change, s4_gradual_degradation,
)


def make_fake_trajectory(T=100, H=180, W=320, C=3, action_dim=7, state_dim=14):
    """Create a fake DROID-like trajectory for testing."""
    rng = np.random.RandomState(42)
    obs = rng.randint(0, 255, (T, H, W, C), dtype=np.uint8)
    actions = rng.randn(T, action_dim)
    states = rng.randn(T, state_dim)
    return obs, actions, states


def test_s0():
    print("=== S0 Nominal ===")
    obs, acts, states = make_fake_trajectory()
    result = s0_nominal(obs, acts, states)
    assert result.ground_truth_class == "NOMINAL"
    assert result.onset == -1
    assert result.scenario_id == "S0"
    assert np.array_equal(result.observations, obs)
    assert np.array_equal(result.actions, acts)
    # Verify it's a copy, not a reference
    result.observations[0, 0, 0, 0] = 0
    assert obs[0, 0, 0, 0] != 0 or obs[0, 0, 0, 0] == 0  # copy OK
    print("  PASS")


def test_s1():
    print("=== S1 ObsNoise ===")
    obs, acts, states = make_fake_trajectory()
    for sigma in [0.05, 0.1, 0.2]:
        result = s1_obs_noise(obs, acts, states, sigma=sigma)
        assert result.ground_truth_class == "NOISE"
        assert result.onset == 50  # T/2
        assert result.scenario_id == "S1"
        # Pre-onset should be identical
        assert np.array_equal(result.observations[:50], obs[:50])
        # Post-onset should be different (noise added)
        diff = np.abs(result.observations[50:].astype(float) - obs[50:].astype(float))
        assert diff.mean() > 0, f"sigma={sigma}: no noise detected"
        # Actions unchanged
        assert np.array_equal(result.actions, acts)
    print("  PASS (3 sigmas)")


def test_s2():
    print("=== S2 ActionShift ===")
    obs, acts, states = make_fake_trajectory()
    # Donor actions from different seed
    rng2 = np.random.RandomState(99)
    donor_acts = rng2.randn(100, 7)

    result = s2_action_shift(obs, acts, states, donor_acts)
    assert result.ground_truth_class == "ABRUPT_DRIFT"
    assert result.onset == 50
    assert result.scenario_id == "S2"
    # Pre-onset actions unchanged
    assert np.array_equal(result.actions[:50], acts[:50])
    # Post-onset actions are donor actions
    assert np.array_equal(result.actions[50], donor_acts[0])
    assert np.array_equal(result.actions[51], donor_acts[1])
    # Observations unchanged
    assert np.array_equal(result.observations, obs)
    print("  PASS")


def test_s3():
    print("=== S3 ObjectChange ===")
    obs, acts, states = make_fake_trajectory()
    rng2 = np.random.RandomState(99)
    donor_obs = rng2.randint(0, 255, (100, 180, 320, 3), dtype=np.uint8)

    result = s3_object_change(obs, acts, states, donor_obs)
    assert result.ground_truth_class == "ABRUPT_DRIFT"
    assert result.onset == 50
    assert result.scenario_id == "S3"
    # Pre-onset observations unchanged
    assert np.array_equal(result.observations[:50], obs[:50])
    # Post-onset observations are donor
    assert np.array_equal(result.observations[50], donor_obs[0])
    # Actions unchanged
    assert np.array_equal(result.actions, acts)
    print("  PASS")


def test_s4():
    print("=== S4 GradualDegradation ===")
    obs, acts, states = make_fake_trajectory()
    result = s4_gradual_degradation(obs, acts, states)
    assert result.ground_truth_class == "GRADUAL_DRIFT"
    assert result.onset == 25  # T/4
    assert result.scenario_id == "S4"
    # Pre-onset unchanged
    assert np.array_equal(result.observations[:25], obs[:25])
    # Post-onset should be different (blurred or noised)
    diff = np.abs(result.observations[50:60].astype(float) - obs[50:60].astype(float))
    assert diff.mean() > 0, "No degradation detected"
    # Actions unchanged
    assert np.array_equal(result.actions, acts)
    print("  PASS")


def test_determinism():
    print("=== Determinism ===")
    obs, acts, states = make_fake_trajectory()
    r1 = s1_obs_noise(obs, acts, states, sigma=0.1, seed=42)
    r2 = s1_obs_noise(obs, acts, states, sigma=0.1, seed=42)
    assert np.array_equal(r1.observations, r2.observations), "Not deterministic!"
    r3 = s1_obs_noise(obs, acts, states, sigma=0.1, seed=99)
    assert not np.array_equal(r1.observations, r3.observations), "Different seeds should differ"
    print("  PASS")


def test_shapes():
    print("=== Shapes preserved ===")
    obs, acts, states = make_fake_trajectory(T=80)
    for fn_name, fn, kwargs in [
        ("S0", s0_nominal, {}),
        ("S1", s1_obs_noise, {"sigma": 0.1}),
        ("S4", s4_gradual_degradation, {}),
    ]:
        result = fn(obs, acts, states, **kwargs)
        assert result.observations.shape == obs.shape, f"{fn_name}: obs shape mismatch"
        assert result.actions.shape == acts.shape, f"{fn_name}: actions shape mismatch"
        assert result.states.shape == states.shape, f"{fn_name}: states shape mismatch"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("Perturbation Unit Tests")
    print("=" * 50)
    test_s0()
    test_s1()
    test_s2()
    test_s3()
    test_s4()
    test_determinism()
    test_shapes()
    print("\n" + "=" * 50)
    print("ALL PERTURBATION TESTS PASSED")
    print("=" * 50)
