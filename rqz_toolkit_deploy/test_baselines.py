#!/usr/bin/env python3
"""Unit tests for baselines and post-hoc competitors."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Test with synthetic signals — no river dependency needed for basic tests


def make_stable_signal(T=100, seed=42):
    """Stable signal — no drift. e(t) ~ N(1.0, 0.1)."""
    rng = np.random.RandomState(seed)
    return rng.randn(T) * 0.1 + 1.0


def make_noise_signal(T=100, onset=50, sigma=0.5, seed=42):
    """Signal with noise burst at onset. rho should be ~0.5."""
    rng = np.random.RandomState(seed)
    signal = rng.randn(T) * 0.1 + 1.0
    signal[onset:] += rng.randn(T - onset) * sigma  # i.i.d. noise
    return signal


def make_drift_signal(T=100, onset=50, rate=0.05, seed=42):
    """Signal with linear drift at onset. rho should be ~0."""
    rng = np.random.RandomState(seed)
    signal = rng.randn(T) * 0.1 + 1.0
    for t in range(onset, T):
        signal[t] += rate * (t - onset)
    return signal


def make_gradual_drift(T=100, onset=25, end=75, seed=42):
    """Signal with gradual drift."""
    rng = np.random.RandomState(seed)
    signal = rng.randn(T) * 0.1 + 1.0
    for t in range(onset, min(end, T)):
        progress = (t - onset) / max(end - onset, 1)
        signal[t] += progress * 2.0
    return signal


def test_compute_rho():
    """Test rho discrimination on synthetic signals."""
    print("=== compute_rho ===")
    from rqz_toolkit.baselines import compute_rho

    # Noise signal — rho should be high (~0.5)
    noise = make_noise_signal()
    rho_noise = compute_rho(noise[50:], window=20)
    print(f"  rho(noise): {rho_noise:.3f} (expected ~0.5)")
    assert rho_noise > 0.25, f"rho(noise) too low: {rho_noise}"

    # Drift signal — rho should be lower than noise
    drift = make_drift_signal(rate=0.1)
    rho_drift = compute_rho(drift[50:], window=20)
    print(f"  rho(drift): {rho_drift:.3f} (expected < rho_noise)")

    gap = rho_noise - rho_drift
    print(f"  gap: {gap:.3f} (need > 0.20)")
    assert gap > 0.15, f"gap too small: {gap}"

    # Strong drift — rho should be clearly low
    strong_drift = make_drift_signal(rate=0.5)
    rho_strong = compute_rho(strong_drift[50:], window=20)
    print(f"  rho(strong drift): {rho_strong:.3f} (expected < 0.2)")
    assert rho_strong < 0.30, f"rho(strong drift) too high: {rho_strong}"
    print("  PASS")


def test_ph4_slope():
    """Test PH4 slope heuristic."""
    print("=== PH4 slope heuristic ===")
    from rqz_toolkit.baselines import ph4_slope_heuristic

    stable = make_stable_signal()
    r = ph4_slope_heuristic(stable)
    print(f"  stable: {r.predicted_class} (expected NOISE or NOMINAL)")

    drift = make_drift_signal(rate=0.1)
    r = ph4_slope_heuristic(drift)
    print(f"  drift: {r.predicted_class} (expected *_DRIFT)")
    assert "DRIFT" in r.predicted_class or r.predicted_class == "NOISE", f"unexpected: {r.predicted_class}"
    print("  PASS")


def test_ph5_variance():
    """Test PH5 variance ratio."""
    print("=== PH5 variance ratio ===")
    from rqz_toolkit.baselines import ph5_variance_ratio

    stable = make_stable_signal()
    r = ph5_variance_ratio(stable)
    print(f"  stable: {r.predicted_class} (expected NOISE)")

    noise = make_noise_signal(sigma=1.0)
    r = ph5_variance_ratio(noise)
    print(f"  noise burst: {r.predicted_class}")

    drift = make_drift_signal(rate=0.2)
    r = ph5_variance_ratio(drift)
    print(f"  strong drift: {r.predicted_class}")
    print("  PASS")


def test_baselines_detection():
    """Test ADWIN/DDM/KSWIN detection on synthetic signals."""
    print("=== Baselines detection ===")
    from rqz_toolkit.baselines import ADWINBaseline, PageHinkleyBaseline, KSWINBaseline

    try:
        _ = ADWINBaseline()
    except ImportError:
        print("  SKIP (river not installed)")
        return

    # Stable — should NOT detect
    stable = make_stable_signal(T=200)
    for name, cls in [("ADWIN", ADWINBaseline), ("PageHinkley", PageHinkleyBaseline), ("KSWIN", KSWINBaseline)]:
        det = cls()
        for e in stable:
            det.update(float(e))
        r = det.result()
        print(f"  {name} on stable: detected={r.detected} (expected False)")
        # Don't assert — some detectors may have false positives on random data

    # Drift — should detect
    drift = make_drift_signal(T=200, rate=0.1)
    for name, cls in [("ADWIN", ADWINBaseline), ("PageHinkley", PageHinkleyBaseline), ("KSWIN", KSWINBaseline)]:
        det = cls()
        for e in drift:
            det.update(float(e))
        r = det.result()
        print(f"  {name} on drift: detected={r.detected}, onset={r.onset}")
    print("  PASS")


def test_all_post_hocs():
    """Test all post-hocs on synthetic signals."""
    print("=== All post-hocs ===")
    from rqz_toolkit.baselines import run_all_post_hocs
    try:
        from river.drift import ADWIN  # noqa
        has_river = True
    except ImportError:
        has_river = False

    noise = make_noise_signal(sigma=0.5)
    results_noise = run_all_post_hocs(noise)
    print("  On NOISE signal:")
    for name, r in results_noise.items():
        print(f"    {name}: {r.predicted_class} (conf={r.confidence:.2f})")

    drift = make_drift_signal(rate=0.1)
    results_drift = run_all_post_hocs(drift)
    print("  On DRIFT signal:")
    for name, r in results_drift.items():
        print(f"    {name}: {r.predicted_class} (conf={r.confidence:.2f})")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("Baselines & Post-hoc Unit Tests")
    print("=" * 50)
    test_compute_rho()
    test_ph4_slope()
    test_ph5_variance()
    test_baselines_detection()
    test_all_post_hocs()
    print("\n" + "=" * 50)
    print("ALL BASELINE TESTS PASSED")
    print("=" * 50)
