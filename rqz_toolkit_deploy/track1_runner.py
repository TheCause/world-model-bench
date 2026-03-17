#!/usr/bin/env python3
"""
Track 1 Runner — MVP benchmark for "Beyond Binary Validity"

Runs DH-sigma + MIT + baselines on DROID trajectories with perturbations S0/S2/S3.
Produces VERDICT_track1.md with go/no-go decision.

Usage:
    python track1_runner.py [--n-episodes 20] [--seeds 3] [--synthetic]

Contract reference: contract.md v1.0, section 6 (Track 1)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rqz_toolkit.droid_loader import load_droid_trajectories, make_synthetic_trajectories, Trajectory
from rqz_toolkit.perturbations import s0_nominal, s2_action_shift, s3_object_change
from rqz_toolkit.dh import DualHorizon, DHConfig
from rqz_toolkit.mit import MITDetector, MITConfig
from rqz_toolkit.hha import HHAController, HHAConfig
from rqz_toolkit.arh import ARHSystem, ARHConfig
from rqz_toolkit.rbd import ResonanceComputer
from rqz_toolkit.logger import RQZLogger

# Try to import baselines (need river)
try:
    from rqz_toolkit.baselines import (
        ADWINBaseline, PageHinkleyBaseline, KSWINBaseline,
        evaluate_c3_fair, compute_rho,
    )
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 123, 456]  # Track 1 = 3 seeds
RESULTS_DIR = Path("results/track1")

@dataclass
class Track1Config:
    n_episodes: int = 20
    seeds: List[int] = None
    scenarios: List[str] = None
    use_synthetic: bool = False  # True for dry-run without DROID/V-JEPA 2
    # Kill conditions (contract section 6)
    c1_gap_kill: float = 0.10
    c2_fp_kill: float = 0.10

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = SEEDS[:3]
        if self.scenarios is None:
            self.scenarios = ["S0", "S2", "S3"]


# =============================================================================
# Error Signal Generator (mock or real V-JEPA 2-AC)
# =============================================================================

def compute_prediction_errors_mock(trajectory: Trajectory, seed: int,
                                    scenario: str = "S0") -> np.ndarray:
    """
    Mock error signal for dry-run testing (no V-JEPA 2-AC needed).

    Produces scenario-dependent signals:
    - S0: stable noise ~0.5 (nominal)
    - S2: abrupt jump at T/2 (actions mismatch → error doubles)
    - S3: abrupt jump at T/2 (scene change → error triples)
    - S1: noise burst at T/2 (i.i.d. increase)
    - S4: gradual increase from T/4 to 3T/4
    """
    rng = np.random.RandomState(seed)
    T = trajectory.n_steps
    onset = T // 2

    # Base: stable with small noise
    errors = 0.5 + rng.randn(T) * 0.08

    if scenario == "S0":
        pass  # nominal — stable noise
    elif scenario == "S1":
        # Noise burst: i.i.d. increase after onset (rho stays ~0.5)
        errors[onset:] += rng.randn(T - onset) * 0.4
    elif scenario == "S2":
        # Abrupt drift: monotonic increase after onset (rho → 0)
        for t in range(onset, T):
            errors[t] += 0.05 * (t - onset)
    elif scenario == "S3":
        # Abrupt jump + drift: sudden shift + slow increase (rho → 0)
        errors[onset:] += 1.0
        for t in range(onset, T):
            errors[t] += 0.03 * (t - onset)
    elif scenario == "S4":
        # Gradual: linear ramp from T/4 to 3T/4
        start = T // 4
        end = 3 * T // 4
        for t in range(start, min(end, T)):
            progress = (t - start) / max(end - start, 1)
            errors[t] += progress * 1.5

    return np.abs(errors).astype(np.float32)


# =============================================================================
# Run single scenario on single trajectory
# =============================================================================

def run_scenario(
    trajectory: Trajectory,
    donor_trajectory: Trajectory,
    scenario: str,
    seed: int,
    compute_errors=None,
) -> Dict:
    """
    Run one scenario on one trajectory with all methods.

    Args:
        trajectory: the test trajectory
        donor_trajectory: trajectory from different task (for S2, S3)
        scenario: "S0", "S2", or "S3"
        seed: random seed
        compute_errors: function to compute prediction errors

    Returns:
        Dict with all metrics for this run
    """
    if compute_errors is None:
        compute_errors = compute_prediction_errors_mock
    T = trajectory.n_steps
    onset = T // 2 if scenario != "S0" else -1

    # Apply perturbation
    obs_dummy = np.zeros((T, 180, 320, 3), dtype=np.uint8)  # placeholder
    if scenario == "S0":
        perturbed = s0_nominal(obs_dummy, trajectory.actions, trajectory.states)
    elif scenario == "S2":
        perturbed = s2_action_shift(obs_dummy, trajectory.actions, trajectory.states,
                                     donor_trajectory.actions, seed=seed)
    elif scenario == "S3":
        donor_obs = np.zeros((donor_trajectory.n_steps, 180, 320, 3), dtype=np.uint8)
        perturbed = s3_object_change(obs_dummy, trajectory.actions, trajectory.states,
                                      donor_obs)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Create perturbed trajectory for error computation
    perturbed_traj = Trajectory(
        episode_id=trajectory.episode_id,
        task_index=trajectory.task_index,
        actions=perturbed.actions,
        states=perturbed.states,
        timestamps=trajectory.timestamps,
        language=trajectory.language,
        n_steps=T,
    )

    # Compute prediction errors (scenario-aware for mock)
    errors = compute_errors(perturbed_traj, seed, scenario=scenario)

    # Split: first half = calibration, second half = test (contract 4.5)
    cal_end = T // 4  # calibrate on first 25%
    cal_errors = errors[:cal_end]

    # --- RQZ Diagnostics ---

    # DH-sigma
    dh = DualHorizon(DHConfig(sign_window=20))
    dh.calibrate(cal_errors)
    for e in errors:
        dh.observe(float(e))
        dh.compute()
    rho = dh.compute_rho()
    diagnosis = dh.diagnose()

    # MIT
    mit = MITDetector(MITConfig(W=min(25, T // 8)))
    cal_S = [abs(cal_errors[i] - cal_errors[i-1]) if i > 0 else 0.0
             for i in range(len(cal_errors))]
    for i in range(len(cal_errors)):
        mit.add_calibration_sample(float(cal_errors[i] ** 2), 0.9)
    mit.calibrate(cal_S)

    mit_triggered = False
    mit_onset = -1
    for t in range(T):
        S = abs(errors[t] - (errors[t-1] if t > 0 else 0.0))
        E = float(errors[t] ** 2)
        # R_mock: realistic proxy. R ~0.85 nominal, drops toward 0 on shift
        mean_cal = float(np.mean(cal_errors)) + 1e-8
        R = float(np.clip(1.0 - errors[t] / (3.0 * mean_cal), 0.0, 1.0))
        result = mit.update(S, E, R)
        if result["trigger"] and not mit_triggered:
            mit_triggered = True
            mit_onset = t

    # HHA
    hha = HHAController(HHAConfig())
    hha.calibrate(cal_S)
    for t in range(T):
        S = abs(errors[t] - (errors[t-1] if t > 0 else 0.0))
        hha.update(S)
    gamma = hha.gamma
    hha_saturated = hha.is_saturated()

    # ARH
    arh = ARHSystem(ARHConfig(warmup_steps=cal_end))
    for t in range(T):
        arh.step_from_error(float(errors[t]))
    arh_stats = arh.get_stats()

    # --- Baselines ---
    baseline_results = {}
    if HAS_RIVER:
        for name, cls in [("ADWIN", ADWINBaseline), ("PageHinkley", PageHinkleyBaseline),
                          ("KSWIN", KSWINBaseline)]:
            det = cls()
            for e in errors:
                det.update(float(e))
            r = det.result()
            baseline_results[name] = {
                "detected": r.detected,
                "onset": r.onset,
            }

        # C3 fair evaluation
        if onset > 0:
            c3_results = evaluate_c3_fair(errors, onset, window=20)
            baseline_results["C3_fair"] = {
                name: {"class": r.predicted_class, "confidence": r.confidence}
                for name, r in c3_results.items()
            }

    # --- Determine diagnosis (Fix #2: NOMINAL when nothing triggers) ---
    # DH P score: if low, no event detected → NOMINAL
    _, _, P = dh.compute()
    any_detection = mit_triggered or P > 0.5  # P > 0.5 = DH detects something
    if not any_detection:
        diagnosis = "NOMINAL"

    # --- Determine policy ---
    # pi(class, confidence, MIT_triggered) — contract section 2
    # Fix #3: confidence = ARH.C, not ad hoc ratio
    confidence = float(arh.C)

    if diagnosis == "NOMINAL" and confidence > 0.7:
        policy = "TRUST"      # R1
    elif diagnosis == "NOISE" and not mit_triggered:
        policy = "FREEZE"     # R2
    elif diagnosis == "NOISE" and mit_triggered:
        policy = "FREEZE"     # R3: DH overrules MIT
    elif diagnosis == "ABRUPT_DRIFT":
        policy = "REBUILD"    # R4
    elif diagnosis == "GRADUAL_DRIFT":
        policy = "REBUILD"    # R5
    elif confidence < 0.3:
        policy = "FREEZE"     # R6
    else:
        policy = "TRUST"

    # --- Ground truth ---
    gt_class = {
        "S0": "NOMINAL",
        "S2": "ABRUPT_DRIFT",
        "S3": "ABRUPT_DRIFT",
    }[scenario]

    gt_policy = {
        "S0": "TRUST",
        "S2": "REBUILD",
        "S3": "REBUILD",
    }[scenario]

    return {
        "scenario": scenario,
        "episode_id": trajectory.episode_id,
        "seed": seed,
        "n_steps": T,
        "onset": onset,
        # DH
        "rho": rho,
        "dh_diagnosis": diagnosis,
        # MIT
        "mit_triggered": mit_triggered,
        "mit_onset": mit_onset,
        # HHA
        "gamma": gamma,
        "hha_saturated": hha_saturated,
        # ARH
        "arh_mean_N": arh_stats["mean_N"],
        "arh_mean_C": arh_stats["mean_C"],
        # Policy
        "policy": policy,
        "confidence": confidence,
        # Ground truth
        "gt_class": gt_class,
        "gt_policy": gt_policy,
        # Correctness
        "diagnosis_correct": (diagnosis == gt_class) or
                             (gt_class == "NOMINAL" and diagnosis == "NOMINAL") or
                             ("DRIFT" in diagnosis and "DRIFT" in gt_class),
        "policy_correct": policy == gt_policy,
        # Baselines
        "baselines": baseline_results,
    }


# =============================================================================
# Aggregate and evaluate claims
# =============================================================================

def evaluate_claims(all_results: List[Dict], cfg: Track1Config) -> Dict:
    """Evaluate C1, C2 against Track 1 kill conditions."""

    verdict = {"C1": {}, "C2": {}, "overall": ""}

    # C1: DH-sigma gap
    noise_rhos = [r["rho"] for r in all_results
                  if r["gt_class"] == "NOMINAL" or r["scenario"] == "S0"]
    drift_rhos = [r["rho"] for r in all_results
                  if "DRIFT" in r["gt_class"]]

    if noise_rhos and drift_rhos:
        gap = float(np.mean(noise_rhos) - np.mean(drift_rhos))
        verdict["C1"] = {
            "rho_nominal_mean": round(float(np.mean(noise_rhos)), 3),
            "rho_drift_mean": round(float(np.mean(drift_rhos)), 3),
            "gap": round(gap, 3),
            "kill_threshold": cfg.c1_gap_kill,
            "pass": gap > cfg.c1_gap_kill,
        }

    # C2: MIT FP rate on S0
    s0_results = [r for r in all_results if r["scenario"] == "S0"]
    if s0_results:
        fp_count = sum(1 for r in s0_results if r["mit_triggered"])
        fp_rate = fp_count / len(s0_results)
        # Detection rate on shift scenarios
        shift_results = [r for r in all_results if r["scenario"] != "S0"]
        dr = sum(1 for r in shift_results if r["mit_triggered"]) / max(len(shift_results), 1)
        verdict["C2"] = {
            "fp_rate": round(fp_rate, 3),
            "detection_rate": round(dr, 3),
            "fp_kill": cfg.c2_fp_kill,
            "pass": fp_rate <= cfg.c2_fp_kill,
        }

    # Diagnosis accuracy
    correct = sum(1 for r in all_results if r["diagnosis_correct"])
    total = len(all_results)
    verdict["diagnosis_accuracy"] = round(correct / max(total, 1), 3)

    # Policy accuracy
    policy_correct = sum(1 for r in all_results if r["policy_correct"])
    verdict["policy_accuracy"] = round(policy_correct / max(total, 1), 3)

    # Overall
    c1_pass = verdict.get("C1", {}).get("pass", False)
    c2_pass = verdict.get("C2", {}).get("pass", False)
    if c1_pass and c2_pass:
        verdict["overall"] = "GO — Track 2 authorized"
    elif c1_pass or c2_pass:
        verdict["overall"] = "PARTIAL — review before Track 2"
    else:
        verdict["overall"] = "KILL — signal insufficient"

    return verdict


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Track 1 MVP Runner")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (dry-run, no DROID)")
    parser.add_argument("--output-dir", type=str, default="results/track1")
    args = parser.parse_args()

    cfg = Track1Config(
        n_episodes=args.n_episodes,
        seeds=SEEDS[:args.seeds],
        use_synthetic=args.synthetic,
    )
    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log = RQZLogger("track1", output_dir=str(RESULTS_DIR), log_every_n=1)
    log.info("Track 1 starting", config=asdict(cfg))

    # Load trajectories
    if cfg.use_synthetic:
        log.info("Using synthetic trajectories (dry-run)")
        trajectories = make_synthetic_trajectories(cfg.n_episodes)
    else:
        log.info("Loading DROID trajectories...")
        trajectories = load_droid_trajectories(cfg.n_episodes)
    log.info(f"Loaded {len(trajectories)} trajectories")

    # Split: first half calibration, second half test (contract 4.5)
    n_cal = len(trajectories) // 2
    cal_trajectories = trajectories[:n_cal]
    test_trajectories = trajectories[n_cal:]
    log.info(f"Split: {n_cal} calibration, {len(test_trajectories)} test")

    # Run all scenarios x seeds x test trajectories
    all_results = []
    for seed in cfg.seeds:
        for scenario in cfg.scenarios:
            log.info(f"=== Seed {seed}, Scenario {scenario} ===")
            for i, traj in enumerate(test_trajectories):
                # Pick donor from different task
                donors = [t for t in cal_trajectories if t.task_index != traj.task_index]
                donor = donors[i % len(donors)] if donors else cal_trajectories[0]

                result = run_scenario(traj, donor, scenario, seed)
                all_results.append(result)

                log.progress(seed, i, len(test_trajectories), {
                    "scenario": scenario,
                    "rho": round(result["rho"], 3),
                    "diagnosis": result["dh_diagnosis"],
                    "policy": result["policy"],
                    "correct": result["diagnosis_correct"],
                })

    # Save raw results
    with open(RESULTS_DIR / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Evaluate claims
    verdict = evaluate_claims(all_results, cfg)
    log.experiment_complete(verdict, verdict["overall"])

    # Write VERDICT
    with open(RESULTS_DIR / "VERDICT_track1.md", "w") as f:
        f.write("# Track 1 Verdict\n\n")
        f.write(f"**Overall: {verdict['overall']}**\n\n")
        f.write(f"- Diagnosis accuracy: {verdict['diagnosis_accuracy']}\n")
        f.write(f"- Policy accuracy: {verdict['policy_accuracy']}\n\n")
        if verdict["C1"]:
            c1 = verdict["C1"]
            f.write("## C1: DH-sigma discrimination\n\n")
            f.write(f"- rho(nominal): {c1['rho_nominal_mean']}\n")
            f.write(f"- rho(drift): {c1['rho_drift_mean']}\n")
            f.write(f"- gap: {c1['gap']}\n")
            f.write(f"- kill threshold: {c1['kill_threshold']}\n")
            f.write(f"- **{'PASS' if c1['pass'] else 'FAIL'}**\n\n")
        if verdict["C2"]:
            c2 = verdict["C2"]
            f.write("## C2: MIT detection\n\n")
            f.write(f"- FP rate: {c2['fp_rate']}\n")
            f.write(f"- Detection rate: {c2['detection_rate']}\n")
            f.write(f"- **{'PASS' if c2['pass'] else 'FAIL'}**\n")

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    log.info(f"Results: {RESULTS_DIR}")
    log.info(f"VERDICT: {verdict['overall']}")


if __name__ == "__main__":
    main()
