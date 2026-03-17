"""
Perturbations S0-S4 for "Beyond Binary Validity" benchmark.

Each perturbation is a pure function that modifies trajectory data
(actions, observations) at the DATA level, not the metric level.

All functions are deterministic given a seed, independent of the model,
and testable unitarily.

Taxonomy: C_core = {NOMINAL, NOISE, ABRUPT_DRIFT, GRADUAL_DRIFT}

Contract reference: contract.md v1.0, section 4.3
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PerturbationResult:
    """Result of applying a perturbation to a trajectory."""
    observations: np.ndarray   # [T, H, W, C] uint8 — perturbed observations
    actions: np.ndarray        # [T, 7] float64 — perturbed actions
    states: np.ndarray         # [T, 14] float64 — states (never perturbed)
    ground_truth_class: str    # one of C_core
    onset: int                 # timestep where perturbation starts (-1 if NOMINAL)
    scenario_id: str           # S0, S1, S2, S3, S4


def s0_nominal(observations: np.ndarray, actions: np.ndarray,
               states: np.ndarray) -> PerturbationResult:
    """S0 — Nominal. No perturbation. Control condition."""
    return PerturbationResult(
        observations=observations.copy(),
        actions=actions.copy(),
        states=states.copy(),
        ground_truth_class="NOMINAL",
        onset=-1,
        scenario_id="S0",
    )


def s1_obs_noise(observations: np.ndarray, actions: np.ndarray,
                 states: np.ndarray, sigma: float = 0.1,
                 seed: int = 42) -> PerturbationResult:
    """
    S1 — ObsNoise. Gaussian noise on pixel observations after T/2.

    pixels[t > T/2] += N(0, sigma * 255)

    Expected class: NOISE
    Expected policy: FREEZE (R2)
    """
    T = observations.shape[0]
    onset = T // 2
    rng = np.random.RandomState(seed)

    obs = observations.copy().astype(np.float32)
    noise = rng.randn(T - onset, *obs.shape[1:]).astype(np.float32) * sigma * 255
    obs[onset:] = np.clip(obs[onset:] + noise, 0, 255)
    obs = obs.astype(np.uint8)

    return PerturbationResult(
        observations=obs,
        actions=actions.copy(),
        states=states.copy(),
        ground_truth_class="NOISE",
        onset=onset,
        scenario_id="S1",
    )


def s2_action_shift(observations: np.ndarray, actions: np.ndarray,
                    states: np.ndarray, donor_actions: np.ndarray,
                    seed: int = 42) -> PerturbationResult:
    """
    S2 — ActionShift. Replace actions after T/2 with actions from
    a DIFFERENT trajectory (donor_actions).

    The predictor receives (z_k, wrong_action) -> prediction error increases
    because the observation was produced by the real action, not the donor.

    NOTE: This tests detection of action/observation INCOHERENCE, not a
    true world change. The world model receives inputs that don't match
    the observations. Document in paper as: "S2 tests sensitivity to
    input mismatch, not environmental drift."

    Expected class: ABRUPT_DRIFT
    Expected policy: REBUILD (R4)
    """
    T = observations.shape[0]
    onset = T // 2

    acts = actions.copy()
    # Use donor actions, cycling if donor is shorter
    donor_len = donor_actions.shape[0]
    for t in range(onset, T):
        donor_idx = (t - onset) % donor_len
        acts[t] = donor_actions[donor_idx]

    return PerturbationResult(
        observations=observations.copy(),
        actions=acts,
        states=states.copy(),
        ground_truth_class="ABRUPT_DRIFT",
        onset=onset,
        scenario_id="S2",
    )


def s3_object_change(observations: np.ndarray, actions: np.ndarray,
                     states: np.ndarray,
                     donor_observations: np.ndarray) -> PerturbationResult:
    """
    S3 — ObjectChange. Replace observations after T/2 with frames
    from a DIFFERENT task trajectory (donor_observations).

    Expected class: ABRUPT_DRIFT
    Expected policy: REBUILD (R4)
    """
    T = observations.shape[0]
    onset = T // 2

    obs = observations.copy()
    donor_len = donor_observations.shape[0]
    for t in range(onset, T):
        donor_idx = (t - onset) % donor_len
        obs[t] = donor_observations[donor_idx]

    return PerturbationResult(
        observations=obs,
        actions=actions.copy(),
        states=states.copy(),
        ground_truth_class="ABRUPT_DRIFT",
        onset=onset,
        scenario_id="S3",
    )


def s4_gradual_degradation(observations: np.ndarray, actions: np.ndarray,
                           states: np.ndarray,
                           seed: int = 42) -> PerturbationResult:
    """
    S4 — GradualDegradation. Progressive gaussian blur on observations
    from T/4 to 3T/4.

    sigma_blur(t) = k * (t - T/4) for t in [T/4, 3T/4]

    Expected class: GRADUAL_DRIFT
    Expected policy: REBUILD (R5)
    """
    try:
        import cv2
    except ImportError:
        # Fallback: additive noise increasing linearly (no cv2 needed)
        return _s4_fallback_noise(observations, actions, states, seed)

    T = observations.shape[0]
    onset = T // 4
    end = 3 * T // 4

    obs = observations.copy()
    for t in range(onset, min(end + 1, T)):
        progress = (t - onset) / max(end - onset, 1)  # 0 to 1
        # kernel size must be odd, minimum 3
        ksize = max(3, int(progress * 21)) | 1  # 3 to 21
        sigma_blur = progress * 5.0  # 0 to 5
        obs[t] = cv2.GaussianBlur(obs[t], (ksize, ksize), sigma_blur)

    return PerturbationResult(
        observations=obs,
        actions=actions.copy(),
        states=states.copy(),
        ground_truth_class="GRADUAL_DRIFT",
        onset=onset,
        scenario_id="S4",
    )


def _s4_fallback_noise(observations, actions, states, seed):
    """Fallback S4 without cv2: increasing additive noise."""
    T = observations.shape[0]
    onset = T // 4
    end = 3 * T // 4
    rng = np.random.RandomState(seed)

    obs = observations.copy().astype(np.float32)
    for t in range(onset, min(end + 1, T)):
        progress = (t - onset) / max(end - onset, 1)
        sigma = progress * 0.3 * 255
        noise = rng.randn(*obs[t].shape).astype(np.float32) * sigma
        obs[t] = np.clip(obs[t] + noise, 0, 255)

    return PerturbationResult(
        observations=obs.astype(np.uint8),
        actions=actions.copy(),
        states=states.copy(),
        ground_truth_class="GRADUAL_DRIFT",
        onset=onset,
        scenario_id="S4",
    )
