"""
HHA — Homeostatic Hamiltonian Agent

"HHA regulates how hard to infer"

Regulates friction gamma based on inter-temporal energy surprise:
  S > S_target -> increase gamma (FREEZE)
  S < S_target -> decrease gamma (ADAPT)

Extracted from:
  HHA_MIT_Unified_Submission/code_reproducibility/hha_mit_system.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable


@dataclass
class HHAConfig:
    gamma_min: float = 0.1
    gamma_max: float = 8.0
    gamma_0: float = 0.5
    alpha: float = 0.8       # Proportional gain
    beta: float = 0.75       # Smoothing factor
    w_delta: float = 1.0     # Weight for energy variation
    w_cross: float = 1.0     # Weight for cross-temporal surprise
    target_k: float = 4.0    # S_target = mu + k * sigma


class StressComputer:
    """
    Computes inter-temporal energy surprise S.

    S_t = w_delta * |E*_t - E*_{t-1}| + w_cross * max(0, E(z*_{t-1}, o_t) - E(z*_{t-1}, o_{t-1}))
    """

    def __init__(self, cfg: HHAConfig):
        self.cfg = cfg
        self.E_prev = None
        self.z_prev = None
        self.o_prev = None

    def reset(self):
        self.E_prev = None
        self.z_prev = None
        self.o_prev = None

    def compute(self, z_star: np.ndarray, E_star: float, obs: np.ndarray,
                energy_fn: Callable) -> Tuple[float, float, float]:
        """Returns (S, delta_E, S_cross)."""
        if self.E_prev is None:
            self.E_prev = E_star
            self.z_prev = z_star.copy()
            self.o_prev = obs.copy()
            return 0.0, 0.0, 0.0

        delta_E = abs(E_star - self.E_prev)
        E_cross = energy_fn(self.z_prev, obs)
        E_prev_prev = energy_fn(self.z_prev, self.o_prev)
        S_cross = max(0.0, E_cross - E_prev_prev)
        S = self.cfg.w_delta * delta_E + self.cfg.w_cross * S_cross

        self.E_prev = E_star
        self.z_prev = z_star.copy()
        self.o_prev = obs.copy()

        return S, delta_E, S_cross

    def compute_from_errors(self, errors: np.ndarray) -> np.ndarray:
        """
        Simplified stress from prediction error sequence (no energy_fn needed).
        Suitable for V-JEPA latent errors.
        S_t = |e_t - e_{t-1}|
        """
        if len(errors) < 2:
            return np.zeros(len(errors))
        S = np.zeros(len(errors))
        S[1:] = np.abs(np.diff(errors))
        return S


class HHAController:
    """
    Homeostatic controller: regulates friction gamma via proportional control.
    """

    def __init__(self, cfg: HHAConfig = None):
        self.cfg = cfg or HHAConfig()
        self.gamma = self.cfg.gamma_0
        self.S_target = None
        self.mu_rest = None
        self.sigma_rest = None
        self.calibrated = False

    def reset(self):
        self.gamma = self.cfg.gamma_0

    def calibrate(self, S_history: List[float]) -> float:
        S_arr = np.array(S_history)
        self.mu_rest = float(np.mean(S_arr))
        self.sigma_rest = float(np.std(S_arr)) + 1e-8
        self.S_target = self.mu_rest + self.cfg.target_k * self.sigma_rest
        self.calibrated = True
        return self.S_target

    def update(self, S: float) -> float:
        if not self.calibrated:
            return self.gamma
        gamma_raw = self.gamma + self.cfg.alpha * (S - self.S_target)
        gamma_raw = np.clip(gamma_raw, self.cfg.gamma_min, self.cfg.gamma_max)
        self.gamma = self.cfg.beta * self.gamma + (1 - self.cfg.beta) * gamma_raw
        self.gamma = float(np.clip(self.gamma, self.cfg.gamma_min, self.cfg.gamma_max))
        return self.gamma

    def is_saturated(self, threshold: float = 0.9) -> bool:
        return self.gamma >= threshold * self.cfg.gamma_max

    def get_saturation_ratio(self) -> float:
        return (self.gamma - self.cfg.gamma_min) / (self.cfg.gamma_max - self.cfg.gamma_min)
