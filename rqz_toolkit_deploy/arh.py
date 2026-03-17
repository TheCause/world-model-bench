"""
ARH — Adaptive Resonance Horizon

"ARH defines how long to trust"

Adapts prediction horizon N based on confidence C:
  N_t = N_min + (N_max - N_min) * C_t
  C drops when resonance drops -> shorter horizon

Extracted from:
  ARH_DH_Unified_Submission/code_reproducibility/arh/tier2_dmc.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ARHConfig:
    N_min: int = 4
    N_max: int = 20
    beta_slow: float = 0.95
    beta_fast: float = 0.5
    W_R: int = 20             # Window for running stats
    k: float = 2.0            # R_crit = mu_R - k * sigma_R
    alpha: float = 0.2        # Reset factor when R < R_crit
    tau: float = 0.7          # Stability threshold
    lambda_zscore: float = 3.0
    C_0: float = 0.5
    warmup_steps: int = 30


class _RunningStats:
    """Online mean/std tracker with window."""

    def __init__(self, window: int = 20):
        self.window = window
        self.values: List[float] = []

    def update(self, x: float):
        self.values.append(x)
        if len(self.values) > self.window:
            self.values = self.values[-self.window:]

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def std(self) -> float:
        return float(np.std(self.values)) if len(self.values) > 1 else 1e-8


class ARHSystem:
    """
    Adaptive Resonance Horizon.

    Key equations:
    - N_t = N_min + (N_max - N_min) * C_t
    - R_t = clip(1 - (e_t - mu_e) / (lambda * sigma_e), 0, 1)
    - beta_t = beta_slow * (1 - O_t) + beta_fast * O_t
    - C_{t+1} = alpha * R_t  if R_t < R_crit  else  beta_t * C_t + (1 - beta_t) * R_t
    """

    def __init__(self, config: ARHConfig = None):
        self.cfg = config or ARHConfig()
        self.C = self.cfg.C_0
        self.t = 0
        self.error_stats = _RunningStats(self.cfg.W_R)
        self.resonance_stats = _RunningStats(self.cfg.W_R)
        self.log: Dict[str, List] = {"C": [], "N": [], "R": [], "status": []}

    def reset(self):
        self.C = self.cfg.C_0
        self.t = 0
        self.error_stats = _RunningStats(self.cfg.W_R)
        self.resonance_stats = _RunningStats(self.cfg.W_R)
        self.log = {"C": [], "N": [], "R": [], "status": []}

    def get_horizon(self) -> int:
        return int(self.cfg.N_min + (self.cfg.N_max - self.cfg.N_min) * self.C)

    def step_from_error(self, prediction_error: float) -> Dict:
        """
        Update ARH from a prediction error value.
        Standalone mode — no world model needed.
        """
        self.t += 1

        e_t = prediction_error
        self.error_stats.update(e_t)
        mu_e = self.error_stats.mean()
        sigma_e = self.error_stats.std()

        # Resonance via z-score
        R_t = float(np.clip(1 - (e_t - mu_e) / (self.cfg.lambda_zscore * sigma_e + 1e-8), 0, 1))

        # Adaptive beta
        O_t = 1 - self.C
        beta_t = self.cfg.beta_slow * (1 - O_t) + self.cfg.beta_fast * O_t

        # Critical threshold
        if self.t <= self.cfg.warmup_steps:
            R_crit = 0.0
        else:
            R_crit = max(0, self.resonance_stats.mean() - self.cfg.k * self.resonance_stats.std())

        self.resonance_stats.update(R_t)

        # Confidence update
        if R_t < R_crit:
            C_new = self.cfg.alpha * R_t
        else:
            C_new = beta_t * self.C + (1 - beta_t) * R_t

        self.C = float(np.clip(C_new, 0, 1))
        N_t = self.get_horizon()
        status = "STABLE" if self.C > self.cfg.tau else "VIGILANT"

        self.log["C"].append(self.C)
        self.log["N"].append(N_t)
        self.log["R"].append(R_t)
        self.log["status"].append(1 if status == "STABLE" else 0)

        return {"status": status, "C": self.C, "N": N_t, "R": R_t}

    def get_stats(self) -> Dict:
        statuses = np.array(self.log["status"])
        return {
            "stable_pct": float(np.mean(statuses) * 100) if len(statuses) > 0 else 0,
            "mean_C": float(np.mean(self.log["C"])) if self.log["C"] else 0,
            "mean_N": float(np.mean(self.log["N"])) if self.log["N"] else 0,
            "mean_R": float(np.mean(self.log["R"])) if self.log["R"] else 0,
        }
