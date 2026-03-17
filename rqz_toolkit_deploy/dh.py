"""
DH — DualHorizon (noise vs drift discrimination)

"DH discriminates noise from drift"

Key signals:
  R = exp(-mean_error * scale / N1)  — resilience [0,1]
  trend = slope of R over window     — drift detection
  rho = sign_change_rate             — noise (0.5) vs drift (0.0)
  P = w_R * (1 - R) + |trend|        — detection score

Extracted from:
  ARH_DH_Unified_Submission/code_reproducibility/dualhorizon/tier3_tcpd.py
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DHConfig:
    N1: int = 5                    # Short-term window for R
    sign_window: int = 20          # Window for rho (sign change rate)
    trend_window: int = 10         # Window for trend slope
    error_scale: float = 5.0       # Scaling for R computation
    w_R: float = 0.9               # Weight for (1-R) in P score


class DualHorizon:
    """
    DualHorizon diagnostic: noise vs drift discrimination.

    Usage:
        dh = DualHorizon(DHConfig())
        dh.calibrate(baseline_errors)
        for error in stream:
            dh.observe(error)
            R, trend, P = dh.compute()
            rho = dh.compute_rho()
            # rho ~ 0.5 -> noise, rho ~ 0.0 -> drift
    """

    def __init__(self, cfg: DHConfig = None):
        self.cfg = cfg or DHConfig()
        self.error_history: List[float] = []
        self.R_history: List[float] = []
        self.baseline_error: float = 1.0

    def reset(self):
        self.error_history = []
        self.R_history = []

    def calibrate(self, errors: np.ndarray):
        self.baseline_error = float(np.mean(np.abs(errors))) + 1e-8

    def observe(self, error: float):
        self.error_history.append(error)
        max_len = max(self.cfg.N1, self.cfg.sign_window) * 3
        if len(self.error_history) > max_len:
            self.error_history = self.error_history[-max_len:]

    def compute_R(self) -> float:
        """Resilience R in [0, 1]. High R = small recent errors."""
        if len(self.error_history) < self.cfg.N1:
            return 1.0
        errors = np.array(self.error_history[-self.cfg.N1:])
        mean_abs = np.mean(np.abs(errors))
        normalized = mean_abs / self.baseline_error
        return float(np.clip(math.exp(-normalized * self.cfg.error_scale / self.cfg.N1), 0, 1))

    def compute_trend(self) -> float:
        """Slope of R over trend_window. Negative = R dropping = drift."""
        if len(self.R_history) < self.cfg.trend_window:
            return 0.0
        recent_R = np.array(self.R_history[-self.cfg.trend_window:])
        x = np.arange(len(recent_R))
        slope = np.polyfit(x, recent_R, 1)[0]
        return float(np.clip(slope * 20, -1, 1))

    def compute_rho(self) -> float:
        """
        Sign change rate rho in [0, 1].
        rho ~ 0.5 -> random noise (i.i.d.)
        rho ~ 0.0 -> systematic drift (same sign)
        """
        if len(self.error_history) < self.cfg.sign_window:
            return 0.5
        recent = np.array(self.error_history[-self.cfg.sign_window:])
        signs = np.sign(recent)
        changes = np.sum(signs[1:] != signs[:-1])
        return float(changes / (len(signs) - 1))

    def compute(self) -> Tuple[float, float, float]:
        """
        Compute R, trend, and detection score P.
        P = w_R * (1 - R) + |trend|
        """
        R = self.compute_R()
        self.R_history.append(R)
        if len(self.R_history) > self.cfg.trend_window * 3:
            self.R_history = self.R_history[-(self.cfg.trend_window * 3):]
        trend = self.compute_trend()
        P = self.cfg.w_R * (1 - R) + abs(trend)
        return R, trend, P

    def diagnose(self, rho_noise_threshold: float = 0.35,
                 rho_drift_threshold: float = 0.15) -> str:
        """
        Classify current state based on rho.
        Returns: 'NOISE', 'DRIFT', or 'INVESTIGATE'
        """
        rho = self.compute_rho()
        if rho > rho_noise_threshold:
            return "NOISE"
        elif rho < rho_drift_threshold:
            return "DRIFT"
        else:
            return "INVESTIGATE"
