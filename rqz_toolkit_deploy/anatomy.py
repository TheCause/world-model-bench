"""
Anatomy — Spatial Localization & Targeted Recovery

Extracted from Anatomy of Failure (spatial validity paper).
Three-stage pipeline: Detect → Localize → Recover.

Components:
  - CUSUMDetector: per-dimension CUSUM change-point detection (Page 1954)
  - VectorialMonitor: spatial R + rho for localization
  - PartialWarmStrategy: targeted reconstruction (freeze trunk, retrain failed heads)

Extracted from:
  Anatomy_Failure_Submission/code_reproducibility/exp_a_spatial_localization.py
  Anatomy_Failure_Submission/code_reproducibility/exp_b_targeted_reconstruction.py
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CUSUMConfig:
    slack: float = 0.5           # k parameter
    threshold: float = 8.0       # h alarm threshold
    persist_k: int = 20          # consecutive steps above h


@dataclass
class VectorialConfig:
    w_ebar: int = 20             # MMAE window for smoothed |e|
    w_rho: int = 30              # Window for sign change rate
    cal_percentile: float = 95.0 # Percentile for sigma calibration
    delta: float = 0.3           # R threshold for alarm
    rho_thresh: float = 0.35     # rho threshold (below = drift)
    persist_k: int = 20          # Persistence for spatial alarm


class CUSUMDetector:
    """
    Per-dimension CUSUM change-point detector.

    S_i = max(0, S_i + |e_i| - mu_i * (1 + k))
    Alarm when S_i > h for persist_k consecutive steps.
    """

    def __init__(self, state_dim: int, cfg: CUSUMConfig = None):
        self.cfg = cfg or CUSUMConfig()
        self.d = state_dim
        self.mu = np.ones(state_dim)
        self.S = np.zeros(state_dim)
        self.cal_data: Dict[int, List[float]] = {i: [] for i in range(state_dim)}
        self.persist = np.zeros(state_dim, dtype=int)
        self.detected: Set[int] = set()

    def reset(self):
        self.S = np.zeros(self.d)
        self.persist = np.zeros(self.d, dtype=int)
        self.detected = set()

    def calibrate(self):
        for i in range(self.d):
            if self.cal_data[i]:
                self.mu[i] = np.mean(self.cal_data[i]) + 1e-8

    def step(self, e_raw: np.ndarray, is_burn_in: bool = False) -> Optional[np.ndarray]:
        """
        Process one step. During burn-in, accumulates calibration data.
        After burn-in, returns boolean alarm array (True = dimension flagged).
        """
        if is_burn_in:
            for i in range(self.d):
                self.cal_data[i].append(abs(e_raw[i]))
            return None

        alarms = np.zeros(self.d, dtype=bool)
        for i in range(self.d):
            self.S[i] = max(0.0, self.S[i] + abs(e_raw[i]) - self.mu[i] * (1 + self.cfg.slack))
            if self.S[i] > self.cfg.threshold:
                self.persist[i] += 1
                if self.persist[i] >= self.cfg.persist_k and i not in self.detected:
                    self.detected.add(i)
                    alarms[i] = True
            else:
                self.persist[i] = 0
        return alarms


class VectorialMonitor:
    """
    Spatial localization via per-dimension R and rho.

    R_i = exp(-e_bar_i^2 / sigma_i^2)  — per-dimension resilience
    rho_i = sign_change_rate(e_i)       — noise vs drift per dimension

    Detection: R_i < delta AND rho_i < rho_thresh for persist_k steps.
    """

    def __init__(self, state_dim: int, cfg: VectorialConfig = None):
        self.cfg = cfg or VectorialConfig()
        self.d = state_dim
        self.sigma = np.ones(state_dim)
        self.cal: Dict[int, List[float]] = {i: [] for i in range(state_dim)}
        self.abs_buf: Dict[int, deque] = {i: deque(maxlen=self.cfg.w_ebar) for i in range(state_dim)}
        self.raw_buf: Dict[int, deque] = {i: deque(maxlen=self.cfg.w_rho) for i in range(state_dim)}
        self.persist = np.zeros(state_dim, dtype=int)
        self.detected: Set[int] = set()

    def reset(self):
        self.persist = np.zeros(self.d, dtype=int)
        self.detected = set()
        for i in range(self.d):
            self.abs_buf[i].clear()
            self.raw_buf[i].clear()

    def calibrate(self):
        for i in range(self.d):
            if self.cal[i]:
                self.sigma[i] = np.percentile(self.cal[i], self.cfg.cal_percentile) + 1e-8
        return self.sigma.copy()

    def step(self, e_raw: np.ndarray, is_burn_in: bool = False
             ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (R, rho) arrays, or (None, None) during burn-in."""
        e_bar = np.zeros(self.d)
        for i in range(self.d):
            self.raw_buf[i].append(e_raw[i])
            self.abs_buf[i].append(abs(e_raw[i]))
            e_bar[i] = np.mean(self.abs_buf[i])

        if is_burn_in:
            for i in range(self.d):
                if len(self.abs_buf[i]) == self.cfg.w_ebar:
                    self.cal[i].append(e_bar[i])
            return None, None

        R = np.exp(-(e_bar ** 2) / (self.sigma ** 2))
        rho = np.full(self.d, 0.5)
        for i in range(self.d):
            buf = list(self.raw_buf[i])
            if len(buf) >= self.cfg.w_rho:
                signs = np.sign(buf)
                rho[i] = np.sum(signs[:-1] != signs[1:]) / (len(buf) - 1)
        return R, rho

    def detect(self, R: np.ndarray, rho: np.ndarray) -> Set[int]:
        """Check which dimensions are flagged. Returns newly detected dims."""
        newly_detected = set()
        for i in range(self.d):
            if R[i] < self.cfg.delta and rho[i] < self.cfg.rho_thresh:
                self.persist[i] += 1
                if self.persist[i] >= self.cfg.persist_k and i not in self.detected:
                    self.detected.add(i)
                    newly_detected.add(i)
            else:
                self.persist[i] = 0
        return newly_detected
