"""
MIT — Model Invalidity Test

"MIT decides when to rebuild"

Triggers reconstruction when ALL 3 pillars are satisfied (conjunction):
  1. Cumulative stress Sigma_S > Theta (persistence)
  2. Mean energy E_bar > epsilon (capacity)
  3. Mean resonance R_bar < delta (validity)

Key insight: Conjunction prevents false positives under noise.

Extracted from:
  HHA_MIT_Unified_Submission/code_reproducibility/hha_mit_system.py
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MITConfig:
    W: int = 25                    # Window size
    persist_N: int = 10            # Consecutive windows required
    k_theta: float = 5.0           # Theta = k * sqrt(W)
    q_eps: float = 0.93            # Energy percentile
    R_drop_factor: float = 0.5     # Trigger if R < 50% of nominal
    refractory: int = 50           # Cooldown after rebuild


class MITDetector:
    """
    Model Invalidity Test — conjunction of 3 pillars.
    """

    def __init__(self, cfg: MITConfig = None):
        self.cfg = cfg or MITConfig()
        self.S_buffer = deque(maxlen=self.cfg.W)
        self.E_buffer = deque(maxlen=self.cfg.W)
        self.R_buffer = deque(maxlen=self.cfg.W)

        self.theta: Optional[float] = None
        self.epsilon: Optional[float] = None
        self.R_nominal: Optional[float] = None

        self.mu_S = 0.0
        self.sigma_S = 1.0

        self.persist_count = 0
        self.refractory_until = -1
        self.t = 0
        self.n_reconstructions = 0
        self.calibrated = False

        self._calib_E: List[float] = []
        self._calib_R: List[float] = []

    def reset(self):
        self.S_buffer.clear()
        self.E_buffer.clear()
        self.R_buffer.clear()
        self.persist_count = 0
        self.refractory_until = -1
        self.t = 0
        self.n_reconstructions = 0

    def add_calibration_sample(self, E: float, R: float):
        self._calib_E.append(E)
        self._calib_R.append(R)

    def calibrate(self, S_history: List[float]) -> Dict:
        S_arr = np.array(S_history)
        self.mu_S = float(np.mean(S_arr))
        self.sigma_S = float(np.std(S_arr)) + 1e-8
        self.theta = self.cfg.k_theta * np.sqrt(self.cfg.W)
        self.epsilon = float(np.percentile(self._calib_E, self.cfg.q_eps * 100)) if self._calib_E else 1.0
        self.R_nominal = float(np.mean(self._calib_R)) if self._calib_R else 0.9
        self.calibrated = True
        self._calib_E = []
        self._calib_R = []
        return {"theta": self.theta, "epsilon": self.epsilon, "R_nominal": self.R_nominal}

    def update(self, S: float, E: float, R: float) -> Dict:
        """Returns dict with 'trigger' bool and pillar states."""
        self.t += 1
        self.S_buffer.append(S)
        self.E_buffer.append(E)
        self.R_buffer.append(R)

        result = {
            "trigger": False, "sigma_S": 0.0, "E_bar": 0.0, "R_bar": 0.0,
            "pillars": {"persistence": False, "capacity": False, "validity": False},
        }

        if not self.calibrated or len(self.S_buffer) < self.cfg.W:
            return result
        if self.t < self.refractory_until:
            return result

        # Pillar 1: cumulative normalized stress
        S_norm = [(s - self.mu_S) / self.sigma_S for s in self.S_buffer]
        sigma_S = sum(max(0, s) for s in S_norm)
        p1 = sigma_S > self.theta

        # Pillar 2: mean energy floor
        E_bar = float(np.mean(list(self.E_buffer)))
        p2 = E_bar > self.epsilon

        # Pillar 3: mean resonance drop
        R_bar = float(np.mean(list(self.R_buffer)))
        p3 = R_bar < self.cfg.R_drop_factor * self.R_nominal

        result["sigma_S"] = sigma_S
        result["E_bar"] = E_bar
        result["R_bar"] = R_bar
        result["pillars"] = {"persistence": p1, "capacity": p2, "validity": p3}

        if p1 and p2 and p3:
            self.persist_count += 1
        else:
            self.persist_count = 0

        if self.persist_count >= self.cfg.persist_N:
            result["trigger"] = True
            self.persist_count = 0
            self.refractory_until = self.t + self.cfg.refractory
            self.n_reconstructions += 1

        return result
