"""
RQZ Pipeline — Orchestration of 5 diagnostic modules.

Architecture contract (mandatory order):
  1. DETECTION  (DH-Delta / MIT Pillar 1) — high FP rate
  2. QUALIFICATION (DH-sigma / MIT Pillar 3) — cancels FPs
  3. ACTION (HHA) — Adapt / Freeze / Rebuild

Full pipeline:
  RBD (plausibility) → DH (detect + qualify) → MIT (rebuild?) → HHA (regulate) → ARH (horizon)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from .rbd import ResonanceComputer
from .hha import HHAController, HHAConfig, StressComputer
from .arh import ARHSystem, ARHConfig
from .mit import MITDetector, MITConfig
from .dh import DualHorizon, DHConfig


@dataclass
class PipelineConfig:
    hha: HHAConfig = None
    arh: ARHConfig = None
    mit: MITConfig = None
    dh: DHConfig = None
    rbd_sigma: float = 0.5
    calibration_steps: int = 100

    def __post_init__(self):
        self.hha = self.hha or HHAConfig()
        self.arh = self.arh or ARHConfig()
        self.mit = self.mit or MITConfig()
        self.dh = self.dh or DHConfig()


class RQZPipeline:
    """
    Unified RQZ diagnostic pipeline.

    Usage:
        pipeline = RQZPipeline()
        pipeline.calibrate(nominal_errors, nominal_R_values)
        for error, R in stream:
            result = pipeline.step(error, R)
            # result['regime'] in {'ADAPT', 'FREEZE', 'REBUILD'}
            # result['horizon'] = adaptive N
            # result['diagnosis'] in {'NOISE', 'DRIFT', 'INVESTIGATE'}
    """

    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self.rbd = ResonanceComputer(sigma=self.cfg.rbd_sigma)
        self.hha = HHAController(self.cfg.hha)
        self.arh = ARHSystem(self.cfg.arh)
        self.mit = MITDetector(self.cfg.mit)
        self.dh = DualHorizon(self.cfg.dh)
        self.t = 0
        self.history: List[Dict] = []

    def calibrate(self, errors: np.ndarray, R_values: Optional[np.ndarray] = None,
                  E_values: Optional[np.ndarray] = None):
        """
        Calibrate all modules from nominal data.

        Args:
            errors: prediction error sequence during nominal operation
            R_values: resonance values (optional, defaults to 1.0)
            E_values: energy values (optional, defaults to errors^2)
        """
        n = len(errors)
        if R_values is None:
            R_values = np.ones(n)
        if E_values is None:
            E_values = errors ** 2

        # Stress from error diffs
        S_values = np.zeros(n)
        S_values[1:] = np.abs(np.diff(errors))

        # DH calibration
        self.dh.calibrate(errors)

        # HHA calibration
        self.hha.calibrate(S_values.tolist())

        # MIT calibration
        for i in range(n):
            self.mit.add_calibration_sample(float(E_values[i]), float(R_values[i]))
        self.mit.calibrate(S_values.tolist())

        # RBD calibration
        self.rbd.calibrate(R_values.tolist())

    def step(self, prediction_error: float, R: float = 1.0,
             E: Optional[float] = None) -> Dict:
        """
        Process one timestep through the full pipeline.

        Args:
            prediction_error: scalar prediction error
            R: resonance value (from RBD or latent coherence)
            E: energy value (defaults to error^2)

        Returns:
            Dict with regime, horizon, diagnosis, and all signals.
        """
        self.t += 1
        if E is None:
            E = prediction_error ** 2

        # Stress (simplified: |delta_error|)
        S = abs(prediction_error - (self.history[-1]["error"] if self.history else 0.0))

        # 1. DH: detect + qualify
        self.dh.observe(prediction_error)
        R_dh, trend, P = self.dh.compute()
        diagnosis = self.dh.diagnose()

        # 2. MIT: rebuild decision
        mit_result = self.mit.update(S, E, R)

        # 3. HHA: regulate gamma
        gamma = self.hha.update(S)

        # 4. ARH: adaptive horizon
        arh_result = self.arh.step_from_error(prediction_error)

        # Determine regime
        if mit_result["trigger"]:
            regime = "REBUILD"
        elif self.hha.is_saturated():
            regime = "FREEZE"
        else:
            regime = "ADAPT"

        result = {
            "t": self.t,
            "error": prediction_error,
            "R": R,
            "E": E,
            "S": S,
            "gamma": gamma,
            "regime": regime,
            "horizon": arh_result["N"],
            "confidence": arh_result["C"],
            "diagnosis": diagnosis,
            "rho": self.dh.compute_rho(),
            "P": P,
            "trend": trend,
            "mit_trigger": mit_result["trigger"],
            "mit_pillars": mit_result["pillars"],
        }

        self.history.append(result)
        return result

    def get_summary(self) -> Dict:
        """Summary statistics over the run."""
        if not self.history:
            return {}
        regimes = [h["regime"] for h in self.history]
        return {
            "total_steps": len(self.history),
            "adapt_pct": regimes.count("ADAPT") / len(regimes) * 100,
            "freeze_pct": regimes.count("FREEZE") / len(regimes) * 100,
            "rebuild_count": sum(1 for h in self.history if h["mit_trigger"]),
            "mean_horizon": np.mean([h["horizon"] for h in self.history]),
            "mean_confidence": np.mean([h["confidence"] for h in self.history]),
        }
