"""
Baselines and post-hoc competitors for "Beyond Binary Validity" benchmark.

Baselines (T1 — detection):
  ADWIN, DDM, KSWIN from `river` library

Post-hoc competitors (C3 — non-trivialite):
  PH1-PH5 attempt to discriminate NOISE from DRIFT using simple heuristics

All receive the same signal e(t) as RQZ modules.

Contract reference: contract.md v1.0, sections 4.4 and C3-detail
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DetectionResult:
    """Result from a drift detector."""
    detected: bool
    onset: int            # timestep of first detection (-1 if not detected)
    all_detections: List[int]  # all timesteps where detection fired


@dataclass
class DiagnosisResult:
    """Result from a post-hoc diagnostic attempt."""
    predicted_class: str  # NOISE, ABRUPT_DRIFT, GRADUAL_DRIFT, or UNKNOWN
    confidence: float     # [0, 1]
    rho: Optional[float]  # sign change rate if applicable


# =============================================================================
# Baselines (T1 — detection only)
# =============================================================================

class ADWINBaseline:
    """ADWIN drift detector wrapper."""

    def __init__(self):
        try:
            from river.drift import ADWIN
            self.detector = ADWIN()
        except ImportError:
            raise ImportError("pip install river")
        self.detections: List[int] = []
        self.t = 0

    def reset(self):
        from river.drift import ADWIN
        self.detector = ADWIN()
        self.detections = []
        self.t = 0

    def update(self, error: float) -> bool:
        self.detector.update(error)
        self.t += 1
        if self.detector.drift_detected:
            self.detections.append(self.t)
            return True
        return False

    def result(self) -> DetectionResult:
        return DetectionResult(
            detected=len(self.detections) > 0,
            onset=self.detections[0] if self.detections else -1,
            all_detections=self.detections,
        )


class PageHinkleyBaseline:
    """PageHinkley drift detector wrapper.

    Replaces DDM (which is designed for binary classification, not
    continuous prediction error). PageHinkley monitors the mean of
    a continuous signal and detects significant increases.

    Pivot documented: DDM→PageHinkley, reason: DDM false positives
    on stable continuous signals.
    """

    def __init__(self):
        try:
            from river.drift import PageHinkley
            self.detector = PageHinkley()
        except ImportError:
            raise ImportError("pip install river")
        self.detections: List[int] = []
        self.t = 0

    def reset(self):
        from river.drift import PageHinkley
        self.detector = PageHinkley()
        self.detections = []
        self.t = 0

    def update(self, error: float) -> bool:
        self.detector.update(error)
        self.t += 1
        if self.detector.drift_detected:
            self.detections.append(self.t)
            return True
        return False

    def result(self) -> DetectionResult:
        return DetectionResult(
            detected=len(self.detections) > 0,
            onset=self.detections[0] if self.detections else -1,
            all_detections=self.detections,
        )


class KSWINBaseline:
    """KSWIN drift detector wrapper."""

    def __init__(self):
        try:
            from river.drift import KSWIN
            self.detector = KSWIN()
        except ImportError:
            raise ImportError("pip install river")
        self.detections: List[int] = []
        self.t = 0

    def reset(self):
        from river.drift import KSWIN
        self.detector = KSWIN()
        self.detections = []
        self.t = 0

    def update(self, error: float) -> bool:
        self.detector.update(error)
        self.t += 1
        if self.detector.drift_detected:
            self.detections.append(self.t)
            return True
        return False

    def result(self) -> DetectionResult:
        return DetectionResult(
            detected=len(self.detections) > 0,
            onset=self.detections[0] if self.detections else -1,
            all_detections=self.detections,
        )


# =============================================================================
# Post-hoc competitors (C3 — diagnostic non-trivialite)
# =============================================================================

def compute_rho(errors: np.ndarray, window: int = 20) -> float:
    """Sign change rate on error RESIDUALS. rho ~0.5 = noise, ~0 = drift.

    Operates on differences d(t) = e(t) - e(t-1), not raw values.
    Noise: d(t) alternates sign (i.i.d.) → rho ~ 0.5
    Drift: d(t) same sign (monotonic) → rho ~ 0
    """
    if len(errors) < window + 1:
        return 0.5
    recent = errors[-(window + 1):]
    diffs = np.diff(recent)  # residuals
    signs = np.sign(diffs)
    # Remove zeros (no change)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.5
    changes = np.sum(signs[1:] != signs[:-1])
    return float(changes / (len(signs) - 1))


def ph1_adwin_rho(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH1: ADWIN + rho post-hoc. Run ADWIN, then compute rho on post-detection window."""
    adwin = ADWINBaseline()
    for e in errors:
        adwin.update(float(e))
    det = adwin.result()

    if not det.detected:
        return DiagnosisResult("NOMINAL", 0.5, None)

    # Compute rho on window after first detection
    start = max(0, det.onset)
    post_errors = errors[start:start + window]
    if len(post_errors) < 5:
        return DiagnosisResult("UNKNOWN", 0.3, None)

    rho = compute_rho(post_errors, len(post_errors))
    if rho > 0.35:
        return DiagnosisResult("NOISE", rho, rho)
    elif rho < 0.15:
        return DiagnosisResult("ABRUPT_DRIFT", 1.0 - rho, rho)
    else:
        return DiagnosisResult("GRADUAL_DRIFT", 0.5, rho)


def ph2_ddm_rho(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH2: DDM + rho post-hoc."""
    ddm = PageHinkleyBaseline()
    for e in errors:
        ddm.update(float(e))
    det = ddm.result()

    if not det.detected:
        return DiagnosisResult("NOMINAL", 0.5, None)

    start = max(0, det.onset)
    post_errors = errors[start:start + window]
    if len(post_errors) < 5:
        return DiagnosisResult("UNKNOWN", 0.3, None)

    rho = compute_rho(post_errors, len(post_errors))
    if rho > 0.35:
        return DiagnosisResult("NOISE", rho, rho)
    elif rho < 0.15:
        return DiagnosisResult("ABRUPT_DRIFT", 1.0 - rho, rho)
    else:
        return DiagnosisResult("GRADUAL_DRIFT", 0.5, rho)


def ph3_kswin_rho(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH3: KSWIN + rho post-hoc."""
    kswin = KSWINBaseline()
    for e in errors:
        kswin.update(float(e))
    det = kswin.result()

    if not det.detected:
        return DiagnosisResult("NOMINAL", 0.5, None)

    start = max(0, det.onset)
    post_errors = errors[start:start + window]
    if len(post_errors) < 5:
        return DiagnosisResult("UNKNOWN", 0.3, None)

    rho = compute_rho(post_errors, len(post_errors))
    if rho > 0.35:
        return DiagnosisResult("NOISE", rho, rho)
    elif rho < 0.15:
        return DiagnosisResult("ABRUPT_DRIFT", 1.0 - rho, rho)
    else:
        return DiagnosisResult("GRADUAL_DRIFT", 0.5, rho)


def ph4_slope_heuristic(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH4: Slope heuristic. Positive slope = drift, ~0 slope = noise."""
    if len(errors) < window:
        return DiagnosisResult("UNKNOWN", 0.3, None)

    recent = errors[-window:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]

    if abs(slope) < 0.01:
        return DiagnosisResult("NOISE", 0.6, None)
    elif slope > 0.05:
        return DiagnosisResult("ABRUPT_DRIFT", min(1.0, abs(slope) * 10), None)
    else:
        return DiagnosisResult("GRADUAL_DRIFT", 0.5, None)


def _baseline_plus_continuous_rho(baseline_class, errors: np.ndarray,
                                  window: int = 20) -> DiagnosisResult:
    """Generic: baseline detection + continuous rho (fairness variant).
    rho is computed on the SAME sliding window as DH-sigma, not just post-detection."""
    detector = baseline_class()
    first_detection = -1
    for i, e in enumerate(errors):
        if detector.update(float(e)) and first_detection == -1:
            first_detection = i

    if first_detection == -1:
        return DiagnosisResult("NOMINAL", 0.5, None)

    # Continuous rho: computed on sliding window around detection, same as DH-sigma
    start = max(0, first_detection - window // 2)
    end = min(len(errors), first_detection + window)
    segment = errors[start:end]
    if len(segment) < 5:
        return DiagnosisResult("UNKNOWN", 0.3, None)

    rho = compute_rho(segment, len(segment))
    if rho > 0.35:
        return DiagnosisResult("NOISE", rho, rho)
    elif rho < 0.15:
        return DiagnosisResult("ABRUPT_DRIFT", 1.0 - rho, rho)
    else:
        return DiagnosisResult("GRADUAL_DRIFT", 0.5, rho)


def ph1b_adwin_continuous_rho(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH1b: ADWIN + continuous rho (fairness variant — same window as DH)."""
    return _baseline_plus_continuous_rho(ADWINBaseline, errors, window)


def ph2b_ddm_continuous_rho(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH2b: DDM + continuous rho (fairness variant)."""
    return _baseline_plus_continuous_rho(PageHinkleyBaseline, errors, window)


def ph3b_kswin_continuous_rho(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH3b: KSWIN + continuous rho (fairness variant)."""
    return _baseline_plus_continuous_rho(KSWINBaseline, errors, window)


def ph5_variance_ratio(errors: np.ndarray, window: int = 20) -> DiagnosisResult:
    """PH5: Variance ratio. Var(post) / Var(pre). High ratio = drift, ~1 = noise."""
    if len(errors) < 2 * window:
        return DiagnosisResult("UNKNOWN", 0.3, None)

    mid = len(errors) // 2
    var_pre = np.var(errors[mid - window:mid]) + 1e-8
    var_post = np.var(errors[mid:mid + window]) + 1e-8
    ratio = var_post / var_pre

    if ratio < 2.0:
        return DiagnosisResult("NOISE", 0.6, None)
    elif ratio > 5.0:
        return DiagnosisResult("ABRUPT_DRIFT", min(1.0, ratio / 10), None)
    else:
        return DiagnosisResult("GRADUAL_DRIFT", 0.5, None)


# =============================================================================
# Convenience: run all post-hocs
# =============================================================================

ALL_POST_HOCS = {
    "PH1": ph1_adwin_rho,
    "PH1b": ph1b_adwin_continuous_rho,
    "PH2": ph2_ddm_rho,
    "PH2b": ph2b_ddm_continuous_rho,
    "PH3": ph3_kswin_rho,
    "PH3b": ph3b_kswin_continuous_rho,
    "PH4": ph4_slope_heuristic,
    "PH5": ph5_variance_ratio,
}


def run_all_post_hocs(errors: np.ndarray) -> dict:
    """Run PH1-PH5 and return {name: DiagnosisResult}."""
    results = {}
    for name, fn in ALL_POST_HOCS.items():
        try:
            results[name] = fn(errors)
        except Exception:
            results[name] = DiagnosisResult("ERROR", 0.0, None)
    return results


def evaluate_c3_fair(errors: np.ndarray, onset: int, window: int = 20) -> dict:
    """
    Fair C3 evaluation: run diagnosis on the post-shift segment
    regardless of whether baselines detected the shift.

    This ensures PH1-PH5 are not handicapped by detection failure.
    All methods receive the SAME segment: errors[onset:onset+window].

    Args:
        errors: full error sequence
        onset: ground truth shift timestep
        window: evaluation window size

    Returns:
        {method_name: DiagnosisResult}
    """
    segment = errors[onset:onset + window]
    if len(segment) < 5:
        return {name: DiagnosisResult("UNKNOWN", 0.0, None) for name in ALL_POST_HOCS}

    rho = compute_rho(segment, len(segment))

    # All post-hocs evaluated on same segment
    results = {}

    # rho-based diagnosis (same logic for all)
    def _rho_diagnosis(rho_val):
        if rho_val > 0.35:
            return DiagnosisResult("NOISE", rho_val, rho_val)
        elif rho_val < 0.15:
            return DiagnosisResult("ABRUPT_DRIFT", 1.0 - rho_val, rho_val)
        else:
            return DiagnosisResult("GRADUAL_DRIFT", 0.5, rho_val)

    # PH1-PH3 and PH1b-PH3b: all reduce to rho on the segment
    for name in ["PH1", "PH1b", "PH2", "PH2b", "PH3", "PH3b"]:
        results[name] = _rho_diagnosis(rho)

    # PH4: slope on segment
    results["PH4"] = ph4_slope_heuristic(segment, window=len(segment))

    # PH5: variance ratio (pre/post onset)
    if onset >= window:
        results["PH5"] = ph5_variance_ratio(
            errors[onset - window:onset + window], window=window
        )
    else:
        results["PH5"] = DiagnosisResult("UNKNOWN", 0.0, None)

    return results
