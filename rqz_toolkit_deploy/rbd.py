"""
RBD — Resonance-Based Diagnostics

"RBD defines what is plausible"

Computes model-physics resonance R:
  R = exp(-||f(z) - Phi(z)||^2 / sigma^2)

Extracted from:
  - RBD_Submission/code_reproducibility/resonance_blending.py
  - HHA_MIT_Unified_Submission/code_reproducibility/hha_mit_system.py
"""

import numpy as np
from typing import List, Optional, Callable


class ResonanceComputer:
    """
    Computes model-physics resonance R in [0, 1].

    R = exp(-||model_pred - physics_pred||^2 / sigma^2)

    For video/latent-only mode (no physics prior), use cosine coherence
    via compute_latent() instead.
    """

    def __init__(self, physics_fn: Optional[Callable] = None, sigma: float = 0.5):
        self.physics_fn = physics_fn
        self.sigma = sigma
        self.R_nominal: Optional[float] = None

    def set_physics(self, physics_fn: Callable):
        self.physics_fn = physics_fn

    def calibrate(self, R_history: List[float]) -> float:
        self.R_nominal = float(np.mean(R_history))
        return self.R_nominal

    def compute(self, model_pred: np.ndarray, physics_pred: np.ndarray) -> float:
        """R from model vs physics prior. Returns 1.0 if no physics pred."""
        if physics_pred is None:
            return 1.0
        diff = np.linalg.norm(model_pred - physics_pred)
        return float(np.exp(-diff ** 2 / self.sigma ** 2))

    def compute_latent(self, pred_embedding: np.ndarray, obs_embedding: np.ndarray) -> float:
        """
        R from latent coherence (no physics prior).
        Uses cosine similarity as proxy for resonance.
        Suitable for V-JEPA and similar latent prediction architectures.
        """
        norm_pred = np.linalg.norm(pred_embedding)
        norm_obs = np.linalg.norm(obs_embedding)
        if norm_pred < 1e-8 or norm_obs < 1e-8:
            return 0.0
        cosine = np.dot(pred_embedding, obs_embedding) / (norm_pred * norm_obs)
        return float(np.clip((cosine + 1) / 2, 0, 1))  # map [-1,1] to [0,1]


class ResonanceBlender:
    """
    Blends model and physics predictions weighted by resonance R.

    output = R * model_pred + (1 - R) * physics_pred

    When R is high (model aligns with physics), trust the model.
    When R is low (model diverges), fall back to physics.
    """

    def __init__(self, resonance: ResonanceComputer):
        self.resonance = resonance

    def blend(self, model_pred: np.ndarray, physics_pred: np.ndarray) -> np.ndarray:
        R = self.resonance.compute(model_pred, physics_pred)
        return R * model_pred + (1 - R) * physics_pred

    def blend_vectorial(self, model_pred: np.ndarray, physics_pred: np.ndarray,
                        sigma_per_dim: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-dimension blending with optional per-dim sigma."""
        if sigma_per_dim is None:
            sigma_per_dim = np.full(model_pred.shape, self.resonance.sigma)
        diff = model_pred - physics_pred
        R_vec = np.exp(-diff ** 2 / sigma_per_dim ** 2)
        return R_vec * model_pred + (1 - R_vec) * physics_pred
