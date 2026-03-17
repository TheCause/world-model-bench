"""
RQZ Toolkit — Portable diagnostic modules for world models.

5 modules:
- RBD: Resonance-Based Diagnostics (plausibility filter)
- HHA: Homeostatic Hamiltonian Agent (adapt/freeze regulation)
- ARH: Adaptive Resonance Horizon (trust horizon)
- MIT: Model Invalidity Test (rebuild trigger)
- DH: DualHorizon (noise vs drift discrimination)

Phrase signature:
  "RBD defines what is plausible, HHA regulates how hard to infer,
   ARH defines how long to trust, MIT decides when to rebuild,
   DH discriminates noise from drift"
"""

from .rbd import ResonanceComputer, ResonanceBlender
from .hha import HHAController, HHAConfig, StressComputer
from .arh import ARHSystem, ARHConfig
from .mit import MITDetector, MITConfig
from .dh import DualHorizon, DHConfig
from .anatomy import CUSUMDetector, CUSUMConfig, VectorialMonitor, VectorialConfig
from .pipeline import RQZPipeline, PipelineConfig

__version__ = "0.1.0"
__all__ = [
    "ResonanceComputer", "ResonanceBlender",
    "HHAController", "HHAConfig", "StressComputer",
    "ARHSystem", "ARHConfig",
    "MITDetector", "MITConfig",
    "DualHorizon", "DHConfig",
    "CUSUMDetector", "CUSUMConfig",
    "VectorialMonitor", "VectorialConfig",
    "RQZPipeline", "PipelineConfig",
]
