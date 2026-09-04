"""
Adaptive, training-free AIGC (AI-Generated Content) image detection.

Implements the three-stage early-exit forensic pipeline described in the
research methodology:

  Stage 1 -> global frequency-perturbation sensitivity   (cheap)
  Stage 2 -> spatial / cropping representation consistency (medium)
  Stage 3 -> multi-patch local frequency forensic analysis (expensive)

with a rule-based, two-threshold early-exit policy so that only
ambiguous images pay for the deeper stages. No detector is trained;
only two scalar thresholds per stage are calibrated on a small held-out
calibration split (never on the test set).
"""

from .encoder import FrozenEncoder, DummyEncoder, load_encoder
from .perturbations import frequency_perturbation, crop_views, patch_split
from .stages import Stage1Frequency, Stage2Spatial, Stage3Patch
from .calibration import calibrate_two_thresholds
from .pipeline import AdaptiveDetector, PipelineConfig
from .evaluate import run_ablation, compute_auc, average_cost

__all__ = [
    "FrozenEncoder", "DummyEncoder", "load_encoder",
    "frequency_perturbation", "crop_views", "patch_split",
    "Stage1Frequency", "Stage2Spatial", "Stage3Patch",
    "calibrate_two_thresholds",
    "AdaptiveDetector", "PipelineConfig",
    "run_ablation", "compute_auc", "average_cost",
]
