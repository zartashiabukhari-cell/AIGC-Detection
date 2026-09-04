"""
The adaptive detector itself (Sections 9, 16, 19): wires Stage 1 -> Stage 2
-> Stage 3 together with the rule-based, two-threshold-per-stage early-exit
policy, and exposes fixed-depth variants (A/B/C in Section 19) for the
required ablation study against the proposed adaptive policy (D).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

try:
    from .stages import Stage1Frequency, Stage2Spatial, Stage3Patch
    from .calibration import confidence_margin
except ImportError:
    from stages import Stage1Frequency, Stage2Spatial, Stage3Patch
    from calibration import confidence_margin


@dataclass
class StageThresholds:
    tau_R: float
    tau_F: float
    gamma: float = 1.0  # confidence margin required to exit early (Section 16)


@dataclass
class PipelineConfig:
    stage1: StageThresholds
    stage2: StageThresholds
    stage3_only_final_threshold: float = 0.5  # Stage 3 has no further stage to escalate to
    n_crop_views: int = 4
    n_patches: int = 9
    freq_strength_global: float = 0.08
    freq_strength_patch: float = 0.10


class AdaptiveDetector:
    """
    Label convention: 0 = REAL, 1 = AI-GENERATED.
    Every call to `predict` / `predict_fixed_depth` returns a dict with the
    final label, the stage at which the decision exited, and the total
    computational cost (in encoder-forward-pass units) spent on that image.
    """

    def __init__(self, encoder, config: PipelineConfig):
        self.encoder = encoder
        self.cfg = config
        self.s1 = Stage1Frequency(encoder, strength=config.freq_strength_global)
        self.s2 = Stage2Spatial(encoder, k=config.n_crop_views)
        self.s3 = Stage3Patch(encoder, n_patches=config.n_patches,
                               strength=config.freq_strength_patch)

    # ---------------------------------------------------------------- #
    # Proposed adaptive early-exit policy (variant "D" in Section 19)
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, img: torch.Tensor, seed: int = None) -> dict:
        cost = 0.0

        d_f, z_global = self.s1(img)
        cost += self.s1.cost
        c1 = confidence_margin(d_f, self.cfg.stage1.tau_R, self.cfg.stage1.tau_F)
        if float(c1) > self.cfg.stage1.gamma:
            label = 1 if float(d_f) > self.cfg.stage1.tau_F else 0
            return dict(label=label, stage_exit=1, cost=cost, d_f=float(d_f))

        d_c = self.s2(img, z_global=z_global, seed=seed)
        cost += self.s2.cost
        combined = 0.5 * (d_f + d_c)  # simple fused evidence, Section 14
        c2 = confidence_margin(combined, self.cfg.stage2.tau_R, self.cfg.stage2.tau_F)
        if float(c2) > self.cfg.stage2.gamma:
            label = 1 if float(combined) > self.cfg.stage2.tau_F else 0
            return dict(label=label, stage_exit=2, cost=cost, d_f=float(d_f), d_c=float(d_c))

        d_p = self.s3(img)
        cost += self.s3.cost
        label = 1 if float(d_p) > self.cfg.stage3_only_final_threshold else 0
        return dict(label=label, stage_exit=3, cost=cost,
                     d_f=float(d_f), d_c=float(d_c), d_p=float(d_p))

    # ---------------------------------------------------------------- #
    # Fixed-depth ablation variants (Section 19, variants A / B / C)
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def predict_fixed_depth(self, img: torch.Tensor, depth: int, seed: int = None) -> dict:
        """depth in {1, 2, 3}: always run exactly this many stages, never exit early."""
        assert depth in (1, 2, 3)
        cost = 0.0
        d_f, z_global = self.s1(img)
        cost += self.s1.cost
        score = d_f
        threshold = self.cfg.stage1.tau_F  # midpoint-style single threshold for a fixed depth

        if depth >= 2:
            d_c = self.s2(img, z_global=z_global, seed=seed)
            cost += self.s2.cost
            score = 0.5 * (d_f + d_c)
            threshold = self.cfg.stage2.tau_F

        if depth == 3:
            d_p = self.s3(img)
            cost += self.s3.cost
            score = d_p
            threshold = self.cfg.stage3_only_final_threshold

        label = 1 if float(score) > threshold else 0
        return dict(label=label, stage_exit=depth, cost=cost, score=float(score))

    @torch.no_grad()
    def raw_scores(self, img: torch.Tensor, seed: int = None) -> dict:
        """
        Run all three stages unconditionally and return raw scores, used only
        for building the calibration set (thresholds must be fit on scores,
        not on early-exit decisions) and for full ROC/AUC evaluation of each
        fixed-depth variant.
        """
        d_f, z_global = self.s1(img)
        d_c = self.s2(img, z_global=z_global, seed=seed)
        d_p = self.s3(img)
        combined_12 = 0.5 * (d_f + d_c)
        total_cost = self.s1.cost + self.s2.cost + self.s3.cost
        return dict(d_f=d_f, d_c=d_c, d_p=d_p, combined_12=combined_12, cost=total_cost)
