"""
The three forensic stages (Sections 10, 13, 15).

Every stage returns a scalar "detectability" score in roughly [0, 2] where
higher = more likely AI-generated (1 - cosine similarity based), plus the
number of encoder forward passes it consumed (its computational cost).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

try:
    from .perturbations import frequency_perturbation, crop_views, patch_split
except ImportError:
    from perturbations import frequency_perturbation, crop_views, patch_split


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


class Stage1Frequency:
    """
    D_f = 1 - cos(f(I), f(I + delta_f))
    Cost: 2 encoder forward passes (original + perturbed).
    """

    def __init__(self, encoder, strength: float = 0.08, high_freq_radius_frac: float = 0.6):
        self.encoder = encoder
        self.strength = strength
        self.high_freq_radius_frac = high_freq_radius_frac
        self.cost = 2 * encoder.unit_cost

    @torch.no_grad()
    def __call__(self, img: torch.Tensor):
        img_f = frequency_perturbation(img, self.strength, self.high_freq_radius_frac)
        z = self.encoder(img)
        z_f = self.encoder(img_f)
        d_f = 1.0 - _cosine(z, z_f)
        return d_f, z  # return z so Stage 2/3 can reuse the global embedding


class Stage2Spatial:
    """
    D_c = 1 - mean_i cos(z, z_i) over k spatial (crop/resize) views.
    Cost: k additional encoder forward passes (reuses the global z from Stage 1).
    """

    def __init__(self, encoder, k: int = 4, min_scale: float = 0.5, max_scale: float = 0.9):
        self.encoder = encoder
        self.k = k
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.cost = k * encoder.unit_cost

    @torch.no_grad()
    def __call__(self, img: torch.Tensor, z_global: torch.Tensor = None, seed: int = None):
        if z_global is None:
            z_global = self.encoder(img)
        views = crop_views(img, self.k, self.min_scale, self.max_scale, seed=seed)
        sims = []
        for v in views:
            z_v = self.encoder(v)
            sims.append(_cosine(z_global, z_v))
        d_c = 1.0 - torch.stack(sims, dim=0).mean(dim=0)
        return d_c


class Stage3Patch:
    """
    Multi-patch local frequency-forensic analysis (Section 15):
    for each of n patches, compute its own Stage-1-style frequency
    sensitivity score, then robustly aggregate (trimmed mean / median)
    across patches to reduce sensitivity to outlier patches (e.g. flat sky
    regions with little forensic signal).
    Cost: 2 * n_patches encoder forward passes (each patch needs its own
    original + perturbed embedding).
    """

    def __init__(self, encoder, n_patches: int = 9, strength: float = 0.10,
                 high_freq_radius_frac: float = 0.6, aggregate: str = "trimmed_mean",
                 trim_frac: float = 0.2):
        self.encoder = encoder
        self.n_patches = n_patches
        self.strength = strength
        self.high_freq_radius_frac = high_freq_radius_frac
        self.aggregate = aggregate
        self.trim_frac = trim_frac
        self.cost = 2 * n_patches * encoder.unit_cost

    @torch.no_grad()
    def __call__(self, img: torch.Tensor):
        patches = patch_split(img, self.n_patches)
        patch_scores = []
        for p in patches:
            p_f = frequency_perturbation(p, self.strength, self.high_freq_radius_frac)
            z_p = self.encoder(p)
            z_pf = self.encoder(p_f)
            patch_scores.append(1.0 - _cosine(z_p, z_pf))
        scores = torch.stack(patch_scores, dim=0)  # (n_patches, B)

        if self.aggregate == "median":
            agg = scores.median(dim=0).values
        elif self.aggregate == "trimmed_mean":
            n = scores.shape[0]
            k = max(1, int(n * self.trim_frac))
            sorted_scores, _ = torch.sort(scores, dim=0)
            trimmed = sorted_scores[k: n - k] if n - 2 * k > 0 else sorted_scores
            agg = trimmed.mean(dim=0)
        else:  # plain mean
            agg = scores.mean(dim=0)
        return agg
