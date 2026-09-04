"""
Two-threshold calibration (Section 12).

Given calibration-split scores for real images and fake (AI-generated)
images at a given stage, choose (tau_R, tau_F) with tau_R < tau_F such that:
  D < tau_R          -> exit as REAL
  tau_R <= D <= tau_F -> UNCERTAIN, escalate to next stage
  D > tau_F          -> exit as AI

No gradient-based / learned classifier is fit here -- only two scalars are
chosen according to a predefined statistical criterion, then frozen.
"""
from __future__ import annotations
import numpy as np


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def calibrate_two_thresholds(scores_real, scores_fake, criterion: str = "youden",
                              target_fpr: float = 0.05, uncertain_band: float = 0.5):
    """
    scores_real / scores_fake: 1D array-like of stage scores D on a small,
        held-out CALIBRATION split only (e.g. 500 real + 500 fake, Section 12).
        These must never overlap with the final test set.

    criterion:
      - "youden": pick the single operating point maximizing TPR - FPR
                  (Youden's J), then open an UNCERTAIN band of width
                  `uncertain_band` (in score units) symmetric around it.
      - "target_fpr": pick tau_R as the score achieving `target_fpr` false
                  positive rate on the real calibration set, and tau_F as the
                  score achieving the mirror false-negative rate on fakes.
      - "eer": use the equal-error-rate point as the center of the band.

    Returns: (tau_R, tau_F)
    """
    real = np.sort(_to_numpy(scores_real).ravel())
    fake = np.sort(_to_numpy(scores_fake).ravel())
    all_scores = np.concatenate([real, fake])
    candidates = np.unique(all_scores)

    def fpr_at(t):  # fraction of REAL images incorrectly scored > t
        return (real > t).mean()

    def tpr_at(t):  # fraction of FAKE images correctly scored > t
        return (fake > t).mean()

    if criterion == "target_fpr":
        # tau_R: real images below this are confidently REAL (<= target_fpr escape as false AI)
        tau_R = np.quantile(real, 1 - target_fpr)
        # tau_F: fakes above this are confidently AI (mirror criterion on fakes)
        tau_F = np.quantile(fake, target_fpr)
        if tau_F < tau_R:
            tau_F = tau_R  # degenerate guard: ensure ordering, collapse uncertain band
        return float(tau_R), float(tau_F)

    if criterion in ("youden", "eer"):
        if criterion == "youden":
            j = np.array([tpr_at(t) - fpr_at(t) for t in candidates])
            center = candidates[np.argmax(j)]
        else:  # eer: point where FPR == 1-TPR (FNR)
            fpr = np.array([fpr_at(t) for t in candidates])
            fnr = np.array([1 - tpr_at(t) for t in candidates])
            center = candidates[np.argmin(np.abs(fpr - fnr))]

        half = uncertain_band / 2.0
        tau_R = float(center - half)
        tau_F = float(center + half)
        return tau_R, tau_F

    raise ValueError(f"unknown criterion: {criterion}")


def confidence_margin(d_score, tau_R: float, tau_F: float):
    """
    C = |D - tau| where tau is whichever boundary is nearer, normalized by
    half the uncertain-band width so C=1 sits exactly at the boundary and
    C>1 means confidently outside the uncertain band (Section 16).
    """
    center = (tau_R + tau_F) / 2.0
    half_band = max((tau_F - tau_R) / 2.0, 1e-8)
    result = abs(_to_numpy(d_score) - center) / half_band
    result = np.asarray(result).ravel()
    return float(result[0]) if result.size == 1 else result
