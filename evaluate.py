"""
Evaluation utilities matching Sections 17-19 of the methodology:
  - AUC per pipeline variant
  - Average computational cost per image (C_avg = P1*C1 + P2*C2 + P3*C3)
  - The required ablation table: Stage1-only / +Stage2 / +Stage3 / Adaptive
"""
from __future__ import annotations
from collections import Counter
import numpy as np


def compute_auc(scores, labels) -> float:
    """
    scores: higher = more likely AI-generated (label 1).
    Uses a rank-based (Mann-Whitney U) AUC so no sklearn dependency is required,
    but will use sklearn.metrics.roc_auc_score if it's importable (identical result).
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # Mann-Whitney U statistic -> AUC
        ranks = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1
        rank_pos_sum = ranks[len(neg):].sum()
        n_pos, n_neg = len(pos), len(neg)
        auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)


def average_cost(stage_exits: list[int], stage_costs: dict[int, float]) -> float:
    """
    stage_exits: list of the exit-stage (1, 2, or 3) for every evaluated image.
    stage_costs: {1: C1, 2: C1+C2, 3: C1+C2+C3} -- CUMULATIVE cost of exiting
        at that stage (not the marginal per-stage cost).
    Returns C_avg = sum_s P_s * C_s.
    """
    n = len(stage_exits)
    if n == 0:
        return 0.0
    counts = Counter(stage_exits)
    return sum((counts.get(s, 0) / n) * c for s, c in stage_costs.items())


def run_ablation(detector, images, labels, seed: int = 0) -> dict:
    """
    Runs the four required variants (Section 19: A, B, C, D) over a list/
    iterable of (already-preprocessed) image tensors + integer labels
    (0=real, 1=AI), and returns AUC + average cost for each.

    `detector` must be an AdaptiveDetector (see pipeline.py).
    Returns a dict keyed by variant name, each value a dict with
    {"auc": float, "avg_cost": float, "exit_fractions": {...}}.
    """
    raw = [detector.raw_scores(img.unsqueeze(0), seed=seed) for img in images]
    labels = np.asarray(labels)

    results = {}

    # A: Stage-1 only
    scores_a = np.array([r["d_f"].item() for r in raw])
    results["A_stage1_only"] = dict(
        auc=compute_auc(scores_a, labels),
        avg_cost=detector.s1.cost,
        exit_fractions={1: 1.0},
    )

    # B: Stage 1 + Stage 2
    scores_b = np.array([r["combined_12"].item() for r in raw])
    results["B_stage1+2"] = dict(
        auc=compute_auc(scores_b, labels),
        avg_cost=detector.s1.cost + detector.s2.cost,
        exit_fractions={2: 1.0},
    )

    # C: Stage 1 + Stage 2 + Stage 3 (always run everything)
    scores_c = np.array([r["d_p"].item() for r in raw])
    results["C_stage1+2+3"] = dict(
        auc=compute_auc(scores_c, labels),
        avg_cost=detector.s1.cost + detector.s2.cost + detector.s3.cost,
        exit_fractions={3: 1.0},
    )

    # D: proposed adaptive early exit
    preds = [detector.predict(img.unsqueeze(0), seed=seed) for img in images]
    scores_d = []
    for p in preds:
        # use the finest-grained score available at whichever stage it exited
        scores_d.append(p.get("d_p", p.get("d_c", p.get("d_f"))))
    stage_exits = [p["stage_exit"] for p in preds]
    n = len(stage_exits)
    counts = Counter(stage_exits)
    exit_fractions = {s: counts.get(s, 0) / n for s in (1, 2, 3)}
    stage_costs = {
        1: detector.s1.cost,
        2: detector.s1.cost + detector.s2.cost,
        3: detector.s1.cost + detector.s2.cost + detector.s3.cost,
    }
    results["D_adaptive"] = dict(
        auc=compute_auc(np.array(scores_d), labels),
        avg_cost=average_cost(stage_exits, stage_costs),
        exit_fractions=exit_fractions,
    )
    return results


def print_ablation_table(results: dict, full_cost: float):
    """Pretty-prints the Section 18-style comparison table."""
    print(f"{'Method':<20}{'AUC (%)':>10}{'Avg. cost':>12}{'Cost %':>10}")
    for name, r in results.items():
        auc_pct = r["auc"] * 100
        cost_pct = 100 * r["avg_cost"] / full_cost if full_cost > 0 else float("nan")
        print(f"{name:<20}{auc_pct:>10.1f}{r['avg_cost']:>12.2f}{cost_pct:>9.1f}%")
