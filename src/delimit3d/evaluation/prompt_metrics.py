"""Model-free metrics for point-conditioned retrieval masks."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def retrieval_metrics(
    target: np.ndarray,
    score: np.ndarray,
    *,
    cosine_threshold: float = 0.70,
) -> dict[str, float | bool | int]:
    """Compute ranked AP and fixed-threshold retrieval metrics.

    ``target`` is a boolean candidate-point vector and ``score`` contains the
    query-to-candidate cosine similarity.  The query point itself should be
    removed by the caller.  No class or semantic information is used here.
    """

    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positives = int(target.sum())
    if positives <= 0 or target.shape != score.shape:
        raise ValueError("target must contain positives and match score shape")
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    sorted_target = target[order]
    ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(score) - 1
    ]
    true_positive = np.cumsum(sorted_target)[ends]
    retrieved = ends + 1
    gained = np.diff(np.r_[0, true_positive])
    ap = np.sum(gained / positives * true_positive / retrieved)
    iou_curve = true_positive / (positives + retrieved - true_positive)
    fixed = score >= float(cosine_threshold)
    fixed_tp = int(np.count_nonzero(fixed & target))
    fixed_count = int(fixed.sum())
    fixed_iou = fixed_tp / (positives + fixed_count - fixed_tp)
    precision = fixed_tp / max(fixed_count, 1)
    recall = fixed_tp / positives
    return {
        "ap": float(ap),
        "iou_fixed": float(fixed_iou),
        "precision_fixed": float(precision),
        "recall_fixed": float(recall),
        "oracle_iou": float(iou_curve.max()),
        "coverage50_fixed": bool(fixed_iou >= 0.5),
        "fixed_predicted_points": fixed_count,
    }


__all__ = ["retrieval_metrics"]
