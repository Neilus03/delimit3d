from __future__ import annotations

import numpy as np

from delimit3d.evaluation.prompt_metrics import retrieval_metrics


def _average_precision(y: np.ndarray, s: np.ndarray) -> float:
    total = 0.0
    last_recall = 0.0
    for threshold in sorted(set(s), reverse=True):
        mask = s >= threshold
        tp = int((y & mask).sum())
        recall = tp / int(y.sum())
        total += (recall - last_recall) * tp / int(mask.sum())
        last_recall = recall
    return total


def test_retrieval_metrics_tied_scores_and_fixed_iou() -> None:
    rng = np.random.default_rng(7)
    for _ in range(100):
        target = rng.random(200) < 0.15
        target[0] = True
        score = rng.choice([-.3, .1, .7, .8, .95], 200)
        values = retrieval_metrics(target, score)
        assert abs(float(values["ap"]) - _average_precision(target, score)) < 1e-12
        fixed = score >= 0.7
        expected_iou = float((fixed & target).sum() / (fixed | target).sum())
        assert abs(float(values["iou_fixed"]) - expected_iou) < 1e-12
