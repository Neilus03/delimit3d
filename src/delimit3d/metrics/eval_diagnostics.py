"""Diagnostic utilities for ranked instance AP evaluation.

This module stays deliberately model-free.  It consumes the compact official
AP records produced by :mod:`delimit3d.metrics.official_instance_ap`, plus
scene-local prediction masks only for duplicate/NMS diagnostics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from delimit3d.metrics.official_instance_ap import (
    evaluate_official_and_oracle_ap,
    evaluate_official_instance_ap,
    merge_ap_record_sets,
)


_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
_DUP_THRESHOLDS = (0.25, 0.50, 0.75)
_DUP_SCOPES = (None, 100, 50, 25)


def _metric_key(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _scope_key(k: int | None) -> str:
    return "all" if k is None else f"top{k}"


def _as_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Iterable[Any]) -> float | None:
    clean = [_as_float_or_none(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _median(values: Iterable[Any]) -> float | None:
    clean = [_as_float_or_none(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return None
    return float(np.median(clean))


def _percentile_dict(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p:02d}": None for p in _PERCENTILES}
    return {
        f"p{p:02d}": float(np.percentile(arr, p))
        for p in _PERCENTILES
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation that avoids a scipy dependency for tests."""
    arr = np.asarray(values, dtype=np.float64)
    ranks = np.empty(arr.size, dtype=np.float64)
    if arr.size == 0:
        return ranks
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return None
    if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
        keep = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[keep]
        y_arr = y_arr[keep]
    if x_arr.size < 2:
        return None
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    return corr if math.isfinite(corr) else None


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return None
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 2:
        return None
    return safe_pearson(_rankdata(x_arr), _rankdata(y_arr))


def safe_kendall_tau(x: Iterable[float], y: Iterable[float]) -> float | None:
    try:
        from scipy.stats import kendalltau
    except Exception:
        return None
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return None
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(keep.sum()) < 2:
        return None
    value = kendalltau(x_arr[keep], y_arr[keep]).statistic
    return _as_float_or_none(value)


def filter_records_topk_by_scene(
    records: dict[str, Any],
    *,
    k: int,
    score_key: str = "score",
) -> dict[str, Any]:
    predictions = list(records.get("predictions", []) or [])
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        by_scene[str(pred.get("scene_id", ""))].append(pred)

    kept: list[dict[str, Any]] = []
    for scene_preds in by_scene.values():
        ordered = sorted(
            scene_preds,
            key=lambda p: (
                -float(p.get(score_key, p.get("score", 0.0))),
                int(p.get("pred_id", 0)),
            ),
        )
        kept.extend(ordered[: max(int(k), 0)])

    return {
        "predictions": kept,
        "ground_truths": list(records.get("ground_truths", []) or []),
        "num_predictions": len(kept),
        "total_gt_instances": len(records.get("ground_truths", []) or []),
    }


def filter_records_by_scene_pred_ids(
    records: dict[str, Any],
    keep_by_scene: dict[str, set[int]],
) -> dict[str, Any]:
    kept = [
        pred for pred in list(records.get("predictions", []) or [])
        if int(pred.get("pred_id", -1)) in keep_by_scene.get(str(pred.get("scene_id", "")), set())
    ]
    return {
        "predictions": kept,
        "ground_truths": list(records.get("ground_truths", []) or []),
        "num_predictions": len(kept),
        "total_gt_instances": len(records.get("ground_truths", []) or []),
    }


def matched_recall_from_records(
    records: dict[str, Any],
    *,
    thresholds: tuple[float, ...] = (0.25, 0.50),
) -> dict[str, float | int]:
    gt_keys = {
        (str(gt["scene_id"]), int(gt["gt_id"]))
        for gt in list(records.get("ground_truths", []) or [])
    }
    matched = {float(t): set() for t in thresholds}
    for pred in list(records.get("predictions", []) or []):
        scene = str(pred.get("scene_id", ""))
        for gt_id_raw, iou_raw in zip(
            pred.get("candidate_gt_ids", []) or [],
            pred.get("candidate_ious", []) or [],
        ):
            key = (scene, int(gt_id_raw))
            if key not in gt_keys:
                continue
            iou = float(iou_raw)
            for threshold in thresholds:
                if iou >= float(threshold):
                    matched[float(threshold)].add(key)

    total_gt = len(gt_keys)
    out: dict[str, float | int] = {"total_gt_instances": total_gt}
    for threshold in thresholds:
        suffix = int(round(float(threshold) * 100))
        out[f"recall{suffix}"] = float(len(matched[float(threshold)]) / max(total_gt, 1))
    return out


def compact_ap_recall_metrics(
    records: dict[str, Any],
    *,
    score_key: str = "score",
) -> dict[str, Any]:
    metrics = evaluate_official_instance_ap(records, score_key=score_key)
    recall = matched_recall_from_records(records)
    num_gt = int(metrics.get("total_gt_instances", 0) or 0)
    num_predictions = int(metrics.get("num_predictions", 0) or 0)
    return {
        "AP": _as_float_or_none(metrics.get("AP")),
        "AP50": _as_float_or_none(metrics.get("AP50")),
        "AP25": _as_float_or_none(metrics.get("AP25")),
        "recall50": _as_float_or_none(recall.get("recall50")),
        "recall25": _as_float_or_none(recall.get("recall25")),
        "num_predictions": num_predictions,
        "num_gt": num_gt,
        "predictions_per_gt": float(num_predictions / max(num_gt, 1)),
        "score_key": score_key,
    }


def topk_diagnostics(
    records: dict[str, Any],
    *,
    topk_values: Iterable[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_score: dict[str, Any] = {}
    by_oracle: dict[str, Any] = {}
    for raw_k in topk_values:
        k = int(raw_k)
        score_records = filter_records_topk_by_scene(records, k=k, score_key="score")
        oracle_records = filter_records_topk_by_scene(records, k=k, score_key="oracle_score")
        by_score[str(k)] = compact_ap_recall_metrics(score_records, score_key="score")
        by_oracle[str(k)] = compact_ap_recall_metrics(oracle_records, score_key="oracle_score")
    return by_score, by_oracle


def _prediction_arrays(records: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    predictions = list(records.get("predictions", []) or [])
    scores = np.asarray([float(p.get("score", 0.0)) for p in predictions], dtype=np.float64)
    oracle = np.asarray([float(p.get("oracle_score", 0.0)) for p in predictions], dtype=np.float64)
    scenes = [str(p.get("scene_id", "")) for p in predictions]
    return scores, oracle, scenes


def score_iou_diagnostics(
    records: dict[str, Any],
    *,
    rank_topks: tuple[int, ...] = (10, 25, 50, 100),
) -> dict[str, Any]:
    scores, oracle, scenes = _prediction_arrays(records)
    out: dict[str, Any] = {
        "num_predictions": int(scores.size),
        "score_min": _as_float_or_none(np.min(scores)) if scores.size else None,
        "score_max": _as_float_or_none(np.max(scores)) if scores.size else None,
        "score_mean": _as_float_or_none(np.mean(scores)) if scores.size else None,
        "score_median": _as_float_or_none(np.median(scores)) if scores.size else None,
        "oracle_iou_min": _as_float_or_none(np.min(oracle)) if oracle.size else None,
        "oracle_iou_max": _as_float_or_none(np.max(oracle)) if oracle.size else None,
        "oracle_iou_mean": _as_float_or_none(np.mean(oracle)) if oracle.size else None,
        "oracle_iou_median": _as_float_or_none(np.median(oracle)) if oracle.size else None,
        "score_percentiles": _percentile_dict(scores),
        "oracle_iou_percentiles": _percentile_dict(oracle),
        "fraction_oracle_iou_ge_25": float(np.mean(oracle >= 0.25)) if oracle.size else None,
        "fraction_oracle_iou_ge_50": float(np.mean(oracle >= 0.50)) if oracle.size else None,
        "fraction_oracle_iou_ge_75": float(np.mean(oracle >= 0.75)) if oracle.size else None,
        "fraction_oracle_iou_eq_0": float(np.mean(np.isclose(oracle, 0.0))) if oracle.size else None,
        "mean_score_iou_ge_50": _mean(scores[oracle >= 0.50]) if oracle.size else None,
        "mean_score_iou_lt_25": _mean(scores[oracle < 0.25]) if oracle.size else None,
        "mean_score_iou_eq_0": _mean(scores[np.isclose(oracle, 0.0)]) if oracle.size else None,
        "pearson": safe_pearson(scores, oracle),
        "spearman": safe_spearman(scores, oracle),
        "kendall_tau": safe_kendall_tau(scores, oracle),
    }

    by_scene_idx: dict[str, list[int]] = defaultdict(list)
    for idx, scene_id in enumerate(scenes):
        by_scene_idx[scene_id].append(idx)
    pearsons: list[float] = []
    spearmans: list[float] = []
    rank_quality: dict[str, Any] = {
        "mean_oracle_iou_all_predictions": _as_float_or_none(np.mean(oracle)) if oracle.size else None,
    }
    topk_by_score: dict[int, list[float]] = {k: [] for k in rank_topks}
    topk_by_oracle: dict[int, list[float]] = {k: [] for k in rank_topks}

    for indices in by_scene_idx.values():
        idx_arr = np.asarray(indices, dtype=np.int64)
        s = scores[idx_arr]
        o = oracle[idx_arr]
        pear = safe_pearson(s, o)
        spear = safe_spearman(s, o)
        if pear is not None:
            pearsons.append(pear)
        if spear is not None:
            spearmans.append(spear)
        for k in rank_topks:
            if o.size == 0:
                continue
            score_order = np.argsort(-s, kind="mergesort")[: min(k, o.size)]
            oracle_order = np.argsort(-o, kind="mergesort")[: min(k, o.size)]
            topk_by_score[k].append(float(np.mean(o[score_order])))
            topk_by_oracle[k].append(float(np.mean(o[oracle_order])))

    for k in rank_topks:
        rank_quality[f"mean_oracle_iou_top{k}_by_score_per_scene"] = _mean(topk_by_score[k])
        rank_quality[f"mean_oracle_iou_top{k}_by_oracle_per_scene"] = _mean(topk_by_oracle[k])

    out["per_scene"] = {
        "pearson_mean": _mean(pearsons),
        "pearson_median": _median(pearsons),
        "spearman_mean": _mean(spearmans),
        "spearman_median": _median(spearmans),
        "num_scenes_with_valid_correlation": len(spearmans),
    }
    out["rank_quality"] = rank_quality
    return out


def score_bin_calibration(
    records: dict[str, Any],
    *,
    bin_width: float = 0.05,
) -> list[dict[str, Any]]:
    scores, oracle, _ = _prediction_arrays(records)
    bins: list[dict[str, Any]] = []
    edges = np.round(np.arange(0.0, 1.0 + bin_width, bin_width), 10)
    for low, high in zip(edges[:-1], edges[1:]):
        if math.isclose(float(high), 1.0):
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        vals = oracle[mask]
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "num_predictions": int(vals.size),
                "mean_oracle_iou": _as_float_or_none(np.mean(vals)) if vals.size else None,
                "median_oracle_iou": _as_float_or_none(np.median(vals)) if vals.size else None,
                "fraction_oracle_iou_ge_25": float(np.mean(vals >= 0.25)) if vals.size else None,
                "fraction_oracle_iou_ge_50": float(np.mean(vals >= 0.50)) if vals.size else None,
                "fraction_oracle_iou_ge_75": float(np.mean(vals >= 0.75)) if vals.size else None,
            }
        )
    return bins


def pairwise_mask_iou(proposals: list[np.ndarray]) -> np.ndarray:
    if not proposals:
        return np.zeros((0, 0), dtype=np.float32)
    masks = np.asarray([np.asarray(mask, dtype=bool) for mask in proposals], dtype=bool)
    sizes = masks.sum(axis=1).astype(np.float64)
    if masks.ndim != 2:
        raise ValueError(f"Expected a 2D proposal mask array, got shape {masks.shape}")

    try:
        from scipy import sparse

        sparse_masks = sparse.csr_matrix(masks.astype(np.uint8, copy=False))
        intersection = (sparse_masks @ sparse_masks.T).toarray().astype(np.float64, copy=False)
    except Exception:
        dense = masks.astype(np.float32, copy=False)
        intersection = (dense @ dense.T).astype(np.float64, copy=False)

    union = sizes[:, None] + sizes[None, :] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    np.fill_diagonal(iou, 1.0)
    return iou.astype(np.float32, copy=False)


def _duplicate_components(adj: np.ndarray) -> tuple[int, int]:
    n = int(adj.shape[0])
    seen = np.zeros(n, dtype=bool)
    groups = 0
    largest = 0
    for start in range(n):
        if seen[start] or not bool(adj[start].any()):
            continue
        groups += 1
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in np.flatnonzero(adj[node]):
                if not seen[int(nxt)]:
                    seen[int(nxt)] = True
                    stack.append(int(nxt))
        largest = max(largest, size)
    return groups, largest


def duplicate_summary_from_iou(
    iou: np.ndarray,
    *,
    thresholds: tuple[float, ...] = _DUP_THRESHOLDS,
) -> dict[str, Any]:
    iou_arr = np.asarray(iou, dtype=np.float32)
    n = int(iou_arr.shape[0])
    out: dict[str, Any] = {}
    for threshold in thresholds:
        if n <= 1:
            out[_metric_key(threshold)] = {
                "num_predictions": n,
                "duplicate_pairs": 0,
                "duplicate_pairs_per_prediction": 0.0,
                "num_predictions_involved": 0,
                "fraction_predictions_involved": 0.0,
                "num_duplicate_groups": 0,
                "largest_duplicate_group_size": 0,
            }
            continue
        adj = iou_arr >= float(threshold)
        np.fill_diagonal(adj, False)
        pair_count = int(np.triu(adj, k=1).sum())
        involved = adj.any(axis=1)
        groups, largest = _duplicate_components(adj)
        out[_metric_key(threshold)] = {
            "num_predictions": n,
            "duplicate_pairs": pair_count,
            "duplicate_pairs_per_prediction": float(pair_count / max(n, 1)),
            "num_predictions_involved": int(involved.sum()),
            "fraction_predictions_involved": float(involved.mean()) if n else 0.0,
            "num_duplicate_groups": int(groups),
            "largest_duplicate_group_size": int(largest),
        }
    return out


def mask_nms_from_iou(
    iou: np.ndarray,
    scores: Iterable[float],
    *,
    threshold: float,
) -> list[int]:
    iou_arr = np.asarray(iou, dtype=np.float32)
    score_arr = np.asarray(list(scores), dtype=np.float64)
    if score_arr.size == 0:
        return []
    order = sorted(range(score_arr.size), key=lambda i: (-float(score_arr[i]), int(i)))
    keep: list[int] = []
    suppressed = np.zeros(score_arr.size, dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        overlaps = iou_arr[idx] > float(threshold)
        suppressed |= overlaps
        suppressed[idx] = False
    return keep


def build_scene_mask_diagnostics(
    records: dict[str, Any],
    proposals: list[np.ndarray],
    *,
    nms_thresholds: Iterable[float],
    duplicate_thresholds: tuple[float, ...] = _DUP_THRESHOLDS,
    duplicate_scopes: tuple[int | None, ...] = _DUP_SCOPES,
    max_duplicate_predictions: int = 250,
) -> dict[str, Any]:
    predictions = list(records.get("predictions", []) or [])
    pred_ids = [int(pred.get("pred_id", idx)) for idx, pred in enumerate(predictions)]
    record_proposals = [
        np.asarray(proposals[pred_id], dtype=bool)
        for pred_id in pred_ids
        if 0 <= pred_id < len(proposals)
    ]
    if len(record_proposals) != len(predictions):
        raise ValueError("Could not align compact prediction records to proposal masks")

    scores = np.asarray([float(p.get("score", 0.0)) for p in predictions], dtype=np.float64)
    oracle = np.asarray([float(p.get("oracle_score", 0.0)) for p in predictions], dtype=np.float64)
    iou = pairwise_mask_iou(record_proposals)
    score_order = np.argsort(-scores, kind="mergesort")

    duplicates: dict[str, Any] = {}
    for scope in duplicate_scopes:
        if scope is None:
            limit = min(len(score_order), int(max_duplicate_predictions))
        else:
            limit = min(len(score_order), int(scope))
        subset = score_order[:limit]
        subset_iou = iou[np.ix_(subset, subset)] if subset.size else np.zeros((0, 0), dtype=np.float32)
        duplicates[_scope_key(scope)] = duplicate_summary_from_iou(
            subset_iou,
            thresholds=duplicate_thresholds,
        )

    nms_by_score: dict[str, list[int]] = {}
    nms_by_oracle: dict[str, list[int]] = {}
    for threshold in nms_thresholds:
        key = _metric_key(float(threshold))
        keep_score_idx = mask_nms_from_iou(iou, scores, threshold=float(threshold))
        keep_oracle_idx = mask_nms_from_iou(iou, oracle, threshold=float(threshold))
        nms_by_score[key] = [pred_ids[i] for i in keep_score_idx]
        nms_by_oracle[key] = [pred_ids[i] for i in keep_oracle_idx]

    return {
        "duplicates": duplicates,
        "nms_keep_pred_ids_by_score": nms_by_score,
        "nms_keep_pred_ids_by_oracle": nms_by_oracle,
    }


def gt_assignment_duplicate_stats(records: dict[str, Any]) -> dict[str, Any]:
    gt_keys = [
        (str(gt["scene_id"]), int(gt["gt_id"]))
        for gt in list(records.get("ground_truths", []) or [])
    ]
    counts_by_thr = {
        0.25: {key: 0 for key in gt_keys},
        0.50: {key: 0 for key in gt_keys},
    }
    for pred in list(records.get("predictions", []) or []):
        scene = str(pred.get("scene_id", ""))
        for gt_id_raw, iou_raw in zip(
            pred.get("candidate_gt_ids", []) or [],
            pred.get("candidate_ious", []) or [],
        ):
            key = (scene, int(gt_id_raw))
            if key not in counts_by_thr[0.25]:
                continue
            iou = float(iou_raw)
            if iou >= 0.25:
                counts_by_thr[0.25][key] += 1
            if iou >= 0.50:
                counts_by_thr[0.50][key] += 1

    vals25 = np.asarray(list(counts_by_thr[0.25].values()), dtype=np.float64)
    vals50 = np.asarray(list(counts_by_thr[0.50].values()), dtype=np.float64)
    return {
        "num_gt": int(len(gt_keys)),
        "mean_predictions_per_gt_iou_ge_25": _as_float_or_none(np.mean(vals25)) if vals25.size else None,
        "mean_predictions_per_gt_iou_ge_50": _as_float_or_none(np.mean(vals50)) if vals50.size else None,
        "median_predictions_per_gt_iou_ge_25": _as_float_or_none(np.median(vals25)) if vals25.size else None,
        "median_predictions_per_gt_iou_ge_50": _as_float_or_none(np.median(vals50)) if vals50.size else None,
        "fraction_gt_more_than_1_prediction_iou_ge_50": float(np.mean(vals50 > 1)) if vals50.size else None,
        "fraction_gt_more_than_5_predictions_iou_ge_25": float(np.mean(vals25 > 5)) if vals25.size else None,
        "max_predictions_assigned_to_single_gt_iou_ge_50": int(np.max(vals50)) if vals50.size else 0,
    }


def _aggregate_duplicate_summaries(scene_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scope in ("all", "top100", "top50", "top25"):
        out[scope] = {}
        for threshold in (_metric_key(t) for t in _DUP_THRESHOLDS):
            rows: list[dict[str, Any]] = []
            for scene in scene_outputs:
                row = (
                    scene.get("mask_diagnostics", {})
                    .get("duplicates", {})
                    .get(scope, {})
                    .get(threshold)
                )
                if isinstance(row, dict):
                    rows.append(row)
            total_preds = int(sum(int(r.get("num_predictions", 0) or 0) for r in rows))
            total_pairs = int(sum(int(r.get("duplicate_pairs", 0) or 0) for r in rows))
            total_involved = int(sum(int(r.get("num_predictions_involved", 0) or 0) for r in rows))
            largest_values = [int(r.get("largest_duplicate_group_size", 0) or 0) for r in rows]
            out[scope][threshold] = {
                "num_scenes": len(rows),
                "num_predictions": total_preds,
                "duplicate_pairs": total_pairs,
                "duplicate_pairs_per_prediction": float(total_pairs / max(total_preds, 1)),
                "num_predictions_involved": total_involved,
                "fraction_predictions_involved": float(total_involved / max(total_preds, 1)),
                "fraction_predictions_involved_mean_per_scene": _mean(
                    r.get("fraction_predictions_involved") for r in rows
                ),
                "fraction_predictions_involved_median_per_scene": _median(
                    r.get("fraction_predictions_involved") for r in rows
                ),
                "num_duplicate_groups_mean_per_scene": _mean(r.get("num_duplicate_groups") for r in rows),
                "largest_duplicate_group_size_mean_per_scene": _mean(largest_values),
                "largest_duplicate_group_size_max": int(max(largest_values)) if largest_values else 0,
            }
    return out


def _scene_keep_map(scene_outputs: list[dict[str, Any]], kind: str, threshold: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    keep_key = f"nms_keep_pred_ids_by_{kind}"
    for scene in scene_outputs:
        scene_id = str(scene["scene_id"])
        keep_ids = scene.get("mask_diagnostics", {}).get(keep_key, {}).get(threshold, [])
        out[scene_id] = {int(x) for x in keep_ids}
    return out


def _summarize_nms(
    records: dict[str, Any],
    scene_outputs: list[dict[str, Any]],
    *,
    nms_thresholds: Iterable[float],
    kind: str,
) -> dict[str, Any]:
    score_key = "score" if kind == "score" else "oracle_score"
    out: dict[str, Any] = {}
    for threshold in nms_thresholds:
        key = _metric_key(float(threshold))
        filtered = filter_records_by_scene_pred_ids(records, _scene_keep_map(scene_outputs, kind, key))
        out[key] = compact_ap_recall_metrics(filtered, score_key=score_key)
    return out


def _per_scene_records(records: dict[str, Any], scene_id: str) -> dict[str, Any]:
    return {
        "predictions": [
            pred for pred in list(records.get("predictions", []) or [])
            if str(pred.get("scene_id", "")) == scene_id
        ],
        "ground_truths": [
            gt for gt in list(records.get("ground_truths", []) or [])
            if str(gt.get("scene_id", "")) == scene_id
        ],
    }


def _diagnosis(
    *,
    baseline: dict[str, Any],
    score_iou: dict[str, Any],
    duplicates: dict[str, Any],
    nms_by_score: dict[str, Any],
    nms_by_oracle: dict[str, Any],
) -> dict[str, Any]:
    ap50 = float(baseline.get("official_AP50") or 0.0)
    oracle_ap50 = float(baseline.get("oracle_AP50") or 0.0)
    spearman = score_iou.get("spearman")
    score_gap = oracle_ap50 - ap50
    score_bottleneck = bool(score_gap >= 0.05 or (spearman is not None and float(spearman) < 0.30))

    best_nms_ap50 = max(
        (float(v.get("AP50") or 0.0) for v in nms_by_score.values()),
        default=0.0,
    )
    top50_iou50 = (
        duplicates.get("top50", {})
        .get("0.5", {})
        .get("fraction_predictions_involved")
    )
    duplicate_bottleneck = bool(
        (best_nms_ap50 - ap50) >= 0.02
        or (top50_iou50 is not None and float(top50_iou50) >= 0.30)
    )

    best_oracle_nms_ap50 = max(
        (float(v.get("AP50") or 0.0) for v in nms_by_oracle.values()),
        default=0.0,
    )
    mask_bottleneck = bool(oracle_ap50 < 0.30 or best_oracle_nms_ap50 < 0.30)

    if score_bottleneck and duplicate_bottleneck:
        next_step = "IoU-aware score target + duplicate suppression"
    elif score_bottleneck:
        next_step = "IoU-aware score target"
    elif duplicate_bottleneck:
        next_step = "duplicate suppression / query competition"
    elif mask_bottleneck:
        next_step = "mask refinement architecture after score/duplicate diagnostics"
    else:
        next_step = "continue diagnostics before changing training"

    evidence = [
        f"AP50={ap50:.4f}, oracle_AP50={oracle_ap50:.4f}, oracle_gap={score_gap:.4f}",
        f"Spearman(score, oracle_iou)={spearman if spearman is not None else 'null'}",
        f"Best score-NMS AP50={best_nms_ap50:.4f}",
        f"Best oracle-NMS AP50={best_oracle_nms_ap50:.4f}",
        f"Duplicate fraction top50 IoU50={top50_iou50 if top50_iou50 is not None else 'null'}",
    ]
    return {
        "score_calibration_bottleneck": score_bottleneck,
        "duplicate_bottleneck": duplicate_bottleneck,
        "mask_quality_bottleneck": mask_bottleneck,
        "recommended_next_step": next_step,
        "evidence": evidence,
    }


def build_diagnostic_report(
    *,
    scene_outputs: list[dict[str, Any]],
    checkpoint: str,
    config: str,
    benchmark: str,
    granularity: str,
    nms_thresholds: Iterable[float],
    topk_values: Iterable[int],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_sets = [scene["records"] for scene in scene_outputs if isinstance(scene.get("records"), dict)]
    records = merge_ap_record_sets(record_sets)
    official = evaluate_official_and_oracle_ap(records)

    total_gt = int(official.get("total_gt_instances", 0) or 0)
    num_predictions = int(official.get("num_predictions", 0) or 0)
    legacy25 = _mean(scene.get("legacy", {}).get("legacy_matched_recall25") for scene in scene_outputs)
    legacy50 = _mean(scene.get("legacy", {}).get("legacy_matched_recall50") for scene in scene_outputs)
    matched_iou = _mean(scene.get("legacy", {}).get("matched_mean_iou") for scene in scene_outputs)
    removed_min_points = int(sum(
        int(scene.get("proposal_stats", {}).get("num_min_points_removed", 0) or 0)
        for scene in scene_outputs
    ))
    num_queries = int(sum(
        int(scene.get("proposal_stats", {}).get("num_queries", 0) or 0)
        for scene in scene_outputs
    ))

    scopes = {str(scene.get("eval_scope", "full_scene")) for scene in scene_outputs}
    eval_scope = next(iter(scopes)) if len(scopes) == 1 else ("mixed" if scopes else "full_scene")
    warnings: list[str] = []
    if eval_scope != "full_scene":
        warnings.append("Evaluation scope is not full_scene; metrics are not paper-comparable.")

    baseline = {
        "official_AP": _as_float_or_none(official.get("AP")),
        "official_AP50": _as_float_or_none(official.get("AP50")),
        "official_AP25": _as_float_or_none(official.get("AP25")),
        "oracle_AP": _as_float_or_none(official.get("oracle_AP")),
        "oracle_AP50": _as_float_or_none(official.get("oracle_AP50")),
        "oracle_AP25": _as_float_or_none(official.get("oracle_AP25")),
        "legacy_recall25": legacy25,
        "legacy_recall50": legacy50,
        "legacy_matched_recall25": legacy25,
        "legacy_matched_recall50": legacy50,
        "matched_mean_iou": matched_iou,
        "num_gt": total_gt,
        "total_gt_instances": total_gt,
        "num_predictions": num_predictions,
        "predictions_per_gt": float(num_predictions / max(total_gt, 1)),
        "num_queries_evaluated": num_queries,
        "num_predictions_removed_by_min_points": removed_min_points,
    }

    score_iou = score_iou_diagnostics(records)
    topk_by_score, topk_by_oracle = topk_diagnostics(records, topk_values=topk_values)
    duplicates = _aggregate_duplicate_summaries(scene_outputs)
    duplicates["gt_assignment"] = gt_assignment_duplicate_stats(records)
    nms_by_score = _summarize_nms(
        records, scene_outputs, nms_thresholds=nms_thresholds, kind="score"
    )
    nms_by_oracle = _summarize_nms(
        records, scene_outputs, nms_thresholds=nms_thresholds, kind="oracle"
    )

    per_scene: dict[str, Any] = {}
    for scene in scene_outputs:
        scene_id = str(scene["scene_id"])
        scene_records = _per_scene_records(records, scene_id)
        scene_official = evaluate_official_and_oracle_ap(scene_records, thresholds=[0.5, 0.25])
        scene_score, scene_oracle, _ = _prediction_arrays(scene_records)
        corr = safe_spearman(scene_score, scene_oracle)
        top25_score = filter_records_topk_by_scene(scene_records, k=25, score_key="score")
        top25_oracle = filter_records_topk_by_scene(scene_records, k=25, score_key="oracle_score")
        top25_score_recall = matched_recall_from_records(top25_score)
        top25_oracle_recall = matched_recall_from_records(top25_oracle)
        dup_top50_iou50 = (
            scene.get("mask_diagnostics", {})
            .get("duplicates", {})
            .get("top50", {})
            .get("0.5", {})
        )
        num_gt_scene = len(scene_records.get("ground_truths", []) or [])
        num_pred_scene = len(scene_records.get("predictions", []) or [])
        per_scene[scene_id] = {
            "scene_id": scene_id,
            "eval_scope": scene.get("eval_scope", "full_scene"),
            "num_gt": num_gt_scene,
            "num_predictions": num_pred_scene,
            "predictions_per_gt": float(num_pred_scene / max(num_gt_scene, 1)),
            "legacy_recall25": _as_float_or_none(scene.get("legacy", {}).get("legacy_matched_recall25")),
            "legacy_recall50": _as_float_or_none(scene.get("legacy", {}).get("legacy_matched_recall50")),
            "per_scene_AP50_diagnostic": _as_float_or_none(scene_official.get("AP50")),
            "oracle_AP50_diagnostic": _as_float_or_none(scene_official.get("oracle_AP50")),
            "mean_score": _as_float_or_none(np.mean(scene_score)) if scene_score.size else None,
            "mean_oracle_iou": _as_float_or_none(np.mean(scene_oracle)) if scene_oracle.size else None,
            "score_iou_spearman": corr,
            "duplicate_fraction_top50_iou50": _as_float_or_none(
                dup_top50_iou50.get("fraction_predictions_involved") if isinstance(dup_top50_iou50, dict) else None
            ),
            "largest_duplicate_group_top50_iou50": int(
                dup_top50_iou50.get("largest_duplicate_group_size", 0)
                if isinstance(dup_top50_iou50, dict)
                else 0
            ),
            "top25_recall50_by_score": _as_float_or_none(top25_score_recall.get("recall50")),
            "top25_recall50_by_oracle": _as_float_or_none(top25_oracle_recall.get("recall50")),
        }

    report = {
        "checkpoint": checkpoint,
        "config": config,
        "benchmark": benchmark,
        "granularity": granularity,
        "eval_scope": eval_scope,
        "settings": settings or {},
        "warnings": warnings,
        "baseline": baseline,
        "oracle": {
            "AP": baseline["oracle_AP"],
            "AP50": baseline["oracle_AP50"],
            "AP25": baseline["oracle_AP25"],
        },
        "score_iou": score_iou,
        "topk_by_score": topk_by_score,
        "topk_by_oracle": topk_by_oracle,
        "duplicates": duplicates,
        "nms_by_score": nms_by_score,
        "nms_by_oracle": nms_by_oracle,
        "score_bins": score_bin_calibration(records),
        "per_scene": per_scene,
    }
    report["diagnosis"] = _diagnosis(
        baseline=baseline,
        score_iou=score_iou,
        duplicates=duplicates,
        nms_by_score=nms_by_score,
        nms_by_oracle=nms_by_oracle,
    )
    return report


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
