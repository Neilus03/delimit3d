"""Dataset-level ranked instance AP for 3D mask proposals.

The functions in this module intentionally operate on compact records rather
than dense masks. Scene-level code computes prediction-to-GT IoU candidates once,
then distributed evaluation can gather these small records and finalize AP on
rank 0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

SCANNET_IGNORE_MODE = "scannet_compatible_ignore"
STRICT_IGNORE_MODE = "strict_all_points_iou"
SCANNET_MIN_REGION_SIZE = 100

OBJECT_CLASS = "object"


def get_iou_thresholds(
    preset: str = "scannet_official",
    custom: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    """Return IoU thresholds for a named AP protocol.

    ``scannet_official`` follows the ScanNet/Mask3D evaluator exactly: AP is
    averaged over ``np.arange(0.5, 0.95, 0.05)`` and AP25 is appended as a
    separately reported threshold.
    """
    preset = str(preset).strip().lower()
    if preset == "custom":
        if custom is None:
            raise ValueError("custom threshold preset requires a custom list")
        values = [float(v) for v in custom]
    elif preset == "scannet_official":
        values = [float(v) for v in np.append(np.arange(0.5, 0.95, 0.05), 0.25)]
    elif preset == "coco_style":
        values = [float(v) for v in np.arange(0.5, 0.95 + 1e-9, 0.05)]
    else:
        raise ValueError(
            f"Unknown IoU threshold preset {preset!r}; expected "
            "'scannet_official', 'coco_style', or 'custom'."
        )

    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(float(value), 2)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(rounded)
    return out


def threshold_metric_name(threshold: float) -> str:
    return f"AP{int(round(float(threshold) * 100))}"


def _nanmean(values: list[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def _as_bool_array(values: np.ndarray | list[bool]) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def build_instance_ap_records(
    *,
    scene_id: str,
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    scores: np.ndarray,
    query_indices: np.ndarray | None = None,
    granularity: str | None = None,
    class_agnostic: bool = True,
    gt_instance_class_ids: dict[int, int] | None = None,
    pred_class_ids: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    void_mask: np.ndarray | None = None,
    min_valid_gt_points: int = 1,
    min_valid_pred_points: int = 1,
) -> dict[str, Any]:
    """Build compact AP records for one scene and one prediction head.

    Parameters
    ----------
    gt_ids:
        Per-point valid GT instance ids. ``0`` means ignored/background.
    proposals / scores:
        Predicted binary masks and confidence scores after proposal-size
        filtering. Scores are not thresholded here.
    eval_mask:
        Optional point universe for pseudo-label evaluation. Points outside this
        mask are removed from both predictions and GT before IoU computation.
    void_mask:
        Optional ignored-region mask used by ScanNet-compatible FP suppression.
        If absent, ``gt_ids == 0`` is treated as void.
    min_valid_gt_points / min_valid_pred_points:
        Minimum valid instance/proposal size. For ScanNet-compatible real-GT
        metrics this should be ``SCANNET_MIN_REGION_SIZE`` (100), matching the
        official evaluator's ``min_region_sizes``. Smaller GT instances are
        excluded from valid positives and treated as ignored regions.
    """
    scene_id = str(scene_id)
    gt_local = np.asarray(gt_ids, dtype=np.int64)
    if gt_local.ndim != 1:
        raise ValueError(f"gt_ids must be 1D, got shape {gt_local.shape}")

    if eval_mask is not None:
        eval_bool = _as_bool_array(eval_mask)
        if eval_bool.shape != gt_local.shape:
            raise ValueError(
                f"eval_mask shape {eval_bool.shape} != gt_ids shape {gt_local.shape}"
            )
        gt_local = np.where(eval_bool, gt_local, 0)
    else:
        eval_bool = np.ones_like(gt_local, dtype=bool)

    if void_mask is not None:
        void_bool = _as_bool_array(void_mask)
        if void_bool.shape != gt_local.shape:
            raise ValueError(
                f"void_mask shape {void_bool.shape} != gt_ids shape {gt_local.shape}"
            )
        void_bool = void_bool | ~eval_bool
    else:
        void_bool = (gt_local <= 0) | ~eval_bool

    gt_records: list[dict[str, Any]] = []
    gt_masks: dict[int, np.ndarray] = {}
    gt_classes: dict[int, str | int] = {}
    gt_sizes: dict[int, int] = {}

    for raw_gt_id in sorted(int(x) for x in np.unique(gt_local) if int(x) > 0):
        mask = gt_local == raw_gt_id
        size = int(mask.sum())
        if size < int(min_valid_gt_points):
            void_bool = void_bool | mask
            continue

        if class_agnostic:
            class_id: str | int = OBJECT_CLASS
        else:
            if gt_instance_class_ids is None or raw_gt_id not in gt_instance_class_ids:
                void_bool = void_bool | mask
                continue
            class_id = int(gt_instance_class_ids[raw_gt_id])

        if size <= 0:
            continue
        gt_masks[raw_gt_id] = mask
        gt_classes[raw_gt_id] = class_id
        gt_sizes[raw_gt_id] = size
        gt_records.append(
            {
                "scene_id": scene_id,
                "gt_id": int(raw_gt_id),
                "class_id": class_id,
                "size": size,
                "granularity": granularity,
            }
        )

    score_values = np.asarray(scores, dtype=np.float64)
    if query_indices is None:
        query_values = np.arange(len(proposals), dtype=np.int64)
    else:
        query_values = np.asarray(query_indices, dtype=np.int64)

    if pred_class_ids is not None:
        pred_class_values = np.asarray(pred_class_ids)
        if len(pred_class_values) != len(proposals):
            raise ValueError("pred_class_ids length must match proposals length")
    else:
        pred_class_values = None

    pred_records: list[dict[str, Any]] = []
    for pred_idx, raw_mask in enumerate(proposals):
        pred_mask = _as_bool_array(raw_mask)
        if pred_mask.shape != gt_local.shape:
            raise ValueError(
                f"proposal shape {pred_mask.shape} != gt_ids shape {gt_local.shape}"
            )
        pred_mask = pred_mask & eval_bool
        pred_size = int(pred_mask.sum())
        if pred_size < int(min_valid_pred_points) or pred_size <= 0:
            continue

        if class_agnostic or pred_class_values is None:
            pred_class: str | int = OBJECT_CLASS
        else:
            pred_class = int(pred_class_values[pred_idx])

        candidate_gt_ids: list[int] = []
        candidate_ious: list[float] = []
        for gt_id, gt_mask in gt_masks.items():
            if gt_classes[gt_id] != pred_class:
                continue
            inter = int(np.logical_and(pred_mask, gt_mask).sum())
            if inter <= 0:
                continue
            union = pred_size + gt_sizes[gt_id] - inter
            iou = float(inter / union) if union > 0 else 0.0
            if iou > 0.0:
                candidate_gt_ids.append(int(gt_id))
                candidate_ious.append(iou)

        ignore_inter = int(np.logical_and(pred_mask, void_bool).sum())
        ignore_fraction = float(ignore_inter / max(pred_size, 1))
        oracle_score = float(max(candidate_ious)) if candidate_ious else 0.0
        score = float(score_values[pred_idx]) if pred_idx < len(score_values) else 0.0
        query_id = int(query_values[pred_idx]) if pred_idx < len(query_values) else pred_idx

        pred_records.append(
            {
                "scene_id": scene_id,
                "pred_id": int(pred_idx),
                "score": score,
                "oracle_score": oracle_score,
                "pred_size": pred_size,
                "query_id": query_id,
                "granularity": granularity,
                "class_id": pred_class,
                "candidate_gt_ids": candidate_gt_ids,
                "candidate_ious": candidate_ious,
                "ignore_fraction": ignore_fraction,
            }
        )

    return {
        "predictions": pred_records,
        "ground_truths": gt_records,
        "num_predictions": len(pred_records),
        "total_gt_instances": len(gt_records),
        "granularity": granularity,
        "min_valid_gt_points": int(min_valid_gt_points),
        "min_valid_pred_points": int(min_valid_pred_points),
    }


def merge_ap_record_sets(record_sets: list[dict[str, Any]]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []
    for records in record_sets:
        if not isinstance(records, dict):
            continue
        predictions.extend(records.get("predictions", []) or [])
        ground_truths.extend(records.get("ground_truths", []) or [])
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "num_predictions": len(predictions),
        "total_gt_instances": len(ground_truths),
    }


def _scannet_ap_from_events(
    y_true: list[float],
    y_score: list[float],
    *,
    hard_false_negatives: int,
    has_gt: bool,
    has_pred: bool,
) -> float:
    """Integrate a PR curve using the ScanNet/Mask3D evaluator convention."""
    if has_gt and has_pred:
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        y_score_arr = np.asarray(y_score, dtype=np.float64)
        if y_score_arr.size == 0:
            return 0.0

        score_arg_sort = np.argsort(y_score_arr)
        y_score_sorted = y_score_arr[score_arg_sort]
        y_true_sorted = y_true_arr[score_arg_sort]
        y_true_sorted_cumsum = np.cumsum(y_true_sorted)
        _, unique_indices = np.unique(y_score_sorted, return_index=True)
        num_prec_recall = len(unique_indices) + 1
        num_examples = len(y_score_sorted)
        num_true_examples = (
            y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
        )

        precision = np.zeros(num_prec_recall, dtype=np.float64)
        recall = np.zeros(num_prec_recall, dtype=np.float64)
        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0.0)

        for idx_res, idx_scores in enumerate(unique_indices):
            cumsum = y_true_sorted_cumsum[idx_scores - 1]
            tp = num_true_examples - cumsum
            fp = num_examples - idx_scores - tp
            fn = cumsum + hard_false_negatives
            precision[idx_res] = float(tp) / max(float(tp + fp), 1e-12)
            recall[idx_res] = float(tp) / max(float(tp + fn), 1e-12)

        precision[-1] = 1.0
        recall[-1] = 0.0
        recall_for_conv = np.copy(recall)
        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
        recall_for_conv = np.append(recall_for_conv, 0.0)
        step_widths = np.convolve(recall_for_conv, [-0.5, 0.0, 0.5], "valid")
        return float(np.dot(precision, step_widths))

    if has_gt:
        return 0.0
    return float("nan")


def _evaluate_class_threshold(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    class_id: str | int,
    threshold: float,
    ignore_mode: str,
    score_key: str,
) -> dict[str, Any]:
    gt_keys = {
        (str(gt["scene_id"]), gt["gt_id"])
        for gt in ground_truths
        if gt.get("class_id") == class_id
    }
    class_predictions = [
        pred for pred in predictions
        if pred.get("class_id") == class_id
    ]
    has_gt = len(gt_keys) > 0
    has_pred = len(class_predictions) > 0
    matched: set[tuple[str, int]] = set()
    y_true: list[float] = []
    y_score: list[float] = []

    pred_order = sorted(
        class_predictions,
        key=lambda pred: (
            -float(pred.get(score_key, pred.get("score", 0.0))),
            str(pred.get("scene_id", "")),
            int(pred.get("pred_id", 0)),
        ),
    )

    def passes_threshold(iou: float) -> bool:
        if ignore_mode == SCANNET_IGNORE_MODE:
            return float(iou) > float(threshold)
        if ignore_mode == STRICT_IGNORE_MODE:
            return float(iou) >= float(threshold)
        raise ValueError(
            f"Unknown ignore_mode={ignore_mode!r}; expected "
            f"{SCANNET_IGNORE_MODE!r} or {STRICT_IGNORE_MODE!r}."
        )

    for pred in pred_order:
        scene = str(pred["scene_id"])
        candidate_gt_ids = pred.get("candidate_gt_ids", []) or []
        candidate_ious = pred.get("candidate_ious", []) or []
        best_key: tuple[str, int] | None = None
        best_match_iou = -1.0
        found_any_valid_gt = False

        for gt_id_raw, iou_raw in zip(candidate_gt_ids, candidate_ious):
            gt_key = (scene, int(gt_id_raw))
            if gt_key not in gt_keys:
                continue
            iou = float(iou_raw)
            if passes_threshold(iou):
                found_any_valid_gt = True
                if gt_key not in matched and iou > best_match_iou:
                    best_match_iou = iou
                    best_key = gt_key

        score = float(pred.get(score_key, pred.get("score", 0.0)))
        if best_key is not None:
            matched.add(best_key)
            y_true.append(1.0)
            y_score.append(score)
            continue

        ignore_fraction = float(pred.get("ignore_fraction", 0.0))
        if ignore_mode == SCANNET_IGNORE_MODE and not found_any_valid_gt:
            if ignore_fraction > threshold:
                continue

        y_true.append(0.0)
        y_score.append(score)

    hard_fn = max(len(gt_keys) - len(matched), 0)
    ap = _scannet_ap_from_events(
        y_true,
        y_score,
        hard_false_negatives=hard_fn,
        has_gt=has_gt,
        has_pred=has_pred,
    )
    return {
        "ap": ap,
        "num_gt": len(gt_keys),
        "num_predictions": len(class_predictions),
        "num_tp": int(sum(y_true)),
        "num_fp": int(len(y_true) - sum(y_true)),
        "num_ignored_predictions": int(len(class_predictions) - len(y_true)),
    }


def evaluate_official_instance_ap(
    records: dict[str, Any],
    *,
    threshold_preset: str = "scannet_official",
    thresholds: list[float] | tuple[float, ...] | None = None,
    ignore_mode: str = SCANNET_IGNORE_MODE,
    score_key: str = "score",
) -> dict[str, Any]:
    """Finalize global ranked AP from compact prediction/GT records."""
    predictions = list(records.get("predictions", []) or [])
    ground_truths = list(records.get("ground_truths", []) or [])
    if thresholds is None:
        thresholds = get_iou_thresholds(threshold_preset)
    thresholds = [round(float(t), 2) for t in thresholds]

    classes = sorted(
        {gt.get("class_id") for gt in ground_truths}
        | {pred.get("class_id") for pred in predictions},
        key=lambda x: str(x),
    )
    if not classes:
        classes = [OBJECT_CLASS]

    by_threshold: dict[str, float] = {}
    by_threshold_detail: dict[str, Any] = {}
    per_class: dict[str, dict[str, float]] = defaultdict(dict)
    total_gt = len(ground_truths)

    for threshold in thresholds:
        metric_name = threshold_metric_name(threshold)
        class_aps: list[float] = []
        detail_by_class: dict[str, Any] = {}
        for class_id in classes:
            detail = _evaluate_class_threshold(
                predictions,
                ground_truths,
                class_id=class_id,
                threshold=threshold,
                ignore_mode=ignore_mode,
                score_key=score_key,
            )
            ap = float(detail["ap"])
            per_class[str(class_id)][metric_name] = ap
            detail_by_class[str(class_id)] = detail
            if detail["num_gt"] > 0 and not math.isnan(ap):
                class_aps.append(ap)
        by_threshold[metric_name] = _nanmean(class_aps)
        by_threshold_detail[metric_name] = detail_by_class

    map_thresholds = [
        threshold_metric_name(t)
        for t in thresholds
        if not math.isclose(float(t), 0.25)
    ]
    ap_mean = _nanmean([by_threshold[k] for k in map_thresholds if k in by_threshold])

    return {
        "AP": ap_mean,
        "AP50": by_threshold.get("AP50", float("nan")),
        "AP25": by_threshold.get("AP25", float("nan")),
        "by_threshold": by_threshold,
        "per_class": dict(per_class),
        "detail": by_threshold_detail,
        "threshold_preset": threshold_preset,
        "ignore_mode": ignore_mode,
        "num_predictions": len(predictions),
        "total_gt_instances": total_gt,
        "num_classes": sum(
            1 for class_id in classes
            if any(gt.get("class_id") == class_id for gt in ground_truths)
        ),
    }


def evaluate_official_and_oracle_ap(
    records: dict[str, Any],
    *,
    threshold_preset: str = "scannet_official",
    thresholds: list[float] | tuple[float, ...] | None = None,
    ignore_mode: str = SCANNET_IGNORE_MODE,
) -> dict[str, Any]:
    """Return normal AP plus oracle-score AP for the same masks."""
    normal = evaluate_official_instance_ap(
        records,
        threshold_preset=threshold_preset,
        thresholds=thresholds,
        ignore_mode=ignore_mode,
        score_key="score",
    )
    oracle = evaluate_official_instance_ap(
        records,
        threshold_preset=threshold_preset,
        thresholds=thresholds,
        ignore_mode=ignore_mode,
        score_key="oracle_score",
    )
    normal["oracle_AP"] = oracle["AP"]
    normal["oracle_AP50"] = oracle["AP50"]
    normal["oracle_AP25"] = oracle["AP25"]
    normal["oracle_by_threshold"] = oracle["by_threshold"]
    return normal
