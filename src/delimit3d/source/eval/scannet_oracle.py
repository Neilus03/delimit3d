"""Oracle ScanNet overlap diagnostics used during source preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from delimit3d.identity import project_slug
from delimit3d.source.common.types import ClusterOutput
from delimit3d.source.datasets.base import SceneAdapter
from delimit3d.source.datasets.scannet.benchmark import (
    SCANNET_EVAL_BENCHMARK_20,
    normalize_scannet_eval_benchmark,
)
from delimit3d.source.datasets.scannet.gt import load_scannet_gt_instance_ids
from delimit3d.source.export.visualization import save_labeled_mesh_ply


def build_proposals_from_cluster_outputs(
    cluster_outputs: list[ClusterOutput],
) -> tuple[list[np.ndarray], list[float]]:
    proposals: list[np.ndarray] = []
    proposal_sources: list[float] = []

    for cluster_output in cluster_outputs:
        labels = cluster_output.labels
        unique_instances = np.unique(labels)
        unique_instances = unique_instances[unique_instances >= 0]

        for inst_id in unique_instances:
            proposals.append(labels == inst_id)
            proposal_sources.append(float(cluster_output.granularity))

    return proposals, proposal_sources


def _build_size_buckets(gt_ids: np.ndarray):
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]

    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    gt_areas_list = list(gt_areas.values())
    if len(gt_areas_list) == 0:
        return {}, gt_areas

    p33 = np.percentile(gt_areas_list, 33.33)
    p66 = np.percentile(gt_areas_list, 66.67)

    size_buckets = {
        f"Small (<{p33:.0f} pts)": (0, p33),
        f"Medium ({p33:.0f}-{p66:.0f} pts)": (p33, p66),
        f"Large (>{p66:.0f} pts)": (p66, float("inf")),
    }
    return size_buckets, gt_areas


def _best_iou_and_best_source_per_gt(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    proposal_sources: list[float],
):
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]

    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    best_iou_by_gt = {}
    best_source_by_gt = {}

    for g_id in gt_instances:
        g_id_int = int(g_id)
        g_mask = gt_ids == g_id
        best_iou = 0.0
        best_source = None

        for i, p_mask in enumerate(proposals):
            intersection = int(np.sum(p_mask & g_mask))
            if intersection == 0:
                continue

            union = prop_areas[i] + gt_areas[g_id_int] - intersection
            iou = intersection / max(union, 1)

            if iou > best_iou:
                best_iou = iou
                best_source = proposal_sources[i]

        best_iou_by_gt[g_id_int] = float(best_iou)
        best_source_by_gt[g_id_int] = best_source

    return best_iou_by_gt, best_source_by_gt


def compute_additional_oracle_metrics(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    proposal_sources: list[float],
):
    size_buckets, gt_areas = _build_size_buckets(gt_ids)
    if not size_buckets:
        return {}

    best_iou_by_gt, best_source_by_gt = _best_iou_and_best_source_per_gt(
        gt_ids,
        proposals,
        proposal_sources,
    )

    thresholds = np.arange(0.25, 1.0, 0.05)

    map_by_bucket = {}
    for bucket_name, (min_pts, max_pts) in size_buckets.items():
        valid_gts = [g for g, area in gt_areas.items() if min_pts <= area < max_pts]
        if len(valid_gts) == 0:
            map_by_bucket[bucket_name] = 0.0
            continue

        recalls = []
        for th in thresholds:
            matched = sum(best_iou_by_gt[g] >= float(th) for g in valid_gts)
            recalls.append(matched / len(valid_gts))

        map_by_bucket[bucket_name] = float(np.mean(recalls))

    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas_full = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    topk_coverage = {}
    for th in (0.25, 0.50):
        counts = {1: 0, 3: 0, 5: 0}

        for g_id in gt_instances:
            g_id_int = int(g_id)
            g_mask = gt_ids == g_id
            n_above = 0

            for i, p_mask in enumerate(proposals):
                intersection = int(np.sum(p_mask & g_mask))
                if intersection == 0:
                    continue

                union = prop_areas[i] + gt_areas_full[g_id_int] - intersection
                iou = intersection / max(union, 1)
                if iou >= th:
                    n_above += 1

            for k in counts:
                if n_above >= k:
                    counts[k] += 1

        denom = max(len(gt_instances), 1)
        topk_coverage[f"iou_{th:.2f}"] = {
            "R_at_least_1": counts[1] / denom,
            "R_at_least_3": counts[3] / denom,
            "R_at_least_5": counts[5] / denom,
        }

    unique_granularities = sorted({float(s) for s in proposal_sources})
    winner_counts = {f"g{g}": 0 for g in unique_granularities}
    no_match = 0

    for g_id in gt_instances:
        src = best_source_by_gt[int(g_id)]
        if src is None:
            no_match += 1
            continue
        winner_counts[f"g{src}"] += 1

    total_gt = max(len(gt_instances), 1)
    winner_share = {k: v / total_gt for k, v in winner_counts.items()}
    winner_share["no_match"] = no_match / total_gt

    return {
        "oracle_mAP_25_95_by_bucket": map_by_bucket,
        "topk_proposal_coverage": topk_coverage,
        "winner_granularity_share": winner_share,
    }


def evaluate_oracle_ap(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    thresholds=(0.25, 0.50),
):
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}

    gt_areas_list = list(gt_areas.values())
    if len(gt_areas_list) == 0:
        return {}

    p33 = np.percentile(gt_areas_list, 33.33)
    p66 = np.percentile(gt_areas_list, 66.67)
    size_buckets = {
        f"Small (<{p33:.0f} pts)": (0, p33),
        f"Medium ({p33:.0f}-{p66:.0f} pts)": (p33, p66),
        f"Large (>{p66:.0f} pts)": (p66, float("inf")),
    }

    results = {}
    prop_areas = [int(np.sum(p)) for p in proposals]

    print(f"Evaluating {len(proposals)} pooled proposals against {len(gt_instances)} GT instances")

    for bucket_name, (min_pts, max_pts) in size_buckets.items():
        valid_gts = [g for g, area in gt_areas.items() if min_pts <= area < max_pts]
        num_gt = len(valid_gts)

        results[bucket_name] = {"AP25": 0.0, "AP50": 0.0, "Count": num_gt}
        if num_gt == 0:
            continue

        for th in thresholds:
            matched_this_th = 0
            for g_id in valid_gts:
                g_mask = gt_ids == g_id
                best_iou = 0.0

                for i, p_mask in enumerate(proposals):
                    intersection = int(np.sum(p_mask & g_mask))
                    if intersection == 0:
                        continue

                    union = prop_areas[i] + gt_areas[g_id] - intersection
                    iou = intersection / max(union, 1)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= th:
                    matched_this_th += 1

            if th == 0.25:
                results[bucket_name]["AP25"] = matched_this_th / num_gt
            elif th == 0.50:
                results[bucket_name]["AP50"] = matched_this_th / num_gt

    return results


def _oracle_size_bucket_slug(bucket_name: object) -> str | None:
    lower = str(bucket_name).strip().lower()
    if lower.startswith("small"):
        return "small"
    if lower.startswith("medium"):
        return "medium"
    if lower.startswith("large"):
        return "large"
    return None


def flatten_oracle_ap_bucket_metrics(oracle_results: dict[str, Any] | None) -> dict[str, float | None]:
    """Stable keys aligned with ScanNet++ reporting: oracle_ap25_small, oracle_ap50_large, etc."""
    if not oracle_results:
        return {}
    flat: dict[str, float | None] = {}
    for bucket_name, bucket_metrics in oracle_results.items():
        if not isinstance(bucket_metrics, dict):
            continue
        bucket = _oracle_size_bucket_slug(bucket_name)
        if bucket is None:
            continue
        flat[f"oracle_ap25_{bucket}"] = bucket_metrics.get("AP25")
        flat[f"oracle_ap50_{bucket}"] = bucket_metrics.get("AP50")
    return flat


def flatten_oracle_map_bucket_metrics(additional_metrics: dict[str, Any] | None) -> dict[str, float | None]:
    """mAP@[0.25:0.95] per size bucket: oracle_map_25_95_small, etc."""
    if not additional_metrics:
        return {}
    map_by_bucket = additional_metrics.get("oracle_mAP_25_95_by_bucket") or {}
    flat: dict[str, float | None] = {}
    for bucket_name, val in map_by_bucket.items():
        bucket = _oracle_size_bucket_slug(bucket_name)
        if bucket is None or val is None:
            continue
        try:
            flat[f"oracle_map_25_95_{bucket}"] = float(val)
        except (TypeError, ValueError):
            continue
    return flat


def build_oracle_best_labels(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    min_iou: float = 0.1,
) -> np.ndarray:
    oracle_labels = np.full(gt_ids.shape, -1, dtype=np.int32)

    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    for g_id in gt_instances:
        g_id_int = int(g_id)
        g_mask = gt_ids == g_id
        best_iou = 0.0
        best_idx = -1

        for i, p_mask in enumerate(proposals):
            intersection = int(np.sum(p_mask & g_mask))
            if intersection == 0:
                continue

            union = prop_areas[i] + gt_areas[g_id_int] - intersection
            iou = intersection / max(union, 1)

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= min_iou:
            oracle_labels[proposals[best_idx]] = g_id_int

    return oracle_labels


def save_oracle_best_ply(
    geometry_path: Path,
    out_path: Path,
    oracle_labels: np.ndarray,
) -> None:
    save_labeled_mesh_ply(
        source_ply_path=geometry_path,
        labels=oracle_labels,
        out_path=out_path,
    )


def compute_clustering_metrics(
    gt_ids: np.ndarray,
    oracle_labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute clustering-consistency metrics between GT instance ids and oracle labels.

    Notes:
    - We ignore GT background points (gt_ids <= 0).
    - We keep oracle noise labels (e.g. -1) as their own cluster.
    """
    mask = gt_ids > 0
    if not np.any(mask):
        return {"NMI": float("nan"), "ARI": float("nan")}

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        print(
            "Warning: scikit-learn not installed; cannot compute NMI/ARI. "
            "Install 'scikit-learn' to enable clustering metrics."
        )
        return {"NMI": float("nan"), "ARI": float("nan")}

    y_true = gt_ids[mask].astype(np.int64)
    y_pred = oracle_labels[mask].astype(np.int64)

    # Label-invariant clustering metrics (IDs can be any integers).
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    ari = adjusted_rand_score(y_true, y_pred)

    return {"NMI": float(nmi), "ARI": float(ari)}


def evaluate_and_save_scannet_oracle(
    adapter: SceneAdapter,
    cluster_outputs: list[ClusterOutput],
    eval_benchmark: str | None = SCANNET_EVAL_BENCHMARK_20,
    min_iou_for_ply: float = 0.1,
    save_artifacts: bool = True,
    output_dir: Path | None = None,
    gt_scene_root: Path | None = None,
) -> dict:
    if adapter.dataset_name == "scannet":
        resolved_benchmark = normalize_scannet_eval_benchmark(eval_benchmark)
        suffix = (
            ""
            if resolved_benchmark == SCANNET_EVAL_BENCHMARK_20
            else f"_{resolved_benchmark}"
        )
        gt_ids = load_scannet_gt_instance_ids(
            Path(gt_scene_root) if gt_scene_root is not None else adapter.scene_root,
            adapter.scene_id,
            eval_benchmark=resolved_benchmark,
        )
    else:
        resolved_benchmark = (
            None if eval_benchmark is None else str(eval_benchmark).strip().lower() or None
        )
        suffix = "" if resolved_benchmark in {None, "", "default"} else f"_{resolved_benchmark}"
        gt_ids = adapter.load_gt_instance_ids()
        if gt_ids is None:
            raise RuntimeError(
                f"Oracle evaluation for dataset '{adapter.dataset_name}' requires GT instance ids."
            )
        if adapter.dataset_name == "structured3d":
            n_gt = int(gt_ids.shape[0])
            for co in cluster_outputs:
                if len(co.labels) != n_gt:
                    raise RuntimeError(
                        f"Structured3D oracle: len(gt_instance_ids)={n_gt} but cluster labels "
                        f"len={len(co.labels)} at granularity {co.granularity}. "
                        "Regenerate the prepared scene or verify mesh vs cluster indexing."
                    )

    proposals, proposal_sources = build_proposals_from_cluster_outputs(cluster_outputs)
    if len(proposals) == 0:
        raise RuntimeError("No proposals available for oracle evaluation.")

    oracle_results = evaluate_oracle_ap(gt_ids, proposals)
    additional_metrics = compute_additional_oracle_metrics(
        gt_ids,
        proposals,
        proposal_sources,
    )
    oracle_best_labels = build_oracle_best_labels(
        gt_ids,
        proposals,
        min_iou=min_iou_for_ply,
    )
    clustering_metrics = compute_clustering_metrics(gt_ids, oracle_best_labels)

    scene_root = Path(output_dir) if output_dir is not None else adapter.scene_root
    if save_artifacts:
        scene_root.mkdir(parents=True, exist_ok=True)
    metrics_path = scene_root / f"oracle_metrics{suffix}.json"
    label_prefix = project_slug()
    labels_path = scene_root / f"{label_prefix}_oracle_best_combined_labels{suffix}.npy"
    ply_path = scene_root / f"{label_prefix}_oracle_best_combined{suffix}.ply"

    results_to_save = dict(oracle_results)
    results_to_save["_extras"] = additional_metrics
    results_to_save["_clustering"] = clustering_metrics

    if save_artifacts:
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(results_to_save, f, indent=2)

        np.save(labels_path, oracle_best_labels)

        geometry_record = adapter.get_geometry_record()
        save_oracle_best_ply(
            geometry_path=geometry_record.geometry_path,
            out_path=ply_path,
            oracle_labels=oracle_best_labels,
        )

        print(f"Saved oracle metrics: {metrics_path}")
        print(f"Saved oracle pooled labels: {labels_path}")
        print(f"Saved oracle pooled ply: {ply_path}")
        if resolved_benchmark is not None:
            print(f"Oracle benchmark: {resolved_benchmark}")
        print(
            "Clustering consistency metrics (GT vs oracle labels on foreground points): "
            f"NMI={clustering_metrics['NMI']:.4f}, ARI={clustering_metrics['ARI']:.4f}"
        )
    else:
        metrics_path = None
        labels_path = None
        ply_path = None

    return {
        "eval_benchmark": resolved_benchmark,
        "metrics_path": metrics_path,
        "labels_path": labels_path,
        "ply_path": ply_path,
        "oracle_results": oracle_results,
        "additional_metrics": additional_metrics,
        "clustering_metrics": clustering_metrics,
    }
