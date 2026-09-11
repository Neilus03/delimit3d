"""AGILE3D-compatible ScanNet++ multi-object protocol helpers.

The functions in this module are model-agnostic except for the small decoder
interface used by the runner. They freeze object/episode construction,
implement the AGILE3D correction-click policy, expand token predictions to
mesh vertices, and aggregate interactive metrics with paired scene bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(array: np.ndarray | torch.Tensor) -> str:
    if torch.is_tensor(array):
        array = array.detach().cpu().contiguous().numpy()
    value = np.asarray(array)
    return hashlib.sha256(
        str(value.dtype).encode("ascii")
        + str(tuple(value.shape)).encode("ascii")
        + value.tobytes(order="C")
    ).hexdigest()


def scene_seed(seed: int, scene: str, operation: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{scene}:{operation}".encode()).digest()[:8], "little"
    )


def read_scene_ids(path: str | Path) -> list[str]:
    values = [
        line.strip()
        for line in Path(path).expanduser().resolve(strict=True).read_text().splitlines()
    ]
    values = [value for value in values if value and not value.startswith("#")]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{path}: split is empty or contains duplicate scene IDs")
    return values


def load_class_mapping(scannetpp_root: str | Path) -> tuple[list[str], dict[str, str]]:
    root = Path(scannetpp_root).expanduser().resolve(strict=True) / "metadata" / "semantic_benchmark"
    classes = (root / "top100_instance.txt").read_text().splitlines()
    import csv

    with (root / "map_benchmark.csv").open(newline="") as handle:
        mapping = {
            row["class"]: (row["instance_map_to"] or row["class"])
            for row in csv.DictReader(handle)
        }
    if not classes or not mapping:
        raise ValueError("ScanNet++ semantic benchmark metadata is empty")
    return classes, mapping


def _segment_group_indices(
    seg_indices: np.ndarray,
    group_segments: Sequence[int],
    *,
    sorted_seg_indices: np.ndarray | None = None,
    sort_order: np.ndarray | None = None,
) -> np.ndarray:
    if sorted_seg_indices is None or sort_order is None:
        sort_order = np.argsort(seg_indices, kind="stable")
        sorted_seg_indices = seg_indices[sort_order]
    values = np.unique(np.asarray(group_segments, dtype=np.int64))
    starts = np.searchsorted(sorted_seg_indices, values, side="left")
    ends = np.searchsorted(sorted_seg_indices, values, side="right")
    chunks = [
        sort_order[int(start) : int(end)]
        for start, end in zip(starts, ends, strict=True)
        if end > start
    ]
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    indices = np.concatenate(chunks).astype(np.int64, copy=False)
    indices.sort(kind="stable")
    return indices


def load_training_scene(
    scene: str,
    *,
    scannetpp_root: str | Path,
    pack_root: str | Path,
    classes: Sequence[str],
    class_mapping: Mapping[str, str],
    minimum_instance_points: int = 100,
    maximum_objects_per_scene: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[int, np.ndarray], dict[str, str]]:
    """Load one ScanNet++ training scene and category-agnostic object masks."""
    pack = Path(pack_root).expanduser().resolve(strict=True) / scene / "training_pack"
    scan = Path(scannetpp_root).expanduser().resolve(strict=True) / "data" / scene / "scans"
    array_paths = [pack / f"{key}.npy" for key in ("points", "colors", "normals")]
    source_paths = array_paths + [scan / "segments.json", scan / "segments_anno.json"]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{scene}: missing {missing}")

    points, colors, normals = [
        np.asarray(np.load(path), dtype=np.float32) for path in array_paths
    ]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{scene}: points must be [N,3]")
    if colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError(f"{scene}: RGBN arrays are misaligned")

    segments = json.loads((scan / "segments.json").read_text())
    seg_indices = np.asarray(segments["segIndices"], dtype=np.int64)
    if seg_indices.shape != (len(points),):
        raise ValueError(f"{scene}: segments do not align with points")
    groups = json.loads((scan / "segments_anno.json").read_text())["segGroups"]
    order = np.argsort(seg_indices, kind="stable")
    sorted_segments = seg_indices[order]

    masks: dict[int, np.ndarray] = {}
    objects: list[dict[str, Any]] = []
    for group in groups:
        label = class_mapping.get(str(group["label"]), str(group["label"]))
        if label not in classes:
            continue
        instance = int(group["objectId"])
        if instance in masks:
            raise ValueError(f"{scene}: duplicate object ID {instance}")
        indices = _segment_group_indices(
            seg_indices,
            group["segments"],
            sorted_seg_indices=sorted_segments,
            sort_order=order,
        )
        if len(indices) < int(minimum_instance_points):
            continue
        masks[instance] = indices
        objects.append(
            {
                "instance": instance,
                "semantic_class": int(classes.index(label)),
                "semantic_name": str(label),
                "instance_points": int(len(indices)),
            }
        )

    objects.sort(key=lambda item: int(item["instance"]))
    before_cap = len(objects)
    if before_cap > int(maximum_objects_per_scene):
        rng = np.random.default_rng(scene_seed(20260911, scene, "object-cap"))
        chosen = sorted(
            int(index)
            for index in rng.choice(before_cap, int(maximum_objects_per_scene), replace=False)
        )
        objects = [objects[index] for index in chosen]
    selected_masks = {int(item["instance"]): masks[int(item["instance"])] for item in objects}
    hashes = {str(path): file_sha256(path) for path in source_paths}
    hashes["points_resolved"] = file_sha256((pack / "points.npy").resolve())
    hashes["colors_resolved"] = file_sha256((pack / "colors.npy").resolve())
    return points, colors, normals, objects, selected_masks, hashes


def load_bundle_scene(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[int, np.ndarray]]:
    """Load a scene from the existing frozen validation bundle."""
    path = Path(str(record["data_path"])).expanduser().resolve(strict=True)
    if file_sha256(path) != record["data_sha256"]:
        raise ValueError(f"{record['scene']}: validation scene data hash mismatch")
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        objects = [dict(item) for item in record["objects"]]
        masks = {
            int(item["instance"]): np.asarray(
                data[f"target_{int(item['instance'])}"], dtype=np.int64
            )
            for item in objects
        }
    return points, colors, normals, objects, masks


def objects_non_overlapping(
    object_ids: Sequence[int],
    masks: Mapping[int, np.ndarray],
) -> bool:
    seen: set[int] = set()
    for object_id in object_ids:
        values = set(np.asarray(masks[int(object_id)], dtype=np.int64).tolist())
        if seen.intersection(values):
            return False
        seen.update(values)
    return True


def choose_non_overlapping_group(
    object_ids: Sequence[int],
    masks: Mapping[int, np.ndarray],
    count: int,
    *,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> tuple[list[int], int]:
    """Choose a deterministic tuple, recording the number of rejected draws."""
    candidates = sorted(int(value) for value in object_ids)
    if count <= 0 or count > len(candidates):
        raise ValueError(f"cannot choose {count} objects from {len(candidates)} candidates")
    rejected = 0
    for _ in range(int(max_attempts)):
        trial = sorted(int(value) for value in rng.choice(candidates, count, replace=False))
        if objects_non_overlapping(trial, masks):
            return trial, rejected
        rejected += 1
    # Fail closed only after deterministic exhaustive fallback. This preserves
    # the requested K whenever the annotations contain a valid tuple.
    import itertools

    for trial_tuple in itertools.combinations(candidates, count):
        trial = list(trial_tuple)
        if objects_non_overlapping(trial, masks):
            return trial, rejected
    raise RuntimeError(f"no non-overlapping tuple of size {count} exists")


def _token_membership(representatives: np.ndarray, raw_indices: np.ndarray) -> np.ndarray:
    representatives = np.asarray(representatives, dtype=np.int64)
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    if raw_indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_raw = np.sort(raw_indices, kind="stable")
    positions = np.searchsorted(sorted_raw, representatives)
    valid = positions < sorted_raw.size
    valid &= sorted_raw[np.minimum(positions, sorted_raw.size - 1)] == representatives
    return np.flatnonzero(valid).astype(np.int64)


def build_token_targets(
    representative_indices: np.ndarray,
    masks: Mapping[int, np.ndarray],
    object_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    reps = np.asarray(representative_indices, dtype=np.int64)
    return {
        int(object_id): _token_membership(reps, masks[int(object_id)])
        for object_id in object_ids
    }


def center_click(token_indices: np.ndarray, token_xyz: np.ndarray) -> int:
    indices = np.asarray(token_indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("cannot choose a click for an empty object")
    center = np.asarray(token_xyz, dtype=np.float64)[indices].mean(axis=0)
    distances = np.sum((np.asarray(token_xyz)[indices] - center[None, :]) ** 2, axis=1)
    # Stable argmin gives the lowest token index on equal distances.
    return int(indices[int(np.argmin(distances))])


def enforce_click_labels(
    prediction: np.ndarray,
    clicks: Mapping[int | str, Sequence[int]],
) -> np.ndarray:
    result = np.asarray(prediction, dtype=np.int64).copy()
    for key, values in clicks.items():
        label = int(key)
        for index in values:
            index = int(index)
            if index < 0 or index >= len(result):
                raise IndexError(f"click index {index} outside prediction of length {len(result)}")
            result[index] = label
    return result


def _cluster_click_candidates(
    prediction: np.ndarray,
    target: np.ndarray,
    xyz: np.ndarray,
    *,
    method: str = "kdtree",
) -> list[dict[str, Any]]:
    """Return error clusters sorted by AGILE3D size and stable ID."""
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float64)
    if prediction.shape != target.shape or xyz.shape != (len(target), 3):
        raise ValueError("prediction, target, and xyz are not aligned")
    error = prediction != target
    if not bool(error.any()):
        return []
    cluster_ids = target.astype(np.int64) * 96 + prediction.astype(np.int64) * 11
    candidates: list[dict[str, Any]] = []
    for cluster_id in sorted(int(value) for value in np.unique(cluster_ids[error])):
        region = error & (cluster_ids == cluster_id)
        region_indices = np.flatnonzero(region)
        complement_indices = np.flatnonzero(~region)
        if len(complement_indices) == 0:
            distances = np.full(len(region_indices), np.inf)
        elif method == "kdtree":
            distances, _ = cKDTree(xyz[complement_indices]).query(
                xyz[region_indices], k=1, workers=1
            )
        elif method == "dense":
            pairwise = np.linalg.norm(
                xyz[region_indices, None, :] - xyz[complement_indices, :][None, :, :], axis=-1
            )
            distances = pairwise.min(axis=1)
        else:
            raise ValueError("click-center method must be kdtree or dense")
        max_distance = float(np.max(distances))
        farthest = np.flatnonzero(np.isclose(distances, max_distance, rtol=0.0, atol=1e-12))
        center_index = int(region_indices[int(farthest[0])])
        candidates.append(
            {
                "cluster_id": int(cluster_id),
                "target_label": int(target[center_index]),
                "size": max_distance,
                "center_index": center_index,
                "region_points": int(len(region_indices)),
            }
        )
    candidates.sort(key=lambda value: (-float(value["size"]), int(value["cluster_id"]), int(value["center_index"])))
    return candidates


def simulated_corrections(
    prediction: np.ndarray,
    target: np.ndarray,
    xyz: np.ndarray,
    clicks: Mapping[int | str, Sequence[int]],
    click_times: Mapping[int | str, Sequence[int]],
    *,
    training: bool,
    click_center_method: str = "kdtree",
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[dict[str, Any]]]:
    """Select and return the next AGILE3D correction clicks."""
    candidates = _cluster_click_candidates(
        prediction, target, xyz, method=click_center_method
    )
    if not candidates:
        return {}, {}, []
    object_count = len([key for key in clicks if int(key) > 0])
    selected = candidates[:object_count if training else 1]
    total_events = sum(len(values) for values in clicks.values())
    new_clicks: dict[str, list[int]] = {}
    new_times: dict[str, list[int]] = {}
    events: list[dict[str, Any]] = []
    for order, candidate in enumerate(selected):
        label = int(candidate["target_label"])
        key = str(label)
        new_clicks.setdefault(key, []).append(int(candidate["center_index"]))
        new_times.setdefault(key, []).append(int(total_events + order))
        events.append(
            {
                "event": int(total_events + order),
                "label": label,
                "token": int(candidate["center_index"]),
                "cluster_id": int(candidate["cluster_id"]),
                "error_size": float(candidate["size"]),
                "error_points": int(candidate["region_points"]),
            }
        )
    return new_clicks, new_times, events


def append_clicks(
    clicks: Mapping[int | str, Sequence[int]],
    click_times: Mapping[int | str, Sequence[int]],
    new_clicks: Mapping[int | str, Sequence[int]],
    new_times: Mapping[int | str, Sequence[int]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    result = {str(int(key)): [int(value) for value in values] for key, values in clicks.items()}
    times = {str(int(key)): [int(value) for value in values] for key, values in click_times.items()}
    for key, values in new_clicks.items():
        normalized_key = str(int(key))
        result.setdefault(normalized_key, []).extend(int(value) for value in values)
        times.setdefault(normalized_key, []).extend(
            int(value) for value in new_times.get(key, new_times.get(int(key), []))
        )
    result.setdefault("0", [])
    times.setdefault("0", [])
    return result, times


def dense_click_reference(
    prediction: np.ndarray, target: np.ndarray, xyz: np.ndarray
) -> list[dict[str, Any]]:
    return _cluster_click_candidates(prediction, target, xyz, method="dense")


def kdtree_click_reference(
    prediction: np.ndarray, target: np.ndarray, xyz: np.ndarray
) -> list[dict[str, Any]]:
    return _cluster_click_candidates(prediction, target, xyz, method="kdtree")


def token_iou(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    pred_mask = np.asarray(prediction) == int(label)
    target_mask = np.asarray(target) == int(label)
    union = int(np.count_nonzero(pred_mask | target_mask))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred_mask & target_mask) / union)


def raw_object_ious(
    token_prediction: np.ndarray,
    raw_target: np.ndarray,
    inverse_map: np.ndarray,
    object_count: int,
) -> dict[str, float]:
    raw_prediction = np.asarray(token_prediction, dtype=np.int64)[
        np.asarray(inverse_map, dtype=np.int64)
    ]
    return {
        str(object_id): token_iou(raw_prediction, raw_target, object_id)
        for object_id in range(1, int(object_count) + 1)
    }


def panel_click_thresholds(object_count: int, max_clicks_per_object: int = 20) -> list[int]:
    values = [1, 3, 5, 10, 15, 20]
    return [
        int(value * int(object_count))
        for value in values
        if value <= int(max_clicks_per_object)
    ]


def metric_at_threshold(
    states: Sequence[Mapping[str, Any]],
    threshold_total_clicks: int,
) -> Mapping[str, Any]:
    """Select the last state at or below a requested per-object threshold."""
    if not states:
        return {"total_clicks": 0, "clicks_per_object": 0.0, "mean_iou": 0.0, "object_ious": {}}
    eligible = [state for state in states if int(state["total_clicks"]) <= int(threshold_total_clicks)]
    state = eligible[-1] if eligible else states[0]
    return state


def paired_scene_bootstrap(
    public: Mapping[str, Mapping[str, float]],
    delimit3d: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    samples: int = 10000,
    seed: int = 20260911,
) -> dict[str, Any]:
    scenes = sorted(set(public).intersection(delimit3d))
    if not scenes:
        raise ValueError("no paired scenes for bootstrap")
    def value(table: Mapping[str, Mapping[str, float] | float], scene: str) -> float:
        item = table[scene]
        if isinstance(item, Mapping):
            return float(item[metric])
        return float(item)

    public_values = np.asarray([value(public, scene) for scene in scenes], dtype=np.float64)
    delimit_values = np.asarray([value(delimit3d, scene) for scene in scenes], dtype=np.float64)
    differences = delimit_values - public_values
    rng = np.random.default_rng(int(seed))
    draw_indices = rng.integers(0, len(scenes), size=(int(samples), len(scenes)))
    draws = differences[draw_indices].mean(axis=1)
    return {
        "metric": metric,
        "scene_count": len(scenes),
        "samples": int(samples),
        "seed": int(seed),
        "observed_difference": float(differences.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "scene_differences": {
            scene: float(value(delimit3d, scene) - value(public, scene))
            for scene in scenes
        },
    }


def aggregate_panel_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot aggregate an empty panel")
    thresholds = sorted(
        {
            int(key)
            for row in rows
            for key in row.get("thresholds", {}).keys()
        }
    )
    output: dict[str, Any] = {"episodes": len(rows), "thresholds": {}}
    for threshold in thresholds:
        values = [
            row["thresholds"][str(threshold)]["mean_iou"]
            for row in rows
            if str(threshold) in row.get("thresholds", {})
        ]
        if values:
            output["thresholds"][str(threshold)] = {
                "mean_iou": float(np.mean(values)),
                "episode_count": len(values),
            }
    return output


__all__ = [
    "aggregate_panel_rows",
    "append_clicks",
    "array_sha256",
    "build_token_targets",
    "center_click",
    "choose_non_overlapping_group",
    "dense_click_reference",
    "enforce_click_labels",
    "file_sha256",
    "json_sha256",
    "kdtree_click_reference",
    "load_bundle_scene",
    "load_class_mapping",
    "load_training_scene",
    "metric_at_threshold",
    "objects_non_overlapping",
    "paired_scene_bootstrap",
    "panel_click_thresholds",
    "raw_object_ious",
    "scene_seed",
    "simulated_corrections",
    "token_iou",
]

