from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

from delimit3d.source.datasets.scannet.benchmark import (
    SCANNET_EVAL_BENCHMARK_20,
    SCANNET_EVAL_BENCHMARK_200,
    SCANNET_EVAL_BENCHMARK_ALL,
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
    get_valid_class_ids_for_benchmark,
    load_raw_category_label_map,
    normalize_scannet_eval_benchmark,
)
from delimit3d.source.datasets.scannet.metadata import IGNORE_INSTANCE_CLASSES


@dataclass(frozen=True)
class ScannetGTInstances:
    """ScanNet instance ids plus benchmark-contiguous class labels."""

    instance_ids: np.ndarray
    instance_class_ids: dict[int, int]
    segment_ids: np.ndarray | None = None
    vertex_xyz: np.ndarray | None = None


def _is_valid_instance_group_label(
    label: str,
    eval_benchmark: str,
    raw_category_label_map: dict[str, dict[str, int]] | None,
) -> bool:
    normalized_label = str(label).strip().lower()
    if not normalized_label:
        return False

    if normalized_label in IGNORE_INSTANCE_CLASSES:
        return False

    if eval_benchmark == SCANNET_EVAL_BENCHMARK_ALL:
        return True

    if raw_category_label_map is None:
        return False

    label_info = raw_category_label_map.get(normalized_label)
    if label_info is None:
        return False

    valid_class_ids = get_valid_class_ids_for_benchmark(eval_benchmark)
    if valid_class_ids is None:
        return True

    if eval_benchmark == SCANNET_EVAL_BENCHMARK_20:
        return int(label_info["nyu40id"]) in valid_class_ids
    if eval_benchmark == SCANNET_EVAL_BENCHMARK_200:
        return int(label_info["id"]) in valid_class_ids

    return False


def _benchmark_class_id_for_label(
    label: str,
    eval_benchmark: str,
    raw_category_label_map: dict[str, dict[str, int]] | None,
) -> int | None:
    """Return contiguous benchmark class id for a raw ScanNet label."""
    if eval_benchmark == SCANNET_EVAL_BENCHMARK_ALL:
        return None
    if raw_category_label_map is None:
        return None

    normalized_label = str(label).strip().lower()
    label_info = raw_category_label_map.get(normalized_label)
    if label_info is None:
        return None

    if eval_benchmark == SCANNET_EVAL_BENCHMARK_20:
        raw_class_id = int(label_info["nyu40id"])
        if raw_class_id not in VALID_CLASS_IDS_20:
            return None
        return VALID_CLASS_IDS_20.index(raw_class_id)

    if eval_benchmark == SCANNET_EVAL_BENCHMARK_200:
        raw_class_id = int(label_info["id"])
        if raw_class_id not in VALID_CLASS_IDS_200:
            return None
        return VALID_CLASS_IDS_200.index(raw_class_id)

    return None


def _load_instances_from_aggregation(
    scene_dir: Path,
    scene_name: str,
    n_vertices: int,
    eval_benchmark: str,
) -> ScannetGTInstances | None:
    seg_paths = [
        scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_name}_vh_clean.segs.json",
    ]
    agg_paths = [
        scene_dir / f"{scene_name}.aggregation.json",
        scene_dir / f"{scene_name}_vh_clean.aggregation.json",
    ]

    seg_path = next((p for p in seg_paths if p.exists()), None)
    agg_path = next((p for p in agg_paths if p.exists()), None)
    if seg_path is None or agg_path is None:
        return None

    raw_category_label_map = None
    if eval_benchmark != SCANNET_EVAL_BENCHMARK_ALL:
        raw_category_label_map = load_raw_category_label_map()

    with seg_path.open("r", encoding="utf-8") as f:
        seg_json = json.load(f)
    with agg_path.open("r", encoding="utf-8") as f:
        agg_json = json.load(f)

    seg_indices = np.asarray(seg_json.get("segIndices", []), dtype=np.int64)
    if seg_indices.shape[0] != n_vertices:
        raise RuntimeError(
            f"segIndices length ({seg_indices.shape[0]}) != num vertices ({n_vertices})"
        )

    seg_to_instance: dict[int, int] = {}
    instance_class_ids: dict[int, int] = {}
    for group in agg_json.get("segGroups", []):
        label = group.get("label", "")
        if not _is_valid_instance_group_label(
            label=label,
            eval_benchmark=eval_benchmark,
            raw_category_label_map=raw_category_label_map,
        ):
            continue

        raw_inst_id = int(group.get("objectId", group.get("id", -1)))
        if raw_inst_id < 0:
            continue
        # Reserve 0 for ignored/background points so every kept instance is foreground.
        inst_id = raw_inst_id + 1
        class_id = _benchmark_class_id_for_label(
            label=label,
            eval_benchmark=eval_benchmark,
            raw_category_label_map=raw_category_label_map,
        )
        if eval_benchmark != SCANNET_EVAL_BENCHMARK_ALL and class_id is None:
            continue
        if class_id is not None:
            instance_class_ids[inst_id] = int(class_id)

        for seg_id in group.get("segments", []):
            seg_to_instance[int(seg_id)] = inst_id

    gt_instance_ids = np.zeros(n_vertices, dtype=np.int64)
    for i, seg_id in enumerate(seg_indices):
        gt_instance_ids[i] = seg_to_instance.get(int(seg_id), 0)

    return ScannetGTInstances(
        instance_ids=gt_instance_ids,
        instance_class_ids=instance_class_ids,
        segment_ids=seg_indices,
    )


def load_scannet_segment_ids(scene_dir: Path, scene_name: str) -> np.ndarray:
    """Load mesh-aligned ScanNet oversegmentation ids.

    The length is checked against the labeled mesh vertex array so callers can
    safely preserve the ids through augmentation and point subsampling.
    """
    labels_ply = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    if not labels_ply.exists():
        raise FileNotFoundError(f"Missing ScanNet labels ply: {labels_ply}")
    n_vertices = len(PlyData.read(str(labels_ply)).elements[0].data)
    candidates = (
        scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_name}_vh_clean.segs.json",
    )
    segment_path = next((path for path in candidates if path.exists()), None)
    if segment_path is None:
        raise FileNotFoundError(
            f"Missing ScanNet segment json for {scene_name}: {list(candidates)}"
        )
    with segment_path.open("r", encoding="utf-8") as handle:
        segment_ids = np.asarray(
            json.load(handle).get("segIndices", []), dtype=np.int64
        )
    if segment_ids.shape != (n_vertices,):
        raise RuntimeError(
            f"segIndices shape {segment_ids.shape} != mesh vertices ({n_vertices},)"
        )
    return segment_ids


def load_scannet_gt_instance_ids(
    scene_dir: Path,
    scene_name: str,
    eval_benchmark: str = SCANNET_EVAL_BENCHMARK_ALL,
) -> np.ndarray:
    eval_benchmark = normalize_scannet_eval_benchmark(eval_benchmark)
    labels_ply = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    if not labels_ply.exists():
        raise FileNotFoundError(f"Missing ScanNet labels ply: {labels_ply}")

    plydata = PlyData.read(str(labels_ply))
    n_vertices = len(plydata.elements[0].data)

    gt_instances = _load_instances_from_aggregation(
        scene_dir,
        scene_name,
        n_vertices,
        eval_benchmark=eval_benchmark,
    )
    if gt_instances is not None:
        return gt_instances.instance_ids

    raise RuntimeError("Could not find GT instance ids in aggregation+segments files.")


def load_scannet_gt_instances(
    scene_dir: Path,
    scene_name: str,
    eval_benchmark: str = SCANNET_EVAL_BENCHMARK_ALL,
) -> ScannetGTInstances:
    """Load ScanNet GT instances and contiguous benchmark class ids.

    ``instance_ids`` follows the existing convention: 0 is ignored/background,
    foreground ids are positive. ``instance_class_ids`` maps those positive ids
    to contiguous class indices for the requested benchmark.
    """
    eval_benchmark = normalize_scannet_eval_benchmark(eval_benchmark)
    labels_ply = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    if not labels_ply.exists():
        raise FileNotFoundError(f"Missing ScanNet labels ply: {labels_ply}")

    plydata = PlyData.read(str(labels_ply))
    n_vertices = len(plydata.elements[0].data)

    gt_instances = _load_instances_from_aggregation(
        scene_dir,
        scene_name,
        n_vertices,
        eval_benchmark=eval_benchmark,
    )
    if gt_instances is not None:
        vertices = plydata.elements[0].data
        vertex_xyz = np.stack(
            [vertices[axis] for axis in ("x", "y", "z")], axis=1
        ).astype(np.float32, copy=False)
        return ScannetGTInstances(
            instance_ids=gt_instances.instance_ids,
            instance_class_ids=gt_instances.instance_class_ids,
            segment_ids=gt_instances.segment_ids,
            vertex_xyz=vertex_xyz,
        )

    raise RuntimeError("Could not find GT instance ids in aggregation+segments files.")
