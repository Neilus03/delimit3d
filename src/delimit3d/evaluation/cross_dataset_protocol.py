"""Official AGILE3D accounting for S3DIS and KITTI-360.

The released AGILE3D PLY label field remains the ground-truth authority.
This module keeps that accounting independent of MinkowskiEngine and any
particular encoder. Normal arrays are separate, row-aligned assets. MO
records retain the official local-to-instance mapping and SO records retain
the original object identity/class.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from .agile3d_protocol import file_sha256, json_sha256, kdtree_click_reference

DATASETS = ("S3DIS", "KITTI360")
CLICKS = (1, 3, 5, 10, 15, 20)
THRESHOLDS = (50, 65, 80, 85, 90)
MAX_CLICKS_PER_OBJECT = 20
SCHEMA = "delimit3d_agile3d_cross_dataset/v1"


def normalize_dataset(dataset: str) -> str:
    """Return the canonical AGILE3D dataset name."""
    value = str(dataset).strip().upper().replace("-", "").replace("_", "")
    if value == "S3DIS":
        return "S3DIS"
    if value in {"KITTI360", "KITTI"}:
        return "KITTI360"
    raise ValueError(f"unsupported cross-dataset protocol: {dataset!r}")


def native_cuda_representative_indices(
    points: np.ndarray,
    *,
    voxel_size: float = 0.02,
) -> np.ndarray:
    """Mirror LitePT CUDA float32 scalar-division representative sampling."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("points must be a non-empty finite Nx3 array")
    if not np.isfinite(values).all():
        raise ValueError("points must be a non-empty finite Nx3 array")
    size = np.float32(voxel_size)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    reciprocal = np.float32(1.0) / size
    grid = np.floor(values * reciprocal).astype(np.int64)
    _unique, representatives = np.unique(grid, axis=0, return_index=True)
    return representatives.astype(np.int64, copy=False)


representative_indices_for_points = native_cuda_representative_indices


def _coerce_integral(value: Any, *, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(numeric)


def _path(value: str | Path, resolver: Callable[[Any], Any] | None = None) -> Path:
    result = resolver(value) if resolver is not None else value
    return Path(result).expanduser()


def _existing_file(
    value: str | Path,
    *,
    resolver: Callable[[Any], Any] | None = None,
    label: str,
) -> Path:
    result = _path(value, resolver)
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _scene_and_count(official_key: str) -> tuple[str, int]:
    key = str(official_key)
    if "_obj_" not in key:
        raise ValueError(f"official multi-object key lacks _obj_: {official_key!r}")
    scene, count_text = key.rsplit("_obj_", 1)
    if not scene:
        raise ValueError(f"official multi-object key has an empty scene: {official_key!r}")
    count = _coerce_integral(count_text, field=f"{official_key}.object_count")
    if count <= 0:
        raise ValueError(f"official multi-object count must be positive: {official_key!r}")
    return scene, count


def official_scene_from_key(official_key: str) -> str:
    """Extract the exact scene stem from an AGILE3D MO key."""
    return _scene_and_count(official_key)[0]


def _normalise_clicks(raw: Any, count: int, *, key: str) -> dict[str, list[int]]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key}.clicks must be a mapping")
    expected = {"0", *[str(index) for index in range(1, count + 1)]}
    unknown = set(str(name) for name in raw) - expected
    if unknown:
        raise ValueError(f"{key}.clicks has unknown local labels: {sorted(unknown)}")
    result: dict[str, list[int]] = {}
    for label in ["0", *[str(index) for index in range(1, count + 1)]]:
        values = raw.get(label, raw.get(int(label), []))
        if values is None:
            values = []
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"{key}.clicks[{label}] must be a sequence")
        converted = [_coerce_integral(value, field=f"{key}.clicks[{label}]") for value in values]
        if any(value < 0 for value in converted):
            raise ValueError(f"{key}.clicks[{label}] contains a negative index")
        result[label] = converted
    return result


def read_official_multi_list(
    path: str | Path,
    *,
    resolver: Callable[[Any], Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read and canonicalise official val_list.json multi-object entries.

    Entries map local labels 1..K to original PLY instance IDs. The returned
    mapping is sorted by exact official key and uses integer IDs.
    """
    source = _existing_file(path, resolver=resolver, label="multi-object list")
    try:
        raw = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"official multi-object list is empty or not a mapping: {source}")

    result: dict[str, dict[str, Any]] = {}
    for official_key in sorted((str(key) for key in raw)):
        scene, count = _scene_and_count(official_key)
        entry = raw[official_key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{official_key} must map to an object")
        raw_objects = entry.get("obj")
        if not isinstance(raw_objects, Mapping):
            raise ValueError(f"{official_key}.obj must be a mapping")
        expected_labels = [str(index) for index in range(1, count + 1)]
        if set(str(label) for label in raw_objects) != set(expected_labels):
            raise ValueError(f"{official_key}.obj must contain exactly local labels 1..{count}")
        objects: dict[str, int] = {}
        for local_label in expected_labels:
            instance = _coerce_integral(
                raw_objects.get(local_label, raw_objects.get(int(local_label))),
                field=f"{official_key}.obj[{local_label}]",
            )
            if instance <= 0:
                raise ValueError(f"{official_key}.obj[{local_label}] must be positive")
            objects[local_label] = instance
        if len(set(objects.values())) != count:
            raise ValueError(f"{official_key}.obj contains duplicate official instance IDs")
        result[official_key] = {
            "official_key": official_key,
            "scene": scene,
            "object_count": count,
            "obj": objects,
            "clicks": _normalise_clicks(entry.get("clicks"), count, key=official_key),
        }
    return result


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_official_single_list(
    object_ids_path: str | Path,
    object_classes_path: str | Path | None = None,
    *,
    resolver: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    """Read official single-object IDs/classes while retaining row order."""
    ids_source = _existing_file(object_ids_path, resolver=resolver, label="single-object IDs")
    try:
        values = np.asarray(np.load(ids_source, allow_pickle=False))
    except (ValueError, OSError) as exc:
        raise ValueError(f"cannot load official single-object IDs: {ids_source}") from exc
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"{ids_source} must be a non-empty [N,2] array")

    classes: list[str] | None = None
    classes_source: Path | None = None
    if object_classes_path is not None:
        classes_source = _existing_file(
            object_classes_path, resolver=resolver, label="single-object classes"
        )
        classes = [line.strip() for line in classes_source.read_text().splitlines()]
        if len(classes) != len(values):
            raise ValueError(
                f"single-object class count {len(classes)} differs from ID count {len(values)}"
            )
        if any(not value for value in classes):
            raise ValueError(f"single-object classes contain an empty class: {classes_source}")

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row_index, row in enumerate(values):
        scene = _decode_text(row[0]).strip()
        object_id_text = _decode_text(row[1]).strip()
        if not scene:
            raise ValueError(f"single-object row {row_index} has an empty scene")
        instance = _coerce_integral(object_id_text, field=f"single-object row {row_index}")
        if instance < 0:
            raise ValueError(f"single-object row {row_index} has a negative object ID")
        identity = (scene, instance)
        if identity in seen:
            raise ValueError(f"duplicate single-object identity: {scene}/{instance}")
        seen.add(identity)
        result.append(
            {
                "order": row_index,
                "scene": scene,
                "instance": instance,
                "object_id": object_id_text,
                "semantic_name": classes[row_index] if classes is not None else None,
            }
        )
    return result



def _normal_candidates(
    root: Path,
    scene: str,
    *,
    split: str | None = None,
    object_id: int | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if object_id is not None:
        crop = f"{scene}_crop_{int(object_id)}"
        candidates.extend(
            [
                root / f"{crop}.npy",
                root / f"{crop}.normal.npy",
                root / crop / "normal.npy",
                root / scene / crop / "normal.npy",
                root / crop / "normals.npy",
                root / scene / crop / "normals.npy",
            ]
        )
        if split:
            candidates.extend(
                [
                    root / split / crop / "normal.npy",
                    root / split / scene / crop / "normal.npy",
                    root / split / crop / "normals.npy",
                    root / split / scene / crop / "normals.npy",
                ]
            )
    candidates.extend(
        [
            root / f"{scene}.npy",
            root / f"{scene}.normal.npy",
            root / scene / "normal.npy",
        ]
    )
    if split:
        candidates.append(root / split / scene / "normal.npy")
    for split_name in ("train", "val", "test"):
        candidates.append(root / split_name / scene / "normal.npy")
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def find_aligned_normal_path(
    normal_root: str | Path,
    scene: str,
    *,
    split: str | None = None,
    object_id: int | None = None,
    resolver: Callable[[Any], Any] | None = None,
) -> Path:
    """Find one row-aligned normal sidecar for a scene or crop."""
    root = _path(normal_root, resolver)
    candidates = [
        _path(candidate, resolver)
        for candidate in _normal_candidates(root, scene, split=split, object_id=object_id)
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    suffix = f"/{object_id}" if object_id is not None else ""
    raise FileNotFoundError(f"no aligned normal sidecar for {scene}{suffix}; searched: {searched}")


def load_aligned_normals(
    normal_root: str | Path | None,
    scene: str,
    expected_points: int,
    *,
    normal_path: str | Path | None = None,
    expected_sha256: str | None = None,
    split: str | None = None,
    object_id: int | None = None,
    resolver: Callable[[Any], Any] | None = None,
) -> np.ndarray:
    """Load a finite N by 3 normal array without changing row order."""
    if normal_path is None:
        if normal_root is None:
            raise ValueError("aligned normals require normal_root or normal_path")
        path = find_aligned_normal_path(
            normal_root, scene, split=split, object_id=object_id, resolver=resolver
        )
    else:
        path = _existing_file(normal_path, resolver=resolver, label="normal sidecar")
    if expected_sha256 is not None and file_sha256(path) != str(expected_sha256):
        raise ValueError(f"{scene}: normal sidecar hash mismatch: {path}")
    try:
        normals = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{scene}: cannot load normal sidecar {path}") from exc
    if normals.shape != (int(expected_points), 3):
        raise ValueError(
            f"{scene}: normal sidecar has shape {normals.shape}, expected {(int(expected_points), 3)}"
        )
    if not np.isfinite(normals).all():
        raise ValueError(f"{scene}: normal sidecar contains non-finite values: {path}")
    return normals


def _asset_record(
    *,
    dataset: str,
    split: str,
    scene: str,
    data_path: Path,
    normal_path: Path | None,
    mode: str,
    resolver: Callable[[Any], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data_path = _existing_file(data_path, resolver=resolver, label=f"{scene} PLY")
    if normal_path is not None:
        normal_path = _existing_file(
            normal_path, resolver=resolver, label=f"{scene} normal sidecar"
        )
    source_hashes = {str(data_path): file_sha256(data_path)}
    result: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "scene": scene,
        "mode": mode,
        "data_path": str(data_path),
        "data_sha256": source_hashes[str(data_path)],
        "normal_alignment_verified": normal_path is not None,
        "labels_authority": "official_AGILE3D_PLY_label",
        "source_hashes": source_hashes,
    }
    if normal_path is not None:
        result["normal_path"] = str(normal_path)
        result["normal_sha256"] = file_sha256(normal_path)
        source_hashes[str(normal_path)] = result["normal_sha256"]
    if metadata:
        result.update(dict(metadata))
    return result


def _normal_for_record(
    normal_root: str | Path | None,
    scene: str,
    *,
    split: str,
    object_id: int | None,
    normal_paths: Mapping[str, str | Path] | None,
    resolver: Callable[[Any], Any] | None,
) -> Path | None:
    if normal_paths:
        keys = [scene]
        if object_id is not None:
            keys = [
                f"{scene}/{int(object_id)}",
                f"{scene}_obj_{int(object_id)}",
                f"{scene}_crop_{int(object_id)}",
                scene,
            ]
        for key in keys:
            if key in normal_paths:
                return _existing_file(
                    normal_paths[key], resolver=resolver, label=f"{scene} normal sidecar"
                )
    if normal_root is None:
        return None
    return find_aligned_normal_path(
        normal_root, scene, split=split, object_id=object_id, resolver=resolver
    )


def build_official_records(
    official_root: str | Path,
    *,
    dataset: str,
    normal_root: str | Path | None = None,
    normal_paths: Mapping[str, str | Path] | None = None,
    split: str = "val",
    multi_list_path: str | Path | None = None,
    single_object_ids_path: str | Path | None = None,
    single_object_classes_path: str | Path | None = None,
    include_single: bool = True,
    single_crop: bool = True,
    require_normals: bool = True,
    resolver: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic metadata records for official MO and SO units.

    MO paths are official_root/scans/scene.ply. SO paths are official binary
    crops when single_crop is true. For crop-specific sidecars, normal_paths
    can use keys scene/object_id.
    """
    dataset_name = normalize_dataset(dataset)
    root = _path(official_root, resolver)
    if not root.is_dir():
        raise FileNotFoundError(f"official data root does not exist: {root}")
    multi_path = (
        _path(multi_list_path, resolver)
        if multi_list_path is not None
        else root / "val_list.json"
    )
    multi = read_official_multi_list(multi_path, resolver=resolver)

    ids_path = (
        _path(single_object_ids_path, resolver)
        if single_object_ids_path is not None
        else root / "single" / "object_ids.npy"
    )
    classes_path = (
        _path(single_object_classes_path, resolver)
        if single_object_classes_path is not None
        else root / "single" / "object_classes.txt"
    )
    single: list[dict[str, Any]] = []
    if include_single:
        single = read_official_single_list(ids_path, classes_path, resolver=resolver)

    asset_by_scene: dict[str, dict[str, Any]] = {}

    def scene_asset(scene: str) -> dict[str, Any]:
        if scene in asset_by_scene:
            return asset_by_scene[scene]
        normal = _normal_for_record(
            normal_root,
            scene,
            split=split,
            object_id=None,
            normal_paths=normal_paths,
            resolver=resolver,
        )
        if require_normals and normal is None:
            raise FileNotFoundError(f"{scene}: aligned normal sidecar is required")
        asset = _asset_record(
            dataset=dataset_name,
            split=split,
            scene=scene,
            data_path=root / "scans" / f"{scene}.ply",
            normal_path=normal,
            mode="SCENE",
            resolver=resolver,
        )
        asset_by_scene[scene] = asset
        return asset

    mo_records: list[dict[str, Any]] = []
    for index, (official_key, entry) in enumerate(multi.items()):
        asset = scene_asset(str(entry["scene"]))
        objects = [
            {
                "local_id": int(local_id),
                "instance": int(entry["obj"][local_id]),
                "semantic_name": None,
                "semantic_class": -1,
            }
            for local_id in sorted(entry["obj"], key=lambda value: int(value))
        ]
        record = dict(asset)
        record.update(
            {
                "mode": "MO",
                "official_index": index,
                "official_key": official_key,
                "object_count": int(entry["object_count"]),
                "object_ids": [item["instance"] for item in objects],
                "local_to_official": {
                    str(item["local_id"]): int(item["instance"]) for item in objects
                },
                "objects": objects,
                "official_click_scaffold": entry["clicks"],
                "official_multi_list_path": str(multi_path),
            }
        )
        mo_records.append(record)

    so_records: list[dict[str, Any]] = []
    if include_single:
        for item in single:
            scene = str(item["scene"])
            instance = int(item["instance"])
            if single_crop:
                data_path = root / "single" / "crops" / scene / f"{scene}_crop_{instance}.ply"
                role = "single_crop"
            else:
                data_path = root / "scans" / f"{scene}.ply"
                role = "single_full_scene"
            normal = _normal_for_record(
                normal_root,
                scene,
                split=split,
                object_id=instance if single_crop else None,
                normal_paths=normal_paths,
                resolver=resolver,
            )
            if require_normals and normal is None:
                raise FileNotFoundError(
                    f"{scene}/{instance}: aligned normal sidecar is required"
                )
            asset = _asset_record(
                dataset=dataset_name,
                split=split,
                scene=scene,
                data_path=data_path,
                normal_path=normal,
                mode="SO",
                resolver=resolver,
                metadata={"data_role": role},
            )
            record = dict(asset)
            record.update(
                {
                    "official_index": int(item["order"]),
                    "object_count": 1,
                    "object_ids": [instance],
                    "instance": instance,
                    "object_id": str(item["object_id"]),
                    "semantic_name": item["semantic_name"],
                    "semantic_class": -1,
                    "objects": [
                        {
                            "local_id": 1,
                            "instance": instance,
                            "semantic_name": item["semantic_name"],
                            "semantic_class": -1,
                        }
                    ],
                    "official_single_ids_path": str(ids_path),
                    "official_single_classes_path": str(classes_path),
                }
            )
            so_records.append(record)

    official_files: dict[str, dict[str, str]] = {
        "multi_list": {"path": str(multi_path), "sha256": file_sha256(multi_path)}
    }
    if include_single:
        official_files["single_object_ids"] = {
            "path": str(ids_path),
            "sha256": file_sha256(ids_path),
        }
        official_files["single_object_classes"] = {
            "path": str(classes_path),
            "sha256": file_sha256(classes_path),
        }

    scene_records = []
    for scene in sorted(asset_by_scene):
        asset = dict(asset_by_scene[scene])
        asset["official_object_ids"] = sorted(
            {
                int(record_id)
                for record in mo_records
                if record["scene"] == scene
                for record_id in record["object_ids"]
            }
        )
        scene_records.append(asset)

    records = {
        "schema": SCHEMA,
        "dataset": dataset_name,
        "split": split,
        "official_files": official_files,
        "scenes": scene_records,
        "MO": mo_records,
        "SO": so_records,
    }
    records["records_sha256"] = json_sha256(records)
    return records


build_official_scene_records = build_official_records



def object_records_from_labels(
    labels: np.ndarray,
    scene: str,
    *,
    semantic_by_object: Mapping[tuple[str, int], str] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted official object summaries without changing label IDs."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("labels must be a non-empty vector")
    lookup = semantic_by_object or {}
    result: list[dict[str, Any]] = []
    ids, counts = np.unique(values, return_counts=True)
    for instance, count in zip(ids, counts, strict=True):
        instance = int(instance)
        if instance <= 0:
            continue
        result.append(
            {
                "instance": instance,
                "instance_points": int(count),
                "semantic_name": lookup.get((str(scene), instance)),
                "semantic_class": -1,
            }
        )
    return result


def objects_from_official_labels(
    labels: np.ndarray,
    scene: str,
    semantic_by_object: Mapping[tuple[str, int], str] | None = None,
) -> list[dict[str, Any]]:
    """Return raw positive official labels in deterministic instance order."""
    return object_records_from_labels(
        labels, scene, semantic_by_object=semantic_by_object
    )


def load_scene_with_labels(
    record: Mapping[str, Any],
    resolver: Callable[[Any], Any] | None = None,
    *,
    require_normals: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[int, np.ndarray],
]:
    """Load RGB, aligned normals, and the untouched official PLY labels."""
    from plyfile import PlyData

    data_path = _existing_file(record["data_path"], resolver=resolver, label="official PLY")
    expected_data_hash = record.get("data_sha256")
    if expected_data_hash is not None and file_sha256(data_path) != str(expected_data_hash):
        raise ValueError(f"{record.get('scene', '<scene>')}: official PLY hash mismatch")
    vertex = PlyData.read(data_path)["vertex"].data
    required = ("x", "y", "z", "R", "G", "B", "label")
    missing = [name for name in required if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"{data_path}: missing official PLY fields {missing}")
    points = np.stack(
        [np.asarray(vertex[name]) for name in ("x", "y", "z")], axis=1
    ).astype(np.float32)
    if points.ndim != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"{data_path}: invalid XYZ geometry")
    colors = np.stack(
        [np.asarray(vertex[name]) for name in ("R", "G", "B")], axis=1
    ).astype(np.float32)
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError(f"{data_path}: invalid RGB fields")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError(f"{data_path}: RGB values must be in [0,255]")
    colors /= 255.0
    labels = np.asarray(vertex["label"], dtype=np.int64).copy()

    normal_path = record.get("normal_path")
    if require_normals and not normal_path:
        raise ValueError(f"{record.get('scene', '<scene>')}: aligned normal sidecar required")
    expected_normal_hash = record.get("normal_sha256")
    if expected_normal_hash is None and normal_path:
        expected_normal_hash = record.get("source_hashes", {}).get(str(normal_path))
    normals = (
        load_aligned_normals(
            None,
            str(record.get("scene", "")),
            len(points),
            normal_path=normal_path,
            expected_sha256=expected_normal_hash,
            resolver=resolver,
        )
        if normal_path
        else np.zeros_like(points, dtype=np.float32)
    )

    mode = str(record.get("mode", "SCENE")).upper()
    object_ids = [
        _coerce_integral(value, field="record.object_ids")
        for value in record.get("object_ids", [])
    ]
    if mode == "SO":
        if len(object_ids) != 1:
            raise ValueError("SO record must contain one original object ID")
        instance = int(object_ids[0])
        target_mask = (
            labels > 0
            if str(record.get("data_role", "")) == "single_crop"
            else labels == instance
        )
        masks = {instance: np.flatnonzero(target_mask).astype(np.int64, copy=False)}
        if masks[instance].size == 0:
            raise ValueError(f"{record.get('scene', '<scene>')}/{instance}: object absent from PLY")
    elif object_ids:
        masks = {
            instance: np.flatnonzero(labels == instance).astype(np.int64, copy=False)
            for instance in object_ids
        }
        missing_ids = [instance for instance, indices in masks.items() if len(indices) == 0]
        if missing_ids:
            raise ValueError(
                f"{record.get('scene', '<scene>')}: official objects absent from PLY: {missing_ids}"
            )
    else:
        masks = {
            int(instance): np.flatnonzero(labels == int(instance)).astype(np.int64, copy=False)
            for instance in np.unique(labels)
            if int(instance) > 0
        }

    semantic_lookup = {
        int(item["instance"]): item.get("semantic_name")
        for item in record.get("objects", [])
        if "instance" in item
    }
    objects = [
        {
            "instance": int(instance),
            "instance_points": int(len(masks[instance])),
            "semantic_name": semantic_lookup.get(instance),
            "semantic_class": -1,
        }
        for instance in sorted(masks)
    ]
    return points, colors, normals, labels, objects, masks


def load_scene(
    record: Mapping[str, Any],
    resolver: Callable[[Any], Any] | None = None,
    *,
    require_normals: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[int, np.ndarray]]:
    """Compatibility wrapper matching the ScanNet40 protocol return shape."""
    points, colors, normals, _labels, objects, masks = load_scene_with_labels(
        record, resolver=resolver, require_normals=require_normals
    )
    return points, colors, normals, objects, masks


def target_from_official_labels(
    labels: np.ndarray,
    object_ids: Sequence[int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Map official IDs to local episode labels and return the mapping."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("official labels must be a non-empty vector")
    requested = [_coerce_integral(value, field="object_ids") for value in object_ids]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("object_ids must be a non-empty sequence of unique IDs")
    target = np.zeros(values.shape, dtype=np.int64)
    mapping: dict[str, int] = {}
    for local_id, instance in enumerate(requested, start=1):
        mask = values == instance
        if not bool(mask.any()):
            raise ValueError(f"official object {instance} is absent from labels")
        target[mask] = local_id
        mapping[str(local_id)] = instance
    return target, mapping


def initial_clicks(
    target: np.ndarray,
    xyz: np.ndarray,
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Choose deepest-interior initial clicks using a deterministic order."""
    target = np.asarray(target, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float32)
    if target.ndim != 1 or xyz.shape != (len(target), 3):
        raise ValueError("target and xyz are not aligned")
    if len(target) == 0 or not np.isfinite(xyz).all():
        raise ValueError("target/xyz must be non-empty and finite")
    labels = sorted(int(value) for value in np.unique(target) if int(value) > 0)
    if not labels:
        raise ValueError("initial clicks require at least one foreground object")
    candidates = kdtree_click_reference(np.zeros_like(target), target, xyz)
    by_label: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        by_label.setdefault(int(candidate["target_label"]), candidate)
    if set(labels) != set(by_label):
        raise ValueError("initial clicks require every requested object to be present")
    order = list(labels)
    random.Random(int(seed)).shuffle(order)
    clicks = {"0": []}
    times = {"0": []}
    for event, label in enumerate(order):
        clicks[str(label)] = [int(by_label[label]["center_index"])]
        times[str(label)] = [event]
    return clicks, times


def _trace_count(trace: Mapping[str, Any]) -> int:
    count = trace.get("object_count", trace.get("requested_count"))
    if count is None:
        count = len(trace.get("object_ids", []))
    count = _coerce_integral(count, field="trace.object_count")
    if count <= 0:
        raise ValueError("trace.object_count must be positive")
    return count


def _trace_states(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_states = trace.get("states")
    if (
        not isinstance(raw_states, Sequence)
        or isinstance(raw_states, (str, bytes))
        or not raw_states
    ):
        raise ValueError("trace.states must be a non-empty sequence")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    previous = -1
    for raw in raw_states:
        if not isinstance(raw, Mapping):
            raise ValueError("trace states must be mappings")
        total = _coerce_integral(raw.get("total_clicks"), field="state.total_clicks")
        if total < 0 or total in seen or total < previous:
            raise ValueError("trace total_clicks must be sorted, unique, and non-negative")
        previous = total
        seen.add(total)
        if "mean_iou" in raw:
            mean = float(raw["mean_iou"])
        elif isinstance(raw.get("object_ious"), Mapping) and raw["object_ious"]:
            mean = float(np.mean([float(value) for value in raw["object_ious"].values()]))
        else:
            raise ValueError("state requires mean_iou or object_ious")
        if not np.isfinite(mean) or mean < 0.0 or mean > 1.0:
            raise ValueError("state.mean_iou must be finite and within [0,1]")
        item = dict(raw)
        item["total_clicks"] = total
        item["mean_iou"] = mean
        result.append(item)
    return result


def metrics_from_trace(
    trace: Mapping[str, Any],
    *,
    max_clicks_per_object: int = MAX_CLICKS_PER_OBJECT,
) -> dict[str, float]:
    """Compute IoU at every official click budget and NoC thresholds."""
    max_clicks = _coerce_integral(max_clicks_per_object, field="max_clicks_per_object")
    if max_clicks <= 0:
        raise ValueError("max_clicks_per_object must be positive")
    count = _trace_count(trace)
    states = _trace_states(trace)
    by_total = {int(state["total_clicks"]): state for state in states}
    metrics: dict[str, float] = {}
    for click in (value for value in CLICKS if value <= max_clicks):
        total = int(click * count)
        if total not in by_total:
            raise ValueError(f"missing exact click budget {click} for {count} objects")
        metrics[f"IoU@{click}"] = float(by_total[total]["mean_iou"])
    max_total = int(max_clicks * count)
    for threshold in THRESHOLDS:
        reached = [
            int(state["total_clicks"]) / count
            for state in states
            if int(state["total_clicks"]) <= max_total
            and float(state["mean_iou"]) >= threshold / 100.0
        ]
        metrics[f"NoC@{threshold}"] = float(min(reached)) if reached else float(max_clicks)
    return metrics


def _trace_identity(trace: Mapping[str, Any], mode: str) -> int:
    if mode == "MO":
        return _trace_count(trace)
    # SO output identity is the original object_ids.npy crop index. It can
    # differ from the binary crop target label, which is always 1.
    official = trace.get("official_object_index")
    if official is not None:
        return _coerce_integral(official, field="trace.official_object_index")
    official = trace.get("official_object_id")
    if official is not None:
        return _coerce_integral(official, field="trace.official_object_id")
    ids = trace.get("object_ids")
    if ids is None:
        ids = [trace.get("object_id")]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or len(ids) != 1:
        raise ValueError("SO trace must carry exactly one original object ID")
    return _coerce_integral(ids[0], field="trace.object_id")


def official_lines(index: int, trace: Mapping[str, Any], mode: str) -> list[str]:
    """Serialize trace states as official AGILE3D result lines."""
    normalized_mode = str(mode).upper()
    if normalized_mode not in {"MO", "SO"}:
        raise ValueError("mode must be MO or SO")
    index = _coerce_integral(index, field="result index")
    if index < 0:
        raise ValueError("result index must be non-negative")
    count = _trace_count(trace)
    identity = _trace_identity(trace, normalized_mode)
    scene = str(trace.get("official_scene", trace.get("scene", ""))).replace("scene", "")
    lines: list[str] = []
    for state in _trace_states(trace):
        total = int(state["total_clicks"])
        clicks = str(float(total / count)) if normalized_mode == "MO" else str(total)
        lines.append(f"{index} {scene} {identity} {clicks} {str(float(state['mean_iou']))}\n")
    return lines


def paired_bootstrap(
    public: Mapping[str, Mapping[str, float] | float],
    adapted: Mapping[str, Mapping[str, float] | float],
    samples: int = 10000,
    seed: int = 20260912,
) -> dict[str, dict[str, float | int]]:
    """Compute paired bootstrap deltas over exactly matched identities."""
    if not public or not adapted or set(public) != set(adapted):
        raise ValueError("paired identities differ or are empty")
    sample_count = _coerce_integral(samples, field="bootstrap.samples")
    if sample_count <= 0:
        raise ValueError("bootstrap.samples must be positive")
    keys = sorted(public, key=lambda value: str(value))
    metric_names = [f"IoU@{click}" for click in CLICKS] + [
        f"NoC@{threshold}" for threshold in THRESHOLDS
    ]
    rng = np.random.default_rng(int(seed))

    def value(row: Mapping[str, float] | float, metric: str, key: Any) -> float:
        if isinstance(row, Mapping):
            if metric not in row:
                raise ValueError(f"paired row {key!r} is missing {metric}")
            raw = row[metric]
        else:
            raise ValueError(f"paired row {key!r} is scalar but {metric} was requested")
        result = float(raw)
        if not np.isfinite(result):
            raise ValueError(f"paired row {key!r} has non-finite {metric}")
        return result

    output: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        differences = np.asarray(
            [value(adapted[key], metric, key) - value(public[key], metric, key) for key in keys],
            dtype=np.float64,
        )
        draws = np.empty(sample_count, dtype=np.float64)
        for start in range(0, sample_count, 128):
            end = min(start + 128, sample_count)
            indices = rng.integers(0, len(keys), size=(end - start, len(keys)))
            draws[start:end] = differences[indices].mean(axis=1)
        output[metric] = {
            "delta": float(differences.mean()),
            "ci95_lower": float(np.quantile(draws, 0.025)),
            "ci95_upper": float(np.quantile(draws, 0.975)),
            "paired_units": len(keys),
            "samples": sample_count,
            "seed": int(seed),
        }
    return output


__all__ = [
    "CLICKS",
    "DATASETS",
    "MAX_CLICKS_PER_OBJECT",
    "SCHEMA",
    "THRESHOLDS",
    "build_official_records",
    "file_sha256",
    "build_official_scene_records",
    "find_aligned_normal_path",
    "initial_clicks",
    "json_sha256",
    "load_aligned_normals",
    "load_scene",
    "load_scene_with_labels",
    "metrics_from_trace",
    "native_cuda_representative_indices",
    "normalize_dataset",
    "object_records_from_labels",
    "objects_from_official_labels",
    "official_lines",
    "official_scene_from_key",
    "paired_bootstrap",
    "read_official_multi_list",
    "representative_indices_for_points",
    "read_official_single_list",
    "target_from_official_labels",
]

