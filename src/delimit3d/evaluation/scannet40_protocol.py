"""Official ScanNet40 identity and result accounting, independent of backbone.

The data and metric protocol follow AGILE3D b73638da41edbabe52a1b578d52ddeb8fa552173.
Features retain the native LitePT RGBN6/2cm/representative-first contract.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .agile3d_protocol import file_sha256, kdtree_click_reference

CLICKS = (1, 3, 5, 10, 15)
THRESHOLDS = (50, 65, 80, 85, 90)
EXPERIMENT = "delimit3d_scannet40_agile3d_matched_v1"


def read_official_list(path):
    data = json.loads(Path(path).read_text())
    scenes = [key.rsplit("_obj_", 1)[0] for key in data]
    if not data or len(scenes) != len(set(scenes)):
        raise ValueError("official scene manifest empty or has duplicate scenes")
    return data


def load_scene(record, resolve=lambda value: Path(value)):
    """Read official PLY order; normals are supplied by verified aligned arrays."""
    from plyfile import PlyData
    path = resolve(record["data_path"])
    if file_sha256(path) != record["data_sha256"]:raise ValueError(f"{record['scene']}: official PLY changed")
    normal_path=resolve(record["normal_path"])
    if file_sha256(normal_path) != record["source_hashes"][record["normal_path"]]:raise ValueError("normal source drift")
    ply = PlyData.read(path)["vertex"].data
    points = np.stack([ply[k] for k in ("x", "y", "z")], 1).astype(np.float32)
    rgb = np.stack([ply[k] for k in ("R", "G", "B")], 1).astype(np.float32) / 255.0
    labels = np.asarray(ply["label"], dtype=np.int64)
    normals = np.asarray(np.load(resolve(record["normal_path"])), dtype=np.float32)
    if normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError(f"{record['scene']}: malformed normals")
    objects = list(record["objects"])
    masks = {int(o["instance"]): np.flatnonzero(labels == int(o["instance"])) for o in objects}
    if any(len(v) == 0 for v in masks.values()):
        raise ValueError(f"{record['scene']}: absent official object")
    return points, rgb, normals, objects, masks


def objects_from_official_labels(labels,scene,semantic_by_object=None):
    semantic_by_object=semantic_by_object or {}
    ids,counts=np.unique(np.asarray(labels,dtype=np.int64),return_counts=True)
    return [{"instance":int(obj),"instance_points":int(n),"semantic_name":semantic_by_object.get((scene,int(obj)),"unavailable"),"semantic_class":-1} for obj,n in zip(ids,counts) if obj>0]


def inspect_scene(scene, split, official_root, processed_root, semantic_by_object=None):
    """Require exact XYZ/RGB for normals; official PLY alone owns target IDs."""
    from plyfile import PlyData
    official_root, processed_root = Path(official_root), Path(processed_root)
    path = official_root / "scans" / f"{scene}.ply"
    pack = processed_root / split / scene
    ply = PlyData.read(path)["vertex"].data
    points = np.stack([ply[k] for k in ("x", "y", "z")], 1).astype(np.float32)
    colors = np.stack([ply[k] for k in ("R", "G", "B")], 1)
    labels = np.asarray(ply["label"], dtype=np.int64)
    coord = np.load(pack / "coord.npy", mmap_mode="r")
    rgb = np.load(pack / "color.npy", mmap_mode="r")
    instance = np.load(pack / "instance.npy", mmap_mode="r").reshape(-1)
    if not np.array_equal(coord, points) or not np.array_equal(rgb, colors):
        raise ValueError(f"{scene}: official PLY coordinate/RGB ordering differs from normals source")
    mapped = np.where(instance >= 0, instance + 1, -1)
    processed_ids_match=bool(np.array_equal(mapped, labels))
    normals = np.load(pack / "normal.npy", mmap_mode="r")
    if normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError(f"{scene}: invalid normals")
    objects = []
    semantic_by_object = semantic_by_object or {}
    for obj in sorted(int(x) for x in np.unique(labels) if x > 0):
        objects.append({"instance": obj, "instance_points": int(np.count_nonzero(labels == obj)),
                        "semantic_name": semantic_by_object.get((scene, obj), "unavailable"),
                        "semantic_class": -1})
    sources = [path] + [pack / (name + ".npy") for name in ("coord", "color", "normal", "instance")]
    hashes = {str(p): file_sha256(p) for p in sources}
    return {"scene": scene, "kind": split, "points": len(points), "objects": objects,
            "data_path": str(path), "data_sha256": hashes[str(path)],
            "normal_path": str(pack / "normal.npy"), "source_hashes": hashes,
            "normal_alignment_verified": True, "ground_truth_authority":"official_AGILE3D_PLY_label", "processed_instance_plus_one_exact_diagnostic":processed_ids_match}


def initial_clicks(target, xyz, seed):
    """Official deepest-interior initial clicks, shuffled with a paired seed."""
    candidates = kdtree_click_reference(np.zeros_like(target), target, xyz)
    by_label = {int(x["target_label"]): x for x in candidates}
    labels = sorted(int(x) for x in np.unique(target) if x > 0)
    if set(labels) != set(by_label):
        raise ValueError("initial clicks require all requested objects to survive voxelization")
    import random
    random.Random(int(seed)).shuffle(labels)
    clicks = {str(k): [] for k in [0] + sorted(labels)}
    times = {str(k): [] for k in [0] + sorted(labels)}
    for time, label in enumerate(labels):
        clicks[str(label)] = [int(by_label[label]["center_index"])]
        times[str(label)] = [time]
    return clicks, times


def metrics_from_trace(trace):
    """MO operates on the scene mean; SO is the K=1 special case."""
    states, count = trace["states"], int(trace["object_count"])
    values = {}
    for click in CLICKS:
        rows = [r for r in states if int(r["total_clicks"]) == click * count]
        if len(rows) != 1:
            raise ValueError(f"missing or duplicate exact click budget {click}")
        values[f"IoU@{click}"] = float(rows[0]["mean_iou"])
    for threshold in THRESHOLDS:
        reached = [r["total_clicks"] / count for r in states if r["mean_iou"] >= threshold / 100]
        values[f"NoC@{threshold}"] = float(min(reached)) if reached else 20.0
    return values


def official_lines(index, trace, mode):
    scene = trace["scene"].removeprefix("scene")
    count = int(trace["object_count"])
    identity = count if mode == "MO" else int(trace["object_ids"][0])
    for row in trace["states"]:
        clicks = str(float(row["total_clicks"] / count)) if mode == "MO" else str(int(row["total_clicks"]))
        yield f"{index} {scene} {identity} {clicks} {float(row['mean_iou'])!r}\n"


def paired_bootstrap(public, adapted, samples=10000, seed=20260912):
    """Paired unit bootstrap. Reject missing/extra identities instead of intersecting."""
    if not public or set(public) != set(adapted):
        raise ValueError("paired identities differ or are empty")
    keys = sorted(public)
    rng = np.random.default_rng(seed)
    result = {}
    for metric in [f"IoU@{x}" for x in CLICKS] + [f"NoC@{x}" for x in THRESHOLDS]:
        differences = np.array([adapted[k][metric] - public[k][metric] for k in keys])
        draws = np.empty(samples)
        # Bound memory for the 10,357-object secondary endpoint.
        for start in range(0, samples, 100):
            n = min(100, samples - start)
            draws[start:start+n] = differences[rng.integers(0, len(keys), (n, len(keys)))].mean(1)
        result[metric] = {"delta": float(differences.mean()), "ci95_lower": float(np.quantile(draws, .025)),
                          "ci95_upper": float(np.quantile(draws, .975)), "paired_units": len(keys), "samples": samples}
    return result
