#!/usr/bin/env python3
"""Matched LitePT transfer evaluation on official AGILE3D S3DIS/KITTI-360 data.

This runner evaluates the two frozen LitePT encoders with the decoder
checkpoints trained by the matched ScanNet40 transfer. It deliberately does
not train on S3DIS or KITTI-360 labels: the official target labels are used
only for downstream evaluation. The multi-object and single-object panels
follow the official AGILE3D file and metric accounting, while the model
remains native LitePT dec0 rather than the original MinkowskiEngine backbone.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

import run_agile3d_multio as core
from delimit3d.evaluation import cross_dataset_protocol as protocol
from delimit3d.evaluation.agile3d_protocol import (
    append_clicks,
    enforce_click_labels,
    file_sha256,
    json_sha256,
    raw_object_ious,
    scene_seed,
    simulated_corrections,
)
from delimit3d.evaluation.agile3d_decoder import load_initialization
from mesh_utils import (
    face_colors,
    labels_to_colors,
    mesh_from_points,
    write_mesh_ply,
)

SCHEMA = "delimit3d_cross_dataset_matched/v1"
REPO = Path(__file__).resolve().parents[2]
SEED = 20260912
DATASETS = {"S3DIS", "KITTI360"}


def dump(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def resolve(value: str | Path, *, strict: bool = True) -> Path:
    path = Path(str(value)).expanduser()
    if path.exists() or not str(path).startswith("/cluster/"):
        return path.resolve(strict=strict)
    mount = os.environ.get("DELIMIT3D_EULER_MOUNT")
    if not mount:
        raise RuntimeError("explicit DELIMIT3D_EULER_MOUNT required off Euler")
    return (Path(mount) / str(path).removeprefix("/cluster/")).resolve(strict=strict)


def effective_commit(config: Mapping[str, Any]) -> str:
    configured = str(config.get("repo_commit", "auto"))
    if configured not in {"", "auto", "unknown"}:
        return configured
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def output_root(config: Mapping[str, Any]) -> Path:
    override = os.environ.get("DELIMIT3D_OUTPUT_ROOT")
    root = Path(override).expanduser().resolve() if override else resolve(config["paths"]["output_root"], strict=False)
    if config["experiment_id"] not in str(root):
        raise ValueError("refusing an output root without the explicit experiment ID")
    root.mkdir(parents=True, exist_ok=True)
    return root


def decoder_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "feature_dim", "hidden_dim", "num_heads", "dim_feedforward", "num_decoders",
        "num_bg_queries", "dropout", "pre_norm", "max_click_events",
        "normalize_pos_enc", "gauss_scale", "aux",
    }
    values = {key: config["decoder"][key] for key in allowed if key in config["decoder"]}
    defaults = {
        "feature_dim": 72, "hidden_dim": 128, "num_heads": 8,
        "dim_feedforward": 1024, "num_decoders": 3, "num_bg_queries": 10,
        "dropout": 0.0, "pre_norm": False, "max_click_events": 200,
        "normalize_pos_enc": True, "gauss_scale": 1.0, "aux": True,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    return values


def verify_config(config: Mapping[str, Any]) -> None:
    required = ("experiment_id", "dataset", "seed", "paths", "protocol", "decoder", "arms", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"configuration is missing {missing}")
    if str(config["dataset"]) not in DATASETS:
        raise ValueError("dataset must be S3DIS or KITTI360")
    if int(config["seed"]) != SEED:
        raise ValueError("paired seed is locked to 20260912")
    protocol_cfg = config["protocol"]
    expected_protocol = {
        "active_litept_level": "dec0",
        "voxel_size_m": 0.02,
        "voxel_reduce": "representative",
        "representative_sampling": "first",
        "click_budget": 20,
        "click_center_method": "kdtree",
        "single_object_label_offset": 1,
    }
    for key, value in expected_protocol.items():
        if protocol_cfg.get(key) != value:
            raise ValueError(f"protocol.{key} drift: expected {value!r}")
    if protocol_cfg.get("normal_policy") != "open3d_knn_pca_centroid_orient_v1":
        raise ValueError("normal_policy must name the frozen synthesized-normal method")
    if int(config["training"]["updates"]) != 5000:
        raise ValueError("decoder checkpoint update must be exactly 5000")
    expected_decoder = {
        "feature_dim": 72, "hidden_dim": 128, "num_heads": 8,
        "dim_feedforward": 1024, "num_decoders": 3, "num_bg_queries": 10,
        "dropout": 0.0, "pre_norm": False, "max_click_events": 200,
        "normalize_pos_enc": True, "gauss_scale": 1.0, "aux": True,
    }
    actual = decoder_kwargs(config)
    for key, value in expected_decoder.items():
        if actual[key] != value:
            raise ValueError(f"decoder.{key}={actual[key]!r}, expected {value!r}")
    for arm in ("public", "delimit3d"):
        if arm not in config["arms"]:
            raise ValueError(f"missing {arm} arm")
        for key in ("checkpoint", "checkpoint_sha256", "decoder_checkpoint", "decoder_checkpoint_sha256", "encoder_tensor_sha256"):
            if key not in config["arms"][arm]:
                raise ValueError(f"{arm} arm missing {key}")
    if not config["paths"].get("normal_root"):
        raise ValueError("normal_root is required; RGBN6 cannot silently use zeros")
    if str(config.get("ground_truth_authority")) != "official_AGILE3D_PLY_label":
        raise ValueError("official PLY labels must remain target authority")


def official_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = resolve(config["paths"]["official_root"])
    return {
        "val": root / "val_list.json",
        "so_ids": root / "single" / "object_ids.npy",
        "so_classes": root / "single" / "object_classes.txt",
    }


def verify_normal_generation_report(
    config: Mapping[str, Any], report_path: Path
) -> dict[str, Any]:
    """Validate the frozen RGBN6 sidecar report against the official lists."""
    if not report_path.is_file():
        raise FileNotFoundError(f"normal generation report missing: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("schema") != "delimit3d_cross_dataset_normals/v1":
        raise ValueError("normal generation report schema mismatch")
    if str(report.get("dataset")) != str(config["dataset"]):
        raise ValueError("normal generation report dataset mismatch")
    if str(report.get("scope")) != "all":
        raise ValueError("normal generation must cover both scenes and crops")
    if str(report.get("method")) != "open3d_estimate_normals_knn_pca_centroid_orient":
        raise ValueError("normal generation method drift")
    if int(report.get("knn", -1)) != 30:
        raise ValueError("normal generation KNN drift")
    paths = official_paths(config)
    expected = {
        "val_list.json": paths["val"],
        "single/object_ids.npy": paths["so_ids"],
        "single/object_classes.txt": paths["so_classes"],
    }
    observed = report.get("source_list_hashes")
    if not isinstance(observed, Mapping):
        raise ValueError("normal generation report lacks source_list_hashes")
    for name, official_path in expected.items():
        if observed.get(name) != file_sha256(official_path):
            raise ValueError(f"normal generation source-list hash drift: {name}")
    entries = report.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("normal generation report has no entries")
    return report


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    points = np.stack([vertex[key] for key in ("x", "y", "z")], axis=1).astype(np.float32)
    colors = np.stack([vertex[key] for key in ("R", "G", "B")], axis=1).astype(np.float32) / 255.0
    labels = np.asarray(vertex["label"], dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(colors) or len(points) != len(labels):
        raise ValueError(f"{path}: malformed PLY columns")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError(f"{path}: non-finite XYZ/RGB")
    return points, colors, labels


def _normal_path(config: Mapping[str, Any], name: str) -> Path:
    return resolve(config["paths"]["normal_root"], strict=False) / str(config["dataset"]) / f"{name}.npy"


def _scene_record(
    config: Mapping[str, Any],
    *,
    scene: str,
    kind: str,
    data_path: Path,
    normal_path: Path,
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    points, _colors, labels = read_ply(data_path)
    if not normal_path.exists():
        raise FileNotFoundError(f"{scene}: missing synthesized normal sidecar {normal_path}")
    normals = np.asarray(np.load(normal_path), dtype=np.float32)
    if normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError(f"{scene}: normal sidecar shape/value mismatch")
    if np.any(np.abs(np.linalg.norm(normals, axis=1) - 1.0) > 2.0e-3):
        raise ValueError(f"{scene}: synthesized normals are not unit length")
    shifted, _shift = core._center_shift_numpy(points)
    representatives = protocol.native_cuda_representative_indices(
        shifted, voxel_size=float(config["protocol"]["voxel_size_m"])
    )
    ids = [int(item["instance"]) for item in objects]
    surviving = set(int(value) for value in np.unique(labels[representatives]))
    missing = [obj for obj in ids if obj not in surviving]
    if missing:
        raise ValueError(f"{scene}: official objects without representative tokens: {missing}")
    return {
        "scene": str(scene),
        "kind": str(kind),
        "mode": "SO" if str(kind) == "crop" else "SCENE",
        "object_ids": [int(item["instance"]) for item in objects],
        "points": int(len(points)),
        "objects": [dict(item) for item in objects],
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "normal_path": str(normal_path),
        "normal_sha256": file_sha256(normal_path),
        "source_hashes": {str(data_path): file_sha256(data_path), str(normal_path): file_sha256(normal_path)},
        "normal_alignment_verified": True,
        "ground_truth_authority": "official_AGILE3D_PLY_label",
        "representative_indices_sha256": core.array_sha256(representatives),
        "representative_token_count": int(len(representatives)),
        "representative_preflight_arithmetic": "pytorch_2.4.1_cuda_float32_scalar_reciprocal",
        "single_object_label_offset": int(config["protocol"]["single_object_label_offset"]),
    }


def _semantics(config: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[tuple[str, int], str]:
    ids = np.load(paths["so_ids"], allow_pickle=True)
    classes = np.loadtxt(paths["so_classes"], dtype=str)
    if ids.ndim != 2 or ids.shape[1] != 2 or len(ids) != len(classes):
        raise ValueError("official single-object IDs/classes are not aligned")
    offset = int(config["protocol"]["single_object_label_offset"])
    return {(str(scene), int(index) + offset): str(cls) for (scene, index), cls in zip(ids, classes, strict=True)}


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_config(config)
    root = output_root(config)
    provenance = load_provenance(config)
    paths = official_paths(config)
    val = json.loads(paths["val"].read_text())
    if not isinstance(val, dict) or not val:
        raise ValueError("official validation list must be a non-empty mapping")
    semantics = _semantics(config, paths)
    official_root = resolve(config["paths"]["official_root"])
    normal_root = resolve(config["paths"]["normal_root"])
    validation: list[dict[str, Any]] = []
    scene_by_name: dict[str, dict[str, Any]] = {}
    for key, value in val.items():
        scene = str(key).rsplit("_obj_", 1)[0]
        if scene in scene_by_name:
            raise ValueError(f"duplicate official scene panel: {scene}")
        data_path = official_root / "scans" / f"{scene}.ply"
        points, _colors, labels = read_ply(data_path)
        objects = protocol.objects_from_official_labels(
            labels, scene, semantic_by_object=semantics
        )
        object_ids = [int(item["instance"]) for item in objects]
        if any(int(obj) not in set(np.unique(labels).tolist()) for obj in object_ids):
            raise ValueError(f"{scene}: object extraction mismatch")
        record = _scene_record(
            config, scene=scene, kind="scene", data_path=data_path,
            normal_path=normal_root / str(config["dataset"]) / f"{scene}.npy",
            objects=objects,
        )
        record["official_val_keys"] = [str(key)]
        validation.append(record)
        scene_by_name[scene] = record
    # Single-object AGILE3D uses crop PLYs. Keep them as separate records so
    # their geometry and LitePT cache are exactly the official crop geometry.
    ids = np.load(paths["so_ids"], allow_pickle=True)
    classes = np.loadtxt(paths["so_classes"], dtype=str)
    offset = int(config["protocol"]["single_object_label_offset"])
    crop_records: list[dict[str, Any]] = []
    so: list[dict[str, Any]] = []
    for (base_scene, crop_index), semantic_name in zip(ids, classes, strict=True):
        base_scene, crop_index = str(base_scene), int(crop_index)
        cache_scene = f"{base_scene}__crop_{crop_index}"
        data_path = official_root / "single" / "crops" / base_scene / f"{base_scene}_crop_{crop_index}.ply"
        points, _colors, labels = read_ply(data_path)
        if set(np.unique(labels).tolist()) - {0, 1}:
            raise ValueError(f"{data_path}: crop labels are not binary 0/1")
        object_points = int(np.count_nonzero(labels == 1))
        if object_points == 0:
            raise ValueError(f"{data_path}: empty target crop")
        objects = [{"instance": 1, "instance_points": object_points, "semantic_name": str(semantic_name), "semantic_class": -1}]
        record = _scene_record(
            config, scene=cache_scene, kind="crop", data_path=data_path,
            normal_path=normal_root / str(config["dataset"]) / f"{base_scene}_crop_{crop_index}.npy",
            objects=objects,
        )
        record["official_scene"] = base_scene
        record["official_object_index"] = crop_index
        record["data_role"] = "single_crop"
        record["single_label"] = 1
        validation.append(record)
        crop_records.append(record)
        so.append({
            "scene": cache_scene,
            "objects": [1],
            "semantic_name": str(semantic_name),
            "official_scene": base_scene,
            "official_object_index": crop_index,
            "official_key": f"{base_scene}_obj_{crop_index}",
        })
    mo: list[dict[str, Any]] = []
    for key, value in val.items():
        scene = str(key).rsplit("_obj_", 1)[0]
        objects = [int(value["obj"][str(i)]) for i in sorted(map(int, value["obj"]))]
        if scene not in scene_by_name:
            raise ValueError(f"{key}: no scene record")
        mo.append({"scene": scene, "objects": objects, "official_key": str(key), "official_scene": scene})
    manifest = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "repo_commit": effective_commit(config),
        "dataset": str(config["dataset"]),
        "validation_scenes": validation,
        "train_scenes": [],
        "panels": {"MO": mo, "SO": so},
        "source_lists": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "normal_generation_report": str(resolve(config["paths"]["normal_report"])),
        "normal_generation_report_sha256": file_sha256(resolve(config["paths"]["normal_report"])),
        "decoder_initialization_tensor_sha256": config["training"]["decoder_initialization_tensor_sha256"],
        "training_source": dict(config["training"]),
        "provenance_sha256": file_sha256(root / "freeze" / "provenance.json"),
        "record_count": len(validation),
        "manifest_notes": [
            "MO follows the official multi-object val_list mapping and raw PLY instance IDs.",
            "SO follows the official crop/object_ids.npy and object_classes.txt files; crop index is mapped to raw binary target label 1.",
            "S3DIS/KITTI-360 labels are evaluation-only; no target labels enter encoder adaptation or decoder training.",
        ],
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    manifest_path = root / "selection_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("refusing to overwrite a changed cross-dataset manifest")
    else:
        dump(manifest_path, manifest)
    dump(root / "prepare_report.json", {
        "schema": "delimit3d_cross_dataset_prepare/v1",
        "dataset": config["dataset"],
        "MO": len(mo),
        "MO_unique_scenes": len(scene_by_name),
        "SO": len(so),
        "validation_records": len(validation),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_lists": manifest["source_lists"],
        "normal_report_sha256": manifest["normal_generation_report_sha256"],
        "single_object_label_offset": offset,
    })
    return manifest


def freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_config(config)
    root = output_root(config)
    if effective_commit(config) != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip():
        raise ValueError("freeze must run from the current committed worktree")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError(f"commit the follow-up worktree before freezing: {dirty}")
    paths = official_paths(config)
    normal_report = resolve(config["paths"]["normal_report"])
    normal_payload = verify_normal_generation_report(config, normal_report)
    audit_path = resolve(config["paths"]["official_data_audit"])
    if not audit_path.is_file():
        raise FileNotFoundError(f"official data audit missing: {audit_path}")
    out = root / "freeze"
    out.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["repo_commit"] = effective_commit(config)
    config_text = yaml.safe_dump(config, sort_keys=False)
    resolved = out / "resolved_config.yaml"
    if resolved.exists() and resolved.read_text() != config_text:
        raise RuntimeError("frozen config differs; use a new experiment ID")
    if not resolved.exists():
        resolved.write_text(config_text)
    archive = out / f"source_{config['repo_commit']}.tar"
    if not archive.exists():
        subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", config["repo_commit"]], cwd=REPO, check=True)
    init_source = resolve(config["training"]["decoder_initialization_path"])
    init_target = root / "decoder_init.pt"
    if init_target.exists() and file_sha256(init_target) != file_sha256(init_source):
        raise RuntimeError("decoder initialization copy drift")
    if not init_target.exists():
        init_target.write_bytes(init_source.read_bytes())
    lists = {}
    frozen_lists = out / "manifests"
    for name, path in paths.items():
        expected = config["official_manifest_sha256"][name]
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"official list changed: {name}")
        target = frozen_lists / path.relative_to(resolve(config["paths"]["official_root"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != observed:
            raise RuntimeError(f"frozen list drift: {target}")
        if not target.exists():
            target.write_bytes(path.read_bytes())
        lists[name] = {"path": str(path), "frozen_path": str(target), "sha256": observed}
    audit_target = frozen_lists / "official_data_audit_v1.json"
    audit_sha256 = file_sha256(audit_path)
    if audit_target.exists() and file_sha256(audit_target) != audit_sha256:
        raise RuntimeError(f"frozen official data audit drift: {audit_target}")
    if not audit_target.exists():
        audit_target.write_bytes(audit_path.read_bytes())
    checkpoints = {}
    for arm, arm_cfg in config["arms"].items():
        for key in ("checkpoint", "decoder_checkpoint"):
            path = resolve(arm_cfg[key])
            observed = file_sha256(path)
            expected = arm_cfg[f"{key}_sha256"]
            if observed != expected:
                raise ValueError(f"{arm} {key} hash mismatch")
            checkpoints[f"{arm}_{key}"] = {"path": str(path), "sha256": observed}
    environment = {
        "schema": "delimit3d_cross_dataset_environment/v1",
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
    }
    dump(out / "environment_cpu.json", environment)
    report = {
        "schema": "delimit3d_cross_dataset_provenance/v1",
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "repo_commit": config["repo_commit"],
        "resolved_config": str(resolved),
        "resolved_config_sha256": file_sha256(resolved),
        "source_archive": str(archive),
        "source_archive_sha256": file_sha256(archive),
        "decoder_initialization": {
            "path": str(init_target),
            "sha256": file_sha256(init_target),
            "tensor_state_sha256": config["training"]["decoder_initialization_tensor_sha256"],
        },
        "normal_generation_report": {
            "path": str(normal_report),
            "sha256": file_sha256(normal_report),
            "method": normal_payload["method"],
            "knn": int(normal_payload["knn"]),
            "entry_count": len(normal_payload["entries"]),
        },
        "official_data_audit": {
            "path": str(audit_path),
            "frozen_path": str(audit_target),
            "sha256": audit_sha256,
        },
        "official_lists": lists,
        "checkpoints": checkpoints,
        "environment": str(out / "environment_cpu.json"),
        "environment_sha256": file_sha256(out / "environment_cpu.json"),
        "training_source": dict(config["training"]),
        "metric_scope": "matched LitePT comparison under AGILE3D metric protocol; not literal AGILE3D model reproduction",
    }
    dump(out / "provenance.json", report)
    return report


def load_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    path = root / "freeze" / "provenance.json"
    report = json.loads(path.read_text())
    if report.get("experiment_id") != config["experiment_id"] or report.get("dataset") != config["dataset"]:
        raise ValueError("cross-dataset provenance experiment mismatch")
    if file_sha256(root / "freeze" / "resolved_config.yaml") != report["resolved_config_sha256"]:
        raise ValueError("resolved config hash drift")
    if file_sha256(resolve(report["source_archive"])) != report["source_archive_sha256"]:
        raise ValueError("source archive hash drift")
    for name, entry in report["official_lists"].items():
        if file_sha256(resolve(entry["path"])) != entry["sha256"]:
            raise ValueError(f"official list drift: {name}")
        if file_sha256(resolve(entry["frozen_path"])) != entry["sha256"]:
            raise ValueError(f"frozen official list drift: {name}")
    normal_report = report["normal_generation_report"]
    normal_path = resolve(normal_report["path"])
    if file_sha256(normal_path) != normal_report["sha256"]:
        raise ValueError("normal generation report drift")
    verified_normals = verify_normal_generation_report(config, normal_path)
    if verified_normals.get("method") != normal_report.get("method") or int(verified_normals.get("knn", -1)) != int(normal_report.get("knn", -2)):
        raise ValueError("normal generation report metadata drift")
    audit = report.get("official_data_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("official data audit missing from provenance")
    if file_sha256(resolve(audit["path"])) != audit["sha256"]:
        raise ValueError("official data audit drift")
    if file_sha256(resolve(audit["frozen_path"])) != audit["sha256"]:
        raise ValueError("frozen official data audit drift")
    init_entry = report.get("decoder_initialization")
    if not isinstance(init_entry, Mapping):
        raise ValueError("decoder initialization missing from provenance")
    init_path = resolve(init_entry["path"])
    if file_sha256(init_path) != init_entry["sha256"]:
        raise ValueError("decoder initialization file drift")
    _, init_report = load_initialization(init_path, expected_kwargs=decoder_kwargs(config))
    if init_report.get("tensor_state_sha256") != init_entry.get("tensor_state_sha256"):
        raise ValueError("decoder initialization tensor hash drift")
    for arm, arm_cfg in config["arms"].items():
        for key in ("checkpoint", "decoder_checkpoint"):
            path = resolve(arm_cfg[key])
            if file_sha256(path) != arm_cfg[f"{key}_sha256"]:
                raise ValueError(f"{arm} {key} changed")
    return report


def load_prepared(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest_path = output_root(config) / "selection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run prepare first")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.pop("manifest_sha256", None)
    if expected is None or json_sha256(manifest) != expected:
        raise ValueError("selection manifest hash mismatch")
    manifest["manifest_sha256"] = expected
    if manifest.get("schema") != SCHEMA or manifest.get("experiment_id") != config["experiment_id"]:
        raise ValueError("selection manifest schema/experiment mismatch")
    if manifest.get("provenance_sha256") != file_sha256(output_root(config) / "freeze" / "provenance.json"):
        raise ValueError("manifest provenance linkage mismatch")
    for record in manifest["validation_scenes"]:
        if file_sha256(resolve(record["data_path"])) != record["data_sha256"]:
            raise ValueError(f"{record['scene']}: scene source hash drift")
        if file_sha256(resolve(record["normal_path"])) != record["normal_sha256"]:
            raise ValueError(f"{record['scene']}: normal source hash drift")
    return manifest, resolve(output_root(config) / "decoder_init.pt")


def target_from_episode(cache: Mapping[str, Any], object_ids: Sequence[int]) -> tuple[torch.Tensor, dict[str, list[int]], dict[str, list[int]], list[int]]:
    target = torch.zeros(cache["features"].shape[0], dtype=torch.long)
    for local_id, object_id in enumerate(object_ids, start=1):
        indices = cache["object_token_indices"].get(str(int(object_id)))
        if indices is None or len(indices) == 0:
            raise ValueError(f"{cache['scene']}: object {object_id} has no representative token")
        if bool(torch.any(target[indices] != 0)):
            raise ValueError("selected objects overlap after representative-token mapping")
        target[indices] = local_id
    seed = scene_seed(SEED, cache["scene"], "initial:" + ",".join(map(str, object_ids)))
    clicks, times = protocol.initial_clicks(target.numpy(), cache["scene_xyz"].numpy(), seed)
    return target, clicks, times, list(object_ids)


class CacheLRU:
    def __init__(self, root: Path, arm: str, capacity: int = 1):
        self.root, self.arm, self.capacity = root, arm, capacity
        self.values: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, scene: str) -> dict[str, Any]:
        if scene in self.values:
            self.values.move_to_end(scene)
        else:
            self.values[scene] = core.load_cache(self.root, self.arm, scene)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return self.values[scene]


def install_adapter() -> None:
    core.SCHEMA = SCHEMA
    core.output_root = output_root
    core.resolve_external = lambda value, strict=True: resolve(value, strict=strict)
    core.effective_commit = effective_commit
    core._load_record_arrays = lambda record: protocol.load_scene(record, resolver=resolve)
    core.load_prepared = lambda config: load_prepared(config)
    core._target_from_episode = target_from_episode
    core.json_dump = dump


def cache_features(config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    root = output_root(config)
    manifest, _ = load_prepared(config)
    records = {record["scene"]: record for record in manifest["validation_scenes"]}
    previous_path = root / f"cache_report_{arm}.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for row in previous.get("cache_rows", []):
            if not row.get("file_sha256") or file_sha256(core._cache_path(root, arm, row["scene"])) != row["file_sha256"]:
                raise ValueError("existing cache binary changed")
    original_dump = core.json_dump

    def publish(path: Path, report: Mapping[str, Any]) -> None:
        report = dict(report)
        if Path(path).name == f"cache_report_{arm}.json":
            rows = [dict(row) for row in report["cache_rows"]]
            for row in rows:
                cache_path = core._cache_path(root, arm, row["scene"])
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                record = records[row["scene"]]
                if core.array_sha256(payload["representative_indices"]) != record["representative_indices_sha256"]:
                    raise ValueError(f"{row['scene']}: GPU/CPU representative geometry mismatch")
                payload["arm"] = arm
                core_temporary = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
                torch.save(payload, core_temporary)
                os.replace(core_temporary, cache_path)
                row["file_sha256"] = file_sha256(cache_path)
            report["cache_rows"] = rows
        original_dump(path, report)

    core.json_dump = publish
    try:
        return core.cache_features(config, arm)
    finally:
        core.json_dump = original_dump


def verify_caches(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    manifest, _ = load_prepared(config)
    expected = {record["scene"] for record in manifest["validation_scenes"]}
    for arm in ("public", "delimit3d"):
        report = json.loads((root / f"cache_report_{arm}.json").read_text())
        if report.get("arm") != arm or report.get("repo_commit") != effective_commit(config):
            raise ValueError(f"{arm}: cache report provenance mismatch")
        if set(row["scene"] for row in report["cache_rows"]) != expected:
            raise ValueError(f"{arm}: cache set does not cover exact manifest")
        if not report["encoder_state_unchanged"] or report["encoder_tensor_sha256_before"] != report["encoder_tensor_sha256_after"]:
            raise ValueError(f"{arm}: encoder invariant failed")
        for row in report["cache_rows"]:
            path = core._cache_path(root, arm, row["scene"])
            if file_sha256(path) != row.get("file_sha256"):
                raise ValueError(f"{arm}/{row['scene']}: cache hash mismatch")
            payload = core.load_cache(root, arm, row["scene"])
            if payload.get("arm") != arm or payload.get("empty_objects"):
                raise ValueError(f"{arm}/{row['scene']}: cache payload invariant failed")
    geometry = core._verify_cross_arm_geometry(root, manifest)
    if not geometry or not geometry.get("geometry_byte_identity"):
        raise ValueError("public/Delimit3D geometry hashes are not byte-identical")
    return geometry


def _checkpoint_for(config: Mapping[str, Any], arm: str) -> Path:
    path = resolve(config["arms"][arm]["decoder_checkpoint"])
    if file_sha256(path) != config["arms"][arm]["decoder_checkpoint_sha256"]:
        raise ValueError(f"{arm}: decoder checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema": core.CHECKPOINT_SCHEMA,
        "experiment_id": config["training"]["experiment_id"],
        "arm": arm,
        "update": int(config["training"]["updates"]),
        "repo_commit": config["training"]["repo_commit"],
        "decoder_initialization_sha256": config["training"]["decoder_initialization_tensor_sha256"],
        "decoder_kwargs": decoder_kwargs(config),
        "encoder_frozen": True,
        "episode_schedule_sha256": config["training"]["episode_schedule_sha256"],
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"{arm}: decoder checkpoint metadata drift at {key}")
    expected_encoder_hash = config["arms"][arm]["encoder_tensor_sha256"]
    if payload.get("encoder_tensor_sha256") != expected_encoder_hash:
        raise ValueError(f"{arm}: decoder checkpoint encoder hash mismatch")
    return path


def evaluate(config: Mapping[str, Any], arm: str, mode: str) -> dict[str, Any]:
    if mode not in ("MO", "SO"):
        raise ValueError("mode must be MO or SO")
    root = output_root(config)
    manifest, _ = load_prepared(config)
    verify_caches(config)
    checkpoint = _checkpoint_for(config, arm)
    decoder = core._checkpoint_decoder(checkpoint, config).to("cuda")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    records = {record["scene"]: record for record in manifest["validation_scenes"]}
    episodes = manifest["panels"][mode]
    out = root / arm / "evaluation" / mode
    out.mkdir(parents=True, exist_ok=True)
    cache = CacheLRU(root, arm, capacity=1)
    rows: list[dict[str, Any]] = []
    csv_path = out / "official_results.csv"
    temporary_csv = out / "official_results.csv.tmp"
    with temporary_csv.open("w") as handle:
        for index, episode in enumerate(episodes):
            scene = str(episode["scene"])
            if mode == "MO":
                identity = str(episode["official_key"])
            else:
                identity = str(episode["official_key"])
            requested_ids = [int(value) for value in episode["objects"]]
            requested_objects = [
                dict(item)
                for item in records[scene].get("objects", [])
                if int(item.get("instance", -1)) in requested_ids
            ]
            if len(requested_objects) != len(requested_ids):
                raise ValueError(f"{identity}: requested object metadata is incomplete")
            trace_path = out / "traces" / f"{index:05d}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.json"
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                if trace.get("checkpoint_sha256") != file_sha256(checkpoint):
                    raise ValueError(f"{identity}: trace checkpoint drift")
                if not trace.get("objects"):
                    trace["objects"] = requested_objects
                    dump(trace_path, trace)
            else:
                prediction_path = out / "predictions" / f"{index:05d}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.npz"
                trace = core._evaluate_episode(
                    decoder, cache.get(scene), records[scene], requested_ids,
                    device=device, config=config, mask_path=prediction_path,
                )
                trace["object_ids"] = list(trace.get("panel_objects", requested_ids))
                trace["objects"] = requested_objects
                trace.update({
                    "identity": identity,
                    "official_key": identity,
                    "official_scene": str(episode.get("official_scene", records[scene].get("official_scene", scene))),
                    "official_object_index": episode.get("official_object_index"),
                    "mode": mode,
                    "dataset": config["dataset"],
                    "cache_scene": scene,
                    "checkpoint_sha256": file_sha256(checkpoint),
                })
                dump(trace_path, trace)
            metrics = protocol.metrics_from_trace(trace)
            rows.append({
                "identity": identity,
                "scene": str(trace["scene"]),
                "official_scene": str(trace.get("official_scene", episode.get("official_scene", scene))),
                "object_count": int(trace["object_count"]),
                "objects": trace.get("objects", requested_objects),
                "metrics": metrics,
                "trace": str(trace_path),
                "official_key": identity,
            })
            handle.writelines(protocol.official_lines(index, trace, mode))
            handle.flush()
            if index == 0 or (index + 1) % 10 == 0:
                print(json.dumps({"mode": mode, "arm": arm, "dataset": config["dataset"], "completed": index + 1, "total": len(episodes)}), flush=True)
    os.replace(temporary_csv, csv_path)
    report = {
        "schema": "delimit3d_cross_dataset_evaluation_report/v1",
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "arm": arm,
        "mode": mode,
        "units": rows,
        "count": len(rows),
        "metrics": {metric: float(np.mean([row["metrics"][metric] for row in rows])) for metric in rows[0]["metrics"]} if rows else {},
        "official_csv": str(csv_path),
        "official_csv_sha256": file_sha256(csv_path),
        "decoder_checkpoint": str(checkpoint),
        "decoder_checkpoint_sha256": file_sha256(checkpoint),
        "encoder_checkpoint_sha256": config["arms"][arm]["checkpoint_sha256"],
        "encoder_tensor_sha256": config["arms"][arm]["encoder_tensor_sha256"],
        "metric_scope": "matched LitePT comparison under AGILE3D metric protocol",
    }
    dump(out / "evaluation_report.json", report)
    return report


def _metric_map(report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {str(row["identity"]): dict(row["metrics"]) for row in report["units"]}


def _object_trace(trace: Mapping[str, Any], local_id: int, obj: Mapping[str, Any]) -> dict[str, Any]:
    states = []
    for state in trace["states"]:
        state = dict(state)
        state["mean_iou"] = float(state["object_ious"][str(local_id)])
        states.append(state)
    return {"scene": str(trace["scene"]), "object_count": 1, "states": states, "objects": [dict(obj)]}


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    aggregate_result: dict[str, Any] = {
        "schema": "delimit3d_cross_dataset_aggregate/v1",
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "endpoints": {},
        "metric_scope": "matched LitePT comparison under AGILE3D metric protocol; not literal AGILE3D reproduction",
    }
    for mode in ("MO", "SO"):
        paths = {arm: root / arm / "evaluation" / mode / "evaluation_report.json" for arm in ("public", "delimit3d")}
        if any(not path.exists() for path in paths.values()):
            continue
        reports = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
        public_map, adapted_map = _metric_map(reports["public"]), _metric_map(reports["delimit3d"])
        paired = protocol.paired_bootstrap(public_map, adapted_map, **config["bootstrap"])
        groups: defaultdict[str, list[str]] = defaultdict(list)
        for row in reports["public"]["units"]:
            groups[f"requested_count/{row['object_count']}"].append(str(row["identity"]))
            if row.get("objects"):
                obj = row["objects"][0]
                n = int(obj.get("instance_points", 0))
                groups["size/" + ("small_lt1000" if n < 1000 else "medium_1000to9999" if n < 10000 else "large_ge10000")].append(str(row["identity"]))
                groups[f"semantic/{obj.get('semantic_name', 'unavailable')}"] .append(str(row["identity"]))
        strata = {}
        for name, keys in sorted(groups.items()):
            if len(keys) >= 2:
                strata[name] = protocol.paired_bootstrap(
                    {key: public_map[key] for key in keys},
                    {key: adapted_map[key] for key in keys},
                    **config["bootstrap"],
                )
        endpoint = {
            "metrics": {arm: reports[arm]["metrics"] for arm in reports},
            "paired_bootstrap": paired,
            "bootstrap_unit": "scene" if mode == "MO" else "object",
            "strata": strata,
        }
        # For MO, add object-level descriptive strata without changing the
        # official scene unit. This exposes size/class behavior explicitly.
        if mode == "MO":
            object_maps = {"public": {}, "delimit3d": {}}
            object_groups: defaultdict[str, list[str]] = defaultdict(list)
            for arm in ("public", "delimit3d"):
                for row in reports[arm]["units"]:
                    trace = json.loads(Path(row["trace"]).read_text())
                    for local_id, obj in enumerate(trace.get("objects", []), start=1):
                        key = f"{row['identity']}/{int(obj['instance'])}"
                        object_maps[arm][key] = protocol.metrics_from_trace(_object_trace(trace, local_id, obj))
                        if arm == "public":
                            n = int(obj.get("instance_points", 0))
                            object_groups["size/" + ("small_lt1000" if n < 1000 else "medium_1000to9999" if n < 10000 else "large_ge10000")].append(key)
                            object_groups[f"semantic/{obj.get('semantic_name', 'unavailable')}"] .append(key)
            endpoint["descriptive_object_strata"] = {
                name: protocol.paired_bootstrap(
                    {key: object_maps["public"][key] for key in keys if key in object_maps["public"]},
                    {key: object_maps["delimit3d"][key] for key in keys if key in object_maps["delimit3d"]},
                    **config["bootstrap"],
                )
                for name, keys in sorted(object_groups.items())
                if len(keys) >= 2 and set(keys).issubset(object_maps["public"]) and set(keys).issubset(object_maps["delimit3d"])
            }
        aggregate_result["endpoints"][mode] = endpoint
    dump(root / "aggregate.json", aggregate_result)
    return aggregate_result


def _trace_state_prediction(trace: Mapping[str, Any], npz: Mapping[str, Any], click: int) -> np.ndarray:
    count = int(trace["object_count"])
    target_total = int(click) * count
    for index, state in enumerate(trace["states"]):
        if int(state["total_clicks"]) == target_total:
            key = f"state_{index:03d}"
            if key not in npz:
                raise KeyError(f"prediction archive lacks {key}")
            return np.asarray(npz[key], dtype=np.int64)
    raise KeyError(f"trace lacks exact {click} clicks/object state")


def _render_axes(ax, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, title: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if len(faces):
        collection = Poly3DCollection(vertices[faces], facecolors=face_colors(colors, faces), linewidths=0.05, alpha=0.95)
        ax.add_collection3d(collection)
        ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    else:
        ax.scatter(*vertices.T, c=colors, s=1.0, depthshade=False)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    ax.view_init(28, -55)


def visualize(config: Mapping[str, Any]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = output_root(config)
    manifest, _ = load_prepared(config)
    records = {record["scene"]: record for record in manifest["validation_scenes"]}
    mo_units = json.loads((root / "public" / "evaluation" / "MO" / "evaluation_report.json").read_text())["units"]
    visual_root = root / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    selected = mo_units[: int(config["protocol"].get("visual_episode_count", 3))]
    outputs: list[dict[str, Any]] = []
    for unit in selected:
        trace_public = json.loads(Path(unit["trace"]).read_text())
        scene = str(trace_public["cache_scene"])
        record = records[scene]
        points, _colors, normals, objects, masks = protocol.load_scene(record, resolver=resolve)
        object_id = int(trace_public["object_ids"][0])
        local_id = 1
        target = np.zeros(len(points), dtype=np.int64)
        target[masks[object_id]] = local_id
        close_mask = np.zeros(len(points), dtype=bool)
        target_points = points[target == local_id]
        lo, hi = target_points.min(axis=0), target_points.max(axis=0)
        pad = max(0.25, float(np.max(hi - lo)) * 0.35)
        close_mask = np.all((points >= lo[None, :] - pad) & (points <= hi[None, :] + pad), axis=1)
        close_indices = np.flatnonzero(close_mask)
        if len(close_indices) < 40:
            close_indices = np.flatnonzero(target == local_id)
        if len(close_indices) > int(config["protocol"].get("visual_max_points", 12000)):
            close_indices = close_indices[np.linspace(0, len(close_indices) - 1, int(config["protocol"]["visual_max_points"]), dtype=np.int64)]
        safe = hashlib.sha256(str(unit["identity"]).encode()).hexdigest()[:16]
        closeup_path = visual_root / "closeups" / f"{config['dataset']}_{safe}_segmentation.png"
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), subplot_kw={"projection": "3d"})
        for row_index, arm in enumerate(("public", "delimit3d")):
            # Both arms use the same deterministic trace filename.
            trace_path = root / arm / "evaluation" / "MO" / "traces" / Path(unit["trace"]).name
            trace = json.loads(trace_path.read_text())
            pred_path = Path(trace["mask_path"])
            with np.load(pred_path) as archive:
                cache = core.load_cache(root, arm, scene)
                inverse = cache["inverse_map"].numpy()
                for column, click in enumerate((1, 5, 15)):
                    token_prediction = _trace_state_prediction(trace, archive, click)
                    prediction = token_prediction[inverse]
                    colors = labels_to_colors(prediction[close_indices], target[close_indices], local_id=local_id)
                    axes[row_index, column].scatter(*points[close_indices].T, c=colors, s=0.5, depthshade=False)
                    axes[row_index, column].set_title(f"{arm}  {click} clicks/object", fontsize=9)
                    axes[row_index, column].set_axis_off()
                    axes[row_index, column].view_init(28, -55)
        fig.tight_layout()
        closeup_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(closeup_path, dpi=220)
        plt.close(fig)
        vertices, faces, keep = mesh_from_points(points[close_indices], normals=normals[close_indices], max_points=int(config["protocol"].get("mesh_max_points", 8000)), knn=30)
        mesh_entry = {"identity": unit["identity"], "scene": scene, "object_instance": object_id, "closeup": str(closeup_path), "closeup_sha256": file_sha256(closeup_path), "meshes": []}
        for arm in ("public", "delimit3d"):
            trace_path = root / arm / "evaluation" / "MO" / "traces" / Path(unit["trace"]).name
            trace = json.loads(trace_path.read_text())
            with np.load(trace["mask_path"]) as archive:
                cache = core.load_cache(root, arm, scene); inverse = cache["inverse_map"].numpy()
                for click in (1, 5, 15):
                    token_prediction = _trace_state_prediction(trace, archive, click)
                    prediction = token_prediction[inverse][close_indices][keep]
                    vertex_target = target[close_indices][keep]
                    colors = labels_to_colors(prediction, vertex_target, local_id=local_id)
                    mesh_path = visual_root / "meshes" / f"{config['dataset']}_{safe}_{arm}_click{click}.ply"
                    mesh_hash = write_mesh_ply(mesh_path, vertices, faces, colors)
                    png_path = mesh_path.with_suffix(".png")
                    fig = plt.figure(figsize=(5, 4))
                    ax = fig.add_subplot(111, projection="3d")
                    _render_axes(ax, vertices, faces, colors, f"{arm}  {click} clicks/object")
                    fig.tight_layout(); fig.savefig(png_path, dpi=220); plt.close(fig)
                    mesh_entry["meshes"].append({"arm": arm, "click": click, "ply": str(mesh_path), "ply_sha256": mesh_hash, "png": str(png_path), "png_sha256": file_sha256(png_path), "vertices": int(len(vertices)), "faces": int(len(faces))})
        outputs.append(mesh_entry)
    dump(visual_root / "visualization_report.json", {"schema": "delimit3d_cross_dataset_visualization/v1", "dataset": config["dataset"], "outputs": outputs, "mesh_method": "Open3D ball-pivoting with SciPy convex-hull fallback", "metric_scope": "descriptive visualization only"})
    return {"outputs": len(outputs), "root": str(visual_root)}


def gpu_preflight() -> dict[str, Any]:
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], text=True)
    print(inventory, flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one bound CUDA GPU is required")
    name = torch.cuda.get_device_name(0)
    if not any(value in name for value in ("RTX 3090", "RTX 4090", "A100")) or "2080" in name:
        raise RuntimeError(f"unapproved GPU: {name}")
    return {"host": platform.node(), "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": name, "nvidia_smi": inventory, "torch": torch.__version__, "cuda": torch.version.cuda}


@contextlib.contextmanager
def worker_files(config: Mapping[str, Any], name: str):
    directory = output_root(config) / "workers"
    directory.mkdir(parents=True, exist_ok=True)
    dump(directory / f"{name}.pid.json", {"pid": os.getpid(), "host": platform.node(), "name": name, "tmux": os.environ.get("TMUX"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    stop = threading.Event()
    heartbeat = directory / f"{name}.heartbeat.jsonl"
    def beat():
        while not stop.is_set():
            with heartbeat.open("a") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "host": platform.node(), "time": time.time(), "name": name}) + "\n")
            stop.wait(30)
    thread = threading.Thread(target=beat, daemon=True); thread.start()
    try:
        yield
    finally:
        stop.set(); thread.join(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("freeze", "prepare", "preflight", "cache-features", "evaluate", "aggregate", "visualize"), required=True)
    parser.add_argument("--arm", choices=("public", "delimit3d"))
    parser.add_argument("--panel", choices=("MO", "SO"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    verify_config(config)
    install_adapter()
    if args.mode == "freeze":
        print(json.dumps(freeze(config), indent=2)); return
    if args.mode == "prepare":
        print(json.dumps(prepare(config), indent=2)); return
    load_provenance(config)
    if args.mode == "preflight":
        print(json.dumps(load_prepared(config)[0], indent=2)); return
    if args.mode == "aggregate":
        print(json.dumps(aggregate(config), indent=2)); return
    if args.mode == "visualize":
        print(json.dumps(visualize(config), indent=2)); return
    if args.mode == "cache-features":
        if args.arm is None: raise ValueError("--arm required")
        gpu_preflight()
        with worker_files(config, f"cache-features_{args.arm}"):
            print(json.dumps(cache_features(config, args.arm), indent=2))
        return
    if args.mode == "evaluate":
        if args.arm is None or args.panel is None: raise ValueError("--arm and --panel required")
        gpu_preflight()
        with worker_files(config, f"evaluate_{args.arm}_{args.panel}"):
            print(json.dumps(evaluate(config, args.arm, args.panel), indent=2))
        return


if __name__ == "__main__":
    main()
