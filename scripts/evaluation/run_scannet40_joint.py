#!/usr/bin/env python3
"""Train a Delimit3D LitePT encoder and an AGILE3D click decoder jointly.

This is a single-arm extension of the completed ScanNet40 matched decoder
protocol.  It deliberately reuses that run's frozen episode schedule and
decoder initialization, but recomputes LitePT dec0 features with gradients on
each update.  No public-LitePT control is launched here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.evaluation import scannet40_protocol as protocol
from delimit3d.evaluation.agile3d_decoder import (
    Agile3DClickDecoder,
    compute_agile3d_losses,
    load_initialization,
    state_sha256,
)
from delimit3d.evaluation.agile3d_protocol import (
    append_clicks,
    array_sha256,
    build_token_targets,
    enforce_click_labels,
    file_sha256,
    json_sha256,
    raw_object_ious,
    scene_seed,
    simulated_corrections,
)
from delimit3d.evaluation.features import load_backbone


EXPERIMENT = "delimit3d_scannet40_agile3d_joint_v1"
MANIFEST_SCHEMA = "delimit3d_scannet40_agile3d_joint_manifest/v1"
CHECKPOINT_SCHEMA = "delimit3d_scannet40_agile3d_joint_checkpoint/v1"
TRAIN_SCHEMA = "delimit3d_scannet40_agile3d_joint_training/v1"
ARM = "delimit3d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("freeze", "prepare", "preflight", "train"), required=True
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve(strict=True).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: YAML root must be a mapping")
    return dict(value)


def resolve(value: str | Path, *, strict: bool = True) -> Path:
    path = Path(str(value)).expanduser()
    if path.exists() or not strict:
        return path.resolve(strict=strict)
    if str(path).startswith("/cluster/"):
        configured = os.environ.get("DELIMIT3D_EULER_MOUNT")
        if configured:
            candidate = Path(configured) / str(path).removeprefix("/cluster/")
            return candidate.resolve(strict=strict)
    return path.resolve(strict=strict)


def dump(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def save_torch(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def effective_commit(config: Mapping[str, Any]) -> str:
    value = str(config.get("repo_commit", "auto"))
    return git_commit() if value in {"", "auto", "unknown"} else value


def root_of(config: Mapping[str, Any]) -> Path:
    root = resolve(config["paths"]["output_root"], strict=False)
    if EXPERIMENT not in str(root):
        raise ValueError(f"refusing non-joint artifact root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def decoder_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    decoder = dict(config["decoder"])
    allowed = {
        "feature_dim",
        "hidden_dim",
        "num_heads",
        "dim_feedforward",
        "num_decoders",
        "num_bg_queries",
        "dropout",
        "pre_norm",
        "max_click_events",
        "normalize_pos_enc",
        "gauss_scale",
        "aux",
    }
    return {key: decoder[key] for key in allowed}


def verify_config(config: Mapping[str, Any]) -> None:
    required = {
        "experiment_id",
        "seed",
        "paths",
        "encoder",
        "protocol",
        "decoder",
        "optimizer",
        "precision",
        "loss",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing {missing}")
    if str(config["experiment_id"]) != EXPERIMENT:
        raise ValueError("unexpected experiment_id")
    if config.get("encoder_adaptation_uses_scannet_labels") is not True:
        raise ValueError("joint encoder training must be explicitly label-supervised")
    if config["encoder"].get("trainable") is not True:
        raise ValueError("joint run requires encoder.trainable=true")
    protocol_config = config["protocol"]
    expected_protocol = {
        "train_updates": 5000,
        "click_budget": 20,
        "active_litept_level": "dec0",
        "voxel_size_m": 0.02,
        "voxel_reduce": "representative",
        "representative_sampling": "first",
        "click_center_method": "kdtree",
    }
    for key, expected in expected_protocol.items():
        if protocol_config.get(key) != expected:
            raise ValueError(f"protocol.{key}={protocol_config.get(key)!r}, expected {expected!r}")
    expected_decoder = {
        "feature_dim": 72,
        "hidden_dim": 128,
        "num_heads": 8,
        "dim_feedforward": 1024,
        "num_decoders": 3,
        "num_bg_queries": 10,
        "dropout": 0.0,
        "pre_norm": False,
        "max_click_events": 200,
        "normalize_pos_enc": True,
        "gauss_scale": 1.0,
        "aux": True,
    }
    actual_decoder = decoder_kwargs(config)
    for key, expected in expected_decoder.items():
        if actual_decoder.get(key) != expected:
            raise ValueError(f"decoder.{key}={actual_decoder.get(key)!r}, expected {expected!r}")
    optimizer = config["optimizer"]
    if optimizer.get("gpu") not in {"RTX 4090", "RTX4090"}:
        raise ValueError("joint protocol is locked to RTX 4090")
    if int(optimizer.get("gpu_count", 0)) != 1:
        raise ValueError("joint protocol requires one GPU")
    if int(optimizer.get("batch_size", 0)) != 1:
        raise ValueError("joint protocol requires batch_size=1")
    learning_rates = [
        float(optimizer.get("learning_rate", 0.0)),
        float(optimizer.get("encoder_lr", 0.0)),
        float(optimizer.get("decoder_lr", 0.0)),
    ]
    if learning_rates != [1.0e-4, 1.0e-4, 1.0e-4]:
        raise ValueError(f"encoder and decoder learning rates must all be 1e-4, got {learning_rates}")
    if float(optimizer.get("weight_decay", 0.0)) != 1.0e-4:
        raise ValueError("weight_decay must be 1e-4")
    if float(optimizer.get("clip_norm", 0.0)) != 0.1:
        raise ValueError("clip_norm must be 0.1")
    if str(config["precision"].get("dtype")) != "float16":
        raise ValueError("the single-4090 joint path currently requires float16 AMP")
    if config["precision"].get("amp") is not True:
        raise ValueError("the single-4090 joint path requires AMP")
    cache_capacity = int(optimizer.get("cache_memory_scenes", 0))
    if cache_capacity < 1:
        raise ValueError("cache_memory_scenes must be positive")


def _copy_tree(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or "build" in path.parts:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != file_sha256(path):
            raise RuntimeError(f"refusing to overwrite changed dependency file: {target}")
        if not target.exists():
            shutil.copy2(path, target)


def _dependency_entries(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze this source, the parent schedule, and the LitePT dependency."""
    verify_config(config)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("commit the joint worktree before freezing")

    parent = resolve(config["paths"]["parent_run_root"])
    parent_manifest_path = parent / "selection_manifest.json"
    parent_manifest_sha = file_sha256(parent_manifest_path)
    if parent_manifest_sha != str(config["parent_manifest_sha256"]):
        raise ValueError("parent selection manifest hash does not match the declared protocol")
    parent_init = parent / "decoder_init.pt"
    if file_sha256(parent_init) != str(config["parent_decoder_initialization_file_sha256"]):
        raise ValueError("parent decoder initialization hash does not match the declaration")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if len(parent_manifest.get("train_scenes", [])) != 1200:
        raise ValueError("parent manifest is not the official 1,200-scene training protocol")
    if len(parent_manifest.get("validation_scenes", [])) != 312:
        raise ValueError("parent manifest is not the official 312-scene validation protocol")
    if len(parent_manifest.get("panels", {}).get("MO", [])) != 312:
        raise ValueError("parent manifest is missing the official MO panel")
    if len(parent_manifest.get("panels", {}).get("SO", [])) != 10357:
        raise ValueError("parent manifest is missing the official SO panel")

    source_commit = git_commit()
    root = root_of(config)
    freeze_root = root / "freeze"
    freeze_root.mkdir(parents=True, exist_ok=True)
    frozen_dependency = freeze_root / "litept_source"
    parent_dependency = parent / "freeze" / "litept_source"
    _copy_tree(parent_dependency, frozen_dependency)
    dependency_entries = _dependency_entries(frozen_dependency)
    dependency_archive = freeze_root / "litept_source.tar"
    if not dependency_archive.exists():
        with tarfile.open(dependency_archive, "w") as archive:
            for name in dependency_entries:
                archive.add(frozen_dependency / name, arcname=name, recursive=False)

    resolved = copy.deepcopy(dict(config))
    resolved["repo_commit"] = source_commit
    resolved["paths"] = dict(resolved["paths"])
    resolved["paths"]["litept_root"] = str(frozen_dependency)
    resolved["parent_manifest_sha256"] = parent_manifest_sha
    resolved["parent_decoder_initialization_file_sha256"] = file_sha256(parent_init)
    resolved_path = freeze_root / "resolved_config.yaml"
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    if resolved_path.exists() and resolved_path.read_text() != resolved_text:
        raise RuntimeError("frozen resolved YAML differs; use a new artifact root")
    if not resolved_path.exists():
        resolved_path.write_text(resolved_text)

    decoder_init = root / "decoder_init.pt"
    if decoder_init.exists() and file_sha256(decoder_init) != file_sha256(parent_init):
        raise RuntimeError("joint decoder initialization already exists with another hash")
    if not decoder_init.exists():
        shutil.copy2(parent_init, decoder_init)
    _decoder, init_report = load_initialization(
        decoder_init, expected_kwargs=decoder_kwargs(resolved)
    )

    source_archive = freeze_root / f"source_{source_commit}.tar"
    if not source_archive.exists():
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={source_archive}", source_commit],
            cwd=REPO_ROOT,
            check=True,
        )
    source_entries = {
        name: file_sha256(REPO_ROOT / name)
        for name in subprocess.check_output(
            ["git", "ls-files"], cwd=REPO_ROOT, text=True
        ).splitlines()
        if (REPO_ROOT / name).is_file()
    }
    environment = {
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    dump(freeze_root / "environment_cpu.json", environment)
    encoder_checkpoint = resolve(resolved["encoder"]["checkpoint"])
    encoder_checkpoint_sha = file_sha256(encoder_checkpoint)
    if encoder_checkpoint_sha != str(resolved["encoder"]["checkpoint_sha256"]):
        raise ValueError("Delimit3D encoder checkpoint hash mismatch")
    provenance = {
        "schema": "delimit3d_scannet40_agile3d_joint_provenance/v1",
        "experiment_id": EXPERIMENT,
        "repo_commit": source_commit,
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": file_sha256(resolved_path),
        "source_archive": str(source_archive),
        "source_archive_sha256": file_sha256(source_archive),
        "source_entries": source_entries,
        "decoder_initialization": init_report,
        "decoder_initialization_file_sha256": file_sha256(decoder_init),
        "encoder_checkpoint": str(encoder_checkpoint),
        "encoder_checkpoint_sha256": encoder_checkpoint_sha,
        "parent_run_root": str(parent),
        "parent_manifest_sha256": parent_manifest_sha,
        "parent_decoder_initialization_file_sha256": file_sha256(parent_init),
        "litept_dependency": {
            "source_root": str(parent_dependency),
            "frozen_root": str(frozen_dependency),
            "source_entries": dependency_entries,
            "archive": str(dependency_archive),
            "archive_sha256": file_sha256(dependency_archive),
        },
        "environment": str(freeze_root / "environment_cpu.json"),
        "environment_sha256": file_sha256(freeze_root / "environment_cpu.json"),
    }
    dump(freeze_root / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)
    return provenance


def verify_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    root = root_of(config)
    provenance_path = root / "freeze" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if str(config.get("repo_commit")) != provenance["repo_commit"]:
        raise ValueError("runtime config commit differs from frozen provenance")
    if file_sha256(root / "freeze" / "resolved_config.yaml") != provenance["resolved_config_sha256"]:
        raise ValueError("resolved configuration changed")
    if file_sha256(resolve(provenance["source_archive"])) != provenance["source_archive_sha256"]:
        raise ValueError("source archive changed")
    if file_sha256(root / "decoder_init.pt") != provenance["decoder_initialization_file_sha256"]:
        raise ValueError("decoder initialization changed")
    for name, expected in provenance["source_entries"].items():
        path = REPO_ROOT / name
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"executing source differs from frozen source: {name}")
    encoder_checkpoint = resolve(config["encoder"]["checkpoint"])
    if file_sha256(encoder_checkpoint) != provenance["encoder_checkpoint_sha256"]:
        raise ValueError("Delimit3D encoder checkpoint changed")
    dependency = provenance["litept_dependency"]
    if file_sha256(resolve(dependency["archive"])) != dependency["archive_sha256"]:
        raise ValueError("frozen LitePT dependency archive changed")
    for name, expected in dependency["source_entries"].items():
        path = resolve(config["paths"]["litept_root"]) / name
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"frozen LitePT dependency differs: {name}")
    frozen = yaml.safe_load(resolve(provenance["resolved_config"]).read_text())
    if json_sha256(config) != json_sha256(frozen):
        raise ValueError("runtime config differs from frozen resolved YAML")
    parent_manifest = resolve(config["paths"]["parent_run_root"]) / "selection_manifest.json"
    if file_sha256(parent_manifest) != str(config["parent_manifest_sha256"]):
        raise ValueError("parent selection manifest changed")
    return provenance


def _load_parent_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = resolve(config["paths"]["parent_run_root"])
    path = parent / "selection_manifest.json"
    manifest = json.loads(path.read_text())
    expected = manifest.pop("manifest_sha256", None)
    if expected is None or json_sha256(manifest) != expected:
        raise ValueError("parent selection manifest hash mismatch")
    manifest["manifest_sha256"] = expected
    if len(manifest.get("train_scenes", [])) != 1200 or len(manifest.get("validation_scenes", [])) != 312:
        raise ValueError("parent manifest scene coverage mismatch")
    schedule = resolve(manifest["train_episodes_path"])
    if file_sha256(schedule) != manifest["train_episodes_sha256"]:
        raise ValueError("parent episode schedule changed")
    return manifest


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_freeze(config)
    root = root_of(config)
    manifest_path = root / "selection_manifest.json"
    if manifest_path.exists():
        return load_prepared(config)
    parent_root = resolve(config["paths"]["parent_run_root"])
    parent = _load_parent_manifest(config)
    schedule_source = resolve(parent["train_episodes_path"])
    schedule_path = root / "train_episodes.jsonl"
    shutil.copy2(schedule_source, schedule_path)
    parent_init = parent_root / "decoder_init.pt"
    init_path = root / "decoder_init.pt"
    if file_sha256(init_path) != file_sha256(parent_init):
        raise ValueError("copied decoder initialization does not match parent")
    _decoder, init_report = load_initialization(
        init_path, expected_kwargs=decoder_kwargs(config)
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT,
        "repo_commit": effective_commit(config),
        "parent_manifest_sha256": str(config["parent_manifest_sha256"]),
        "train_scenes": parent["train_scenes"],
        "validation_scenes": parent["validation_scenes"],
        "panels": parent["panels"],
        "train_episodes_path": str(schedule_path),
        "train_episodes_sha256": file_sha256(schedule_path),
        "decoder_initialization_path": str(init_path),
        "decoder_initialization": init_report,
        "protocol": dict(config["protocol"]),
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    dump(manifest_path, manifest)
    dump(
        root / "prepare_report.json",
        {
            "schema": "delimit3d_scannet40_agile3d_joint_prepare/v1",
            "train_scenes": len(manifest["train_scenes"]),
            "validation_scenes": len(manifest["validation_scenes"]),
            "MO": len(manifest["panels"]["MO"]),
            "SO": len(manifest["panels"]["SO"]),
            "updates": sum(1 for _ in schedule_path.open()),
            "parent_manifest_sha256": manifest["parent_manifest_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def load_prepared(config: Mapping[str, Any]) -> dict[str, Any]:
    root = root_of(config)
    path = root / "selection_manifest.json"
    manifest = json.loads(path.read_text())
    expected = manifest.pop("manifest_sha256", None)
    if expected is None or json_sha256(manifest) != expected:
        raise ValueError("joint selection manifest hash mismatch")
    manifest["manifest_sha256"] = expected
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("experiment_id") != EXPERIMENT:
        raise ValueError("joint manifest schema or experiment mismatch")
    if manifest.get("repo_commit") != effective_commit(config):
        raise ValueError("joint manifest commit mismatch")
    if manifest.get("parent_manifest_sha256") != str(config["parent_manifest_sha256"]):
        raise ValueError("joint manifest parent mismatch")
    schedule = resolve(manifest["train_episodes_path"])
    if file_sha256(schedule) != manifest["train_episodes_sha256"]:
        raise ValueError("joint episode schedule changed")
    init = resolve(manifest["decoder_initialization_path"])
    _decoder, report = load_initialization(init, expected_kwargs=decoder_kwargs(config))
    if report["tensor_state_sha256"] != manifest["decoder_initialization"]["tensor_state_sha256"]:
        raise ValueError("joint decoder initialization tensor hash changed")
    return manifest


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _normalise_normals(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


class SceneInputCache:
    """CPU cache for verified raw RGBN6 inputs; GPU holds one scene at a time."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.values: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, record: Mapping[str, Any]) -> dict[str, Any]:
        scene = str(record["scene"])
        if scene in self.values:
            self.values.move_to_end(scene)
            return self.values[scene]
        points, colors, normals, objects, masks = protocol.load_scene(record, resolve)
        shifted, _shift = _center_shift_numpy(points)
        features = build_input_features(
            points,
            colors,
            use_colors=True,
            use_normals=True,
            normals=_normalise_normals(normals),
        )
        value = {
            "scene": scene,
            "coord": np.ascontiguousarray(shifted, dtype=np.float32),
            "features": np.ascontiguousarray(features, dtype=np.float32),
            "masks": {int(key): np.asarray(indices, dtype=np.int64) for key, indices in masks.items()},
            "objects": [dict(item) for item in objects],
        }
        self.values[scene] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def _center_shift_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float32)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    shift = np.asarray(
        [(minimum[0] + maximum[0]) / 2.0, (minimum[1] + maximum[1]) / 2.0, minimum[2]],
        dtype=np.float32,
    )
    return (values - shift).astype(np.float32, copy=False), shift


def _records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["scene"]): dict(record)
        for record in list(manifest["train_scenes"]) + list(manifest["validation_scenes"])
    }


def _amp_context(config: Mapping[str, Any]):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(config["precision"]["amp"]),
    )


def _token_context(
    encoder: torch.nn.Module,
    raw: Mapping[str, Any],
    record: Mapping[str, Any],
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[int]], dict[str, list[int]]]:
    coord = torch.from_numpy(raw["coord"]).to(device=device, dtype=torch.float32)
    features_in = torch.from_numpy(raw["features"]).to(device=device, dtype=torch.float32)
    with _amp_context(config):
        output = encoder(coord, features_in)
    tokens = output.scene_tokens
    scene_xyz = output.scene_xyz.detach()
    inverse = output.inverse_map.detach().cpu()
    representatives = output.representative_indices.detach().cpu()
    del output, coord, features_in
    if tokens.ndim != 2 or tokens.shape[1] != 72:
        raise RuntimeError(f"{record['scene']}: unexpected dec0 shape {tuple(tokens.shape)}")
    if scene_xyz.shape != (tokens.shape[0], 3):
        raise RuntimeError(f"{record['scene']}: token coordinate shape mismatch")
    if representatives.shape != (tokens.shape[0],):
        raise RuntimeError(f"{record['scene']}: representative shape mismatch")
    expected_hash = record.get("representative_indices_sha256")
    if expected_hash and array_sha256(representatives.numpy()) != expected_hash:
        raise RuntimeError(f"{record['scene']}: representative geometry changed")
    expected_count = record.get("representative_token_count")
    if expected_count is not None and int(tokens.shape[0]) != int(expected_count):
        raise RuntimeError(f"{record['scene']}: representative token count changed")

    object_ids = [int(value) for value in episode["objects"]]
    memberships = build_token_targets(representatives.numpy(), raw["masks"], object_ids)
    target_cpu = torch.zeros(tokens.shape[0], dtype=torch.long)
    for local_id, object_id in enumerate(object_ids, start=1):
        indices = torch.from_numpy(memberships[object_id]).long()
        if indices.numel() == 0:
            raise RuntimeError(f"{record['scene']}: requested object {object_id} has no token")
        if bool(torch.any(target_cpu[indices] != 0)):
            raise RuntimeError(f"{record['scene']}: selected object tuple overlaps in token space")
        target_cpu[indices] = local_id
    seed = scene_seed(
        int(config["seed"]),
        str(record["scene"]),
        "initial:" + ",".join(str(value) for value in object_ids),
    )
    clicks, times = protocol.initial_clicks(
        target_cpu.numpy(), scene_xyz.detach().cpu().numpy(), seed
    )
    return (
        tokens,
        scene_xyz,
        inverse,
        representatives,
        target_cpu.to(device=device),
        clicks,
        times,
    )


def _prefix_interaction(
    decoder: Agile3DClickDecoder,
    tokens: torch.Tensor,
    scene_xyz: torch.Tensor,
    target: torch.Tensor,
    clicks: dict[str, list[int]],
    times: dict[str, list[int]],
    rounds: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    decoder.eval()
    with torch.inference_mode():
        for _ in range(int(rounds)):
            with _amp_context(config):
                output = decoder(tokens, scene_xyz, clicks=clicks, click_times=times)
            prediction = output["pred_masks"].argmax(dim=1).cpu().numpy()
            prediction = enforce_click_labels(prediction, clicks)
            new_clicks, new_times, selected = simulated_corrections(
                prediction,
                target.cpu().numpy(),
                scene_xyz.cpu().numpy(),
                clicks,
                times,
                training=True,
                click_center_method=str(config["protocol"]["click_center_method"]),
            )
            if not new_clicks:
                break
            clicks, times = append_clicks(clicks, times, new_clicks, new_times)
            events.extend(selected)
    decoder.train()
    return clicks, times, events


def _grad_l2(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float()
            total += float(torch.sum(value * value).item())
    return float(total**0.5)


def _train_one_update(
    encoder: torch.nn.Module,
    decoder: Agile3DClickDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    raw: Mapping[str, Any],
    record: Mapping[str, Any],
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    encoder.train()
    decoder.train()
    (
        tokens,
        scene_xyz,
        _inverse,
        _representatives,
        target,
        clicks,
        times,
    ) = _token_context(encoder, raw, record, episode, config, device)
    clicks, times, prefix_events = _prefix_interaction(
        decoder,
        tokens,
        scene_xyz,
        target,
        clicks,
        times,
        int(episode["click_prefix"]),
        config,
    )
    with _amp_context(config):
        output = decoder(tokens, scene_xyz, clicks=clicks, click_times=times)
    loss, details = compute_agile3d_losses(
        output,
        target,
        scene_xyz=scene_xyz,
        clicks=clicks,
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
        alpha=float(config["loss"]["alpha"]),
        beta=float(config["loss"]["beta"]),
        radius=float(config["loss"]["radius"]),
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite joint loss: {loss}")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    decoder_parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    encoder_grad_norm = _grad_l2(encoder_parameters)
    decoder_grad_norm = _grad_l2(decoder_parameters)
    all_parameters = encoder_parameters + decoder_parameters
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        all_parameters, float(config["optimizer"]["clip_norm"])
    )
    if not np.isfinite(float(clipped_norm.detach().cpu() if torch.is_tensor(clipped_norm) else clipped_norm)):
        raise FloatingPointError("non-finite joint gradient norm")
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach().float().cpu()),
        "loss_bce": float(details["loss_bce"].detach().float().cpu()),
        "loss_dice": float(details["loss_dice"].detach().float().cpu()),
        "grad_norm": float(clipped_norm.detach().float().cpu() if torch.is_tensor(clipped_norm) else clipped_norm),
        "encoder_grad_norm": encoder_grad_norm,
        "decoder_grad_norm": decoder_grad_norm,
        "objects": int(len(episode["objects"])),
        "prefix_rounds": int(episode["click_prefix"]),
        "prefix_events": int(len(prefix_events)),
        "tokens": int(tokens.shape[0]),
        "amp_scale": float(scaler.get_scale()),
    }


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous().clone() for key, value in module.state_dict().items()}


def _parameter_hash(module: torch.nn.Module) -> str:
    return state_sha256({key: value for key, value in module.named_parameters()})


def _make_optimizer(
    encoder: torch.nn.Module,
    decoder: Agile3DClickDecoder,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    decoder_parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    if len(encoder_parameters) == 0 or len(decoder_parameters) == 0:
        raise RuntimeError("joint optimizer requires trainable encoder and decoder parameters")
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "encoder",
                "params": encoder_parameters,
                "lr": float(config["optimizer"]["encoder_lr"]),
                "weight_decay": float(config["optimizer"]["weight_decay"]),
            },
            {
                "name": "decoder",
                "params": decoder_parameters,
                "lr": float(config["optimizer"]["decoder_lr"]),
                "weight_decay": float(config["optimizer"]["weight_decay"]),
            },
        ]
    )
    parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected_ids = {id(parameter) for parameter in encoder_parameters + decoder_parameters}
    if parameter_ids != expected_ids:
        raise RuntimeError("joint optimizer does not own exactly all encoder and decoder parameters")
    if [group.get("name") for group in optimizer.param_groups] != ["encoder", "decoder"]:
        raise RuntimeError("joint optimizer groups are not ordered encoder, decoder")
    if any(group["lr"] != 1.0e-4 for group in optimizer.param_groups):
        raise RuntimeError("joint optimizer encoder and decoder learning rates differ from 1e-4")
    return optimizer


def _checkpoint_payload(
    *,
    encoder: torch.nn.Module,
    decoder: Agile3DClickDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    update: int,
    encoder_checkpoint_sha256: str,
    encoder_initial_state_sha256: str,
    encoder_initial_parameter_sha256: str,
    decoder_initialization_sha256: str,
    recent_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    encoder_state = _cpu_state(encoder)
    decoder_state = _cpu_state(decoder)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT,
        "arm": ARM,
        "update": int(update),
        "repo_commit": effective_commit(config),
        "decoder_kwargs": decoder_kwargs(config),
        "encoder_state_dict": encoder_state,
        "decoder_state_dict": decoder_state,
        "encoder_state_sha256": state_sha256(encoder_state),
        "encoder_parameter_sha256": state_sha256(
            {key: encoder_state[key] for key, _value in encoder.named_parameters()}
        ),
        "decoder_tensor_sha256": state_sha256(decoder_state),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": rng_state(),
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "encoder_initial_state_sha256": encoder_initial_state_sha256,
        "encoder_initial_parameter_sha256": encoder_initial_parameter_sha256,
        "decoder_initialization_sha256": decoder_initialization_sha256,
        "encoder_trainable": True,
        "encoder_frozen": False,
        "encoder_optimizer_parameter_count": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "decoder_optimizer_parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
        ),
        "optimizer_param_group_names": [group.get("name") for group in optimizer.param_groups],
        "optimizer_param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "precision": dict(config["precision"]),
        "episode_schedule_sha256": manifest["train_episodes_sha256"],
        "recent_train_stats": list(recent_stats[-100:]),
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> dict[str, Any]:
    payload = _checkpoint_payload(**kwargs)
    save_torch(path, payload)
    return payload


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    encoder: torch.nn.Module,
    decoder: Agile3DClickDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    encoder_checkpoint_sha256: str,
    encoder_initial_state_sha256: str,
    encoder_initial_parameter_sha256: str,
    decoder_initialization_sha256: str,
) -> int:
    required = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT,
        "arm": ARM,
        "repo_commit": effective_commit(config),
        "decoder_kwargs": decoder_kwargs(config),
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "encoder_initial_state_sha256": encoder_initial_state_sha256,
        "encoder_initial_parameter_sha256": encoder_initial_parameter_sha256,
        "decoder_initialization_sha256": decoder_initialization_sha256,
        "episode_schedule_sha256": manifest["train_episodes_sha256"],
        "encoder_trainable": True,
        "encoder_frozen": False,
        "precision": dict(config["precision"]),
    }
    for key, value in required.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint mismatch: {key}")
    update = int(checkpoint.get("update", 0))
    if not 0 < update < int(config["protocol"]["train_updates"]):
        raise ValueError("resume checkpoint update must be inside the unfinished schedule")
    encoder_state = checkpoint.get("encoder_state_dict")
    decoder_state = checkpoint.get("decoder_state_dict")
    if not isinstance(encoder_state, Mapping) or not isinstance(decoder_state, Mapping):
        raise ValueError("resume checkpoint is missing encoder or decoder state")
    if state_sha256(encoder_state) != checkpoint.get("encoder_state_sha256"):
        raise ValueError("resume encoder tensor hash mismatch")
    if state_sha256(decoder_state) != checkpoint.get("decoder_tensor_sha256"):
        raise ValueError("resume decoder tensor hash mismatch")
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    saved_optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(saved_optimizer, Mapping):
        raise ValueError("resume checkpoint is missing optimizer state")
    if saved_optimizer.get("param_groups") and [
        group.get("name") for group in saved_optimizer["param_groups"]
    ] != ["encoder", "decoder"]:
        raise ValueError("resume optimizer groups are not encoder, decoder")
    optimizer.load_state_dict(saved_optimizer)
    if [group.get("name") for group in optimizer.param_groups] != ["encoder", "decoder"]:
        raise ValueError("restored optimizer groups are not encoder, decoder")
    if any(float(group["lr"]) != 1.0e-4 for group in optimizer.param_groups):
        raise ValueError("restored optimizer learning rates are not equal 1e-4")
    scaler_state = checkpoint.get("scaler_state_dict")
    if not isinstance(scaler_state, Mapping):
        raise ValueError("resume checkpoint is missing AMP scaler state")
    scaler.load_state_dict(scaler_state)
    saved_rng = checkpoint.get("rng_state")
    if not isinstance(saved_rng, Mapping) or set(("python", "numpy", "torch", "cuda")) - set(saved_rng):
        raise ValueError("resume checkpoint is missing complete RNG state")
    restore_rng(saved_rng)
    return update


def startup_gradient_check(
    *,
    encoder: torch.nn.Module,
    decoder: Agile3DClickDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    raw: Mapping[str, Any],
    record: Mapping[str, Any],
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    encoder_state = _cpu_state(encoder)
    decoder_state = _cpu_state(decoder)
    saved_rng = rng_state()
    torch.cuda.reset_peak_memory_stats(device)
    values = _train_one_update(
        encoder, decoder, optimizer, scaler, raw, record, episode, config, device
    )
    if float(values["encoder_grad_norm"]) <= 0.0:
        raise RuntimeError("startup gradient check found no encoder gradient")
    if float(values["decoder_grad_norm"]) <= 0.0:
        raise RuntimeError("startup gradient check found no decoder gradient")
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    restore_rng(saved_rng)
    del encoder_state, decoder_state
    return {
        "schema": "delimit3d_scannet40_agile3d_joint_gradient_check/v1",
        "passed": True,
        "update_was_not_committed": True,
        "values": values,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def gpu_preflight() -> dict[str, Any]:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        text=True,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("joint run requires exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in name:
        raise RuntimeError(f"joint run requires RTX 4090, got {name}")
    report = {
        "host": platform.node(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": name,
        "nvidia_smi": inventory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
    }
    print(inventory, flush=True)
    return report


@contextlib.contextmanager
def heartbeat(config: Mapping[str, Any]):
    directory = root_of(config) / "workers"
    directory.mkdir(parents=True, exist_ok=True)
    name = "joint_train"
    dump(
        directory / f"{name}.pid.json",
        {
            "pid": os.getpid(),
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    stop = threading.Event()

    def beat() -> None:
        interval = int(config.get("runtime", {}).get("heartbeat_seconds", 30))
        while not stop.is_set():
            with (directory / f"{name}.heartbeat.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "pid": os.getpid(),
                            "host": platform.node(),
                            "update": None,
                        }
                    )
                    + "\n"
                )
            stop.wait(interval)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def train(config: Mapping[str, Any], resume: Path | None = None) -> dict[str, Any]:
    provenance = verify_freeze(config)
    manifest = load_prepared(config)
    root = root_of(config)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    encoder_checkpoint = resolve(config["encoder"]["checkpoint"])
    encoder_checkpoint_sha256 = file_sha256(encoder_checkpoint)
    encoder = load_backbone(
        encoder_checkpoint,
        litept_root=resolve(config["paths"]["litept_root"]),
        in_channels=6,
        device=device,
    )
    encoder.requires_grad_(True)
    encoder.train()
    if not all(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("joint encoder contains a frozen parameter")
    decoder, init_report = load_initialization(
        resolve(manifest["decoder_initialization_path"]),
        expected_kwargs=decoder_kwargs(config),
    )
    decoder.to(device)
    decoder.train()
    set_seed(int(config["seed"]))
    encoder_initial_state_sha256 = state_sha256(encoder.state_dict())
    encoder_initial_parameter_sha256 = _parameter_hash(encoder)
    decoder_initialization_sha256 = init_report["tensor_state_sha256"]
    optimizer = _make_optimizer(encoder, decoder, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["precision"]["amp"]))
    begin = 0
    if resume is not None:
        checkpoint = torch.load(resolve(resume), map_location="cpu", weights_only=False)
        begin = _validate_resume(
            checkpoint,
            config=config,
            manifest=manifest,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scaler=scaler,
            encoder_checkpoint_sha256=encoder_checkpoint_sha256,
            encoder_initial_state_sha256=encoder_initial_state_sha256,
            encoder_initial_parameter_sha256=encoder_initial_parameter_sha256,
            decoder_initialization_sha256=decoder_initialization_sha256,
        )
    output = root / ARM
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"
    if log_path.exists() and resume is None:
        raise RuntimeError(f"existing joint training log requires --resume: {log_path}")
    episodes = read_jsonl(resolve(manifest["train_episodes_path"]))
    if len(episodes) != int(config["protocol"]["train_updates"]):
        raise RuntimeError("joint episode schedule length differs from 5,000 updates")
    records = _records(manifest)
    scene_cache = SceneInputCache(int(config["optimizer"]["cache_memory_scenes"]))

    if begin == 0 and bool(config.get("runtime", {}).get("startup_gradient_check", False)):
        first = episodes[0]
        first_record = records[str(first["scene"])]
        check = startup_gradient_check(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scaler=scaler,
            raw=scene_cache.get(first_record),
            record=first_record,
            episode=first,
            config=config,
            device=device,
        )
        dump(root / "startup_gradient_check.json", check)
        del optimizer, scaler
        optimizer = _make_optimizer(encoder, decoder, config)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config["precision"]["amp"]))

    stats: list[dict[str, Any]] = []
    start = time.time()
    log_mode = "a" if resume is not None else "w"
    with log_path.open(log_mode) as log:
        for episode in episodes[begin:]:
            record = records[str(episode["scene"])]
            values = _train_one_update(
                encoder,
                decoder,
                optimizer,
                scaler,
                scene_cache.get(record),
                record,
                episode,
                config,
                device,
            )
            row = {
                "update": int(episode["update"]),
                "scene": str(episode["scene"]),
                "arm": ARM,
                **values,
                "seconds": float(time.time() - start),
            }
            stats.append(row)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            if int(episode["update"]) % 100 == 0 or int(episode["update"]) == begin + 1:
                log.flush()
                print(json.dumps({"mode": "joint_train", **row}), flush=True)
            update = int(episode["update"])
            if update % 250 == 0 or update == len(episodes):
                _save_checkpoint(
                    output / f"joint_u{update:05d}.pt",
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    manifest=manifest,
                    update=update,
                    encoder_checkpoint_sha256=encoder_checkpoint_sha256,
                    encoder_initial_state_sha256=encoder_initial_state_sha256,
                    encoder_initial_parameter_sha256=encoder_initial_parameter_sha256,
                    decoder_initialization_sha256=decoder_initialization_sha256,
                    recent_stats=stats,
                )

    final_path = output / f"joint_u{len(episodes):05d}.pt"
    if not final_path.exists():
        _save_checkpoint(
            final_path,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            manifest=manifest,
            update=len(episodes),
            encoder_checkpoint_sha256=encoder_checkpoint_sha256,
            encoder_initial_state_sha256=encoder_initial_state_sha256,
            encoder_initial_parameter_sha256=encoder_initial_parameter_sha256,
            decoder_initialization_sha256=decoder_initialization_sha256,
            recent_stats=stats,
        )
    encoder_after_state_sha256 = state_sha256(encoder.state_dict())
    encoder_after_parameter_sha256 = _parameter_hash(encoder)
    report = {
        "schema": TRAIN_SCHEMA,
        "experiment_id": EXPERIMENT,
        "arm": ARM,
        "repo_commit": effective_commit(config),
        "provenance_sha256": file_sha256(root / "freeze" / "provenance.json"),
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
        "updates": len(episodes),
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "encoder_initial_state_sha256": encoder_initial_state_sha256,
        "encoder_initial_parameter_sha256": encoder_initial_parameter_sha256,
        "encoder_state_sha256_after": encoder_after_state_sha256,
        "encoder_parameter_sha256_after": encoder_after_parameter_sha256,
        "encoder_state_changed": encoder_after_state_sha256 != encoder_initial_state_sha256,
        "encoder_parameters_changed": encoder_after_parameter_sha256 != encoder_initial_parameter_sha256,
        "encoder_trainable": True,
        "encoder_frozen": False,
        "encoder_optimizer_parameter_count": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "decoder_optimizer_parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
        ),
        "optimizer_param_group_names": [group.get("name") for group in optimizer.param_groups],
        "optimizer_param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "decoder_initialization_sha256": decoder_initialization_sha256,
        "episode_schedule_sha256": manifest["train_episodes_sha256"],
        "precision": dict(config["precision"]),
        "resume_update": int(begin),
        "seconds": float(time.time() - start),
        "last_stats": stats[-10:],
        "source_archive_sha256": provenance["source_archive_sha256"],
    }
    dump(output / "train_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    verify_config(config)
    if args.mode == "freeze":
        freeze(config)
        return
    if args.mode == "prepare":
        prepare(config)
        return
    verify_freeze(config)
    if args.mode == "preflight":
        print(json.dumps(verify_freeze(config), indent=2), flush=True)
        return
    gpu = gpu_preflight()
    dump(root_of(config) / "workers" / "environment_joint_train.json", gpu)
    with heartbeat(config):
        train(config, args.resume)


if __name__ == "__main__":
    main()
