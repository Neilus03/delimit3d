#!/usr/bin/env python3
"""Evaluate the completed joint LitePT + AGILE3D ScanNet40 checkpoint.

The decoder-only evaluator cannot be reused for this arm because its feature
cache was produced by a frozen encoder.  This runner loads the trained
encoder and decoder, recomputes one verified RGBN6/dec0 feature cache per
validation scene, and then applies the same official MO/SO click protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
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
import yaml

import run_scannet40_joint as joint


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EXPERIMENT = "delimit3d_scannet40_agile3d_joint_v1"
EVALUATION_SCHEMA = "delimit3d_scannet40_agile3d_joint_evaluation/v1"
EVALUATION_MANIFEST_SCHEMA = "delimit3d_scannet40_agile3d_joint_evaluation_manifest/v1"
FEATURE_CACHE_SCHEMA = "delimit3d_scannet40_joint_feature_cache/v1"


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


def dump(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_torch(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    joint.torch.save(payload, temporary)
    os.replace(temporary, path)


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


def root_of(config: Mapping[str, Any]) -> Path:
    root = resolve(config["paths"]["output_root"], strict=False)
    if EXPERIMENT not in str(root):
        raise ValueError(f"refusing non-joint evaluation artifact root: {root}")
    training_root = resolve(config["paths"]["training_root"], strict=False)
    if root == training_root:
        raise ValueError("evaluation output must not overwrite the immutable training root")
    root.mkdir(parents=True, exist_ok=True)
    return root


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


def verify_config(config: Mapping[str, Any]) -> None:
    # The shared verifier locks the exact training/decoder/input protocol.
    joint.verify_config(config)
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation section is required")
    panels = list(evaluation.get("panels", []))
    if panels != ["MO", "SO"]:
        raise ValueError("joint evaluation must run MO then SO")
    checkpoint = evaluation.get("checkpoint")
    checkpoint_sha256 = evaluation.get("checkpoint_sha256")
    if not checkpoint or not checkpoint_sha256:
        raise ValueError("evaluation checkpoint and checkpoint_sha256 are required")
    training_root = config.get("paths", {}).get("training_root")
    if not training_root:
        raise ValueError("paths.training_root is required")
    if Path(str(training_root)).expanduser() == Path(
        str(config["paths"]["output_root"])
    ).expanduser():
        raise ValueError("training and evaluation roots must differ")


def _load_canonical_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    expected = payload.pop(field, None)
    if not expected or json_sha256(payload) != expected:
        raise ValueError(f"{path}: canonical {field} mismatch")
    payload[field] = expected
    return payload, str(expected)


def _load_training_provenance(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    training_root = resolve(config["paths"]["training_root"])
    path = training_root / "freeze" / "provenance.json"
    payload = json.loads(path.read_text())
    if payload.get("schema") != "delimit3d_scannet40_agile3d_joint_provenance/v1":
        raise ValueError("training provenance schema mismatch")
    if payload.get("experiment_id") != EXPERIMENT:
        raise ValueError("training provenance experiment mismatch")
    if payload.get("arm", joint.DEFAULT_ARM) != joint.arm_name(config):
        raise ValueError("training provenance arm mismatch")
    return path, payload


def _load_training_manifest(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    training_root = resolve(config["paths"]["training_root"])
    path = training_root / "selection_manifest.json"
    manifest, expected = _load_canonical_json(path, "manifest_sha256")
    if manifest.get("schema") != joint.MANIFEST_SCHEMA:
        raise ValueError("training selection manifest schema mismatch")
    if manifest.get("experiment_id") != EXPERIMENT:
        raise ValueError("training selection manifest experiment mismatch")
    if manifest.get("arm", joint.DEFAULT_ARM) != joint.arm_name(config):
        raise ValueError("training selection manifest arm mismatch")
    if len(manifest.get("train_scenes", [])) != 1200:
        raise ValueError("training manifest does not cover 1,200 train scenes")
    if len(manifest.get("validation_scenes", [])) != 312:
        raise ValueError("training manifest does not cover 312 validation scenes")
    if len(manifest.get("panels", {}).get("MO", [])) != 312:
        raise ValueError("training manifest MO panel is not the official 312-scene panel")
    if len(manifest.get("panels", {}).get("SO", [])) != 10357:
        raise ValueError("training manifest SO panel is not the official 10,357-object panel")
    schedule = resolve(manifest["train_episodes_path"])
    if file_sha256(schedule) != manifest.get("train_episodes_sha256"):
        raise ValueError("training episode schedule changed")
    if expected != str(config["evaluation"]["training_manifest_sha256"]):
        raise ValueError("training manifest hash differs from evaluation declaration")
    return path, manifest


def verify_training_artifacts(
    config: Mapping[str, Any], *, require_checkpoint_payload: bool = False
) -> dict[str, Any]:
    provenance_path, provenance = _load_training_provenance(config)
    manifest_path, manifest = _load_training_manifest(config)
    training_root = resolve(config["paths"]["training_root"])
    arm = joint.arm_name(config)
    report_path = training_root / arm / "train_report.json"
    report = json.loads(report_path.read_text())
    if report.get("arm", joint.DEFAULT_ARM) != arm:
        raise ValueError("joint training report arm mismatch")
    decoder_init = training_root / "decoder_init.pt"
    if file_sha256(decoder_init) != str(config["parent_decoder_initialization_file_sha256"]):
        raise ValueError("training decoder initialization file hash mismatch")
    checkpoint = resolve(config["evaluation"]["checkpoint"])
    declared_checkpoint = str(config["evaluation"]["checkpoint_sha256"])
    if file_sha256(checkpoint) != declared_checkpoint:
        raise ValueError("joint evaluation checkpoint hash mismatch")
    if resolve(report["checkpoint"]) != checkpoint:
        raise ValueError("training report points to another final checkpoint")
    if report.get("checkpoint_sha256") != declared_checkpoint:
        raise ValueError("training report checkpoint hash mismatch")
    if int(report.get("updates", -1)) != 5000:
        raise ValueError("joint training report is not complete")
    if report.get("encoder_trainable") is not True or report.get("encoder_frozen") is not False:
        raise ValueError("joint training report does not prove an unfrozen encoder")
    if report.get("encoder_state_changed") is not True or report.get("encoder_parameters_changed") is not True:
        raise ValueError("joint training report does not prove encoder adaptation")
    if report.get("optimizer_param_group_names") != ["encoder", "decoder"]:
        raise ValueError("joint training optimizer groups are not encoder, decoder")
    if [float(value) for value in report.get("optimizer_param_group_lrs", [])] != [1.0e-4, 1.0e-4]:
        raise ValueError("joint training encoder/decoder learning rates are not equal")
    if report.get("episode_schedule_sha256") != manifest["train_episodes_sha256"]:
        raise ValueError("training report schedule hash mismatch")
    if manifest.get("decoder_initialization", {}).get("file_sha256") != file_sha256(decoder_init):
        raise ValueError("training manifest decoder initialization linkage mismatch")
    if report.get("provenance_sha256") != file_sha256(provenance_path):
        raise ValueError("training report provenance linkage mismatch")
    if provenance.get("parent_manifest_sha256") != config.get("parent_manifest_sha256"):
        raise ValueError("training parent manifest hash differs from evaluation config")
    if provenance.get("repo_commit") != manifest.get("repo_commit"):
        raise ValueError("training provenance and manifest commits differ")
    if provenance.get("encoder_checkpoint_sha256") != config["encoder"]["checkpoint_sha256"]:
        raise ValueError("training encoder initialization hash differs from evaluation config")
    if provenance.get("source_archive_sha256") != report.get("source_archive_sha256"):
        raise ValueError("training source archive linkage mismatch")
    dependency = provenance.get("litept_dependency", {})
    if file_sha256(resolve(dependency["archive"])) != dependency["archive_sha256"]:
        raise ValueError("training frozen LitePT dependency archive changed")
    if require_checkpoint_payload:
        payload = joint.torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != joint.CHECKPOINT_SCHEMA:
            raise ValueError("joint checkpoint schema mismatch")
        if payload.get("experiment_id") != EXPERIMENT or payload.get("arm") != arm:
            raise ValueError("joint checkpoint identity mismatch")
        if int(payload.get("update", -1)) != 5000:
            raise ValueError("joint checkpoint is not the final 5,000-update checkpoint")
        if payload.get("repo_commit") != report.get("repo_commit"):
            raise ValueError("joint checkpoint source commit mismatch")
        if payload.get("episode_schedule_sha256") != manifest["train_episodes_sha256"]:
            raise ValueError("joint checkpoint schedule hash mismatch")
        if payload.get("encoder_frozen") is not False or payload.get("encoder_trainable") is not True:
            raise ValueError("joint checkpoint is not an all-learnable checkpoint")
        if payload.get("optimizer_param_group_names") != ["encoder", "decoder"]:
            raise ValueError("joint checkpoint optimizer groups mismatch")
        if [float(value) for value in payload.get("optimizer_param_group_lrs", [])] != [1.0e-4, 1.0e-4]:
            raise ValueError("joint checkpoint learning rates are not equal")
        if payload.get("decoder_kwargs") != joint.decoder_kwargs(config):
            raise ValueError("joint checkpoint decoder architecture mismatch")
        if joint.state_sha256(payload["encoder_state_dict"]) != payload.get("encoder_state_sha256"):
            raise ValueError("joint checkpoint encoder tensor hash mismatch")
        if joint.state_sha256(payload["decoder_state_dict"]) != payload.get("decoder_tensor_sha256"):
            raise ValueError("joint checkpoint decoder tensor hash mismatch")
    return {
        "training_root": str(training_root),
        "provenance_path": str(provenance_path),
        "provenance_sha256": file_sha256(provenance_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "train_report_path": str(report_path),
        "train_report_sha256": file_sha256(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": declared_checkpoint,
        "repo_commit": str(report["repo_commit"]),
        "episode_schedule_sha256": str(manifest["train_episodes_sha256"]),
        "litept_dependency_archive_sha256": str(dependency["archive_sha256"]),
    }


def _copy_tree(source: Path, destination: Path, entries: Mapping[str, str]) -> None:
    source = source.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in entries.items():
        source_path = source / name
        if not source_path.exists() or file_sha256(source_path) != expected:
            raise ValueError(f"training frozen LitePT dependency drift: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != expected:
            raise ValueError(f"evaluation frozen LitePT dependency drift: {name}")
        if not target.exists():
            shutil.copy2(source_path, target)


def freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_config(config)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("commit the evaluation worktree before freezing")
    training = verify_training_artifacts(config)
    provenance_path, training_provenance = _load_training_provenance(config)
    training_root = resolve(config["paths"]["training_root"])
    root = root_of(config)
    freeze_root = root / "freeze"
    freeze_root.mkdir(parents=True, exist_ok=True)
    frozen_dependency = freeze_root / "litept_source"
    dependency = training_provenance["litept_dependency"]
    _copy_tree(
        resolve(dependency["frozen_root"]),
        frozen_dependency,
        dependency["source_entries"],
    )
    dependency_archive = freeze_root / "litept_source.tar"
    source_dependency_archive = resolve(dependency["archive"])
    if dependency_archive.exists() and file_sha256(dependency_archive) != dependency["archive_sha256"]:
        raise ValueError("evaluation LitePT dependency archive already differs")
    if not dependency_archive.exists():
        shutil.copy2(source_dependency_archive, dependency_archive)

    decoder_init_source = training_root / "decoder_init.pt"
    decoder_init = root / "decoder_init.pt"
    decoder_init_sha = file_sha256(decoder_init_source)
    if decoder_init.exists() and file_sha256(decoder_init) != decoder_init_sha:
        raise ValueError("evaluation decoder initialization already differs")
    if not decoder_init.exists():
        shutil.copy2(decoder_init_source, decoder_init)

    source_commit = git_commit()
    resolved = copy.deepcopy(dict(config))
    resolved["repo_commit"] = source_commit
    resolved["paths"] = dict(resolved["paths"])
    resolved["paths"]["litept_root"] = str(frozen_dependency)
    resolved_path = freeze_root / "resolved_config.yaml"
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    if resolved_path.exists() and resolved_path.read_text() != resolved_text:
        raise RuntimeError("frozen evaluation YAML differs; use a new artifact root")
    if not resolved_path.exists():
        resolved_path.write_text(resolved_text)

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
        "torch": "deferred-to-gpu-evaluation",
        "cuda_version": "deferred-to-gpu-evaluation",
    }
    dump(freeze_root / "environment_cpu.json", environment)
    provenance = {
        "schema": EVALUATION_SCHEMA,
        "experiment_id": EXPERIMENT,
        "arm": joint.arm_name(config),
        "repo_commit": source_commit,
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": file_sha256(resolved_path),
        "source_archive": str(source_archive),
        "source_archive_sha256": file_sha256(source_archive),
        "source_entries": source_entries,
        "decoder_initialization": {
            "path": str(decoder_init),
            "file_sha256": decoder_init_sha,
        },
        "environment": str(freeze_root / "environment_cpu.json"),
        "environment_sha256": file_sha256(freeze_root / "environment_cpu.json"),
        "training": training,
        "training_provenance_path": str(provenance_path),
        "training_provenance_sha256": file_sha256(provenance_path),
        "litept_dependency": {
            "source_root": str(resolve(dependency["frozen_root"])),
            "frozen_root": str(frozen_dependency),
            "source_entries": dict(dependency["source_entries"]),
            "archive": str(dependency_archive),
            "archive_sha256": file_sha256(dependency_archive),
        },
    }
    dump(freeze_root / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)
    return provenance


def verify_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    root = root_of(config)
    provenance_path = root / "freeze" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("evaluation provenance schema mismatch")
    if provenance.get("arm", joint.DEFAULT_ARM) != joint.arm_name(config):
        raise ValueError("evaluation provenance arm mismatch")
    if config.get("repo_commit") != provenance.get("repo_commit"):
        raise ValueError("evaluation config commit differs from frozen provenance")
    if file_sha256(resolve(provenance["resolved_config"])) != provenance["resolved_config_sha256"]:
        raise ValueError("evaluation resolved configuration changed")
    if file_sha256(resolve(provenance["source_archive"])) != provenance["source_archive_sha256"]:
        raise ValueError("evaluation source archive changed")
    if file_sha256(resolve(provenance["environment"])) != provenance["environment_sha256"]:
        raise ValueError("evaluation environment record changed")
    for name, expected in provenance["source_entries"].items():
        path = REPO_ROOT / name
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"executing evaluation source differs from frozen source: {name}")
    decoder_init = resolve(provenance["decoder_initialization"]["path"])
    if file_sha256(decoder_init) != provenance["decoder_initialization"]["file_sha256"]:
        raise ValueError("evaluation decoder initialization changed")
    dependency = provenance["litept_dependency"]
    if file_sha256(resolve(dependency["archive"])) != dependency["archive_sha256"]:
        raise ValueError("evaluation LitePT dependency archive changed")
    for name, expected in dependency["source_entries"].items():
        path = resolve(config["paths"]["litept_root"]) / name
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"evaluation LitePT dependency differs: {name}")
    frozen = yaml.safe_load(resolve(provenance["resolved_config"]).read_text())
    if json_sha256(config) != json_sha256(frozen):
        raise ValueError("runtime evaluation config differs from frozen resolved YAML")
    training = verify_training_artifacts(config)
    if training != provenance["training"]:
        raise ValueError("training provenance changed since evaluation freeze")
    if file_sha256(resolve(provenance["training_provenance_path"])) != provenance["training_provenance_sha256"]:
        raise ValueError("training provenance file changed since evaluation freeze")
    return provenance


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_freeze(config)
    training = verify_training_artifacts(config)
    _path, manifest = _load_training_manifest(config)
    root = root_of(config)
    payload = {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT,
        "arm": joint.arm_name(config),
        "repo_commit": str(config["repo_commit"]),
        "training": training,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "panels": {name: len(manifest["panels"][name]) for name in ("MO", "SO")},
        "protocol": dict(config["protocol"]),
        "decoder": dict(config["decoder"]),
    }
    payload["manifest_sha256"] = json_sha256(payload)
    path = root / "evaluation_manifest.json"
    if path.exists():
        old, _ = _load_canonical_json(path, "manifest_sha256")
        old["manifest_sha256"] = json_sha256({k: v for k, v in old.items() if k != "manifest_sha256"})
        if old != payload:
            raise RuntimeError("existing evaluation manifest differs; use a new evaluation root")
    else:
        dump(path, payload)
    dump(
        root / "prepare_report.json",
        {
            "schema": "delimit3d_scannet40_joint_evaluation_prepare/v1",
            "training_manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "MO": len(manifest["panels"]["MO"]),
            "SO": len(manifest["panels"]["SO"]),
            "evaluation_manifest_sha256": payload["manifest_sha256"],
        },
    )
    return payload


def load_evaluation_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    root = root_of(config)
    payload, _ = _load_canonical_json(root / "evaluation_manifest.json", "manifest_sha256")
    if payload.get("schema") != EVALUATION_MANIFEST_SCHEMA:
        raise ValueError("evaluation manifest schema mismatch")
    if payload.get("experiment_id") != EXPERIMENT:
        raise ValueError("evaluation manifest experiment mismatch")
    if payload.get("arm", joint.DEFAULT_ARM) != joint.arm_name(config):
        raise ValueError("evaluation manifest arm mismatch")
    _path, training_manifest = _load_training_manifest(config)
    if payload.get("training_manifest_sha256") != training_manifest["manifest_sha256"]:
        raise ValueError("evaluation manifest training selection drift")
    if payload.get("checkpoint_sha256") != config["evaluation"]["checkpoint_sha256"]:
        raise ValueError("evaluation manifest checkpoint drift")
    return payload


def _feature_cache_path(root: Path, scene: str) -> Path:
    return root / "feature_cache" / "joint" / f"{scene}.pt"


def _validate_feature_payload(
    payload: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    manifest_sha256: str,
    checkpoint_sha256: str,
) -> None:
    scene = str(record["scene"])
    required = {
        "schema": FEATURE_CACHE_SCHEMA,
        "experiment_id": EXPERIMENT,
        "scene": scene,
        "source_data_sha256": str(record["data_sha256"]),
        "selection_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{scene}: feature cache {key} mismatch")
    features = payload.get("features")
    xyz = payload.get("scene_xyz")
    inverse = payload.get("inverse_map")
    representatives = payload.get("representative_indices")
    raw_labels = payload.get("raw_labels")
    if not joint.torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != 72:
        raise ValueError(f"{scene}: feature cache token shape mismatch")
    if not joint.torch.is_tensor(xyz) or tuple(xyz.shape) != (features.shape[0], 3):
        raise ValueError(f"{scene}: feature cache coordinate shape mismatch")
    if not joint.torch.is_tensor(inverse) or tuple(inverse.shape) != (int(record["points"]),):
        raise ValueError(f"{scene}: feature cache inverse map shape mismatch")
    if not joint.torch.is_tensor(representatives) or tuple(representatives.shape) != (features.shape[0],):
        raise ValueError(f"{scene}: feature cache representative shape mismatch")
    if not isinstance(raw_labels, np.ndarray) or raw_labels.shape != (int(record["points"]),):
        raise ValueError(f"{scene}: feature cache raw label shape mismatch")
    if not bool(joint.torch.isfinite(features).all()) or not bool(joint.torch.isfinite(xyz).all()):
        raise ValueError(f"{scene}: feature cache contains non-finite tensors")
    if joint.array_sha256(representatives) != record.get("representative_indices_sha256"):
        raise ValueError(f"{scene}: representative geometry changed in feature cache")
    if int(features.shape[0]) != int(record.get("representative_token_count")):
        raise ValueError(f"{scene}: representative token count changed in feature cache")
    object_ids = [int(item["instance"]) for item in record["objects"]]
    indices = payload.get("object_token_indices")
    if not isinstance(indices, Mapping):
        raise ValueError(f"{scene}: feature cache missing object-token indices")
    for object_id in object_ids:
        values = indices.get(str(object_id))
        if not joint.torch.is_tensor(values) or values.numel() == 0:
            raise ValueError(f"{scene}: object {object_id} has no representative token")


class SceneFeatureCache:
    """Disk-backed dynamic encoder cache with a small GPU LRU."""

    def __init__(
        self,
        config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        encoder: Any,
        device: Any,
        checkpoint_sha256: str,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.encoder = encoder
        self.device = device
        self.checkpoint_sha256 = checkpoint_sha256
        self.root = root_of(config)
        self.manifest_sha256 = str(manifest["manifest_sha256"])
        self.commit = str(config["repo_commit"])
        self.capacity = int(config["optimizer"].get("cache_memory_scenes", 2))
        self.values: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.rows: dict[str, dict[str, Any]] = {}
        self.report_path = self.root / "feature_cache_manifest.json"
        if self.report_path.exists():
            report, _ = _load_canonical_json(self.report_path, "manifest_sha256")
            if report.get("schema") != FEATURE_CACHE_SCHEMA:
                raise ValueError("feature cache manifest schema mismatch")
            if report.get("selection_manifest_sha256") != self.manifest_sha256:
                raise ValueError("feature cache manifest selection drift")
            if report.get("checkpoint_sha256") != self.checkpoint_sha256:
                raise ValueError("feature cache manifest checkpoint drift")
            if report.get("repo_commit") != self.commit:
                raise ValueError("feature cache manifest commit drift")
            self.rows = {str(row["scene"]): dict(row) for row in report.get("rows", [])}
            for scene, row in self.rows.items():
                path = _feature_cache_path(self.root, scene)
                if not path.exists() or file_sha256(path) != row.get("file_sha256"):
                    raise ValueError(f"feature cache binary drift: {scene}")

    def path_for(self, scene: str) -> Path:
        return _feature_cache_path(self.root, str(scene))

    def _publish_row(self, scene: str, path: Path, payload: Mapping[str, Any]) -> None:
        row = {
            "scene": str(scene),
            "path": str(path),
            "file_sha256": file_sha256(path),
            "geometry_sha256": str(payload["geometry_sha256"]),
            "tokens": int(payload["features"].shape[0]),
            "points": int(payload["points"]),
        }
        previous = self.rows.get(str(scene))
        if previous is not None and previous != row:
            raise ValueError(f"feature cache row changed for {scene}")
        self.rows[str(scene)] = row
        report = {
            "schema": FEATURE_CACHE_SCHEMA,
            "experiment_id": EXPERIMENT,
            "repo_commit": self.commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "selection_manifest_sha256": self.manifest_sha256,
            "rows": [self.rows[key] for key in sorted(self.rows)],
            "complete": False,
        }
        report["manifest_sha256"] = json_sha256(report)
        dump(self.report_path, report)

    def _encode(self, record: Mapping[str, Any]) -> dict[str, Any]:
        scene = str(record["scene"])
        points, colors, normals, objects, masks = joint.protocol.load_scene(record, resolve)
        shifted, shift = joint._center_shift_numpy(points)
        input_features = joint.build_input_features(
            points,
            colors,
            use_colors=True,
            use_normals=True,
            normals=joint._normalise_normals(normals),
        )
        coord = joint.torch.from_numpy(shifted).to(self.device, dtype=joint.torch.float32)
        feat = joint.torch.from_numpy(input_features).to(self.device, dtype=joint.torch.float32)
        with joint.torch.inference_mode():
            output = self.encoder(coord, feat)
        tokens = output.scene_tokens.detach().float().cpu().contiguous()
        xyz = output.scene_xyz.detach().float().cpu().contiguous()
        inverse = output.inverse_map.detach().long().cpu().contiguous()
        representatives = output.representative_indices.detach().long().cpu().contiguous()
        del output, coord, feat
        if tokens.ndim != 2 or tokens.shape[1] != 72:
            raise RuntimeError(f"{scene}: unexpected dynamic dec0 token shape {tuple(tokens.shape)}")
        if tuple(inverse.shape) != (len(points),):
            raise RuntimeError(f"{scene}: dynamic inverse map shape mismatch")
        if tuple(representatives.shape) != (tokens.shape[0],):
            raise RuntimeError(f"{scene}: dynamic representative shape mismatch")
        object_ids = [int(item["instance"]) for item in objects]
        token_targets = joint.build_token_targets(representatives.numpy(), masks, object_ids)
        if any(len(values) == 0 for values in token_targets.values()):
            raise RuntimeError(f"{scene}: an official object has no dynamic representative token")
        raw_labels = np.zeros(len(points), dtype=np.int64)
        for object_id in object_ids:
            raw_labels[masks[object_id]] = object_id
        geometry_hash = json_sha256(
            {
                "scene": scene,
                "shift": joint.array_sha256(shifted),
                "scene_xyz": joint.array_sha256(xyz),
                "inverse_map": joint.array_sha256(inverse),
                "representative_indices": joint.array_sha256(representatives),
            }
        )
        normal_hash = str(record["source_hashes"][record["normal_path"]])
        payload = {
            "schema": FEATURE_CACHE_SCHEMA,
            "experiment_id": EXPERIMENT,
            "repo_commit": self.commit,
            "scene": scene,
            "kind": str(record["kind"]),
            "source_data_path": str(record["data_path"]),
            "source_data_sha256": str(record["data_sha256"]),
            "source_normal_path": str(record["normal_path"]),
            "source_normal_sha256": normal_hash,
            "points": int(len(points)),
            "features": tokens,
            "scene_xyz": xyz,
            "inverse_map": inverse,
            "representative_indices": representatives,
            "shift": joint.torch.from_numpy(shift),
            "object_token_indices": {
                str(object_id): joint.torch.from_numpy(values).long()
                for object_id, values in token_targets.items()
            },
            "raw_labels": raw_labels,
            "objects": [dict(item) for item in objects],
            "geometry_sha256": geometry_hash,
            "selection_manifest_sha256": self.manifest_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        return payload

    def _validate(self, payload: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        _validate_feature_payload(
            payload,
            record=record,
            manifest_sha256=self.manifest_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
        )
        if payload.get("repo_commit") != self.commit:
            raise ValueError(f"{record['scene']}: feature cache commit mismatch")
        if payload.get("source_normal_sha256") != str(
            record["source_hashes"][record["normal_path"]]
        ):
            raise ValueError(f"{record['scene']}: feature cache normal source mismatch")

    def get(self, record: Mapping[str, Any]) -> dict[str, Any]:
        scene = str(record["scene"])
        if scene in self.values:
            self.values.move_to_end(scene)
            return self.values[scene]
        path = self.path_for(scene)
        if path.exists():
            payload = joint.torch.load(path, map_location="cpu", weights_only=False)
            self._validate(payload, record)
            row = self.rows.get(scene)
            if row is not None and file_sha256(path) != row.get("file_sha256"):
                raise ValueError(f"{scene}: feature cache file hash changed")
            if row is None:
                self._publish_row(scene, path, payload)
            reused = True
        else:
            payload = self._encode(record)
            self._validate(payload, record)
            save_torch(path, payload)
            self._publish_row(scene, path, payload)
            reused = False
            print(
                json.dumps(
                    {
                        "mode": "joint_eval_feature_cache",
                        "scene": scene,
                        "completed": len(self.rows),
                        "total": 312,
                        "tokens": int(payload["features"].shape[0]),
                        "reused": reused,
                    }
                ),
                flush=True,
            )
        features = payload["features"].to(self.device, non_blocking=True)
        scene_xyz = payload["scene_xyz"].to(self.device, non_blocking=True)
        value = {
            "scene": scene,
            "features": features,
            "scene_xyz": scene_xyz,
            "scene_xyz_cpu": payload["scene_xyz"],
            "inverse_map": payload["inverse_map"].numpy(),
            "representative_indices": payload["representative_indices"],
            "object_token_indices": payload["object_token_indices"],
            "raw_labels": payload["raw_labels"],
            "objects": [dict(item) for item in payload["objects"]],
            "feature_cache_path": str(path),
            "feature_cache_sha256": file_sha256(path),
        }
        self.values[scene] = value
        self.values.move_to_end(scene)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value

    def finalize(self, *, require_complete: bool = True) -> dict[str, Any]:
        report = {
            "schema": FEATURE_CACHE_SCHEMA,
            "experiment_id": EXPERIMENT,
            "repo_commit": self.commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "selection_manifest_sha256": self.manifest_sha256,
            "rows": [self.rows[key] for key in sorted(self.rows)],
            "complete": len(self.rows) == 312,
        }
        report["manifest_sha256"] = json_sha256(report)
        dump(self.report_path, report)
        if require_complete and len(self.rows) != 312:
            raise RuntimeError(f"dynamic validation feature cache is incomplete: {len(self.rows)}/312")
        return report


def _target_from_episode(cache: Mapping[str, Any], object_ids: Sequence[int], seed: int, scene: str):
    target = joint.torch.zeros(cache["features"].shape[0], dtype=joint.torch.long)
    for local_id, object_id in enumerate(object_ids, start=1):
        values = cache["object_token_indices"].get(str(int(object_id)))
        if values is None or values.numel() == 0:
            raise ValueError(f"{scene}: selected object {object_id} has no representative token")
        if bool(joint.torch.any(target[values] != 0)):
            raise ValueError(f"{scene}: selected objects overlap in token space")
        target[values] = int(local_id)
    click_seed = joint.scene_seed(
        int(seed), scene, "initial:" + ",".join(str(value) for value in object_ids)
    )
    clicks, times = joint.protocol.initial_clicks(
        target.numpy(), cache["scene_xyz_cpu"].numpy(), click_seed
    )
    return target, clicks, times


def evaluate_episode(
    decoder: Any,
    cache: Mapping[str, Any],
    record: Mapping[str, Any],
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
    mode: str,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    from delimit3d.evaluation.agile3d_protocol import raw_object_ious

    object_ids = [int(value) for value in episode["objects"]]
    count = len(object_ids)
    target, clicks, times = _target_from_episode(
        cache, object_ids, int(config["seed"]), str(record["scene"])
    )
    raw_labels = cache["raw_labels"]
    raw_target = np.zeros(len(raw_labels), dtype=np.int64)
    for local_id, object_id in enumerate(object_ids, start=1):
        raw_target[raw_labels == object_id] = local_id
    states: list[dict[str, Any]] = [
        {
            "total_clicks": 0,
            "mean_iou": 0.0,
            "object_ious": {str(index): 0.0 for index in range(1, count + 1)},
            "clicks": {},
            "click_times": {},
        }
    ]
    saved: dict[str, np.ndarray] = {}
    decoder.eval()
    perfect = False
    with joint.torch.inference_mode():
        for total in range(count, int(config["protocol"]["click_budget"]) * count + 1):
            if not perfect:
                output = decoder(
                    cache["features"],
                    cache["scene_xyz"],
                    clicks=clicks,
                    click_times=times,
                )
                prediction = output["pred_masks"].argmax(1).cpu().numpy()
                prediction = joint.enforce_click_labels(prediction, clicks)
            ious = raw_object_ious(
                prediction,
                raw_target,
                cache["inverse_map"],
                count,
            )
            states.append(
                {
                    "total_clicks": total,
                    "mean_iou": float(np.mean(list(ious.values()))),
                    "object_ious": ious,
                    "clicks": {key: list(value) for key, value in clicks.items()},
                    "click_times": {key: list(value) for key, value in times.items()},
                    "perfect_prediction_padded": perfect,
                }
            )
            if prediction_path and total in {
                count,
                3 * count,
                5 * count,
                10 * count,
                15 * count,
                20 * count,
            }:
                saved[f"click_{total // count}"] = prediction.astype(np.int16)
            if total == int(config["protocol"]["click_budget"]) * count:
                break
            new_clicks, new_times, _events = joint.simulated_corrections(
                prediction,
                target.numpy(),
                cache["scene_xyz_cpu"].numpy(),
                clicks,
                times,
                training=False,
                click_center_method=str(config["protocol"]["click_center_method"]),
            )
            if not new_clicks:
                perfect = True
            else:
                clicks, times = joint.append_clicks(clicks, times, new_clicks, new_times)
    trace = {
        "schema": "delimit3d_scannet40_joint_trace/v1",
        "mode": str(mode),
        "scene": str(record["scene"]),
        "object_ids": object_ids,
        "object_count": count,
        "states": states,
        "objects": [
            next(item for item in cache["objects"] if int(item["instance"]) == object_id)
            for object_id in object_ids
        ],
    }
    if prediction_path:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, **saved)
        trace["prediction_path"] = str(prediction_path)
    trace["metrics"] = joint.protocol.metrics_from_trace(trace)
    return trace


def _trace_identity(mode: str, episode: Mapping[str, Any]) -> str:
    scene = str(episode["scene"])
    return scene if mode == "MO" else f"{scene}_obj_{int(episode['objects'][0])}"


def evaluate_panel(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    decoder: Any,
    feature_cache: SceneFeatureCache,
    mode: str,
    *,
    max_units: int | None = None,
) -> dict[str, Any]:
    root = root_of(config)
    output = root / "delimit3d" / "evaluation" / mode
    output.mkdir(parents=True, exist_ok=True)
    records = {
        str(record["scene"]): record
        for record in list(
            _load_training_manifest(config)[1]["validation_scenes"]
        )
    }
    episodes = list(_load_training_manifest(config)[1]["panels"][mode])
    if max_units is not None:
        episodes = episodes[: int(max_units)]
    checkpoint_sha256 = str(config["evaluation"]["checkpoint_sha256"])
    completed: list[dict[str, Any]] = []
    temporary_csv = output / "official_results.csv.tmp"
    with temporary_csv.open("w") as results:
        for index, episode in enumerate(episodes):
            identity = _trace_identity(mode, episode)
            trace_path = output / "traces" / f"{identity}.json"
            record = records[str(episode["scene"])]
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                if (
                    trace.get("mode") != mode
                    or trace.get("checkpoint_sha256") != checkpoint_sha256
                    or trace.get("training_manifest_sha256")
                    != manifest["training_manifest_sha256"]
                ):
                    raise ValueError(f"{identity}: existing trace belongs to another evaluation")
                cache_path = feature_cache.path_for(str(episode["scene"]))
                expected_cache_sha = trace.get("feature_cache_sha256")
                if not cache_path.exists() or file_sha256(cache_path) != expected_cache_sha:
                    raise ValueError(f"{identity}: feature cache changed after trace publication")
            else:
                cache = feature_cache.get(record)
                prediction_path = None
                if index < int(config["evaluation"].get("visual_scene_count", 4)):
                    prediction_path = output / "predictions" / f"{identity}.npz"
                trace = evaluate_episode(
                    decoder,
                    cache,
                    record,
                    episode,
                    config,
                    mode,
                    prediction_path,
                )
                trace["checkpoint_sha256"] = checkpoint_sha256
                trace["training_manifest_sha256"] = str(manifest["training_manifest_sha256"])
                trace["feature_cache_sha256"] = str(cache["feature_cache_sha256"])
                trace["feature_cache_path"] = str(cache["feature_cache_path"])
                dump(trace_path, trace)
            results.writelines(joint.protocol.official_lines(index, trace, mode))
            completed.append(
                {
                    "identity": identity,
                    "scene": str(episode["scene"]),
                    "object_count": int(trace["object_count"]),
                    "objects": trace["objects"],
                    "metrics": trace["metrics"],
                    "trace": str(trace_path),
                }
            )
            dump(
                root / "workers" / "status.json",
                {
                    "phase": "evaluate",
                    "panel": mode,
                    "completed": index + 1,
                    "total": len(episodes),
                    "scene": str(episode["scene"]),
                    "time": time.time(),
                },
            )
            if index % 10 == 0 or index + 1 == len(episodes):
                print(
                    json.dumps(
                        {
                            "mode": "joint_eval",
                            "panel": mode,
                            "completed": index + 1,
                            "total": len(episodes),
                        }
                    ),
                    flush=True,
                )
    os.replace(temporary_csv, output / "official_results.csv")
    if not completed:
        raise RuntimeError(f"{mode}: no evaluation units completed")
    report = {
        "schema": "delimit3d_scannet40_joint_panel_report/v1",
        "experiment_id": EXPERIMENT,
        "arm": f"{joint.arm_name(config)}_joint",
        "mode": mode,
        "units": completed,
        "count": len(completed),
        "expected_count": int(manifest["panels"][mode]),
        "complete": len(completed) == int(manifest["panels"][mode]),
        "metrics": {
            metric: float(np.mean([row["metrics"][metric] for row in completed]))
            for metric in completed[0]["metrics"]
        },
        "official_csv": str(output / "official_results.csv"),
        "official_csv_sha256": file_sha256(output / "official_results.csv"),
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_sha256": str(manifest["training_manifest_sha256"]),
        "metric_scope": "joint trained LitePT encoder plus AGILE3D decoder under official ScanNet40 protocol",
    }
    dump(output / "evaluation_report.json", report)
    return report


def write_status(root: Path, **values: Any) -> None:
    payload = {"time": time.time(), **values}
    dump(root / "workers" / "status.json", payload)


@contextlib.contextmanager
def heartbeat(config: Mapping[str, Any]):
    root = root_of(config)
    directory = root / "workers"
    directory.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    dump(
        directory / "joint_eval.pid.json",
        {
            "pid": os.getpid(),
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": joint.torch.cuda.get_device_name(0) if joint.torch.cuda.is_available() else None,
        },
    )

    def beat() -> None:
        interval = int(config.get("runtime", {}).get("heartbeat_seconds", 30))
        path = directory / "joint_eval.heartbeat.jsonl"
        while not stop.is_set():
            status = {}
            status_path = directory / "status.json"
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text())
                except json.JSONDecodeError:
                    status = {}
            with path.open("a") as handle:
                handle.write(json.dumps({"time": time.time(), **status}, sort_keys=True) + "\n")
            stop.wait(interval)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def load_models(config: Mapping[str, Any], checkpoint_path: Path, device: Any):
    payload = joint.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = joint.load_backbone(
        resolve(config["encoder"]["checkpoint"]),
        litept_root=resolve(config["paths"]["litept_root"]),
        in_channels=6,
        device=device,
    )
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    decoder, _init_report = joint.load_initialization(
        root_of(config) / "decoder_init.pt",
        expected_kwargs=joint.decoder_kwargs(config),
    )
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    decoder.to(device)
    decoder.eval()
    if joint.state_sha256(encoder.state_dict()) != payload.get("encoder_state_sha256"):
        raise ValueError("loaded encoder state does not match checkpoint tensor hash")
    if joint.state_sha256(decoder.state_dict()) != payload.get("decoder_tensor_sha256"):
        raise ValueError("loaded decoder state does not match checkpoint tensor hash")
    return encoder, decoder, payload


def evaluate(
    config: Mapping[str, Any],
    *,
    panel: str | None = None,
    checkpoint: Path | None = None,
    max_units: int | None = None,
) -> dict[str, Any]:
    verify_freeze(config)
    eval_manifest = load_evaluation_manifest(config)
    joint._load_train_runtime()
    verify_training_artifacts(config, require_checkpoint_payload=True)
    gpu = joint.gpu_preflight()
    root = root_of(config)
    dump(root / "workers" / "environment_joint_eval.json", gpu)
    device = joint.torch.device("cuda")
    joint.torch.cuda.set_device(0)
    checkpoint_path = resolve(checkpoint or config["evaluation"]["checkpoint"])
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != str(config["evaluation"]["checkpoint_sha256"]):
        raise ValueError("requested evaluation checkpoint is not the declared final joint checkpoint")
    encoder, decoder, _payload = load_models(config, checkpoint_path, device)
    _records_path, training_manifest = _load_training_manifest(config)
    cache = SceneFeatureCache(config, training_manifest, encoder, device, checkpoint_sha256)
    modes = [panel] if panel else [str(value) for value in config["evaluation"]["panels"]]
    reports: dict[str, Any] = {}
    with heartbeat(config):
        write_status(root, phase="starting", panels=modes, checkpoint_sha256=checkpoint_sha256)
        for mode in modes:
            reports[mode] = evaluate_panel(
                config,
                eval_manifest,
                decoder,
                cache,
                mode,
                max_units=max_units,
            )
            joint.torch.cuda.empty_cache()
        cache_report = cache.finalize(require_complete=max_units is None)
        complete = all(bool(report["complete"]) for report in reports.values())
        result = {
            "schema": EVALUATION_SCHEMA,
            "experiment_id": EXPERIMENT,
            "arm": f"{joint.arm_name(config)}_joint",
            "repo_commit": str(config["repo_commit"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "training_manifest_sha256": str(eval_manifest["training_manifest_sha256"]),
            "feature_cache_manifest": str(cache.report_path),
            "feature_cache_manifest_sha256": file_sha256(cache.report_path),
            "panels": reports,
            "complete": complete,
            "time": time.time(),
        }
        dump(root / "evaluation_report.json", result)
        write_status(root, phase="complete" if complete else "partial", panels=modes)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("freeze", "prepare", "preflight", "evaluate"), required=True
    )
    parser.add_argument("--panel", choices=("MO", "SO"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-units", type=int)
    return parser.parse_args()


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
        print(json.dumps(verify_training_artifacts(config), indent=2), flush=True)
        return
    if args.mode == "evaluate":
        evaluate(
            config,
            panel=args.panel,
            checkpoint=args.checkpoint,
            max_units=args.max_units,
        )
        return
    raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
