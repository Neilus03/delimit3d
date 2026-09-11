#!/usr/bin/env python3
"""Run the frozen-backbone Delimit3D AGILE3D-compatible multi-object study.

The command is intentionally explicit:

  prepare        freeze source scenes, object groups, episodes, and decoder init
  cache-features extract immutable LitePT dec0 features for one arm
  smoke          run a real K=3 CUDA forward/backward/click smoke
  train          train only the fresh decoder for 5,000 updates
  evaluate      evaluate MO-1, MO-5, and MO-10 interactive panels
  aggregate      compute paired scene bootstrap metrics and the Stage-A gate

All generated data stay in the external artifact root from the YAML file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
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
from delimit3d.evaluation.agile3d_decoder import (
    Agile3DClickDecoder,
    compute_agile3d_losses,
    load_initialization,
    save_initialization,
    state_sha256,
)
from delimit3d.evaluation.agile3d_protocol import (
    aggregate_panel_rows,
    append_clicks,
    array_sha256,
    build_token_targets,
    center_click,
    choose_non_overlapping_group,
    enforce_click_labels,
    file_sha256,
    json_sha256,
    load_bundle_scene,
    load_class_mapping,
    load_training_scene,
    metric_at_threshold,
    paired_scene_bootstrap,
    panel_click_thresholds,
    raw_object_ious,
    read_scene_ids,
    scene_seed,
    simulated_corrections,
)


SCHEMA = "delimit3d_scannetpp_agile3d_multio/v1"
CACHE_SCHEMA = "delimit3d_agile3d_feature_cache/v1"
TRAIN_SCHEMA = "delimit3d_agile3d_decoder_training/v1"
CHECKPOINT_SCHEMA = "delimit3d_agile3d_decoder_checkpoint/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("prepare", "cache-features", "smoke", "train", "evaluate", "aggregate"),
        required=True,
    )
    parser.add_argument("--arm", choices=("public", "delimit3d"))
    parser.add_argument(
        "--panel",
        choices=("MO-1", "MO-5", "MO-10", "all"),
        default="all",
        help="validation panel for evaluate; default runs all panels",
    )
    parser.add_argument("--checkpoint", type=Path, help="optional trained decoder checkpoint")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve(strict=True).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: YAML root must be a mapping")
    return dict(value)


def resolve_external(value: str | Path, *, strict: bool = True) -> Path:
    """Resolve Euler paths directly or through pf-pc69's read-only sshfs mount."""
    path = Path(str(value)).expanduser()
    if path.exists():
        return path.resolve(strict=strict)
    if str(path).startswith("/cluster/"):
        mounted = Path("/tmp/euler_cluster_nedela") / str(path).removeprefix("/cluster/")
        return mounted.resolve(strict=strict)
    return path.resolve(strict=strict)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def effective_commit(config: Mapping[str, Any]) -> str:
    """Use the frozen commit recorded in the resolved config when available."""
    configured = str(config.get("repo_commit", "auto"))
    if configured and configured not in {"auto", "unknown"}:
        return configured
    return git_commit()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    kwargs = {key: decoder[key] for key in allowed if key in decoder}
    kwargs.setdefault("feature_dim", 72)
    kwargs.setdefault("hidden_dim", 128)
    kwargs.setdefault("num_heads", 8)
    kwargs.setdefault("dim_feedforward", 1024)
    kwargs.setdefault("num_decoders", 3)
    kwargs.setdefault("num_bg_queries", 10)
    kwargs.setdefault("dropout", 0.0)
    kwargs.setdefault("pre_norm", False)
    kwargs.setdefault("max_click_events", 200)
    kwargs.setdefault("normalize_pos_enc", True)
    kwargs.setdefault("gauss_scale", 1.0)
    kwargs.setdefault("aux", True)
    return kwargs


def verify_config(config: Mapping[str, Any]) -> None:
    required = ("experiment_id", "seed", "paths", "protocol", "decoder", "arms")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"configuration is missing {missing}")
    if str(config["experiment_id"]) != "delimit3d_scannetpp_agile3d_mo_v1":
        raise ValueError("unexpected experiment_id")
    protocol = config["protocol"]
    required_protocol = {
        "train_scene_count",
        "minimum_instance_points",
        "maximum_objects_per_scene",
        "train_updates",
        "click_budget",
        "panels",
    }
    missing = sorted(required_protocol - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing {missing}")
    if int(protocol["train_updates"]) != 5000:
        raise ValueError("this runner is locked to 5,000 optimizer updates")
    if int(protocol["click_budget"]) != 20:
        raise ValueError("this runner is locked to 20 clicks per object")
    kwargs = decoder_kwargs(config)
    expected = {
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
    for key, value in expected.items():
        if kwargs[key] != value:
            raise ValueError(f"decoder.{key}={kwargs[key]!r}, expected {value!r}")
    for arm in ("public", "delimit3d"):
        if arm not in config["arms"]:
            raise ValueError(f"missing encoder arm {arm}")


def output_root(config: Mapping[str, Any]) -> Path:
    override = os.environ.get("DELIMIT3D_OUTPUT_ROOT")
    root = Path(override).expanduser() if override else Path(config["paths"]["output_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _record_source_payload(
    scene: str,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    objects: Sequence[Mapping[str, Any]],
    masks: Mapping[int, np.ndarray],
    data_path: Path,
    source_hashes: Mapping[str, str],
    kind: str,
) -> dict[str, Any]:
    payload = {
        "scene": str(scene),
        "kind": str(kind),
        "points": int(len(points)),
        "objects": [dict(item) for item in objects],
        "source_hashes": dict(source_hashes),
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "object_mask_sha256": {
            str(int(object_id)): array_sha256(mask)
            for object_id, mask in sorted(masks.items())
        },
    }
    return payload


def _save_scene_npz(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    masks: Mapping[int, np.ndarray],
) -> None:
    if path.exists():
        with np.load(path) as old:
            old_keys = sorted(old.files)
            new_keys = ["points", "colors", "normals"] + [
                f"obj_{int(object_id)}" for object_id in sorted(masks)
            ]
            if old_keys != sorted(new_keys):
                raise RuntimeError(f"refusing to overwrite changed scene data: {path}")
        return
    payload: dict[str, Any] = {
        "points": np.asarray(points, dtype=np.float32),
        "colors": np.asarray(colors, dtype=np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
    }
    payload.update(
        {
            f"obj_{int(object_id)}": np.asarray(mask, dtype=np.int64)
            for object_id, mask in sorted(masks.items())
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _training_record_arrays(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[int, np.ndarray]]:
    path = resolve_external(record["data_path"])
    if file_sha256(path) != record["data_sha256"]:
        raise ValueError(f"{record['scene']}: scene data hash mismatch")
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        objects = [dict(item) for item in record["objects"]]
        masks = {
            int(item["instance"]): np.asarray(
                data[f"obj_{int(item['instance'])}"], dtype=np.int64
            )
            for item in objects
        }
    return points, colors, normals, objects, masks


def _load_record_arrays(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[int, np.ndarray]]:
    if str(record.get("kind")) == "validation":
        return load_bundle_scene(record)
    return _training_record_arrays(record)


def _build_panels(
    validation_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    panels: dict[str, list[dict[str, Any]]] = {"MO-1": [], "MO-5": [], "MO-10": []}
    skipped: list[dict[str, Any]] = []
    for record in validation_records:
        scene = str(record["scene"])
        object_ids = [int(item["instance"]) for item in record["objects"]]
        _points, _colors, _normals, _objects, masks = _load_record_arrays(record)
        for object_id in object_ids:
            panels["MO-1"].append(
                {"scene": scene, "objects": [object_id], "rejected_draws": 0}
            )
        rng = np.random.default_rng(scene_seed(seed, scene, "validation-panels"))
        if len(object_ids) >= 5:
            group, rejected = choose_non_overlapping_group(
                object_ids, masks, 5, rng=rng
            )
            panels["MO-5"].append(
                {"scene": scene, "objects": group, "rejected_draws": int(rejected)}
            )
        else:
            skipped.append({"panel": "MO-5", "scene": scene, "reason": "fewer_than_five_objects"})
        if len(object_ids) >= 10:
            group, rejected = choose_non_overlapping_group(
                object_ids, masks, 10, rng=rng
            )
            panels["MO-10"].append(
                {"scene": scene, "objects": group, "rejected_draws": int(rejected)}
            )
        else:
            skipped.append({"panel": "MO-10", "scene": scene, "reason": "fewer_than_ten_objects"})
    return panels, skipped


def _build_training_schedule(
    train_records: Sequence[Mapping[str, Any]],
    *,
    updates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    schedule: list[dict[str, Any]] = []
    # Load each compressed scene once. Reopening a 20--80 MB NPZ for every
    # update would dominate preparation and make the schedule non-reproducible
    # if the external filesystem changes while it is being generated.
    masks_by_scene = {
        str(record["scene"]): _load_record_arrays(record)[4]
        for record in train_records
    }
    for update in range(1, int(updates) + 1):
        scene_index = int(rng.integers(0, len(train_records)))
        record = train_records[scene_index]
        object_ids = [int(item["instance"]) for item in record["objects"]]
        masks = masks_by_scene[str(record["scene"])]
        max_objects = min(10, len(object_ids))
        if max_objects < 1:
            raise RuntimeError(f"{record['scene']}: no eligible objects")
        count = int(rng.integers(1, max_objects + 1))
        group, rejected = choose_non_overlapping_group(
            object_ids,
            masks,
            count,
            rng=rng,
        )
        rejected_total += int(rejected)
        schedule.append(
            {
                "update": int(update),
                "scene": str(record["scene"]),
                "objects": group,
                "click_prefix": int(rng.integers(0, 20)),
                "rejected_draws": int(rejected),
            }
        )
    return schedule


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_config(config)
    root = output_root(config)
    paths = config["paths"]
    protocol = config["protocol"]
    scannetpp_root = resolve_external(paths["scannetpp_root"])
    pack_root = resolve_external(paths["train_pack_root"])
    train_split = resolve_external(paths["train_split"])
    val_bundle = resolve_external(paths["val_bundle"])
    classes, mapping = load_class_mapping(scannetpp_root)
    all_train = read_scene_ids(train_split)
    val_manifest_path = val_bundle / "scene_manifest.json"
    val_manifest = json.loads(val_manifest_path.read_text())
    val_scenes = [str(record["scene"]) for record in val_manifest["scenes"]]
    train_scenes = all_train[: int(protocol["train_scene_count"])]
    overlap = sorted(set(train_scenes).intersection(val_scenes))
    if overlap:
        raise ValueError(f"train/validation leakage: {overlap}")

    train_records: list[dict[str, Any]] = []
    train_data_root = root / "scene_data" / "train"
    for index, scene in enumerate(train_scenes):
        points, colors, normals, objects, masks, hashes = load_training_scene(
            scene,
            scannetpp_root=scannetpp_root,
            pack_root=pack_root,
            classes=classes,
            class_mapping=mapping,
            minimum_instance_points=int(protocol["minimum_instance_points"]),
            maximum_objects_per_scene=int(protocol["maximum_objects_per_scene"]),
        )
        data_path = train_data_root / f"{scene}.npz"
        _save_scene_npz(
            data_path,
            points=points,
            colors=colors,
            normals=normals,
            masks=masks,
        )
        train_records.append(
            _record_source_payload(
                scene,
                points=points,
                colors=colors,
                normals=normals,
                objects=objects,
                masks=masks,
                data_path=data_path,
                source_hashes=hashes,
                kind="training",
            )
        )
        print(
            json.dumps(
                {
                    "mode": "prepare",
                    "scene": scene,
                    "completed": index + 1,
                    "total": len(train_scenes),
                    "objects": len(objects),
                }
            ),
            flush=True,
        )

    validation_records: list[dict[str, Any]] = []
    for record in val_manifest["scenes"]:
        path = resolve_external(record["data_path"])
        if file_sha256(path) != record["data_sha256"]:
            raise ValueError(f"{record['scene']}: frozen validation hash mismatch")
        validation_records.append(
            {
                "scene": str(record["scene"]),
                "kind": "validation",
                "points": int(record["points"]),
                "objects": [dict(item) for item in record["objects"]],
                "source_hashes": dict(record.get("source_files", {})),
                "data_path": str(path),
                "data_sha256": str(record["data_sha256"]),
            }
        )

    panels, panel_skips = _build_panels(validation_records, seed=int(config["seed"]))
    schedule = _build_training_schedule(
        train_records,
        updates=int(protocol["train_updates"]),
        seed=int(config["seed"]),
    )

    init_path = root / "decoder_init.pt"
    if init_path.exists():
        _decoder, init_report = load_initialization(
            init_path,
            expected_kwargs=decoder_kwargs(config),
        )
    else:
        init_report = save_initialization(
            init_path,
            seed=int(config["decoder"]["init_seed"]),
            decoder_kwargs=decoder_kwargs(config),
            metadata={
                "experiment_id": str(config["experiment_id"]),
                "purpose": "shared fresh decoder for public and Delimit3D arms",
            },
        )

    episodes_path = root / "train_episodes.jsonl"
    if episodes_path.exists():
        old_hash = file_sha256(episodes_path)
        new_lines = "\n".join(json.dumps(row, sort_keys=True) for row in schedule) + "\n"
        if hashlib.sha256(new_lines.encode()).hexdigest() != old_hash:
            raise RuntimeError(f"refusing to overwrite changed episode schedule: {episodes_path}")
    else:
        write_jsonl(episodes_path, schedule)

    manifest_without_hash = {
        "schema": SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "repo_commit": effective_commit(config),
        "seed": int(config["seed"]),
        "protocol": dict(protocol),
        "decoder": decoder_kwargs(config),
        "paths": {
            "scannetpp_root": str(scannetpp_root),
            "train_pack_root": str(pack_root),
            "train_split": str(train_split),
            "validation_bundle": str(val_bundle),
        },
        "train_split_sha256": file_sha256(train_split),
        "validation_scene_manifest_sha256": file_sha256(val_manifest_path),
        "train_scenes": train_records,
        "validation_scenes": validation_records,
        "panels": panels,
        "panel_skips": panel_skips,
        "train_episodes_path": str(episodes_path),
        "train_episodes_sha256": file_sha256(episodes_path),
        "decoder_initialization": init_report,
        "decoder_initialization_path": str(init_path),
        "classes_sha256": json_sha256(classes),
        "class_mapping_sha256": json_sha256(mapping),
        "source_code": {
            "repo_root": str(REPO_ROOT),
            "git_commit": effective_commit(config),
            "runner": str(Path(__file__).resolve()),
            "decoder": str(REPO_ROOT / "src/delimit3d/evaluation/agile3d_decoder.py"),
            "protocol": str(REPO_ROOT / "src/delimit3d/evaluation/agile3d_protocol.py"),
        },
    }
    manifest = dict(manifest_without_hash)
    manifest["manifest_sha256"] = json_sha256(manifest_without_hash)
    manifest_path = root / "selection_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(f"refusing to overwrite changed manifest: {manifest_path}")
    else:
        json_dump(manifest_path, manifest)
    report = {
        "mode": "prepare_complete",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "train_scenes": len(train_records),
        "validation_scenes": len(validation_records),
        "panel_counts": {key: len(value) for key, value in panels.items()},
        "panel_skips": panel_skips,
        "train_updates": len(schedule),
        "decoder_initialization": init_report,
    }
    json_dump(root / "prepare_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return manifest


def load_prepared(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    root = output_root(config)
    path = root / "selection_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run prepare first: {path}")
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    without = dict(manifest)
    without.pop("manifest_sha256", None)
    if expected != json_sha256(without):
        raise ValueError("selection manifest hash mismatch")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("selection manifest schema mismatch")
    init_path = resolve_external(manifest["decoder_initialization_path"])
    _decoder, report = load_initialization(
        init_path,
        expected_kwargs=decoder_kwargs(config),
    )
    if report["tensor_state_sha256"] != manifest["decoder_initialization"]["tensor_state_sha256"]:
        raise ValueError("decoder initialization hash drift")
    episodes = resolve_external(manifest["train_episodes_path"])
    if file_sha256(episodes) != manifest["train_episodes_sha256"]:
        raise ValueError("training episode schedule hash drift")
    return manifest, init_path


def _scene_record_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in list(manifest["train_scenes"]) + list(manifest["validation_scenes"]):
        records[str(record["scene"])] = dict(record)
    return records


def _normalise_normals(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _extract_one_scene(
    model: torch.nn.Module,
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    points, colors, normals, objects, masks = _load_record_arrays(record)
    shifted, shift = _center_shift_numpy(points)
    features = build_input_features(
        points,
        colors,
        use_colors=True,
        use_normals=True,
        normals=_normalise_normals(normals),
    )
    coord = torch.from_numpy(shifted).to(device=device, dtype=torch.float32)
    feat = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(coord, feat)
    if not hasattr(output, "scene_tokens") or not hasattr(output, "representative_indices"):
        raise RuntimeError("LitePT wrapper did not expose native dec0 representative indices")
    tokens = output.scene_tokens.detach().float().cpu()
    xyz = output.scene_xyz.detach().float().cpu()
    inverse = output.inverse_map.detach().long().cpu()
    representatives = output.representative_indices.detach().long().cpu()
    if tokens.ndim != 2 or tokens.shape[1] != 72:
        raise RuntimeError(f"{record['scene']}: unexpected dec0 token shape {tuple(tokens.shape)}")
    if xyz.shape != (tokens.shape[0], 3):
        raise RuntimeError(f"{record['scene']}: token coordinate shape mismatch")
    if inverse.shape != (len(points),):
        raise RuntimeError(f"{record['scene']}: inverse map shape mismatch")
    if representatives.shape != (tokens.shape[0],):
        raise RuntimeError(f"{record['scene']}: representative index shape mismatch")
    token_targets = build_token_targets(
        representatives.numpy(),
        masks,
        [int(item["instance"]) for item in objects],
    )
    geometry_hash = json_sha256(
        {
            "scene": str(record["scene"]),
            "shift": array_sha256(shifted),
            "scene_xyz": array_sha256(xyz),
            "inverse_map": array_sha256(inverse),
            "representative_indices": array_sha256(representatives),
        }
    )
    return {
        "schema": CACHE_SCHEMA,
        "scene": str(record["scene"]),
        "kind": str(record["kind"]),
        "source_data_path": str(record["data_path"]),
        "source_data_sha256": str(record["data_sha256"]),
        "points": int(len(points)),
        "features": tokens,
        "scene_xyz": xyz,
        "inverse_map": inverse,
        "representative_indices": representatives,
        "shifted_points": torch.from_numpy(shifted),
        "shift": torch.from_numpy(shift),
        "object_token_indices": {
            str(int(key)): torch.from_numpy(value).long()
            for key, value in token_targets.items()
        },
        "objects": [dict(item) for item in objects],
        "geometry_sha256": geometry_hash,
        "empty_objects": [
            int(key) for key, value in token_targets.items() if len(value) == 0
        ],
    }


def _cache_path(root: Path, arm: str, scene: str) -> Path:
    return root / "feature_cache" / arm / f"{scene}.pt"


def cache_features(config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    manifest, _init_path = load_prepared(config)
    arm_cfg = config["arms"][arm]
    checkpoint = resolve_external(arm_cfg["checkpoint"])
    observed = file_sha256(checkpoint)
    if observed != str(arm_cfg["checkpoint_sha256"]):
        raise ValueError(f"{arm}: checkpoint hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("feature caching requires CUDA")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    set_seed(int(config["seed"]))
    model = __import__("delimit3d.evaluation.features", fromlist=["load_backbone"]).load_backbone(
        checkpoint,
        litept_root=resolve_external(config["paths"]["litept_root"]),
        in_channels=6,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    encoder_before = state_sha256(model.state_dict())
    root = output_root(config)
    records = _scene_record_map(manifest)
    cache_rows: list[dict[str, Any]] = []
    all_records = list(manifest["train_scenes"]) + list(manifest["validation_scenes"])
    for index, record in enumerate(all_records):
        scene = str(record["scene"])
        cache_path = _cache_path(root, arm, scene)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("schema") != CACHE_SCHEMA:
                raise ValueError(f"{cache_path}: cache schema mismatch")
            if payload.get("checkpoint_sha256") != observed:
                raise ValueError(f"{cache_path}: cache belongs to another checkpoint")
            cache_rows.append(
                {
                    "scene": scene,
                    "path": str(cache_path),
                    "geometry_sha256": payload["geometry_sha256"],
                    "tokens": int(payload["features"].shape[0]),
                    "empty_objects": payload.get("empty_objects", []),
                    "reused": True,
                }
            )
            continue
        payload = _extract_one_scene(model, record, config=config, device=device)
        payload["checkpoint_sha256"] = observed
        payload["encoder_tensor_sha256"] = encoder_before
        payload["repo_commit"] = effective_commit(config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        cache_rows.append(
            {
                "scene": scene,
                "path": str(cache_path),
                "geometry_sha256": payload["geometry_sha256"],
                "tokens": int(payload["features"].shape[0]),
                "empty_objects": payload["empty_objects"],
                "reused": False,
            }
        )
        print(
            json.dumps(
                {
                    "mode": "cache-features",
                    "arm": arm,
                    "scene": scene,
                    "completed": index + 1,
                    "total": len(all_records),
                    "tokens": int(payload["features"].shape[0]),
                }
            ),
            flush=True,
        )
        del payload
        torch.cuda.empty_cache()
    encoder_after = state_sha256(model.state_dict())
    if encoder_after != encoder_before:
        raise RuntimeError(f"{arm}: encoder hash changed during feature export")
    report = {
        "schema": "delimit3d_agile3d_feature_cache_report/v1",
        "experiment_id": config["experiment_id"],
        "arm": arm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": observed,
        "encoder_tensor_sha256_before": encoder_before,
        "encoder_tensor_sha256_after": encoder_after,
        "encoder_state_unchanged": encoder_before == encoder_after,
        "repo_commit": effective_commit(config),
        "cache_rows": cache_rows,
        "cache_count": len(cache_rows),
    }
    report_path = root / f"cache_report_{arm}.json"
    json_dump(report_path, report)
    _verify_cross_arm_geometry(root, manifest)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _verify_cross_arm_geometry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    public_report_path = root / "cache_report_public.json"
    delimit_report_path = root / "cache_report_delimit3d.json"
    if not public_report_path.exists() or not delimit_report_path.exists():
        return None
    public = json.loads(public_report_path.read_text())
    delimit = json.loads(delimit_report_path.read_text())
    public_map = {row["scene"]: row["geometry_sha256"] for row in public["cache_rows"]}
    delimit_map = {row["scene"]: row["geometry_sha256"] for row in delimit["cache_rows"]}
    if public_map != delimit_map:
        differing = sorted(
            scene for scene in set(public_map).union(delimit_map)
            if public_map.get(scene) != delimit_map.get(scene)
        )
        raise RuntimeError(f"public/Delimit3D geometry hashes differ: {differing[:10]}")
    report = {
        "schema": "delimit3d_agile3d_geometry_comparison/v1",
        "scene_count": len(public_map),
        "geometry_byte_identity": True,
        "scene_geometry_sha256": public_map,
        "public_encoder_tensor_sha256": public["encoder_tensor_sha256_after"],
        "delimit3d_encoder_tensor_sha256": delimit["encoder_tensor_sha256_after"],
    }
    json_dump(root / "geometry_comparison.json", report)
    return report


def load_cache(root: Path, arm: str, scene: str) -> dict[str, Any]:
    path = _cache_path(root, arm, scene)
    if not path.exists():
        raise FileNotFoundError(f"missing {arm} feature cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"{path}: cache schema mismatch")
    return payload


def _target_from_episode(
    cache: Mapping[str, Any],
    object_ids: Sequence[int],
) -> tuple[torch.Tensor, dict[str, list[int]], dict[str, list[int]], list[int]]:
    features = cache["features"]
    target = torch.zeros(features.shape[0], dtype=torch.long)
    clicks: dict[str, list[int]] = {"0": []}
    times: dict[str, list[int]] = {"0": []}
    valid_ids: list[int] = []
    for local_id, object_id in enumerate(object_ids, start=1):
        values = cache["object_token_indices"].get(str(int(object_id)))
        if values is None:
            raise KeyError(f"cache has no object {object_id}")
        indices = values.detach().cpu().numpy().astype(np.int64)
        if len(indices) == 0:
            raise RuntimeError(f"object {object_id} has no surviving representative token")
        if bool(torch.any(target[torch.from_numpy(indices)] != 0)):
            raise RuntimeError("selected object tuple overlaps after representative-token mapping")
        target[torch.from_numpy(indices)] = int(local_id)
        click = center_click(indices, cache["scene_xyz"].numpy())
        clicks[str(local_id)] = [int(click)]
        times[str(local_id)] = [0]
        valid_ids.append(int(object_id))
    return target, clicks, times, valid_ids


def _prefix_interaction(
    decoder: Agile3DClickDecoder,
    features: torch.Tensor,
    xyz: torch.Tensor,
    target: torch.Tensor,
    clicks: dict[str, list[int]],
    times: dict[str, list[int]],
    rounds: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    decoder.eval()
    with torch.inference_mode():
        for _ in range(int(rounds)):
            output = decoder(features, xyz, clicks=clicks, click_times=times)
            prediction = output["pred_masks"].argmax(dim=1).detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            prediction = enforce_click_labels(prediction, clicks)
            new_clicks, new_times, selected = simulated_corrections(
                prediction,
                target_np,
                xyz.detach().cpu().numpy(),
                clicks,
                times,
                training=True,
            )
            if not new_clicks:
                break
            clicks, times = append_clicks(clicks, times, new_clicks, new_times)
            events.extend(selected)
    decoder.train()
    return clicks, times, events


def _train_one_update(
    decoder: Agile3DClickDecoder,
    optimizer: torch.optim.Optimizer,
    cache: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, float | int]:
    object_ids = [int(value) for value in episode["objects"]]
    target_cpu, clicks, times, _valid = _target_from_episode(cache, object_ids)
    features = cache["features"].to(device=device, dtype=torch.float32)
    xyz = cache["scene_xyz"].to(device=device, dtype=torch.float32)
    target = target_cpu.to(device=device)
    clicks, times, prefix_events = _prefix_interaction(
        decoder,
        features,
        xyz,
        target,
        clicks,
        times,
        int(episode["click_prefix"]),
    )
    decoder.train()
    output = decoder(features, xyz, clicks=clicks, click_times=times)
    loss, details = compute_agile3d_losses(
        output,
        target,
        scene_xyz=xyz,
        clicks=clicks,
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
        alpha=float(config["loss"]["alpha"]),
        beta=float(config["loss"]["beta"]),
        radius=float(config["loss"]["radius"]),
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss {loss}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        decoder.parameters(), float(config["optimizer"]["clip_norm"])
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "loss_bce": float(details["loss_bce"].detach().cpu()),
        "loss_dice": float(details["loss_dice"].detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
        "objects": int(len(object_ids)),
        "prefix_rounds": int(episode["click_prefix"]),
        "prefix_events": int(len(prefix_events)),
        "tokens": int(features.shape[0]),
    }


def _save_training_checkpoint(
    path: Path,
    *,
    decoder: Agile3DClickDecoder,
    optimizer: torch.optim.Optimizer,
    arm: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    update: int,
    encoder_hash: str,
    init_hash: str,
    train_stats: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "arm": str(arm),
        "update": int(update),
        "repo_commit": effective_commit(config),
        "decoder_kwargs": decoder_kwargs(config),
        "decoder_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in decoder.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "decoder_tensor_sha256": state_sha256(decoder.state_dict()),
        "decoder_initialization_sha256": init_hash,
        "encoder_tensor_sha256": encoder_hash,
        "encoder_frozen": True,
        "episode_schedule_sha256": manifest["train_episodes_sha256"],
        "recent_train_stats": list(train_stats[-100:]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    manifest, init_path = load_prepared(config)
    root = output_root(config)
    cache_report_path = root / f"cache_report_{arm}.json"
    if not cache_report_path.exists():
        raise FileNotFoundError(f"run cache-features for {arm} first")
    cache_report = json.loads(cache_report_path.read_text())
    if not cache_report["encoder_state_unchanged"]:
        raise RuntimeError(f"{arm}: feature cache encoder invariant failed")
    decoder, init_report = load_initialization(
        init_path, expected_kwargs=decoder_kwargs(config)
    )
    init_hash = init_report["tensor_state_sha256"]
    decoder.to("cuda")
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    if not all(parameter.requires_grad for parameter in decoder.parameters()):
        raise RuntimeError("decoder contains a frozen parameter")
    if any(parameter is None for group in optimizer.param_groups for parameter in group["params"]):
        raise RuntimeError("invalid decoder optimizer parameter")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    set_seed(int(config["seed"]) + (0 if arm == "public" else 1))
    episodes = read_jsonl(Path(str(manifest["train_episodes_path"])))
    if len(episodes) != int(config["protocol"]["train_updates"]):
        raise RuntimeError("episode schedule length does not match training budget")
    records = _scene_record_map(manifest)
    cache_memory: dict[str, dict[str, Any]] = {}
    arm_root = root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    log_path = arm_root / "train_log.jsonl"
    stats: list[dict[str, Any]] = []
    start = time.time()
    with log_path.open("w") as log:
        for index, episode in enumerate(episodes):
            scene = str(episode["scene"])
            if scene not in cache_memory:
                cache_memory[scene] = load_cache(root, arm, scene)
            values = _train_one_update(
                decoder,
                optimizer,
                cache_memory[scene],
                episode,
                device=device,
                config=config,
            )
            row = {
                "update": int(index + 1),
                "scene": scene,
                "arm": arm,
                **values,
                "seconds": float(time.time() - start),
            }
            stats.append(row)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            if (index + 1) in {1000, 2500, 5000}:
                _save_training_checkpoint(
                    arm_root / f"decoder_u{index + 1:05d}.pt",
                    decoder=decoder,
                    optimizer=optimizer,
                    arm=arm,
                    config=config,
                    manifest=manifest,
                    update=index + 1,
                    encoder_hash=str(cache_report["encoder_tensor_sha256_after"]),
                    init_hash=init_hash,
                    train_stats=stats,
                )
            if (index + 1) % 100 == 0 or index == 0:
                log.flush()
                print(json.dumps({"mode": "train", **row}), flush=True)
    encoder_hash = str(cache_report["encoder_tensor_sha256_after"])
    final_path = arm_root / "decoder_u05000.pt"
    if not final_path.exists():
        _save_training_checkpoint(
            final_path,
            decoder=decoder,
            optimizer=optimizer,
            arm=arm,
            config=config,
            manifest=manifest,
            update=len(episodes),
            encoder_hash=encoder_hash,
            init_hash=init_hash,
            train_stats=stats,
        )
    report = {
        "schema": TRAIN_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "arm": arm,
        "repo_commit": effective_commit(config),
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
        "decoder_tensor_sha256": state_sha256(decoder.state_dict()),
        "decoder_initialization_sha256": init_hash,
        "encoder_tensor_sha256_before": encoder_hash,
        "encoder_tensor_sha256_after": encoder_hash,
        "encoder_state_unchanged": True,
        "episode_schedule_sha256": manifest["train_episodes_sha256"],
        "updates": len(episodes),
        "optimizer_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
        "seconds": float(time.time() - start),
        "last_stats": stats[-10:],
    }
    json_dump(arm_root / "train_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _center_shift_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    shift = np.asarray(
        [(minimum[0] + maximum[0]) / 2.0, (minimum[1] + maximum[1]) / 2.0, minimum[2]],
        dtype=np.float32,
    )
    return (points - shift).astype(np.float32, copy=False), shift


def _checkpoint_decoder(path: Path, config: Mapping[str, Any]) -> Agile3DClickDecoder:
    payload = torch.load(path.expanduser().resolve(strict=True), map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: decoder checkpoint schema mismatch")
    decoder = Agile3DClickDecoder(**decoder_kwargs(config))
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    if state_sha256(decoder.state_dict()) != payload["decoder_tensor_sha256"]:
        raise ValueError(f"{path}: decoder tensor hash mismatch")
    return decoder


def _evaluate_episode(
    decoder: Agile3DClickDecoder,
    cache: Mapping[str, Any],
    record: Mapping[str, Any],
    object_ids: Sequence[int],
    *,
    device: torch.device,
    config: Mapping[str, Any],
    mask_path: Path,
) -> dict[str, Any]:
    points, _colors, _normals, objects, masks = _load_record_arrays(record)
    object_ids = [int(value) for value in object_ids]
    target_token, clicks, times, _valid = _target_from_episode(cache, object_ids)
    raw_target = np.zeros(len(points), dtype=np.int64)
    for local_id, object_id in enumerate(object_ids, start=1):
        raw_indices = np.asarray(masks[object_id], dtype=np.int64)
        if bool(np.any(raw_target[raw_indices] != 0)):
            raise RuntimeError(f"{record['scene']}: selected raw objects overlap")
        raw_target[raw_indices] = int(local_id)
    features = cache["features"].to(device=device, dtype=torch.float32)
    xyz = cache["scene_xyz"].to(device=device, dtype=torch.float32)
    inverse = cache["inverse_map"].numpy()
    states: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    decoder.eval()
    max_total = int(config["protocol"]["click_budget"]) * len(object_ids)
    with torch.inference_mode():
        while True:
            output = decoder(features, xyz, clicks=clicks, click_times=times)
            prediction = output["pred_masks"].argmax(dim=1).detach().cpu().numpy()
            prediction = enforce_click_labels(prediction, clicks)
            total_clicks = sum(len(values) for values in clicks.values())
            object_ious = raw_object_ious(
                prediction,
                raw_target,
                inverse,
                len(object_ids),
            )
            state_index = len(states)
            predictions[f"state_{state_index:03d}"] = prediction.astype(np.int16)
            state = {
                "state": int(state_index),
                "total_clicks": int(total_clicks),
                "clicks_per_object": float(total_clicks / len(object_ids)),
                "clicks": {key: list(values) for key, values in clicks.items()},
                "click_times": {key: list(values) for key, values in times.items()},
                "object_ious": object_ious,
                "mean_iou": float(np.mean(list(object_ious.values()))),
            }
            states.append(state)
            if total_clicks >= max_total:
                break
            new_clicks, new_times, events = simulated_corrections(
                prediction,
                target_token.numpy(),
                xyz.detach().cpu().numpy(),
                clicks,
                times,
                training=False,
                click_center_method=str(config["protocol"]["click_center_method"]),
            )
            if not new_clicks:
                break
            state["new_click_events"] = events
            clicks, times = append_clicks(clicks, times, new_clicks, new_times)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, **predictions)
    threshold_values = [1, 3, 5, 10, 15, 20]
    thresholds: dict[str, Any] = {}
    for value in threshold_values:
        selected = metric_at_threshold(states, value * len(object_ids))
        thresholds[str(value)] = selected
    noc: dict[str, float] = {}
    for iou_threshold in (0.50, 0.65, 0.80, 0.85, 0.90):
        object_clicks: list[float] = []
        for local_id in range(1, len(object_ids) + 1):
            reached = [
                float(state["clicks_per_object"])
                for state in states
                if float(state["object_ious"][str(local_id)]) >= iou_threshold
            ]
            object_clicks.append(min(reached) if reached else float(config["protocol"]["click_budget"]))
        noc[f"{iou_threshold:.2f}"] = float(np.mean(object_clicks))
    return {
        "scene": str(record["scene"]),
        "panel_objects": object_ids,
        "object_count": len(object_ids),
        "states": states,
        "thresholds": thresholds,
        "noc": noc,
        "mask_path": str(mask_path),
        "target_points": {
            str(local_id): int(np.count_nonzero(raw_target == local_id))
            for local_id in range(1, len(object_ids) + 1)
        },
        "semantic": {
            str(local_id): next(
                (
                    dict(item)
                    for item in objects
                    if int(item["instance"]) == object_id
                ),
                {"instance": object_id},
            )
            for local_id, object_id in enumerate(object_ids, start=1)
        },
    }


def evaluate(config: Mapping[str, Any], arm: str, panel: str) -> dict[str, Any]:
    manifest, init_path = load_prepared(config)
    root = output_root(config)
    cache_report = json.loads((root / f"cache_report_{arm}.json").read_text())
    decoder_path = (
        Path(str(config["arms"][arm].get("decoder_checkpoint", ""))).expanduser()
        if config["arms"][arm].get("decoder_checkpoint")
        else root / arm / "decoder_u05000.pt"
    )
    if not decoder_path.exists():
        raise FileNotFoundError(f"missing trained decoder checkpoint: {decoder_path}")
    decoder = _checkpoint_decoder(decoder_path, config).to("cuda")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    set_seed(int(config["seed"]) + (0 if arm == "public" else 1))
    records = _scene_record_map(manifest)
    panels = [panel] if panel != "all" else ["MO-1", "MO-5", "MO-10"]
    arm_root = root / arm
    reports: dict[str, Any] = {}
    for panel_name in panels:
        panel_root = arm_root / "evaluation" / panel_name
        episodes_path = panel_root / "episodes.jsonl"
        scenes_path = panel_root / "scenes.jsonl"
        mask_root = panel_root / "masks"
        rows: list[dict[str, Any]] = []
        for index, episode in enumerate(manifest["panels"][panel_name]):
            scene = str(episode["scene"])
            cache = load_cache(root, arm, scene)
            row = _evaluate_episode(
                decoder,
                cache,
                records[scene],
                [int(value) for value in episode["objects"]],
                device=device,
                config=config,
                mask_path=mask_root / f"{index:05d}_{scene}.npz",
            )
            row["episode_index"] = int(index)
            row["panel"] = panel_name
            rows.append(row)
            if (index + 1) % 5 == 0 or index == 0:
                print(
                    json.dumps(
                        {
                            "mode": "evaluate",
                            "arm": arm,
                            "panel": panel_name,
                            "completed": index + 1,
                            "total": len(manifest["panels"][panel_name]),
                            "scene": scene,
                        }
                    ),
                    flush=True,
                )
        write_jsonl(episodes_path, rows)
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scene[str(row["scene"])].append(row)
        scene_rows: list[dict[str, Any]] = []
        for scene, scene_episodes in sorted(by_scene.items()):
            threshold_summary: dict[str, Any] = {}
            for value in (1, 3, 5, 10, 15, 20):
                candidates = [
                    float(episode["thresholds"][str(value)]["mean_iou"])
                    for episode in scene_episodes
                ]
                if candidates:
                    threshold_summary[str(value)] = {
                        "mean_iou": float(np.mean(candidates)),
                        "episode_count": len(candidates),
                    }
            noc_summary = {
                key: float(np.mean([float(episode["noc"][key]) for episode in scene_episodes]))
                for key in ("0.50", "0.65", "0.80", "0.85", "0.90")
            }
            scene_rows.append(
                {
                    "scene": scene,
                    "episodes": len(scene_episodes),
                    "thresholds": threshold_summary,
                    "noc": noc_summary,
                }
            )
        write_jsonl(scenes_path, scene_rows)
        report = {
            "schema": "delimit3d_agile3d_interactive_panel_report/v1",
            "experiment_id": config["experiment_id"],
            "arm": arm,
            "panel": panel_name,
            "decoder_checkpoint": str(decoder_path),
            "decoder_checkpoint_sha256": file_sha256(decoder_path),
            "encoder_checkpoint_sha256": cache_report["checkpoint_sha256"],
            "encoder_tensor_sha256": cache_report["encoder_tensor_sha256_after"],
            "encoder_state_unchanged": cache_report["encoder_state_unchanged"],
            "episodes": len(rows),
            "scenes": len(scene_rows),
            "thresholds": {
                str(value): float(
                    np.mean(
                        [
                            float(row["thresholds"][str(value)]["mean_iou"])
                            for row in rows
                        ]
                    )
                )
                for value in (1, 3, 5, 10, 15, 20)
                if rows
            },
            "noc": {
                key: float(np.mean([float(row["noc"][key]) for row in rows]))
                for key in ("0.50", "0.65", "0.80", "0.85", "0.90")
            },
            "scene_rows": str(scenes_path),
            "episode_rows": str(episodes_path),
        }
        json_dump(panel_root / "report.json", report)
        reports[panel_name] = report
        del decoder
        decoder = _checkpoint_decoder(decoder_path, config).to(device)
    aggregate = {
        "schema": "delimit3d_agile3d_interactive_evaluation/v1",
        "arm": arm,
        "reports": reports,
        "repo_commit": effective_commit(config),
    }
    json_dump(arm_root / "evaluation_report.json", aggregate)
    return aggregate


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    reports: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    for panel in ("MO-1", "MO-5", "MO-10"):
        public_path = root / "public" / "evaluation" / panel / "scenes.jsonl"
        delimit_path = root / "delimit3d" / "evaluation" / panel / "scenes.jsonl"
        if not public_path.exists() or not delimit_path.exists():
            continue
        public_rows = read_jsonl(public_path)
        delimit_rows = read_jsonl(delimit_path)
        public_map = {
            str(row["scene"]): {
                f"iou@{value}": float(row["thresholds"][str(value)]["mean_iou"])
                for value in (1, 3, 5, 10, 15, 20)
                if str(value) in row["thresholds"]
            }
            | {
                f"noc@{key}": float(row["noc"][key])
                for key in ("0.50", "0.65", "0.80", "0.85", "0.90")
            }
            for row in public_rows
        }
        delimit_map = {
            str(row["scene"]): {
                f"iou@{value}": float(row["thresholds"][str(value)]["mean_iou"])
                for value in (1, 3, 5, 10, 15, 20)
                if str(value) in row["thresholds"]
            }
            | {
                f"noc@{key}": float(row["noc"][key])
                for key in ("0.50", "0.65", "0.80", "0.85", "0.90")
            }
            for row in delimit_rows
        }
        panel_report: dict[str, Any] = {
            "scene_count": len(set(public_map).intersection(delimit_map)),
            "public": {
                key: float(np.mean([row[key] for row in public_map.values() if key in row]))
                for key in sorted({key for row in public_map.values() for key in row})
            },
            "delimit3d": {
                key: float(np.mean([row[key] for row in delimit_map.values() if key in row]))
                for key in sorted({key for row in delimit_map.values() for key in row})
            },
            "bootstrap": {},
        }
        for metric in ("iou@1", "iou@3", "iou@5", "iou@10", "iou@15", "iou@20"):
            public_metric = {
                scene: row[metric] for scene, row in public_map.items() if metric in row
            }
            delimit_metric = {
                scene: row[metric] for scene, row in delimit_map.items() if metric in row
            }
            if public_metric and delimit_metric:
                panel_report["bootstrap"][metric] = paired_scene_bootstrap(
                    public_metric,
                    delimit_metric,
                    metric,
                    samples=int(config["bootstrap"]["samples"]),
                    seed=int(config["bootstrap"]["seed"]),
                )
        for metric in ("noc@0.50", "noc@0.65", "noc@0.80", "noc@0.85", "noc@0.90"):
            public_metric = {
                scene: row[metric] for scene, row in public_map.items() if metric in row
            }
            delimit_metric = {
                scene: row[metric] for scene, row in delimit_map.items() if metric in row
            }
            if public_metric and delimit_metric:
                panel_report["bootstrap"][metric] = paired_scene_bootstrap(
                    public_metric,
                    delimit_metric,
                    metric,
                    samples=int(config["bootstrap"]["samples"]),
                    seed=int(config["bootstrap"]["seed"]),
                )
        reports[panel] = panel_report

    if "MO-5" in reports:
        mo5 = reports["MO-5"]
        b1 = mo5["bootstrap"].get("iou@1")
        b5 = mo5["bootstrap"].get("iou@5")
        public_noc = mo5["public"].get("noc@0.80")
        delimit_noc = mo5["delimit3d"].get("noc@0.80")
        conditions = {
            "iou_at_1_delta_ge_2pp": bool(
                b1 is not None and float(b1["observed_difference"]) >= 0.02
            ),
            "iou_at_5_delta_ge_1_5pp": bool(
                b5 is not None and float(b5["observed_difference"]) >= 0.015
            ),
            "iou_at_1_ci_lower_gt_zero": bool(
                b1 is not None and float(b1["ci95_lower"]) > 0.0
            ),
            "iou_at_5_ci_lower_gt_zero": bool(
                b5 is not None and float(b5["ci95_lower"]) > 0.0
            ),
            "noc_at_80_not_worse": bool(
                public_noc is not None
                and delimit_noc is not None
                and delimit_noc <= public_noc
            ),
        }
        gate = {
            "panel": "MO-5",
            "conditions": conditions,
            "passed": bool(all(conditions.values())),
            "stage_b_authorized": bool(all(conditions.values())),
        }
    report = {
        "schema": "delimit3d_agile3d_stage_a_aggregate/v1",
        "experiment_id": config["experiment_id"],
        "repo_commit": effective_commit(config),
        "panels": reports,
        "stage_a_gate": gate,
        "stage_b_launched": False,
    }
    json_dump(root / "aggregate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _dense_smoke_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    target = np.asarray([1, 1, 0, 0], dtype=np.int64)
    prediction = np.asarray([0, 0, 1, 0], dtype=np.int64)
    return prediction, target, xyz


def smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest, init_path = load_prepared(config)
    root = output_root(config)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    set_seed(int(config["seed"]))
    geometry = json.loads((root / "geometry_comparison.json").read_text())
    reports: list[dict[str, Any]] = []
    episode = next(
        row for row in read_jsonl(Path(str(manifest["train_episodes_path"])))
        if len(row["objects"]) >= 3
    )
    for arm in ("public", "delimit3d"):
        cache = load_cache(root, arm, str(episode["scene"]))
        decoder, init_report = load_initialization(
            init_path, expected_kwargs=decoder_kwargs(config)
        )
        decoder.to(device)
        target, clicks, times, _ = _target_from_episode(
            cache, [int(value) for value in episode["objects"][:3]]
        )
        features = cache["features"].to(device=device, dtype=torch.float32)
        xyz = cache["scene_xyz"].to(device=device, dtype=torch.float32)
        decoder.eval()
        with torch.inference_mode():
            output = decoder(features, xyz, clicks=clicks, click_times=times)
        logits = output["pred_masks"]
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"{arm}: smoke logits are non-finite")
        decoder.train()
        loss, details = compute_agile3d_losses(
            output,
            target.to(device),
            scene_xyz=xyz,
            clicks=clicks,
            bce_weight=float(config["loss"]["bce_weight"]),
            dice_weight=float(config["loss"]["dice_weight"]),
            alpha=float(config["loss"]["alpha"]),
            beta=float(config["loss"]["beta"]),
            radius=float(config["loss"]["radius"]),
        )
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(config["optimizer"]["clip_norm"]))
        optimizer.step()
        # Ten evaluation events exercise the KD-tree policy and background handling.
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        pred = enforce_click_labels(pred, clicks)
        event_count = 0
        for _ in range(10):
            new_clicks, new_times, events = simulated_corrections(
                pred,
                target.numpy(),
                cache["scene_xyz"].numpy(),
                clicks,
                times,
                training=False,
            )
            if not new_clicks:
                break
            clicks, times = append_clicks(clicks, times, new_clicks, new_times)
            event_count += len(events)
            with torch.inference_mode():
                logits = decoder(features, xyz, clicks=clicks, click_times=times)["pred_masks"]
            pred = enforce_click_labels(
                logits.argmax(dim=1).detach().cpu().numpy(), clicks
            )
        reports.append(
            {
                "arm": arm,
                "scene": episode["scene"],
                "objects": episode["objects"][:3],
                "init_hash": init_report["tensor_state_sha256"],
                "loss": float(loss.detach().cpu()),
                "loss_bce": float(details["loss_bce"].detach().cpu()),
                "loss_dice": float(details["loss_dice"].detach().cpu()),
                "finite": True,
                "event_count": int(event_count),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "geometry_sha256": cache["geometry_sha256"],
                "encoder_tensor_sha256": cache.get("encoder_tensor_sha256"),
                "encoder_state_unchanged": True,
            }
        )
        del decoder
        torch.cuda.empty_cache()
    dense_prediction, dense_target, dense_xyz = _dense_smoke_fixture()
    dense = [
        {
            "cluster_id": int(row["cluster_id"]),
            "center_index": int(row["center_index"]),
            "target_label": int(row["target_label"]),
        }
        for row in __import__("delimit3d.evaluation.agile3d_protocol", fromlist=["dense_click_reference"]).dense_click_reference(
            dense_prediction, dense_target, dense_xyz
        )
    ]
    kdtree = [
        {
            "cluster_id": int(row["cluster_id"]),
            "center_index": int(row["center_index"]),
            "target_label": int(row["target_label"]),
        }
        for row in __import__("delimit3d.evaluation.agile3d_protocol", fromlist=["kdtree_click_reference"]).kdtree_click_reference(
            dense_prediction, dense_target, dense_xyz
        )
    ]
    if dense != kdtree:
        raise RuntimeError("KD-tree click selection does not match dense fixture")
    report = {
        "schema": "delimit3d_agile3d_smoke/v1",
        "experiment_id": config["experiment_id"],
        "geometry_byte_identity": geometry["geometry_byte_identity"],
        "arms": reports,
        "dense_vs_kdtree_equivalent": True,
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in reports),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "hostname": platform.node(),
    }
    json_dump(root / "smoke_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    verify_config(config)
    if args.mode == "prepare":
        prepare(config)
    elif args.mode == "cache-features":
        if not args.arm:
            raise ValueError("--arm is required for cache-features")
        cache_features(config, args.arm)
    elif args.mode == "smoke":
        smoke(config)
    elif args.mode == "train":
        if not args.arm:
            raise ValueError("--arm is required for train")
        train(config, args.arm)
    elif args.mode == "evaluate":
        if not args.arm:
            raise ValueError("--arm is required for evaluate")
        evaluate(config, args.arm, args.panel)
    elif args.mode == "aggregate":
        aggregate(config)


if __name__ == "__main__":
    main()

