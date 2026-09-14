#!/usr/bin/env python3
"""Prepare, smoke-test, and run the public PTv3 Delimit3D adaptation.

The runner is intentionally a separate experiment family from the LitePT and
Sonata arms.  It uses the audited Structured3D source manifests and the same
deterministic multigranular sampler, while loading the public ScanNet PTv3
backbone and adapting it with the Delimit3D contrastive objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


CHECKPOINT_SCHEMA = "delimit3d_ptv3_adaptation_checkpoint/v1"
PROVENANCE_SCHEMA = "delimit3d_ptv3_adaptation_provenance/v1"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    config = _expand(raw)
    config["_config_path"] = str(path.resolve())
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                *args,
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def required(config: Mapping[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {}).get(key)
    if value in (None, ""):
        raise ValueError(f"Missing config value {section}.{key}")
    return value


def resolved_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "repo_root": Path(config["_config_path"]).resolve().parents[2],
        "ptv3_root": Path(required(config, "model", "ptv3_root")).expanduser(),
        "checkpoint": Path(required(config, "model", "checkpoint")).expanduser(),
        "source_root": Path(required(config, "data", "source_root")).expanduser(),
        "train_split": Path(required(config, "data", "train_split")).expanduser(),
        "output": Path(required(config, "paths", "output")).expanduser(),
    }


def build_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = resolved_paths(config)
    checkpoint_sha = (
        sha256_file(paths["checkpoint"]) if paths["checkpoint"].is_file() else None
    )
    expected_sha = str(config.get("model", {}).get("checkpoint_sha256", "")) or None
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "experiment": dict(config.get("experiment", {})),
        "model": {
            **dict(config.get("model", {})),
            "checkpoint": str(paths["checkpoint"]),
            "checkpoint_sha256_actual": checkpoint_sha,
            "checkpoint_sha256_expected": expected_sha,
            "checkpoint_sha256_match": (
                None
                if checkpoint_sha is None or expected_sha is None
                else checkpoint_sha == expected_sha
            ),
            "ptv3_source_commit_actual": git_value(
                paths["ptv3_root"], "rev-parse", "HEAD"
            ),
            "ptv3_source_status": git_value(paths["ptv3_root"], "status", "--short"),
        },
        "data": {
            **dict(config.get("data", {})),
            "source_root": str(paths["source_root"]),
            "train_split": str(paths["train_split"]),
            "train_split_sha256": (
                sha256_file(paths["train_split"])
                if paths["train_split"].is_file()
                else None
            ),
        },
        "sampling": dict(config.get("sampling", {})),
        "optimizer": dict(config.get("optimizer", {})),
        "code": {
            "repo_commit": git_value(paths["repo_root"], "rev-parse", "HEAD"),
            "repo_status_porcelain": git_value(paths["repo_root"], "status", "--short"),
        },
        "resolved_config_sha256": sha256_json(config),
    }


def prepare(config: Mapping[str, Any]) -> None:
    paths = resolved_paths(config)
    paths["output"].mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config)
    (paths["output"] / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (paths["output"] / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {key: value for key, value in config.items() if not key.startswith("_")},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    missing = [
        str(path)
        for path in (
            paths["ptv3_root"],
            paths["checkpoint"],
            paths["source_root"],
            paths["train_split"],
        )
        if not path.exists()
    ]
    if provenance["model"]["checkpoint_sha256_match"] is False:
        raise RuntimeError("Public PTv3 checkpoint SHA-256 does not match the resolved config")
    status = {
        "ready_for_gpu_preflight": not missing
        and provenance["model"]["checkpoint_sha256_match"] in (True, None),
        "missing_now": missing,
        "source_root_exists": paths["source_root"].is_dir(),
        "train_split_exists": paths["train_split"].is_file(),
        "checkpoint_sha256": provenance["model"]["checkpoint_sha256_actual"],
        "ptv3_source_commit": provenance["model"]["ptv3_source_commit_actual"],
        "provenance": str(paths["output"] / "provenance.json"),
    }
    (paths["output"] / "prepare_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_scene(scene_id: str, config: Mapping[str, Any]):
    from delimit3d.training.adaptation import load_scene_source

    paths = resolved_paths(config)
    manifest = paths["source_root"] / scene_id / "source_manifest.json"
    return load_scene_source(
        manifest,
        input_features="rgbn6",
        coordinate_normalization_policy=str(
            config.get("data", {}).get(
                "coordinate_normalization_policy",
                "structured3d_local_y_up_to_scannet_z_up_v1",
            )
        ),
        verify_hashes=True,
    )


def _load_runtime(config: Mapping[str, Any]):
    import torch
    from delimit3d.training.adaptation import sampling_config_for_profile
    from delimit3d.training.ptv3_adaptation import PTv3ContrastiveModel

    paths = resolved_paths(config)
    model_config = config.get("model", {})
    model = PTv3ContrastiveModel(
        ptv3_root=paths["ptv3_root"],
        checkpoint=paths["checkpoint"],
        grid_size=float(model_config.get("grid_size", 0.02)),
        projection_dim=int(model_config.get("projection_dim", 128)),
        projection_hidden_dim=int(model_config.get("projection_hidden_dim", 128)),
        enable_flash=bool(model_config.get("enable_flash", True)),
        shuffle_orders=bool(model_config.get("shuffle_orders", True)),
        multiscale_supervision=bool(
            model_config.get("multiscale_supervision", False)
        ),
        multiscale_loss_weight=float(
            model_config.get("multiscale_loss_weight", 0.4)
        ),
        multiscale_warmup_updates=int(
            model_config.get("multiscale_warmup_updates", 32)
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PTv3 adaptation requires a CUDA compute node")
    model.to(device)
    profile = str(config.get("sampling", {}).get("profile", "pf_hnm_512_v1"))
    return model, device, sampling_config_for_profile(profile)


def _state_hash(module: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _optimizer_for(model: Any, config: Mapping[str, Any]):
    import torch

    optimizer_config = config.get("optimizer", {})
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    if not encoder_parameters or not head_parameters:
        raise RuntimeError("PTv3 optimizer parameter groups are unexpectedly empty")
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_ids = {id(parameter) for parameter in head_parameters}
    if encoder_ids & head_ids:
        raise RuntimeError("A parameter appears in both PTv3 optimizer groups")
    return torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(optimizer_config.get("backbone_lr", 1e-4)),
            },
            {
                "params": head_parameters,
                "lr": float(optimizer_config.get("head_lr", 3e-3)),
            },
        ],
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
    )


def _plan_and_inputs(
    *,
    scene: Any,
    scene_epoch: int,
    update: int,
    seed: int,
    device: Any,
    sampling: Any,
    coverage_states: Mapping[str, Any],
    source_cells: list[str],
):
    import torch
    from delimit3d.training.adaptation import build_scene_visit_input
    from delimit3d.training.ptv3_adaptation import build_ptv3_v2_plan

    visit = build_scene_visit_input(
        scene=scene,
        epoch=update - 1,
        global_seed=seed,
        input_features="rgbn6",
        augmentation_profile="none",
    )
    points = torch.from_numpy(visit.points).to(device)
    features = torch.from_numpy(visit.features).to(device)
    plan = build_ptv3_v2_plan(
        scene=scene,
        epoch=scene_epoch,
        seed=seed,
        points=points,
        cells=source_cells,
        sampling=sampling,
        coverage_states=coverage_states,
    )
    return points, features, plan


def _one_step(config: Mapping[str, Any], *, updates: int, output: Path) -> dict[str, Any]:
    import torch
    from delimit3d.data.contrastive_sampler_v2 import DeterministicCoverageState
    from delimit3d.training.contracts import load_scene_ids, stable_seed

    model, device, sampling = _load_runtime(config)
    paths = resolved_paths(config)
    scene_ids = load_scene_ids(paths["train_split"])
    if not scene_ids:
        raise RuntimeError("Training split is empty")
    seed = int(config.get("experiment", {}).get("seed", 20260911))
    optimizer = _optimizer_for(model, config)
    initial_hash = _state_hash(model.encoder)
    model.train()
    log_path = output / "metrics.jsonl"
    source_cells = [
        str(cell)
        for cell in config.get("sampling", {}).get(
            "source_cells", ["2d/g02", "2d/g05", "2d/g08"]
        )
    ]
    scene_cache: dict[str, Any] = {}
    coverage_states_by_scene: dict[str, dict[str, Any]] = {}
    scene_visit_epochs: dict[str, int] = {}
    last: dict[str, Any] = {}
    checkpoint_updates = {
        int(item)
        for item in config.get("experiment", {}).get("checkpoint_updates", [64, 128, 256])
    }
    for update in range(1, int(updates) + 1):
        scene_id = scene_ids[(update - 1) % len(scene_ids)]
        if scene_id not in scene_cache:
            scene_cache[scene_id] = _load_scene(scene_id, config)
        scene = scene_cache[scene_id]
        scene_epoch = scene_visit_epochs.get(scene_id, 0)
        coverage_states = coverage_states_by_scene.setdefault(
            scene_id, {cell: None for cell in source_cells}
        )
        points, features, plan = _plan_and_inputs(
            scene=scene,
            scene_epoch=scene_epoch,
            update=update,
            seed=seed,
            device=device,
            sampling=sampling,
            coverage_states=coverage_states,
            source_cells=source_cells,
        )
        for cell, coverage in plan.coverage_by_cell.items():
            state_after = coverage.get("state_after")
            if state_after is not None:
                coverage_states[str(cell)] = DeterministicCoverageState.from_dict(
                    state_after
                ).for_epoch(scene_epoch + 1)
        scene_visit_epochs[scene_id] = scene_epoch + 1

        optimizer.zero_grad(set_to_none=True)
        forward_seed = stable_seed(seed, scene_id, update, "ptv3-forward")
        result = model(
            points=points,
            features=features,
            batch=None,
            frame_groups=plan.groups,
            seed=forward_seed,
        )
        loss = result["loss_total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite PTv3 loss at update {update}")
        loss.backward()
        clip = float(config.get("optimizer", {}).get("gradient_clip_norm", 0.1))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        optimizer.step()
        record = {
            "update": update,
            "scene_id": scene_id,
            "loss": float(loss.detach()),
            "cosine_gap": float(result.get("cosine_gap", 0.0)),
            "positive_cosine_mean": float(result.get("positive_cosine_mean", 0.0)),
            "negative_cosine_mean": float(result.get("negative_cosine_mean", 0.0)),
            "hardest_negative_ranking_accuracy": float(
                result.get("hardest_negative_ranking_accuracy", 0.0)
            ),
            "temperature": float(result.get("temperature", 0.0)),
            "group_count": int(result.get("group_count", len(plan.groups))),
            "final_token_count": int(result.get("final_token_count", 0)),
            "feature_dim": int(result.get("feature_dim", 0)),
            "grad_norm_before_clip": grad_norm,
            "seed": int(forward_seed),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        last = record
        if update in checkpoint_updates or update == int(updates):
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "encoder_state_sha256": _state_hash(model.encoder),
                "initial_encoder_state_sha256": initial_hash,
                "config_sha256": sha256_json(config),
                "metrics": record,
            }
            torch.save(checkpoint, output / f"checkpoint_u{update:06d}.pt")
        if update % 16 == 0 or update == 1:
            print(json.dumps(record, sort_keys=True), flush=True)
    final_hash = _state_hash(model.encoder)
    report = {
        "schema_version": "delimit3d_ptv3_adaptation_train_report/v1",
        "updates": int(updates),
        "initial_encoder_state_sha256": initial_hash,
        "final_encoder_state_sha256": final_hash,
        "encoder_changed": initial_hash != final_hash,
        "last": last,
        "output": str(output),
    }
    (output / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def smoke(config: Mapping[str, Any]) -> None:
    """Run one real PTv3 forward/backward/step on the first train scene."""

    import torch
    from delimit3d.data.contrastive_sampler_v2 import DeterministicCoverageState
    from delimit3d.training.contracts import load_scene_ids, stable_seed

    model, device, sampling = _load_runtime(config)
    paths = resolved_paths(config)
    scene_ids = load_scene_ids(paths["train_split"])
    if not scene_ids:
        raise RuntimeError("Training split is empty")
    seed = int(config.get("experiment", {}).get("seed", 20260911))
    source_cells = [
        str(cell)
        for cell in config.get("sampling", {}).get(
            "source_cells", ["2d/g02", "2d/g05", "2d/g08"]
        )
    ]
    scene = _load_scene(scene_ids[0], config)
    coverage_states = {cell: None for cell in source_cells}
    points, features, plan = _plan_and_inputs(
        scene=scene,
        scene_epoch=0,
        update=1,
        seed=seed,
        device=device,
        sampling=sampling,
        coverage_states=coverage_states,
        source_cells=source_cells,
    )
    for cell, coverage in plan.coverage_by_cell.items():
        if coverage.get("state_after") is not None:
            coverage_states[str(cell)] = DeterministicCoverageState.from_dict(
                coverage["state_after"]
            )
    before = _state_hash(model.encoder)
    optimizer = _optimizer_for(model, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad():
        result = model(
            points=points,
            features=features,
            batch=None,
            frame_groups=plan.groups,
            seed=stable_seed(seed, scene.scene_id, "ptv3-smoke"),
        )
        loss = result["loss_total"]
        if not torch.isfinite(loss):
            raise RuntimeError("PTv3 smoke produced a non-finite loss")
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("optimizer", {}).get("gradient_clip_norm", 0.1)))
        )
        optimizer.step()
    after = _state_hash(model.encoder)
    feats = result["point_features"]
    geometry = {
        "scene_id": scene.scene_id,
        "raw_point_count": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "final_token_count": int(result["final_token_count"]),
        "group_count": len(plan.groups),
        "raw_to_token_min": int(result["raw_to_token"].min().item()),
        "raw_to_token_max": int(result["raw_to_token"].max().item()),
        "representative_count": int(result["representative_indices"].shape[0]),
        "finite_features": bool(torch.isfinite(feats).all().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "loss": float(loss.detach()),
        "grad_norm_before_clip": grad_norm,
        "encoder_state_sha256_before": before,
        "encoder_state_sha256_after_one_step": after,
        "encoder_changed_after_one_step": before != after,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "source_manifest_sha256": scene.manifest_sha256,
    }
    if not geometry["finite_features"] or not geometry["finite_loss"]:
        raise RuntimeError(f"PTv3 smoke contract failed: {geometry}")
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(
        json.dumps(geometry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(geometry, indent=2, sort_keys=True))


def preflight(config: Mapping[str, Any]) -> None:
    import torch
    from delimit3d.training.contracts import load_scene_ids

    paths = resolved_paths(config)
    scene_ids = load_scene_ids(paths["train_split"])
    if not scene_ids:
        raise RuntimeError("Training split is empty")
    model, _, _ = _load_runtime(config)
    status: dict[str, Any] = {
        "ready_for_train": True,
        "scene_count": len(scene_ids),
        "first_scene": scene_ids[0],
        "encoder_state_sha256": _state_hash(model.encoder),
        "feature_dim": int(model.feature_dim),
        "expected_updates": int(config.get("experiment", {}).get("updates", 256)),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        status["gpu"] = {
            "name": props.name,
            "total_memory": int(props.total_memory),
        }
    else:
        status["gpu"] = {"name": "none", "total_memory": 0}
    paths["output"].mkdir(parents=True, exist_ok=True)
    (paths["output"] / "preflight.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


def train(config: Mapping[str, Any]) -> None:
    paths = resolved_paths(config)
    paths["output"].mkdir(parents=True, exist_ok=True)
    updates = int(config.get("experiment", {}).get("updates", 256))
    report = _one_step(config, updates=updates, output=paths["output"])
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("prepare", "smoke", "preflight", "train"),
        required=True,
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve(strict=True))
    if args.mode == "prepare":
        prepare(config)
    elif args.mode == "smoke":
        smoke(config)
    elif args.mode == "preflight":
        preflight(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
