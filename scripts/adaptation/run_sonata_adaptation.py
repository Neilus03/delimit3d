#!/usr/bin/env python3
"""Prepare, preflight, or run the 256-update Sonata Delimit3D adaptation.

This runner is intentionally independent from the validated LitePT runner.  It
uses the same audited Structured3D scene/proposal loader and the same
coverage_multigranular_v2 sampling semantics, while the backbone is the public
Sonata encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


CHECKPOINT_SCHEMA = "delimit3d_sonata_adaptation_checkpoint/v1"
PROVENANCE_SCHEMA = "delimit3d_sonata_adaptation_provenance/v1"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    cfg = _expand(raw)
    cfg["_config_path"] = str(path.resolve())
    return cfg


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
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def required(cfg: Mapping[str, Any], section: str, key: str) -> Any:
    value = cfg.get(section, {}).get(key)
    if value in (None, ""):
        raise ValueError(f"Missing config value {section}.{key}")
    return value


def resolved_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    return {
        "repo_root": Path(cfg["_config_path"]).resolve().parents[2],
        "sonata_root": Path(required(cfg, "model", "sonata_root")).expanduser(),
        "checkpoint": Path(required(cfg, "model", "checkpoint")).expanduser(),
        "source_root": Path(required(cfg, "data", "source_root")).expanduser(),
        "train_split": Path(required(cfg, "data", "train_split")).expanduser(),
        "output": Path(required(cfg, "paths", "output")).expanduser(),
    }


def build_provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    p = resolved_paths(cfg)
    repo = p["repo_root"]
    checkpoint_sha = (
        sha256_file(p["checkpoint"]) if p["checkpoint"].is_file() else None
    )
    expected_sha = str(cfg.get("model", {}).get("checkpoint_sha256", "")) or None
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "experiment": cfg.get("experiment", {}),
        "model": {
            **dict(cfg.get("model", {})),
            "checkpoint": str(p["checkpoint"]),
            "checkpoint_sha256_actual": checkpoint_sha,
            "checkpoint_sha256_expected": expected_sha,
            "checkpoint_sha256_match": (
                None if checkpoint_sha is None or expected_sha is None else checkpoint_sha == expected_sha
            ),
            "sonata_source_commit": git_value(p["sonata_root"], "rev-parse", "HEAD"),
        },
        "data": {
            **dict(cfg.get("data", {})),
            "source_root": str(p["source_root"]),
            "train_split": str(p["train_split"]),
            "train_split_sha256": (
                sha256_file(p["train_split"]) if p["train_split"].is_file() else None
            ),
        },
        "sampling": dict(cfg.get("sampling", {})),
        "optimizer": dict(cfg.get("optimizer", {})),
        "code": {
            "repo_commit": git_value(repo, "rev-parse", "HEAD"),
            "repo_status_porcelain": git_value(repo, "status", "--short"),
        },
        "resolved_config_sha256": sha256_json(cfg),
    }


def prepare(cfg: Mapping[str, Any]) -> None:
    p = resolved_paths(cfg)
    p["output"].mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(cfg)
    (p["output"] / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (p["output"] / "resolved_config.yaml").write_text(
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False),
        encoding="utf-8",
    )
    missing = [str(path) for path in (p["sonata_root"], p["checkpoint"]) if not path.exists()]
    if provenance["model"]["checkpoint_sha256_match"] is False:
        raise RuntimeError("Sonata checkpoint SHA-256 does not match the resolved config")
    status = {
        "ready_for_gpu_preflight": not missing and provenance["model"]["checkpoint_sha256_match"] in (True, None),
        "missing_now": missing,
        "source_root_exists": p["source_root"].is_dir(),
        "train_split_exists": p["train_split"].is_file(),
        "provenance": str(p["output"] / "provenance.json"),
    }
    (p["output"] / "prepare_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_scene(scene_id: str, cfg: Mapping[str, Any]):
    from delimit3d.training.adaptation import load_scene_source

    p = resolved_paths(cfg)
    manifest = p["source_root"] / scene_id / "source_manifest.json"
    return load_scene_source(
        manifest,
        input_features="rgbn6",
        coordinate_normalization_policy=str(
            cfg.get("data", {}).get(
                "coordinate_normalization_policy", "native"
            )
        ),
        verify_hashes=True,
    )


def _load_runtime(cfg: Mapping[str, Any]):
    import torch
    from delimit3d.training.adaptation import sampling_config_for_profile
    from delimit3d.training.sonata_adaptation import SonataContrastiveModel

    p = resolved_paths(cfg)
    model_cfg = cfg.get("model", {})
    model = SonataContrastiveModel(
        sonata_root=p["sonata_root"],
        checkpoint=p["checkpoint"],
        grid_size=float(model_cfg.get("grid_size", 0.02)),
        projection_dim=int(model_cfg.get("projection_dim", 128)),
        projection_hidden_dim=int(model_cfg.get("projection_hidden_dim", 128)),
        enable_flash=bool(model_cfg.get("enable_flash", True)),
        patch_size=int(model_cfg.get("patch_size", 1024)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Sonata preflight/train requires a CUDA compute node")
    model.to(device)
    profile = str(cfg.get("sampling", {}).get("profile", "pf_hnm_512_v1"))
    sampling = sampling_config_for_profile(profile)
    return model, device, sampling


def _state_hash(module: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _one_step(cfg: Mapping[str, Any], *, updates: int, output: Path) -> dict[str, Any]:
    import torch
    from delimit3d.data.contrastive_sampler_v2 import DeterministicCoverageState
    from delimit3d.training.adaptation import build_scene_visit_input
    from delimit3d.training.sonata_adaptation import build_sonata_v2_plan
    from delimit3d.training.contracts import load_scene_ids, stable_seed

    model, device, sampling = _load_runtime(cfg)
    p = resolved_paths(cfg)
    scene_ids = load_scene_ids(p["train_split"])
    if not scene_ids:
        raise RuntimeError("Training split is empty")
    seed = int(cfg.get("experiment", {}).get("seed", 20260911))
    optimizer_cfg = cfg.get("optimizer", {})
    encoder_params = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    head_params = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    if not encoder_params or not head_params:
        raise RuntimeError("Sonata optimizer parameter groups are unexpectedly empty")
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(optimizer_cfg.get("backbone_lr", 1e-4))},
            {"params": head_params, "lr": float(optimizer_cfg.get("head_lr", 3e-3))},
        ],
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
    )
    initial_hash = _state_hash(model.encoder)
    model.train()
    log_path = output / "metrics.jsonl"
    scene_cache: dict[str, Any] = {}
    coverage_states: dict[str, DeterministicCoverageState | None] = {
        str(cell): None
        for cell in cfg.get("sampling", {}).get(
            "source_cells", ["2d/g02", "2d/g05", "2d/g08"]
        )
    }
    last: dict[str, Any] = {}
    for update in range(1, int(updates) + 1):
        scene_id = scene_ids[(update - 1) % len(scene_ids)]
        if scene_id not in scene_cache:
            scene_cache[scene_id] = _load_scene(scene_id, cfg)
        scene = scene_cache[scene_id]
        points_np = build_scene_visit_input(
            scene=scene,
            epoch=update - 1,
            global_seed=seed,
            input_features="rgbn6",
            augmentation_profile="none",
        ).points
        points = torch.from_numpy(points_np).to(device)
        plan = build_sonata_v2_plan(
            scene=scene,
            epoch=update - 1,
            seed=seed,
            points=points,
            cells=cfg.get("sampling", {}).get(
                "source_cells", ["2d/g02", "2d/g05", "2d/g08"]
            ),
            sampling=sampling,
            coverage_states=coverage_states,
        )
        for cell, coverage in plan.coverage_by_cell.items():
            state_after = coverage.get("state_after")
            if state_after is not None:
                coverage_states[str(cell)] = DeterministicCoverageState.from_dict(
                    state_after
                ).for_epoch(update)
        optimizer.zero_grad(set_to_none=True)
        forward_seed = stable_seed(seed, scene_id, update, "sonata-forward")
        result = model(
            points=points,
            colors=scene.colors,
            normals=scene.normals,
            batch=None,
            frame_groups=plan.groups,
            seed=forward_seed,
        )
        loss = result["loss_total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {update}")
        loss.backward()
        clip = float(optimizer_cfg.get("gradient_clip_norm", 0.1))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        optimizer.step()
        record = {
            "update": update,
            "scene_id": scene_id,
            "loss": float(loss.detach()),
            "cosine_gap": float(result.get("cosine_gap", 0.0)),
            "hardest_negative_ranking_accuracy": float(
                result.get("hardest_negative_ranking_accuracy", 0.0)
            ),
            "temperature": float(result.get("temperature", 0.0)),
            "group_count": int(result.get("group_count", len(plan.groups))),
            "final_token_count": int(result.get("final_token_count", 0)),
            "grad_norm_before_clip": grad_norm,
            "seed": int(forward_seed),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        last = record
        checkpoints = set(int(x) for x in cfg.get("experiment", {}).get("checkpoint_updates", [256]))
        if update in checkpoints or update == int(updates):
            ckpt = {
                "schema_version": CHECKPOINT_SCHEMA,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "encoder_state_sha256": _state_hash(model.encoder),
                "initial_encoder_state_sha256": initial_hash,
                "config_sha256": sha256_json(cfg),
                "metrics": record,
            }
            torch.save(ckpt, output / f"checkpoint_u{update:06d}.pt")
        if update % 16 == 0 or update == 1:
            print(json.dumps(record, sort_keys=True), flush=True)
    final_hash = _state_hash(model.encoder)
    report = {
        "schema_version": "delimit3d_sonata_adaptation_train_report/v1",
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


def smoke(cfg: Mapping[str, Any]) -> None:
    """Run one no-grad scene forward and prove the dense mapping contract."""
    import torch
    from delimit3d.training.adaptation import build_scene_visit_input
    from delimit3d.training.contracts import load_scene_ids, stable_seed
    from delimit3d.data.contrastive_sampler_v2 import DeterministicCoverageState
    from delimit3d.training.sonata_adaptation import build_sonata_v2_plan

    model, device, sampling = _load_runtime(cfg)
    p = resolved_paths(cfg)
    scene_ids = load_scene_ids(p["train_split"])
    if not scene_ids:
        raise RuntimeError("Training split is empty")
    seed = int(cfg.get("experiment", {}).get("seed", 20260911))
    scene = _load_scene(scene_ids[0], cfg)
    visit = build_scene_visit_input(
        scene=scene,
        epoch=0,
        global_seed=seed,
        input_features="rgbn6",
        augmentation_profile="none",
    )
    points = torch.from_numpy(visit.points).to(device)
    plan = build_sonata_v2_plan(
        scene=scene,
        epoch=0,
        seed=seed,
        points=points,
        cells=cfg.get("sampling", {}).get(
            "source_cells", ["2d/g02", "2d/g05", "2d/g08"]
        ),
        sampling=sampling,
    )
    before = _state_hash(model.encoder)
    model.eval()
    with torch.no_grad():
        result = model(
            points=points,
            colors=scene.colors,
            normals=scene.normals,
            batch=None,
            frame_groups=plan.groups,
            seed=stable_seed(seed, scene.scene_id, "sonata-smoke"),
        )
    feats = result["point_features"]
    raw_to_token = result["raw_to_token"]
    finite = bool(torch.isfinite(feats).all().item())
    geometry = {
        "raw_point_count": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "final_token_count": int(result["final_token_count"]),
        "group_count": len(plan.groups),
        "raw_to_token_min": int(raw_to_token.min().item()),
        "raw_to_token_max": int(raw_to_token.max().item()),
        "finite_features": finite,
        "encoder_state_sha256_before": before,
        "encoder_state_sha256_after": _state_hash(model.encoder),
        "encoder_unchanged": before == _state_hash(model.encoder),
        "scene_id": scene.scene_id,
        "source_manifest_sha256": scene.manifest_sha256,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    if not finite or not geometry["encoder_unchanged"] or geometry["final_token_count"] <= 0:
        raise RuntimeError(f"Sonata smoke contract failed: {geometry}")
    p["output"].mkdir(parents=True, exist_ok=True)
    (p["output"] / "smoke.json").write_text(
        json.dumps(geometry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(geometry, indent=2, sort_keys=True))


def preflight(cfg: Mapping[str, Any]) -> None:
    import torch
    from delimit3d.training.contracts import load_scene_ids

    p = resolved_paths(cfg)
    scene_ids = load_scene_ids(p["train_split"])
    if not scene_ids:
        raise RuntimeError("Training split is empty")
    output = p["output"]
    output.mkdir(parents=True, exist_ok=True)
    before = _state_hash(_load_runtime(cfg)[0].encoder)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {"name": props.name, "total_memory": int(props.total_memory)}
    else:
        gpu = {"name": "none", "total_memory": 0}
    status = {
        "ready_for_train": True,
        "scene_count": len(scene_ids),
        "first_scene": scene_ids[0],
        "encoder_state_sha256": before,
        "gpu": gpu,
        "expected_updates": int(cfg.get("experiment", {}).get("updates", 256)),
    }
    (output / "preflight.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


def train(cfg: Mapping[str, Any]) -> None:
    p = resolved_paths(cfg)
    p["output"].mkdir(parents=True, exist_ok=True)
    updates = int(cfg.get("experiment", {}).get("updates", 256))
    report = _one_step(cfg, updates=updates, output=p["output"])
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "smoke", "preflight", "train"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve(strict=True))
    if args.mode == "prepare":
        prepare(cfg)
    elif args.mode == "smoke":
        smoke(cfg)
    elif args.mode == "preflight":
        preflight(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
