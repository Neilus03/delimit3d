#!/usr/bin/env python3
"""Create a no-clobber checkpoint with recalibrated LitePT BN buffers.

Only BatchNorm ``running_mean``, ``running_var``, and
``num_batches_tracked`` entries are replaced.  All parameters, optimizer state,
projection heads, and source checkpoint bytes remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm


_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _STUDENT_ROOT.parent
for _path in (_SCRIPT_DIR, _STUDENT_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from delimit3d.evaluation.features import (  # noqa: E402
    _map_to_wrapper_key,
    _payload_state,
    sha256_file,
)
from delimit3d.training.adaptation import (  # noqa: E402
    build_scene_visit_input,
    load_scene_source,
)
from delimit3d.cli.recalibration_contract import (  # noqa: E402
    build_bn_recalibration_scene_selection,
    load_and_validate_bn_recalibration_normal_audit,
)
from delimit3d.models.litept_wrapper import LitePTBackbone  # noqa: E402


MANIFEST_SCHEMA = "litept_rgbn6_bn_recalibration/v2"
COORDINATE_NORMALIZATION_POLICY = "structured3d_local_y_up_to_scannet_z_up_v1"
NORMALIZATION_POLICIES = ("batchnorm", "sync_batchnorm")


def _state_container(payload: Any, path: Path) -> MutableMapping[str, torch.Tensor]:
    if not isinstance(payload, MutableMapping):
        raise TypeError(f"{path}: checkpoint payload is not mutable mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, MutableMapping):
            return value
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise KeyError(f"{path}: no state_dict, model_state_dict, or model")


def bn_buffer_keys(model: torch.nn.Module) -> set[str]:
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, _BatchNorm):
            continue
        prefix = f"{module_name}." if module_name else ""
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and name in {
                "running_mean", "running_var", "num_batches_tracked",
            }:
                keys.add(prefix + name)
    return keys


def _tensor_digest(records: Mapping[str, torch.Tensor]) -> str:
    """Content hash a named tensor mapping independent of torch serialization."""

    digest = hashlib.sha256()
    for key in sorted(records):
        tensor = records[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_hashes(
    state: Mapping[str, torch.Tensor], buffer_keys: set[str]
) -> dict[str, str]:
    return {
        "bn_buffers": _tensor_digest(
            {key: value for key, value in state.items() if key in buffer_keys}
        ),
        "non_bn_state": _tensor_digest(
            {key: value for key, value in state.items() if key not in buffer_keys}
        ),
        "complete_model_state": _tensor_digest(state),
    }


def apply_recalibrated_buffers(
    *,
    source_state: MutableMapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    expected_model_keys: set[str],
    buffer_keys: set[str],
) -> list[str]:
    """Patch only mapped BN buffers and return changed source-state keys."""

    changed: list[str] = []
    covered: set[str] = set()
    for source_key, original in list(source_state.items()):
        if not isinstance(original, torch.Tensor):
            continue
        mapped = _map_to_wrapper_key(str(source_key), expected_keys=expected_model_keys)
        if mapped is None or mapped not in buffer_keys:
            continue
        replacement = model_state[mapped].detach().to(device=original.device, dtype=original.dtype)
        source_state[source_key] = replacement.clone()
        covered.add(mapped)
        if not torch.equal(original, replacement):
            changed.append(str(source_key))
    missing = sorted(buffer_keys - covered)
    if missing:
        raise RuntimeError(f"Checkpoint does not map every LitePT BN buffer: {missing}")
    return sorted(changed)


def _load_backbone(checkpoint: Path, litept_root: Path, device: torch.device) -> tuple[LitePTBackbone, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LitePTBackbone(
        litept_root=str(litept_root),
        in_channels=6,
        grid_size=0.02,
        litept_variant="litept_s_star",
        multi_scale=False,
        feature_pyramid_mode="mask3d_fpn",
        cache_training_voxelization=False,
        voxel_reduce="representative",
        representative_sampling="first",
    )
    source = _payload_state(payload, checkpoint)
    expected = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for source_key, tensor in source.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        target = _map_to_wrapper_key(str(source_key), expected_keys=set(expected))
        if target is not None:
            mapped[target] = tensor.detach().cpu()
    missing = sorted(set(expected) - set(mapped))
    mismatched = sorted(
        key for key in set(expected) & set(mapped)
        if tuple(expected[key].shape) != tuple(mapped[key].shape)
    )
    if missing or mismatched:
        raise RuntimeError(
            f"Strict RGBN6 backbone load failed: missing={missing} shape={mismatched}"
        )
    model.load_state_dict(mapped, strict=True)
    model.to(device)
    return model, payload


def _configure_bn_recalibration(model: torch.nn.Module) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, _BatchNorm):
            continue
        module.reset_running_stats()
        module.momentum = None
        module.train()
        records.append({
            "module": name,
            "features": int(module.num_features),
            "eps": float(module.eps),
            "momentum": None,
        })
    if not records:
        raise RuntimeError("LitePT backbone exposes no BatchNorm modules")
    return {"count": len(records), "modules": records}


def _runtime_context(
    normalization_policy: str, requested_world_size: int, device_arg: str
) -> tuple[int, int, torch.device]:
    """Initialize exactly the process topology required by the BN policy."""

    if normalization_policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"Unknown normalization policy: {normalization_policy}")
    if normalization_policy == "batchnorm":
        if requested_world_size != 1:
            raise ValueError("batchnorm recalibration is scene-balanced and requires world_size=1")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("batchnorm recalibration must not be launched with torchrun")
        rank, world_size = 0, 1
    else:
        if requested_world_size != 2:
            raise ValueError("sync_batchnorm recalibration requires world_size=2")
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            raise ValueError("sync_batchnorm recalibration must run under torchrun with two ranks")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != requested_world_size:
            raise ValueError(f"Distributed world-size drift: {world_size}")
    if device_arg == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA BN recalibration requested without CUDA")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_arg)
    if normalization_policy == "sync_batchnorm" and device.type != "cuda":
        raise ValueError("PyTorch SyncBatchNorm recalibration requires CUDA")
    return rank, world_size, device


def _assert_sync_bn_buffers_identical(
    state: Mapping[str, torch.Tensor], buffer_keys: set[str]
) -> str:
    digest = _state_hashes(state, buffer_keys)["bn_buffers"]
    if dist.is_initialized():
        gathered: list[str | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, digest)
        if len(set(gathered)) != 1:
            raise RuntimeError(f"SyncBatchNorm buffers differ across ranks: {gathered}")
    return digest


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite recalibrated checkpoint: {path}")
    fd, temporary_name = __import__("tempfile").mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite BN manifest: {path}")
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--normal-audit", type=Path, required=True)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument(
        "--normalization-policy", choices=NORMALIZATION_POLICIES, required=True
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--augmentation-profile",
        default="mask3d_scannet_rgbn6_sphere50k_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, world_size, device = _runtime_context(
        str(args.normalization_policy), int(args.world_size), str(args.device)
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    train_split = args.train_split.resolve(strict=True)
    litept_root = args.litept_root.resolve(strict=True)
    output_checkpoint = args.output_checkpoint.resolve()
    output_manifest = args.output_manifest.resolve()
    for output in (output_checkpoint, output_manifest):
        if rank == 0 and output.exists():
            raise FileExistsError(f"Refusing existing BN output: {output}")
    scene_selection = build_bn_recalibration_scene_selection(
        train_split,
        scene_count=int(args.scenes),
        seed=int(args.seed),
    )
    selected = list(scene_selection["selected_scene_ids"])
    _, normal_audit = load_and_validate_bn_recalibration_normal_audit(
        args.normal_audit,
        source_root=source_root,
        train_split=train_split,
        scene_count=int(args.scenes),
        seed=int(args.seed),
    )
    model, payload = _load_backbone(checkpoint, litept_root, device)
    if args.normalization_policy == "sync_batchnorm":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    bn_contract = _configure_bn_recalibration(model)

    with torch.no_grad():
        local_selected = (
            selected
            if args.normalization_policy == "batchnorm"
            else selected[rank::world_size]
        )
        if args.normalization_policy == "sync_batchnorm" and len(selected) % world_size:
            raise ValueError("SyncBatchNorm scene count must be divisible by world size")
        for local_index, scene_id in enumerate(local_selected):
            global_index = (
                local_index
                if args.normalization_policy == "batchnorm"
                else local_index * world_size + rank
            )
            scene = load_scene_source(
                source_root / scene_id / "source_manifest.json",
                input_features="rgbn6",
                verify_hashes=True,
                coordinate_normalization_policy=COORDINATE_NORMALIZATION_POLICY,
            )
            visit = build_scene_visit_input(
                scene=scene,
                epoch=global_index,
                global_seed=int(args.seed),
                input_features="rgbn6",
                augmentation_profile=str(args.augmentation_profile),
            )
            coordinates = torch.from_numpy(np.asarray(visit.points, dtype=np.float32)).to(device)
            features = torch.from_numpy(np.asarray(visit.features, dtype=np.float32)).to(device)
            model(coordinates, features)
            if (local_index + 1) % 25 == 0 or local_index + 1 == len(local_selected):
                print(
                    f"BN recalibration rank={rank} {local_index + 1}/"
                    f"{len(local_selected)} {scene_id}", flush=True,
                )

    after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    buffers = bn_buffer_keys(model)
    before_hashes = _state_hashes(before, buffers)
    after_hashes = _state_hashes(after, buffers)
    _assert_sync_bn_buffers_identical(after, buffers)
    parameter_drift = [
        key for key in before
        if key not in buffers and not torch.equal(before[key], after[key])
    ]
    if parameter_drift:
        raise RuntimeError(f"Non-BN state changed during recalibration: {parameter_drift}")
    state = _state_container(payload, checkpoint)
    changed_source_keys = apply_recalibrated_buffers(
        source_state=state,
        model_state=after,
        expected_model_keys=set(after),
        buffer_keys=buffers,
    )
    if not changed_source_keys:
        raise RuntimeError("BN recalibration did not change any checkpoint buffer")
    if before_hashes["non_bn_state"] != after_hashes["non_bn_state"]:
        raise RuntimeError("Non-BN model-state hash changed during recalibration")
    record = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_root": str(source_root),
        "train_split": str(train_split),
        "train_split_sha256": sha256_file(train_split),
        "scene_selection": scene_selection,
        "normal_audit": normal_audit,
        "input_contract": {
            "features": "rgbn6",
            "channels": 6,
            "channel_order": ["R", "G", "B", "nx", "ny", "nz"],
            "voxel_reduce": "representative",
            "representative_sampling": "first",
            "augmentation_profile": str(args.augmentation_profile),
            "coordinate_normalization_policy": COORDINATE_NORMALIZATION_POLICY,
        },
        "normalization_policy": str(args.normalization_policy),
        "world_size": world_size,
        "policy": (
            "reset_running_stats_then_scene_balanced_cumulative_average_bn_only"
            if args.normalization_policy == "batchnorm"
            else "reset_running_stats_then_two_rank_global_sync_batchnorm"
        ),
        "selected_scene_ids": selected,
        "scene_count": len(selected),
        "seed": int(args.seed),
        "bn_modules": bn_contract,
        "bn_buffer_keys": sorted(buffers),
        "model_state_hashes_before": before_hashes,
        "model_state_hashes_after": after_hashes,
        "changed_source_state_keys": changed_source_keys,
        "non_bn_state_unchanged": True,
        "claim_boundary": "running-stat calibration only; no representation or downstream claim",
        "passed": True,
    }
    if rank == 0:
        if isinstance(payload, dict):
            payload["bn_recalibration"] = dict(record)
        _atomic_torch_save(output_checkpoint, payload)
        record["output_checkpoint"] = str(output_checkpoint)
        record["output_checkpoint_sha256"] = sha256_file(output_checkpoint)
        _atomic_json(output_manifest, record)
        print(json.dumps({
            "output_checkpoint": str(output_checkpoint),
            "output_checkpoint_sha256": record["output_checkpoint_sha256"],
            "output_manifest": str(output_manifest),
            "passed": True,
        }, sort_keys=True))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
