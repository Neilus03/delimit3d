#!/usr/bin/env python3
"""Matched two-GPU public PTv3 Delimit3D adaptation."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import yaml

from run_ptv3_adaptation import (
    _load_runtime,
    _load_scene,
    _optimizer_for,
    _set_seed,
    _state_hash,
    load_config,
    resolved_paths,
    sha256_json,
)
from delimit3d.training.contracts import load_scene_ids, stable_seed
from delimit3d.data.contrastive_sampler_v2 import NoRetainedFrameGroupsError

RUN_SCHEMA = "delimit3d_ptv3_matched_adaptation_run/v1"
CHECKPOINT_SCHEMA = "delimit3d_ptv3_matched_adaptation_checkpoint/v1"
WORLD_SIZE_EXPECTED = 2
SCENES_PER_RANK_DEFAULT = 6
PROPOSALS_PER_SCENE_DEFAULT = 16


def _init_dist(config: Mapping[str, Any]):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    expected = int(config.get("matched", {}).get("world_size", WORLD_SIZE_EXPECTED))
    if world != expected:
        raise RuntimeError(f"expected {expected} torchrun ranks, got {world}")
    if not torch.cuda.is_available():
        raise RuntimeError("matched PTv3 requires CUDA")
    if not 0 <= local < world:
        raise RuntimeError(f"invalid LOCAL_RANK={local}")
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.get_rank() != rank or dist.get_world_size() != world:
        raise RuntimeError("distributed rank contract drift")
    return rank, world, local, torch.device("cuda", local)


def _destroy() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _sampling(config: Mapping[str, Any]):
    _, _, sampling = _load_runtime(config)
    matched = config.get("matched", {})
    proposals = int(matched.get("proposals_per_scene", PROPOSALS_PER_SCENE_DEFAULT))
    if int(sampling.proposals_per_forward) != proposals:
        sampling = replace(sampling, proposals_per_forward=proposals)
    policy = str(config.get("sampling", {}).get("feature_hard_mining", "disabled")).lower()
    if policy in {"disabled", "off", "false", "0"}:
        sampling = replace(sampling, feature_hard_negatives=0)
    return sampling


def _visit_and_plan(
    *,
    scene: Any,
    scene_epoch: int,
    update: int,
    seed: int,
    device: torch.device,
    sampling: Any,
    coverage: Mapping[str, Any],
    cells: list[str],
    config: Mapping[str, Any],
):
    from delimit3d.training.adaptation import build_scene_visit_input
    from delimit3d.training.ptv3_adaptation import build_ptv3_v2_plan

    visit = build_scene_visit_input(
        scene=scene,
        epoch=int(scene_epoch),
        global_seed=int(seed),
        input_features="rgbn6",
        augmentation_profile=str(
            config.get("data", {}).get("augmentation_profile", "none")
        ),
    )
    points = torch.from_numpy(visit.points).to(device)
    features = torch.from_numpy(visit.features).to(device)
    plan = build_ptv3_v2_plan(
        scene=scene,
        epoch=int(scene_epoch),
        seed=int(stable_seed(seed, scene.scene_id, update, "matched-plan")),
        points=points,
        cells=cells,
        sampling=sampling,
        coverage_states=coverage,
    )
    return points, features, plan


def _lr_factor(config: Mapping[str, Any], update: int) -> float:
    matched = config.get("matched", {})
    total = int(config.get("experiment", {}).get("updates", 256))
    warmup = int(matched.get("warmup_updates", 245))
    peak = float(matched.get("peak_lr_factor", 1.0))
    final = float(matched.get("final_lr_factor", 0.01))
    if warmup > 0 and update <= warmup:
        return peak * float(update) / float(warmup)
    if total <= warmup:
        return final
    progress = min(1.0, max(0.0, float(update - warmup) / float(total - warmup)))
    return final + (peak - final) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def _set_lr(optimizer: Any, config: Mapping[str, Any], update: int) -> float:
    factor = _lr_factor(config, update)
    optimizer.param_groups[0]["lr"] = float(config["optimizer"]["backbone_lr"]) * factor
    optimizer.param_groups[1]["lr"] = float(config["optimizer"]["head_lr"]) * factor
    return factor


def _parameter_hash(module: Any) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _same_hash(module: Any, world: int, *, parameters_only: bool = False) -> str:
    value = _parameter_hash(module) if parameters_only else _state_hash(module)
    values: list[str | None] = [None for _ in range(world)]
    dist.all_gather_object(values, value)
    if len(set(str(item) for item in values)) != 1:
        raise RuntimeError(f"distributed state differs across ranks: {values}")
    return value


def _rank_scene_ids(scene_ids: list[str], rank: int, world: int) -> list[str]:
    result = scene_ids[rank::world]
    if not result:
        raise RuntimeError(f"rank {rank} has no scenes")
    return result


def _step_scene_ids(pool: list[str], update: int, count: int) -> list[str]:
    start = (int(update) - 1) * int(count)
    return [pool[(start + i) % len(pool)] for i in range(count)]


def _aggregate(
    payloads: list[dict[str, Any]],
    update: int,
    world: int,
    spr: int,
    pps: int,
    factor: float,
    clip: float,
) -> dict[str, Any]:
    records = [
        record
        for payload in payloads
        for record in payload["scene_records"]
    ]
    valid_records = [
        record for record in records
        if bool(record.get("optimization_valid", True))
    ]
    if not valid_records:
        raise RuntimeError(f"global update {update} retained zero valid scenes")
    names = (
        "loss",
        "cosine_gap",
        "positive_cosine_mean",
        "negative_cosine_mean",
        "hardest_negative_ranking_accuracy",
    )
    out = {
        name: sum(float(record[name]) for record in valid_records)
        / float(len(valid_records))
        for name in names
    }
    return {
        "schema_version": RUN_SCHEMA,
        "kind": "train",
        "update": int(update),
        "world_size": int(world),
        "scenes_per_rank": int(spr),
        "scenes_per_global_update": int(world * spr),
        "proposals_per_scene": int(pps),
        "proposals_per_global_update": int(world * spr * pps),
        "scene_count_in_update": len(records),
        "valid_scene_count": len(valid_records),
        "invalid_scene_count": len(records) - len(valid_records),
        "valid_proposal_count": sum(
            int(record.get("realized_proposals", 0))
            for record in valid_records
        ),
        "requested_proposal_count": int(world * spr * pps),
        **out,
        "temperature": sum(
            float(record["temperature"]) for record in valid_records
        ) / float(len(valid_records)),
        "grad_norm_before_clip_by_rank": [
            float(payload["grad_norm_before_clip"]) for payload in payloads
        ],
        "grad_norm_before_clip_mean": sum(
            float(payload["grad_norm_before_clip"]) for payload in payloads
        ) / float(world),
        "gradient_clip_norm": float(clip),
        "lr_factor": float(factor),
        "lr_backbone": float(payloads[0]["lr_backbone"]),
        "lr_head": float(payloads[0]["lr_head"]),
        "scene_records": records,
        "rank_scene_ids": {
            str(payload["rank"]): payload["scene_ids"]
            for payload in payloads
        },
        "requested_rank_scene_ids": {
            str(payload["rank"]): payload["requested_scene_ids"]
            for payload in payloads
        },
        "skipped_rank_scene_ids": {
            str(payload["rank"]): payload["skipped_scene_ids"]
            for payload in payloads
        },
        "resampled_scene_count": 0,
    }


def _zero_model_loss(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Build a graph-connected zero so invalid visits stay synchronized."""
    zero = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            zero = zero + parameter.sum() * 0.0
    return zero


def train(config: Mapping[str, Any], *, updates_override: int | None = None, smoke: bool = False) -> None:
    rank, world, local, device = _init_dist(config)
    seed = int(config.get("experiment", {}).get("seed", 20260911))
    _set_seed(seed)
    paths = resolved_paths(config)
    paths["output"].mkdir(parents=True, exist_ok=True)
    model, _, _ = _load_runtime(config)
    sampling = _sampling(config)
    matched = config.get("matched", {})
    spr = int(matched.get("scenes_per_rank", SCENES_PER_RANK_DEFAULT))
    pps = int(matched.get("proposals_per_scene", PROPOSALS_PER_SCENE_DEFAULT))
    spg = int(matched.get("scenes_per_global_update", world * spr))
    ppg = int(matched.get("proposals_per_global_update", world * spr * pps))
    if world != WORLD_SIZE_EXPECTED or spr * world != spg or spg * pps != ppg:
        raise RuntimeError("matched world/scene/proposal contract drift")
    if int(sampling.proposals_per_forward) != pps:
        raise RuntimeError("sampler proposals_per_forward does not match contract")
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], output_device=local,
        broadcast_buffers=True, find_unused_parameters=False
    )
    optimizer = _optimizer_for(model, config)
    initial_full_hash = _same_hash(model, world)
    initial_encoder_hash = _same_hash(model.encoder, world)
    model.train()
    scene_ids = load_scene_ids(paths["train_split"])
    pool = _rank_scene_ids(scene_ids, rank, world)
    cells = [str(cell) for cell in config.get("sampling", {}).get("source_cells", ["2d/g02", "2d/g05", "2d/g08"])]
    cache: OrderedDict[str, Any] = OrderedDict()
    coverage: dict[str, dict[str, Any]] = {}
    epochs: dict[str, int] = {}
    updates = int(updates_override if updates_override is not None else config.get("experiment", {}).get("updates", 256))
    clip = float(config.get("optimizer", {}).get("gradient_clip_norm", 5.0))
    checkpoints = {int(value) for value in config.get("experiment", {}).get("checkpoint_updates", [64, 128, 256])}
    log_path = paths["output"] / "metrics.jsonl"
    if rank == 0 and log_path.exists() and not smoke:
        raise RuntimeError(f"refusing to append to {log_path}")
    if rank == 0:
        (paths["output"] / "matched_contract.json").write_text(
            json.dumps({
                "schema_version": RUN_SCHEMA,
                "initial_full_model_hash": initial_full_hash,
                "initial_encoder_hash": initial_encoder_hash,
                "world_size": world,
                "scenes_per_rank": spr,
                "scenes_per_global_update": spg,
                "proposals_per_scene": pps,
                "proposals_per_global_update": ppg,
                "updates": updates,
                "seed": seed,
                "augmentation_profile": config.get("data", {}).get("augmentation_profile"),
                "feature_hard_mining": config.get("sampling", {}).get("feature_hard_mining"),
                "source_cells": cells,
                "invalid_scene_policy": "fixed_schedule_zero_gradient",
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    start_time = time.time()
    last: dict[str, Any] | None = None
    for update in range(1, updates + 1):
        requested = _step_scene_ids(pool, update, spr)
        selected = list(requested)
        skipped_scene_ids: list[dict[str, Any]] = []
        optimizer.zero_grad(set_to_none=True)
        records: list[dict[str, Any]] = []
        for index, scene_id in enumerate(selected):
            if scene_id not in cache:
                cache[scene_id] = _load_scene(scene_id, config)
                while len(cache) > 24:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(scene_id)
            scene = cache[scene_id]
            scene_epoch = int(epochs.get(scene_id, 0))
            state = coverage.setdefault(scene_id, {cell: None for cell in cells})
            context = (
                ddp.no_sync()
                if index < spr - 1
                else contextlib.nullcontext()
            )
            try:
                points, features, plan = _visit_and_plan(
                    scene=scene,
                    scene_epoch=scene_epoch,
                    update=update,
                    seed=seed,
                    device=device,
                    sampling=sampling,
                    coverage=state,
                    cells=cells,
                    config=config,
                )
            except (NoRetainedFrameGroupsError, ValueError) as error:
                message = str(error)
                if isinstance(error, NoRetainedFrameGroupsError):
                    reason = "no_retained_frame_groups"
                elif "coverage state" in message:
                    reason = "coverage_state_invalid"
                else:
                    raise
                skipped_scene_ids.append({
                    "scene_id": scene_id,
                    "reason": reason,
                    "detail": message,
                })
                with context:
                    zero = _zero_model_loss(model, device)
                    zero.backward()
                records.append({
                    "rank": rank,
                    "scene_id": scene_id,
                    "loss": 0.0,
                    "cosine_gap": 0.0,
                    "positive_cosine_mean": 0.0,
                    "negative_cosine_mean": 0.0,
                    "hardest_negative_ranking_accuracy": 0.0,
                    "temperature": 0.0,
                    "group_count": 0,
                    "feature_dim": 0,
                    "final_token_count": 0,
                    "requested_proposals": pps,
                    "realized_proposals": 0,
                    "forward_seed": None,
                    "optimization_valid": False,
                })
                continue

            for cell, item in plan.coverage_by_cell.items():
                state_after = item.get("state_after")
                if state_after is not None:
                    from delimit3d.data.contrastive_sampler_v2 import (
                        DeterministicCoverageState,
                    )
                    state[str(cell)] = (
                        DeterministicCoverageState.from_dict(state_after)
                        .for_epoch(scene_epoch + 1)
                    )
            epochs[scene_id] = scene_epoch + 1
            forward_seed = stable_seed(
                seed, scene_id, update, "matched-forward"
            )
            with context:
                result = ddp(
                    points=points,
                    features=features,
                    batch=None,
                    frame_groups=plan.groups,
                    seed=forward_seed,
                )
                loss = result["loss_total"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at global update {update}"
                    )
                (loss / float(spr)).backward()
            records.append({
                "rank": rank,
                "scene_id": scene_id,
                "loss": float(loss.detach()),
                "cosine_gap": float(result.get("cosine_gap", 0.0)),
                "positive_cosine_mean": float(
                    result.get("positive_cosine_mean", 0.0)
                ),
                "negative_cosine_mean": float(
                    result.get("negative_cosine_mean", 0.0)
                ),
                "hardest_negative_ranking_accuracy": float(
                    result.get("hardest_negative_ranking_accuracy", 0.0)
                ),
                "temperature": float(result.get("temperature", 0.0)),
                "group_count": int(
                    result.get("group_count", len(plan.groups))
                ),
                "feature_dim": int(result.get("feature_dim", 0)),
                "final_token_count": int(
                    result.get("final_token_count", 0)
                ),
                "requested_proposals": pps,
                "realized_proposals": int(
                    sum(
                        len(group.selected_proposal_indices)
                        for group in plan.groups
                    )
                ),
                "forward_seed": int(forward_seed),
                "optimization_valid": True,
            })

        local_valid = torch.tensor(
            sum(
                1
                for record in records
                if bool(record.get("optimization_valid", False))
            ),
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(local_valid, op=dist.ReduceOp.SUM)
        global_valid = int(local_valid.item())
        if global_valid <= 0:
            raise RuntimeError(
                f"global update {update} retained zero valid scenes"
            )
        valid_rescale = float(world * spr) / float(global_valid)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(valid_rescale)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        factor = _set_lr(optimizer, config, update)
        optimizer.step()
        payload = {
            "rank": rank,
            "scene_ids": selected,
            "requested_scene_ids": requested,
            "skipped_scene_ids": skipped_scene_ids,
            "scene_records": records,
            "grad_norm_before_clip": grad_norm,
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(world)]
        dist.all_gather_object(gathered, payload)
        if any(item is None for item in gathered):
            raise RuntimeError("empty distributed payload")
        if rank == 0:
            row = _aggregate([item for item in gathered if item is not None], update, world, spr, pps, factor, clip)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            last = row
            if update in checkpoints or update == updates:
                torch.save({
                    "schema_version": CHECKPOINT_SCHEMA,
                    "update": update,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "encoder_state_sha256": _parameter_hash(model.encoder),
                    "initial_model_state_sha256": initial_full_hash,
                    "initial_encoder_state_sha256": initial_encoder_hash,
                    "config_sha256": sha256_json(config),
                    "metrics": row,
                    "matched_contract": {"world_size": world, "scenes_per_rank": spr, "scenes_per_global_update": spg, "proposals_per_scene": pps, "proposals_per_global_update": ppg},
                }, paths["output"] / f"checkpoint_u{update:06d}.pt")
            if update == 1 or update % 8 == 0 or update == updates:
                print(json.dumps(row, sort_keys=True), flush=True)
        dist.barrier()
    final_full_hash = _same_hash(model, world, parameters_only=True)
    final_encoder_hash = _same_hash(model.encoder, world, parameters_only=True)
    if rank == 0:
        report = {
            "schema_version": RUN_SCHEMA,
            "status": "smoke_completed" if smoke else "completed",
            "updates": updates,
            "initial_model_state_sha256": initial_full_hash,
            "final_model_state_sha256": final_full_hash,
            "initial_encoder_state_sha256": initial_encoder_hash,
            "final_encoder_state_sha256": final_encoder_hash,
            "encoder_changed": initial_encoder_hash != final_encoder_hash,
            "world_size": world,
            "scenes_per_rank": spr,
            "scenes_per_global_update": spg,
            "proposals_per_scene": pps,
            "proposals_per_global_update": ppg,
            "elapsed_seconds": time.time() - start_time,
            "last": last,
            "output": str(paths["output"]),
        }
        name = "matched_smoke.json" if smoke else "train_report.json"
        (paths["output"] / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    dist.barrier()
    _destroy()


def prepare(config: Mapping[str, Any]) -> None:
    from run_ptv3_adaptation import prepare as base_prepare
    base_prepare(config)
    paths = resolved_paths(config)
    matched = config.get("matched", {})
    contract = {
        "world_size": int(matched.get("world_size", WORLD_SIZE_EXPECTED)),
        "scenes_per_rank": int(matched.get("scenes_per_rank", SCENES_PER_RANK_DEFAULT)),
        "scenes_per_global_update": int(matched.get("scenes_per_global_update", 12)),
        "proposals_per_scene": int(matched.get("proposals_per_scene", PROPOSALS_PER_SCENE_DEFAULT)),
        "proposals_per_global_update": int(matched.get("proposals_per_global_update", 192)),
        "updates": int(config.get("experiment", {}).get("updates", 256)),
    }
    if contract["world_size"] != 2 or contract["scenes_per_global_update"] != 12 or contract["proposals_per_global_update"] != 192:
        raise RuntimeError(f"invalid matched contract: {contract}")
    (paths["output"] / "matched_prepare_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(contract, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "smoke", "train"), required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve(strict=True))
    try:
        if args.mode == "prepare":
            prepare(config)
        elif args.mode == "smoke":
            train(config, updates_override=1, smoke=True)
        else:
            train(config)
    finally:
        if dist.is_available() and dist.is_initialized():
            _destroy()


if __name__ == "__main__":
    main()

