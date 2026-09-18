#!/usr/bin/env python3
"""Matched two-GPU public Sonata Delimit3D adaptation.

The runner keeps the public Sonata bridge and audited multigranular sampler,
but makes the requested global-update contract explicit:
2 ranks x 6 scenes/rank x 16 proposals/scene = 12 scenes and 192 proposals
per optimizer update.  Contract drift is a hard error.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import yaml

from run_sonata_adaptation import (
    _load_runtime,
    _load_scene,
    _set_seed,
    _state_hash,
    build_provenance,
    load_config,
    resolved_paths,
    sha256_json,
)
from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    NoRetainedFrameGroupsError,
)
from delimit3d.training.contracts import load_scene_ids, stable_seed

RUN_SCHEMA = "delimit3d_sonata_matched_adaptation_run/v1"
CHECKPOINT_SCHEMA = "delimit3d_sonata_matched_adaptation_checkpoint/v1"
WORLD_SIZE_EXPECTED = 2
SCENES_PER_RANK_DEFAULT = 6
PROPOSALS_PER_SCENE_DEFAULT = 16
EXPECTED_SOURCE_CELLS = ("2d/g02", "2d/g05", "2d/g08")


def _validate_static_contract(config: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    matched = config.get("matched", {})
    model = config.get("model", {})
    data = config.get("data", {})
    sampling = config.get("sampling", {})
    if str(experiment.get("name", "")) != "delimit3d_sonata_rgbn6_256_matched_v1":
        raise RuntimeError("unexpected matched Sonata experiment name")
    if int(experiment.get("updates", 0)) != 256:
        raise RuntimeError("matched Sonata requires exactly 256 optimizer updates")
    if str(model.get("family", "")) != "sonata":
        raise RuntimeError("matched Sonata runner requires model.family=sonata")
    if str(data.get("input_features", "")) != "rgbn6":
        raise RuntimeError("matched Sonata requires data.input_features=rgbn6")
    if float(model.get("grid_size", 0.0)) != 0.02:
        raise RuntimeError("matched Sonata requires a 2 cm grid")
    if int(model.get("projection_dim", 0)) != 128:
        raise RuntimeError("matched Sonata requires a 128D projection output")
    if str(sampling.get("mode", "")) != "coverage_multigranular_v2":
        raise RuntimeError("matched Sonata requires coverage_multigranular_v2")
    data_cells = tuple(str(cell) for cell in data.get("source_cells", ()))
    sampling_cells = tuple(str(cell) for cell in sampling.get("source_cells", ()))
    if data_cells != EXPECTED_SOURCE_CELLS or sampling_cells != EXPECTED_SOURCE_CELLS:
        raise RuntimeError("matched Sonata requires source cells g02 + g05 + g08")
    if str(sampling.get("negative_membership_policy", "")) != (
        "require_negative_proposal_membership"
    ):
        raise RuntimeError("matched Sonata negative-membership policy drifted")
    world = int(matched.get("world_size", 0))
    scenes_per_rank = int(matched.get("scenes_per_rank", 0))
    scenes_global = int(matched.get("scenes_per_global_update", 0))
    proposals_scene = int(matched.get("proposals_per_scene", 0))
    proposals_global = int(matched.get("proposals_per_global_update", 0))
    if (
        world != WORLD_SIZE_EXPECTED
        or scenes_per_rank != SCENES_PER_RANK_DEFAULT
        or scenes_global != 12
        or proposals_scene != PROPOSALS_PER_SCENE_DEFAULT
        or proposals_global != 192
        or world * scenes_per_rank != scenes_global
        or scenes_global * proposals_scene != proposals_global
    ):
        raise RuntimeError(
            "matched Sonata contract must be 2 GPUs, 12 scenes, and 192 proposals globally"
        )


def _matched_contract(config: Mapping[str, Any], world: int) -> dict[str, int]:
    matched = config.get("matched", {})
    contract = {
        "world_size": int(world),
        "scenes_per_rank": int(
            matched.get("scenes_per_rank", SCENES_PER_RANK_DEFAULT)
        ),
        "scenes_per_global_update": int(
            matched.get("scenes_per_global_update", 12)
        ),
        "proposals_per_scene": int(
            matched.get("proposals_per_scene", PROPOSALS_PER_SCENE_DEFAULT)
        ),
        "proposals_per_global_update": int(
            matched.get("proposals_per_global_update", 192)
        ),
    }
    if (
        contract["world_size"] != WORLD_SIZE_EXPECTED
        or contract["scenes_per_rank"] * contract["world_size"]
        != contract["scenes_per_global_update"]
        or contract["scenes_per_global_update"]
        * contract["proposals_per_scene"]
        != contract["proposals_per_global_update"]
    ):
        raise RuntimeError(f"invalid matched Sonata contract: {contract}")
    return contract


def _init_dist(config: Mapping[str, Any]):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    expected = int(config.get("matched", {}).get("world_size", WORLD_SIZE_EXPECTED))
    if world != expected:
        raise RuntimeError(f"expected {expected} torchrun ranks, got {world}")
    if not torch.cuda.is_available():
        raise RuntimeError("matched Sonata requires CUDA")
    if not 0 <= local < world:
        raise RuntimeError(f"invalid LOCAL_RANK={local}")
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.get_rank() != rank or dist.get_world_size() != world:
        raise RuntimeError("distributed rank contract drift")
    _matched_contract(config, world)
    return rank, world, local, torch.device("cuda", local)


def _destroy() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _load_matched_runtime(config: Mapping[str, Any]):
    model, device, sampling = _load_runtime(config)
    pps = int(
        config.get("matched", {}).get(
            "proposals_per_scene", PROPOSALS_PER_SCENE_DEFAULT
        )
    )
    if int(sampling.proposals_per_forward) != pps:
        raise RuntimeError(
            "Sonata sampler profile does not provide the requested proposals/scene: "
            f"{sampling.proposals_per_forward} != {pps}"
        )
    return model, device, sampling


def _plan_and_inputs(
    *,
    scene: Any,
    scene_epoch: int,
    update: int,
    seed: int,
    device: torch.device,
    sampling: Any,
    coverage_states: Mapping[str, Any],
    source_cells: list[str],
    config: Mapping[str, Any],
):
    from delimit3d.training.adaptation import build_scene_visit_input
    from delimit3d.training.sonata_adaptation import build_sonata_v2_plan

    visit = build_scene_visit_input(
        scene=scene,
        epoch=int(update - 1),
        global_seed=int(seed),
        input_features="rgbn6",
        augmentation_profile=str(
            config.get("data", {}).get("augmentation_profile", "none")
        ),
    )
    points = torch.from_numpy(visit.points).to(device)
    plan = build_sonata_v2_plan(
        scene=scene,
        epoch=int(scene_epoch),
        seed=int(stable_seed(seed, scene.scene_id, update, "sonata-matched-plan")),
        points=points,
        cells=source_cells,
        sampling=sampling,
        coverage_states=coverage_states,
    )
    return points, plan


def _optimizer_for(model: Any, config: Mapping[str, Any]):
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
        raise RuntimeError("Sonata optimizer parameter groups are unexpectedly empty")
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_ids = {id(parameter) for parameter in head_parameters}
    if encoder_ids & head_ids:
        raise RuntimeError("a parameter appears in both Sonata optimizer groups")
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


def _parameter_hash(module: Any) -> str:
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


def _set_lr(optimizer: Any, config: Mapping[str, Any], update: int) -> float:
    matched = config.get("matched", {})
    total = int(config.get("experiment", {}).get("updates", 256))
    warmup = int(matched.get("warmup_updates", 0))
    peak = float(matched.get("peak_lr_factor", 1.0))
    final = float(matched.get("final_lr_factor", 1.0))
    if warmup > 0 and update <= warmup:
        factor = peak * float(update) / float(warmup)
    elif total <= warmup:
        factor = final
    else:
        progress = min(1.0, max(0.0, float(update - warmup) / float(total - warmup)))
        factor = final + (peak - final) * 0.5 * (
            1.0 + torch.cos(torch.tensor(progress * torch.pi)).item()
        )
    optimizer.param_groups[0]["lr"] = float(config["optimizer"]["backbone_lr"]) * factor
    optimizer.param_groups[1]["lr"] = float(config["optimizer"]["head_lr"]) * factor
    return factor


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
    contract: Mapping[str, int],
    factor: float,
    clip: float,
) -> dict[str, Any]:
    records = [
        record
        for payload in payloads
        for record in payload["scene_records"]
    ]
    valid_records = [
        record for record in records if bool(record.get("optimization_valid", True))
    ]
    if not valid_records:
        raise RuntimeError(f"global update {update} retained zero valid scenes")
    metric_names = (
        "loss",
        "cosine_gap",
        "positive_cosine_mean",
        "negative_cosine_mean",
        "triplet_ranking_accuracy",
        "hardest_negative_ranking_accuracy",
    )
    means = {
        name: sum(float(record.get(name, 0.0)) for record in valid_records)
        / float(len(valid_records))
        for name in metric_names
    }
    return {
        "schema_version": RUN_SCHEMA,
        "kind": "train",
        "update": int(update),
        **{key: int(value) for key, value in contract.items()},
        "scene_count_in_update": len(records),
        "valid_scene_count": len(valid_records),
        "invalid_scene_count": len(records) - len(valid_records),
        "valid_proposal_count": sum(
            int(record.get("realized_proposals", 0)) for record in valid_records
        ),
        "requested_proposal_count": int(contract["proposals_per_global_update"]),
        "retained_group_count": sum(
            int(record.get("group_count", 0)) for record in valid_records
        ),
        **means,
        "temperature": sum(
            float(record.get("temperature", 0.0)) for record in valid_records
        )
        / float(len(valid_records)),
        "grad_norm_before_clip_by_rank": [
            float(payload["grad_norm_before_clip"]) for payload in payloads
        ],
        "grad_norm_before_clip_mean": sum(
            float(payload["grad_norm_before_clip"]) for payload in payloads
        )
        / float(len(payloads)),
        "gradient_clip_norm": float(clip),
        "lr_factor": float(factor),
        "lr_backbone": float(payloads[0]["lr_backbone"]),
        "lr_head": float(payloads[0]["lr_head"]),
        "scene_records": records,
        "rank_scene_ids": {
            str(payload["rank"]): payload["scene_ids"] for payload in payloads
        },
        "requested_rank_scene_ids": {
            str(payload["rank"]): payload["requested_scene_ids"]
            for payload in payloads
        },
        "skipped_rank_scene_ids": {
            str(payload["rank"]): payload["skipped_scene_ids"]
            for payload in payloads
        },
    }


def _zero_model_loss(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            zero = zero + parameter.sum() * 0.0
    return zero


def train(
    config: Mapping[str, Any],
    *,
    updates_override: int | None = None,
    smoke: bool = False,
) -> None:
    _validate_static_contract(config)
    rank, world, local, device = _init_dist(config)
    _set_seed(int(config.get("experiment", {}).get("seed", 20260911)))
    paths = resolved_paths(config)
    paths["output"].mkdir(parents=True, exist_ok=True)
    model, device, sampling = _load_matched_runtime(config)
    contract = _matched_contract(config, world)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local],
        output_device=local,
        broadcast_buffers=True,
        find_unused_parameters=False,
    )
    optimizer = _optimizer_for(model, config)
    initial_full_hash = _same_hash(model, world)
    initial_encoder_hash = _same_hash(model.encoder, world)
    model.train()
    scene_ids = load_scene_ids(paths["train_split"])
    expected_scene_count = int(
        config.get("matched", {}).get("expected_train_scene_count", 2930)
    )
    if len(scene_ids) != expected_scene_count:
        raise RuntimeError(
            f"expected {expected_scene_count} training scenes, got {len(scene_ids)}"
        )
    pool = _rank_scene_ids(scene_ids, rank, world)
    source_cells = [
        str(cell)
        for cell in config.get("sampling", {}).get(
            "source_cells", list(EXPECTED_SOURCE_CELLS)
        )
    ]
    cache: OrderedDict[str, Any] = OrderedDict()
    coverage: dict[str, dict[str, Any]] = {}
    epochs: dict[str, int] = {}
    updates = int(
        updates_override
        if updates_override is not None
        else config.get("experiment", {}).get("updates", 256)
    )
    if updates <= 0 or updates > 256:
        raise RuntimeError(f"invalid matched Sonata update count: {updates}")
    clip = float(config.get("optimizer", {}).get("gradient_clip_norm", 0.1))
    checkpoints = {
        int(value)
        for value in config.get("experiment", {}).get(
            "checkpoint_updates", [64, 128, 256]
        )
    }
    log_path = paths["output"] / "metrics.jsonl"
    if rank == 0 and not smoke and log_path.exists():
        raise RuntimeError(f"refusing to append to existing {log_path}")
    if rank == 0:
        (paths["output"] / "matched_contract.json").write_text(
            json.dumps(
                {
                    "schema_version": RUN_SCHEMA,
                    **contract,
                    "updates": updates,
                    "seed": int(config.get("experiment", {}).get("seed", 20260911)),
                    "source_cells": source_cells,
                    "augmentation_profile": config.get("data", {}).get(
                        "augmentation_profile"
                    ),
                    "feature_hard_mining": config.get("sampling", {}).get(
                        "feature_hard_mining"
                    ),
                    "initial_full_model_hash": initial_full_hash,
                    "initial_encoder_hash": initial_encoder_hash,
                    "invalid_scene_policy": "fixed_schedule_zero_gradient",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    start_time = time.time()
    last: dict[str, Any] | None = None
    seed = int(config.get("experiment", {}).get("seed", 20260911))
    for update in range(1, updates + 1):
        requested = _step_scene_ids(pool, update, contract["scenes_per_rank"])
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
            state = coverage.setdefault(
                scene_id, {cell: None for cell in source_cells}
            )
            context = (
                ddp.no_sync()
                if index < contract["scenes_per_rank"] - 1
                else contextlib.nullcontext()
            )
            try:
                points, plan = _plan_and_inputs(
                    scene=scene,
                    scene_epoch=scene_epoch,
                    update=update,
                    seed=seed,
                    device=device,
                    sampling=sampling,
                    coverage_states=state,
                    source_cells=source_cells,
                    config=config,
                )
            except (NoRetainedFrameGroupsError, ValueError) as error:
                if isinstance(error, NoRetainedFrameGroupsError):
                    reason = "no_retained_frame_groups"
                elif "coverage state" in str(error):
                    reason = "coverage_state_invalid"
                else:
                    raise
                skipped_scene_ids.append(
                    {"scene_id": scene_id, "reason": reason, "detail": str(error)}
                )
                with context:
                    _zero_model_loss(model, device).backward()
                records.append(
                    {
                        "rank": rank,
                        "scene_id": scene_id,
                        "loss": 0.0,
                        "cosine_gap": 0.0,
                        "positive_cosine_mean": 0.0,
                        "negative_cosine_mean": 0.0,
                        "triplet_ranking_accuracy": 0.0,
                        "hardest_negative_ranking_accuracy": 0.0,
                        "temperature": 0.0,
                        "group_count": 0,
                        "feature_dim": 0,
                        "final_token_count": 0,
                        "requested_proposals": contract["proposals_per_scene"],
                        "realized_proposals": 0,
                        "forward_seed": None,
                        "optimization_valid": False,
                    }
                )
                continue

            for cell, item in plan.coverage_by_cell.items():
                state_after = item.get("state_after")
                if state_after is not None:
                    state[str(cell)] = DeterministicCoverageState.from_dict(
                        state_after
                    ).for_epoch(scene_epoch + 1)
            epochs[scene_id] = scene_epoch + 1
            forward_seed = stable_seed(
                seed, scene_id, update, "sonata-matched-forward"
            )
            with context:
                result = ddp(
                    points=points,
                    colors=scene.colors,
                    normals=scene.normals,
                    batch=None,
                    frame_groups=plan.groups,
                    seed=forward_seed,
                )
                loss = result["loss_total"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite Sonata loss at update {update}")
                (loss / float(contract["scenes_per_rank"])).backward()
            records.append(
                {
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
                    "triplet_ranking_accuracy": float(
                        result.get("triplet_ranking_accuracy", 0.0)
                    ),
                    "hardest_negative_ranking_accuracy": float(
                        result.get("hardest_negative_ranking_accuracy", 0.0)
                    ),
                    "temperature": float(result.get("temperature", 0.0)),
                    "group_count": int(result.get("group_count", len(plan.groups))),
                    "feature_dim": int(result.get("feature_dim", 0)),
                    "final_token_count": int(result.get("final_token_count", 0)),
                    "requested_proposals": contract["proposals_per_scene"],
                    "realized_proposals": int(
                        sum(
                            len(group.selected_proposal_indices)
                            for group in plan.groups
                        )
                    ),
                    "forward_seed": int(forward_seed),
                    "optimization_valid": True,
                }
            )

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
            raise RuntimeError(f"global update {update} retained zero valid scenes")
        valid_rescale = float(contract["scenes_per_global_update"]) / float(global_valid)
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
            row = _aggregate(
                [item for item in gathered if item is not None],
                update,
                contract,
                factor,
                clip,
            )
            last = row
            if not smoke:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                if update in checkpoints or update == updates:
                    torch.save(
                        {
                            "schema_version": CHECKPOINT_SCHEMA,
                            "update": update,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "encoder_state_sha256": _parameter_hash(model.encoder),
                            "initial_model_state_sha256": initial_full_hash,
                            "initial_encoder_state_sha256": initial_encoder_hash,
                            "config_sha256": sha256_json(config),
                            "metrics": row,
                            "matched_contract": dict(contract),
                        },
                        paths["output"] / f"checkpoint_u{update:06d}.pt",
                    )
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
            **contract,
            "elapsed_seconds": time.time() - start_time,
            "last": last,
            "output": str(paths["output"]),
        }
        name = "matched_smoke.json" if smoke else "train_report.json"
        (paths["output"] / name).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    dist.barrier()
    _destroy()


def prepare(config: Mapping[str, Any]) -> None:
    _validate_static_contract(config)
    paths = resolved_paths(config)
    paths["output"].mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config)
    (paths["output"] / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paths["output"] / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {key: value for key, value in config.items() if not key.startswith("_")},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    paths_to_check = (
        paths["sonata_root"],
        paths["checkpoint"],
        paths["source_root"],
        paths["train_split"],
    )
    missing = [str(path) for path in paths_to_check if not path.exists()]
    checkpoint_match = provenance["model"]["checkpoint_sha256_match"]
    scene_count = (
        len(load_scene_ids(paths["train_split"]))
        if paths["train_split"].is_file()
        else None
    )
    expected_scene_count = int(
        config.get("matched", {}).get("expected_train_scene_count", 2930)
    )
    if checkpoint_match is False:
        raise RuntimeError("Sonata checkpoint SHA-256 does not match the resolved config")
    if missing:
        raise RuntimeError(f"matched Sonata preparation missing paths: {missing}")
    if scene_count != expected_scene_count:
        raise RuntimeError(
            f"expected {expected_scene_count} training scenes, got {scene_count}"
        )
    contract = _matched_contract(config, WORLD_SIZE_EXPECTED)
    (paths["output"] / "matched_prepare_contract.json").write_text(
        json.dumps(
            {
                "schema_version": RUN_SCHEMA,
                **contract,
                "updates": int(config["experiment"]["updates"]),
                "train_scene_count": scene_count,
                "checkpoint_sha256": provenance["model"]["checkpoint_sha256_actual"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status = {
        "ready_for_gpu_preflight": True,
        "missing_now": [],
        "source_root_exists": True,
        "train_split_exists": True,
        "train_scene_count": scene_count,
        "checkpoint_sha256": provenance["model"]["checkpoint_sha256_actual"],
        "provenance": str(paths["output"] / "provenance.json"),
    }
    (paths["output"] / "prepare_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, sort_keys=True))


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
