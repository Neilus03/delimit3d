#!/usr/bin/env python3
"""Paired fixed-pack evaluation of public and adapted Sonata encoders.

The public checkpoint is encoder-only.  To keep the comparison causal, this
evaluator measures both checkpoints in the shared native final-token feature
space with one fixed, non-learnable temperature.  It does not use either
checkpoint's disposable projection head.  The serialized fixed-comparable
pack supplies the same scene order, pseudomask proposal occurrences, base
negative candidates, and seed for both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from delimit3d.losses.litept_hierarchy_supervision import (
    build_token_pure_dec0_contrastive_batch,
)
from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion
from delimit3d.training.adaptation import (
    FIXED_EVAL_PACK_SCHEMA,
    SceneSource,
    _fixed_scene_batch_views,
    load_scene_source,
    sampling_config_for_profile,
    validate_fixed_holdout_eval_pack,
)
from delimit3d.training.contracts import stable_seed
from delimit3d.training.sonata_adaptation import SonataContrastiveModel


REPORT_SCHEMA = "delimit3d_sonata_fixed_pair_evaluation/v1"
CHECKPOINT_SCHEMA = "delimit3d_sonata_adaptation_checkpoint/v1"
FIXED_TEMPERATURE = 0.07
MAX_POSITIVE_PAIRS_PER_PROPOSAL = 64
METRIC_NAMES = (
    "loss",
    "cosine_gap",
    "triplet_ranking_accuracy",
    "hardest_negative_ranking_accuracy",
    "paired_retrieval_ap",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mean_positive_rank",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_digest(digest: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())


def batch_digest(batch: Any) -> str:
    digest = hashlib.sha256()
    for name in (
        "positive_pairs",
        "negative_indices",
        "proposal_labels",
        "feature_candidate_indices",
        "proposal_member_offsets",
        "proposal_member_indices",
        "pair_proposal_indices",
        "eligible_indices",
        "relation_proposal_offsets",
        "relation_proposal_member_indices",
    ):
        value = getattr(batch, name, None)
        if value is None:
            digest.update(f"{name}:None".encode())
        else:
            tensor_digest(digest, name, value)
    for name in (
        "num_feature_hard_negatives",
        "num_uniform_negatives",
        "num_spatial_hard_negatives",
        "spatial_candidate_pool",
        "feature_candidate_pool",
        "multiscale_seed",
        "algorithm_version",
        "require_negative_proposal_membership",
    ):
        digest.update(f"{name}:{getattr(batch, name, None)!r}".encode())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    raw["_config_path"] = str(path.resolve())
    return raw


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_fixed_scenes(
    *,
    pack_dir: Path,
    source_root: Path,
    coordinate_policy: str,
) -> tuple[list[SceneSource], dict[str, Any]]:
    manifest_path = (pack_dir / "manifest.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FIXED_EVAL_PACK_SCHEMA:
        raise ValueError(f"Unexpected fixed-pack schema: {manifest_path}")
    scene_rows = manifest.get("scenes")
    if not isinstance(scene_rows, list) or not scene_rows:
        raise ValueError(f"Fixed pack has no ordered scenes: {manifest_path}")
    scenes: list[SceneSource] = []
    for row in scene_rows:
        scene_id = str(row["scene_id"])
        scene_manifest = source_root / scene_id / "source_manifest.json"
        scene = load_scene_source(
            scene_manifest,
            input_features="rgbn6",
            coordinate_normalization_policy=coordinate_policy,
            verify_hashes=True,
        )
        if not isinstance(scene, SceneSource):
            raise TypeError(
                f"Fixed Sonata evaluation requires one SceneSource per scene, got {scene_id}"
            )
        scenes.append(scene)
    return scenes, manifest


def fixed_pack_digest(manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(manifest["schema_version"]).encode())
    digest.update(str(manifest["sampling_profile"]).encode())
    digest.update(str(manifest["seed"]).encode())
    for row in manifest["scenes"]:
        digest.update(str(row["scene_id"]).encode())
        digest.update(str(row["source_manifest_sha256"]).encode())
        digest.update(str(row["sha256"]).encode())
    return digest.hexdigest()


def load_model(
    *,
    cfg: Mapping[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> tuple[SonataContrastiveModel, dict[str, Any] | None, str]:
    model_cfg = cfg["model"]
    model = SonataContrastiveModel(
        sonata_root=resolved_path(model_cfg["sonata_root"]),
        checkpoint=resolved_path(model_cfg["checkpoint"]),
        grid_size=float(model_cfg.get("grid_size", 0.02)),
        projection_dim=int(model_cfg.get("projection_dim", 128)),
        projection_hidden_dim=int(model_cfg.get("projection_hidden_dim", 128)),
        enable_flash=bool(model_cfg.get("enable_flash", True)),
        patch_size=int(model_cfg.get("patch_size", 1024)),
    )
    payload: dict[str, Any] | None = None
    if checkpoint != resolved_path(model_cfg["checkpoint"]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError(f"Unexpected adapted checkpoint schema: {checkpoint}")
        if int(payload.get("update", -1)) != 256:
            raise ValueError(f"Expected update-256 checkpoint, got {checkpoint}")
        state = payload.get("model")
        if not isinstance(state, Mapping):
            raise ValueError(f"Adapted checkpoint has no model state: {checkpoint}")
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, payload, state_hash(model.encoder)


def retrieval_statistics(
    embeddings: torch.Tensor,
    batch: Any,
) -> dict[str, float]:
    """Rank one paired positive against the exact fixed negative rows."""

    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-8)
    positive = normalized[batch.positive_pairs]
    negatives = normalized[batch.negative_indices]
    positive_cosine = (positive[:, 0] * positive[:, 1]).sum(dim=-1)
    cosine_a = torch.einsum("pd,pmd->pm", positive[:, 0], negatives)
    cosine_b = torch.einsum("pd,pmd->pm", positive[:, 1], negatives)
    positive_cosine = positive_cosine[:, None]
    ranks = torch.cat(
        [
            1 + (cosine_a >= positive_cosine).sum(dim=1),
            1 + (cosine_b >= positive_cosine).sum(dim=1),
        ]
    ).float()
    return {
        "paired_retrieval_ap": float((1.0 / ranks).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "recall_at_10": float((ranks <= 10).float().mean()),
        "mean_positive_rank": float(ranks.mean()),
    }


def empty_accumulator() -> dict[str, Any]:
    return {
        "weight": 0,
        "groups": 0,
        "metrics": {name: 0.0 for name in METRIC_NAMES},
        "mapping": {
            "attempted_groups": 0,
            "valid_groups": 0,
            "requested_positive_pairs": 0,
            "realized_positive_pairs": 0,
        },
    }


def add_metrics(
    accumulator: dict[str, Any],
    values: Mapping[str, float],
    weight: int,
) -> None:
    accumulator["weight"] += int(weight)
    accumulator["groups"] += 1
    for name in METRIC_NAMES:
        accumulator["metrics"][name] += float(values[name]) * int(weight)


def finish_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    weight = int(accumulator["weight"])
    if weight <= 0:
        raise RuntimeError("Evaluation retained zero positive pairs")
    return {
        "groups": int(accumulator["groups"]),
        "positive_pairs": weight,
        "metrics": {
            name: float(value / weight)
            for name, value in accumulator["metrics"].items()
        },
        "mapping": dict(accumulator["mapping"]),
    }


def evaluate_arm(
    *,
    arm_name: str,
    checkpoint: Path,
    cfg: Mapping[str, Any],
    scenes: list[SceneSource],
    pack_dir: Path,
    sampling: Any,
    pack_seed: int,
    device: torch.device,
    reference_inputs: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(17)
    model, checkpoint_payload, encoder_hash = load_model(
        cfg=cfg,
        checkpoint=checkpoint,
        device=device,
    )
    criterion = PartFieldContrastiveCriterion(
        temperature=FIXED_TEMPERATURE,
        learnable_temperature=False,
    ).to(device)
    overall = empty_accumulator()
    per_cell: dict[str, dict[str, Any]] = defaultdict(empty_accumulator)
    batch_digests: dict[str, str] = {}
    mapping_digests: dict[str, str] = {}
    forward_seeds: dict[str, int] = {}
    scene_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for scene_index, scene in enumerate(scenes):
            forward_seed = stable_seed(pack_seed, scene.scene_id, "eval-forward")
            forward_seeds[scene.scene_id] = int(forward_seed)
            seed_everything(forward_seed)
            points = torch.from_numpy(scene.points).to(device)
            if scene.colors is None or scene.normals is None:
                raise RuntimeError(f"{scene.scene_id}: RGBN6 arrays are incomplete")
            encoded = model.encoder(
                points=points,
                colors=scene.colors,
                normals=scene.normals,
                seed=int(forward_seed),
            )
            token_features = encoded.token_features.float()
            raw_to_token = encoded.raw_to_token.long()
            token_xyz = encoded.token_xyz.float()
            token_map_digest = hashlib.sha256(
                raw_to_token.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            xyz_digest = hashlib.sha256(
                token_xyz.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            mapping_digests[scene.scene_id] = f"{token_map_digest}:{xyz_digest}"
            views = _fixed_scene_batch_views(
                pack_dir=pack_dir,
                scene=scene,
                device=device,
                sampling=sampling,
            )
            fixed_groups = views["fixed_comparable"]
            scene_accumulator = empty_accumulator()
            for cell, group, point_batch in fixed_groups:
                token_batch, mapping = build_token_pure_dec0_contrastive_batch(
                    point_batch,
                    raw_to_token,
                    token_xyz,
                    max_positive_pairs_per_proposal=MAX_POSITIVE_PAIRS_PER_PROPOSAL,
                )
                key = f"{scene.scene_id}|{cell}|{group}"
                if token_batch is None:
                    raise RuntimeError(f"{key}: token remapping retained no batch")
                digest = batch_digest(token_batch)
                batch_digests[key] = digest
                result = criterion(token_features, token_batch)
                retrieval = retrieval_statistics(token_features, token_batch)
                values = {
                    "loss": float(result["loss_total"]),
                    "cosine_gap": float(result["cosine_gap"]),
                    "triplet_ranking_accuracy": float(
                        result["triplet_ranking_accuracy"]
                    ),
                    "hardest_negative_ranking_accuracy": float(
                        result["hardest_negative_ranking_accuracy"]
                    ),
                    **retrieval,
                }
                weight = int(token_batch.num_pairs)
                add_metrics(overall, values, weight)
                add_metrics(per_cell[cell], values, weight)
                add_metrics(scene_accumulator, values, weight)
                for accumulator in (overall, per_cell[cell], scene_accumulator):
                    accumulator["mapping"]["attempted_groups"] += 1
                    accumulator["mapping"]["valid_groups"] += int(
                        mapping.get("realized_positive_pairs", 0) > 0
                    )
                    accumulator["mapping"]["requested_positive_pairs"] += int(
                        sum(
                            int(proposal.get("requested_token_pairs", 0))
                            for proposal in mapping.get("proposals", ())
                        )
                    )
                    accumulator["mapping"]["realized_positive_pairs"] += int(
                        mapping.get("realized_positive_pairs", 0)
                    )
            scene_rows.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_index": scene_index,
                    "forward_seed": int(forward_seed),
                    "token_count": int(token_features.shape[0]),
                    "raw_point_count": int(points.shape[0]),
                    "groups": int(scene_accumulator["groups"]),
                    "positive_pairs": int(scene_accumulator["weight"]),
                    "metrics": finish_accumulator(scene_accumulator)["metrics"],
                    "mapping": dict(scene_accumulator["mapping"]),
                }
            )
            del encoded, token_features, raw_to_token, token_xyz, points
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (scene_index + 1) % 16 == 0 or scene_index == 0:
                print(
                    json.dumps(
                        {
                            "arm": arm_name,
                            "scene_index": scene_index + 1,
                            "scene_count": len(scenes),
                            "scene_id": scene.scene_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    finished_cells = {
        cell: finish_accumulator(accumulator)
        for cell, accumulator in sorted(per_cell.items())
    }
    report = {
        "name": arm_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_update": (
            int(checkpoint_payload["update"])
            if checkpoint_payload is not None
            else None
        ),
        "encoder_state_sha256": encoder_hash,
        "metrics": finish_accumulator(overall),
        "by_cell": finished_cells,
        "scenes": scene_rows,
        "forward_seed_digest": sha256_json(forward_seeds),
        "fixed_batch_digest": sha256_json(batch_digests),
        "token_mapping_digest": sha256_json(mapping_digests),
    }
    inputs = {
        "forward_seeds": forward_seeds,
        "batch_digests": batch_digests,
        "mapping_digests": mapping_digests,
    }
    if checkpoint_payload is not None:
        report["checkpoint_payload"] = {
            "schema_version": checkpoint_payload.get("schema_version"),
            "update": int(checkpoint_payload.get("update", -1)),
            "initial_encoder_state_sha256": checkpoint_payload.get(
                "initial_encoder_state_sha256"
            ),
            "payload_encoder_state_sha256": checkpoint_payload.get(
                "encoder_state_sha256"
            ),
            "config_sha256": checkpoint_payload.get("config_sha256"),
        }
    return report, inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing comparison; pass --overwrite: {output}"
        )
    cfg = load_config(args.config.resolve(strict=True))
    pack_dir = args.pack_dir.resolve(strict=True)
    adapted_checkpoint = args.adapted_checkpoint.resolve(strict=True)
    model_cfg = cfg["model"]
    public_checkpoint = resolved_path(model_cfg["checkpoint"])
    if sha256_file(public_checkpoint) != str(model_cfg["checkpoint_sha256"]):
        raise RuntimeError("Public Sonata checkpoint SHA-256 does not match config")
    sonata_root = resolved_path(model_cfg["sonata_root"])
    expected_sonata_commit = str(model_cfg.get("source_commit", ""))
    actual_sonata_commit = git_head(sonata_root)
    if expected_sonata_commit and actual_sonata_commit != expected_sonata_commit:
        raise RuntimeError(
            f"Sonata source commit drift: expected {expected_sonata_commit}, got {actual_sonata_commit}"
        )
    source_root = resolved_path(cfg["data"]["source_root"])
    coordinate_policy = str(cfg["data"]["coordinate_normalization_policy"])
    scenes, manifest = load_fixed_scenes(
        pack_dir=pack_dir,
        source_root=source_root,
        coordinate_policy=coordinate_policy,
    )
    sampling_profile = str(manifest["sampling_profile"])
    sampling = sampling_config_for_profile(sampling_profile)
    pack_report = validate_fixed_holdout_eval_pack(
        pack_dir=pack_dir,
        scenes=scenes,
        sampling=sampling,
        sampling_profile=sampling_profile,
        seed=int(manifest["seed"]),
        verify_scene_hashes=True,
        cells=manifest["source_cells"],
    )
    if not isinstance(pack_report.get("token_pure_dec0_fixed_pack"), dict):
        raise RuntimeError("The paired Sonata evaluation requires the token-pure fixed pack")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        gpu_name = "cpu"

    # Evaluate the original public arm first so its encoder hash anchors the
    # adapted checkpoint's recorded initialization and the fixed-input digest.
    public_report, public_inputs = evaluate_arm(
        arm_name="public_sonata",
        checkpoint=public_checkpoint,
        cfg=cfg,
        scenes=scenes,
        pack_dir=pack_dir,
        sampling=sampling,
        pack_seed=int(manifest["seed"]),
        device=device,
        reference_inputs=None,
    )
    adapted_report, adapted_inputs = evaluate_arm(
        arm_name="sonata_adapted_u256",
        checkpoint=adapted_checkpoint,
        cfg=cfg,
        scenes=scenes,
        pack_dir=pack_dir,
        sampling=sampling,
        pack_seed=int(manifest["seed"]),
        device=device,
        reference_inputs=public_inputs,
    )
    same_seeds = public_inputs["forward_seeds"] == adapted_inputs["forward_seeds"]
    same_batches = public_inputs["batch_digests"] == adapted_inputs["batch_digests"]
    same_mappings = public_inputs["mapping_digests"] == adapted_inputs["mapping_digests"]
    if not same_seeds or not same_batches or not same_mappings:
        raise RuntimeError(
            "Fixed evaluation inputs drifted between arms: "
            f"seeds={same_seeds}, batches={same_batches}, mappings={same_mappings}"
        )
    initial_hash = adapted_report.get("checkpoint_payload", {}).get(
        "initial_encoder_state_sha256"
    )
    if initial_hash != public_report["encoder_state_sha256"]:
        raise RuntimeError(
            "The adapted checkpoint does not record the loaded public encoder as its initialization"
        )
    payload_hash = adapted_report.get("checkpoint_payload", {}).get(
        "payload_encoder_state_sha256"
    )
    if payload_hash != adapted_report["encoder_state_sha256"]:
        raise RuntimeError("Adapted checkpoint encoder hash does not match loaded state")
    expected_config_hash = sha256_json(cfg)
    payload_config_hash = adapted_report.get("checkpoint_payload", {}).get("config_sha256")
    if payload_config_hash != expected_config_hash:
        raise RuntimeError(
            f"Adapted checkpoint config hash drift: expected {expected_config_hash}, got {payload_config_hash}"
        )

    report = {
        "schema_version": REPORT_SCHEMA,
        "feature_space": "native_sonata_final_tokens",
        "objective": {
            "criterion": "PartFieldContrastiveCriterion",
            "temperature": FIXED_TEMPERATURE,
            "learnable_temperature": False,
            "loss_and_cosine_gap": "same fixed-comparable token batch for both arms",
            "retrieval": "one paired positive ranked against the fixed base negatives, both endpoints",
        },
        "contract": {
            "config": str(args.config.resolve()),
            "config_sha256": expected_config_hash,
            "sonata_source": str(sonata_root),
            "sonata_source_commit": actual_sonata_commit,
            "source_root": str(source_root),
            "coordinate_normalization_policy": coordinate_policy,
            "input_contract": model_cfg.get("input_contract"),
            "grid_size": float(model_cfg.get("grid_size", 0.02)),
            "augmentation_profile": cfg["data"].get("augmentation_profile"),
            "fixed_pack": pack_report,
            "fixed_pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
            "fixed_pack_content_digest": fixed_pack_digest(manifest),
            "fixed_view": "fixed_comparable",
            "source_cells": list(manifest["source_cells"]),
            "scene_order": [scene.scene_id for scene in scenes],
            "scene_count": len(scenes),
            "pack_seed": int(manifest["seed"]),
            "per_scene_forward_seed": "stable_seed(pack_seed, scene_id, 'eval-forward')",
            "max_positive_pairs_per_proposal": MAX_POSITIVE_PAIRS_PER_PROPOSAL,
        },
        "input_identity": {
            "same_scene_order": True,
            "same_pseudomask_proposals": True,
            "same_fixed_base_negatives": True,
            "same_forward_seeds": same_seeds,
            "same_token_mappings": same_mappings,
            "same_remapped_batches": same_batches,
            "remapped_batch_digest": public_report["fixed_batch_digest"],
            "token_mapping_digest": public_report["token_mapping_digest"],
            "forward_seed_digest": public_report["forward_seed_digest"],
        },
        "runtime": {
            "device": str(device),
            "gpu": gpu_name,
            "torch_version": torch.__version__,
        },
        "arms": [public_report, adapted_report],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "scene_count": len(scenes),
                "fixed_pack_manifest_sha256": report["contract"]["fixed_pack_manifest_sha256"],
                "same_remapped_batches": same_batches,
                "public": public_report["metrics"],
                "adapted": adapted_report["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
