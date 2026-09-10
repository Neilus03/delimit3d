#!/usr/bin/env python3
"""Select the best native holdout checkpoint and apply the upstream gate."""

from __future__ import annotations

from delimit3d.identity import project_name, project_slug

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    LOGICAL_BATCH_EVAL_EPOCHS,
    SOURCE_CELLS,
    SOURCE_CELL_SCHEDULE_CHOICES,
    SOURCE_CELL_SCHEDULE_FIXED,
    SOURCE_FAMILY_2D,
    SOURCE_FAMILY_2D3D,
    SOURCE_FAMILY_CHOICES,
    SCENES_PER_RANK,
    exposure_counts,
    load_scene_ids,
    sha256_file,
    source_cells_for_family,
)
from delimit3d.training.adaptation import (
    MULTISCALE_AUXILIARY_WEIGHT,
    MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
    MULTISCALE_LEVEL_LOSS_WEIGHTS,
    MULTISCALE_LEVEL_NAMES,
    MULTISCALE_RECIPE_NAME,
    TRAIN_AUGMENTATION_CHOICES,
    augmentation_contract_for_profile,
)


def _atomic_json_noclobber(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing existing checkpoint selection: {path}")
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".candidate",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        candidate = Path(handle.name)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100,
            os.fsencode(candidate),
            -100,
            os.fsencode(path),
            1,
        ) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-epochs",
        type=int,
        choices=(336, 1008),
        default=336,
        help="Fixed complete schedule to validate before checkpoint selection.",
    )
    parser.add_argument(
        "--expected-source-family",
        choices=SOURCE_FAMILY_CHOICES,
        default=SOURCE_FAMILY_2D3D,
        help="Strict active supervision family expected in the completed run.",
    )
    parser.add_argument(
        "--expected-three-d-mode",
        choices=("gt3", "g08_repeated"),
        default="gt3",
    )
    parser.add_argument(
        "--expected-sampling-profile",
        choices=("pf_hnm_512_v1", "pf_hnm_512_p16_v1", "pf_hnm_768_v1"),
        default=None,
        help="Optional strict training sampling contract required by the run.",
    )
    parser.add_argument(
        "--expected-evaluation-sampling-profile",
        choices=("pf_hnm_512_v1", "pf_hnm_512_p16_v1", "pf_hnm_768_v1"),
        default=None,
        help="Optional strict holdout sampling contract required by the run.",
    )
    parser.add_argument(
        "--expected-input-features",
        choices=("rgb3", "rgbn6"),
        default=None,
        help="Optional strict LitePT input-feature contract required by the run.",
    )
    parser.add_argument(
        "--expected-pretraining-recipe",
        choices=("final_dec0_only_v1", MULTISCALE_RECIPE_NAME),
        default=None,
        help="Optional strict objective and feature-level recipe contract.",
    )
    parser.add_argument(
        "--expected-train-augmentation-profile",
        choices=TRAIN_AUGMENTATION_CHOICES,
        default=None,
        help="Optional strict training-only input-augmentation contract.",
    )
    parser.add_argument(
        "--expected-required-crop-mode",
        choices=("sphere_50k", "full_scene"),
        default=None,
        help="Optional strict source point-coverage contract.",
    )
    parser.add_argument(
        "--expected-scenes-per-rank-per-update",
        type=int,
        default=None,
        help="Optional strict per-rank logical scene-batch contract.",
    )
    parser.add_argument(
        "--expected-proposals-per-forward",
        type=int,
        default=None,
        help="Optional strict proposal-mask count per scene forward.",
    )
    parser.add_argument(
        "--expected-evaluation-proposals-per-forward",
        type=int,
        default=None,
        help="Optional strict proposal count used for checkpoint evaluation.",
    )
    parser.add_argument(
        "--expected-source-cell-schedule",
        choices=SOURCE_CELL_SCHEDULE_CHOICES,
        default=None,
        help="Optional strict balanced source-cell schedule contract.",
    )
    parser.add_argument(
        "--expected-source-cell-schedule-seed",
        type=int,
        default=None,
        help="Optional strict seed for the source-cell schedule.",
    )
    return parser.parse_args()


def _eval_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == "eval":
            epoch = int(row["epoch"])
            if epoch in rows:
                raise ValueError(f"Duplicate evaluation epoch {epoch}")
            rows[epoch] = row
    return rows


def main() -> None:
    args = parse_args()
    summary = json.loads(
        (args.run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (args.run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    if summary.get("experiment_id") != EXPERIMENT_ID or not summary.get("passed"):
        raise RuntimeError("Full upstream run did not pass its execution checks")
    source_family = str(
        resolved.get("source_family", SOURCE_FAMILY_2D3D)
    )
    if source_family != args.expected_source_family:
        raise RuntimeError(
            "Full upstream run uses an unexpected source family: "
            f"{source_family!r} != {args.expected_source_family!r}"
        )
    source_cells = source_cells_for_family(source_family)
    resolved_source_cells = tuple(
        resolved.get("source_cells", SOURCE_CELLS)
    )
    if resolved_source_cells != source_cells:
        raise RuntimeError(
            "Full upstream run uses an unexpected source-cell contract: "
            f"{resolved_source_cells!r} != {source_cells!r}"
        )
    if source_family == SOURCE_FAMILY_2D3D:
        if resolved.get("three_d_mode") != args.expected_three_d_mode:
            raise RuntimeError(
                "Full upstream run uses an unexpected three-dimensional source mode"
            )
    elif source_family == SOURCE_FAMILY_2D:
        if resolved.get("three_d_mode") is not None:
            raise RuntimeError("2D-only upstream run unexpectedly declares a 3D mode")
    input_features = resolved.get("input_features")
    input_channels = resolved.get("input_channels")
    pretraining_recipe = str(
        resolved.get("pretraining_recipe", "final_dec0_only_v1")
    )
    if (
        args.expected_pretraining_recipe is not None
        and pretraining_recipe != args.expected_pretraining_recipe
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected pretraining recipe: "
            f"{pretraining_recipe!r} != {args.expected_pretraining_recipe!r}"
        )
    if args.expected_pretraining_recipe == MULTISCALE_RECIPE_NAME:
        if resolved.get("multiscale_supervision") is not True:
            raise RuntimeError("Multi-scale recipe is not enabled in the run")
        if resolved.get("multiscale_levels") != list(MULTISCALE_LEVEL_NAMES):
            raise RuntimeError("Multi-scale decoder-level contract drift")
        if resolved.get("multiscale_level_loss_weights") != dict(
            MULTISCALE_LEVEL_LOSS_WEIGHTS
        ):
            raise RuntimeError("Multi-scale stage-weight contract drift")
        if resolved.get("multiscale_loss_weight") != MULTISCALE_AUXILIARY_WEIGHT:
            raise RuntimeError("Multi-scale auxiliary-budget contract drift")
        if (
            resolved.get("multiscale_warmup_epochs")
            != MULTISCALE_AUXILIARY_WARMUP_EPOCHS
        ):
            raise RuntimeError("Multi-scale auxiliary-ramp contract drift")
        if (
            resolved.get("multiscale_token_policy")
            != "strict_pure_token_resampling_v1"
        ):
            raise RuntimeError("Multi-scale token-purity contract drift")
        if (
            resolved.get("multiscale_temperature_policy")
            != "separate_learnable_temperature_per_auxiliary_level"
        ):
            raise RuntimeError("Multi-scale temperature-isolation contract drift")
        if (
            resolved.get("multiscale_feature_hard_policy")
            != "separate_candidate_selection_per_positive_endpoint"
        ):
            raise RuntimeError("Multi-scale feature-hard symmetry contract drift")
        if resolved.get("multiscale_transfer_contract") != "native_mask3d_fpn_required":
            raise RuntimeError("Multi-scale transfer contract drift")
    if args.expected_input_features is not None:
        expected_channels = 3 if args.expected_input_features == "rgb3" else 6
        if (
            input_features != args.expected_input_features
            or int(input_channels or -1) != expected_channels
        ):
            raise RuntimeError(
                "Full upstream run uses an unexpected input-feature contract: "
                f"features={input_features!r} channels={input_channels!r}"
            )
        if resolved.get("voxel_reduce") != "representative":
            raise RuntimeError(
                "Full upstream run does not use representative voxel reduction"
            )
    train_augmentation_profile = str(
        resolved.get("train_augmentation_profile", "none")
    )
    if (
        args.expected_train_augmentation_profile is not None
        and train_augmentation_profile
        != args.expected_train_augmentation_profile
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected train augmentation profile: "
            f"{train_augmentation_profile!r} != "
            f"{args.expected_train_augmentation_profile!r}"
        )
    if resolved.get("evaluation_augmentation_profile", "none") != "none":
        raise RuntimeError("Checkpoint evaluation unexpectedly uses augmentation")
    expected_train_augmentation = augmentation_contract_for_profile(
        train_augmentation_profile
    )
    resolved_train_augmentation = resolved.get(
        "train_augmentation", expected_train_augmentation
    )
    if resolved_train_augmentation != expected_train_augmentation:
        raise RuntimeError("Full upstream run augmentation parameters drifted")
    required_crop_mode = str(resolved.get("required_crop_mode", "any"))
    if (
        args.expected_required_crop_mode is not None
        and required_crop_mode != args.expected_required_crop_mode
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected source crop mode: "
            f"{required_crop_mode!r} != {args.expected_required_crop_mode!r}"
        )
    scenes_per_rank_per_update = int(
        resolved.get("scenes_per_rank_per_update", 1)
    )
    if scenes_per_rank_per_update <= 0:
        raise RuntimeError("Full upstream run has an invalid logical scene batch")
    if (
        args.expected_scenes_per_rank_per_update is not None
        and scenes_per_rank_per_update
        != int(args.expected_scenes_per_rank_per_update)
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected per-rank scene batch: "
            f"{scenes_per_rank_per_update} != "
            f"{args.expected_scenes_per_rank_per_update}"
        )
    sampling = resolved.get("sampling") or {}
    proposals_per_forward = int(sampling.get("proposals_per_forward", 4))
    if (
        args.expected_proposals_per_forward is not None
        and proposals_per_forward != int(args.expected_proposals_per_forward)
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected proposal count: "
            f"{proposals_per_forward} != {args.expected_proposals_per_forward}"
        )
    evaluation_sampling_profile = str(
        resolved.get(
            "evaluation_sampling_profile",
            resolved.get("sampling_profile", "pf_hnm_512_v1"),
        )
    )
    evaluation_sampling = resolved.get("evaluation_sampling") or sampling
    evaluation_proposals_per_forward = int(
        evaluation_sampling.get("proposals_per_forward", proposals_per_forward)
    )
    if (
        args.expected_evaluation_sampling_profile is not None
        and evaluation_sampling_profile
        != args.expected_evaluation_sampling_profile
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected evaluation sampling profile: "
            f"{evaluation_sampling_profile!r} != "
            f"{args.expected_evaluation_sampling_profile!r}"
        )
    if (
        args.expected_evaluation_proposals_per_forward is not None
        and evaluation_proposals_per_forward
        != int(args.expected_evaluation_proposals_per_forward)
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected evaluation proposal count: "
            f"{evaluation_proposals_per_forward} != "
            f"{args.expected_evaluation_proposals_per_forward}"
        )
    source_cell_schedule = str(
        resolved.get("source_cell_schedule", SOURCE_CELL_SCHEDULE_FIXED)
    )
    if (
        args.expected_source_cell_schedule is not None
        and source_cell_schedule != args.expected_source_cell_schedule
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected source-cell schedule: "
            f"{source_cell_schedule!r} != "
            f"{args.expected_source_cell_schedule!r}"
        )
    schedule_seed = int(
        resolved.get("source_cell_schedule_seed", resolved.get("seed", 42))
    )
    if (
        args.expected_source_cell_schedule_seed is not None
        and schedule_seed != int(args.expected_source_cell_schedule_seed)
    ):
        raise RuntimeError(
            "Full upstream run uses an unexpected source-cell schedule seed: "
            f"{schedule_seed} != {args.expected_source_cell_schedule_seed}"
        )
    fixed_eval_pack = resolved.get("fixed_eval_pack")
    code_manifest = resolved.get("code_manifest")
    if args.expected_sampling_profile is not None:
        if resolved.get("sampling_profile") != args.expected_sampling_profile:
            raise RuntimeError("Full upstream run uses an unexpected sampling profile")
        if (
            resolved.get("cache_training_voxelization") is not False
            or resolved.get("voxelization_mode")
            != "per_forward_single_scene_recompute"
        ):
            raise RuntimeError("Full upstream run lacks corrected voxelization provenance")
        if args.expected_sampling_profile == "pf_hnm_768_v1":
            if not fixed_eval_pack or not fixed_eval_pack.get("passed"):
                raise RuntimeError("Full upstream run lacks a passed fixed-eval pack")
            manifest_path = Path(
                fixed_eval_pack["manifest_path"]
            ).resolve(strict=True)
            if sha256_file(manifest_path) != fixed_eval_pack["manifest_sha256"]:
                raise RuntimeError("Fixed-eval pack manifest hash drift")
            if not code_manifest or not code_manifest.get("passed"):
                raise RuntimeError("Full upstream run lacks a pinned code manifest")
            code_manifest_path = Path(code_manifest["path"]).resolve(strict=True)
            if sha256_file(code_manifest_path) != code_manifest["sha256"]:
                raise RuntimeError("Code-snapshot manifest hash drift")
    source_audit = resolved.get("source_dataset_audit")
    if source_audit is None and resolved.get("dataset") == "structured3d":
        # Structured3D has no ScanNet-shaped aggregate audit file.  Its
        # training path audits every source_manifest.json through
        # load_scene_source, so preserve that provenance contract explicitly
        # instead of treating the intentional null field as a failed audit.
        split_metadata_path = args.train_split.with_name(
            args.train_split.name + ".json"
        ).resolve(strict=True)
        split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
        if split_metadata.get("dataset") != "structured3d":
            raise RuntimeError("Structured3D split metadata dataset drift")
        source_root = Path(split_metadata["source_root"]).resolve(strict=True)
        checks = summary.get("checks", {})
        source_audit = {
            "schema_version": "structured3d_per_scene_manifest_validation/v1",
            "passed": bool(
                checks.get("source_cells_exact")
                and checks.get("source_exposure_exact")
            ),
            "mode": "per_scene_manifest_validation_during_training",
            "metadata_path": str(split_metadata_path),
            "metadata_sha256": sha256_file(split_metadata_path),
            "source_root": str(source_root),
            "checks": {
                "source_cells_exact": bool(checks.get("source_cells_exact")),
                "source_exposure_exact": bool(
                    checks.get("source_exposure_exact")
                ),
                "training_run_passed": bool(summary.get("passed")),
            },
        }
        if not source_audit["passed"]:
            raise RuntimeError("Structured3D per-scene source audit contract failed")
    elif not source_audit or not source_audit.get("passed"):
        raise RuntimeError("Full upstream run lacks a passed frozen source audit")
    else:
        source_audit_path = Path(source_audit["path"]).resolve(strict=True)
        if sha256_file(source_audit_path) != source_audit["sha256"]:
            raise RuntimeError("Frozen dataset source-audit hash drift")
    scene_visits_per_epoch_per_rank = int(
        resolved.get(
            "scene_visits_per_epoch_per_rank",
            resolved.get("updates_per_epoch", SCENES_PER_RANK),
        )
    )
    expected_scene_visits = (
        int(args.expected_epochs) * scene_visits_per_epoch_per_rank
    )
    if expected_scene_visits % scenes_per_rank_per_update != 0:
        raise RuntimeError(
            "Expected schedule does not end on a complete logical scene batch"
        )
    expected_updates = expected_scene_visits // scenes_per_rank_per_update
    if (
        int(summary.get("epochs", -1)) != int(args.expected_epochs)
        or int(summary.get("updates", -1)) != expected_updates
    ):
        raise RuntimeError(
            f"Expected the complete {args.expected_epochs}-epoch / "
            f"{expected_updates:,}-update run"
        )
    evaluations = _eval_rows(args.run_dir / "metrics.jsonl")
    continuation = summary.get("continuation") or {}
    if continuation.get("enabled"):
        parent_epoch = int(continuation["parent_checkpoint"]["epoch"])
        expected_epochs = (
            {
                epoch
                for epoch in LOGICAL_BATCH_EVAL_EPOCHS
                if parent_epoch <= epoch <= int(args.expected_epochs)
            }
            | {parent_epoch, int(args.expected_epochs)}
            if scenes_per_rank_per_update > 1
            else {parent_epoch, 504, 672, 840, 1008}
        )
    else:
        if scenes_per_rank_per_update > 1:
            expected_epochs = {
                epoch
                for epoch in LOGICAL_BATCH_EVAL_EPOCHS
                if epoch <= int(args.expected_epochs)
            } | {int(args.expected_epochs)}
        else:
            expected_epochs = (
                {0, 17, 34, 85, 169, 252, 336}
                if int(args.expected_epochs) == 336
                else {
                    0,
                    17,
                    34,
                    85,
                    169,
                    252,
                    336,
                    504,
                    672,
                    756,
                    840,
                    924,
                    1008,
                }
            )
    if set(evaluations) != expected_epochs:
        raise RuntimeError(
            f"Checkpoint-evaluation epochs drift: {sorted(evaluations)}"
        )
    if args.expected_sampling_profile == "pf_hnm_768_v1":
        invalid_views = {
            epoch: row.get("selection_view")
            for epoch, row in evaluations.items()
            if row.get("selection_view") != "fixed_comparable"
        }
        invalid_pack_hashes = {
            epoch: row.get("fixed_eval_pack", {}).get("manifest_sha256")
            for epoch, row in evaluations.items()
            if row.get("fixed_eval_pack", {}).get("manifest_sha256")
            != fixed_eval_pack["manifest_sha256"]
        }
        if invalid_views:
            raise RuntimeError(f"Evaluation selection-view drift: {invalid_views}")
        if invalid_pack_hashes:
            raise RuntimeError(
                f"Evaluation fixed-pack hash drift: {invalid_pack_hashes}"
            )
    initial_epoch = min(expected_epochs)
    initial = evaluations[initial_epoch]
    candidates = [
        evaluations[epoch] for epoch in sorted(expected_epochs - {initial_epoch})
    ]
    selected = max(
        candidates,
        key=lambda row: (
            float(row["native_selection_macro"]["overall"]),
            -int(row["epoch"]),
        ),
    )
    selected_epoch = int(selected["epoch"])
    selected_checkpoint = (
        args.run_dir / "checkpoints" / f"epoch_{selected_epoch:04d}.pt"
    )
    checkpoint = torch.load(
        selected_checkpoint, map_location="cpu", weights_only=False
    )
    if int(checkpoint["epoch"]) != selected_epoch:
        raise RuntimeError("Selected checkpoint epoch drift")
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Selected checkpoint experiment drift")
    if checkpoint.get("public_checkpoint_used") is not False:
        raise RuntimeError("Selected checkpoint does not exclude public initialization")
    checkpoint_resolved = checkpoint.get("resolved_config", {})
    if args.expected_pretraining_recipe is not None:
        checkpoint_recipe = str(
            checkpoint_resolved.get("pretraining_recipe", "final_dec0_only_v1")
        )
        if checkpoint_recipe != args.expected_pretraining_recipe:
            raise RuntimeError("Selected checkpoint pretraining recipe drift")
        if args.expected_pretraining_recipe == MULTISCALE_RECIPE_NAME:
            for key in (
                "multiscale_supervision",
                "multiscale_levels",
                "multiscale_level_loss_weights",
                "multiscale_loss_weight",
                "multiscale_warmup_epochs",
                "multiscale_token_policy",
                "multiscale_temperature_policy",
                "multiscale_feature_hard_policy",
                "multiscale_transfer_contract",
            ):
                if checkpoint_resolved.get(key) != resolved.get(key):
                    raise RuntimeError(
                        f"Selected checkpoint {key} contract drift"
                    )
    if args.expected_input_features is not None and (
        checkpoint_resolved.get("input_features") != input_features
        or checkpoint_resolved.get("input_channels") != input_channels
        or checkpoint_resolved.get("voxel_reduce") != "representative"
        or checkpoint_resolved.get("source_family") != source_family
        or tuple(checkpoint_resolved.get("source_cells", ())) != source_cells
    ):
        raise RuntimeError("Selected checkpoint feature/source contract drift")
    checkpoint_sampling = checkpoint_resolved.get("sampling") or {}
    checkpoint_evaluation_sampling = (
        checkpoint_resolved.get("evaluation_sampling")
        or checkpoint_sampling
    )
    if (
        int(
            checkpoint_resolved.get("scenes_per_rank_per_update", 1)
        )
        != scenes_per_rank_per_update
        or int(checkpoint_sampling.get("proposals_per_forward", 4))
        != proposals_per_forward
        or str(
            checkpoint_resolved.get(
                "evaluation_sampling_profile", evaluation_sampling_profile
            )
        )
        != evaluation_sampling_profile
        or int(
            checkpoint_evaluation_sampling.get(
                "proposals_per_forward", evaluation_proposals_per_forward
            )
        )
        != evaluation_proposals_per_forward
        or str(
            checkpoint_resolved.get(
                "source_cell_schedule", SOURCE_CELL_SCHEDULE_FIXED
            )
        )
        != source_cell_schedule
        or int(
            checkpoint_resolved.get(
                "source_cell_schedule_seed", schedule_seed
            )
        )
        != schedule_seed
        or str(
            checkpoint_resolved.get(
                "train_augmentation_profile", train_augmentation_profile
            )
        )
        != train_augmentation_profile
        or str(
            checkpoint_resolved.get("required_crop_mode", required_crop_mode)
        )
        != required_crop_mode
        or checkpoint_resolved.get(
            "train_augmentation", resolved_train_augmentation
        )
        != resolved_train_augmentation
    ):
        raise RuntimeError("Selected checkpoint logical-batch contract drift")
    checkpoint_macro = float(
        checkpoint["evaluation"]["native_selection_macro"]["overall"]
    )
    selected_macro = float(selected["native_selection_macro"]["overall"])
    if not math.isclose(checkpoint_macro, selected_macro, abs_tol=1e-12):
        raise RuntimeError("Selected checkpoint evaluation differs from metrics log")

    train_scene_ids = load_scene_ids(args.train_split)
    expected_counts = exposure_counts(
        train_scene_ids,
        epochs=selected_epoch,
        cells=source_cells,
        proposals_per_forward=proposals_per_forward,
        cell_schedule=source_cell_schedule,
        seed=schedule_seed,
    )
    observed_counts = {
        key: int(value)
        for key, value in checkpoint["global_exposure_counts"].items()
    }
    initial_native = initial["native_selection_macro"]
    selected_native = selected["native_selection_macro"]
    active_macro_keys = (
        ("overall", "2d", "3d")
        if source_family == SOURCE_FAMILY_2D3D
        else ("overall", "2d")
    )
    missing_macro_keys = {
        "initial": sorted(set(active_macro_keys) - set(initial_native)),
        "selected": sorted(set(active_macro_keys) - set(selected_native)),
    }
    if any(missing_macro_keys.values()):
        raise RuntimeError(
            f"Active source-family macros are missing: {missing_macro_keys}"
        )
    improvements = {
        key: float(selected_native[key]) - float(initial_native[key])
        for key in active_macro_keys
    }
    health = selected["native_health"]
    gate_checks = {
        "overall_macro_improves_0.02": improvements["overall"] >= 0.02,
        "effective_rank_at_least_2": float(health["effective_rank"]) >= 2.0,
        "minimum_std_at_least_1e-3": float(health["feature_std_min"]) >= 1e-3,
        "source_counts_exact": observed_counts == expected_counts,
        "source_cell_set_exact": set(observed_counts) == set(source_cells),
        "logical_scene_batch_exact": (
            args.expected_scenes_per_rank_per_update is None
            or scenes_per_rank_per_update
            == int(args.expected_scenes_per_rank_per_update)
        ),
        "proposal_count_exact": (
            args.expected_proposals_per_forward is None
            or proposals_per_forward
            == int(args.expected_proposals_per_forward)
        ),
        "evaluation_sampling_profile_exact": (
            args.expected_evaluation_sampling_profile is None
            or evaluation_sampling_profile
            == args.expected_evaluation_sampling_profile
        ),
        "evaluation_proposal_count_exact": (
            args.expected_evaluation_proposals_per_forward is None
            or evaluation_proposals_per_forward
            == int(args.expected_evaluation_proposals_per_forward)
        ),
        "source_cell_schedule_exact": (
            args.expected_source_cell_schedule is None
            or source_cell_schedule == args.expected_source_cell_schedule
        ),
        "source_cell_schedule_seed_exact": (
            args.expected_source_cell_schedule_seed is None
            or schedule_seed == int(args.expected_source_cell_schedule_seed)
        ),
        "train_augmentation_profile_exact": (
            args.expected_train_augmentation_profile is None
            or train_augmentation_profile
            == args.expected_train_augmentation_profile
        ),
        "source_crop_mode_exact": (
            args.expected_required_crop_mode is None
            or required_crop_mode == args.expected_required_crop_mode
        ),
        "pretraining_recipe_exact": (
            args.expected_pretraining_recipe is None
            or pretraining_recipe == args.expected_pretraining_recipe
        ),
        "checkpoint_strict_execution_passed": True,
    }
    if source_family == SOURCE_FAMILY_2D3D:
        gate_checks.update(
            {
                "3d_macro_improves_0.02": improvements["3d"] >= 0.02,
                "2d_macro_nondecrease": improvements["2d"] >= 0.0,
            }
        )
    else:
        gate_checks["2d_macro_improves_0.02"] = improvements["2d"] >= 0.02
    if args.expected_sampling_profile == "pf_hnm_768_v1":
        gate_checks["fixed_eval_pack_exact"] = bool(
            fixed_eval_pack
            and fixed_eval_pack.get("passed")
            and selected.get("selection_view") == "fixed_comparable"
        )
        gate_checks["corrected_voxelization_exact"] = (
            resolved.get("cache_training_voxelization") is False
            and resolved.get("voxelization_mode")
            == "per_forward_single_scene_recompute"
        )
        gate_checks["code_manifest_exact"] = bool(
            code_manifest and code_manifest.get("passed")
        )
    report = {
        "schema_version": "litept_gt3_unsam3_checkpoint_selection/v3",
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "selection_metric": selected.get(
            "selection_metric",
            "native 72D six-cell macro hardest-negative ranking",
        ),
        "checkpoint_selection_policy": "final_dec0_native_selection_macro",
        "selection_view": selected.get(
            "selection_view", "adaptive_feature_hard"
        ),
        "sampling_profile": resolved.get(
            "sampling_profile", "pf_hnm_512_v1"
        ),
        "evaluation_sampling_profile": evaluation_sampling_profile,
        "source_family": source_family,
        "source_cells": list(source_cells),
        "three_d_mode": resolved.get("three_d_mode"),
        "input_features": input_features,
        "input_channels": input_channels,
        "pretraining_recipe": pretraining_recipe,
        "multiscale_supervision": bool(
            resolved.get("multiscale_supervision", False)
        ),
        "multiscale_levels": list(
            resolved.get("multiscale_levels", MULTISCALE_LEVEL_NAMES)
        ),
        "multiscale_level_loss_weights": resolved.get(
            "multiscale_level_loss_weights"
        ),
        "multiscale_loss_weight": resolved.get("multiscale_loss_weight"),
        "multiscale_warmup_epochs": resolved.get("multiscale_warmup_epochs"),
        "multiscale_token_policy": resolved.get("multiscale_token_policy"),
        "multiscale_temperature_policy": resolved.get(
            "multiscale_temperature_policy"
        ),
        "multiscale_feature_hard_policy": resolved.get(
            "multiscale_feature_hard_policy"
        ),
        "multiscale_transfer_contract": resolved.get(
            "multiscale_transfer_contract"
        ),
        "train_augmentation_profile": train_augmentation_profile,
        "train_augmentation": resolved_train_augmentation,
        "evaluation_augmentation_profile": "none",
        "required_crop_mode": required_crop_mode,
        "voxel_reduce": resolved.get("voxel_reduce"),
        "scenes_per_rank_per_update": scenes_per_rank_per_update,
        "scenes_per_global_update": int(
            resolved.get(
                "scenes_per_global_update",
                scenes_per_rank_per_update * int(resolved.get("world_size", 1)),
            )
        ),
        "proposals_per_forward": proposals_per_forward,
        "evaluation_proposals_per_forward": (
            evaluation_proposals_per_forward
        ),
        "proposals_per_global_update": int(
            resolved.get(
                "proposals_per_global_update",
                proposals_per_forward
                * scenes_per_rank_per_update
                * int(resolved.get("world_size", 1)),
            )
        ),
        "source_cell_schedule": source_cell_schedule,
        "source_cell_schedule_seed": schedule_seed,
        "tie_break": "earliest epoch",
        "evaluated_epochs": sorted(evaluations),
        "scores": {
            str(epoch): {
                "native": evaluations[epoch]["native_selection_macro"],
                "health": evaluations[epoch]["native_health"],
            }
            for epoch in sorted(evaluations)
        },
        "selected_epoch": selected_epoch,
        "selected_update": int(selected["update"]),
        "selected_checkpoint": str(selected_checkpoint.resolve(strict=True)),
        "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
        "initial_native_macro": initial_native,
        "selected_native_macro": selected_native,
        "improvements": improvements,
        "selected_native_health": health,
        "observed_exposure_counts": observed_counts,
        "expected_exposure_counts": expected_counts,
        "gate_checks": gate_checks,
        "upstream_gate_passed": all(gate_checks.values()),
        "initial_checkpoint_sha256": resolved["initialization"]["sha256"],
        "source_dataset_audit": source_audit,
        "fixed_eval_pack": fixed_eval_pack,
        "code_manifest": code_manifest,
        "cache_training_voxelization": resolved.get(
            "cache_training_voxelization"
        ),
        "voxelization_mode": resolved.get("voxelization_mode"),
        "public_checkpoint_used": False,
    }
    _atomic_json_noclobber(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
