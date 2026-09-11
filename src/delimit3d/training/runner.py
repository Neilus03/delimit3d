#!/usr/bin/env python3
"""Distributed scratch LitePT GT3 + UnSAM3 contrastive pretraining."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from delimit3d.identity import project_name, project_slug
from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    LOGICAL_BATCH_EVAL_EPOCHS,
    OFFICIAL_TRAIN_SCENES,
    SCENES_PER_RANK,
    SOURCE_CELLS,
    SOURCE_CELL_SCHEDULE_CHOICES,
    SOURCE_CELL_SCHEDULE_FIXED,
    SOURCE_FAMILY_2D,
    SOURCE_FAMILY_CHOICES,
    WORLD_SIZE,
    audit_scene_source_manifest,
    extension_lr_factor,
    exposure_counts,
    lr_factor,
    load_scene_ids,
    normalize_initialization_report,
    sha256_file,
    source_cells_for_family,
    split_digest,
    stable_seed,
)
from delimit3d.training.adaptation import (
    INPUT_FEATURE_CHOICES,
    HIERARCHY_QUERY_PROJECTION_POLICIES,
    LitePTContrastiveModel,
    MULTISCALE_AUXILIARY_WEIGHT,
    MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
    MULTISCALE_LEVEL_LOSS_WEIGHTS,
    MULTISCALE_LEVEL_NAMES,
    SAMPLING_PROFILES,
    RGBN6_COLOR_NORMALIZATION,
    COMMON_RGB_COLOR_MEAN,
    COMMON_RGB_COLOR_STD,
    TRAIN_AUGMENTATION_CHOICES,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
    augmentation_contract_for_profile,
    build_scene_visit_input,
    evaluate_holdout,
    hierarchy_gradient_telemetry,
    load_scene_source,
    normalize_hierarchy_mask_stage_weight_schedule,
    normalize_hierarchy_mask_stage_weights,
    optimizer_parameter_groups,
    prepare_fixed_holdout_eval_pack,
    sampling_config_for_profile,
    scene_source_for_visit,
    set_optimizer_lr,
    training_batch,
)
from delimit3d.data.contrastive_sampler_v2 import (
    DEFAULT_2D_CELLS as V2_DEFAULT_2D_CELLS,
    DeterministicCoverageState,
    FrameProposalCatalog,
    NoRetainedFrameGroupsError,
    sample_multigranular_frame_group_plan,
)
from delimit3d.data.point_augmentations import (
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
)


log = logging.getLogger(f"{project_slug()}.adaptation")
# The 336-epoch base schedule and its 3x extension share the first checkpoints.
# Each later checkpoint lands on a complete six-cell exposure cycle.
DEFAULT_EVAL_EPOCHS = (
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
)
V2_SAMPLING_MODE = "coverage_multigranular_v2"
TRAINING_SAMPLING_MODES = (
    "random_with_replacement",
    "coverage_without_replacement",
    V2_SAMPLING_MODE,
)
HIERARCHY_SUPERVISION_MODES = (
    "off",
    "legacy",
    "query_mask",
    "query_mask_vicreg",
)
NORMALIZATION_POLICIES = ("batchnorm", "sync_batchnorm")
COORDINATE_NORMALIZATION_POLICIES = (
    "native",
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
)
CUMULATIVE_HEALTH_SCHEMA = "litept_pretraining_cumulative_health/v1"
RANK_CUMULATIVE_HEALTH_SCHEMA = "litept_pretraining_rank_cumulative_health/v1"
TOKEN_CUMULATIVE_HEALTH_SCHEMA = "litept_token_pure_delivery_cumulative/v1"
HIERARCHY_CUMULATIVE_HEALTH_SCHEMA = "litept_hierarchy_delivery_cumulative/v1"
LEGACY_MULTISCALE_CUMULATIVE_SCHEMA = (
    "litept_legacy_multiscale_delivery_cumulative/v1"
)
GRADIENT_PANEL_CUMULATIVE_SCHEMA = "litept_gradient_panel_cumulative/v1"
HIERARCHY_HEALTH_STAGES = ("enc4", "dec3", "dec2", "dec1", "dec0")
HIERARCHY_ROUTE_STAGES = {
    "g02": ("dec1", "dec2"),
    "g05": ("dec2", "dec3"),
    "g08": ("dec3", "enc4"),
}
LEGACY_MULTISCALE_CUMULATIVE_STAGES = tuple(MULTISCALE_LEVEL_NAMES)
LEGACY_MULTISCALE_INTEGER_KEYS = (
    "input_pairs",
    "valid_pairs",
    "kept_pairs",
    "input_proposals",
    "kept_proposals",
    "collapsed_positive_pairs",
    "dropped_proposals_insufficient_pure_positive",
    "dropped_proposals_no_pure_negative",
    "pure_positive_tokens",
    "pure_negative_tokens",
    "mixed_or_unknown_tokens_ignored",
    "accepted_mixed_tokens",
    "zero_negative_pairs",
    "base_negative_collisions",
    "base_negative_duplicates",
    "feature_candidate_pool",
    "feature_hard_negatives",
    "feature_candidate_collisions",
    "feature_candidate_duplicates",
)
LEGACY_MULTISCALE_FLOAT_KEYS = (
    "loss",
    "hardest_negative_ranking_accuracy",
    "cosine_gap",
    "temperature",
    "base_negatives_per_pair",
    "uniform_negatives_per_pair",
    "spatial_hard_negatives_per_pair",
    "stage_weight",
    "effective_loss_weight",
)
LEGACY_MULTISCALE_CUMULATIVE_KEYS = (
    *LEGACY_MULTISCALE_INTEGER_KEYS,
    *LEGACY_MULTISCALE_FLOAT_KEYS,
)
RANK_CUMULATIVE_HEALTH_KEYS = frozenset(
    {
        "schema_version",
        "rank",
        "through_update",
        "optimizer_update_count",
        "source_cells",
        "granularities",
        "expected_microbatches_per_update",
        "clipping",
        "token_pure_dec0",
        "hierarchy_supervision",
        "legacy_multiscale_delivery",
        "gradient_panels",
    }
)
TOKEN_CUMULATIVE_COUNT_KEYS = (
    "input_groups",
    "retained_groups",
    "input_proposals",
    "retained_proposals",
    "input_point_pairs",
    "requested_token_pairs",
    "realized_positive_pairs",
    "zero_denominator_groups",
)
HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS = (
    "mask_valid_fold_denominator",
    "mask_proposals_attempted",
    "mask_proposals_valid",
    "mask_folds_attempted",
    "mask_folds_valid",
)


def _comma_separated_nonnegative_ints(value: str) -> tuple[int, ...]:
    """Parse a sorted, unique epoch list while keeping CLI provenance explicit."""

    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integer epochs, got {value!r}"
        ) from exc
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("Epoch lists must contain non-negative values")
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("Epoch lists must be sorted and unique")
    return parsed


def _update_for_completed_epoch(
    *,
    epoch: int,
    scene_visits_per_epoch_per_rank: int,
    scenes_per_rank_per_update: int,
) -> int:
    scene_visits = int(epoch) * int(scene_visits_per_epoch_per_rank)
    logical_batch = int(scenes_per_rank_per_update)
    if logical_batch <= 0:
        raise ValueError("scenes_per_rank_per_update must be positive")
    if scene_visits % logical_batch != 0:
        raise ValueError(
            "Epoch boundary does not align with a complete logical batch: "
            f"epoch={epoch} scene_visits={scene_visits} "
            f"scenes_per_rank_per_update={logical_batch}"
        )
    return scene_visits // logical_batch


def _final_evaluation_endpoint(
    *,
    nominal_epochs: int,
    total_updates: int,
    maximum_updates: int | None,
    snapshot_label_by_update: Mapping[int, int],
    checkpoint_epochs: Collection[int],
    checkpoint_updates: Collection[int],
    checkpoint_dir: Path,
) -> tuple[int, Path]:
    """Resolve the exact retained evaluation/checkpoint at a run endpoint.

    A ``--maximum-updates`` screen normally stops between true scene-epoch
    boundaries. Its endpoint is therefore an update-aligned audit snapshot,
    not the nominal ``--epochs`` checkpoint.
    """

    nominal_epochs = int(nominal_epochs)
    total_updates = int(total_updates)
    if nominal_epochs < 0 or total_updates <= 0:
        raise ValueError("Final evaluation endpoint is invalid")
    snapshot_labels = {
        int(update): int(label)
        for update, label in snapshot_label_by_update.items()
    }
    retained_epochs = {int(value) for value in checkpoint_epochs}
    retained_updates = {int(value) for value in checkpoint_updates}

    if maximum_updates is not None:
        if int(maximum_updates) != total_updates:
            raise ValueError("Maximum-update endpoint disagrees with total updates")
        if total_updates not in snapshot_labels:
            raise ValueError(
                "Maximum-update endpoint must be an exact audit snapshot"
            )
        if total_updates not in retained_updates:
            raise ValueError(
                "Maximum-update endpoint audit snapshot must retain a checkpoint"
            )
        label = snapshot_labels[total_updates]
        return (
            label,
            checkpoint_dir / f"audit_e{label:04d}_u{total_updates:08d}.pt",
        )

    if nominal_epochs not in retained_epochs:
        raise ValueError("Final scene epoch must retain a resumable checkpoint")
    return nominal_epochs, checkpoint_dir / f"epoch_{nominal_epochs:04d}.pt"


def _updates_for_scene_visit_duration(
    *,
    epochs: int,
    scene_visits_per_epoch_per_rank: int,
    scenes_per_rank_per_update: int,
) -> int:
    """Convert a scene-epoch duration to complete updates without dropping visits."""

    if int(epochs) <= 0:
        raise ValueError("duration epochs must be positive")
    scene_visits = int(epochs) * int(scene_visits_per_epoch_per_rank)
    logical_batch = int(scenes_per_rank_per_update)
    if logical_batch <= 0:
        raise ValueError("scenes_per_rank_per_update must be positive")
    return (scene_visits + logical_batch - 1) // logical_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("scannet", "structured3d"),
        default="scannet",
        help="Dataset contract used for split validation and source auditing.",
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help=(
            "Required DDP world size. Defaults to the frozen ScanNet value "
            "when omitted."
        ),
    )
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument(
        "--input-features",
        choices=INPUT_FEATURE_CHOICES,
        default="rgbn6",
        help="Input feature contract: RGB (3) or RGB+normal (6).",
    )
    parser.add_argument(
        "--train-augmentation-profile",
        choices=TRAIN_AUGMENTATION_CHOICES,
        default="none",
        help=(
            "Training-only input augmentation. Evaluation is always "
            "deterministic and unaugmented."
        ),
    )
    parser.add_argument(
        "--coordinate-normalization-policy",
        choices=COORDINATE_NORMALIZATION_POLICIES,
        default="native",
        help=(
            "Deterministic source-frame conversion applied to points and normals "
            "before both training augmentation and unaugmented evaluation."
        ),
    )
    parser.add_argument(
        "--required-crop-mode",
        choices=("any", "sphere_50k", "full_scene"),
        default="any",
        help="Fail unless every loaded source manifest uses this crop mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        choices=(3, 18, 34, 36, 84, 168, 252, 336, 1008),
        required=True,
    )
    parser.add_argument("--backbone-lr", type=float, choices=(3e-4, 1e-3), required=True)
    parser.add_argument("--head-lr", type=float, default=3e-3)
    parser.add_argument(
        "--representative-sampling",
        choices=("first", "random"),
        default="first",
        help="Representative point policy for deterministic transfer parity.",
    )
    parser.add_argument(
        "--color-mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="RGB mean applied after RGBN6 augmentation, in [0,1] space.",
    )
    parser.add_argument(
        "--color-std",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="RGB standard deviation applied after RGBN6 augmentation.",
    )
    parser.add_argument(
        "--multiscale-supervision",
        action="store_true",
        help=(
            "Add disposable dec3/dec2/dec1 contrastive heads while retaining "
            "the final dec0 objective as the primary loss."
        ),
    )
    parser.add_argument(
        "--multiscale-loss-weight",
        type=float,
        default=MULTISCALE_AUXILIARY_WEIGHT,
        help="Maximum multiplier for the tapered auxiliary multi-scale sum.",
    )
    parser.add_argument(
        "--multiscale-warmup-epochs",
        type=int,
        default=MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
        help="Linearly ramp the auxiliary loss over this many scene epochs.",
    )
    parser.add_argument(
        "--hierarchy-supervision",
        choices=HIERARCHY_SUPERVISION_MODES,
        default="off",
        help=(
            "Opt-in V2 supervision recipe. 'legacy' must accompany "
            "--multiscale-supervision; query modes expose enc4 through dec0 "
            "and use proposal-conditioned occupancy supervision."
        ),
    )
    parser.add_argument("--hierarchy-mask-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--hierarchy-mask-stage-weights",
        default="{}",
        help=(
            "Optional JSON mapping of relative proposal-query weights for "
            "enc4/dec3/dec2/dec1; omitted stages default to 1.0."
        ),
    )
    parser.add_argument(
        "--hierarchy-mask-stage-weight-schedule",
        default="null",
        help=(
            "Optional JSON object {transition_fraction,early,late} for a "
            "deterministic hierarchy-mask stage-weight transition."
        ),
    )
    parser.add_argument(
        "--hierarchy-query-projection-policy",
        choices=HIERARCHY_QUERY_PROJECTION_POLICIES,
        default="shared_dec0",
        help=(
            "Detached dec0 query-head policy: the legacy shared head or "
            "a disposable target-stage-specific head."
        ),
    )
    parser.add_argument("--hierarchy-vicreg-loss-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-warmup-fraction", type=float, default=0.05)
    parser.add_argument("--hierarchy-variance-target", type=float, default=1.0)
    parser.add_argument("--hierarchy-covariance-weight", type=float, default=0.04)
    parser.add_argument("--hierarchy-vicreg-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--feature-hard-warmup-fraction",
        type=float,
        default=0.05,
        help="Minimum update fraction before V2 feature-hard mining may activate.",
    )
    parser.add_argument("--feature-hard-gap-threshold", type=float, default=0.05)
    parser.add_argument("--feature-hard-required-panels", type=int, default=2)
    parser.add_argument(
        "--normalization-policy",
        choices=NORMALIZATION_POLICIES,
        default="batchnorm",
        help="Backbone normalization policy; SyncBN conversion occurs after initial load.",
    )
    parser.add_argument(
        "--bn-recalibration-scenes",
        type=int,
        default=0,
        help=(
            "Number of fixed training scenes used by the retained-checkpoint BN "
            "recalibration sidecar contract. Zero preserves the legacy runner."
        ),
    )
    parser.add_argument(
        "--evaluation-epochs",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help="Optional sorted epoch labels for fixed health evaluation.",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help=(
            "Optional sorted retained epoch labels. Unaligned labels are saved as "
            "non-resumable update-aligned audit snapshots."
        ),
    )
    parser.add_argument(
        "--audit-snapshot-updates",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help=(
            "Optional update-aligned health-panel snapshots. These never alter "
            "the scene stream and may therefore carry an approximate epoch label."
        ),
    )
    parser.add_argument(
        "--audit-snapshot-labels",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help="Epoch-style labels paired one-to-one with --audit-snapshot-updates.",
    )
    parser.add_argument(
        "--checkpoint-updates",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help=(
            "Subset of audit snapshot updates whose raw state is retained as a "
            "non-resumable checkpoint."
        ),
    )
    parser.add_argument(
        "--maximum-updates",
        type=int,
        default=None,
        help=(
            "Optional exact update horizon for short funnel panels. It may stop "
            "before the nominal final scene boundary but never executes a partial "
            "logical batch."
        ),
    )
    parser.add_argument(
        "--pause-after-update",
        type=int,
        default=None,
        help=(
            "Operational V2 production pause at an exact resumable epoch "
            "boundary. Unlike --maximum-updates, this does not alter the "
            "declared optimization horizon or LR/supervision schedules."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=17)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Training/sampling seed. Fixed-evaluation and initialization seeds "
            "default to this value unless explicitly separated below."
        ),
    )
    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=None,
        help=(
            "Seed declared by the immutable initialization checkpoint. This "
            "may differ from --seed for a controlled training-seed repeat."
        ),
    )
    parser.add_argument(
        "--fixed-eval-seed",
        type=int,
        default=None,
        help=(
            "Seed declared by the immutable fixed holdout pack. This may "
            "differ from --seed for a controlled training-seed repeat."
        ),
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--sampling-profile",
        choices=tuple(sorted(SAMPLING_PROFILES)),
        default="pf_hnm_512_v1",
        help="Frozen training positive/negative sampling contract.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=TRAINING_SAMPLING_MODES,
        default="random_with_replacement",
        help=(
            "Training tuple scheduler. V2 coverage visits all three 2D "
            "granularities with frame-local, checkpoint-reproducible groups."
        ),
    )
    parser.add_argument(
        "--evaluation-sampling-profile",
        choices=tuple(sorted(SAMPLING_PROFILES)),
        default=None,
        help=(
            "Optional holdout sampling contract. Defaults to the training "
            "profile; the matched p16 arm pins pf_hnm_512_v1 so evaluation "
            "remains comparable to earlier runs."
        ),
    )
    parser.add_argument(
        "--scenes-per-rank-per-update",
        type=int,
        default=1,
        help=(
            "Number of single-scene forwards accumulated on each DDP rank "
            "before one optimizer update."
        ),
    )
    parser.add_argument(
        "--source-loading-mode",
        choices=("preload", "logical_batch"),
        default="preload",
        help=(
            "Keep every rank-local training source resident, or load only the "
            "current logical scene group from a pre-audited source archive."
        ),
    )
    parser.add_argument(
        "--source-loader-workers",
        type=int,
        default=1,
        help="Parallel source-loader threads used by logical_batch mode.",
    )
    parser.add_argument(
        "--source-cell-schedule",
        choices=SOURCE_CELL_SCHEDULE_CHOICES,
        default=SOURCE_CELL_SCHEDULE_FIXED,
        help=(
            "Per-scene source-cell ordering. The balanced-random schedule "
            "uses one reproducible permutation per complete cell cycle."
        ),
    )
    parser.add_argument(
        "--fixed-eval-pack",
        type=Path,
        default=None,
        help=(
            "Directory containing or receiving the immutable heldout tuple pack. "
            "Required for pf_hnm_768_v1."
        ),
    )
    parser.add_argument(
        "--code-manifest",
        type=Path,
        default=None,
        help=(
            "Hash manifest for the immutable code snapshot used by this run. "
            "Required for pf_hnm_768_v1."
        ),
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=None,
        help=(
            "Optional frozen dataset source-audit JSON. The ScanNet contract "
            "uses source_root/../audits/source_dataset_audit.json by default; "
            "Structured3D currently validates each scene manifest directly."
        ),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--resume-sha256",
        type=str,
        default=None,
        help="Required content digest for --resume.",
    )
    parser.add_argument(
        "--resume-into-fresh-attempt",
        action="store_true",
        help=(
            "Continue a hash-pinned checkpoint in a new output directory, "
            "preserving the parent attempt unchanged."
        ),
    )
    parser.add_argument(
        "--execution-provenance",
        type=Path,
        default=None,
        help="Frozen pre-run trust-root manifest embedded in every checkpoint.",
    )
    parser.add_argument(
        "--execution-provenance-sha256",
        type=str,
        default=None,
        help="Required SHA-256 of --execution-provenance.",
    )
    parser.add_argument(
        "--extension-start-epoch",
        type=int,
        default=None,
        help=(
            "Start a new warm-restarted cosine phase from a completed resume "
            "checkpoint at this global epoch."
        ),
    )
    parser.add_argument(
        "--extension-parent-checkpoint",
        type=Path,
        default=None,
        help=(
            "Pinned completed checkpoint that begins the extension lineage. "
            "For the first extension launch this must equal --resume; retain "
            "the same value when exactly resuming an interrupted extension."
        ),
    )
    parser.add_argument(
        "--three-d-mode",
        choices=("gt3", "g08_repeated"),
        default="gt3",
        help=(
            "Use g02/g05/g08 in their three logical 3D slots, or repeat exact "
            "g08 GT supervision in all three slots for the matched control. "
            "Forbidden when --source-family 2d."
        ),
    )
    parser.add_argument(
        "--source-family",
        choices=SOURCE_FAMILY_CHOICES,
        default="2d3d",
        help=(
            "Active source cells. 2d3d is the six-cell default; 2d trains and "
            "evaluates only UnSAM 2d/g02, 2d/g05, and 2d/g08."
        ),
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help=(
            "Explicit four-rank integration smoke: repeat one audited scene on "
            "every rank and evaluate that scene only."
        ),
    )
    parser.add_argument(
        "--smoke-manifest-list",
        type=Path,
        default=None,
        help=(
            "Text file containing at least two audited scene manifests. Every "
            "rank cycles through the listed scenes, exercising scene-changing "
            "voxelization inside one process."
        ),
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        choices=(2, 6),
        default=2,
        help=(
            "Number of multi-scene smoke epochs. Six exercises every source "
            "cell exactly once per scene and rank."
        ),
    )
    return parser.parse_args()


def _resolve_seed_contract(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve immutable-artifact seeds while preserving legacy defaults.

    The training/sampling seed is intentionally kept in ``args.seed``.  A
    controlled seed repeat can retain the exact initialization checkpoint and
    fixed holdout pack by supplying their declared seeds separately.
    """

    initialization_seed = (
        int(args.seed)
        if args.initialization_seed is None
        else int(args.initialization_seed)
    )
    fixed_eval_seed = (
        int(args.seed)
        if args.fixed_eval_seed is None
        else int(args.fixed_eval_seed)
    )
    if initialization_seed < 0 or fixed_eval_seed < 0 or int(args.seed) < 0:
        raise ValueError("Seeds must be non-negative")
    return initialization_seed, fixed_eval_seed


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temp_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        temp_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _content_addressed_report(value: Any) -> Any:
    """Drop extraction-location fields while preserving identity hashes."""

    if isinstance(value, dict):
        return {
            key: _content_addressed_report(item)
            for key, item in value.items()
            if key != "path"
            and key != "pack_dir"
            and not key.endswith("_path")
            and not key.endswith("_root")
        }
    if isinstance(value, list):
        return [_content_addressed_report(item) for item in value]
    return value


def _semantic_config_sha256(resolved: dict[str, Any]) -> str:
    """Hash every semantic field while excluding attempt-local lineage."""

    payload = _content_addressed_report(resolved)
    payload.pop("semantic_config_sha256", None)
    payload.pop("resume_parent", None)
    payload.pop("resume_into_fresh_attempt", None)
    payload.pop("lineage_parent_checkpoints", None)
    payload.pop("pause_after_update", None)
    payload.pop("execution_stop_update", None)
    payload.pop("executed_scene_visits_per_rank", None)
    execution = payload.get("execution_provenance")
    if isinstance(execution, dict):
        payload["execution_provenance"] = {
            "schema_version": execution.get("schema_version"),
            "trust_root": execution.get("trust_root"),
            "passed": execution.get("passed"),
        }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _atomic_jsonl_noclobber(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        os.link(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _gradient_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def _clip_gradient_diagnostics(
    parameters: Any,
    *,
    max_norm: float,
) -> dict[str, float]:
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    parameters = list(parameters)
    pre_clip_norm = _gradient_norm(parameters)
    clip_return_norm = float(
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    )
    post_clip_norm = _gradient_norm(parameters)
    clip_coefficient = min(1.0, max_norm / (clip_return_norm + 1.0e-6))
    return {
        "pre_clip_norm": pre_clip_norm,
        "post_clip_norm": post_clip_norm,
        "clip_return_norm": clip_return_norm,
        "clip_coefficient": clip_coefficient,
        "was_clipped": float(clip_return_norm > max_norm),
    }


def _merge_count_dicts(rows: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _json_copy(value: Any) -> Any:
    """Copy JSON telemetry without retaining mutable controller references."""

    return json.loads(json.dumps(value))


def _strict_nonnegative_counter(value: Any, *, name: str) -> int:
    """Read a JSON counter without accepting booleans or negative values."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid cumulative counter: {name}")
    return int(value)


def _strict_finite_real(value: Any, *, name: str) -> float:
    """Read a JSON metric without accepting booleans or non-finite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid cumulative metric: {name}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid cumulative metric: {name}")
    return parsed


def _strict_nonnegative_real(value: Any, *, name: str) -> float:
    parsed = _strict_finite_real(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"Invalid cumulative metric: {name}")
    return parsed


def _canonical_string_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"Invalid cumulative string set: {name}")
    parsed = tuple(value)
    if any(not isinstance(item, str) or not item for item in parsed):
        raise ValueError(f"Invalid cumulative string set: {name}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate cumulative string in {name}")
    return parsed


def _canonical_panel_schedule(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"Invalid cumulative gradient-panel schedule: {name}")
    parsed: list[int] = []
    for item in value:
        parsed.append(_strict_nonnegative_counter(item, name=f"{name}[]"))
    result = tuple(parsed)
    if result != tuple(sorted(set(result))) or any(item <= 0 for item in result):
        raise ValueError(f"Invalid cumulative gradient-panel schedule: {name}")
    return result


def _canonical_source_metadata(
    cells: Sequence[str], *, hierarchy_enabled: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered_cells = _canonical_string_sequence(cells, name="source_cells")
    granularities = tuple(
        dict.fromkeys(cell.rsplit("/", 1)[-1] for cell in ordered_cells)
    )
    if any(granularity not in HIERARCHY_ROUTE_STAGES for granularity in granularities):
        raise ValueError("Cumulative source cells contain an unknown granularity")
    if hierarchy_enabled and set(granularities) != set(HIERARCHY_ROUTE_STAGES):
        raise ValueError(
            "Hierarchy cumulative state requires exactly g02/g05/g08 granularities"
        )
    if hierarchy_enabled and tuple(ordered_cells) != tuple(V2_DEFAULT_2D_CELLS):
        raise ValueError("Hierarchy cumulative state requires canonical 2D source cells")
    if hierarchy_enabled and len(granularities) != len(HIERARCHY_ROUTE_STAGES):
        raise ValueError("Hierarchy cumulative granularities are duplicated")
    return ordered_cells, granularities


def _legacy_multiscale_stage_metrics(
    value: Any, *, stage: str
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Legacy multiscale stage {stage} is not a mapping")
    missing = set(LEGACY_MULTISCALE_CUMULATIVE_KEYS) - set(value)
    if missing:
        raise ValueError(
            f"Legacy multiscale stage {stage} is missing canonical metrics: "
            f"{sorted(missing)}"
        )
    parsed: dict[str, int | float] = {}
    for key in LEGACY_MULTISCALE_INTEGER_KEYS:
        raw = value.get(key, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Invalid legacy multiscale counter: {stage}.{key}")
        raw_float = float(raw)
        # The legacy mapper uses -1 for an unmeasured feature-candidate
        # duplicate count when feature-hard mining requested zero candidates.
        # Canonical cumulative state has no unknown sentinel: no candidate
        # pool means zero candidate duplicates; a negative value otherwise is
        # rejected below.
        if (
            key == "feature_candidate_duplicates"
            and raw_float == -1.0
            and value.get("feature_hard_negatives", 0) == 0
            and value.get("feature_candidate_pool", 0) == 0
        ):
            raw_float = 0.0
        if not math.isfinite(raw_float) or raw_float < 0.0 or not raw_float.is_integer():
            raise ValueError(f"Invalid legacy multiscale counter: {stage}.{key}")
        parsed[key] = int(raw_float)
    for key in LEGACY_MULTISCALE_FLOAT_KEYS:
        parsed[key] = _strict_finite_real(
            value.get(key, 0.0), name=f"legacy_multiscale.{stage}.{key}"
        )
    if parsed["valid_pairs"] > parsed["input_pairs"]:
        raise ValueError(f"Legacy multiscale pair denominator drift: {stage}")
    if parsed["kept_pairs"] > parsed["valid_pairs"]:
        raise ValueError(f"Legacy multiscale kept-pair drift: {stage}")
    if parsed["kept_proposals"] > parsed["input_proposals"]:
        raise ValueError(f"Legacy multiscale proposal denominator drift: {stage}")
    return parsed


def _scheduled_gradient_panel_updates(
    *,
    total_updates: int,
    scene_visits_per_epoch_per_rank: int,
    scenes_per_rank_per_update: int,
    eval_epochs: Sequence[int],
    snapshot_updates: Sequence[int],
    auxiliary_enabled: bool,
) -> list[int]:
    """Return the exact update schedule used by ``collect_global_gradient_panel``."""

    if not auxiliary_enabled:
        return []
    if int(total_updates) <= 0:
        raise ValueError("total_updates must be positive")
    # The evaluator's checkpoint fallback always expects a panel at the
    # checkpoint boundary, even when the final update is not also an eval or
    # audit boundary.
    updates = {1, int(total_updates)}
    updates.update(int(value) for value in snapshot_updates)
    for epoch in eval_epochs:
        if int(epoch) <= 0:
            continue
        panel_update = _update_for_completed_epoch(
            epoch=int(epoch),
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
        )
        if panel_update <= int(total_updates):
            updates.add(panel_update)
    if any(value <= 0 or value > int(total_updates) for value in updates):
        raise ValueError("Gradient-panel update lies outside the training horizon")
    return sorted(updates)


def _new_rank_cumulative_health(
    *,
    rank: int,
    cells: Sequence[str],
    hierarchy_enabled: bool,
    scheduled_gradient_panel_updates: Sequence[int],
    expected_microbatches_per_update: int = 1,
    legacy_multiscale_enabled: bool = False,
) -> dict[str, Any]:
    """Create rank-local, checkpointable counters with no sampled-row proxies."""

    if not isinstance(hierarchy_enabled, bool) or not isinstance(
        legacy_multiscale_enabled, bool
    ):
        raise ValueError("Cumulative-health enablement flags must be boolean")
    if hierarchy_enabled and legacy_multiscale_enabled:
        raise ValueError("Hierarchy and legacy multiscale delivery are exclusive")
    ordered_cells, granularities = _canonical_source_metadata(
        cells, hierarchy_enabled=hierarchy_enabled
    )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    expected_microbatches = _strict_nonnegative_counter(
        expected_microbatches_per_update,
        name="expected_microbatches_per_update",
    )
    if expected_microbatches <= 0:
        raise ValueError("expected_microbatches_per_update must be positive")
    schedule = _canonical_panel_schedule(
        scheduled_gradient_panel_updates, name="scheduled_updates"
    )
    hierarchy_per_granularity_stage = {
        granularity: {
            stage: {
                "visit_count": 0,
                "route_active_count": 0,
                "route_required_count": 0,
                "optimization_valid_count": 0,
                "optimization_invalid_count": 0,
                **{key: 0 for key in HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS},
            }
            for stage in HIERARCHY_ROUTE_STAGES[granularity]
        }
        for granularity in HIERARCHY_ROUTE_STAGES
    }
    legacy_stage_template = {
        key: (0 if key in LEGACY_MULTISCALE_INTEGER_KEYS else 0.0)
        for key in LEGACY_MULTISCALE_CUMULATIVE_KEYS
    }
    return {
        "schema_version": RANK_CUMULATIVE_HEALTH_SCHEMA,
        "rank": int(rank),
        "through_update": 0,
        "optimizer_update_count": 0,
        "source_cells": list(ordered_cells),
        "granularities": list(granularities),
        "expected_microbatches_per_update": expected_microbatches,
        "clipping": {
            "definition": "any_rank_was_clipped_per_optimizer_update",
            "local_rank_update_event_count": 0,
            "local_rank_clipped_update_count": 0,
            "any_rank_clipped_update_count": 0,
            "all_ranks_unclipped_update_count": 0,
        },
        "token_pure_dec0": {
            "schema_version": TOKEN_CUMULATIVE_HEALTH_SCHEMA,
            "microbatch_count": 0,
            "recorded_microbatch_count": 0,
            "optimization_valid_count": 0,
            "optimization_invalid_count": 0,
            "per_cell": {
                cell: {key: 0 for key in TOKEN_CUMULATIVE_COUNT_KEYS}
                for cell in ordered_cells
            },
        },
        "hierarchy_supervision": {
            "schema_version": HIERARCHY_CUMULATIVE_HEALTH_SCHEMA,
            "enabled": bool(hierarchy_enabled),
            "microbatch_count": 0,
            "recorded_microbatch_count": 0,
            "optimization_valid_count": 0,
            "optimization_invalid_count": 0,
            "vicreg_valid_count": 0,
            "vicreg_invalid_count": 0,
            "per_cell": {
                granularity: {
                    "visit_count": 0,
                    "optimization_valid_count": 0,
                    "optimization_invalid_count": 0,
                    "input_groups": 0,
                    "input_proposals": 0,
                    "assigned_proposals": 0,
                    **{
                        key: 0
                        for key in HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS
                    },
                }
                for granularity in granularities
            },
            "per_stage": {
                stage: {
                    "visit_count": 0,
                    **{
                        key: 0
                        for key in HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS
                    },
                    "vicreg_valid_count": 0,
                    "vicreg_invalid_count": 0,
                }
                for stage in HIERARCHY_HEALTH_STAGES
            },
            "per_granularity_stage": hierarchy_per_granularity_stage,
        },
        "legacy_multiscale_delivery": {
            "schema_version": LEGACY_MULTISCALE_CUMULATIVE_SCHEMA,
            "enabled": bool(legacy_multiscale_enabled),
            "microbatch_count": 0,
            "recorded_microbatch_count": 0,
            "optimization_valid_count": 0,
            "optimization_invalid_count": 0,
            "per_stage": {
                stage: dict(legacy_stage_template)
                for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES
            },
        },
        "gradient_panels": {
            "schema_version": GRADIENT_PANEL_CUMULATIVE_SCHEMA,
            "scheduled_updates": list(schedule),
            "panels": [],
        },
    }


def _accumulate_rank_cumulative_health_update(
    state: dict[str, Any],
    *,
    update: int,
    local_was_clipped: bool,
    clipped_rank_count: int,
    world_size: int,
    attempted_microbatch_count: int,
    optimization_valid_count: int,
    token_records: Sequence[Mapping[str, Any]],
    hierarchy_records: Sequence[Mapping[str, Any]],
    legacy_multiscale_records: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Commit one completed optimizer update into exact rank-local counters."""

    if state.get("schema_version") != RANK_CUMULATIVE_HEALTH_SCHEMA:
        raise ValueError("Rank cumulative-health schema drift")
    observed_update = _strict_nonnegative_counter(update, name="update")
    expected_update = _strict_nonnegative_counter(
        state.get("through_update"), name="through_update"
    ) + 1
    if observed_update != expected_update:
        raise ValueError(
            "Cumulative-health update sequence drift: "
            f"observed={observed_update} expected={expected_update}"
        )
    if (
        not isinstance(local_was_clipped, bool)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or isinstance(clipped_rank_count, bool)
        or not isinstance(clipped_rank_count, int)
        or not 0 <= clipped_rank_count <= world_size
    ):
        raise ValueError("Invalid clipping event types")
    attempted = _strict_nonnegative_counter(
        attempted_microbatch_count, name="attempted_microbatch_count"
    )
    valid = _strict_nonnegative_counter(
        optimization_valid_count, name="optimization_valid_count"
    )
    if attempted <= 0 or not 0 <= valid <= attempted:
        raise ValueError("Invalid cumulative microbatch validity counts")
    expected_microbatches = _strict_nonnegative_counter(
        state.get("expected_microbatches_per_update"),
        name="expected_microbatches_per_update",
    )
    if attempted != expected_microbatches:
        raise ValueError(
            "Cumulative microbatch denominator drift: "
            f"observed={attempted} expected={expected_microbatches}"
        )
    if (
        len(token_records) > attempted
        or len(hierarchy_records) > attempted
        or len(legacy_multiscale_records) > attempted
    ):
        raise ValueError("More delivery records than attempted microbatches")
    for records, name in (
        (token_records, "token"),
        (hierarchy_records, "hierarchy"),
        (legacy_multiscale_records, "legacy multiscale"),
    ):
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError(f"Malformed {name} cumulative delivery record")

    legacy = state["legacy_multiscale_delivery"]
    if not isinstance(legacy.get("enabled"), bool):
        raise ValueError("Legacy multiscale enablement is invalid")
    if legacy["enabled"]:
        if len(legacy_multiscale_records) != valid:
            raise ValueError(
                "Legacy multiscale record denominator does not match valid microbatches"
            )
    elif legacy_multiscale_records:
        raise ValueError(
            "Legacy multiscale records present while legacy delivery is disabled"
        )

    clipping = state["clipping"]
    clipping["local_rank_update_event_count"] += 1
    clipping["local_rank_clipped_update_count"] += int(bool(local_was_clipped))
    clipping["any_rank_clipped_update_count"] += int(clipped_rank_count > 0)
    clipping["all_ranks_unclipped_update_count"] += int(clipped_rank_count == 0)

    token = state["token_pure_dec0"]
    token["microbatch_count"] += attempted
    token["recorded_microbatch_count"] += len(token_records)
    token["optimization_valid_count"] += valid
    token["optimization_invalid_count"] += attempted - valid
    for record in token_records:
        for group in record.get("groups", ()):
            cell = str(group.get("cell"))
            if cell not in token["per_cell"]:
                continue
            summary = token["per_cell"][cell]
            realized_pairs = int(group.get("realized_positive_pairs", 0))
            summary["input_groups"] += 1
            summary["retained_groups"] += int(realized_pairs > 0)
            summary["input_proposals"] += int(group.get("input_proposals", 0))
            summary["retained_proposals"] += int(
                group.get("retained_proposals", 0)
            )
            summary["input_point_pairs"] += int(group.get("input_point_pairs", 0))
            summary["requested_token_pairs"] += sum(
                int(proposal.get("requested_token_pairs", 0))
                for proposal in group.get("proposals", ())
            )
            summary["realized_positive_pairs"] += realized_pairs
            summary["zero_denominator_groups"] += int(
                bool(group.get("zero_denominator", realized_pairs <= 0))
            )

    if legacy["enabled"]:
        legacy["microbatch_count"] += attempted
        legacy["recorded_microbatch_count"] += len(legacy_multiscale_records)
        legacy["optimization_valid_count"] += valid
        legacy["optimization_invalid_count"] += attempted - valid
        for record in legacy_multiscale_records:
            levels = record.get("levels")
            if not isinstance(levels, Mapping) or set(levels) != set(
                LEGACY_MULTISCALE_CUMULATIVE_STAGES
            ):
                raise ValueError("Legacy multiscale stage set drift")
            for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES:
                parsed = _legacy_multiscale_stage_metrics(
                    levels[stage], stage=stage
                )
                stage_summary = legacy["per_stage"][stage]
                for key in LEGACY_MULTISCALE_INTEGER_KEYS:
                    stage_summary[key] += int(parsed[key])
                for key in LEGACY_MULTISCALE_FLOAT_KEYS:
                    stage_summary[key] += float(parsed[key])

    hierarchy = state["hierarchy_supervision"]
    if bool(hierarchy["enabled"]):
        hierarchy["microbatch_count"] += attempted
        hierarchy["recorded_microbatch_count"] += len(hierarchy_records)
        hierarchy["optimization_valid_count"] += valid
        hierarchy["optimization_invalid_count"] += attempted - valid
        vicreg_valid_records = 0
        per_cell_optimization_valid_counts = {
            granularity: 0 for granularity in hierarchy["per_cell"]
        }
        per_edge_optimization_valid_counts = {
            (granularity, stage): 0
            for granularity, stages in HIERARCHY_ROUTE_STAGES.items()
            for stage in stages
        }
        stage_vicreg_valid_counts = {
            stage: 0 for stage in HIERARCHY_HEALTH_STAGES
        }
        for record in hierarchy_records:
            if not isinstance(record, Mapping):
                continue
            stage_records = record.get("stages", {})
            if not isinstance(stage_records, Mapping):
                stage_records = {}
            record_vicreg_flag = record.get("vicreg_all_stages_valid")
            if isinstance(record_vicreg_flag, bool):
                record_vicreg_valid = record_vicreg_flag
            else:
                record_vicreg_valid = all(
                    isinstance(stage_record, Mapping)
                    and isinstance(stage_record.get("vicreg"), Mapping)
                    and bool(stage_record["vicreg"].get("valid", False))
                    for stage in HIERARCHY_HEALTH_STAGES
                    for stage_record in (stage_records.get(stage),)
                )
            vicreg_valid_records += int(record_vicreg_valid)
            route_records = record.get("route_stage_validity", {})
            if not isinstance(route_records, Mapping) or set(route_records) != set(
                HIERARCHY_ROUTE_STAGES
            ):
                raise ValueError("Hierarchy route granularity set drift")
            for granularity, cell_summary in hierarchy["per_cell"].items():
                route = route_records.get(granularity)
                if not isinstance(route, Mapping):
                    raise ValueError(
                        f"Hierarchy route record missing for {granularity}"
                    )
                route_valid = bool(route.get("optimization_valid", False))
                per_cell_optimization_valid_counts[granularity] += int(
                    route_valid
                )
                cell_summary["input_groups"] += int(
                    route.get("input_group_count", 0)
                )
                cell_summary["input_proposals"] += int(
                    route.get("input_proposal_count", 0)
                )
                cell_summary["assigned_proposals"] += int(
                    route.get("assigned_proposal_count", 0)
                )
                route_stages = route.get("stages", {})
                expected_route_stages = HIERARCHY_ROUTE_STAGES[granularity]
                if not isinstance(route_stages, Mapping) or set(route_stages) != set(
                    expected_route_stages
                ):
                    raise ValueError(
                        f"Hierarchy route stage set drift for {granularity}"
                    )
                for stage in expected_route_stages:
                    stage_route = route_stages[stage]
                    if not isinstance(stage_route, Mapping):
                        raise ValueError(
                            f"Hierarchy route stage record missing for "
                            f"{granularity}/{stage}"
                        )
                    cell_summary["mask_valid_fold_denominator"] += int(
                        stage_route.get("loss_denominator_valid_folds", 0)
                    )
                    for source, target in (
                        ("proposals_attempted", "mask_proposals_attempted"),
                        ("proposals_valid", "mask_proposals_valid"),
                        ("folds_attempted", "mask_folds_attempted"),
                        ("folds_valid", "mask_folds_valid"),
                    ):
                        cell_summary[target] += int(stage_route.get(source, 0))
                    edge_summary = hierarchy["per_granularity_stage"][granularity][
                        stage
                    ]
                    edge_summary["route_active_count"] += int(
                        bool(stage_route.get("route_active", False))
                    )
                    edge_summary["route_required_count"] += int(
                        bool(stage_route.get("route_required", False))
                    )
                    edge_valid = bool(stage_route.get("optimization_valid", False))
                    per_edge_optimization_valid_counts[(granularity, stage)] += int(
                        edge_valid
                    )
                    edge_summary["mask_valid_fold_denominator"] += int(
                        stage_route.get("loss_denominator_valid_folds", 0)
                    )
                    for source, target in (
                        ("proposals_attempted", "mask_proposals_attempted"),
                        ("proposals_valid", "mask_proposals_valid"),
                        ("folds_attempted", "mask_folds_attempted"),
                        ("folds_valid", "mask_folds_valid"),
                    ):
                        edge_summary[target] += int(stage_route.get(source, 0))
            for stage, stage_summary in hierarchy["per_stage"].items():
                stage_record = stage_records.get(stage, {})
                if not isinstance(stage_record, Mapping):
                    stage_record = {}
                mask = stage_record.get("mask", {})
                if not isinstance(mask, Mapping):
                    mask = {}
                vicreg = stage_record.get("vicreg", {})
                if not isinstance(vicreg, Mapping):
                    vicreg = {}
                stage_summary["mask_valid_fold_denominator"] += int(
                    mask.get("loss_denominator_valid_folds", 0)
                )
                for source, target in (
                    ("proposals_attempted", "mask_proposals_attempted"),
                    ("proposals_valid", "mask_proposals_valid"),
                    ("folds_attempted", "mask_folds_attempted"),
                    ("folds_valid", "mask_folds_valid"),
                ):
                    stage_summary[target] += int(mask.get(source, 0))
                stage_vicreg_valid_counts[stage] += int(
                    bool(vicreg.get("valid", False))
                )
        for (granularity, stage), valid_count in (
            per_edge_optimization_valid_counts.items()
        ):
            edge_summary = hierarchy["per_granularity_stage"][granularity][stage]
            edge_summary["visit_count"] += attempted
            edge_summary["optimization_valid_count"] += valid_count
            edge_summary["optimization_invalid_count"] += attempted - valid_count
        hierarchy["vicreg_valid_count"] += vicreg_valid_records
        hierarchy["vicreg_invalid_count"] += attempted - vicreg_valid_records
        for granularity, cell_summary in hierarchy["per_cell"].items():
            optimization_valid_count = per_cell_optimization_valid_counts[
                granularity
            ]
            cell_summary["optimization_valid_count"] += optimization_valid_count
            cell_summary["visit_count"] += attempted
            cell_summary["optimization_invalid_count"] += (
                attempted - optimization_valid_count
            )
        for stage, stage_summary in hierarchy["per_stage"].items():
            stage_summary["visit_count"] += attempted
            stage_summary["vicreg_valid_count"] += stage_vicreg_valid_counts[stage]
            stage_summary["vicreg_invalid_count"] += (
                attempted - stage_vicreg_valid_counts[stage]
            )
    elif hierarchy_records:
        raise ValueError("Hierarchy records present while hierarchy health is disabled")

    state["optimizer_update_count"] += 1
    state["through_update"] = observed_update


def _gradient_objective_stage_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count finite/nonzero stage gradients instead of averaging them away."""

    objectives = sorted(
        {
            str(objective)
            for record in records
            for objective in record.get("objectives", {})
        }
    )
    output: dict[str, Any] = {
        "schema_version": "litept_gradient_objective_stage_counts/v1",
        "record_count": len(records),
        "objectives": {},
    }
    for objective in objectives:
        objective_records = [
            record.get("objectives", {}).get(objective, {})
            for record in records
            if objective in record.get("objectives", {})
        ]
        stages = sorted(
            {
                str(stage)
                for objective_record in objective_records
                for stage in objective_record.get("stages", {})
            }
        )
        stage_counts: dict[str, Any] = {}
        for stage in stages:
            metrics_rows = [
                objective_record.get("stages", {}).get(stage)
                for objective_record in objective_records
            ]
            metrics_rows = [
                metrics for metrics in metrics_rows if isinstance(metrics, Mapping)
            ]
            finite_count = 0
            nonzero_count = 0
            for metrics in metrics_rows:
                norm = metrics.get("grad_norm")
                norm_finite = isinstance(norm, (int, float)) and math.isfinite(
                    float(norm)
                )
                finite = bool(metrics.get("finite", False)) and norm_finite
                finite_count += int(finite)
                nonzero_count += int(finite and float(norm) > 0.0)
            stage_counts[stage] = {
                "observation_count": len(metrics_rows),
                "finite_count": finite_count,
                "nonzero_count": nonzero_count,
                "cosine_valid_count": sum(
                    int(bool(metrics.get("cosine_valid", False)))
                    for metrics in metrics_rows
                ),
                "norm_ratio_valid_count": sum(
                    int(bool(metrics.get("norm_ratio_valid", False)))
                    for metrics in metrics_rows
                ),
            }
        output["objectives"][objective] = {
            "record_count": len(objective_records),
            "finite_record_count": sum(
                int(
                    bool(objective_record.get("stages"))
                    and all(
                        bool(metrics.get("finite", False))
                        and isinstance(metrics.get("grad_norm"), (int, float))
                        and math.isfinite(float(metrics["grad_norm"]))
                        for metrics in objective_record.get("stages", {}).values()
                    )
                )
                for objective_record in objective_records
            ),
            "nonzero_record_count": sum(
                int(
                    any(
                        bool(metrics.get("finite", False))
                        and isinstance(metrics.get("grad_norm"), (int, float))
                        and math.isfinite(float(metrics["grad_norm"]))
                        and float(metrics["grad_norm"]) > 0.0
                        for metrics in objective_record.get("stages", {}).values()
                    )
                )
                for objective_record in objective_records
            ),
            "stages": stage_counts,
        }
    return output


def _append_rank_gradient_panel(
    state: dict[str, Any], panel: Mapping[str, Any]
) -> None:
    if not isinstance(panel, Mapping):
        raise ValueError("Incomplete cumulative gradient-panel record")
    update = _strict_nonnegative_counter(
        panel.get("update"), name="gradient_panel.update"
    )
    gradient_panels = state["gradient_panels"]
    if update not in set(gradient_panels["scheduled_updates"]):
        raise ValueError(f"Unexpected gradient panel update {update}")
    if any(int(row.get("update", -1)) == update for row in gradient_panels["panels"]):
        raise ValueError(f"Duplicate gradient panel update {update}")
    required = {
        "kind",
        "update",
        "global_shared_gradient",
        "controller",
        "conflict_threshold",
        "objective_stage_counts",
    }
    if panel.get("kind") != "gradient_health" or not required.issubset(panel):
        raise ValueError("Incomplete cumulative gradient-panel record")
    if not isinstance(panel.get("global_shared_gradient"), Mapping):
        raise ValueError("Invalid cumulative gradient-panel gradient report")
    if not isinstance(panel.get("controller"), Mapping):
        raise ValueError("Invalid cumulative gradient-panel controller")
    if not isinstance(panel.get("objective_stage_counts"), Mapping):
        raise ValueError("Invalid cumulative gradient-panel stage counts")
    _strict_finite_real(
        panel.get("conflict_threshold"), name="gradient_panel.conflict_threshold"
    )
    gradient_panels["panels"].append(_json_copy(dict(panel)))


def _merge_rank_cumulative_health(
    rank_states: Sequence[Mapping[str, Any]],
    *,
    through_update: int,
    world_size: int,
    gradient_conflict_controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge exact per-rank state into the checkpoint-native health record."""

    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or isinstance(through_update, bool)
        or not isinstance(through_update, int)
        or through_update < 0
    ):
        raise ValueError("Invalid cumulative-health merge boundary")
    if not isinstance(gradient_conflict_controller, Mapping):
        raise ValueError("Gradient conflict-controller state is not a mapping")
    if len(rank_states) != int(world_size):
        raise ValueError("Incomplete rank cumulative-health state")
    if any(not isinstance(row, Mapping) for row in rank_states):
        raise ValueError("Incomplete rank cumulative-health state")
    if any(set(row) != RANK_CUMULATIVE_HEALTH_KEYS for row in rank_states):
        raise ValueError("Rank cumulative-health canonical key set drift")
    try:
        ordered = sorted(rank_states, key=lambda row: int(row.get("rank", -1)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cumulative-health rank identities are invalid") from exc
    if any(
        isinstance(row.get("rank"), bool) or not isinstance(row.get("rank"), int)
        for row in ordered
    ) or [int(row.get("rank", -1)) for row in ordered] != list(range(world_size)):
        raise ValueError("Cumulative-health rank identities are incomplete")
    for row in ordered:
        try:
            row_through_update = _strict_nonnegative_counter(
                row.get("through_update"), name="through_update"
            )
        except ValueError as exc:
            raise ValueError("Rank cumulative-health boundary drift") from exc
        if (
            row.get("schema_version") != RANK_CUMULATIVE_HEALTH_SCHEMA
            or row_through_update != int(through_update)
        ):
            raise ValueError("Rank cumulative-health boundary drift")
    try:
        optimizer_counts = {
            _strict_nonnegative_counter(
                row["optimizer_update_count"],
                name="optimizer_update_count",
            )
            for row in ordered
        }
    except (KeyError, ValueError) as exc:
        raise ValueError("Cumulative optimizer-update count drift") from exc
    if optimizer_counts != {int(through_update)}:
        raise ValueError("Cumulative optimizer-update count drift")
    metadata: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []
    for row in ordered:
        hierarchy_state_for_metadata = row.get("hierarchy_supervision")
        if not isinstance(hierarchy_state_for_metadata, Mapping):
            raise ValueError("Rank hierarchy-delivery state is not a mapping")
        hierarchy_enabled_for_metadata = hierarchy_state_for_metadata.get(
            "enabled"
        )
        if not isinstance(hierarchy_enabled_for_metadata, bool):
            raise ValueError("Rank hierarchy-delivery enablement is invalid")
        try:
            source_cells, granularities = _canonical_source_metadata(
                row.get("source_cells"),
                hierarchy_enabled=hierarchy_enabled_for_metadata,
            )
            expected_microbatches = _strict_nonnegative_counter(
                row.get("expected_microbatches_per_update"),
                name="expected_microbatches_per_update",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Rank cumulative-health source metadata is invalid") from exc
        if expected_microbatches <= 0:
            raise ValueError("Rank cumulative-health microbatch denominator is invalid")
        if list(source_cells) != list(row.get("source_cells")) or list(
            granularities
        ) != list(row.get("granularities")):
            raise ValueError("Rank cumulative-health source metadata order drift")
        metadata.append((source_cells, granularities, expected_microbatches))
    if len(set(metadata)) != 1:
        raise ValueError("Ranks disagree on cumulative source metadata")
    source_cells, granularities, expected_microbatches_per_update = metadata[0]
    expected_rank_microbatches = int(through_update) * expected_microbatches_per_update
    schedule_values: list[tuple[int, ...]] = []
    for row in ordered:
        panel_state = row.get("gradient_panels")
        if not isinstance(panel_state, Mapping):
            raise ValueError("Rank gradient-panel state is missing")
        if set(panel_state) != {"schema_version", "scheduled_updates", "panels"}:
            raise ValueError("Rank gradient-panel canonical key set drift")
        if panel_state.get("schema_version") != GRADIENT_PANEL_CUMULATIVE_SCHEMA:
            raise ValueError("Rank gradient-panel schema drift")
        try:
            schedule_values.append(
                _canonical_panel_schedule(
                    panel_state.get("scheduled_updates"),
                    name="scheduled_updates",
                )
            )
        except ValueError as exc:
            raise ValueError("Rank gradient-panel schedule is invalid") from exc
    schedules = set(schedule_values)
    if len(schedules) != 1:
        raise ValueError("Ranks disagree on the gradient-panel schedule")
    scheduled_updates = list(next(iter(schedules)))

    clipping_states: list[Mapping[str, Any]] = []
    for row in ordered:
        clipping_state = row.get("clipping")
        if not isinstance(clipping_state, Mapping):
            raise ValueError("Rank clipping state is missing")
        if set(clipping_state) != {
            "definition",
            "local_rank_update_event_count",
            "local_rank_clipped_update_count",
            "any_rank_clipped_update_count",
            "all_ranks_unclipped_update_count",
        }:
            raise ValueError("Rank clipping canonical key set drift")
        if clipping_state.get("definition") != (
            "any_rank_was_clipped_per_optimizer_update"
        ):
            raise ValueError("Rank clipping definition drift")
        clipping_states.append(clipping_state)
    try:
        any_rank_counts = {
            _strict_nonnegative_counter(
                state["any_rank_clipped_update_count"],
                name="clipping.any_rank_clipped_update_count",
            )
            for state in clipping_states
        }
        all_unclipped_counts = {
            _strict_nonnegative_counter(
                state["all_ranks_unclipped_update_count"],
                name="clipping.all_ranks_unclipped_update_count",
            )
            for state in clipping_states
        }
    except (KeyError, ValueError) as exc:
        raise ValueError("Rank clipping counters are invalid") from exc
    if len(any_rank_counts) != 1 or len(all_unclipped_counts) != 1:
        raise ValueError("Ranks disagree on global clipping events")
    clipped_update_count = next(iter(any_rank_counts))
    unclipped_update_count = next(iter(all_unclipped_counts))
    if (
        clipped_update_count > int(through_update)
        or unclipped_update_count > int(through_update)
    ):
        raise ValueError("Cumulative clipping counts are invalid")
    if clipped_update_count + unclipped_update_count != int(through_update):
        raise ValueError("Cumulative clipping denominator drift")
    try:
        per_rank_clipped = [
            _strict_nonnegative_counter(
                state["local_rank_clipped_update_count"],
                name="clipping.local_rank_clipped_update_count",
            )
            for state in clipping_states
        ]
        per_rank_events = [
            _strict_nonnegative_counter(
                state["local_rank_update_event_count"],
                name="clipping.local_rank_update_event_count",
            )
            for state in clipping_states
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError("Rank clipping counters are invalid") from exc
    if any(
        events != int(through_update)
        or clipped < 0
        or clipped > events
        for events, clipped in zip(per_rank_events, per_rank_clipped)
    ):
        raise ValueError("Rank clipping event denominator drift")
    if sum(per_rank_clipped) < clipped_update_count:
        raise ValueError("Rank clipping events undercount any-rank events")

    token_states: list[Mapping[str, Any]] = []
    token_cell_sets: list[frozenset[str]] = []
    for rank, row in enumerate(ordered):
        token_state = row.get("token_pure_dec0")
        if not isinstance(token_state, Mapping):
            raise ValueError(f"Rank {rank} token-delivery state is missing")
        if set(token_state) != {
            "schema_version",
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "per_cell",
        }:
            raise ValueError("Rank token-delivery canonical key set drift")
        if token_state.get("schema_version") != TOKEN_CUMULATIVE_HEALTH_SCHEMA:
            raise ValueError("Rank token-delivery schema drift")
        try:
            microbatch_count = _strict_nonnegative_counter(
                token_state.get("microbatch_count"),
                name="token.microbatch_count",
            )
            recorded_microbatch_count = _strict_nonnegative_counter(
                token_state.get("recorded_microbatch_count"),
                name="token.recorded_microbatch_count",
            )
            optimization_valid_count = _strict_nonnegative_counter(
                token_state.get("optimization_valid_count"),
                name="token.optimization_valid_count",
            )
            optimization_invalid_count = _strict_nonnegative_counter(
                token_state.get("optimization_invalid_count"),
                name="token.optimization_invalid_count",
            )
        except ValueError as exc:
            raise ValueError("Rank token-delivery counters are invalid") from exc
        if (
            recorded_microbatch_count > microbatch_count
            or optimization_valid_count + optimization_invalid_count
            != microbatch_count
            or microbatch_count != expected_rank_microbatches
        ):
            raise ValueError("Rank token-delivery count partition drift")
        per_cell = token_state.get("per_cell")
        if not isinstance(per_cell, Mapping) or tuple(per_cell) != source_cells:
            raise ValueError("Rank token-delivery cell set drift")
        token_cell_sets.append(frozenset(str(cell) for cell in per_cell))
        for cell in per_cell:
            cell_state = per_cell.get(cell)
            if not isinstance(cell_state, Mapping):
                raise ValueError("Rank token-delivery cell state is missing")
            if set(cell_state) != set(TOKEN_CUMULATIVE_COUNT_KEYS):
                raise ValueError("Rank token-delivery counter key set drift")
            try:
                cell_counts = {
                    key: _strict_nonnegative_counter(
                        cell_state.get(key), name=f"token.{cell}.{key}"
                    )
                    for key in TOKEN_CUMULATIVE_COUNT_KEYS
                }
            except ValueError as exc:
                raise ValueError("Rank token-delivery cell counters are invalid") from exc
            if (
                cell_counts["retained_groups"] > cell_counts["input_groups"]
                or cell_counts["retained_proposals"]
                > cell_counts["input_proposals"]
                or cell_counts["realized_positive_pairs"]
                > cell_counts["requested_token_pairs"]
            ):
                raise ValueError("Rank token-delivery cell counter order drift")
        token_states.append(token_state)

    if len(set(token_cell_sets)) != 1:
        raise ValueError("Ranks disagree on token-delivery cell set")
    token_template = token_states[0]
    token_cells = tuple(token_template["per_cell"])
    token = {
        "schema_version": TOKEN_CUMULATIVE_HEALTH_SCHEMA,
        "expected_microbatch_count": expected_rank_microbatches * int(world_size),
        **{
            key: sum(int(state[key]) for state in token_states)
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
            )
        },
        "per_cell": {
            cell: {
                key: sum(
                    int(state["per_cell"][cell][key]) for state in token_states
                )
                for key in TOKEN_CUMULATIVE_COUNT_KEYS
            }
            for cell in token_cells
        },
    }
    token["optimization_valid_fraction"] = int(
        token["optimization_valid_count"]
    ) / max(int(token["microbatch_count"]), 1)
    for cell_summary in token["per_cell"].values():
        cell_summary["group_survival_fraction"] = int(
            cell_summary["retained_groups"]
        ) / max(int(cell_summary["input_groups"]), 1)
        cell_summary["proposal_survival_fraction"] = int(
            cell_summary["retained_proposals"]
        ) / max(int(cell_summary["input_proposals"]), 1)
        cell_summary["realized_pair_yield"] = int(
            cell_summary["realized_positive_pairs"]
        ) / max(int(cell_summary["requested_token_pairs"]), 1)

    hierarchy_states: list[Mapping[str, Any]] = []
    hierarchy_cell_sets: list[frozenset[str]] = []
    hierarchy_stage_sets: list[frozenset[str]] = []
    for rank, row in enumerate(ordered):
        hierarchy_state = row.get("hierarchy_supervision")
        if not isinstance(hierarchy_state, Mapping):
            raise ValueError(f"Rank {rank} hierarchy-delivery state is missing")
        if hierarchy_state.get("schema_version") != HIERARCHY_CUMULATIVE_HEALTH_SCHEMA:
            raise ValueError("Rank hierarchy-delivery schema drift")
        if set(hierarchy_state) != {
            "schema_version",
            "enabled",
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "vicreg_valid_count",
            "vicreg_invalid_count",
            "per_cell",
            "per_stage",
            "per_granularity_stage",
        }:
            raise ValueError("Rank hierarchy-delivery canonical key set drift")
        if not isinstance(hierarchy_state.get("enabled"), bool):
            raise ValueError("Rank hierarchy-delivery enablement is invalid")
        for key in (
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "vicreg_valid_count",
            "vicreg_invalid_count",
        ):
            _strict_nonnegative_counter(
                hierarchy_state.get(key), name=f"hierarchy.{key}"
            )
        if (
            int(hierarchy_state["recorded_microbatch_count"])
            > int(hierarchy_state["microbatch_count"])
            or int(hierarchy_state["optimization_valid_count"])
            + int(hierarchy_state["optimization_invalid_count"])
            != int(hierarchy_state["microbatch_count"])
            or int(hierarchy_state["vicreg_valid_count"])
            + int(hierarchy_state["vicreg_invalid_count"])
            != int(hierarchy_state["microbatch_count"])
            or (
                bool(hierarchy_state["enabled"])
                and int(hierarchy_state["microbatch_count"])
                != expected_rank_microbatches
            )
            or (
                not bool(hierarchy_state["enabled"])
                and any(
                    int(hierarchy_state[key]) != 0
                    for key in (
                        "microbatch_count",
                        "recorded_microbatch_count",
                        "optimization_valid_count",
                        "optimization_invalid_count",
                        "vicreg_valid_count",
                        "vicreg_invalid_count",
                    )
                )
            )
        ):
            raise ValueError("Rank hierarchy-delivery count partition drift")
        per_cell = hierarchy_state.get("per_cell")
        per_stage = hierarchy_state.get("per_stage")
        per_granularity_stage = hierarchy_state.get("per_granularity_stage")
        if not isinstance(per_cell, Mapping) or tuple(per_cell) != granularities:
            raise ValueError("Rank hierarchy-delivery cell set drift")
        if not isinstance(per_stage, Mapping) or tuple(per_stage) != HIERARCHY_HEALTH_STAGES:
            raise ValueError("Rank hierarchy-delivery stage set drift")
        if not isinstance(per_granularity_stage, Mapping) or tuple(
            per_granularity_stage
        ) != tuple(HIERARCHY_ROUTE_STAGES):
            raise ValueError("Rank hierarchy-delivery route set drift")
        hierarchy_cell_sets.append(frozenset(str(cell) for cell in per_cell))
        hierarchy_stage_sets.append(frozenset(str(stage) for stage in per_stage))
        for cell, cell_state in per_cell.items():
            if not isinstance(cell_state, Mapping):
                raise ValueError("Rank hierarchy-delivery cell state is missing")
            if set(cell_state) != {
                "visit_count",
                "optimization_valid_count",
                "optimization_invalid_count",
                "input_groups",
                "input_proposals",
                "assigned_proposals",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
            }:
                raise ValueError("Rank hierarchy-delivery cell key set drift")
            for key in (
                "visit_count",
                "optimization_valid_count",
                "optimization_invalid_count",
                "input_groups",
                "input_proposals",
                "assigned_proposals",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
            ):
                _strict_nonnegative_counter(
                    cell_state.get(key), name=f"hierarchy.{cell}.{key}"
                )
            if (
                bool(hierarchy_state["enabled"])
                and int(cell_state["visit_count"]) != expected_rank_microbatches
            ):
                raise ValueError("Rank hierarchy-delivery cell visit denominator drift")
            if (
                int(cell_state["optimization_valid_count"])
                + int(cell_state["optimization_invalid_count"])
                != int(cell_state["visit_count"])
                or int(cell_state["mask_proposals_valid"])
                > int(cell_state["mask_proposals_attempted"])
                or int(cell_state["mask_folds_valid"])
                > int(cell_state["mask_folds_attempted"])
                or int(cell_state["mask_valid_fold_denominator"])
                > int(cell_state["mask_folds_valid"])
            ):
                raise ValueError("Rank hierarchy-delivery cell counter drift")
        for granularity in HIERARCHY_ROUTE_STAGES:
            edge_states = per_granularity_stage[granularity]
            expected_edges = HIERARCHY_ROUTE_STAGES[granularity]
            if not isinstance(edge_states, Mapping) or tuple(edge_states) != expected_edges:
                raise ValueError(
                    f"Rank hierarchy-delivery route stage set drift for {granularity}"
                )
            for stage in expected_edges:
                edge_state = edge_states[stage]
                if not isinstance(edge_state, Mapping):
                    raise ValueError("Rank hierarchy-delivery route state is missing")
                for key in (
                    "visit_count",
                    "route_active_count",
                    "route_required_count",
                    "optimization_valid_count",
                    "optimization_invalid_count",
                    *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
                ):
                    _strict_nonnegative_counter(
                        edge_state.get(key),
                        name=f"hierarchy.{granularity}.{stage}.{key}",
                    )
                if (
                    bool(hierarchy_state["enabled"])
                    and int(edge_state["visit_count"]) != expected_rank_microbatches
                ):
                    raise ValueError(
                        "Rank hierarchy-delivery route visit denominator drift"
                    )
                if (
                    int(edge_state["optimization_valid_count"])
                    + int(edge_state["optimization_invalid_count"])
                    != int(edge_state["visit_count"])
                    or int(edge_state["route_active_count"])
                    > int(edge_state["visit_count"])
                    or int(edge_state["route_required_count"])
                    > int(edge_state["visit_count"])
                    or int(edge_state["mask_proposals_valid"])
                    > int(edge_state["mask_proposals_attempted"])
                    or int(edge_state["mask_folds_valid"])
                    > int(edge_state["mask_folds_attempted"])
                    or int(edge_state["mask_valid_fold_denominator"])
                    > int(edge_state["mask_folds_valid"])
                ):
                    raise ValueError("Rank hierarchy-delivery route counter drift")
        for stage, stage_state in per_stage.items():
            if not isinstance(stage_state, Mapping):
                raise ValueError("Rank hierarchy-delivery stage state is missing")
            if set(stage_state) != {
                "visit_count",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
                "vicreg_valid_count",
                "vicreg_invalid_count",
            }:
                raise ValueError("Rank hierarchy-delivery stage key set drift")
            for key in (
                "visit_count",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
                "vicreg_valid_count",
                "vicreg_invalid_count",
            ):
                _strict_nonnegative_counter(
                    stage_state.get(key), name=f"hierarchy.{stage}.{key}"
                )
            if (
                bool(hierarchy_state["enabled"])
                and int(stage_state["visit_count"]) != expected_rank_microbatches
            ):
                raise ValueError(
                    "Rank hierarchy-delivery stage visit denominator drift"
                )
            if (
                int(stage_state["vicreg_valid_count"])
                + int(stage_state["vicreg_invalid_count"])
                != int(stage_state["visit_count"])
                or int(stage_state["mask_proposals_valid"])
                > int(stage_state["mask_proposals_attempted"])
                or int(stage_state["mask_folds_valid"])
                > int(stage_state["mask_folds_attempted"])
                or int(stage_state["mask_valid_fold_denominator"])
                > int(stage_state["mask_folds_valid"])
            ):
                raise ValueError("Rank hierarchy-delivery stage counter drift")
        hierarchy_states.append(hierarchy_state)

    if len(set(hierarchy_cell_sets)) != 1:
        raise ValueError("Ranks disagree on hierarchy-delivery cell set")
    if len(set(hierarchy_stage_sets)) != 1:
        raise ValueError("Ranks disagree on hierarchy-delivery stage set")
    hierarchy_template = hierarchy_states[0]
    hierarchy_enabled = bool(hierarchy_template["enabled"])
    if any(bool(state["enabled"]) != hierarchy_enabled for state in hierarchy_states):
        raise ValueError("Ranks disagree on hierarchy-health enablement")
    hierarchy = {
        "schema_version": HIERARCHY_CUMULATIVE_HEALTH_SCHEMA,
        "enabled": hierarchy_enabled,
        "expected_microbatch_count": expected_rank_microbatches * int(world_size),
        **{
            key: sum(int(state[key]) for state in hierarchy_states)
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
                "vicreg_valid_count",
                "vicreg_invalid_count",
            )
        },
        "per_cell": {
            cell: {
                key: sum(
                    int(state["per_cell"][cell][key])
                    for state in hierarchy_states
                )
                for key in cell_summary
            }
            for cell, cell_summary in hierarchy_template["per_cell"].items()
        },
        "per_stage": {
            stage: {
                key: sum(
                    int(state["per_stage"][stage][key])
                    for state in hierarchy_states
                )
                for key in stage_summary
            }
            for stage, stage_summary in hierarchy_template["per_stage"].items()
        },
        "per_granularity_stage": {
            granularity: {
                stage: {
                    key: sum(
                        int(
                            state["per_granularity_stage"][granularity][stage][key]
                        )
                        for state in hierarchy_states
                    )
                    for key in hierarchy_template["per_granularity_stage"][
                        granularity
                    ][stage]
                }
                for stage in HIERARCHY_ROUTE_STAGES[granularity]
            }
            for granularity in HIERARCHY_ROUTE_STAGES
        },
    }
    hierarchy["optimization_valid_fraction"] = int(
        hierarchy["optimization_valid_count"]
    ) / max(int(hierarchy["microbatch_count"]), 1)
    hierarchy["vicreg_valid_fraction"] = int(
        hierarchy["vicreg_valid_count"]
    ) / max(int(hierarchy["microbatch_count"]), 1)
    for cell_summary in hierarchy["per_cell"].values():
        cell_summary["optimization_valid_fraction"] = int(
            cell_summary["optimization_valid_count"]
        ) / max(int(cell_summary["visit_count"]), 1)
        cell_summary["mask_proposal_survival_fraction"] = int(
            cell_summary["mask_proposals_valid"]
        ) / max(int(cell_summary["mask_proposals_attempted"]), 1)
        cell_summary["mask_fold_survival_fraction"] = int(
            cell_summary["mask_folds_valid"]
        ) / max(int(cell_summary["mask_folds_attempted"]), 1)
    for stage_summary in hierarchy["per_stage"].values():
        stage_summary["mask_proposal_survival_fraction"] = int(
            stage_summary["mask_proposals_valid"]
        ) / max(int(stage_summary["mask_proposals_attempted"]), 1)
        stage_summary["mask_fold_survival_fraction"] = int(
            stage_summary["mask_folds_valid"]
        ) / max(int(stage_summary["mask_folds_attempted"]), 1)
        stage_summary["vicreg_valid_fraction"] = int(
            stage_summary["vicreg_valid_count"]
        ) / max(int(stage_summary["visit_count"]), 1)
    for edge_stages in hierarchy["per_granularity_stage"].values():
        for edge_summary in edge_stages.values():
            edge_summary["optimization_valid_fraction"] = int(
                edge_summary["optimization_valid_count"]
            ) / max(int(edge_summary["visit_count"]), 1)
            edge_summary["mask_proposal_survival_fraction"] = int(
                edge_summary["mask_proposals_valid"]
            ) / max(int(edge_summary["mask_proposals_attempted"]), 1)
            edge_summary["mask_fold_survival_fraction"] = int(
                edge_summary["mask_folds_valid"]
            ) / max(int(edge_summary["mask_folds_attempted"]), 1)

    legacy_states: list[Mapping[str, Any]] = []
    legacy_enabled_values: list[bool] = []
    for rank, row in enumerate(ordered):
        legacy_state = row.get("legacy_multiscale_delivery")
        if not isinstance(legacy_state, Mapping):
            raise ValueError(f"Rank {rank} legacy multiscale state is missing")
        if set(legacy_state) != {
            "schema_version",
            "enabled",
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "per_stage",
        }:
            raise ValueError("Rank legacy multiscale canonical key set drift")
        if legacy_state.get("schema_version") != LEGACY_MULTISCALE_CUMULATIVE_SCHEMA:
            raise ValueError("Rank legacy multiscale schema drift")
        enabled = legacy_state.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("Rank legacy multiscale enablement is invalid")
        legacy_enabled_values.append(enabled)
        counters = {
            key: _strict_nonnegative_counter(
                legacy_state.get(key), name=f"legacy.{key}"
            )
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
            )
        }
        if enabled:
            if counters["microbatch_count"] != expected_rank_microbatches:
                raise ValueError("Rank legacy multiscale denominator drift")
        elif any(counters.values()):
            raise ValueError("Disabled legacy multiscale state is non-empty")
        if (
            counters["recorded_microbatch_count"] > counters["microbatch_count"]
            or counters["optimization_valid_count"]
            + counters["optimization_invalid_count"]
            != counters["microbatch_count"]
            or (
                enabled
                and counters["recorded_microbatch_count"]
                != counters["optimization_valid_count"]
            )
        ):
            raise ValueError("Rank legacy multiscale count partition drift")
        per_stage = legacy_state.get("per_stage")
        if not isinstance(per_stage, Mapping) or tuple(per_stage) != (
            *LEGACY_MULTISCALE_CUMULATIVE_STAGES,
        ):
            raise ValueError("Rank legacy multiscale stage set drift")
        for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES:
            stage_state = per_stage[stage]
            if not isinstance(stage_state, Mapping) or set(stage_state) != set(
                LEGACY_MULTISCALE_CUMULATIVE_KEYS
            ):
                raise ValueError("Rank legacy multiscale metric key set drift")
            for key in LEGACY_MULTISCALE_INTEGER_KEYS:
                count = _strict_nonnegative_counter(
                    stage_state.get(key), name=f"legacy.{stage}.{key}"
                )
                if not enabled and count != 0:
                    raise ValueError("Disabled legacy multiscale metrics are non-empty")
            for key in LEGACY_MULTISCALE_FLOAT_KEYS:
                metric = _strict_finite_real(
                    stage_state.get(key), name=f"legacy.{stage}.{key}"
                )
                if not enabled and metric != 0.0:
                    raise ValueError("Disabled legacy multiscale metrics are non-empty")
            if (
                int(stage_state["valid_pairs"]) > int(stage_state["input_pairs"])
                or int(stage_state["kept_pairs"]) > int(stage_state["valid_pairs"])
                or int(stage_state["kept_proposals"])
                > int(stage_state["input_proposals"])
            ):
                raise ValueError("Rank legacy multiscale metric denominator drift")
        legacy_states.append(legacy_state)
    if len(set(legacy_enabled_values)) != 1:
        raise ValueError("Ranks disagree on legacy multiscale enablement")
    legacy_enabled = legacy_enabled_values[0]
    if hierarchy_enabled and legacy_enabled:
        raise ValueError(
            "Hierarchy and legacy multiscale delivery cannot both be enabled"
        )
    legacy = {
        "schema_version": LEGACY_MULTISCALE_CUMULATIVE_SCHEMA,
        "enabled": legacy_enabled,
        "expected_microbatch_count": (
            expected_rank_microbatches * int(world_size) if legacy_enabled else 0
        ),
        **{
            key: sum(int(state[key]) for state in legacy_states)
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
            )
        },
        "per_stage": {
            stage: {
                key: (
                    sum(int(state["per_stage"][stage][key]) for state in legacy_states)
                    if key in LEGACY_MULTISCALE_INTEGER_KEYS
                    else sum(
                        float(state["per_stage"][stage][key])
                        for state in legacy_states
                    )
                )
                for key in LEGACY_MULTISCALE_CUMULATIVE_KEYS
            }
            for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES
        },
    }
    legacy["optimization_valid_fraction"] = int(
        legacy["optimization_valid_count"]
    ) / max(int(legacy["microbatch_count"]), 1)
    for stage_summary in legacy["per_stage"].values():
        stage_summary["pair_survival_fraction"] = int(
            stage_summary["kept_pairs"]
        ) / max(int(stage_summary["input_pairs"]), 1)
        stage_summary["proposal_survival_fraction"] = int(
            stage_summary["kept_proposals"]
        ) / max(int(stage_summary["input_proposals"]), 1)

    panels_by_update: dict[int, dict[str, Any]] = {}
    for row in ordered:
        panel_state = row.get("gradient_panels")
        if not isinstance(panel_state, Mapping):
            raise ValueError("Rank gradient-panel state is missing")
        panel_records = panel_state.get("panels")
        if not isinstance(panel_records, list):
            raise ValueError("Rank gradient-panel records are invalid")
        rank_panel_updates: set[int] = set()
        for panel in panel_records:
            if not isinstance(panel, Mapping):
                raise ValueError("Rank gradient-panel record is invalid")
            try:
                panel_update = _strict_nonnegative_counter(
                    panel.get("update"), name="gradient_panel.update"
                )
            except ValueError as exc:
                raise ValueError("Rank gradient-panel record update is invalid") from exc
            if panel_update in rank_panel_updates:
                raise ValueError("Duplicate gradient panel update in rank state")
            rank_panel_updates.add(panel_update)
            if panel_update not in scheduled_updates or panel_update > int(
                through_update
            ):
                raise ValueError("Gradient panel lies outside cumulative schedule")
            required = {
                "kind",
                "update",
                "global_shared_gradient",
                "controller",
                "conflict_threshold",
                "objective_stage_counts",
            }
            if panel.get("kind") != "gradient_health" or not required.issubset(
                panel
            ):
                raise ValueError("Incomplete cumulative gradient-panel record")
            if not isinstance(panel.get("global_shared_gradient"), Mapping):
                raise ValueError("Invalid cumulative gradient-panel gradient report")
            if not isinstance(panel.get("controller"), Mapping):
                raise ValueError("Invalid cumulative gradient-panel controller")
            if not isinstance(panel.get("objective_stage_counts"), Mapping):
                raise ValueError("Invalid cumulative gradient-panel stage counts")
            _strict_finite_real(
                panel.get("conflict_threshold"),
                name="gradient_panel.conflict_threshold",
            )
            copied = _json_copy(dict(panel))
            if panel_update in panels_by_update and panels_by_update[panel_update] != copied:
                raise ValueError("Ranks disagree on a cumulative gradient panel")
            panels_by_update[panel_update] = copied
    observed_updates = sorted(panels_by_update)
    expected_updates = [
        value for value in scheduled_updates if value <= int(through_update)
    ]
    missing_updates = sorted(set(expected_updates) - set(observed_updates))
    unexpected_updates = sorted(set(observed_updates) - set(expected_updates))
    controller = _json_copy(dict(gradient_conflict_controller))
    rank_event_count = int(through_update) * int(world_size)
    rank_clipped_event_count = sum(per_rank_clipped)
    return {
        "schema_version": CUMULATIVE_HEALTH_SCHEMA,
        "through_update": int(through_update),
        "optimizer_update_count": int(through_update),
        "world_size": int(world_size),
        "source_cells": list(source_cells),
        "granularities": list(granularities),
        "expected_microbatches_per_update": expected_microbatches_per_update,
        "clipping": {
            "definition": "any_rank_was_clipped_per_optimizer_update",
            "clipped_update_count": clipped_update_count,
            "unclipped_update_count": unclipped_update_count,
            "clipped_update_fraction": clipped_update_count
            / max(int(through_update), 1),
            "rank_update_event_count": rank_event_count,
            "rank_clipped_event_count": rank_clipped_event_count,
            "rank_update_clipping_fraction": rank_clipped_event_count
            / max(rank_event_count, 1),
            "per_rank_clipped_update_counts": per_rank_clipped,
        },
        # Keep the checkpoint-native names aligned with the evaluator's
        # cumulative_health/v1 fallback.  Rank-local state intentionally keeps
        # the model-facing names above because it is restored by the runner.
        "token_delivery": token,
        "hierarchy_delivery": hierarchy,
        "legacy_multiscale_delivery": legacy,
        "gradient_panels": {
            "schema_version": GRADIENT_PANEL_CUMULATIVE_SCHEMA,
            "scheduled_updates": scheduled_updates,
            "expected_updates": expected_updates,
            "observed_updates": observed_updates,
            "missing_updates": missing_updates,
            "unexpected_updates": unexpected_updates,
            "complete_through_update": not missing_updates and not unexpected_updates,
            "panels": [panels_by_update[value] for value in observed_updates],
        },
        "gradient_conflict_controller": controller,
        "auxiliary_budget": _json_copy(controller.get("auxiliary_budget")),
        "rank_states": [_json_copy(dict(row)) for row in ordered],
    }


def _restore_rank_cumulative_health(
    payload: Mapping[str, Any],
    *,
    rank: int,
    checkpoint_update: int,
    world_size: int,
    scheduled_gradient_panel_updates: Sequence[int],
    allow_schedule_extension: bool,
) -> dict[str, Any]:
    """Validate the aggregate record and restore this rank's exact counters."""

    if not isinstance(payload, Mapping):
        raise ValueError("Resume cumulative-health payload is not a mapping")
    if payload.get("schema_version") != CUMULATIVE_HEALTH_SCHEMA:
        raise ValueError("Resume cumulative-health schema drift")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or isinstance(checkpoint_update, bool)
        or not isinstance(checkpoint_update, int)
        or checkpoint_update < 0
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or rank >= world_size
        or not isinstance(allow_schedule_extension, bool)
    ):
        raise ValueError("Resume rank identity is invalid")
    rank_states = payload.get("rank_states")
    controller = payload.get("gradient_conflict_controller")
    if not isinstance(rank_states, list) or not isinstance(controller, Mapping):
        raise ValueError("Resume lacks exact cumulative rank/controller state")
    reconstructed = _merge_rank_cumulative_health(
        rank_states,
        through_update=checkpoint_update,
        world_size=world_size,
        gradient_conflict_controller=controller,
    )
    if reconstructed != payload:
        raise ValueError("Resume cumulative-health aggregate is corrupt")
    current_schedule = list(
        _canonical_panel_schedule(
            scheduled_gradient_panel_updates, name="current_schedule"
        )
    )
    previous_schedule = list(
        _canonical_panel_schedule(
            reconstructed["gradient_panels"]["scheduled_updates"],
            name="previous_schedule",
        )
    )
    observed_updates = list(reconstructed["gradient_panels"]["observed_updates"])
    expected_previous_updates = [
        value for value in previous_schedule if value <= int(checkpoint_update)
    ]
    if observed_updates != expected_previous_updates:
        raise ValueError(
            "Resume gradient-panel history is incomplete or non-canonical"
        )
    current_past_updates = [
        value for value in current_schedule if value <= int(checkpoint_update)
    ]
    if current_past_updates != expected_previous_updates:
        raise ValueError(
            "Resume gradient-panel history is not a subset of the current schedule"
        )
    if previous_schedule != current_schedule and not allow_schedule_extension:
        raise ValueError("Resume gradient-panel schedule differs")
    if not set(observed_updates).issubset(set(current_schedule)):
        raise ValueError("Resume has panels outside the current schedule")
    restored = _json_copy(rank_states[int(rank)])
    restored["gradient_panels"]["scheduled_updates"] = current_schedule
    return restored


def _tensor_mapping_sha256(
    state: Mapping[str, torch.Tensor],
    *,
    keys: Sequence[str] | None = None,
) -> str:
    """Content hash tensor values without depending on torch.save serialization."""

    selected = tuple(sorted(state if keys is None else keys))
    digest = hashlib.sha256()
    for key in selected:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _batchnorm_state_report(model: LitePTContrastiveModel) -> dict[str, Any]:
    state = model.state_dict()
    keys = tuple(
        sorted(
            key
            for key in state
            if key.endswith(".running_mean")
            or key.endswith(".running_var")
            or key.endswith(".num_batches_tracked")
        )
    )
    return {
        "schema_version": "litept_batchnorm_state/v1",
        "key_count": len(keys),
        "keys": list(keys),
        "sha256": _tensor_mapping_sha256(state, keys=keys),
    }


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture all mutable per-rank generators used by the runner."""

    return {
        "schema_version": "litept_pretraining_rng_state/v2",
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    if state.get("schema_version") != "litept_pretraining_rng_state/v2":
        raise ValueError("Resume RNG-state schema drift")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device=device)


def _native_fixed_pack_cosine_gap(evaluation: Mapping[str, Any]) -> float:
    """Return legacy dense-point dec0 gap as a diagnostic-only value."""

    if evaluation.get("selection_view") != "fixed_comparable":
        raise ValueError("Feature-hard gating requires a fixed-comparable panel")
    native = evaluation.get("cell_metrics", {}).get("native", {})
    gaps = [
        float(metrics["cosine_gap"])
        for cell, metrics in native.items()
        if str(cell).startswith("2d/")
    ]
    if not gaps or not all(math.isfinite(value) for value in gaps):
        raise ValueError("Fixed panel lacks finite native 2D cosine gaps")
    return float(np.mean(gaps))


def _native_token_pure_fixed_pack_cosine_gap(
    evaluation: Mapping[str, Any],
) -> float:
    """Return the exact native dec0 token-space gap or fail closed."""

    if evaluation.get("selection_view") != "fixed_comparable":
        raise ValueError("Token-pure feature-hard gating requires a fixed panel")
    pack = evaluation.get("fixed_eval_pack")
    if not isinstance(pack, Mapping) or not isinstance(
        pack.get("token_pure_dec0_fixed_pack"), Mapping
    ):
        raise ValueError("Fixed panel is not a V2 token-pure pack")
    report = evaluation.get("native_fixed_comparable_dec0")
    expected_cells = ("2d/g02", "2d/g05", "2d/g08")
    if (
        not isinstance(report, Mapping)
        or report.get("valid") is not True
        or tuple(report.get("expected_cells", ())) != expected_cells
        or set(report.get("cells", {})) != set(expected_cells)
        or any(
            not bool(report["cells"][cell].get("valid", False))
            or int(report["cells"][cell].get("denominator", 0)) <= 0
            for cell in expected_cells
        )
    ):
        raise ValueError("Fixed panel lacks exact valid g02/g05/g08 token metrics")
    value = float(report.get("macro_cosine_gap", float("nan")))
    if not math.isfinite(value):
        raise ValueError("Fixed token-space cosine gap is non-finite")
    return value


def _summarize_v2_sampling(
    records: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[str],
) -> dict[str, Any]:
    """Compact per-update delivery report without serializing coverage queues."""

    per_cell = {
        cell: {
            "requested_proposals": 0,
            "realized_unique_proposals": 0,
            "contrastive_retained_proposals": 0,
            "duplicate_proposals": 0,
            "frame_groups": 0,
            "realized_positive_pairs": 0,
            "requested_positive_pairs": 0,
            "visible_points": 0,
            "known_points": 0,
            "unknown_points": 0,
            "minimum_safe_negatives_per_pair": None,
        }
        for cell in cells
    }
    for record in records:
        coverage_by_cell = record["coverage_by_cell"]
        for cell in cells:
            coverage = coverage_by_cell[cell]
            row = per_cell[cell]
            row["requested_proposals"] += int(coverage["requested_count"])
            row["realized_unique_proposals"] += int(coverage["realized_count"])
            row["contrastive_retained_proposals"] += int(
                coverage["contrastive_retained_count"]
            )
            row["duplicate_proposals"] += int(coverage["duplicate_count"])
            row["frame_groups"] += int(coverage["frame_group_count"])
        for group in record.get("groups", []):
            row = per_cell[str(group["cell"])]
            sampler = group["sampler"]
            row["realized_positive_pairs"] += int(
                sampler["realized_positive_pairs"]
            )
            row["requested_positive_pairs"] += int(
                sampler["requested_positive_pairs_per_proposal"]
            ) * len(sampler["retained_proposal_indices"])
            row["visible_points"] += int(sampler["visible_count"])
            row["known_points"] += int(sampler["known_count"])
            row["unknown_points"] += int(sampler["unknown_count"])
            safe_min = int(sampler["safe_negatives_per_pair_min"])
            previous = row["minimum_safe_negatives_per_pair"]
            row["minimum_safe_negatives_per_pair"] = (
                safe_min if previous is None else min(int(previous), safe_min)
            )
    totals = {
        key: sum(int(row[key]) for row in per_cell.values())
        for key in (
            "requested_proposals",
            "realized_unique_proposals",
            "contrastive_retained_proposals",
            "duplicate_proposals",
            "frame_groups",
            "realized_positive_pairs",
            "requested_positive_pairs",
        )
    }
    totals["proposal_survival_fraction"] = totals[
        "contrastive_retained_proposals"
    ] / max(totals["realized_unique_proposals"], 1)
    totals["positive_pair_yield"] = totals["realized_positive_pairs"] / max(
        totals["requested_positive_pairs"], 1
    )
    realized_by_cell = [
        int(per_cell[cell]["realized_unique_proposals"]) for cell in cells
    ]
    total_realized = sum(realized_by_cell)
    per_cell_share = {
        cell: int(per_cell[cell]["realized_unique_proposals"])
        / max(total_realized, 1)
        for cell in cells
    }
    return {
        "schema_version": "litept_multigranular_sampling_health/v2",
        "scene_visits": len(records),
        "optimization_valid_scene_visits": sum(
            int(bool(record.get("optimization_valid", False)))
            for record in records
        ),
        "optimization_invalid_scene_visits": sum(
            int(not bool(record.get("optimization_valid", False)))
            for record in records
        ),
        "optimization_valid_fraction": sum(
            int(bool(record.get("optimization_valid", False)))
            for record in records
        )
        / max(len(records), 1),
        "optimization_drop_reasons": {
            reason: sum(
                int(record.get("optimization_drop_reason") == reason)
                for record in records
            )
            for reason in sorted(
                {
                    str(record["optimization_drop_reason"])
                    for record in records
                    if record.get("optimization_drop_reason")
                }
            )
        },
        "feature_hard_enabled": None,
        "totals": totals,
        "per_cell_optimization_share": per_cell_share,
        "per_cell": per_cell,
        "hard_invariants": {
            "zero_duplicate_proposals": totals["duplicate_proposals"] == 0,
            "frame_local_groups": True,
            "teacher_comembership_filtered": True,
            "unknown_background_excluded": True,
        },
    }


def _mean_numeric_tree(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average common scalar telemetry leaves while retaining stable contracts."""

    if not records:
        return {}
    output: dict[str, Any] = {}
    common_keys = set(records[0])
    for record in records[1:]:
        common_keys &= set(record)
    for key in sorted(common_keys):
        values = [record[key] for record in records]
        if all(isinstance(value, bool) for value in values):
            output[key] = all(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            output[key] = float(np.mean([float(value) for value in values]))
        elif all(isinstance(value, Mapping) for value in values):
            output[key] = _mean_numeric_tree(values)
        elif all(value == values[0] for value in values):
            output[key] = values[0]
    output["microbatch_count"] = len(records)
    return output


def _summarize_token_pure_records(
    records: Sequence[Mapping[str, Any]], *, cells: Sequence[str]
) -> dict[str, Any]:
    """Reduce token-space delivery with explicit denominators and worst cases."""

    ordered_cells = tuple(str(cell) for cell in cells)
    per_cell: dict[str, dict[str, Any]] = {
        cell: {
            "input_groups": 0,
            "retained_groups": 0,
            "input_proposals": 0,
            "retained_proposals": 0,
            "input_point_pairs": 0,
            "requested_token_pairs": 0,
            "realized_positive_pairs": 0,
            "zero_denominator_groups": 0,
            "minimum_pure_positive_survival_fraction": None,
            "minimum_safe_negative_tokens": None,
            "drop_reasons": {},
        }
        for cell in ordered_cells
    }
    worst_groups: list[dict[str, Any]] = []
    valid_visits = 0
    exact_three_cell_visits = 0
    for visit_index, record in enumerate(records):
        retained_cells = {str(cell) for cell in record.get("retained_cells", ())}
        optimization_valid = bool(
            record.get("optimization_valid", retained_cells == set(ordered_cells))
        )
        valid_visits += int(optimization_valid)
        exact_three_cell_visits += int(retained_cells == set(ordered_cells))
        for group_index, group in enumerate(record.get("groups", ())):
            cell = str(group.get("cell"))
            if cell not in per_cell:
                continue
            summary = per_cell[cell]
            summary["input_groups"] += 1
            realized_pairs = int(group.get("realized_positive_pairs", 0))
            summary["retained_groups"] += int(realized_pairs > 0)
            summary["input_proposals"] += int(group.get("input_proposals", 0))
            summary["retained_proposals"] += int(
                group.get("retained_proposals", 0)
            )
            summary["input_point_pairs"] += int(group.get("input_point_pairs", 0))
            summary["realized_positive_pairs"] += realized_pairs
            summary["zero_denominator_groups"] += int(
                bool(group.get("zero_denominator", realized_pairs <= 0))
            )
            requested_token_pairs = 0
            group_min_survival: float | None = None
            group_min_safe: int | None = None
            for proposal in group.get("proposals", ()):
                requested_token_pairs += int(
                    proposal.get("requested_token_pairs", 0)
                )
                survival = proposal.get("pure_positive_survival_fraction")
                if isinstance(survival, (int, float)) and math.isfinite(
                    float(survival)
                ):
                    group_min_survival = (
                        float(survival)
                        if group_min_survival is None
                        else min(group_min_survival, float(survival))
                    )
                safe = proposal.get("safe_negative_tokens_min")
                if isinstance(safe, (int, float)):
                    group_min_safe = (
                        int(safe)
                        if group_min_safe is None
                        else min(group_min_safe, int(safe))
                    )
                reason = proposal.get("drop_reason")
                if reason:
                    reasons = summary["drop_reasons"]
                    reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1
            summary["requested_token_pairs"] += requested_token_pairs
            if group_min_survival is not None:
                previous = summary["minimum_pure_positive_survival_fraction"]
                summary["minimum_pure_positive_survival_fraction"] = (
                    group_min_survival
                    if previous is None
                    else min(float(previous), group_min_survival)
                )
            if group_min_safe is not None:
                previous_safe = summary["minimum_safe_negative_tokens"]
                summary["minimum_safe_negative_tokens"] = (
                    group_min_safe
                    if previous_safe is None
                    else min(int(previous_safe), group_min_safe)
                )
            worst_groups.append(
                {
                    "visit_index": visit_index,
                    "group_index": group_index,
                    "cell": cell,
                    "group_id": group.get("group_id"),
                    "input_proposals": int(group.get("input_proposals", 0)),
                    "retained_proposals": int(group.get("retained_proposals", 0)),
                    "requested_token_pairs": requested_token_pairs,
                    "realized_positive_pairs": realized_pairs,
                    "zero_denominator": bool(
                        group.get("zero_denominator", realized_pairs <= 0)
                    ),
                }
            )
    for summary in per_cell.values():
        summary["group_survival_fraction"] = int(
            summary["retained_groups"]
        ) / max(int(summary["input_groups"]), 1)
        summary["proposal_survival_fraction"] = int(
            summary["retained_proposals"]
        ) / max(int(summary["input_proposals"]), 1)
        summary["realized_pair_yield"] = int(
            summary["realized_positive_pairs"]
        ) / max(int(summary["requested_token_pairs"]), 1)
    worst_groups.sort(
        key=lambda row: (
            row["realized_positive_pairs"] / max(row["requested_token_pairs"], 1),
            row["retained_proposals"] / max(row["input_proposals"], 1),
            row["visit_index"],
            row["group_index"],
        )
    )
    return {
        "schema_version": "litept_token_pure_delivery_summary/v2",
        "microbatch_count": len(records),
        "optimization_valid_count": valid_visits,
        "optimization_valid_fraction": valid_visits / max(len(records), 1),
        "exact_three_cell_count": exact_three_cell_visits,
        "exact_three_cell_fraction": exact_three_cell_visits
        / max(len(records), 1),
        "per_cell": per_cell,
        "worst_groups": worst_groups[:12],
    }


def _summarize_hierarchy_records(
    records: Sequence[Mapping[str, Any]], *, cells: Sequence[str]
) -> dict[str, Any]:
    """Reduce mask/VICReg delivery without discarding variable rank evidence."""

    granularities = tuple(str(cell).rsplit("/", 1)[-1] for cell in cells)
    per_cell: dict[str, dict[str, Any]] = {
        granularity: {
            "frame_groups": 0,
            "valid_fold_denominator": 0,
            "active_visits": 0,
            "g02_routed_dec1_proposals": 0,
            "g02_routed_dec2_proposals": 0,
            "route_valid_visits": 0,
            "route_input_proposals": 0,
            "route_assigned_proposals": 0,
        }
        for granularity in granularities
    }
    stage_names = ("enc4", "dec3", "dec2", "dec1", "dec0")
    per_stage: dict[str, dict[str, Any]] = {
        stage: {
            "minimum_token_count": None,
            "mask_valid_fold_denominator": 0,
            "mask_proposals_attempted": 0,
            "mask_proposals_valid": 0,
            "mask_folds_attempted": 0,
            "mask_folds_valid": 0,
            "vicreg_all_valid": True,
            "vicreg_minimum_std": None,
            "vicreg_minimum_participation_rank": None,
            "vicreg_local_sample_count": 0,
            "vicreg_global_sample_count": 0,
            "vicreg_discarded_token_count": 0,
        }
        for stage in stage_names
    }
    optimization_valid_count = 0
    for record in records:
        optimization_valid_count += int(record.get("optimization_valid", True))
        mask = record.get("mask", {})
        for granularity, route_record in record.get(
            "route_stage_validity", {}
        ).items():
            if granularity not in per_cell:
                continue
            per_cell[granularity]["route_valid_visits"] += int(
                bool(route_record.get("optimization_valid", False))
            )
            per_cell[granularity]["route_input_proposals"] += int(
                route_record.get("input_proposal_count", 0)
            )
            per_cell[granularity]["route_assigned_proposals"] += int(
                route_record.get("assigned_proposal_count", 0)
            )
        for granularity, cell_record in mask.get("cells", {}).items():
            if granularity not in per_cell:
                continue
            summary = per_cell[granularity]
            summary["frame_groups"] += int(cell_record.get("group_count", 0))
            denominator = int(
                cell_record.get("loss_denominator_valid_folds", 0)
            )
            summary["valid_fold_denominator"] += denominator
            summary["active_visits"] += int(denominator > 0)
            if granularity == "g02":
                for route in cell_record.get("route", ()):
                    summary["g02_routed_dec1_proposals"] += int(
                        route.get("routed_dec1_proposals", 0)
                    )
                    summary["g02_routed_dec2_proposals"] += int(
                        route.get("routed_dec2_proposals", 0)
                    )
        for stage, stage_record in record.get("stages", {}).items():
            if stage not in per_stage:
                continue
            summary = per_stage[stage]
            token_count = int(stage_record.get("token_count", 0))
            summary["minimum_token_count"] = (
                token_count
                if summary["minimum_token_count"] is None
                else min(int(summary["minimum_token_count"]), token_count)
            )
            mask_stage = stage_record.get("mask", {})
            summary["mask_valid_fold_denominator"] += int(
                mask_stage.get("loss_denominator_valid_folds", 0)
            )
            for source, target in (
                ("proposals_attempted", "mask_proposals_attempted"),
                ("proposals_valid", "mask_proposals_valid"),
                ("folds_attempted", "mask_folds_attempted"),
                ("folds_valid", "mask_folds_valid"),
            ):
                summary[target] += int(mask_stage.get(source, 0))
            vicreg = stage_record.get("vicreg", {})
            summary["vicreg_all_valid"] = bool(
                summary["vicreg_all_valid"] and vicreg.get("valid", True)
            )
            for source, target in (
                ("local_sampled_token_count", "vicreg_local_sample_count"),
                ("global_sampled_token_count", "vicreg_global_sample_count"),
            ):
                summary[target] += int(vicreg.get(source, 0))
            summary["vicreg_discarded_token_count"] += sum(
                int(value)
                for value in vicreg.get("per_rank_discarded_token_counts", ())
            )
            for source, target in (
                ("component_std_min", "vicreg_minimum_std"),
                ("participation_rank", "vicreg_minimum_participation_rank"),
            ):
                value = vicreg.get(source)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    summary[target] = (
                        float(value)
                        if summary[target] is None
                        else min(float(summary[target]), float(value))
                    )
    for summary in per_stage.values():
        summary["mask_proposal_survival_fraction"] = int(
            summary["mask_proposals_valid"]
        ) / max(int(summary["mask_proposals_attempted"]), 1)
        summary["mask_fold_survival_fraction"] = int(
            summary["mask_folds_valid"]
        ) / max(int(summary["mask_folds_attempted"]), 1)
    for summary in per_cell.values():
        summary["route_valid_fraction"] = int(
            summary["route_valid_visits"]
        ) / max(len(records), 1)
        summary["route_assignment_fraction"] = int(
            summary["route_assigned_proposals"]
        ) / max(int(summary["route_input_proposals"]), 1)
    return {
        "schema_version": "litept_hierarchy_delivery_summary/v2",
        "microbatch_count": len(records),
        "optimization_valid_count": optimization_valid_count,
        "optimization_valid_fraction": optimization_valid_count
        / max(len(records), 1),
        "per_cell": per_cell,
        "per_stage": per_stage,
    }


def _telemetry_objective_cosine(
    records: Sequence[Mapping[str, Any]], objective: str
) -> float | None:
    values: list[float] = []
    for record in records:
        stages = (
            record.get("objectives", {})
            .get(objective, {})
            .get("stages", {})
        )
        for metrics in stages.values():
            if metrics.get("cosine_valid"):
                value = float(metrics["cosine_vs_final_dec0"])
                if math.isfinite(value):
                    values.append(value)
    return float(np.mean(values)) if values else None


def _telemetry_auxiliary_budget(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_record: list[float] = []
    per_objective: dict[str, list[float]] = {}
    for record in records:
        objectives = record.get("objectives", {})
        stage_sums: dict[str, float] = {}
        for objective, objective_record in objectives.items():
            if objective == "final_dec0":
                continue
            ratios = [
                float(metrics["grad_norm_ratio_vs_final_dec0"])
                for metrics in objective_record.get("stages", {}).values()
                if metrics.get("norm_ratio_valid")
            ]
            if ratios:
                value = max(ratios)
                per_objective.setdefault(str(objective), []).append(value)
                stage_sums[str(objective)] = value
        if stage_sums:
            per_record.append(sum(stage_sums.values()))
    maximum = max(per_record) if per_record else None
    return {
        "conservative_sum_aux_norm_ratio_max": maximum,
        "conservative_sum_aux_norm_ratio_mean": (
            float(np.mean(per_record)) if per_record else None
        ),
        "per_objective_max_ratio": {
            objective: max(values)
            for objective, values in per_objective.items()
        },
        "limit": 0.5,
        "passed": maximum is not None and maximum <= 0.5,
    }


def _manual_objective_gradients(
    *,
    components: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    scale: float,
) -> dict[str, list[torch.Tensor]]:
    """Materialize one microbatch's objective gradients for rare PCGrad fallback."""

    if scale <= 0.0:
        raise ValueError("manual gradient scale must be positive")
    output: dict[str, list[torch.Tensor]] = {}
    for objective, component in components.items():
        gradients = torch.autograd.grad(
            component / float(scale),
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        output[objective] = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if gradient is None
            else gradient.detach().clone()
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    return output


def _flat_objective_gradients(
    *,
    components: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    scale: float,
) -> dict[str, torch.Tensor]:
    """Return compact FP32 objective gradients over one shared parameter set.

    These vectors are used only on predeclared telemetry panels or after the
    conditional PCGrad fallback has been enabled.  Keeping one flat vector per
    objective avoids retaining full-model zero tensors for private heads and
    permits one collective per objective instead of one per parameter.
    """

    if scale <= 0.0:
        raise ValueError("flat gradient scale must be positive")
    if not parameters:
        raise ValueError("shared parameter set must not be empty")
    parameter_numels = [int(parameter.numel()) for parameter in parameters]
    total_numel = sum(parameter_numels)
    output: dict[str, torch.Tensor] = {}
    for objective, component in components.items():
        gradients = torch.autograd.grad(
            component / float(scale),
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        flat = torch.empty(
            total_numel,
            dtype=torch.float32,
            device=parameters[0].device,
        )
        offset = 0
        for parameter, gradient, numel in zip(
            parameters, gradients, parameter_numels, strict=True
        ):
            target = flat[offset : offset + numel]
            if gradient is None:
                target.zero_()
            else:
                target.copy_(gradient.detach().reshape(-1).float())
            offset += numel
        output[str(objective)] = flat
    return output


def _add_flat_objective_gradients(
    destination: dict[str, torch.Tensor] | None,
    source: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if destination is None:
        return {objective: vector.clone() for objective, vector in source.items()}
    if set(destination) != set(source):
        raise RuntimeError("Flat objective set drift inside a logical batch")
    for objective, vector in source.items():
        if destination[objective].shape != vector.shape:
            raise RuntimeError("Flat objective shape drift inside a logical batch")
        destination[objective].add_(vector)
    return destination


def _synchronize_accumulated_gradients(
    *,
    parameters: Sequence[torch.nn.Parameter],
    world_size: int,
) -> None:
    """Average gradients accumulated inside a ``DistributedDataParallel.no_sync`` block.

    DDP normally performs this reduction from the next synchronized backward.
    A V2 logical batch can skip its nominal final scene before constructing a
    DDP graph, though, so there is no backward pass available to flush the
    preceding ``no_sync`` gradients.  Reduce a shared presence mask first so
    ranks with different local autograd usage still execute identical
    collectives, and preserve ``None`` for parameters unused on every rank.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    trainable = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    if not trainable or world_size == 1:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Accumulated-gradient synchronization requires an initialized process group"
        )
    device = trainable[0].device
    if any(parameter.device != device for parameter in trainable):
        raise ValueError("Accumulated-gradient parameters must share one device")
    present = torch.tensor(
        [int(parameter.grad is not None) for parameter in trainable],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    sparse = torch.tensor(
        int(
            any(
                parameter.grad is not None and parameter.grad.is_sparse
                for parameter in trainable
            )
        ),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(sparse, op=dist.ReduceOp.MAX)
    if int(sparse.item()) != 0:
        raise RuntimeError(
            "Accumulated-gradient synchronization does not support sparse gradients"
        )
    for parameter, globally_present in zip(trainable, present.tolist(), strict=True):
        if int(globally_present) == 0:
            parameter.grad = None
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
            parameter.grad = gradient
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(float(world_size))


def _global_shared_gradient_report(
    *,
    accumulated: Mapping[str, torch.Tensor],
    shared_parameters: Sequence[torch.nn.Parameter],
    projected_objectives: set[str],
    world_size: int,
    install_projected_gradient: bool,
) -> dict[str, Any]:
    """Reduce true logical-batch gradients and optionally install shared PCGrad.

    Private head gradients are deliberately left untouched.  Only the common
    LitePT backbone is eligible for projection, which prevents an auxiliary
    objective from injecting the primary objective into private parameters.
    """

    if "final_dec0" not in accumulated:
        raise ValueError("Shared-gradient report requires final_dec0")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    reduced = {
        objective: vector.detach().clone().float()
        for objective, vector in accumulated.items()
    }
    expected_numel = sum(int(parameter.numel()) for parameter in shared_parameters)
    if not shared_parameters or any(
        int(vector.numel()) != expected_numel for vector in reduced.values()
    ):
        raise ValueError("Shared-gradient vector/parameter layout mismatch")
    for vector in reduced.values():
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        vector.div_(float(world_size))

    reference = reduced["final_dec0"]
    reference_square = reference.square().sum()
    projected_vectors = {
        objective: vector.clone() for objective, vector in reduced.items()
    }
    objective_reports: dict[str, Any] = {}
    for objective, vector in reduced.items():
        if objective == "final_dec0":
            continue
        objective_square = vector.square().sum()
        dot_before = (vector * reference).sum()
        denominator_before = torch.sqrt(
            reference_square.clamp_min(1.0e-20)
            * objective_square.clamp_min(1.0e-20)
        )
        should_project = (
            objective in projected_objectives
            and float(dot_before.detach()) < 0.0
            and float(reference_square.detach()) > 0.0
        )
        projected = projected_vectors[objective]
        if should_project:
            coefficient = dot_before / reference_square.clamp_min(1.0e-20)
            projected.sub_(coefficient * reference)
        projected_square = projected.square().sum()
        dot_after = (projected * reference).sum()
        denominator_after = torch.sqrt(
            reference_square.clamp_min(1.0e-20)
            * projected_square.clamp_min(1.0e-20)
        )
        objective_reports[objective] = {
            "enabled": objective in projected_objectives,
            "applied": bool(should_project),
            "global_shared_dot_before": float(dot_before.detach()),
            "global_shared_cosine_before": float(
                (dot_before / denominator_before).detach()
            ),
            "global_shared_norm_ratio_before": float(
                torch.sqrt(
                    objective_square.clamp_min(0.0)
                    / reference_square.clamp_min(1.0e-20)
                ).detach()
            ),
            "global_shared_dot_after": float(dot_after.detach()),
            "global_shared_cosine_after": float(
                (dot_after / denominator_after).detach()
            ),
            "global_shared_norm_ratio_after": float(
                torch.sqrt(
                    projected_square.clamp_min(0.0)
                    / reference_square.clamp_min(1.0e-20)
                ).detach()
            ),
        }

    auxiliary_names = [
        objective for objective in reduced if objective != "final_dec0"
    ]
    auxiliary_before = sum(
        (reduced[objective] for objective in auxiliary_names),
        torch.zeros_like(reference),
    )
    auxiliary_after = sum(
        (projected_vectors[objective] for objective in auxiliary_names),
        torch.zeros_like(reference),
    )
    reference_norm = reference.norm().clamp_min(1.0e-20)
    budget_before = float((auxiliary_before.norm() / reference_norm).detach())
    budget_after = float((auxiliary_after.norm() / reference_norm).detach())

    if install_projected_gradient:
        combined = sum(
            projected_vectors.values(), torch.zeros_like(reference)
        )
        offset = 0
        for parameter in shared_parameters:
            numel = int(parameter.numel())
            replacement = combined[offset : offset + numel].reshape_as(parameter)
            parameter.grad = replacement.to(dtype=parameter.dtype).clone()
            offset += numel
        if offset != int(combined.numel()):
            raise RuntimeError("Shared-gradient installation layout drift")

    return {
        "schema_version": "litept_global_shared_backbone_gradient/v2",
        "reference": "final_dec0",
        "shared_parameter_count": len(shared_parameters),
        "shared_parameter_numel": expected_numel,
        "projected_objectives": sorted(projected_objectives),
        "projection_installed": bool(install_projected_gradient),
        "objectives": objective_reports,
        "auxiliary_budget": {
            "global_sum_aux_norm_ratio_before": budget_before,
            "global_sum_aux_norm_ratio_after": budget_after,
            "limit": 0.5,
            "passed_before": budget_before <= 0.5,
            "passed_after": budget_after <= 0.5,
        },
    }


def _install_global_flat_total_gradient(
    *,
    accumulated: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    world_size: int,
) -> None:
    """All-reduce and install one compact total-gradient vector."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    expected_numel = sum(int(parameter.numel()) for parameter in parameters)
    if int(accumulated.numel()) != expected_numel:
        raise ValueError("Flat total-gradient vector/parameter layout mismatch")
    reduced = accumulated.detach().clone().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced.div_(float(world_size))
    offset = 0
    for parameter in parameters:
        numel = int(parameter.numel())
        parameter.grad = (
            reduced[offset : offset + numel]
            .reshape_as(parameter)
            .to(dtype=parameter.dtype)
            .clone()
        )
        offset += numel


def _add_manual_gradients(
    destination: dict[str, list[torch.Tensor]] | None,
    source: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    if destination is None:
        return {
            objective: [tensor.clone() for tensor in tensors]
            for objective, tensors in source.items()
        }
    if set(destination) != set(source):
        raise RuntimeError("Manual objective set drift inside a logical batch")
    for objective, tensors in source.items():
        for accumulated, current in zip(
            destination[objective], tensors, strict=True
        ):
            accumulated.add_(current)
    return destination


def _install_global_pcgrad(
    *,
    accumulated: Mapping[str, Sequence[torch.Tensor]],
    parameters: Sequence[torch.nn.Parameter],
    projected_objectives: set[str],
    world_size: int,
    shared_parameter_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Compatibility PCGrad helper with projection restricted to shared tensors."""

    if "final_dec0" not in accumulated:
        raise ValueError("PCGrad requires final_dec0 as reference")
    reduced = {
        objective: [tensor.clone() for tensor in tensors]
        for objective, tensors in accumulated.items()
    }
    for tensors in reduced.values():
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(float(world_size))
    reference = reduced["final_dec0"]
    if shared_parameter_indices is None:
        shared_parameter_indices = set(range(len(parameters)))
    if not shared_parameter_indices or any(
        index < 0 or index >= len(parameters)
        for index in shared_parameter_indices
    ):
        raise ValueError("shared_parameter_indices are empty or out of range")
    projection_report: dict[str, Any] = {}
    for objective, gradients in reduced.items():
        if objective == "final_dec0":
            continue
        dot = sum(
            (gradient.float() * ref.float()).sum()
            for index, (gradient, ref) in enumerate(
                zip(gradients, reference, strict=True)
            )
            if index in shared_parameter_indices
        )
        reference_square = sum(
            ref.float().square().sum()
            for index, ref in enumerate(reference)
            if index in shared_parameter_indices
        )
        objective_square = sum(
            gradient.float().square().sum()
            for index, gradient in enumerate(gradients)
            if index in shared_parameter_indices
        )
        projected = objective in projected_objectives and float(dot) < 0.0
        if projected and float(reference_square) > 0.0:
            coefficient = dot / reference_square.clamp_min(1.0e-20)
            for index, (gradient, ref) in enumerate(
                zip(gradients, reference, strict=True)
            ):
                if index in shared_parameter_indices:
                    gradient.sub_(coefficient.to(gradient.dtype) * ref)
        dot_after = sum(
            (gradient.float() * ref.float()).sum()
            for index, (gradient, ref) in enumerate(
                zip(gradients, reference, strict=True)
            )
            if index in shared_parameter_indices
        )
        objective_square_after = sum(
            gradient.float().square().sum()
            for index, gradient in enumerate(gradients)
            if index in shared_parameter_indices
        )
        denominator = torch.sqrt(
            reference_square.clamp_min(1.0e-20)
            * objective_square.clamp_min(1.0e-20)
        )
        projection_report[objective] = {
            "enabled": objective in projected_objectives,
            "applied": bool(projected),
            "global_dot_before": float(dot),
            "global_cosine_before": float(dot / denominator),
            "global_shared_dot_after": float(dot_after),
            "global_shared_cosine_after": float(
                dot_after
                / torch.sqrt(
                    reference_square.clamp_min(1.0e-20)
                    * objective_square_after.clamp_min(1.0e-20)
                )
            ),
        }
    for parameter_index, parameter in enumerate(parameters):
        combined = sum(
            (
                tensors[parameter_index]
                for tensors in reduced.values()
            ),
            torch.zeros_like(parameter, memory_format=torch.preserve_format),
        )
        parameter.grad = combined
    return {
        "schema_version": "litept_global_shared_parameter_pcgrad/v2",
        "reference": "final_dec0",
        "shared_parameter_indices": sorted(shared_parameter_indices),
        "projected_objectives": sorted(projected_objectives),
        "objectives": projection_report,
    }


def _checkpoint_payload(
    *,
    epoch: int,
    update: int,
    model: LitePTContrastiveModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    resolved: dict[str, Any],
    evaluation: dict[str, Any] | None,
    evaluation_history: dict[int, dict[str, Any]],
    rank_exposure_counts: list[dict[str, int]],
    rank_coverage_states: list[dict[str, Any]] | None = None,
    rank_sampling_totals: list[dict[str, Any]] | None = None,
    rank_rng_states: list[dict[str, Any]] | None = None,
    feature_hard_state: dict[str, Any] | None = None,
    gradient_conflict_state: dict[str, Any] | None = None,
    cumulative_health: dict[str, Any] | None = None,
    checkpoint_kind: str = "resumable_epoch_boundary",
    nominal_epoch: int | None = None,
) -> dict[str, Any]:
    backbone_state = model.backbone.state_dict()
    pointgroup_state = {
        f"backbone.{key.removeprefix('model.')}": value
        for key, value in backbone_state.items()
        if key.startswith("model.")
    }
    if len(pointgroup_state) != len(backbone_state):
        invalid = sorted(
            key for key in backbone_state if not key.startswith("model.")
        )
        raise RuntimeError(
            f"Cannot map wrapper backbone into PointGroup state: {invalid}"
        )
    is_v2 = str(getattr(args, "sampling_mode", "")) == V2_SAMPLING_MODE
    auxiliary_prefixes = (
        "multiscale_projectors.",
        "multiscale_criteria.",
        "hierarchy_mask_supervisor.",
    )
    auxiliary_keys = sorted(
        key
        for key in model.state_dict()
        if key.startswith(auxiliary_prefixes)
    )
    if (is_v2 or bool(getattr(args, "multiscale_supervision", False))) and (
        cumulative_health is None
    ):
        raise ValueError(
            "V2/legacy-multiscale checkpoint requires cumulative health telemetry"
        )
    return {
        "schema_version": (
            "litept_rgbn6_multigranular_pretrain_checkpoint/v2"
            if is_v2
            else "litept_gt3_unsam3_pretrain_checkpoint/v1"
        ),
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "epoch": int(epoch),
        "nominal_epoch": int(epoch if nominal_epoch is None else nominal_epoch),
        "update": int(update),
        "checkpoint_kind": str(checkpoint_kind),
        "resumable": checkpoint_kind == "resumable_epoch_boundary",
        "seed": int(args.seed),
        "model_state_dict": model.state_dict(),
        "backbone_state_dict": backbone_state,
        "pointgroup_model_state_dict": pointgroup_state,
        "projection_head_state_dict": model.projector.state_dict(),
        "multiscale_projection_head_state_dict": (
            model.multiscale_projectors.state_dict()
        ),
        "multiscale_criterion_state_dict": (
            model.multiscale_criteria.state_dict()
        ),
        "criterion_state_dict": model.criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rank_exposure_counts": rank_exposure_counts,
        "global_exposure_counts": _merge_count_dicts(rank_exposure_counts),
        "rank_coverage_states": rank_coverage_states,
        "rank_sampling_totals": rank_sampling_totals,
        "rank_rng_states": rank_rng_states,
        "feature_hard_state": feature_hard_state,
        "gradient_conflict_state": gradient_conflict_state,
        "cumulative_health": cumulative_health,
        "batchnorm_raw": _batchnorm_state_report(model),
        "batchnorm_recalibrated": None,
        "auxiliary_state_keys": auxiliary_keys,
        "transfer_export_contract": {
            "allowed_prefixes": ["backbone."],
            "rejected_prefixes": [
                "projector.",
                "criterion.",
                *auxiliary_prefixes,
            ],
            "chosen_batchnorm_buffers_only": True,
        },
        "evaluation": evaluation,
        "evaluation_history": {
            str(key): value
            for key, value in sorted(evaluation_history.items())
        },
        "resolved_config": resolved,
        "public_checkpoint_used": bool(
            resolved["initialization"]["public_checkpoint_used"]
        ),
    }


def _strict_initial_load(
    *,
    model: LitePTContrastiveModel,
    path: Path,
    input_features: str,
    multiscale_supervision: bool = False,
    v2_auxiliary_initialization: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "litept_gt3_unsam3_initial/v1":
        raise ValueError(f"Unexpected initialization schema: {path}")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Initialization experiment drift: {path}")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get(
        "voxel_reduce"
    ) != "representative":
        raise ValueError(
            "Initialization does not declare representative-point voxelization; "
            "regenerate it rather than silently mixing preprocessing contracts"
        )
    declared_input_features = architecture.get("input_features")
    if declared_input_features is None:
        # The original v1 RGBN6 initialization predates the explicit feature
        # field. Preserve its compatibility while rejecting it for RGB3.
        declared_input_features = (
            "rgbn6" if architecture.get("in_channels") == 6 else None
        )
    if declared_input_features != str(input_features):
        raise ValueError(
            "Initialization input-feature contract differs from this run: "
            f"checkpoint={declared_input_features!r} run={input_features!r}"
        )
    declared_representative_sampling = architecture.get(
        "representative_sampling", "first"
    )
    if declared_representative_sampling != model.backbone.representative_sampling:
        raise ValueError(
            "Initialization representative-point sampling differs from this run: "
            f"checkpoint={declared_representative_sampling!r} "
            f"run={model.backbone.representative_sampling!r}"
        )
    declared_multiscale = bool(
        architecture.get("multiscale_supervision", False)
    )
    if v2_auxiliary_initialization and declared_multiscale:
        raise ValueError(
            "V2 arms require the common dec0-only initialization; disposable "
            "auxiliary heads are initialized deterministically by the V2 model"
        )
    if not v2_auxiliary_initialization and declared_multiscale != bool(
        multiscale_supervision
    ):
        raise ValueError(
            "Initialization multi-scale recipe differs from this run: "
            f"checkpoint={declared_multiscale!r} run={bool(multiscale_supervision)!r}"
        )
    if declared_multiscale:
        expected_multiscale = {
            "pretraining_recipe": model.recipe_name,
            "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
            "multiscale_level_loss_weights": dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            ),
            "multiscale_loss_weight": float(model.multiscale_loss_weight),
            "multiscale_warmup_epochs": int(model.multiscale_warmup_epochs),
            "multiscale_token_policy": "strict_pure_token_resampling_v1",
            "multiscale_temperature_policy": (
                "separate_learnable_temperature_per_auxiliary_level"
            ),
            "multiscale_feature_hard_policy": (
                "separate_candidate_selection_per_positive_endpoint"
            ),
        }
        multiscale_mismatches = {
            key: {
                "checkpoint": architecture.get(key),
                "run": expected,
            }
            for key, expected in expected_multiscale.items()
            if architecture.get(key) != expected
        }
        if multiscale_mismatches:
            raise ValueError(
                "Initialization multi-scale contract drift: "
                f"{multiscale_mismatches}"
            )
    expected_channels = 3 if input_features == "rgb3" else 6
    if int(architecture.get("in_channels", -1)) != expected_channels:
        raise ValueError(
            "Initialization input-channel count differs from this run: "
            f"checkpoint={architecture.get('in_channels')!r} "
            f"run={expected_channels}"
        )
    public_checkpoint_used = payload.get("public_checkpoint_used")
    if not isinstance(public_checkpoint_used, bool):
        raise ValueError(f"Initialization lacks an explicit public-checkpoint flag: {path}")
    initialization_kind = payload.get(
        "initialization_kind",
        "scratch" if not public_checkpoint_used else None,
    )
    if initialization_kind not in ("scratch", "public_litept_backbone"):
        raise ValueError(
            f"Unsupported initialization kind {initialization_kind!r}: {path}"
        )
    if public_checkpoint_used != (initialization_kind == "public_litept_backbone"):
        raise ValueError(f"Initialization kind/public-checkpoint flag drift: {path}")
    public_checkpoint = payload.get("public_checkpoint")
    if public_checkpoint_used:
        if not isinstance(public_checkpoint, dict):
            raise ValueError(f"Public initialization lacks checkpoint provenance: {path}")
        public_path = Path(public_checkpoint["path"]).resolve(strict=True)
        if sha256_file(public_path) != public_checkpoint["sha256"]:
            raise ValueError(f"Public initialization checkpoint hash drift: {public_path}")
        if int(public_checkpoint.get("loaded_backbone_keys", -1)) != 302:
            raise ValueError("Public initialization did not load exactly 302 backbone keys")
    checkpoint_state = payload.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Initialization lacks model_state_dict: {path}")
    auxiliary_prefixes = (
        "multiscale_projectors.",
        "multiscale_criteria.",
        "hierarchy_mask_supervisor.",
    )
    if v2_auxiliary_initialization:
        current_state = model.state_dict()
        expected_base_keys = {
            key
            for key in current_state
            if key.startswith(("backbone.", "projector.", "criterion."))
        }
        checkpoint_keys = set(checkpoint_state)
        if checkpoint_keys != expected_base_keys:
            raise ValueError(
                "V2 common initialization is not an exact backbone/primary-head "
                "state: "
                f"missing={sorted(expected_base_keys - checkpoint_keys)} "
                f"extra={sorted(checkpoint_keys - expected_base_keys)}"
            )
        for key in expected_base_keys:
            if (
                checkpoint_state[key].shape != current_state[key].shape
                or checkpoint_state[key].dtype != current_state[key].dtype
            ):
                raise ValueError(f"V2 initialization tensor contract drift: {key}")
        auxiliary_keys = sorted(
            key for key in current_state if key.startswith(auxiliary_prefixes)
        )
        merged_state = dict(current_state)
        merged_state.update(checkpoint_state)
        model.load_state_dict(merged_state, strict=True)
        base_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=sorted(expected_base_keys)
        )
        auxiliary_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=auxiliary_keys
        )
    else:
        model.load_state_dict(checkpoint_state, strict=True)
        expected_base_keys = set(checkpoint_state)
        auxiliary_keys = sorted(
            key for key in model.state_dict() if key.startswith(auxiliary_prefixes)
        )
        base_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=sorted(expected_base_keys)
        )
        auxiliary_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=auxiliary_keys
        )
    report = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "seed": int(payload["seed"]),
        "strict": True,
        "strict_scope": (
            "backbone_primary_exact_auxiliary_deterministic"
            if v2_auxiliary_initialization
            else "complete_model_exact"
        ),
        "loaded_base_key_count": len(expected_base_keys),
        "loaded_base_state_sha256": base_state_sha256,
        "auxiliary_key_count": len(auxiliary_keys),
        "auxiliary_keys": auxiliary_keys,
        "auxiliary_initial_state_sha256": auxiliary_state_sha256,
        "input_features": str(input_features),
        "in_channels": expected_channels,
        "representative_sampling": model.backbone.representative_sampling,
        "initialization_kind": initialization_kind,
        "public_checkpoint_used": public_checkpoint_used,
    }
    if public_checkpoint_used:
        report["public_checkpoint"] = public_checkpoint
        report["source_random_initialization"] = payload[
            "source_random_initialization"
        ]
    return report


def _preload_sources(
    *,
    scene_ids: list[str],
    source_root: Path,
    smoke_manifest: Path | None,
    expected_manifest_hashes: dict[str, str] | None,
    input_features: str,
    coordinate_normalization_policy: str = "native",
    color_mean: tuple[float, ...] | None = None,
    color_std: tuple[float, ...] | None = None,
    verify_hashes: bool = True,
) -> list[Any]:
    scenes = []
    for scene_id in scene_ids:
        manifest_path = (
            smoke_manifest
            if smoke_manifest is not None
            else source_root / scene_id / "source_manifest.json"
        )
        scene = load_scene_source(
            manifest_path,
            input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            color_mean=color_mean,
            color_std=color_std,
            verify_hashes=verify_hashes,
        )
        if smoke_manifest is None and scene.scene_id != scene_id:
            raise ValueError(
                f"Source scene {scene.scene_id} != split scene {scene_id}"
            )
        if expected_manifest_hashes is not None:
            expected_hash = expected_manifest_hashes.get(scene.scene_id)
            if expected_hash is None:
                raise ValueError(
                    f"{scene.scene_id}: absent from frozen dataset source audit"
                )
            if scene.manifest_sha256 != expected_hash:
                raise ValueError(
                    f"{scene.scene_id}: source manifest differs from frozen "
                    "dataset source audit"
                )
        scenes.append(scene)
    return scenes


def _audit_lazy_sources(
    *,
    scene_ids: list[str],
    source_root: Path,
    expected_manifest_hashes: dict[str, str] | None,
) -> list[Any]:
    """Validate lazy source identity and structure without materializing arrays."""

    audits = []
    for scene_id in scene_ids:
        audit = audit_scene_source_manifest(
            source_root / scene_id / "source_manifest.json",
            expected_scene_id=scene_id,
            verify_hashes=False,
        )
        if expected_manifest_hashes is not None:
            expected_hash = expected_manifest_hashes.get(scene_id)
            if expected_hash is None:
                raise ValueError(
                    f"{scene_id}: absent from frozen dataset source audit"
                )
            if audit.manifest_sha256 != expected_hash:
                raise ValueError(
                    f"{scene_id}: source manifest differs from frozen "
                    "dataset source audit"
                )
        audits.append(audit)
    return audits


def _load_logical_scene_group(
    *,
    scene_indices: list[int],
    scene_ids: list[str],
    source_root: Path,
    expected_manifest_hashes: dict[str, str] | None,
    input_features: str,
    coordinate_normalization_policy: str,
    executor: ThreadPoolExecutor,
    color_mean: tuple[float, ...] | None = None,
    color_std: tuple[float, ...] | None = None,
) -> list[Any]:
    """Load one ordered logical scene group from the trusted local archive."""

    futures = [
        executor.submit(
            _preload_sources,
            scene_ids=[scene_ids[scene_index]],
            source_root=source_root,
            smoke_manifest=None,
            expected_manifest_hashes=expected_manifest_hashes,
            input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            color_mean=color_mean,
            color_std=color_std,
            # The complete archive and every artifact were already verified by
            # the pack audit and extraction gate. Rehashing arrays on all 336
            # visits would change I/O cost, not integrity.
            verify_hashes=False,
        )
        for scene_index in scene_indices
    ]
    return [future.result()[0] for future in futures]


def _load_smoke_manifest_paths(path: Path) -> list[Path]:
    paths = [
        Path(line.strip()).resolve(strict=True)
        for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(paths) < 2:
        raise ValueError(
            "--smoke-manifest-list must contain at least two scene manifests"
        )
    if len(set(paths)) != len(paths):
        raise ValueError("--smoke-manifest-list contains duplicate paths")
    return paths


def _load_dataset_source_audit(
    *,
    path: Path,
    dataset: str,
    source_root: Path,
    train_split: Path,
    holdout_split: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.resolve(strict=True)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if dataset == "structured3d":
        if (
            audit.get("schema_version")
            != "structured3d_fullscene_source_audit/v1"
            or audit.get("experiment_id") != EXPERIMENT_ID
            or audit.get("dataset") != "structured3d"
            or audit.get("crop_mode") != "full_scene"
            or audit.get("artifact_hashes_verified") is not True
            or not audit.get("passed")
            or not all(audit.get("checks", {}).values())
        ):
            raise RuntimeError(
                f"Structured3D full-scene source audit failed or drifted: {path}"
            )
        split_contract = (
            ("optimization", train_split, 2930),
            ("validation", holdout_split, 326),
        )
        for split_name, current_path, expected_count in split_contract:
            record = audit.get("splits", {}).get(split_name, {})
            resolved_current = current_path.resolve(strict=True)
            if int(record.get("scene_count", -1)) != expected_count:
                raise RuntimeError(
                    f"Structured3D {split_name} scene-count drift in source audit"
                )
            if sha256_file(resolved_current) != record.get("sha256"):
                raise RuntimeError(
                    f"Structured3D {split_name} split hash differs from source audit"
                )
        rows = audit.get("scenes", [])
        expected_hashes = {
            str(row["scene_id"]): str(row["manifest_sha256"]) for row in rows
        }
        if (
            int(audit.get("scene_count", -1)) != 3256
            or int(audit.get("train_scene_count", -1)) != 2930
            or int(audit.get("validation_scene_count", -1)) != 326
            or len(rows) != 3256
            or len(expected_hashes) != 3256
        ):
            raise RuntimeError(
                f"Structured3D source-audit scene map drift: {path}"
            )
        expected_ids = set(load_scene_ids(train_split)) | set(
            load_scene_ids(holdout_split)
        )
        if set(expected_hashes) != expected_ids:
            raise RuntimeError(
                f"Structured3D source-audit membership differs from current splits: {path}"
            )
        report = {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": audit["schema_version"],
            "scene_count": int(audit["scene_count"]),
            "source_root": str(source_root.resolve(strict=True)),
            "audited_source_root": str(audit["source_root"]),
            "crop_mode": "full_scene",
            "artifact_hashes_verified": True,
            "point_count_summary": audit.get("point_count_summary"),
            "total_artifact_bytes": int(audit["total_artifact_bytes"]),
            "passed": True,
        }
        return report, expected_hashes
    if dataset != "scannet":
        raise ValueError(f"Unsupported source-audit dataset: {dataset!r}")
    if (
        audit.get("schema_version")
        != "litept_gt3_unsam3_dataset_source_audit/v1"
        or audit.get("experiment_id") != EXPERIMENT_ID
        or not audit.get("passed")
        or not all(audit.get("checks", {}).values())
    ):
        raise RuntimeError(f"Dataset source audit failed or drifted: {path}")
    if int(audit.get("scene_count", -1)) != OFFICIAL_TRAIN_SCENES:
        raise RuntimeError(f"Dataset source audit scene-count drift: {path}")
    if Path(audit["source_root"]).resolve(strict=True) != source_root.resolve(
        strict=True
    ):
        raise RuntimeError(f"Dataset source audit root drift: {path}")
    dataset_record = audit.get("dataset_manifest", {})
    dataset_manifest_path = Path(dataset_record["path"]).resolve(strict=True)
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    if dataset_manifest_sha256 != dataset_record["sha256"]:
        raise RuntimeError(f"Dataset manifest hash drift from source audit: {path}")
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    if (
        dataset_manifest.get("schema_version")
        != "litept_gt3_unsam3_dataset/v1"
        or dataset_manifest.get("experiment_id") != EXPERIMENT_ID
        or not all(dataset_manifest.get("checks", {}).values())
    ):
        raise RuntimeError(f"Dataset manifest failed or drifted: {dataset_manifest_path}")
    split_contract = (
        ("optimization", train_split),
        ("upstream_holdout", holdout_split),
    )
    for split_name, current_path in split_contract:
        record = dataset_manifest["splits"][split_name]
        resolved_current = current_path.resolve(strict=True)
        if Path(record["path"]).resolve(strict=True) != resolved_current:
            raise RuntimeError(f"{split_name} path differs from dataset manifest")
        current_hash = sha256_file(resolved_current)
        if (
            current_hash != record["file_sha256"]
            or current_hash != record["ordered_scene_digest"]
        ):
            raise RuntimeError(f"{split_name} hash differs from dataset manifest")
    rows = audit.get("scenes", [])
    expected_hashes = {
        str(row["scene_id"]): str(row["manifest_sha256"]) for row in rows
    }
    if (
        len(rows) != OFFICIAL_TRAIN_SCENES
        or len(expected_hashes) != OFFICIAL_TRAIN_SCENES
    ):
        raise RuntimeError(f"Dataset source audit scene map drift: {path}")
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": audit["schema_version"],
        "scene_count": int(audit["scene_count"]),
        "source_root": str(source_root.resolve(strict=True)),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "passed": True,
    }
    return report, expected_hashes


def _v2_frame_catalogs(
    *,
    scene: Any,
    device: torch.device,
    cells: tuple[str, ...],
) -> dict[str, tuple[FrameProposalCatalog, ...]]:
    """Move one scene's immutable frame-local proposal catalogs to a rank device."""

    output: dict[str, tuple[FrameProposalCatalog, ...]] = {}
    for cell in cells:
        if not cell.startswith("2d/"):
            raise ValueError(f"V2 multigranular sampling is 2D-only, got {cell!r}")
        granularity = cell.split("/", 1)[1]
        output[cell] = tuple(
            FrameProposalCatalog(
                cell=cell,
                frame_id=frame.physical_frame_id,
                visible_indices=torch.as_tensor(
                    frame.visible_indices,
                    dtype=torch.long,
                    device=device,
                ),
                proposal_offsets=torch.as_tensor(
                    frame.proposal_offsets,
                    dtype=torch.long,
                    device=device,
                ),
                proposal_point_indices=torch.as_tensor(
                    frame.proposal_point_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
            for frame in scene.frames(granularity, "train")
        )
    return output


def _run() -> None:
    args = parse_args()
    # Keep the training/sampling seed independent from the immutable artifacts
    # when a controlled seed repeat is requested.  Legacy invocations retain
    # the previous behavior because both defaults fall back to --seed.
    (
        args.initialization_seed,
        args.fixed_eval_seed,
    ) = _resolve_seed_contract(args)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size = int(
        args.expected_world_size
        if args.expected_world_size is not None
        else WORLD_SIZE
    )
    if expected_world_size <= 0:
        raise ValueError("--expected-world-size must be positive")
    if world_size != expected_world_size:
        raise RuntimeError(
            f"This experiment requires exactly {expected_world_size} ranks, "
            f"got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=30),
    )
    local_device_report = {
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
    }
    device_reports: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(device_reports, local_device_report)
    source_loader_pool: ThreadPoolExecutor | None = None
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        if (args.resume is None) != (args.resume_sha256 is None):
            raise ValueError("--resume and --resume-sha256 are required together")
        if args.resume_into_fresh_attempt and args.resume is None:
            raise ValueError("--resume-into-fresh-attempt requires --resume")
        if (args.execution_provenance is None) != (
            args.execution_provenance_sha256 is None
        ):
            raise ValueError(
                "--execution-provenance and its SHA-256 are required together"
            )
        resume_sha256: str | None = None
        if args.resume is not None:
            resume_path = args.resume.resolve(strict=True)
            resume_sha256 = sha256_file(resume_path)
            if resume_sha256 != args.resume_sha256:
                raise ValueError(
                    f"Resume checkpoint hash drift: {resume_sha256} != "
                    f"{args.resume_sha256}"
                )
            gathered_resume_hashes: list[str | None] = [None] * world_size
            dist.all_gather_object(gathered_resume_hashes, resume_sha256)
            if set(gathered_resume_hashes) != {resume_sha256}:
                raise RuntimeError(
                    f"Ranks disagree on resume checkpoint identity: "
                    f"{gathered_resume_hashes}"
                )
        execution_provenance_report: dict[str, Any] | None = None
        if args.execution_provenance is not None:
            provenance_path = args.execution_provenance.resolve(strict=True)
            provenance_sha256 = sha256_file(provenance_path)
            if provenance_sha256 != args.execution_provenance_sha256:
                raise ValueError(
                    "Execution-provenance hash differs from the Slurm-spooled "
                    "trust root"
                )
            gathered_provenance_hashes: list[str | None] = [None] * world_size
            dist.all_gather_object(
                gathered_provenance_hashes, provenance_sha256
            )
            if set(gathered_provenance_hashes) != {provenance_sha256}:
                raise RuntimeError(
                    "Ranks disagree on execution-provenance identity"
                )
            provenance_payload = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            if (
                provenance_payload.get("schema_version")
                not in {
                    "structured3d_rgbn6_fullscene_execution_provenance/v1",
                    "structured3d_rgbn6_sphere50k_execution_provenance/v1",
                    "structured3d_rgbn6_pretraining_v2_execution_provenance/v1",
                }
                or provenance_payload.get("passed") is not True
            ):
                raise ValueError("Execution-provenance manifest is not passed")
            execution_provenance_report = {
                "path": str(provenance_path),
                "sha256": provenance_sha256,
                "schema_version": provenance_payload["schema_version"],
                "trust_root": provenance_payload.get("trust_root"),
                "passed": True,
            }
        source_family = str(args.source_family)
        source_cells = source_cells_for_family(source_family)
        sampling_mode = str(args.sampling_mode)
        hierarchy_mode = str(args.hierarchy_supervision)
        hierarchy_enabled = hierarchy_mode in {"query_mask", "query_mask_vicreg"}
        if hierarchy_mode == "legacy" and not args.multiscale_supervision:
            raise ValueError(
                "--hierarchy-supervision legacy requires --multiscale-supervision"
            )
        if hierarchy_enabled and args.multiscale_supervision:
            raise ValueError(
                "Proposal-query hierarchy and legacy multiscale supervision are "
                "mutually exclusive"
            )
        if hierarchy_mode == "query_mask" and float(
            args.hierarchy_vicreg_loss_weight
        ) != 0.0:
            raise ValueError("query_mask requires zero VICReg loss weight")
        if hierarchy_mode == "query_mask_vicreg" and float(
            args.hierarchy_vicreg_loss_weight
        ) <= 0.0:
            raise ValueError("query_mask_vicreg requires a positive VICReg loss weight")
        try:
            raw_hierarchy_mask_stage_weights = json.loads(
                str(args.hierarchy_mask_stage_weights)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--hierarchy-mask-stage-weights must be a JSON object"
            ) from exc
        if not isinstance(raw_hierarchy_mask_stage_weights, dict):
            raise ValueError(
                "--hierarchy-mask-stage-weights must decode to an object"
            )
        hierarchy_mask_stage_weights = normalize_hierarchy_mask_stage_weights(
            raw_hierarchy_mask_stage_weights
        )
        try:
            raw_hierarchy_mask_stage_weight_schedule = json.loads(
                str(args.hierarchy_mask_stage_weight_schedule)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--hierarchy-mask-stage-weight-schedule must be JSON"
            ) from exc
        if raw_hierarchy_mask_stage_weight_schedule is not None and not isinstance(
            raw_hierarchy_mask_stage_weight_schedule,
            dict,
        ):
            raise ValueError(
                "--hierarchy-mask-stage-weight-schedule must decode to null or an object"
            )
        hierarchy_mask_stage_weight_schedule = (
            normalize_hierarchy_mask_stage_weight_schedule(
                raw_hierarchy_mask_stage_weight_schedule
            )
        )
        if any(
            value < 0.0
            for value in (
                float(args.hierarchy_mask_loss_weight),
                float(args.hierarchy_vicreg_loss_weight),
                float(args.hierarchy_covariance_weight),
            )
        ):
            raise ValueError("Hierarchy loss weights must be non-negative")
        if not 0.0 <= float(args.hierarchy_warmup_fraction) <= 1.0:
            raise ValueError("--hierarchy-warmup-fraction must lie in [0,1]")
        if not 0.0 <= float(args.feature_hard_warmup_fraction) <= 1.0:
            raise ValueError("--feature-hard-warmup-fraction must lie in [0,1]")
        if int(args.feature_hard_required_panels) <= 0:
            raise ValueError("--feature-hard-required-panels must be positive")
        if int(args.bn_recalibration_scenes) < 0:
            raise ValueError("--bn-recalibration-scenes must be non-negative")
        if sampling_mode == V2_SAMPLING_MODE:
            v2_errors: list[str] = []
            if args.dataset != "structured3d":
                v2_errors.append("dataset must be structured3d")
            if args.input_features != "rgbn6":
                v2_errors.append("input features must be rgbn6")
            if (
                args.coordinate_normalization_policy
                != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
            ):
                v2_errors.append(
                    "Structured3D stored Y-up points/normals must be canonicalized "
                    "to the ScanNet Z-up model frame"
                )
            if source_family != SOURCE_FAMILY_2D:
                v2_errors.append("source family must be 2d")
            if tuple(source_cells) != tuple(V2_DEFAULT_2D_CELLS):
                v2_errors.append("source cells must be g02/g05/g08")
            if args.sampling_profile != "pf_hnm_512_p16_v1":
                v2_errors.append("training sampling profile must be p16")
            if int(args.scenes_per_rank_per_update) != 6:
                v2_errors.append("six scenes per rank/update are required")
            if expected_world_size != 2:
                v2_errors.append("exactly two homogeneous GPU ranks are required")
            if args.representative_sampling != "first":
                v2_errors.append("representative sampling must be first")
            if args.required_crop_mode != "sphere_50k":
                v2_errors.append("crop mode must be sphere_50k")
            if hierarchy_mode == "off":
                v2_errors.append("an explicit legacy/query hierarchy arm is required")
            if int(args.bn_recalibration_scenes) <= 0:
                v2_errors.append("a fixed BN recalibration panel is required")
            if args.fixed_eval_pack is None:
                v2_errors.append("an immutable fixed evaluation pack is required")
            typed_device_reports = [
                report for report in device_reports if report is not None
            ]
            if len(typed_device_reports) != world_size or len(
                {report["name"] for report in typed_device_reports}
            ) != 1:
                v2_errors.append(
                    f"GPU ranks must be homogeneous, observed={typed_device_reports}"
                )
            if v2_errors:
                raise ValueError("Invalid V2 RGBN6 contract: " + "; ".join(v2_errors))
        train_augmentation_profile = str(args.train_augmentation_profile)
        color_mean = (
            tuple(float(value) for value in args.color_mean)
            if args.color_mean is not None
            else None
        )
        color_std = (
            tuple(float(value) for value in args.color_std)
            if args.color_std is not None
            else None
        )
        if (color_mean is None) != (color_std is None):
            raise ValueError("--color-mean and --color-std must be supplied together")
        if train_augmentation_profile in {
            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
            TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
        }:
            if color_mean is None:
                color_mean = tuple(COMMON_RGB_COLOR_MEAN)
                color_std = tuple(COMMON_RGB_COLOR_STD)
            if color_mean != tuple(COMMON_RGB_COLOR_MEAN) or color_std != tuple(
                COMMON_RGB_COLOR_STD
            ):
                raise ValueError(
                    "Normal-aware RGBN6 augmentation requires the fixed LitePT "
                    "RGB /255 transfer contract"
                )
        required_crop_mode = str(args.required_crop_mode)
        source_loading_mode = str(args.source_loading_mode)
        source_loader_workers = int(args.source_loader_workers)
        if source_loader_workers <= 0:
            raise ValueError("--source-loader-workers must be positive")
        augmentation_crop_modes = {
            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE: "full_scene",
            TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K: "sphere_50k",
        }
        if train_augmentation_profile in augmentation_crop_modes:
            if args.dataset != "structured3d":
                raise ValueError(
                    f"{train_augmentation_profile} is Structured3D-only"
                )
            if args.input_features != "rgbn6":
                raise ValueError(
                    f"{train_augmentation_profile} requires --input-features rgbn6"
                )
            augmentation_crop_mode = augmentation_crop_modes[
                train_augmentation_profile
            ]
            if required_crop_mode != augmentation_crop_mode:
                raise ValueError(
                    f"{train_augmentation_profile} requires "
                    f"--required-crop-mode {augmentation_crop_mode}"
                )
        scenes_per_rank_per_update = int(args.scenes_per_rank_per_update)
        if scenes_per_rank_per_update <= 0:
            raise ValueError("--scenes-per-rank-per-update must be positive")
        if source_family == SOURCE_FAMILY_2D and args.three_d_mode != "gt3":
            raise ValueError(
                "--three-d-mode is not used with --source-family 2d; omit it "
                "or leave the default"
            )
        if (
            args.smoke_manifest is not None
            and args.smoke_manifest_list is not None
        ):
            raise ValueError(
                "--smoke-manifest and --smoke-manifest-list are mutually exclusive"
            )
        smoke_manifest_paths = (
            _load_smoke_manifest_paths(args.smoke_manifest_list)
            if args.smoke_manifest_list is not None
            else [args.smoke_manifest.resolve(strict=True)]
            if args.smoke_manifest is not None
            else []
        )
        smoke = bool(smoke_manifest_paths)
        if (
            (args.multiscale_supervision or hierarchy_enabled or sampling_mode == V2_SAMPLING_MODE)
            and not smoke
            and args.execution_provenance is None
        ):
            raise ValueError(
                "Deep-supervision/V2 production requires frozen execution provenance"
            )
        if smoke and source_loading_mode != "preload":
            raise ValueError("Smoke runs require --source-loading-mode preload")
        if smoke:
            epochs = int(args.smoke_epochs)
            scene_visits_per_epoch_per_rank = len(smoke_manifest_paths)
            scenes_per_rank = scene_visits_per_epoch_per_rank
            local_scene_ids = [
                f"smoke-rank-{rank}-scene-{index}"
                for index in range(scene_visits_per_epoch_per_rank)
            ]
            holdout_scene_ids = [
                f"smoke-holdout-{index}"
                for index in range(scene_visits_per_epoch_per_rank)
            ]
            eval_epochs = {0, epochs}
            if epochs == 6:
                eval_epochs.add(3)
            source_audit_report = None
            expected_manifest_hashes = None
        else:
            train_scene_ids = load_scene_ids(args.train_split)
            holdout_scene_ids = load_scene_ids(args.holdout_split)
            if not train_scene_ids:
                raise ValueError("Training split is empty")
            if not holdout_scene_ids:
                raise ValueError("Validation/holdout split is empty")
            if len(train_scene_ids) % world_size != 0:
                raise ValueError(
                    "Training split must divide evenly across ranks: "
                    f"{len(train_scene_ids)} scenes / {world_size} ranks"
                )
            if args.dataset == "scannet":
                if len(train_scene_ids) != WORLD_SIZE * SCENES_PER_RANK:
                    raise ValueError(
                        f"Expected {WORLD_SIZE * SCENES_PER_RANK} ScanNet train "
                        f"scenes, got {len(train_scene_ids)}"
                    )
                if len(holdout_scene_ids) != 21:
                    raise ValueError(
                        "Expected 21 ScanNet upstream holdout scenes, "
                        f"got {len(holdout_scene_ids)}"
                    )
            if set(train_scene_ids) & set(holdout_scene_ids):
                raise ValueError("Training and validation/holdout splits overlap")
            scenes_per_rank = len(train_scene_ids) // world_size
            start = rank * scenes_per_rank
            local_scene_ids = train_scene_ids[start : start + scenes_per_rank]
            epochs = int(args.epochs)
            scene_visits_per_epoch_per_rank = scenes_per_rank
            eval_epochs = {
                epoch for epoch in DEFAULT_EVAL_EPOCHS if epoch <= epochs
            } | {epochs}
            if args.source_audit is not None or args.dataset == "scannet":
                source_audit_path = (
                    args.source_audit.resolve(strict=True)
                    if args.source_audit is not None
                    else args.source_root.parent
                    / "audits"
                    / "source_dataset_audit.json"
                )
                source_audit_report, expected_manifest_hashes = (
                    _load_dataset_source_audit(
                        path=source_audit_path,
                        dataset=args.dataset,
                        source_root=args.source_root,
                        train_split=args.train_split,
                        holdout_split=args.holdout_split,
                    )
                )
            else:
                # Structured3D has a different scene/source-manifest contract;
                # each manifest and every referenced artifact is still audited
                # by load_scene_source below, but there is not yet a ScanNet-
                # shaped dataset-level audit file to reuse here.
                source_audit_report = None
                expected_manifest_hashes = None
            if (
                source_loading_mode == "logical_batch"
                and expected_manifest_hashes is None
            ):
                raise ValueError(
                    "logical_batch source loading requires a frozen dataset "
                    "source audit"
                )

        nominal_scene_visits_per_rank = (
            int(epochs) * int(scene_visits_per_epoch_per_rank)
        )
        if (
            args.maximum_updates is None
            and nominal_scene_visits_per_rank % scenes_per_rank_per_update != 0
        ):
            raise ValueError(
                "The complete run must contain full per-rank logical batches: "
                f"scene_visits={nominal_scene_visits_per_rank} "
                f"scenes_per_rank_per_update={scenes_per_rank_per_update}"
            )
        if args.maximum_updates is None:
            total_updates = (
                nominal_scene_visits_per_rank // scenes_per_rank_per_update
            )
        else:
            total_updates = int(args.maximum_updates)
            if total_updates <= 0:
                raise ValueError("--maximum-updates must be positive")
            consumed = total_updates * scenes_per_rank_per_update
            if consumed > nominal_scene_visits_per_rank:
                raise ValueError(
                    "--maximum-updates exceeds the nominal scene horizon: "
                    f"consumed={consumed} available={nominal_scene_visits_per_rank}"
                )
            if sampling_mode != V2_SAMPLING_MODE:
                raise ValueError("--maximum-updates is reserved for V2 funnel panels")
        total_scene_visits_per_rank = total_updates * scenes_per_rank_per_update
        if args.pause_after_update is not None:
            if args.maximum_updates is not None:
                raise ValueError(
                    "--pause-after-update and --maximum-updates are mutually exclusive"
                )
            if sampling_mode != V2_SAMPLING_MODE:
                raise ValueError("--pause-after-update is reserved for V2 production")
            execution_stop_update = int(args.pause_after_update)
            if not 0 < execution_stop_update <= total_updates:
                raise ValueError("--pause-after-update lies outside the run horizon")
            if (
                execution_stop_update * scenes_per_rank_per_update
            ) % scene_visits_per_epoch_per_rank != 0:
                raise ValueError(
                    "--pause-after-update must land on an exact epoch boundary"
                )
        else:
            execution_stop_update = total_updates

        if args.evaluation_epochs is not None:
            eval_epochs = set(args.evaluation_epochs)
            if 0 not in eval_epochs:
                raise ValueError("--evaluation-epochs must include epoch 0")
            if any(value > epochs for value in eval_epochs):
                raise ValueError("An evaluation epoch exceeds --epochs")
        snapshot_updates = tuple(args.audit_snapshot_updates or ())
        snapshot_labels = tuple(args.audit_snapshot_labels or ())
        if bool(snapshot_updates) != bool(snapshot_labels) or len(
            snapshot_updates
        ) != len(snapshot_labels):
            raise ValueError(
                "Audit snapshot updates and labels must be supplied one-to-one"
            )
        if any(value <= 0 or value > total_updates for value in snapshot_updates):
            raise ValueError("Audit snapshot updates must lie inside the run horizon")
        if len(set(snapshot_labels)) != len(snapshot_labels):
            raise ValueError("Audit snapshot labels must be unique")
        snapshot_label_by_update = dict(zip(snapshot_updates, snapshot_labels))
        checkpoint_updates = set(args.checkpoint_updates or ())
        if not checkpoint_updates.issubset(set(snapshot_updates)):
            raise ValueError("Checkpoint updates must be audit snapshot updates")
        if scenes_per_rank_per_update > 1 and args.evaluation_epochs is None:
            if smoke:
                eval_epochs = {
                    epoch
                    for epoch in eval_epochs
                    if (
                        int(epoch) * int(scene_visits_per_epoch_per_rank)
                    )
                    % scenes_per_rank_per_update
                    == 0
                }
                eval_epochs.add(0)
                eval_epochs.add(epochs)
            else:
                eval_epochs = {
                    epoch
                    for epoch in LOGICAL_BATCH_EVAL_EPOCHS
                    if epoch <= epochs
                } | {epochs}
        checkpoint_epochs = (
            set(args.checkpoint_epochs)
            if args.checkpoint_epochs is not None
            else set(eval_epochs)
        )
        if not checkpoint_epochs.issubset(eval_epochs):
            raise ValueError("Checkpoint epochs must also be evaluation epochs")
        # Validate the final retention contract before model initialization or
        # GPU work. Maximum-update screens must end on an explicitly retained
        # audit snapshot; complete runs must retain their true final epoch.
        _final_evaluation_endpoint(
            nominal_epochs=epochs,
            total_updates=total_updates,
            maximum_updates=args.maximum_updates,
            snapshot_label_by_update=snapshot_label_by_update,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_updates=checkpoint_updates,
            checkpoint_dir=args.output_dir / "checkpoints",
        )
        unaligned_eval_epochs = sorted(
            epoch
            for epoch in eval_epochs
            if (
                int(epoch) * int(scene_visits_per_epoch_per_rank)
            )
            % scenes_per_rank_per_update
            != 0
        )
        if unaligned_eval_epochs:
            raise ValueError(
                "Evaluation epochs must end on complete logical batches: "
                f"{unaligned_eval_epochs}"
            )

        extension_enabled = args.extension_start_epoch is not None
        if extension_enabled != (args.extension_parent_checkpoint is not None):
            raise ValueError(
                "--extension-start-epoch and --extension-parent-checkpoint "
                "must be supplied together"
            )
        if extension_enabled and args.resume is None:
            raise ValueError("An extension requires --resume")
        if extension_enabled and not 0 < int(args.extension_start_epoch) < epochs:
            raise ValueError("Extension start epoch must be inside the target schedule")
        extension_start_epoch = (
            int(args.extension_start_epoch) if extension_enabled else 0
        )
        extension_start_update = _update_for_completed_epoch(
            epoch=extension_start_epoch,
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
        )
        if (
            args.sampling_profile == "pf_hnm_768_v1"
            and (args.fixed_eval_pack is None or args.code_manifest is None)
        ):
            raise ValueError(
                "pf_hnm_768_v1 requires --fixed-eval-pack and --code-manifest"
            )
        if (
            (args.multiscale_supervision or hierarchy_enabled or sampling_mode == V2_SAMPLING_MODE)
            and not smoke
            and args.code_manifest is None
        ):
            raise ValueError(
                "Deep-supervision/V2 production requires --code-manifest"
            )
        code_manifest_report: dict[str, Any] | None = None
        if args.code_manifest is not None:
            code_manifest_path = args.code_manifest.resolve(strict=True)
            code_manifest_payload = json.loads(
                code_manifest_path.read_text(encoding="utf-8")
            )
            accepted_code_manifest_schemas = {
                "litept_gt3_unsam3_clean1008_code_snapshot/v1",
                "chorus_rgbn6_sphere50k_pretrain_code/v1",
                "chorus_fullscene_rgbn6_multiscale_pretrain_code/v1",
                "structured3d_rgbn6_pretraining_v2_code_manifest/v1",
            }
            if (
                code_manifest_payload.get("schema_version")
                not in accepted_code_manifest_schemas
                or not code_manifest_payload.get("passed")
            ):
                raise ValueError(
                    f"Code-snapshot manifest did not pass: {code_manifest_path}"
                )
            code_manifest_report = {
                "path": str(code_manifest_path),
                "sha256": sha256_file(code_manifest_path),
                "schema_version": code_manifest_payload["schema_version"],
                "snapshot_root": code_manifest_payload.get(
                    "snapshot_root", code_manifest_payload.get("archive")
                ),
                "passed": True,
            }

        output_exists = torch.tensor(
            int(args.output_dir.exists()) if rank == 0 else 0,
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(output_exists, src=0)
        if int(output_exists) == 1 and args.resume is None:
            raise FileExistsError(
                f"Refusing to overwrite run directory: {args.output_dir}"
            )
        if int(output_exists) == 1 and args.resume_into_fresh_attempt:
            raise FileExistsError(
                "Fresh-attempt resume requires a new output directory: "
                f"{args.output_dir}"
            )
        if (
            int(output_exists) == 0
            and args.resume is not None
            and not extension_enabled
            and not args.resume_into_fresh_attempt
        ):
            raise FileNotFoundError(
                f"Resume requested but run directory is missing: {args.output_dir}"
            )
        if int(output_exists) == 0 and extension_enabled:
            if (
                args.resume.resolve(strict=True)
                != args.extension_parent_checkpoint.resolve(strict=True)
            ):
                raise ValueError(
                    "A new extension must resume directly from its pinned parent"
                )
        log.info(
            "rank=%d preparing %d source scenes with mode=%s workers=%d",
            rank,
            len(local_scene_ids),
            source_loading_mode,
            source_loader_workers,
        )
        local_source_audits: list[Any] = []
        if smoke:
            smoke_scenes = [
                load_scene_source(
                    path,
                    input_features=args.input_features,
                    coordinate_normalization_policy=(
                        args.coordinate_normalization_policy
                    ),
                    color_mean=color_mean,
                    color_std=color_std,
                    verify_hashes=True,
                )
                for path in smoke_manifest_paths
            ]
            if len({scene.scene_id for scene in smoke_scenes}) != len(smoke_scenes):
                raise ValueError("Smoke manifest list contains duplicate scene IDs")
            rotation = rank % len(smoke_scenes)
            local_scenes = smoke_scenes[rotation:] + smoke_scenes[:rotation]
        elif source_loading_mode == "preload":
            local_scenes = _preload_sources(
                scene_ids=local_scene_ids,
                source_root=args.source_root,
                smoke_manifest=None,
                expected_manifest_hashes=expected_manifest_hashes,
                input_features=args.input_features,
                coordinate_normalization_policy=(
                    args.coordinate_normalization_policy
                ),
                color_mean=color_mean,
                color_std=color_std,
            )
        else:
            local_source_audits = _audit_lazy_sources(
                scene_ids=local_scene_ids,
                source_root=args.source_root,
                expected_manifest_hashes=expected_manifest_hashes,
            )
            local_scenes = []
            source_loader_pool = ThreadPoolExecutor(
                max_workers=source_loader_workers,
                thread_name_prefix=f"scene-loader-rank{rank}",
            )
        if len(set(local_scene_ids)) != len(local_scene_ids) and not smoke:
            raise ValueError(f"rank={rank}: duplicate local source scene")
        if rank == 0:
            holdout_scenes = (
                [
                    load_scene_source(
                        path,
                        input_features=args.input_features,
                        coordinate_normalization_policy=(
                            args.coordinate_normalization_policy
                        ),
                        color_mean=color_mean,
                        color_std=color_std,
                        verify_hashes=True,
                    )
                    for path in smoke_manifest_paths
                ]
                if smoke
                else _preload_sources(
                    scene_ids=holdout_scene_ids,
                    source_root=args.source_root,
                    smoke_manifest=None,
                    expected_manifest_hashes=expected_manifest_hashes,
                    input_features=args.input_features,
                    coordinate_normalization_policy=(
                        args.coordinate_normalization_policy
                    ),
                    color_mean=color_mean,
                    color_std=color_std,
                )
            )
        else:
            holdout_scenes = []

        loaded_crop_modes = {scene.crop_mode for scene in local_scenes}
        loaded_crop_modes.update(audit.crop_mode for audit in local_source_audits)
        if rank == 0:
            loaded_crop_modes.update(scene.crop_mode for scene in holdout_scenes)
        if required_crop_mode != "any" and loaded_crop_modes != {
            required_crop_mode
        }:
            raise ValueError(
                "Loaded source crop mode differs from the required contract: "
                f"required={required_crop_mode!r} observed={sorted(loaded_crop_modes)}"
            )

        sampling = sampling_config_for_profile(args.sampling_profile)
        evaluation_sampling_profile = str(
            args.evaluation_sampling_profile or args.sampling_profile
        )
        if (
            evaluation_sampling_profile != args.sampling_profile
            and (
                args.sampling_profile,
                evaluation_sampling_profile,
            )
            != ("pf_hnm_512_p16_v1", "pf_hnm_512_v1")
        ):
            raise ValueError(
                "The only supported split train/evaluation sampling contract "
                "is pf_hnm_512_p16_v1 -> pf_hnm_512_v1"
            )
        evaluation_sampling = sampling_config_for_profile(
            evaluation_sampling_profile
        )
        fixed_eval_pack_report: dict[str, Any] | None = None
        if args.fixed_eval_pack is not None:
            report_broadcast: list[dict[str, Any] | None] = [None]
            if rank == 0:
                report_broadcast[0] = prepare_fixed_holdout_eval_pack(
                    pack_dir=args.fixed_eval_pack,
                    scenes=holdout_scenes,
                    sampling=evaluation_sampling,
                    sampling_profile=evaluation_sampling_profile,
                    seed=args.fixed_eval_seed,
                    cells=source_cells,
                )
            dist.broadcast_object_list(report_broadcast, src=0)
            fixed_eval_pack_report = report_broadcast[0]
            if fixed_eval_pack_report is None:
                raise RuntimeError("Fixed-eval pack report broadcast failed")
            dist.barrier()

        # Every arm loads an explicit, hash-verified initialization artifact.
        torch.manual_seed(args.seed)
        model = LitePTContrastiveModel(
            litept_root=args.litept_root,
            input_features=args.input_features,
            representative_sampling=args.representative_sampling,
            token_pure_dec0=(sampling_mode == V2_SAMPLING_MODE),
            multiscale_supervision=args.multiscale_supervision,
            multiscale_loss_weight=args.multiscale_loss_weight,
            multiscale_warmup_epochs=args.multiscale_warmup_epochs,
            hierarchy_supervision=hierarchy_enabled,
            hierarchy_mask_loss_weight=args.hierarchy_mask_loss_weight,
            hierarchy_mask_stage_weights=hierarchy_mask_stage_weights,
            hierarchy_mask_stage_weight_schedule=(
                hierarchy_mask_stage_weight_schedule
            ),
            hierarchy_query_projection_policy=(
                args.hierarchy_query_projection_policy
            ),
            hierarchy_vicreg_loss_weight=args.hierarchy_vicreg_loss_weight,
            hierarchy_warmup_fraction=args.hierarchy_warmup_fraction,
            hierarchy_variance_target=args.hierarchy_variance_target,
            hierarchy_covariance_weight=args.hierarchy_covariance_weight,
            hierarchy_vicreg_max_tokens=args.hierarchy_vicreg_max_tokens,
        ).to(device)
        initial_report = _strict_initial_load(
            model=model,
            path=args.initial_checkpoint,
            input_features=args.input_features,
            multiscale_supervision=args.multiscale_supervision,
            v2_auxiliary_initialization=(sampling_mode == V2_SAMPLING_MODE),
        )
        if initial_report["seed"] != int(args.initialization_seed):
            raise ValueError("Initialization seed differs from --initialization-seed")
        if args.normalization_policy == "sync_batchnorm":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
        optimizer_groups, optimizer_group_report = optimizer_parameter_groups(
            model,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_groups)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

        if smoke:
            warmup_updates = 1
        else:
            warmup_updates = _updates_for_scene_visit_duration(
                epochs=int(args.warmup_epochs),
                scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
                scenes_per_rank_per_update=scenes_per_rank_per_update,
            )
        scheduled_gradient_panel_updates = _scheduled_gradient_panel_updates(
            total_updates=total_updates,
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
            # ``checkpoint_epochs`` is the evaluator's source of truth for
            # checkpoint-native gradient-panel expectations.  It defaults to
            # all evaluation epochs, while an explicit subset must not create
            # undeclared panels in cumulative_health/v1.
            eval_epochs=sorted(checkpoint_epochs),
            snapshot_updates=snapshot_updates,
            auxiliary_enabled=bool(
                hierarchy_enabled or args.multiscale_supervision
            ),
        )
        extension_parent_report = None
        if extension_enabled:
            extension_parent_path = args.extension_parent_checkpoint.resolve(
                strict=True
            )
            parent_digest: list[str | None] = [
                sha256_file(extension_parent_path) if rank == 0 else None
            ]
            dist.broadcast_object_list(parent_digest, src=0)
            if parent_digest[0] is None:
                raise RuntimeError("Extension parent digest broadcast failed")
            extension_parent_report = {
                "path": str(extension_parent_path),
                "sha256": parent_digest[0],
                "epoch": extension_start_epoch,
                "update": extension_start_update,
            }
        resolved = {
            "schema_version": "litept_gt3_unsam3_pretrain_resolved/v1",
            "project_name": project_name(),
            "project_slug": project_slug(),
            "experiment_id": EXPERIMENT_ID,
            "dataset": args.dataset,
            "mode": (
                f"{args.dataset}_smoke" if smoke else f"{args.dataset}_production"
            ),
            "world_size": world_size,
            "seed": int(args.seed),
            "initialization_seed": int(args.initialization_seed),
            "fixed_eval_seed": int(args.fixed_eval_seed),
            "epochs": epochs,
            "updates_per_epoch": (
                scene_visits_per_epoch_per_rank
                if scenes_per_rank_per_update == 1
                else None
            ),
            "scene_visits_per_epoch_per_rank": scene_visits_per_epoch_per_rank,
            "nominal_scene_visits_per_rank": nominal_scene_visits_per_rank,
            "executed_scene_visits_per_rank": (
                execution_stop_update * scenes_per_rank_per_update
            ),
            "scenes_per_rank_per_update": scenes_per_rank_per_update,
            "scenes_per_global_update": scenes_per_rank_per_update * world_size,
            "total_updates": total_updates,
            "pause_after_update": (
                None
                if args.pause_after_update is None
                else int(args.pause_after_update)
            ),
            "execution_stop_update": execution_stop_update,
            "maximum_updates": (
                None if args.maximum_updates is None else int(args.maximum_updates)
            ),
            "warmup_epochs": 1 if smoke else int(args.warmup_epochs),
            "warmup_updates": warmup_updates,
            "backbone_lr": float(args.backbone_lr),
            "head_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
            "min_lr_factor": 0.01,
            "lr_schedule": (
                {
                    "kind": "completed_checkpoint_cosine_restart",
                    "start_epoch": extension_start_epoch,
                    "start_update": extension_start_update,
                    "warmup_epochs": 1 if smoke else int(args.warmup_epochs),
                    "warmup_updates": warmup_updates,
                    "start_factor": 0.01,
                    "peak_factor": 1.0,
                    "final_factor": 0.01,
                }
                if extension_enabled
                else {
                    "kind": "linear_warmup_cosine",
                    "start_epoch": 0,
                    "start_update": 0,
                    "warmup_epochs": 1 if smoke else int(args.warmup_epochs),
                    "warmup_updates": warmup_updates,
                    "start_factor": 0.0,
                    "peak_factor": 1.0,
                    "final_factor": 0.01,
                }
            ),
            "continuation": (
                {
                    "enabled": True,
                    "parent_checkpoint": extension_parent_report,
                    "added_epochs": epochs - extension_start_epoch,
                    "added_updates": total_updates - extension_start_update,
                }
                if extension_enabled
                else {"enabled": False}
            ),
            "precision": "fp32",
            "device_reports": device_reports,
            "input_features": args.input_features,
            "input_channels": int(model.input_channels),
            "coordinate_normalization_policy": str(
                args.coordinate_normalization_policy
            ),
            "model_coordinate_frame": (
                "scannet_z_up_v1"
                if args.coordinate_normalization_policy
                == STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
                else "source_native"
            ),
            "rgb_color_mean": list(color_mean) if color_mean is not None else None,
            "rgb_color_std": list(color_std) if color_std is not None else None,
            "rgb_color_normalization": (
                RGBN6_COLOR_NORMALIZATION if color_mean is not None else None
            ),
            "pretraining_recipe": model.recipe_name,
            "sampling_mode": sampling_mode,
            "token_pure_dec0": bool(model.token_pure_dec0),
            "hierarchy_supervision": {
                "contract_version": "litept_query_mask_hierarchy/v2",
                "mode": hierarchy_mode,
                "enabled": hierarchy_enabled,
                "mask_loss_weight": float(args.hierarchy_mask_loss_weight),
                "mask_stage_weights": dict(hierarchy_mask_stage_weights),
                "mask_stage_weight_schedule": hierarchy_mask_stage_weight_schedule,
                "query_projection_policy": str(
                    args.hierarchy_query_projection_policy
                ),
                "vicreg_loss_weight": float(args.hierarchy_vicreg_loss_weight),
                "warmup_fraction": float(args.hierarchy_warmup_fraction),
                "variance_target": float(args.hierarchy_variance_target),
                "covariance_weight": float(args.hierarchy_covariance_weight),
                "vicreg_max_tokens": int(args.hierarchy_vicreg_max_tokens),
                "query_source": "detached_native_dec0_two_fold_pure_tokens",
                "query_projection": "layernorm_then_linear_128",
                "target": "positive_known_over_total_known",
                "unknown_and_mixed_policy": "ignored",
                "g02_route": (
                    "dec2_iff_each_shared_fold_has_at_least_two_positive_known_"
                    "tokens_and_nonzero_known_negative_mass_else_dec1_exactly_once"
                ),
                "g05_route": ["dec3", "dec2"],
                "g08_route": ["enc4", "dec3"],
                "vicreg_statistics": (
                    "distributed_scene_balanced_native_fp32_global_min_rank_tokens"
                ),
            },
            "normalization": {
                "policy": str(args.normalization_policy),
                "bn_recalibration_scenes": int(args.bn_recalibration_scenes),
                "retained_checkpoint_recalibration": "required_sidecar",
            },
            "feature_hard_gate": {
                "warmup_fraction": float(args.feature_hard_warmup_fraction),
                "cosine_gap_threshold": float(args.feature_hard_gap_threshold),
                "required_consecutive_panels": int(
                    args.feature_hard_required_panels
                ),
                "required_metric_space": "native_token_pure_dec0_fixed_pack",
                "dense_point_holdout_is_diagnostic_only": True,
            },
            "gradient_conflict_policy": {
                "cosine_threshold": -0.05,
                "required_consecutive_panels": 2,
                "first_action": "halve_objective_coefficient_once",
                "controller_gradient": (
                    "globally_summed_six_microbatch_shared_backbone_gradient"
                ),
                "second_action": "shared_backbone_pcgrad_vs_final_dec0",
                "private_head_policy": "unprojected_total_gradient",
                "third_action": "reject_candidate",
                "total_auxiliary_gradient_budget_vs_primary": 0.5,
            },
            "gradient_panel_schedule": {
                "schema_version": GRADIENT_PANEL_CUMULATIVE_SCHEMA,
                "updates": scheduled_gradient_panel_updates,
                "sources": [
                    "first_optimizer_update",
                    "evaluation_epoch_boundaries",
                    "audit_snapshot_updates",
                ],
            },
            "multiscale_supervision": bool(args.multiscale_supervision),
            "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
            "multiscale_level_loss_weights": dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            ),
            "multiscale_loss_weight": float(args.multiscale_loss_weight),
            "multiscale_warmup_epochs": int(args.multiscale_warmup_epochs),
            "multiscale_token_policy": (
                "strict_pure_token_resampling_v1"
                if args.multiscale_supervision
                else None
            ),
            "multiscale_temperature_policy": (
                "separate_learnable_temperature_per_auxiliary_level"
                if args.multiscale_supervision
                else None
            ),
            "multiscale_feature_hard_policy": (
                "separate_candidate_selection_per_positive_endpoint"
                if args.multiscale_supervision
                else None
            ),
            "multiscale_transfer_contract": (
                "native_mask3d_fpn_required"
                if args.multiscale_supervision or hierarchy_enabled
                else "final_dec0_feature"
            ),
            "checkpoint_selection_policy": (
                "external_five_stage_health_then_fixed_dec0_metric"
                if sampling_mode == V2_SAMPLING_MODE
                else "final_dec0_native_selection_macro"
                if args.multiscale_supervision
                else "native_selection_macro"
            ),
            "train_augmentation_profile": train_augmentation_profile,
            "train_augmentation": augmentation_contract_for_profile(
                train_augmentation_profile
            ),
            "evaluation_augmentation_profile": "none",
            "required_crop_mode": required_crop_mode,
            "loaded_crop_modes_rank0": sorted(loaded_crop_modes),
            "point_input_contract": (
                "all_source_points_preserved_before_2cm_voxelization"
                if required_crop_mode == "full_scene"
                else "all_stored_source_points_before_2cm_voxelization"
            ),
            "source_loading_mode": source_loading_mode,
            "source_loader_workers": source_loader_workers,
            "source_archive_integrity": (
                "frozen_archive_and_source_audit_verified_before_lazy_loading"
                if source_loading_mode == "logical_batch"
                else "per_scene_artifact_hashes_verified_during_preload"
            ),
            "grad_clip_norm": float(args.grad_clip_norm),
            "cache_training_voxelization": bool(
                model.backbone.cache_training_voxelization
            ),
            "voxel_reduce": model.backbone.voxel_reduce,
            "representative_sampling": model.backbone.representative_sampling,
            "voxelization_mode": "per_forward_single_scene_recompute",
            "source_family": source_family,
            "source_cells": list(source_cells),
            "source_cell_schedule": args.source_cell_schedule,
            "source_cell_schedule_seed": int(args.seed),
            "three_d_mode": None if source_family == SOURCE_FAMILY_2D else args.three_d_mode,
            "three_d_slot_to_label": (
                {}
                if source_family == SOURCE_FAMILY_2D
                else (
                    {
                        "3d/g02": "g02",
                        "3d/g05": "g05",
                        "3d/g08": "g08",
                    }
                    if args.three_d_mode == "gt3"
                    else {
                        "3d/g02": "g08",
                        "3d/g05": "g08",
                        "3d/g08": "g08",
                    }
                )
            ),
            "sampling_profile": args.sampling_profile,
            "sampling": vars(sampling),
            "sampling_v2_contract": (
                {
                    "schema_version": "multigranular_frame_local_coverage/v2",
                    "quota": "rotating_6_5_5_without_replacement",
                    "loss_group": "physical_frame_local",
                    "negative_known_support": (
                        "visible_intersection_same_frame_all_granularity_relation_union"
                    ),
                    "post_voxel_policy": (
                        "exact_three_cell_token_pure_or_synchronized_zero_skip"
                    ),
                    "resume_policy": (
                        "epoch_keyed_deterministic_reconstruction_with_prior_epoch_"
                        "cursor_audit_validation"
                    ),
                }
                if sampling_mode == V2_SAMPLING_MODE
                else None
            ),
            "evaluation_sampling_profile": evaluation_sampling_profile,
            "evaluation_sampling": vars(evaluation_sampling),
            "evaluation_proposals_per_forward": int(
                evaluation_sampling.proposals_per_forward
            ),
            "proposals_per_rank_per_update": (
                scenes_per_rank_per_update * int(sampling.proposals_per_forward)
            ),
            "proposals_per_global_update": (
                world_size
                * scenes_per_rank_per_update
                * int(sampling.proposals_per_forward)
            ),
            "fixed_eval_pack": fixed_eval_pack_report,
            "code_manifest": code_manifest_report,
            "eval_epochs": sorted(eval_epochs),
            "checkpoint_epochs": sorted(checkpoint_epochs),
            "audit_snapshot_updates": [
                {
                    "update": int(snapshot_update),
                    "label": int(snapshot_label_by_update[snapshot_update]),
                    "checkpoint": snapshot_update in checkpoint_updates,
                }
                for snapshot_update in snapshot_updates
            ],
            "initialization": initial_report,
            "execution_provenance": execution_provenance_report,
            "resume_parent": (
                {
                    "sha256": resume_sha256,
                }
                if resume_sha256 is not None
                else None
            ),
            "resume_into_fresh_attempt": bool(
                args.resume_into_fresh_attempt
            ),
            "source_dataset_audit": source_audit_report,
            "optimizer_parameter_groups": optimizer_group_report,
            "train_split": (
                {
                    "path": str(args.train_split.resolve(strict=True)),
                    "sha256": sha256_file(args.train_split),
                    "ordered_digest": split_digest(
                        load_scene_ids(args.train_split)
                    ),
                }
                if not smoke
                else None
            ),
            "holdout_split": (
                {
                    "path": str(args.holdout_split.resolve(strict=True)),
                    "sha256": sha256_file(args.holdout_split),
                    "ordered_digest": split_digest(
                        load_scene_ids(args.holdout_split)
                    ),
                }
                if not smoke
                else None
            ),
            "local_scene_ids": {
                str(candidate_rank): (
                    load_scene_ids(args.train_split)[
                        candidate_rank * scenes_per_rank :
                        (candidate_rank + 1) * scenes_per_rank
                    ]
                    if not smoke
                    else [
                        smoke_scenes[
                            (candidate_rank + index) % len(smoke_scenes)
                        ].scene_id
                        for index in range(len(smoke_scenes))
                    ]
                )
                for candidate_rank in range(expected_world_size)
            },
            "holdout_scene_ids": (
                load_scene_ids(args.holdout_split)
                if not smoke
                else [scene.scene_id for scene in smoke_scenes]
            ),
            "public_checkpoint_used": bool(
                initial_report["public_checkpoint_used"]
            ),
        }
        resolved["semantic_config_sha256"] = _semantic_config_sha256(resolved)
        metrics_path = args.output_dir / "metrics.jsonl"

        start_epoch = 0
        inherited_evaluation: dict[str, Any] | None = None
        inherited_evaluations: dict[int, dict[str, Any]] = {}
        lineage_parent_checkpoints: list[dict[str, Any]] = []
        extension_bootstrap = bool(extension_enabled and int(output_exists) == 0)
        rank_exposure_counts: dict[str, int] = {cell: 0 for cell in source_cells}
        rank_coverage_states: dict[str, Any] = {}
        rank_sampling_totals: dict[str, dict[str, int]] = {
            cell: {
                "requested_proposals": 0,
                "realized_unique_proposals": 0,
                "contrastive_retained_proposals": 0,
                "duplicate_proposals": 0,
                "realized_positive_pairs": 0,
                "requested_positive_pairs": 0,
                "active_scene_visits": 0,
                "loss_mass_millionths": 0,
                "optimized_scene_visits": 0,
                "invalid_scene_visits": 0,
                "token_realized_positive_pairs": 0,
                "token_requested_positive_pairs": 0,
                "optimized_loss_mass_millionths": 0,
            }
            for cell in source_cells
        }
        feature_hard_state: dict[str, Any] = {
            "schema_version": "litept_feature_hard_gate/v1",
            "enabled": False,
            "qualifying_panel_streak": 0,
            "panel_count": 0,
            "last_panel_update": None,
            "last_panel_cosine_gap": None,
            "last_dense_point_diagnostic_cosine_gap": None,
            "metric_space": "native_token_pure_dec0_fixed_pack",
            "dense_point_holdout_eligible_for_unlock": False,
            "native_token_metric_eligible_for_unlock": False,
            "activation_update": None,
        }
        conflict_objective_names = (
            ("hierarchy_mask_scaled", "hierarchy_vicreg_scaled")
            if hierarchy_enabled
            else ("legacy_auxiliary_scaled",)
            if args.multiscale_supervision
            else ()
        )
        gradient_conflict_state: dict[str, Any] = {
            "schema_version": "litept_auxiliary_conflict_controller/v2",
            "threshold": -0.05,
            "required_consecutive_panels": 2,
            "rejected": False,
            "rejection_reason": None,
            "auxiliary_budget": {
                "limit": 0.5,
                "last_global_sum_aux_norm_ratio": None,
                "violation_count": 0,
                "passed": None,
            },
            "objectives": {
                objective: {
                    "panel_streak": 0,
                    "coefficient_halved": False,
                    "pcgrad_enabled": False,
                    "post_pcgrad_failure": False,
                    "last_cosine": None,
                    "last_action_update": None,
                }
                for objective in conflict_objective_names
            },
        }
        rank_cumulative_health: dict[str, Any] | None = (
            _new_rank_cumulative_health(
                rank=rank,
                cells=source_cells,
                hierarchy_enabled=hierarchy_enabled,
                expected_microbatches_per_update=scenes_per_rank_per_update,
                legacy_multiscale_enabled=bool(
                    args.multiscale_supervision and not hierarchy_enabled
                ),
                scheduled_gradient_panel_updates=(
                    scheduled_gradient_panel_updates
                ),
            )
            if sampling_mode == V2_SAMPLING_MODE or args.multiscale_supervision
            else None
        )
        if args.resume is not None:
            checkpoint = torch.load(
                args.resume, map_location="cpu", weights_only=False
            )
            accepted_checkpoint_schemas = {
                "litept_gt3_unsam3_pretrain_checkpoint/v1",
                "litept_rgbn6_multigranular_pretrain_checkpoint/v2",
            }
            if checkpoint.get("schema_version") not in accepted_checkpoint_schemas:
                raise ValueError("Resume checkpoint schema drift")
            if checkpoint.get("experiment_id") != EXPERIMENT_ID:
                raise ValueError("Resume checkpoint experiment drift")
            raw_evaluation_history = checkpoint.get("evaluation_history")
            if not isinstance(raw_evaluation_history, dict):
                if args.resume_into_fresh_attempt:
                    raise ValueError(
                        "Fresh-attempt resume requires checkpoint evaluation history"
                    )
            else:
                inherited_evaluations = {
                    int(epoch): dict(value)
                    for epoch, value in raw_evaluation_history.items()
                }
            resolved["resume_parent"] = {
                "sha256": resume_sha256,
                "schema_version": checkpoint["schema_version"],
                "epoch": int(checkpoint.get("epoch", -1)),
                "update": int(checkpoint.get("update", -1)),
            }
            previous = checkpoint["resolved_config"]
            if (
                args.multiscale_supervision or sampling_mode == V2_SAMPLING_MODE
            ) and not extension_enabled:
                previous_digest = previous.get("semantic_config_sha256")
                if not isinstance(previous_digest, str):
                    raise ValueError(
                        "Multi-scale resume lacks a semantic-config digest"
                    )
                observed_previous_digest = _semantic_config_sha256(previous)
                if previous_digest != observed_previous_digest:
                    raise ValueError(
                        "Resume checkpoint semantic-config digest is corrupt"
                    )
                if previous_digest != resolved["semantic_config_sha256"]:
                    raise ValueError(
                        "Complete semantic configuration differs on resume"
                    )
            base_immutable = (
                "dataset",
                "world_size",
                "seed",
                "initialization_seed",
                "fixed_eval_seed",
                "updates_per_epoch",
                "scene_visits_per_epoch_per_rank",
                "scenes_per_rank_per_update",
                "source_loading_mode",
                "source_loader_workers",
                "input_features",
                "input_channels",
                "coordinate_normalization_policy",
                "model_coordinate_frame",
                "train_augmentation_profile",
                "train_augmentation",
                "evaluation_augmentation_profile",
                "required_crop_mode",
                "pretraining_recipe",
                "sampling_mode",
                "token_pure_dec0",
                "hierarchy_supervision",
                "normalization",
                "feature_hard_gate",
                "gradient_conflict_policy",
                "multiscale_supervision",
                "multiscale_levels",
                "multiscale_level_loss_weights",
                "multiscale_loss_weight",
                "multiscale_warmup_epochs",
                "multiscale_token_policy",
                "multiscale_temperature_policy",
                "multiscale_feature_hard_policy",
                "multiscale_transfer_contract",
                "checkpoint_selection_policy",
                "backbone_lr",
                "head_lr",
                "weight_decay",
                "grad_clip_norm",
                "initialization",
                "source_dataset_audit",
                "three_d_mode",
                "source_family",
                "source_cells",
                "source_cell_schedule",
                "source_cell_schedule_seed",
                "sampling_profile",
                "sampling",
                "sampling_v2_contract",
                "evaluation_sampling_profile",
                "evaluation_sampling",
                "cache_training_voxelization",
                "voxel_reduce",
                "voxelization_mode",
                "fixed_eval_pack",
                "code_manifest",
                "train_split",
                "holdout_split",
            )
            base_mismatches: list[str] = []
            for key in base_immutable:
                previous_value = previous.get(key)
                resolved_value = resolved.get(key)
                if key == "initialization":
                    previous_value = _content_addressed_report(
                        normalize_initialization_report(previous_value)
                    )
                    resolved_value = _content_addressed_report(
                        normalize_initialization_report(resolved_value)
                    )
                elif key in {
                    "train_split",
                    "holdout_split",
                    "fixed_eval_pack",
                    "code_manifest",
                    "execution_provenance",
                    "source_dataset_audit",
                }:
                    previous_value = _content_addressed_report(previous_value)
                    resolved_value = _content_addressed_report(resolved_value)
                elif key == "sampling_profile" and previous_value is None:
                    if previous.get("sampling") == vars(
                        sampling_config_for_profile("pf_hnm_512_v1")
                    ):
                        previous_value = "pf_hnm_512_v1"
                elif (
                    key == "evaluation_sampling_profile"
                    and previous_value is None
                ):
                    previous_value = previous.get(
                        "sampling_profile", "pf_hnm_512_v1"
                    )
                elif key == "evaluation_sampling" and previous_value is None:
                    previous_value = previous.get("sampling")
                elif key == "scene_visits_per_epoch_per_rank" and previous_value is None:
                    previous_value = previous.get("updates_per_epoch")
                elif key == "scenes_per_rank_per_update" and previous_value is None:
                    previous_value = 1
                elif key == "source_loading_mode" and previous_value is None:
                    previous_value = "preload"
                elif key == "source_loader_workers" and previous_value is None:
                    previous_value = 1
                elif key == "train_augmentation_profile" and previous_value is None:
                    previous_value = "none"
                elif key == "train_augmentation" and previous_value is None:
                    previous_value = augmentation_contract_for_profile("none")
                elif key == "evaluation_augmentation_profile" and previous_value is None:
                    previous_value = "none"
                elif key == "pretraining_recipe" and previous_value is None:
                    previous_value = "final_dec0_only_v1"
                elif key == "multiscale_supervision" and previous_value is None:
                    previous_value = False
                elif key == "multiscale_levels" and previous_value is None:
                    previous_value = list(MULTISCALE_LEVEL_NAMES)
                elif (
                    key == "multiscale_level_loss_weights"
                    and previous_value is None
                ):
                    previous_value = dict(MULTISCALE_LEVEL_LOSS_WEIGHTS)
                elif key == "multiscale_loss_weight" and previous_value is None:
                    previous_value = MULTISCALE_AUXILIARY_WEIGHT
                elif key == "multiscale_warmup_epochs" and previous_value is None:
                    previous_value = MULTISCALE_AUXILIARY_WARMUP_EPOCHS
                elif key == "multiscale_token_policy" and previous_value is None:
                    previous_value = (
                        "strict_pure_token_resampling_v1"
                        if bool(previous.get("multiscale_supervision", False))
                        else None
                    )
                elif (
                    key == "multiscale_temperature_policy"
                    and previous_value is None
                ):
                    previous_value = (
                        "separate_learnable_temperature_per_auxiliary_level"
                        if bool(previous.get("multiscale_supervision", False))
                        else None
                    )
                elif (
                    key == "multiscale_feature_hard_policy"
                    and previous_value is None
                ):
                    previous_value = (
                        "separate_candidate_selection_per_positive_endpoint"
                        if bool(previous.get("multiscale_supervision", False))
                        else None
                    )
                elif key == "multiscale_transfer_contract" and previous_value is None:
                    previous_value = "final_dec0_feature"
                elif key == "required_crop_mode" and previous_value is None:
                    previous_value = "any"
                elif key == "source_cell_schedule" and previous_value is None:
                    previous_value = SOURCE_CELL_SCHEDULE_FIXED
                elif key == "source_cell_schedule_seed" and previous_value is None:
                    previous_value = previous.get("seed")
                elif (
                    key == "cache_training_voxelization"
                    and previous_value is None
                    and args.sampling_profile == "pf_hnm_512_v1"
                ):
                    previous_value = False
                elif (
                    key == "voxelization_mode"
                    and previous_value is None
                    and args.sampling_profile == "pf_hnm_512_v1"
                ):
                    previous_value = "per_forward_single_scene_recompute"
                elif key == "source_family" and previous_value is None:
                    previous_cells = previous.get("source_cells")
                    if previous_cells is None or list(previous_cells) == list(
                        SOURCE_CELLS
                    ):
                        previous_value = "2d3d"
                elif (
                    key == "three_d_mode"
                    and previous_value is None
                    and resolved_value is None
                    and source_family == SOURCE_FAMILY_2D
                ):
                    previous_value = None
                if previous_value != resolved_value:
                    base_mismatches.append(key)
            if base_mismatches:
                raise ValueError(
                    "Resume base configuration differs from current run: "
                    + ", ".join(base_mismatches)
                )
            if extension_bootstrap:
                if int(checkpoint.get("epoch", -1)) != extension_start_epoch:
                    raise ValueError(
                        "Extension parent epoch differs from --extension-start-epoch"
                    )
                if int(checkpoint.get("update", -1)) != extension_start_update:
                    raise ValueError("Extension parent update differs from start update")
                if (
                    int(previous.get("epochs", -1)) != extension_start_epoch
                    or int(previous.get("total_updates", -1))
                    != extension_start_update
                ):
                    raise ValueError(
                        "Extension parent is not the completed source schedule"
                    )
                if sampling_mode == V2_SAMPLING_MODE:
                    parent_rank_counts = checkpoint.get("rank_exposure_counts")
                    if (
                        checkpoint.get("resumable") is not True
                        or not isinstance(parent_rank_counts, list)
                        or checkpoint.get("global_exposure_counts")
                        != _merge_count_dicts(parent_rank_counts)
                    ):
                        raise ValueError(
                            "V2 extension parent exposure/replay state is not exact"
                        )
                else:
                    expected_parent_counts = exposure_counts(
                        load_scene_ids(args.train_split),
                        epochs=extension_start_epoch,
                        cells=source_cells,
                        proposals_per_forward=sampling.proposals_per_forward,
                        cell_schedule=args.source_cell_schedule,
                        seed=args.seed,
                    )
                    if checkpoint.get("global_exposure_counts") != expected_parent_counts:
                        raise ValueError("Extension parent exposure counts are not exact")
                inherited_evaluation = checkpoint.get("evaluation")
                if not isinstance(inherited_evaluation, dict):
                    raise ValueError("Extension parent lacks its boundary evaluation")
            else:
                exact_immutable = (
                    "epochs",
                    "total_updates",
                    "warmup_updates",
                    "eval_epochs",
                    "lr_schedule",
                    "continuation",
                )
                if any(
                    previous.get(key) != resolved.get(key)
                    for key in exact_immutable
                ):
                    raise ValueError("Exact resume configuration differs from current run")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"])
            rank_exposure_counts = {
                key: int(value)
                for key, value in checkpoint["rank_exposure_counts"][rank].items()
            }
            if sampling_mode == V2_SAMPLING_MODE:
                if checkpoint.get("resumable") is not True:
                    raise ValueError("V2 resume requires a resumable epoch checkpoint")
                rank_coverage_records = checkpoint.get("rank_coverage_states")
                rank_rng_records = checkpoint.get("rank_rng_states")
                if (
                    not isinstance(rank_coverage_records, list)
                    or len(rank_coverage_records) != world_size
                    or not isinstance(rank_rng_records, list)
                    or len(rank_rng_records) != world_size
                ):
                    raise ValueError("V2 resume lacks complete per-rank state")
                rank_coverage_states = dict(rank_coverage_records[rank])
                for state_key, state_record in rank_coverage_states.items():
                    if state_record is None:
                        continue
                    state = DeterministicCoverageState.from_dict(state_record)
                    scene_id, separator, sampled_cell = str(state_key).partition("|")
                    if (
                        not separator
                        or state.scene_id != scene_id
                        or not state.operation.endswith(f"/{sampled_cell}")
                        or int(state.epoch) != int(start_epoch) - 1
                        or int(state.cursor) <= 0
                    ):
                        raise ValueError(
                            "V2 saved coverage audit cursor is inconsistent with "
                            "the resumable epoch boundary"
                        )
                saved_sampling_totals = checkpoint.get("rank_sampling_totals")
                if (
                    not isinstance(saved_sampling_totals, list)
                    or len(saved_sampling_totals) != world_size
                ):
                    raise ValueError("V2 resume lacks cumulative sampling totals")
                rank_sampling_totals = {
                    cell: {
                        key: int(value) for key, value in values.items()
                    }
                    for cell, values in saved_sampling_totals[rank].items()
                }
                saved_feature_hard = checkpoint.get("feature_hard_state")
                if not isinstance(saved_feature_hard, dict):
                    raise ValueError("V2 resume lacks feature-hard gate state")
                feature_hard_state = dict(saved_feature_hard)
                feature_hard_state.setdefault(
                    "last_dense_point_diagnostic_cosine_gap", None
                )
                feature_hard_state.setdefault(
                    "metric_space", "native_token_pure_dec0_fixed_pack"
                )
                feature_hard_state.setdefault(
                    "dense_point_holdout_eligible_for_unlock", False
                )
                feature_hard_state.setdefault(
                    "native_token_metric_eligible_for_unlock", False
                )
                saved_conflict_state = checkpoint.get(
                    "gradient_conflict_state"
                )
                if not isinstance(saved_conflict_state, dict):
                    raise ValueError("V2 resume lacks gradient-conflict state")
                gradient_conflict_state = dict(saved_conflict_state)
                if bool(gradient_conflict_state.get("rejected", False)):
                    raise ValueError(
                        "Refusing to resume a checkpoint whose gradient controller "
                        "has already rejected the candidate"
                    )
                gradient_conflict_state.setdefault("rejection_reason", None)
                gradient_conflict_state.setdefault(
                    "auxiliary_budget",
                    {
                        "limit": 0.5,
                        "last_global_sum_aux_norm_ratio": None,
                        "violation_count": 0,
                        "passed": None,
                    },
                )
                saved_cumulative_health = checkpoint.get("cumulative_health")
                if not isinstance(saved_cumulative_health, Mapping):
                    raise ValueError("V2 resume lacks cumulative health state")
                if saved_cumulative_health.get(
                    "gradient_conflict_controller"
                ) != gradient_conflict_state:
                    raise ValueError(
                        "V2 resume cumulative/controller state is inconsistent"
                    )
                rank_cumulative_health = _restore_rank_cumulative_health(
                    saved_cumulative_health,
                    rank=rank,
                    checkpoint_update=int(checkpoint["update"]),
                    world_size=world_size,
                    scheduled_gradient_panel_updates=(
                        scheduled_gradient_panel_updates
                    ),
                    allow_schedule_extension=extension_enabled,
                )
                _restore_rng_state(rank_rng_records[rank], device)
            elif args.multiscale_supervision:
                if checkpoint.get("resumable") is not True:
                    raise ValueError(
                        "Legacy multiscale resume requires a resumable epoch checkpoint"
                    )
                rank_rng_records = checkpoint.get("rank_rng_states")
                if (
                    not isinstance(rank_rng_records, list)
                    or len(rank_rng_records) != world_size
                ):
                    raise ValueError("Legacy multiscale resume lacks complete RNG state")
                saved_conflict_state = checkpoint.get("gradient_conflict_state")
                if not isinstance(saved_conflict_state, dict):
                    raise ValueError(
                        "Legacy multiscale resume lacks gradient-conflict state"
                    )
                gradient_conflict_state = dict(saved_conflict_state)
                if bool(gradient_conflict_state.get("rejected", False)):
                    raise ValueError(
                        "Refusing to resume a checkpoint whose gradient controller "
                        "has already rejected the candidate"
                    )
                saved_cumulative_health = checkpoint.get("cumulative_health")
                if not isinstance(saved_cumulative_health, Mapping):
                    raise ValueError(
                        "Legacy multiscale resume lacks cumulative health state"
                    )
                if saved_cumulative_health.get(
                    "gradient_conflict_controller"
                ) != gradient_conflict_state:
                    raise ValueError(
                        "Legacy multiscale resume cumulative/controller state is inconsistent"
                    )
                rank_cumulative_health = _restore_rank_cumulative_health(
                    saved_cumulative_health,
                    rank=rank,
                    checkpoint_update=int(checkpoint["update"]),
                    world_size=world_size,
                    scheduled_gradient_panel_updates=(
                        scheduled_gradient_panel_updates
                    ),
                    allow_schedule_extension=extension_enabled,
                )
                _restore_rng_state(rank_rng_records[rank], device)
            expected_resume_update = _update_for_completed_epoch(
                epoch=start_epoch,
                scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
                scenes_per_rank_per_update=scenes_per_rank_per_update,
            )
            if int(checkpoint["update"]) != expected_resume_update:
                raise ValueError("Resume update is not at an epoch boundary")
            if extension_enabled and start_epoch < extension_start_epoch:
                raise ValueError("Resume checkpoint predates the extension phase")
            if args.resume_into_fresh_attempt:
                expected_history_epochs = {
                    epoch for epoch in eval_epochs if epoch <= start_epoch
                }
                if set(inherited_evaluations) != expected_history_epochs:
                    raise ValueError(
                        "Fresh-attempt checkpoint evaluation history is incomplete"
                    )
                parent_checkpoint_paths = sorted(
                    args.resume.resolve(strict=True).parent.glob("epoch_*.pt")
                )
                observed_parent_epochs = {
                    int(path.stem.removeprefix("epoch_"))
                    for path in parent_checkpoint_paths
                }
                expected_checkpoint_epochs = (
                    expected_history_epochs
                    if sampling_mode == V2_SAMPLING_MODE
                    else expected_history_epochs - {0}
                )
                if observed_parent_epochs != expected_checkpoint_epochs:
                    raise ValueError(
                        "Fresh-attempt parent checkpoint set is incomplete or has later files"
                    )
                if args.resume.resolve(strict=True) != max(
                    parent_checkpoint_paths,
                    key=lambda path: int(path.stem.removeprefix("epoch_")),
                ):
                    raise ValueError(
                        "Fresh-attempt resume must use the latest parent checkpoint"
                    )
                lineage_parent_checkpoints = [
                    {
                        "epoch": int(path.stem.removeprefix("epoch_")),
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for path in parent_checkpoint_paths
                ]
                resolved["lineage_parent_checkpoints"] = (
                    lineage_parent_checkpoints
                )
        if hierarchy_enabled:
            model.hierarchy_mask_loss_weight = float(
                args.hierarchy_mask_loss_weight
            ) * (
                0.5
                if gradient_conflict_state["objectives"][
                    "hierarchy_mask_scaled"
                ]["coefficient_halved"]
                else 1.0
            )
            model.hierarchy_vicreg_loss_weight = float(
                args.hierarchy_vicreg_loss_weight
            ) * (
                0.5
                if gradient_conflict_state["objectives"][
                    "hierarchy_vicreg_scaled"
                ]["coefficient_halved"]
                else 1.0
            )
        elif args.multiscale_supervision:
            model.multiscale_loss_weight = float(args.multiscale_loss_weight) * (
                0.5
                if gradient_conflict_state["objectives"][
                    "legacy_auxiliary_scaled"
                ]["coefficient_halved"]
                else 1.0
            )
        dist.barrier()
        if rank == 0:
            # A failed restart must leave the previous provenance untouched.
            # All checkpoint/config validation and strict state loading above
            # therefore precede this atomic publication.
            if int(output_exists) == 0:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                (args.output_dir / "checkpoints").mkdir()
            if args.resume_into_fresh_attempt:
                for parent in lineage_parent_checkpoints:
                    source_checkpoint = Path(parent["path"])
                    destination_checkpoint = (
                        args.output_dir / "checkpoints" / source_checkpoint.name
                    )
                    with source_checkpoint.open("rb") as source, (
                        destination_checkpoint.open("xb")
                    ) as destination:
                        shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if sha256_file(destination_checkpoint) != parent["sha256"]:
                        raise RuntimeError(
                            "Copied parent checkpoint hash drifted"
                        )
                inherited_rows = [
                    {
                        "kind": "eval",
                        "epoch": epoch,
                        "update": _update_for_completed_epoch(
                            epoch=epoch,
                            scene_visits_per_epoch_per_rank=(
                                scene_visits_per_epoch_per_rank
                            ),
                            scenes_per_rank_per_update=(
                                scenes_per_rank_per_update
                            ),
                        ),
                        **inherited_evaluations[epoch],
                    }
                    for epoch in sorted(inherited_evaluations)
                ]
                _atomic_jsonl_noclobber(metrics_path, inherited_rows)
            elif int(output_exists) == 0:
                metrics_path.unlink(missing_ok=True)
            _atomic_json(args.output_dir / "resolved_config.json", resolved)
        dist.barrier()

        evaluations: dict[int, dict[str, Any]] = dict(inherited_evaluations)

        def evaluate_and_record(epoch: int, update: int) -> dict[str, Any]:
            evaluation: dict[str, Any] | None = None
            dist.barrier()
            if rank == 0:
                evaluation = evaluate_holdout(
                    model=model,
                    scenes=holdout_scenes,
                    device=device,
                    sampling=evaluation_sampling,
                    seed=args.fixed_eval_seed,
                    fixed_eval_pack=args.fixed_eval_pack,
                    fixed_eval_pack_report=fixed_eval_pack_report,
                    cells=source_cells,
                )
                if sampling_mode == V2_SAMPLING_MODE:
                    token_cosine_gap = _native_token_pure_fixed_pack_cosine_gap(
                        evaluation
                    )
                    qualifies = token_cosine_gap > float(
                        args.feature_hard_gap_threshold
                    )
                    feature_hard_state["qualifying_panel_streak"] = (
                        int(feature_hard_state["qualifying_panel_streak"]) + 1
                        if qualifies
                        else 0
                    )
                    feature_hard_state["panel_count"] = (
                        int(feature_hard_state["panel_count"]) + 1
                    )
                    feature_hard_state["last_panel_update"] = int(update)
                    feature_hard_state["last_panel_cosine_gap"] = token_cosine_gap
                    feature_hard_state[
                        "last_dense_point_diagnostic_cosine_gap"
                    ] = None
                    feature_hard_state[
                        "dense_point_holdout_eligible_for_unlock"
                    ] = False
                    feature_hard_state[
                        "native_token_metric_eligible_for_unlock"
                    ] = True
                    evaluation["feature_hard_gate"] = dict(feature_hard_state)
                evaluations[epoch] = evaluation
                _append_jsonl(
                    metrics_path,
                    {
                        "kind": "eval",
                        "epoch": epoch,
                        "update": update,
                        **evaluation,
                    },
                )
                native_macro = evaluation["native_selection_macro"]
                log.info(
                    "eval epoch=%d update=%d native_macro=%.4f 2d=%s 3d=%s "
                    "effrank=%.2f min_std=%.6f",
                    epoch,
                    update,
                    native_macro["overall"],
                    "n/a"
                    if "2d" not in native_macro
                    else f"{native_macro['2d']:.4f}",
                    "n/a"
                    if "3d" not in native_macro
                    else f"{native_macro['3d']:.4f}",
                    evaluation["native_health"]["effective_rank"],
                    evaluation["native_health"]["feature_std_min"],
                )
            payload: list[dict[str, Any] | None] = [evaluation]
            dist.broadcast_object_list(payload, src=0)
            evaluation = payload[0]
            if evaluation is None:
                raise RuntimeError("Evaluation broadcast failed")
            if sampling_mode == V2_SAMPLING_MODE:
                received_gate = evaluation.get("feature_hard_gate")
                if not isinstance(received_gate, dict):
                    raise RuntimeError("V2 evaluation lacks feature-hard gate state")
                feature_hard_state.clear()
                feature_hard_state.update(received_gate)
            evaluations[epoch] = evaluation
            dist.barrier()
            return evaluation

        def save_checkpoint(
            *,
            epoch_label: int,
            update_value: int,
            evaluation: dict[str, Any] | None,
            resumable: bool,
        ) -> Path:
            """Collect rank-local replay state and atomically retain one raw state."""

            gathered_counts: list[dict[str, int] | None] = [None] * world_size
            gathered_coverage: list[dict[str, Any] | None] = [None] * world_size
            gathered_sampling_totals: list[dict[str, Any] | None] = [
                None
            ] * world_size
            gathered_rng: list[dict[str, Any] | None] = [None] * world_size
            gathered_cumulative_health: list[dict[str, Any] | None] = [
                None
            ] * world_size
            dist.all_gather_object(gathered_counts, rank_exposure_counts)
            dist.all_gather_object(gathered_coverage, rank_coverage_states)
            dist.all_gather_object(
                gathered_sampling_totals, rank_sampling_totals
            )
            dist.all_gather_object(gathered_rng, _capture_rng_state(device))
            dist.all_gather_object(
                gathered_cumulative_health, rank_cumulative_health
            )
            if resumable:
                filename = f"epoch_{int(epoch_label):04d}.pt"
                checkpoint_kind = "resumable_epoch_boundary"
            else:
                filename = (
                    f"audit_e{int(epoch_label):04d}_u{int(update_value):08d}.pt"
                )
                checkpoint_kind = "nonresumable_update_aligned_audit_snapshot"
            path = args.output_dir / "checkpoints" / filename
            if rank == 0:
                typed_counts = [
                    {key: int(value) for key, value in row.items()}
                    for row in gathered_counts
                    if row is not None
                ]
                typed_coverage = [
                    dict(row) for row in gathered_coverage if row is not None
                ]
                typed_rng = [dict(row) for row in gathered_rng if row is not None]
                typed_sampling_totals = [
                    dict(row)
                    for row in gathered_sampling_totals
                    if row is not None
                ]
                typed_cumulative_health = [
                    dict(row)
                    for row in gathered_cumulative_health
                    if row is not None
                ]
                if (
                    len(typed_counts) != world_size
                    or len(typed_coverage) != world_size
                    or len(typed_rng) != world_size
                    or len(typed_sampling_totals) != world_size
                ):
                    raise RuntimeError("Incomplete rank state at checkpoint")
                cumulative_health = None
                if rank_cumulative_health is not None:
                    if len(typed_cumulative_health) != world_size:
                        raise RuntimeError(
                            "Incomplete cumulative health at checkpoint"
                        )
                    cumulative_health = _merge_rank_cumulative_health(
                        typed_cumulative_health,
                        through_update=int(update_value),
                        world_size=world_size,
                        gradient_conflict_controller=(
                            gradient_conflict_state
                        ),
                    )
                payload = _checkpoint_payload(
                    epoch=int(epoch_label),
                    nominal_epoch=int(epoch_label),
                    update=int(update_value),
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    resolved=resolved,
                    evaluation=evaluation,
                    evaluation_history=evaluations,
                    rank_exposure_counts=typed_counts,
                    rank_coverage_states=typed_coverage,
                    rank_sampling_totals=typed_sampling_totals,
                    rank_rng_states=typed_rng,
                    feature_hard_state=dict(feature_hard_state),
                    gradient_conflict_state=json.loads(
                        json.dumps(gradient_conflict_state)
                    ),
                    cumulative_health=cumulative_health,
                    checkpoint_kind=checkpoint_kind,
                )
                _atomic_torch_save(payload, path)
            dist.barrier()
            return path

        if extension_bootstrap and start_epoch in eval_epochs:
            inherited_update = _update_for_completed_epoch(
                epoch=start_epoch,
                scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
                scenes_per_rank_per_update=scenes_per_rank_per_update,
            )
            if rank == 0:
                evaluations[start_epoch] = inherited_evaluation
                _append_jsonl(
                    metrics_path,
                    {
                        "kind": "eval",
                        "epoch": start_epoch,
                        "update": inherited_update,
                        "inherited_from_parent": True,
                        **inherited_evaluation,
                    },
                )
                log.info(
                    "inherited eval epoch=%d update=%d native_macro=%.4f",
                    start_epoch,
                    inherited_update,
                    inherited_evaluation["native_selection_macro"]["overall"],
                )
            dist.barrier()
        elif start_epoch == 0 and 0 in eval_epochs:
            initial_evaluation = evaluate_and_record(0, 0)
            if 0 in checkpoint_epochs:
                save_checkpoint(
                    epoch_label=0,
                    update_value=0,
                    evaluation=initial_evaluation,
                    resumable=True,
                )

        model.train()
        started = time.perf_counter()
        update = _update_for_completed_epoch(
            epoch=start_epoch,
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
        )
        pending_visits: list[tuple[int, int]] = []
        stop_training = False
        training_scene_count = (
            len(local_scenes) if smoke else len(local_scene_ids)
        )
        for epoch in range(start_epoch, epochs):
            if stop_training:
                break
            order = np.random.default_rng(
                stable_seed(args.seed, rank, epoch, "scene-order")
            ).permutation(training_scene_count)
            for scene_position, scene_index in enumerate(order.tolist()):
                if update >= execution_stop_update:
                    stop_training = True
                    break
                pending_visits.append((epoch, int(scene_index)))
                at_epoch_boundary = scene_position + 1 == len(order)
                if len(pending_visits) < scenes_per_rank_per_update:
                    if at_epoch_boundary and epoch + 1 in eval_epochs:
                        raise RuntimeError(
                            "Evaluation boundary contains an incomplete logical batch"
                        )
                    continue
                if len(pending_visits) != scenes_per_rank_per_update:
                    raise AssertionError("Logical scene group cardinality drift")

                update += 1
                update_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                factor = (
                    extension_lr_factor(
                        update=update,
                        start_update=extension_start_update,
                        total_updates=total_updates,
                        warmup_updates=warmup_updates,
                    )
                    if extension_enabled
                    else lr_factor(
                        update=update,
                        total_updates=total_updates,
                        warmup_updates=warmup_updates,
                    )
                )
                lr_values = set_optimizer_lr(optimizer, factor=factor)
                optimizer.zero_grad(set_to_none=True)
                if (
                    sampling_mode == V2_SAMPLING_MODE
                    and not bool(feature_hard_state["enabled"])
                    and bool(
                        feature_hard_state.get(
                            "native_token_metric_eligible_for_unlock", False
                        )
                    )
                    and int(feature_hard_state["qualifying_panel_streak"])
                    >= int(args.feature_hard_required_panels)
                    and update
                    >= max(
                        1,
                        int(
                            math.ceil(
                                total_updates
                                * float(args.feature_hard_warmup_fraction)
                            )
                        ),
                    )
                ):
                    feature_hard_state["enabled"] = True
                    feature_hard_state["activation_update"] = int(update)
                scene_metric_sums = [0.0 for _ in range(6)]
                multiscale_metric_sums: dict[str, dict[str, float]] = {
                    name: {
                        "loss": 0.0,
                        "hardest_negative_ranking_accuracy": 0.0,
                        "cosine_gap": 0.0,
                        "temperature": 0.0,
                        "input_pairs": 0.0,
                        "input_proposals": 0.0,
                        "kept_pairs": 0.0,
                        "kept_proposals": 0.0,
                        "collapsed_positive_pairs": 0.0,
                        "dropped_proposals_insufficient_pure_positive": 0.0,
                        "dropped_proposals_no_pure_negative": 0.0,
                        "pure_positive_tokens": 0.0,
                        "pure_negative_tokens": 0.0,
                        "mixed_or_unknown_tokens_ignored": 0.0,
                        "accepted_mixed_tokens": 0.0,
                        "zero_negative_pairs": 0.0,
                        "base_negatives_per_pair": 0.0,
                        "uniform_negatives_per_pair": 0.0,
                        "spatial_hard_negatives_per_pair": 0.0,
                        "feature_candidate_pool": 0.0,
                        "feature_hard_negatives": 0.0,
                        "strict_token_purity": 0.0,
                        "stage_weight": 0.0,
                        "effective_loss_weight": 0.0,
                    }
                    for name in MULTISCALE_LEVEL_NAMES
                }
                final_loss_sum = 0.0
                auxiliary_loss_sum = 0.0
                multiscale_scale_sum = 0.0
                sampling_audit_records: list[dict[str, Any]] = []
                hierarchy_records: list[dict[str, Any]] = []
                token_pure_records: list[dict[str, Any]] = []
                legacy_multiscale_records: list[dict[str, Any]] = []
                gradient_telemetry_records: list[dict[str, Any]] = []
                shared_parameters = [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ]
                if not shared_parameters:
                    raise RuntimeError("LitePT shared-backbone parameter set is empty")
                shared_parameter_ids = {id(parameter) for parameter in shared_parameters}
                private_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                    and id(parameter) not in shared_parameter_ids
                ]
                projected_pcgrad_objectives = {
                    objective
                    for objective, state in gradient_conflict_state[
                        "objectives"
                    ].items()
                    if bool(state["pcgrad_enabled"])
                }
                pcgrad_active = bool(
                    (hierarchy_enabled or args.multiscale_supervision)
                    and projected_pcgrad_objectives
                )
                collect_global_gradient_panel = bool(
                    (hierarchy_enabled or args.multiscale_supervision)
                    and update in scheduled_gradient_panel_updates
                )
                manual_gradient_mode = bool(
                    collect_global_gradient_panel or pcgrad_active
                )
                shared_gradient_accumulator: dict[str, torch.Tensor] | None = None
                private_gradient_accumulator: dict[str, torch.Tensor] | None = None
                shared_gradient_report: dict[str, Any] | None = None
                globally_valid_microbatches = 0
                globally_invalid_microbatches = 0
                v2_gradients_pending_sync = False
                group_epochs = [visit_epoch for visit_epoch, _ in pending_visits]
                data_wait_started = time.perf_counter()
                if source_loading_mode == "logical_batch":
                    if source_loader_pool is None:
                        raise RuntimeError("Logical source-loader pool is missing")
                    group_scenes = _load_logical_scene_group(
                        scene_indices=[index for _, index in pending_visits],
                        scene_ids=local_scene_ids,
                        source_root=args.source_root,
                        expected_manifest_hashes=expected_manifest_hashes,
                        input_features=args.input_features,
                        coordinate_normalization_policy=(
                            args.coordinate_normalization_policy
                        ),
                        color_mean=color_mean,
                        color_std=color_std,
                        executor=source_loader_pool,
                    )
                else:
                    group_scenes = [
                        local_scenes[index] for _, index in pending_visits
                    ]
                data_wait_seconds = time.perf_counter() - data_wait_started
                for microbatch_index, (
                    (visit_epoch, visit_scene_index),
                    scene,
                ) in enumerate(zip(pending_visits, group_scenes, strict=True)):
                    if (
                        source_loading_mode == "logical_batch"
                        and scene.scene_id != local_scene_ids[visit_scene_index]
                    ):
                        raise RuntimeError("Lazy scene-group ordering drift")
                    scene = scene_source_for_visit(
                        scene,
                        epoch=visit_epoch,
                        global_seed=args.seed,
                    )
                    visit_input = build_scene_visit_input(
                        scene=scene,
                        epoch=visit_epoch,
                        global_seed=args.seed,
                        input_features=args.input_features,
                        augmentation_profile=train_augmentation_profile,
                        color_mean=color_mean,
                        color_std=color_std,
                    )
                    points = torch.from_numpy(visit_input.points).to(device)
                    features = torch.from_numpy(visit_input.features).to(device)
                    frame_groups: Sequence[Any] | None = None
                    plan = None
                    if sampling_mode == V2_SAMPLING_MODE:
                        # Resumable checkpoints retain one cursor per
                        # scene/cell.  The saved record is stamped with the
                        # completed preceding epoch; explicitly rebind it to
                        # this visit before handing it to the sampler.  A
                        # missing record is the deterministic epoch-derived
                        # initial cursor for a scene that has not been seen
                        # yet.  Never pass another scene's cursor into this
                        # call: the sampler identity includes scene and
                        # catalog fingerprint.
                        coverage_states_for_scene: dict[
                            str, DeterministicCoverageState | None
                        ] = {}
                        for sampled_cell in source_cells:
                            saved_state = rank_coverage_states.get(
                                f"{scene.scene_id}|{sampled_cell}"
                            )
                            if saved_state is None:
                                coverage_states_for_scene[sampled_cell] = None
                            elif isinstance(saved_state, Mapping):
                                coverage_states_for_scene[sampled_cell] = (
                                    DeterministicCoverageState.from_dict(
                                        saved_state
                                    ).for_epoch(visit_epoch)
                                )
                            else:
                                raise ValueError(
                                    "V2 coverage state must be a mapping or null"
                                )
                        try:
                            plan = sample_multigranular_frame_group_plan(
                                scene_id=scene.scene_id,
                                epoch=visit_epoch,
                                seed=args.seed,
                                points=points,
                                frames_by_cell=_v2_frame_catalogs(
                                    scene=scene,
                                    device=device,
                                    cells=tuple(source_cells),
                                ),
                                total_proposal_quota=int(
                                    sampling.proposals_per_forward
                                ),
                                cells=tuple(source_cells),
                                positive_pairs_per_proposal=int(
                                    sampling.positive_pairs_per_proposal
                                ),
                                num_uniform_negatives=int(
                                    sampling.uniform_negatives
                                ),
                                num_spatial_hard_negatives=int(
                                    sampling.spatial_hard_negatives
                                ),
                                spatial_candidate_pool=int(
                                    sampling.spatial_candidate_pool
                                ),
                                num_feature_hard_negatives=(
                                    int(sampling.feature_hard_negatives)
                                    if bool(feature_hard_state["enabled"])
                                    else 0
                                ),
                                feature_candidate_pool=int(
                                    sampling.feature_candidate_pool
                                ),
                                require_negative_proposal_membership=True,
                                operation="rgbn6-pretraining-v2",
                                coverage_states=coverage_states_for_scene,
                            )
                        except NoRetainedFrameGroupsError as error:
                            # Expected data-quality exhaustion is represented by
                            # the complete plan and synchronized before any rank
                            # can enter SyncBN/DDP/VICReg collectives.
                            plan = error.plan
                        frame_groups = plan.groups
                        batch = None
                        sampling_audit_records.append(
                            {
                                **plan.audit_metadata(),
                                "source_crop_id": scene.crop_id,
                                "source_variant_id": scene.source_variant_id,
                                "groups": [
                                    {
                                        "cell": group.cell,
                                        "frame_id": group.frame_id,
                                        "selected_proposal_count": len(
                                            group.selected_proposal_indices
                                        ),
                                        "sampler": dict(
                                            group.coverage_metadata["sampler"]
                                        ),
                                    }
                                    for group in plan.groups
                                ],
                            }
                        )
                        for sampled_cell, coverage in plan.coverage_by_cell.items():
                            rank_exposure_counts[sampled_cell] += int(
                                coverage["realized_count"]
                            )
                            cumulative = rank_sampling_totals[sampled_cell]
                            cumulative["requested_proposals"] += int(
                                coverage["requested_count"]
                            )
                            cumulative["realized_unique_proposals"] += int(
                                coverage["realized_count"]
                            )
                            cumulative[
                                "contrastive_retained_proposals"
                            ] += int(coverage["contrastive_retained_count"])
                            cumulative["duplicate_proposals"] += int(
                                coverage["duplicate_count"]
                            )
                            state_after = coverage.get("state_after")
                            rank_coverage_states[
                                f"{scene.scene_id}|{sampled_cell}"
                            ] = state_after
                        for group in plan.groups:
                            sampler_report = group.coverage_metadata["sampler"]
                            cumulative = rank_sampling_totals[group.cell]
                            cumulative["realized_positive_pairs"] += int(
                                sampler_report["realized_positive_pairs"]
                            )
                            cumulative["requested_positive_pairs"] += int(
                                sampler_report[
                                    "requested_positive_pairs_per_proposal"
                                ]
                            ) * len(sampler_report["retained_proposal_indices"])
                        local_sampler_valid = (
                            bool(plan.groups)
                            and {group.cell for group in plan.groups}
                            == set(source_cells)
                        )
                        sampler_valid = torch.tensor(
                            int(local_sampler_valid),
                            dtype=torch.int32,
                            device=device,
                        )
                        dist.all_reduce(sampler_valid, op=dist.ReduceOp.MIN)
                        if int(sampler_valid.item()) != 1:
                            globally_invalid_microbatches += 1
                            for sampled_cell in source_cells:
                                rank_sampling_totals[sampled_cell][
                                    "invalid_scene_visits"
                                ] += 1
                            sampling_audit_records[-1][
                                "optimization_valid"
                            ] = False
                            sampling_audit_records[-1][
                                "optimization_drop_reason"
                            ] = "missing_required_frame_local_cell_on_at_least_one_rank"
                            del visit_input, points, features, batch, plan
                            continue
                        sampling_audit_records[-1]["optimization_valid"] = True
                    else:
                        cell, batch = training_batch(
                            scene=scene,
                            epoch=visit_epoch,
                            points=points,
                            device=device,
                            sampling=sampling,
                            sampling_mode=sampling_mode,
                            repeat_g08_for_all_3d_slots=(
                                args.three_d_mode == "g08_repeated"
                            ),
                            cells=source_cells,
                            cell_schedule=args.source_cell_schedule,
                            schedule_seed=args.seed,
                        )
                    forward_seed = stable_seed(
                        args.seed,
                        rank,
                        scene.scene_id,
                        visit_epoch,
                        "train-forward",
                    )
                    random.seed(forward_seed)
                    np.random.seed(forward_seed % (2**32))
                    torch.manual_seed(forward_seed)
                    is_final_microbatch = (
                        microbatch_index + 1 == scenes_per_rank_per_update
                    )
                    # V2 presents six sequential scene forwards as one logical
                    # update.  Keep the first five backward passes local so
                    # the final DDP backward reduces their accumulated buckets
                    # together with the sixth scene.  Manual telemetry/PCGrad
                    # updates intentionally use the raw model and their own
                    # exact global-gradient reductions.
                    sync_context = (
                        nullcontext()
                        if manual_gradient_mode or is_final_microbatch
                        else ddp_model.no_sync()
                    )
                    deferred_v2_sync = bool(
                        sampling_mode == V2_SAMPLING_MODE
                        and not manual_gradient_mode
                        and not is_final_microbatch
                    )
                    with sync_context:
                        if (
                            manual_gradient_mode
                            and args.normalization_policy == "batchnorm"
                        ):
                            for buffer in model.buffers():
                                dist.broadcast(buffer, src=0)
                        forward_model = model if manual_gradient_mode else ddp_model
                        result = forward_model(
                            points,
                            features,
                            batch,
                            multiscale_loss_scale=(
                                model.multiscale_scale_for_epoch(visit_epoch)
                            ),
                            frame_groups=frame_groups,
                            supervision_step=update,
                            supervision_total_steps=total_updates,
                            return_loss_components=(
                                collect_global_gradient_panel or pcgrad_active
                            ),
                            return_multiscale_proposal_diagnostics=bool(
                                args.multiscale_supervision
                            ),
                        )
                        loss = result["loss_total"]
                        finite = torch.tensor(
                            int(bool(torch.isfinite(loss))),
                            dtype=torch.int32,
                            device=device,
                        )
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if int(finite.item()) != 1:
                            raise FloatingPointError(
                                "Non-finite loss at "
                                f"epoch={visit_epoch} update={update} "
                                f"microbatch={microbatch_index}"
                            )
                        if "hierarchy_supervision" in result:
                            hierarchy_records.append(result["hierarchy_supervision"])
                        if "token_pure_dec0" in result:
                            token_pure_records.append(result["token_pure_dec0"])
                        if sampling_mode == V2_SAMPLING_MODE:
                            token_health = result.get("token_pure_dec0")
                            local_token_valid = bool(
                                result.get("optimization_valid", False)
                                and
                                isinstance(token_health, Mapping)
                                and token_health.get("optimization_valid", False)
                                and set(token_health.get("retained_cells", ()))
                                == set(source_cells)
                            )
                            hierarchy_health = result.get("hierarchy_supervision")
                            if hierarchy_enabled:
                                local_token_valid = bool(
                                    local_token_valid
                                    and isinstance(hierarchy_health, Mapping)
                                    and hierarchy_health.get(
                                        "optimization_valid", False
                                    )
                                )
                            token_valid = torch.tensor(
                                int(local_token_valid),
                                dtype=torch.int32,
                                device=device,
                            )
                            dist.all_reduce(token_valid, op=dist.ReduceOp.MIN)
                            optimization_valid = int(token_valid.item()) == 1
                        else:
                            optimization_valid = True
                        if not optimization_valid:
                            globally_invalid_microbatches += 1
                            if sampling_audit_records:
                                sampling_audit_records[-1][
                                    "optimization_valid"
                                ] = False
                                sampling_audit_records[-1][
                                    "optimization_drop_reason"
                                ] = (
                                    "post_voxel_token_or_hierarchy_starvation_"
                                    "on_at_least_one_rank"
                                )
                            for sampled_cell in source_cells:
                                rank_sampling_totals[sampled_cell][
                                    "invalid_scene_visits"
                                ] += 1
                            # Every rank drains the identical DDP graph.  The
                            # model's V2 zero-yield contract touches disposable
                            # heads, so find_unused_parameters=False remains safe.
                            if not manual_gradient_mode:
                                (loss * 0.0).backward()
                                if sampling_mode == V2_SAMPLING_MODE:
                                    v2_gradients_pending_sync = deferred_v2_sync
                            del visit_input, points, features, batch, result, loss
                            if plan is not None:
                                del plan
                            continue

                        globally_valid_microbatches += 1
                        if args.multiscale_supervision:
                            multiscale_levels = result.get("multiscale")
                            if not isinstance(multiscale_levels, Mapping) or set(
                                multiscale_levels
                            ) != set(LEGACY_MULTISCALE_CUMULATIVE_STAGES):
                                raise RuntimeError(
                                    "Legacy multiscale result stage set drift"
                                )
                            legacy_levels: dict[str, dict[str, Any]] = {}
                            for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES:
                                stage_metrics = multiscale_levels[stage]
                                if not isinstance(stage_metrics, Mapping) or not set(
                                    LEGACY_MULTISCALE_CUMULATIVE_KEYS
                                ).issubset(stage_metrics):
                                    raise RuntimeError(
                                        "Legacy multiscale result metric set drift"
                                    )
                                # Keep only the canonical cumulative metrics;
                                # proposal-level diagnostics are intentionally
                                # not copied into every update record.
                                legacy_levels[stage] = {
                                    key: stage_metrics[key]
                                    for key in LEGACY_MULTISCALE_CUMULATIVE_KEYS
                                }
                            legacy_multiscale_records.append(
                                {
                                    "levels": _json_copy(legacy_levels)
                                }
                            )
                        if collect_global_gradient_panel:
                            loss_components = result.get(
                                "loss_component_tensors"
                            )
                            junctions = result.get("hierarchy_junction_tensors")
                            if not isinstance(loss_components, dict) or not isinstance(
                                junctions, dict
                            ):
                                raise RuntimeError(
                                    "Supervised V2 result lacks gradient telemetry tensors"
                                )
                            if hierarchy_enabled:
                                selected_components = {
                                    key: loss_components[key]
                                    for key in (
                                        "final_dec0",
                                        "hierarchy_mask_scaled",
                                        "hierarchy_vicreg_scaled",
                                    )
                                }
                            else:
                                selected_components = {
                                    key: loss_components[key]
                                    for key in (
                                        "final_dec0",
                                        "legacy_auxiliary_scaled",
                                    )
                                }
                            gradient_telemetry_records.append(
                                hierarchy_gradient_telemetry(
                                    loss_components=selected_components,
                                    junctions=junctions,
                                )
                            )
                        if manual_gradient_mode:
                            manual_components = result.get("loss_component_tensors")
                            if not isinstance(manual_components, dict):
                                raise RuntimeError(
                                    "Shared-gradient mode lacks objective components"
                                )
                            component_names = (
                                (
                                    "final_dec0",
                                    "hierarchy_mask_scaled",
                                    "hierarchy_vicreg_scaled",
                                )
                                if hierarchy_enabled
                                else (
                                    "final_dec0",
                                    "legacy_auxiliary_scaled",
                                )
                            )
                            selected_manual_components = {
                                key: manual_components[key]
                                for key in component_names
                            }
                            shared_gradient_accumulator = (
                                _add_flat_objective_gradients(
                                    shared_gradient_accumulator,
                                    _flat_objective_gradients(
                                    components=selected_manual_components,
                                    parameters=shared_parameters,
                                    scale=float(scenes_per_rank_per_update),
                                ),
                            )
                            )
                            if private_parameters:
                                private_gradient_accumulator = (
                                    _add_flat_objective_gradients(
                                        private_gradient_accumulator,
                                        _flat_objective_gradients(
                                            components={"total": loss},
                                            parameters=private_parameters,
                                            scale=float(
                                                scenes_per_rank_per_update
                                            ),
                                        ),
                                    )
                                )
                        else:
                            (loss / scenes_per_rank_per_update).backward()
                            if sampling_mode == V2_SAMPLING_MODE:
                                v2_gradients_pending_sync = deferred_v2_sync
                    for metric_index, value in enumerate(
                        (
                            float(loss.detach()),
                            float(result["triplet_ranking_accuracy"]),
                            float(result["hardest_negative_ranking_accuracy"]),
                            float(result["positive_cosine_mean"]),
                            float(result["negative_cosine_mean"]),
                            float(result["cosine_gap"]),
                        )
                    ):
                        scene_metric_sums[metric_index] += value
                    final_loss_sum += float(result.get("loss_final_dec0", 0.0))
                    auxiliary_loss_sum += float(result.get("loss_multiscale", 0.0))
                    multiscale_scale_sum += float(
                        result.get("multiscale_loss_scale", 0.0)
                    )
                    if sampling_mode == V2_SAMPLING_MODE:
                        token_health = result["token_pure_dec0"]
                        per_cell_pairs: dict[str, int] = {
                            sampled_cell: 0 for sampled_cell in source_cells
                        }
                        per_cell_requested: dict[str, int] = {
                            sampled_cell: 0 for sampled_cell in source_cells
                        }
                        for group_record in token_health.get("groups", ()):
                            sampled_cell = str(group_record["cell"])
                            per_cell_pairs[sampled_cell] += int(
                                group_record.get("realized_positive_pairs", 0)
                            )
                            proposal_records = group_record.get("proposals", ())
                            requested_token_pairs = sum(
                                int(proposal.get("requested_token_pairs", 0))
                                for proposal in proposal_records
                            )
                            per_cell_requested[sampled_cell] += (
                                requested_token_pairs
                                if proposal_records
                                else int(group_record.get("input_point_pairs", 0))
                            )
                        mass = int(round(1_000_000 / len(source_cells)))
                        for sampled_cell in source_cells:
                            cumulative = rank_sampling_totals[sampled_cell]
                            cumulative["active_scene_visits"] += 1
                            cumulative["optimized_scene_visits"] += 1
                            cumulative["loss_mass_millionths"] += mass
                            cumulative[
                                "optimized_loss_mass_millionths"
                            ] += mass
                            cumulative[
                                "token_realized_positive_pairs"
                            ] += per_cell_pairs[sampled_cell]
                            cumulative[
                                "token_requested_positive_pairs"
                            ] += per_cell_requested[sampled_cell]
                    for name in MULTISCALE_LEVEL_NAMES:
                        level_metrics = result.get("multiscale", {}).get(name, {})
                        for key in multiscale_metric_sums[name]:
                            multiscale_metric_sums[name][key] += float(
                                level_metrics.get(key, 0.0)
                            )
                    if sampling_mode != V2_SAMPLING_MODE:
                        rank_exposure_counts[cell] += sampling.proposals_per_forward
                    del visit_input, points, features, batch, result, loss
                    if plan is not None:
                        del plan

                if source_loading_mode == "logical_batch":
                    del scene, group_scenes

                if (
                    sampling_mode == V2_SAMPLING_MODE
                    and not manual_gradient_mode
                    and (
                        v2_gradients_pending_sync
                        or globally_valid_microbatches <= 0
                    )
                ):
                    _synchronize_accumulated_gradients(
                        parameters=model.parameters(),
                        world_size=world_size,
                    )
                if globally_valid_microbatches <= 0:
                    raise RuntimeError(
                        "V2 logical update retained zero globally valid microbatches"
                    )
                if sampling_mode == V2_SAMPLING_MODE:
                    valid_rescale = (
                        float(scenes_per_rank_per_update)
                        / float(globally_valid_microbatches)
                    )
                    if not manual_gradient_mode:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(valid_rescale)
                    if shared_gradient_accumulator is not None:
                        for vector in shared_gradient_accumulator.values():
                            vector.mul_(valid_rescale)
                    if private_gradient_accumulator is not None:
                        for vector in private_gradient_accumulator.values():
                            vector.mul_(valid_rescale)
                if manual_gradient_mode:
                    if shared_gradient_accumulator is None:
                        raise RuntimeError(
                            "Shared-gradient logical batch accumulated no gradients"
                        )
                    shared_gradient_report = _global_shared_gradient_report(
                        accumulated=shared_gradient_accumulator,
                        shared_parameters=shared_parameters,
                        projected_objectives=projected_pcgrad_objectives,
                        world_size=world_size,
                        install_projected_gradient=True,
                    )
                    if private_parameters:
                        if (
                            private_gradient_accumulator is None
                            or set(private_gradient_accumulator) != {"total"}
                        ):
                            raise RuntimeError(
                                "Private logical-batch gradient is incomplete"
                            )
                        _install_global_flat_total_gradient(
                            accumulated=private_gradient_accumulator["total"],
                            parameters=private_parameters,
                            world_size=world_size,
                        )

                gradient = _clip_gradient_diagnostics(
                    model.parameters(),
                    max_norm=float(args.grad_clip_norm),
                )
                gradient_status = torch.tensor(
                    [
                        int(
                            all(
                                math.isfinite(float(gradient[key]))
                                for key in (
                                    "pre_clip_norm",
                                    "post_clip_norm",
                                    "clip_return_norm",
                                    "clip_coefficient",
                                )
                            )
                        ),
                        int(bool(gradient["was_clipped"])),
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                # One SUM replaces the former finite-only MIN collective.  It
                # also supplies the exact any-rank/rank-event clipping counts
                # without adding a synchronization to the hot path.
                dist.all_reduce(gradient_status, op=dist.ReduceOp.SUM)
                if int(gradient_status[0].item()) != world_size:
                    raise FloatingPointError(
                        f"Non-finite gradient at epoch={epoch} update={update}"
                    )
                clipped_rank_count = int(gradient_status[1].item())
                optimizer.step()
                if rank_cumulative_health is not None:
                    _accumulate_rank_cumulative_health_update(
                        rank_cumulative_health,
                        update=update,
                        local_was_clipped=bool(gradient["was_clipped"]),
                        clipped_rank_count=clipped_rank_count,
                        world_size=world_size,
                        attempted_microbatch_count=(
                            scenes_per_rank_per_update
                        ),
                        optimization_valid_count=(
                            globally_valid_microbatches
                        ),
                        token_records=token_pure_records,
                        hierarchy_records=hierarchy_records,
                        legacy_multiscale_records=legacy_multiscale_records,
                    )
                update_seconds = time.perf_counter() - update_started

                if gradient_telemetry_records:
                    gathered_gradient_telemetry: list[
                        list[dict[str, Any]] | None
                    ] = [None for _ in range(world_size)]
                    dist.all_gather_object(
                        gathered_gradient_telemetry,
                        gradient_telemetry_records,
                    )
                    if rank == 0:
                        flattened_gradient_telemetry = [
                            record
                            for rank_records in gathered_gradient_telemetry
                            if rank_records is not None
                            for record in rank_records
                        ]
                        conflict_actions: list[dict[str, Any]] = []
                        if shared_gradient_report is None:
                            raise RuntimeError(
                                "Gradient panel lacks global shared-backbone report"
                            )
                        if hierarchy_enabled or args.multiscale_supervision:
                            for objective, state in gradient_conflict_state[
                                "objectives"
                            ].items():
                                objective_report = shared_gradient_report[
                                    "objectives"
                                ].get(objective)
                                if not isinstance(objective_report, Mapping):
                                    raise RuntimeError(
                                        f"Missing global shared gradient for {objective}"
                                    )
                                cosine_key = (
                                    "global_shared_cosine_after"
                                    if bool(state["pcgrad_enabled"])
                                    else "global_shared_cosine_before"
                                )
                                cosine = float(objective_report[cosine_key])
                                state["last_cosine"] = cosine
                                state["last_cosine_kind"] = cosine_key
                                state["panel_streak"] = (
                                    int(state["panel_streak"]) + 1
                                    if cosine
                                    < float(gradient_conflict_state["threshold"])
                                    else 0
                                )
                                if int(state["panel_streak"]) < int(
                                    gradient_conflict_state[
                                        "required_consecutive_panels"
                                    ]
                                ):
                                    continue
                                if not bool(state["coefficient_halved"]):
                                    state["coefficient_halved"] = True
                                    action = "halve_coefficient"
                                elif not bool(state["pcgrad_enabled"]):
                                    state["pcgrad_enabled"] = True
                                    action = "enable_pcgrad"
                                else:
                                    state["post_pcgrad_failure"] = True
                                    gradient_conflict_state["rejected"] = True
                                    gradient_conflict_state[
                                        "rejection_reason"
                                    ] = (
                                        "persistent_global_shared_conflict_"
                                        f"after_pcgrad:{objective}"
                                    )
                                    action = "reject_candidate"
                                state["panel_streak"] = 0
                                state["last_action_update"] = int(update)
                                conflict_actions.append(
                                    {
                                        "objective": objective,
                                        "cosine": cosine,
                                        "action": action,
                                    }
                                )
                            budget_report = shared_gradient_report[
                                "auxiliary_budget"
                            ]
                            budget_ratio = float(
                                budget_report[
                                    "global_sum_aux_norm_ratio_before"
                                ]
                            )
                            budget_state = gradient_conflict_state[
                                "auxiliary_budget"
                            ]
                            budget_state[
                                "last_global_sum_aux_norm_ratio"
                            ] = budget_ratio
                            budget_state["passed"] = bool(
                                budget_report["passed_before"]
                            )
                            if not bool(budget_report["passed_before"]):
                                budget_state["violation_count"] = (
                                    int(budget_state["violation_count"]) + 1
                                )
                                gradient_conflict_state["rejected"] = True
                                gradient_conflict_state[
                                    "rejection_reason"
                                ] = "global_shared_auxiliary_gradient_budget_exceeded"
                                conflict_actions.append(
                                    {
                                        "objective": "combined_auxiliary",
                                        "norm_ratio": budget_ratio,
                                        "limit": float(budget_report["limit"]),
                                        "action": "reject_candidate",
                                    }
                                )
                        gradient_health_row = {
                            "kind": "gradient_health",
                            "epoch": group_epochs[-1],
                            "update": update,
                            "diagnostic_local_junction_telemetry": _mean_numeric_tree(
                                flattened_gradient_telemetry
                            ),
                            "objective_stage_counts": (
                                _gradient_objective_stage_counts(
                                    flattened_gradient_telemetry
                                )
                            ),
                            "diagnostic_conservative_auxiliary_budget": _telemetry_auxiliary_budget(
                                flattened_gradient_telemetry
                            ),
                            "global_shared_gradient": shared_gradient_report,
                            "auxiliary_budget": shared_gradient_report[
                                "auxiliary_budget"
                            ],
                            "auxiliary_budget_limit_vs_primary": 0.5,
                            "conflict_threshold": -0.05,
                            "controller": gradient_conflict_state,
                            "actions": conflict_actions,
                            "pcgrad": (
                                shared_gradient_report
                                if pcgrad_active
                                else None
                            ),
                        }
                        _append_jsonl(metrics_path, gradient_health_row)
                        if rank_cumulative_health is not None:
                            _append_rank_gradient_panel(
                                rank_cumulative_health,
                                gradient_health_row,
                            )
                    conflict_payload: list[dict[str, Any] | None] = [
                        gradient_conflict_state if rank == 0 else None
                    ]
                    dist.broadcast_object_list(conflict_payload, src=0)
                    received_conflict_state = conflict_payload[0]
                    if not isinstance(received_conflict_state, dict):
                        raise RuntimeError("Gradient conflict-state broadcast failed")
                    received_conflict_state = json.loads(
                        json.dumps(received_conflict_state)
                    )
                    gradient_conflict_state.clear()
                    gradient_conflict_state.update(received_conflict_state)
                    if hierarchy_enabled:
                        model.hierarchy_mask_loss_weight = float(
                            args.hierarchy_mask_loss_weight
                        ) * (
                            0.5
                            if gradient_conflict_state["objectives"][
                                "hierarchy_mask_scaled"
                            ]["coefficient_halved"]
                            else 1.0
                        )
                        model.hierarchy_vicreg_loss_weight = float(
                            args.hierarchy_vicreg_loss_weight
                        ) * (
                            0.5
                            if gradient_conflict_state["objectives"][
                                "hierarchy_vicreg_scaled"
                            ]["coefficient_halved"]
                            else 1.0
                        )
                    elif args.multiscale_supervision:
                        model.multiscale_loss_weight = float(
                            args.multiscale_loss_weight
                        ) * (
                            0.5
                            if gradient_conflict_state["objectives"][
                                "legacy_auxiliary_scaled"
                            ]["coefficient_halved"]
                            else 1.0
                        )

                if (
                    update == 1
                    or update % int(args.log_every) == 0
                    or update in snapshot_label_by_update
                    or (at_epoch_boundary and epoch + 1 in eval_epochs)
                ):
                    metric_denominator = max(globally_valid_microbatches, 1)
                    scene_metric_means = [
                        value / metric_denominator
                        for value in scene_metric_sums
                    ]
                    values = torch.tensor(
                        [
                            scene_metric_means[0],
                            scene_metric_means[1],
                            scene_metric_means[2],
                            float(gradient["pre_clip_norm"]),
                            float(gradient["post_clip_norm"]),
                            float(gradient["clip_return_norm"]),
                            float(gradient["clip_coefficient"]),
                            float(gradient["was_clipped"]),
                            scene_metric_means[3],
                            scene_metric_means[4],
                            scene_metric_means[5],
                            final_loss_sum / metric_denominator,
                            auxiliary_loss_sum / metric_denominator,
                            multiscale_scale_sum / metric_denominator,
                            *[
                                multiscale_metric_sums[name][key]
                                / metric_denominator
                                for name in MULTISCALE_LEVEL_NAMES
                                for key in multiscale_metric_sums[name]
                            ],
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    values /= world_size
                    gathered_sampling: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    gathered_hierarchy: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    gathered_token_pure: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(
                        gathered_sampling, sampling_audit_records
                    )
                    dist.all_gather_object(gathered_hierarchy, hierarchy_records)
                    dist.all_gather_object(gathered_token_pure, token_pure_records)
                    local_resource = {
                        "rank": rank,
                        "device": local_device_report,
                        "update_seconds": update_seconds,
                        "data_wait_seconds": data_wait_seconds,
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    }
                    gathered_resources: list[dict[str, Any] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(gathered_resources, local_resource)
                    if rank == 0:
                        row = {
                            "kind": "train",
                            "epoch": group_epochs[-1],
                            "scene_epoch_start": group_epochs[0],
                            "scene_epoch_end": group_epochs[-1],
                            "update": update,
                            "loss_total": float(values[0]),
                            "triplet_ranking_accuracy": float(values[1]),
                            "hardest_negative_ranking_accuracy": float(values[2]),
                            "gradient_norm": float(values[3]),
                            "gradient_norm_pre_clip": float(values[3]),
                            "gradient_norm_post_clip": float(values[4]),
                            "clip_return_norm": float(values[5]),
                            "clip_coefficient": float(values[6]),
                            "gradient_clipped_rank_fraction": float(values[7]),
                            "positive_cosine_mean": float(values[8]),
                            "negative_cosine_mean": float(values[9]),
                            "cosine_gap": float(values[10]),
                            "lr": lr_values,
                            "temperature": float(model.criterion.temperature.detach()),
                            "sampling_profile": args.sampling_profile,
                            "train_augmentation_profile": train_augmentation_profile,
                            "required_crop_mode": required_crop_mode,
                            "scenes_per_rank_per_update": scenes_per_rank_per_update,
                            "scenes_per_global_update": (
                                scenes_per_rank_per_update * world_size
                            ),
                            "globally_valid_scenes_per_rank": int(
                                globally_valid_microbatches
                            ),
                            "globally_invalid_scenes_per_rank": int(
                                globally_invalid_microbatches
                            ),
                            "proposals_per_scene": int(
                                sampling.proposals_per_forward
                            ),
                            "proposals_per_global_update": (
                                scenes_per_rank_per_update
                                * world_size
                                * int(sampling.proposals_per_forward)
                            ),
                            "negatives_per_pair": int(sampling.uniform_negatives)
                            + int(sampling.spatial_hard_negatives)
                            + int(sampling.feature_hard_negatives),
                            "wall_seconds": time.perf_counter() - started,
                            "normalization_policy": str(
                                args.normalization_policy
                            ),
                            "feature_hard_gate": dict(feature_hard_state),
                            "resource_health": {
                                "ranks": [
                                    resource
                                    for resource in gathered_resources
                                    if resource is not None
                                ],
                                "rank_step_time_imbalance_fraction": (
                                    max(
                                        float(resource["update_seconds"])
                                        for resource in gathered_resources
                                        if resource is not None
                                    )
                                    / max(
                                        min(
                                            float(resource["update_seconds"])
                                            for resource in gathered_resources
                                            if resource is not None
                                        ),
                                        1.0e-12,
                                    )
                                    - 1.0
                                ),
                                "maximum_reserved_memory_fraction": max(
                                    float(resource["peak_reserved_bytes"])
                                    / float(resource["device"]["total_memory_bytes"])
                                    for resource in gathered_resources
                                    if resource is not None
                                ),
                            },
                        }
                        if sampling_mode == V2_SAMPLING_MODE:
                            flattened_sampling = [
                                record
                                for rank_records in gathered_sampling
                                if rank_records is not None
                                for record in rank_records
                            ]
                            row["sampling_health"] = _summarize_v2_sampling(
                                flattened_sampling,
                                cells=source_cells,
                            )
                            row["sampling_health"]["feature_hard_enabled"] = bool(
                                feature_hard_state["enabled"]
                            )
                            flattened_token_pure = [
                                record
                                for rank_records in gathered_token_pure
                                if rank_records is not None
                                for record in rank_records
                            ]
                            row["token_pure_dec0"] = (
                                _summarize_token_pure_records(
                                    flattened_token_pure,
                                    cells=source_cells,
                                )
                            )
                        if hierarchy_enabled:
                            flattened_hierarchy = [
                                record
                                for rank_records in gathered_hierarchy
                                if rank_records is not None
                                for record in rank_records
                            ]
                            row["hierarchy_supervision"] = (
                                _summarize_hierarchy_records(
                                    flattened_hierarchy,
                                    cells=source_cells,
                                )
                            )
                        row["multiscale"] = {
                            "loss_final_dec0": float(values[11]),
                            "loss_auxiliary": float(values[12]),
                            "loss_scale": float(values[13]),
                            "levels": {},
                        }
                        value_index = 14
                        for name in MULTISCALE_LEVEL_NAMES:
                            row["multiscale"]["levels"][name] = {}
                            for key in multiscale_metric_sums[name]:
                                row["multiscale"]["levels"][name][key] = float(
                                    values[value_index]
                                )
                                value_index += 1
                        _append_jsonl(metrics_path, row)
                        log.info(
                            "epoch=%d update=%d loss=%.4f hard=%.4f lr_bb=%.3e",
                            group_epochs[-1],
                            update,
                            row["loss_total"],
                            row["hardest_negative_ranking_accuracy"],
                            lr_values["backbone/decay"],
                        )
                pending_visits = []

                if bool(gradient_conflict_state["rejected"]):
                    if rank == 0:
                        _atomic_json(
                            args.output_dir / "candidate_rejected.json",
                            {
                                "schema_version": (
                                    "litept_pretraining_candidate_rejection/v2"
                                ),
                                "reason": gradient_conflict_state.get(
                                    "rejection_reason"
                                )
                                or "persistent_post_pcgrad_conflict",
                                "epoch": int(group_epochs[-1]),
                                "update": int(update),
                                "controller": gradient_conflict_state,
                                "checkpoint_published_for_rejected_update": False,
                                "passed": False,
                            },
                        )
                    dist.barrier()
                    raise RuntimeError(
                        "Candidate rejected before evaluation/checkpoint publication: "
                        f"{gradient_conflict_state.get('rejection_reason')}"
                    )

                if update in snapshot_label_by_update:
                    snapshot_label = int(snapshot_label_by_update[update])
                    snapshot_evaluation = evaluate_and_record(
                        snapshot_label, update
                    )
                    if update in checkpoint_updates:
                        save_checkpoint(
                            epoch_label=snapshot_label,
                            update_value=update,
                            evaluation=snapshot_evaluation,
                            resumable=False,
                        )

                if at_epoch_boundary:
                    completed_epoch = epoch + 1
                    completed_update = _update_for_completed_epoch(
                        epoch=completed_epoch,
                        scene_visits_per_epoch_per_rank=(
                            scene_visits_per_epoch_per_rank
                        ),
                        scenes_per_rank_per_update=scenes_per_rank_per_update,
                    )
                    if update != completed_update:
                        raise RuntimeError(
                            "Logical update count drift at epoch boundary: "
                            f"observed={update} expected={completed_update}"
                        )
                    evaluation = None
                    if completed_epoch in eval_epochs:
                        if completed_update in snapshot_label_by_update:
                            if int(snapshot_label_by_update[completed_update]) != int(
                                completed_epoch
                            ):
                                raise RuntimeError(
                                    "Snapshot label conflicts with epoch boundary"
                                )
                            evaluation = evaluations[completed_epoch]
                        else:
                            evaluation = evaluate_and_record(
                                completed_epoch, completed_update
                            )
                    if completed_epoch in checkpoint_epochs:
                        save_checkpoint(
                            epoch_label=completed_epoch,
                            update_value=completed_update,
                            evaluation=evaluation,
                            resumable=True,
                        )
                if update >= execution_stop_update:
                    stop_training = True
                    break

        if pending_visits:
            raise RuntimeError(
                "Complete run ended with an incomplete logical scene batch"
            )
        if execution_stop_update < total_updates:
            if update != execution_stop_update:
                raise RuntimeError(
                    "Operational pause update drift: "
                    f"{update} != {execution_stop_update}"
                )
            pause_epoch = (
                execution_stop_update * scenes_per_rank_per_update
            ) // scene_visits_per_epoch_per_rank
            pause_checkpoint = (
                args.output_dir / "checkpoints" / f"epoch_{pause_epoch:04d}.pt"
            )
            pause_checkpoint_exists = torch.tensor(
                int(pause_checkpoint.is_file()) if rank == 0 else 0,
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(pause_checkpoint_exists, src=0)
            if int(pause_checkpoint_exists.item()) != 1:
                raise RuntimeError(
                    "Operational production pause lacks its resumable epoch "
                    f"checkpoint: {pause_checkpoint}"
                )
            if rank == 0:
                _atomic_json(
                    args.output_dir / "run_paused.json",
                    {
                        "schema_version": "litept_pretraining_operational_pause/v1",
                        "experiment_id": EXPERIMENT_ID,
                        "epoch": int(pause_epoch),
                        "update": int(update),
                        "checkpoint_path": str(pause_checkpoint.resolve()),
                        "checkpoint_sha256": sha256_file(pause_checkpoint),
                        "declared_total_updates": int(total_updates),
                        "lr_schedule_continues_without_restart": True,
                        "candidate_rejected": bool(
                            gradient_conflict_state.get("rejected", False)
                        ),
                        "passed": not bool(
                            gradient_conflict_state.get("rejected", False)
                        ),
                    },
                )
            dist.barrier()
            return
        if update != total_updates:
            raise RuntimeError(
                f"Final logical update count drift: {update} != {total_updates}"
            )

        gathered_counts_final: list[dict[str, int] | None] = [
            None for _ in range(world_size)
        ]
        gathered_sampling_final: list[dict[str, Any] | None] = [
            None for _ in range(world_size)
        ]
        gathered_cumulative_health_final: list[dict[str, Any] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered_counts_final, rank_exposure_counts)
        dist.all_gather_object(gathered_sampling_final, rank_sampling_totals)
        dist.all_gather_object(
            gathered_cumulative_health_final, rank_cumulative_health
        )
        if rank == 0:
            typed_final = [
                {key: int(value) for key, value in row.items()}
                for row in gathered_counts_final
                if row is not None
            ]
            global_counts = _merge_count_dicts(typed_final)
            typed_sampling_final = [
                dict(row) for row in gathered_sampling_final if row is not None
            ]
            typed_cumulative_health_final = [
                dict(row)
                for row in gathered_cumulative_health_final
                if row is not None
            ]
            cumulative_health_final = None
            if rank_cumulative_health is not None:
                if len(typed_cumulative_health_final) != world_size:
                    raise RuntimeError("Incomplete final cumulative health")
                cumulative_health_final = _merge_rank_cumulative_health(
                    typed_cumulative_health_final,
                    through_update=total_updates,
                    world_size=world_size,
                    gradient_conflict_controller=gradient_conflict_state,
                )
            final_eval_label, final_checkpoint_path = _final_evaluation_endpoint(
                nominal_epochs=epochs,
                total_updates=total_updates,
                maximum_updates=args.maximum_updates,
                snapshot_label_by_update=snapshot_label_by_update,
                checkpoint_epochs=checkpoint_epochs,
                checkpoint_updates=checkpoint_updates,
                checkpoint_dir=args.output_dir / "checkpoints",
            )
            final_eval = evaluations.get(final_eval_label)
            if final_eval is None:
                checkpoint = torch.load(
                    final_checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    int(checkpoint.get("epoch", -1)) != final_eval_label
                    or int(checkpoint.get("update", -1)) != total_updates
                ):
                    raise RuntimeError(
                        "Final checkpoint boundary disagrees with the exact run endpoint"
                    )
                final_eval = checkpoint["evaluation"]
            checks = {
                "world_size_exact": world_size == expected_world_size,
                "final_update_exact": (
                    total_scene_visits_per_rank
                    == total_updates * scenes_per_rank_per_update
                ),
                "logical_batch_exact": (
                    resolved["scenes_per_rank_per_update"]
                    == scenes_per_rank_per_update
                    and resolved["scenes_per_global_update"]
                    == scenes_per_rank_per_update * world_size
                    and resolved["proposals_per_global_update"]
                    == (
                        scenes_per_rank_per_update
                        * world_size
                        * int(sampling.proposals_per_forward)
                    )
                ),
                "input_feature_contract_exact": (
                    resolved["input_features"] == args.input_features
                    and resolved["input_channels"]
                    == (3 if args.input_features == "rgb3" else 6)
                    and initial_report["input_features"] == args.input_features
                ),
                "representative_sampling_contract_exact": (
                    resolved["representative_sampling"]
                    == args.representative_sampling
                    and (
                        train_augmentation_profile
                        not in {
                            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
                            TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
                        }
                        or args.representative_sampling == "first"
                    )
                ),
                "rgbn6_color_contract_exact": (
                    train_augmentation_profile
                    not in {
                        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
                        TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
                    }
                    or (
                        resolved["rgb_color_normalization"]
                        == RGBN6_COLOR_NORMALIZATION
                        and resolved["rgb_color_mean"]
                        == list(COMMON_RGB_COLOR_MEAN)
                        and resolved["rgb_color_std"]
                        == list(COMMON_RGB_COLOR_STD)
                    )
                ),
                "pretraining_recipe_exact": (
                    resolved["pretraining_recipe"] == model.recipe_name
                    and resolved["multiscale_supervision"]
                    == bool(args.multiscale_supervision)
                    and resolved["multiscale_levels"]
                    == list(MULTISCALE_LEVEL_NAMES)
                    and resolved["multiscale_level_loss_weights"]
                    == dict(MULTISCALE_LEVEL_LOSS_WEIGHTS)
                    and resolved["multiscale_loss_weight"]
                    == float(args.multiscale_loss_weight)
                    and resolved["multiscale_warmup_epochs"]
                    == int(args.multiscale_warmup_epochs)
                    and resolved["multiscale_token_policy"]
                    == (
                        "strict_pure_token_resampling_v1"
                        if args.multiscale_supervision
                        else None
                    )
                    and resolved["multiscale_temperature_policy"]
                    == (
                        "separate_learnable_temperature_per_auxiliary_level"
                        if args.multiscale_supervision
                        else None
                    )
                    and resolved["multiscale_feature_hard_policy"]
                    == (
                        "separate_candidate_selection_per_positive_endpoint"
                        if args.multiscale_supervision
                        else None
                    )
                    and resolved["multiscale_transfer_contract"]
                    == (
                        "native_mask3d_fpn_required"
                        if args.multiscale_supervision or hierarchy_enabled
                        else "final_dec0_feature"
                    )
                    and resolved["sampling_mode"] == sampling_mode
                    and resolved["token_pure_dec0"]
                    == (sampling_mode == V2_SAMPLING_MODE)
                    and resolved["hierarchy_supervision"]["mode"]
                    == hierarchy_mode
                    and resolved["normalization"]["policy"]
                    == str(args.normalization_policy)
                ),
                "train_augmentation_contract_exact": (
                    resolved["train_augmentation_profile"]
                    == train_augmentation_profile
                    and resolved["train_augmentation"]
                    == augmentation_contract_for_profile(
                        train_augmentation_profile
                    )
                    and resolved["evaluation_augmentation_profile"] == "none"
                ),
                "source_crop_mode_exact": (
                    resolved["required_crop_mode"] == required_crop_mode
                    and (
                        required_crop_mode == "any"
                        or resolved["loaded_crop_modes_rank0"]
                        == [required_crop_mode]
                    )
                ),
                "source_loading_contract_exact": (
                    resolved["source_loading_mode"] == source_loading_mode
                    and resolved["source_loader_workers"]
                    == source_loader_workers
                ),
                "all_metrics_finite": all(
                    math.isfinite(float(value))
                    for value in final_eval["native_selection_macro"].values()
                ),
                "initial_checkpoint_exact": (
                    initial_report["sha256"]
                    == resolved["initialization"]["sha256"]
                ),
                "initialization_seed_exact": (
                    initial_report["seed"] == int(args.initialization_seed)
                    and resolved["initialization_seed"]
                    == int(args.initialization_seed)
                ),
                "fixed_eval_seed_exact": (
                    fixed_eval_pack_report is None
                    or (
                        int(fixed_eval_pack_report.get("seed", -1))
                        == int(args.fixed_eval_seed)
                        and resolved["fixed_eval_seed"]
                        == int(args.fixed_eval_seed)
                    )
                ),
                "source_cells_exact": set(global_counts) == set(source_cells),
                "scene_voxelization_cache_disabled": (
                    resolved["cache_training_voxelization"] is False
                    and resolved["voxelization_mode"]
                    == "per_forward_single_scene_recompute"
                ),
                "sampling_profile_exact": (
                    resolved["sampling_profile"] == args.sampling_profile
                    and resolved["sampling"] == vars(sampling)
                ),
                "evaluation_sampling_exact": (
                    resolved["evaluation_sampling_profile"]
                    == evaluation_sampling_profile
                    and resolved["evaluation_sampling"]
                    == vars(evaluation_sampling)
                ),
                "source_cell_schedule_exact": (
                    resolved["source_cell_schedule"]
                    == args.source_cell_schedule
                    and resolved["source_cell_schedule_seed"] == int(args.seed)
                ),
                "code_manifest_pinned": bool(
                    code_manifest_report
                    and sha256_file(Path(code_manifest_report["path"]))
                    == code_manifest_report["sha256"]
                )
                if (
                    args.sampling_profile == "pf_hnm_768_v1"
                    or args.multiscale_supervision
                    or sampling_mode == V2_SAMPLING_MODE
                )
                else True,
            }
            if args.multiscale_supervision:
                legacy_cumulative = (
                    cumulative_health_final.get("legacy_multiscale_delivery")
                    if isinstance(cumulative_health_final, Mapping)
                    else None
                )
                expected_legacy_microbatches = (
                    total_updates * scenes_per_rank_per_update * world_size
                )
                checks["legacy_multiscale_cumulative_exact"] = bool(
                    isinstance(legacy_cumulative, Mapping)
                    and legacy_cumulative.get("enabled") is True
                    and legacy_cumulative.get("expected_microbatch_count")
                    == expected_legacy_microbatches
                    and legacy_cumulative.get("microbatch_count")
                    == expected_legacy_microbatches
                    and legacy_cumulative.get("recorded_microbatch_count")
                    == legacy_cumulative.get("optimization_valid_count")
                    and legacy_cumulative.get("optimization_valid_count", 0)
                    + legacy_cumulative.get("optimization_invalid_count", 0)
                    == expected_legacy_microbatches
                    and set(legacy_cumulative.get("per_stage", {}))
                    == set(LEGACY_MULTISCALE_CUMULATIVE_STAGES)
                    and isinstance(cumulative_health_final, Mapping)
                    and cumulative_health_final["gradient_panels"][
                        "complete_through_update"
                    ] is True
                    and cumulative_health_final["gradient_panels"][
                        "expected_updates"
                    ]
                    == cumulative_health_final["gradient_panels"][
                        "observed_updates"
                    ]
                )
                diagnostics = final_eval.get("multiscale_diagnostics", {})
                checks["multiscale_holdout_diagnostics_valid"] = (
                    set(diagnostics) == set(MULTISCALE_LEVEL_NAMES)
                    and all(
                        int(diagnostics[name].get("attempted_batches", 0)) > 0
                        and int(diagnostics[name].get("valid_batches", 0)) > 0
                        and float(
                            diagnostics[name].get("valid_batch_fraction", 0.0)
                        )
                        > 0.0
                        and float(
                            diagnostics[name]
                            .get("mapping", {})
                            .get("accepted_mixed_tokens", -1.0)
                        )
                        == 0.0
                        and math.isclose(
                            float(
                                diagnostics[name]
                                .get("mapping", {})
                                .get("strict_token_purity", 0.0)
                            ),
                            1.0,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        and all(
                            math.isfinite(float(value))
                            for section in (
                                "conditional_metrics",
                                "native_health",
                                "projected_health",
                            )
                            for value in diagnostics[name]
                            .get(section, {})
                            .values()
                        )
                        for name in MULTISCALE_LEVEL_NAMES
                    )
                )
            if args.fixed_eval_pack is not None:
                checks["fixed_eval_pack_pinned"] = bool(
                    fixed_eval_pack_report
                    and fixed_eval_pack_report.get("passed")
                    and final_eval.get("selection_view") == "fixed_comparable"
                    and final_eval.get("fixed_eval_pack", {}).get(
                        "manifest_sha256"
                    )
                    == fixed_eval_pack_report["manifest_sha256"]
                )
                if sampling_mode == V2_SAMPLING_MODE:
                    checks["fixed_eval_pack_exact_token_relation_contract"] = bool(
                        isinstance(
                            fixed_eval_pack_report.get(
                                "token_pure_dec0_fixed_pack"
                            ),
                            Mapping,
                        )
                        and final_eval.get(
                            "native_fixed_comparable_dec0_macro_cosine_gap_valid"
                        )
                        is True
                    )
            elif args.sampling_profile == "pf_hnm_768_v1":
                checks["fixed_eval_pack_pinned"] = False
            if extension_enabled:
                checks["continuation_parent_pinned"] = (
                    resolved["continuation"]["enabled"] is True
                    and resolved["continuation"]["parent_checkpoint"]["epoch"]
                    == extension_start_epoch
                    and resolved["continuation"]["parent_checkpoint"]["update"]
                    == extension_start_update
                    and resolved["lr_schedule"]["kind"]
                    == "completed_checkpoint_cosine_restart"
                )
            if initial_report["public_checkpoint_used"]:
                checks["public_checkpoint_used_and_pinned"] = (
                    resolved["public_checkpoint_used"] is True
                    and initial_report["initialization_kind"]
                    == "public_litept_backbone"
                    and int(
                        initial_report["public_checkpoint"][
                            "loaded_backbone_keys"
                        ]
                    )
                    == 302
                )
            else:
                checks["public_checkpoint_unused"] = (
                    resolved["public_checkpoint_used"] is False
                )
            if sampling_mode == V2_SAMPLING_MODE:
                global_sampling_totals = {
                    cell: {
                        key: sum(
                            int(rank_totals[cell][key])
                            for rank_totals in typed_sampling_final
                        )
                        for key in rank_sampling_totals[cell]
                    }
                    for cell in source_cells
                }
                aggregate_sampling = {
                    key: sum(
                        global_sampling_totals[cell][key]
                        for cell in source_cells
                    )
                    for key in rank_sampling_totals[source_cells[0]]
                }
                loss_mass_total = aggregate_sampling[
                    "optimized_loss_mass_millionths"
                ]
                loss_shares = {
                    cell: global_sampling_totals[cell][
                        "optimized_loss_mass_millionths"
                    ]
                    / max(loss_mass_total, 1)
                    for cell in source_cells
                }
                checks["source_exposure_exact"] = (
                    global_counts
                    == {
                        cell: global_sampling_totals[cell][
                            "realized_unique_proposals"
                        ]
                        for cell in source_cells
                    }
                )
                checks["sampling_zero_duplicates"] = (
                    aggregate_sampling["duplicate_proposals"] == 0
                )
                checks["sampling_proposal_survival_at_least_90pct"] = (
                    aggregate_sampling["contrastive_retained_proposals"]
                    / max(aggregate_sampling["realized_unique_proposals"], 1)
                    >= 0.90
                )
                checks["sampling_pair_yield_at_least_85pct"] = (
                    aggregate_sampling["token_realized_positive_pairs"]
                    / max(
                        aggregate_sampling["token_requested_positive_pairs"],
                        1,
                    )
                    >= 0.85
                )
                checks["optimized_visit_fraction_at_least_95pct"] = (
                    aggregate_sampling["optimized_scene_visits"]
                    / max(
                        aggregate_sampling["optimized_scene_visits"]
                        + aggregate_sampling["invalid_scene_visits"],
                        1,
                    )
                    >= 0.95
                )
                checks["all_three_cells_reached_optimized_token_loss"] = all(
                    int(
                        global_sampling_totals[cell][
                            "token_realized_positive_pairs"
                        ]
                    )
                    > 0
                    and int(
                        global_sampling_totals[cell]["optimized_scene_visits"]
                    )
                    > 0
                    for cell in source_cells
                )
                checks["granularity_loss_share_within_5pp"] = all(
                    abs(float(loss_shares[cell]) - 1.0 / 3.0) <= 0.05
                    for cell in source_cells
                )
                checks["global_shared_auxiliary_budget_passed"] = bool(
                    gradient_conflict_state.get("auxiliary_budget", {}).get(
                        "passed", False
                    )
                ) and not bool(gradient_conflict_state.get("rejected", False))
                if cumulative_health_final is None:
                    raise RuntimeError("V2 final cumulative health is missing")
                expected_rank_microbatches = (
                    total_updates
                    * scenes_per_rank_per_update
                    * world_size
                )
                cumulative_clipping = cumulative_health_final["clipping"]
                cumulative_panels = cumulative_health_final["gradient_panels"]
                cumulative_token = cumulative_health_final["token_delivery"]
                cumulative_hierarchy = cumulative_health_final[
                    "hierarchy_delivery"
                ]
                checks["cumulative_health_exact"] = bool(
                    cumulative_health_final["through_update"] == total_updates
                    and cumulative_health_final["optimizer_update_count"]
                    == total_updates
                    and cumulative_clipping["clipped_update_count"]
                    + cumulative_clipping["unclipped_update_count"]
                    == total_updates
                    and cumulative_clipping["rank_update_event_count"]
                    == total_updates * world_size
                    and cumulative_token["microbatch_count"]
                    == expected_rank_microbatches
                    and cumulative_token["optimization_valid_count"]
                    + cumulative_token["optimization_invalid_count"]
                    == expected_rank_microbatches
                    and (
                        not hierarchy_enabled
                        or (
                            cumulative_hierarchy["microbatch_count"]
                            == expected_rank_microbatches
                            and cumulative_hierarchy[
                                "optimization_valid_count"
                            ]
                            + cumulative_hierarchy[
                                "optimization_invalid_count"
                            ]
                            == expected_rank_microbatches
                            and cumulative_hierarchy["vicreg_valid_count"]
                            + cumulative_hierarchy["vicreg_invalid_count"]
                            == expected_rank_microbatches
                        )
                    )
                    and cumulative_panels["complete_through_update"] is True
                    and cumulative_panels["expected_updates"]
                    == cumulative_panels["observed_updates"]
                    and cumulative_health_final[
                        "gradient_conflict_controller"
                    ]
                    == gradient_conflict_state
                )
                expected_counts = {
                    "sampling_totals": global_sampling_totals,
                    "loss_shares": loss_shares,
                }
            elif not smoke:
                expected_counts = exposure_counts(
                    load_scene_ids(args.train_split),
                    epochs=epochs,
                    cells=source_cells,
                    proposals_per_forward=sampling.proposals_per_forward,
                    cell_schedule=args.source_cell_schedule,
                    seed=args.seed,
                )
                checks["source_exposure_exact"] = global_counts == expected_counts
            else:
                expected_counts = exposure_counts(
                    [
                        scene.scene_id
                        for _ in range(world_size)
                        for scene in smoke_scenes
                    ],
                    epochs=epochs,
                    cells=source_cells,
                    proposals_per_forward=sampling.proposals_per_forward,
                    cell_schedule=args.source_cell_schedule,
                    seed=args.seed,
                )
                checks["source_exposure_exact"] = global_counts == expected_counts
            summary = {
                "schema_version": (
                    "litept_rgbn6_multigranular_pretrain_summary/v2"
                    if sampling_mode == V2_SAMPLING_MODE
                    else "litept_gt3_unsam3_pretrain_summary/v1"
                ),
                "project_name": project_name(),
                "project_slug": project_slug(),
                "experiment_id": EXPERIMENT_ID,
                "dataset": args.dataset,
                "mode": (
                    f"{args.dataset}_smoke"
                    if smoke
                    else f"{args.dataset}_production"
                ),
                "epochs": epochs,
                "updates": total_updates,
                "seed": int(args.seed),
                "initialization_seed": int(args.initialization_seed),
                "fixed_eval_seed": int(args.fixed_eval_seed),
                "scene_visits_per_epoch_per_rank": (
                    scene_visits_per_epoch_per_rank
                ),
                "scenes_per_rank_per_update": scenes_per_rank_per_update,
                "scenes_per_global_update": scenes_per_rank_per_update * world_size,
                "backbone_lr": float(args.backbone_lr),
                "head_lr": float(args.head_lr),
                "pretraining_recipe": model.recipe_name,
                "sampling_mode": sampling_mode,
                "token_pure_dec0": bool(model.token_pure_dec0),
                "hierarchy_supervision": resolved["hierarchy_supervision"],
                "normalization": resolved["normalization"],
                "feature_hard_gate": feature_hard_state,
                "cumulative_health": cumulative_health_final,
                "multiscale_supervision": bool(args.multiscale_supervision),
                "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
                "multiscale_level_loss_weights": dict(
                    MULTISCALE_LEVEL_LOSS_WEIGHTS
                ),
                "multiscale_loss_weight": float(args.multiscale_loss_weight),
                "multiscale_warmup_epochs": int(args.multiscale_warmup_epochs),
                "multiscale_token_policy": (
                    "strict_pure_token_resampling_v1"
                    if args.multiscale_supervision
                    else None
                ),
                "multiscale_temperature_policy": (
                    "separate_learnable_temperature_per_auxiliary_level"
                    if args.multiscale_supervision
                    else None
                ),
                "multiscale_feature_hard_policy": (
                    "separate_candidate_selection_per_positive_endpoint"
                    if args.multiscale_supervision
                    else None
                ),
                "multiscale_transfer_contract": (
                    "native_mask3d_fpn_required"
                    if args.multiscale_supervision or hierarchy_enabled
                    else "final_dec0_feature"
                ),
                "input_features": args.input_features,
                "input_channels": int(model.input_channels),
                "coordinate_normalization_policy": str(
                    args.coordinate_normalization_policy
                ),
                "train_augmentation_profile": train_augmentation_profile,
                "evaluation_augmentation_profile": "none",
                "required_crop_mode": required_crop_mode,
                "source_loading_mode": source_loading_mode,
                "source_loader_workers": source_loader_workers,
                "sampling_profile": args.sampling_profile,
                "sampling": vars(sampling),
                "evaluation_sampling_profile": evaluation_sampling_profile,
                "evaluation_sampling": vars(evaluation_sampling),
                "evaluation_proposals_per_forward": int(
                    evaluation_sampling.proposals_per_forward
                ),
                "proposals_per_global_update": (
                    scenes_per_rank_per_update
                    * world_size
                    * int(sampling.proposals_per_forward)
                ),
                "source_cell_schedule": args.source_cell_schedule,
                "source_cell_schedule_seed": int(args.seed),
                "fixed_eval_pack": fixed_eval_pack_report,
                "code_manifest": code_manifest_report,
                "cache_training_voxelization": False,
                "voxelization_mode": "per_forward_single_scene_recompute",
                "lr_schedule": resolved["lr_schedule"],
                "continuation": resolved["continuation"],
                "initialization": initial_report,
                "public_checkpoint_used": bool(
                    initial_report["public_checkpoint_used"]
                ),
                "checks": checks,
                "mechanics_passed": all(checks.values()),
                "health_selection_status": (
                    "pending_external_five_stage_bn_recalibrated_gate"
                    if sampling_mode == V2_SAMPLING_MODE
                    else "not_applicable"
                ),
                "passed": (
                    None
                    if sampling_mode == V2_SAMPLING_MODE
                    else all(checks.values())
                ),
                "global_exposure_counts": global_counts,
                "expected_exposure_counts": expected_counts,
                "final_evaluation": final_eval,
                "training_wall_seconds": time.perf_counter() - started,
                "final_checkpoint": str(final_checkpoint_path),
            }
            _atomic_json(args.output_dir / "run_summary.json", summary)
            if not summary["mechanics_passed"]:
                raise RuntimeError(f"Final run checks failed: {checks}")
        dist.barrier()
    finally:
        if source_loader_pool is not None:
            source_loader_pool.shutdown(wait=True, cancel_futures=True)
        dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _run()


if __name__ == "__main__":
    main()
