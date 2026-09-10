"""Training primitives for scratch LitePT GT3 + UnSAM3 pretraining."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from delimit3d.training.contracts import (
    GRANULARITY_KEYS,
    RAW_CACHE_SCHEMA,
    SOURCE_CELLS,
    SOURCE_CELL_SCHEDULE_FIXED,
    TWO_D_CELLS,
    normalize_source_cells,
    audit_scene_source_manifest,
    lr_factor,
    sha256_file,
    source_cell_for_scene_epoch,
    stable_seed,
)
from delimit3d.data.partfield_contrastive import (
    SAMPLER_ALGORITHM_VERSION,
    SAMPLER_PRESELECTION_POLICY,
    TOKEN_POSITIVE_PAIR_PAYLOAD_SCHEMA,
    ContrastiveTripletBatch,
    DeferredTokenLossPlan,
    concatenate_triplet_batches,
    sample_overlapping_proposal_triplets,
    sample_partition_triplets,
    validate_token_positive_pair_payload,
)
from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    FrameProposalCatalog,
    NoRetainedFrameGroupsError,
    rotating_three_cell_quotas,
    sample_multigranular_frame_group_plan,
)
from delimit3d.data.point_augmentations import (
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    augment_points_mask3d_scannet_rgbn6,
    canonicalize_structured3d_points_normals_to_scannet,
)
from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion
from delimit3d.losses.litept_hierarchy_supervision import (
    HIERARCHY_COARSE_TO_FINE,
    HIERARCHY_QUERY_PROJECTION_POLICIES,
    HIERARCHY_STAGE_CHANNELS,
    MaskSupervisionGroup,
    ProposalConditionedHierarchyLoss,
    build_token_pure_dec0_contrastive_batch,
    hierarchy_gradient_telemetry,
    hierarchy_point_maps,
    native_hierarchy_vicreg_loss,
)
from delimit3d.models.litept_wrapper import LitePTBackbone, LitePTBackboneOutput


@dataclass(frozen=True)
class SamplingConfig:
    proposals_per_forward: int = 4
    positive_pairs_per_proposal: int = 64
    uniform_negatives: int = 256
    spatial_hard_negatives: int = 128
    spatial_candidate_pool: int = 1024
    feature_hard_negatives: int = 128
    feature_candidate_pool: int = 1024


SAMPLING_PROFILES: dict[str, SamplingConfig] = {
    "pf_hnm_512_v1": SamplingConfig(),
    "pf_hnm_512_p16_v1": SamplingConfig(proposals_per_forward=16),
    "pf_hnm_768_v1": SamplingConfig(
        uniform_negatives=256,
        spatial_hard_negatives=256,
        spatial_candidate_pool=1024,
        feature_hard_negatives=256,
        feature_candidate_pool=1024,
    ),
}
FIXED_EVAL_PACK_SCHEMA = "litept_gt3_unsam3_fixed_eval_pack/v1"
FIXED_EVAL_SCENE_SCHEMA = "litept_gt3_unsam3_fixed_eval_scene/v1"
TOKEN_PURE_FIXED_EVAL_SCHEMA = "litept_token_pure_dec0_fixed_pack/v5"
TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY = (
    "rebuilt_after_voxelization_from_saved_deferred_widths"
)
TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION = (
    "runtime_rebuild_from_selected_proposal_csr_after_voxelization"
)
CLEAN_TRAINING_SOURCE_SCHEMA = "structured3d_clean_2d_training_scene_source/v1"
CLEAN_CROP_PACK_SCHEMA = "structured3d_clean_2d_crop_pack/v1"
CLEAN_PROJECTION_CACHE_SCHEMA = "structured3d_clean_2d_projection_cache/v1"
TRAINING_SAMPLING_MODES = (
    "random_with_replacement",
    "coverage_without_replacement",
    "hybrid_2d_random_3d_coverage",
)
INPUT_FEATURE_CHANNELS: dict[str, int] = {
    "rgb3": 3,
    "rgbn6": 6,
}
INPUT_FEATURE_CHOICES = tuple(INPUT_FEATURE_CHANNELS)
# RGBN6 is a transfer contract, not merely a channel-count choice.  LitePT's
# released Pointcept pipelines use NormalizeColor, i.e. RGB / 255, for both
# ScanNet and Structured3D.  Keep that fixed affine map at both sides of this
# transfer so the encoder never sees a dataset-specific input coordinate
# system at fine-tuning time.
COMMON_RGB_COLOR_MEAN = (0.0, 0.0, 0.0)
COMMON_RGB_COLOR_STD = (1.0, 1.0, 1.0)
RGBN6_COLOR_NORMALIZATION = "litept_rgb_0_1_no_mean_std"
MULTISCALE_RECIPE_NAME = "final_dec0_plus_decoder_stage_deep_supervision_v1"
MULTISCALE_LEVEL_NAMES = ("dec3", "dec2", "dec1")
MULTISCALE_LEVEL_INDICES = (0, 1, 2)
# The final dec0 objective remains primary.  The auxiliary budget is tapered
# toward the finer decoder stages rather than treating all resolutions as
# interchangeable: effective full-strength weights are dec3=0.08, dec2=0.12,
# and dec1=0.20.  This mirrors the stagewise weighting used by the closest
# multi-level contrastive precedent while keeping the total auxiliary budget
# below half of the final objective.
MULTISCALE_LEVEL_LOSS_WEIGHTS: dict[str, float] = {
    "dec3": 0.20,
    "dec2": 0.30,
    "dec1": 0.50,
}
MULTISCALE_AUXILIARY_WEIGHT = 0.4
MULTISCALE_AUXILIARY_WARMUP_EPOCHS = 20
HIERARCHY_RECIPE_NAME = "final_dec0_plus_proposal_query_hierarchy_v1"
HIERARCHY_MASK_LOSS_WEIGHT = 0.25
HIERARCHY_VICREG_LOSS_WEIGHT = 0.0
HIERARCHY_WARMUP_FRACTION = 0.05
HIERARCHY_VARIANCE_TARGET = 1.0
HIERARCHY_COVARIANCE_WEIGHT = 0.04
HIERARCHY_VICREG_MAX_TOKENS = 4096
# ``dec0`` is the primary contrastive objective, not a proposal-query mask
# target.  These are the only stages for which a relative hierarchy-mask
# multiplier has meaning.  The default is deliberately all ones so existing
# V2 recipes preserve their exact objective arithmetic.
HIERARCHY_MASK_STAGE_NAMES = tuple(
    stage for stage in HIERARCHY_COARSE_TO_FINE if stage != "dec0"
)
TOKEN_PURE_DEC0_MAX_PAIRS_PER_PROPOSAL = 64
TRAIN_AUGMENTATION_NONE = "none"
TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE = (
    "mask3d_scannet_rgbn6_fullscene_v1"
)
TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K = (
    "mask3d_scannet_rgbn6_sphere50k_v1"
)
TRAIN_AUGMENTATION_CHOICES = (
    TRAIN_AUGMENTATION_NONE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
)


def normalize_hierarchy_mask_stage_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return a complete, validated relative query-mask stage weighting.

    This is intentionally relative to ``hierarchy_mask_loss_weight``.  It
    lets a calibration increase only a demonstrably under-served stage while
    preserving the established global objective, rather than changing every
    routed edge together.  Omitting the option is exactly equivalent to the
    pre-existing all-one weighting.
    """

    supplied = {} if weights is None else dict(weights)
    unknown = sorted(set(supplied) - set(HIERARCHY_MASK_STAGE_NAMES))
    if unknown:
        raise ValueError(
            "Unknown hierarchy-mask stage weights: "
            f"{unknown!r}; expected {list(HIERARCHY_MASK_STAGE_NAMES)!r}"
        )
    normalized: dict[str, float] = {}
    for stage in HIERARCHY_MASK_STAGE_NAMES:
        value = float(supplied.get(stage, 1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"hierarchy-mask stage weight for {stage} must be finite and non-negative"
            )
        normalized[stage] = value
    return normalized


def normalize_hierarchy_mask_stage_weight_schedule(
    schedule: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate an optional deterministic early/late stage-weight schedule.

    The schedule is deliberately a single hard transition rather than a new
    learnable controller: it preserves exact reproducibility across resume and
    makes the causal question explicit.  It is useful when a coarse stage
    needs stronger early shaping while decoder-stage auxiliaries benefit from
    a lower late coefficient.
    """

    if schedule is None:
        return None
    supplied = dict(schedule)
    expected = {"transition_fraction", "early", "late"}
    unknown = sorted(set(supplied) - expected)
    missing = sorted(expected - set(supplied))
    if unknown or missing:
        raise ValueError(
            "hierarchy-mask stage-weight schedule must contain exactly "
            f"{sorted(expected)!r}; missing={missing!r}, unknown={unknown!r}"
        )
    fraction = float(supplied["transition_fraction"])
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            "hierarchy-mask stage-weight schedule transition_fraction must lie in (0, 1)"
        )
    early = normalize_hierarchy_mask_stage_weights(supplied["early"])
    late = normalize_hierarchy_mask_stage_weights(supplied["late"])
    return {
        "transition_fraction": fraction,
        "early": early,
        "late": late,
    }


def hierarchy_mask_stage_weights_for_step(
    *,
    static_weights: Mapping[str, float],
    schedule: Mapping[str, Any] | None,
    supervision_step: int | None,
    supervision_total_steps: int | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select static or scheduled stage weights and emit replayable telemetry."""

    normalized_static = normalize_hierarchy_mask_stage_weights(static_weights)
    normalized_schedule = normalize_hierarchy_mask_stage_weight_schedule(schedule)
    if normalized_schedule is None:
        return normalized_static, {
            "enabled": False,
            "phase": "static",
            "progress_fraction": None,
            "transition_fraction": None,
        }
    if supervision_step is None or supervision_total_steps is None:
        return dict(normalized_schedule["early"]), {
            "enabled": True,
            "phase": "early_missing_step_context",
            "progress_fraction": None,
            "transition_fraction": normalized_schedule["transition_fraction"],
        }
    step = int(supervision_step)
    total = int(supervision_total_steps)
    if step < 0 or total <= 0 or step > total:
        raise ValueError(
            "hierarchy-mask stage-weight schedule requires 0 <= supervision_step <= total"
        )
    # The runner numbers optimizer updates from one.  Select the phase from
    # progress *before* the current update so a 0.5 transition in a 256-step
    # calibration gives exactly 128 early and 128 late objective evaluations,
    # rather than silently giving the early phase only 127 updates.
    progress = float(max(step - 1, 0)) / float(total)
    phase = (
        "early"
        if progress < float(normalized_schedule["transition_fraction"])
        else "late"
    )
    return dict(normalized_schedule[phase]), {
        "enabled": True,
        "phase": phase,
        "progress_fraction": progress,
        "transition_fraction": normalized_schedule["transition_fraction"],
    }


def weighted_hierarchy_mask_loss(
    *,
    loss_total: torch.Tensor,
    stage_loss_tensors: Mapping[str, torch.Tensor],
    stage_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply relative stage weights to the already cell-balanced mask loss.

    ``ProposalConditionedHierarchyLoss`` has already performed the important
    within-cell and cross-granularity normalization before exposing its stage
    contributions.  We only rescale those final contributions; we do not
    reweight masks, frames, or granularities here.  The all-one branch returns
    the original scalar verbatim to preserve legacy arithmetic.
    """

    normalized = normalize_hierarchy_mask_stage_weights(stage_weights)
    weighted = {
        stage: float(normalized[stage])
        * stage_loss_tensors.get(stage, loss_total * 0.0)
        for stage in HIERARCHY_MASK_STAGE_NAMES
    }
    if all(weight == 1.0 for weight in normalized.values()):
        return loss_total, weighted
    total = loss_total * 0.0
    for stage in HIERARCHY_MASK_STAGE_NAMES:
        total = total + weighted[stage]
    return total, weighted
TRAIN_AUGMENTATION_CONTRACTS: dict[str, dict[str, Any]] = {
    TRAIN_AUGMENTATION_NONE: {
        "profile": TRAIN_AUGMENTATION_NONE,
        "stochastic": False,
        "point_removal": False,
    },
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE: {
        "profile": TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
        "stochastic": True,
        "training_only": True,
        "point_removal": False,
        "point_order_preserved": True,
        "rgb_normalization": RGBN6_COLOR_NORMALIZATION,
        "rgb_color_mean": list(COMMON_RGB_COLOR_MEAN),
        "rgb_color_std": list(COMMON_RGB_COLOR_STD),
        "geometry": {
            "center_then_random_translate": True,
            "reflect_x_probability": 0.5,
            "reflect_y_probability": 0.5,
            "anisotropic_scale": {
                "range_per_axis": [0.9, 1.1],
                "probability": 0.5,
                "normal_transform": "inverse_transpose_then_unit_normalize",
            },
            "rotate_z": {
                "range_degrees": [-180.0, 180.0],
                "probability": 0.5,
            },
            "rotate_y": {
                "range_degrees": [-7.5, 7.5],
                "probability": 0.5,
            },
            "rotate_x": {
                "range_degrees": [-7.5, 7.5],
                "probability": 0.5,
            },
            "rigid_normal_transform": True,
            "elastic_distortion": False,
            "coordinate_jitter": False,
        },
        "color": {
            "brightness_contrast": {
                "brightness_range": [-0.2, 0.2],
                "contrast_range": [-0.2, 0.2],
                "probability": 0.5,
            },
            "rgb_shift": {
                "integer_255_range_per_channel": [-20, 20],
                "probability": 0.5,
            },
            "clip_range": [0.0, 1.0],
        },
        "excluded": [
            "point_dropout",
            "sphere_crop",
            "coordinate_jitter",
            "elastic_distortion",
            "normal_jitter",
            "color_dropout",
        ],
        "seed_key": ["global_seed", "scene_id", "epoch", "input-augmentation"],
        "seed_depends_on_rank": False,
        "seed_depends_on_granularity": False,
    },
}

# The fixed-crop arm deliberately uses the same normal-aware transform
# distributions as the full-scene arm.  Its only point-set difference is
# upstream: every source pack is one already-selected 50k sphere crop.  No
# further crop, dropout, or point subsampling is performed during training.
_sphere50k_augmentation_contract = json.loads(
    json.dumps(
        TRAIN_AUGMENTATION_CONTRACTS[
            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE
        ]
    )
)
_sphere50k_augmentation_contract.update(
    {
        "profile": TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
        "required_crop_mode": "sphere_50k",
        "stored_point_contract": (
            "use every stored point from the frozen source crop; at most 50000"
        ),
    }
)
_sphere50k_augmentation_contract["excluded"] = [
    "point_dropout",
    "additional_sphere_crop",
    "point_subsampling",
    "coordinate_jitter",
    "elastic_distortion",
    "normal_jitter",
    "color_dropout",
]
TRAIN_AUGMENTATION_CONTRACTS[
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K
] = _sphere50k_augmentation_contract


def input_channels_for(input_features: str) -> int:
    try:
        return int(INPUT_FEATURE_CHANNELS[str(input_features)])
    except KeyError as exc:
        raise ValueError(
            f"Unknown input feature contract {input_features!r}; "
            f"expected one of {INPUT_FEATURE_CHOICES}"
        ) from exc


def sampling_config_for_profile(profile: str) -> SamplingConfig:
    try:
        return SAMPLING_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sampling profile {profile!r}; "
            f"expected one of {sorted(SAMPLING_PROFILES)}"
        ) from exc


@dataclass(frozen=True)
class RawFrame:
    granularity: str
    physical_frame_id: str
    split: str
    visible_indices: np.ndarray
    proposal_offsets: np.ndarray
    proposal_point_indices: np.ndarray


@dataclass(frozen=True)
class SceneSource:
    scene_id: str
    manifest_path: Path
    manifest_sha256: str
    points: np.ndarray
    features: np.ndarray
    labels_by_key: dict[str, np.ndarray]
    raw_frames: tuple[RawFrame, ...]
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    crop_mode: str = "sphere_50k"
    coordinate_normalization_policy: str = "native"
    crop_id: str | None = None
    source_variant_id: str | None = None

    def frames(self, granularity: str, split: str) -> tuple[RawFrame, ...]:
        return tuple(
            frame
            for frame in self.raw_frames
            if (
                frame.granularity == granularity
                and frame.split == split
                and frame.proposal_offsets.size > 1
            )
        )


@dataclass(frozen=True)
class MultiCropSceneSource:
    """One logical scene whose immutable clean crops rotate across epochs."""

    scene_id: str
    manifest_path: Path
    manifest_sha256: str
    crops: tuple[SceneSource, ...]
    crop_mode: str = "sphere_50k"
    coordinate_normalization_policy: str = (
        STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
    )
    crop_rotation_policy: str = "deterministic_epoch_rotation_v1"

    def __post_init__(self) -> None:
        if not self.crops:
            raise ValueError("a multi-crop source requires at least one crop")
        if any(crop.scene_id != self.scene_id for crop in self.crops):
            raise ValueError("all clean crops must retain the logical scene ID")
        if any(crop.crop_mode != self.crop_mode for crop in self.crops):
            raise ValueError("clean crop modes differ within one logical scene")
        if any(
            crop.coordinate_normalization_policy
            != self.coordinate_normalization_policy
            for crop in self.crops
        ):
            raise ValueError("clean crop coordinate policies differ")


def scene_source_for_visit(
    scene: SceneSource | MultiCropSceneSource,
    *,
    epoch: int,
    global_seed: int,
) -> SceneSource:
    """Resolve one clean crop without changing logical scene optimization mass."""

    if isinstance(scene, SceneSource):
        return scene
    if scene.crop_rotation_policy != "deterministic_epoch_rotation_v1":
        raise ValueError(f"unsupported crop rotation {scene.crop_rotation_policy!r}")
    offset = stable_seed(
        int(global_seed), scene.scene_id, "clean-pack-crop-rotation"
    ) % len(scene.crops)
    return scene.crops[(offset + int(epoch)) % len(scene.crops)]


@dataclass(frozen=True)
class SceneVisitInput:
    points: np.ndarray
    features: np.ndarray
    augmentation_seed: int | None


def augmentation_contract_for_profile(profile: str) -> dict[str, Any]:
    """Return a JSON-safe copy of the frozen train-augmentation contract."""

    try:
        contract = TRAIN_AUGMENTATION_CONTRACTS[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown train augmentation profile {profile!r}; "
            f"expected one of {TRAIN_AUGMENTATION_CHOICES}"
        ) from exc
    return json.loads(json.dumps(contract))


def build_scene_visit_input(
    *,
    scene: SceneSource,
    epoch: int,
    global_seed: int,
    input_features: str,
    augmentation_profile: str,
    color_mean: Sequence[float] | None = None,
    color_std: Sequence[float] | None = None,
) -> SceneVisitInput:
    """Build one point/feature view without mutating the cached scene arrays.

    The augmentation seed intentionally excludes DDP rank, scene-order position,
    selected frame, proposal, and granularity.  A scene therefore receives one
    reproducible transform per epoch/visit, shared by all proposal draws, while
    single- and multi-granularity arms can consume the same augmented views.
    """

    profile = str(augmentation_profile)
    if profile not in TRAIN_AUGMENTATION_CHOICES:
        augmentation_contract_for_profile(profile)
    if profile == TRAIN_AUGMENTATION_NONE:
        if (color_mean is None) != (color_std is None):
            raise ValueError("color_mean and color_std must be supplied together")
        if color_mean is not None:
            features = build_input_features(
                scene.points,
                scene.colors,
                use_colors=True,
                use_normals=input_features == "rgbn6",
                normals=scene.normals,
                color_mean=color_mean,
                color_std=color_std,
            ).astype(np.float32, copy=False)
        else:
            features = scene.features
        return SceneVisitInput(
            points=scene.points,
            features=features,
            augmentation_seed=None,
        )
    required_crop_modes = {
        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE: "full_scene",
        TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K: "sphere_50k",
    }
    if profile not in required_crop_modes:
        raise AssertionError(f"Unhandled train augmentation profile {profile!r}")
    if input_features != "rgbn6":
        raise ValueError(f"{profile} requires input_features='rgbn6'")
    required_crop_mode = required_crop_modes[profile]
    if scene.crop_mode != required_crop_mode:
        raise ValueError(
            f"{scene.scene_id}: {profile} requires "
            f"crop.mode={required_crop_mode!r}, "
            f"got {scene.crop_mode!r}"
        )
    if scene.colors is None or scene.normals is None:
        raise ValueError(f"{scene.scene_id}: {profile} requires RGB and normals")

    augmentation_seed = stable_seed(
        int(global_seed),
        scene.scene_id,
        int(epoch),
        "input-augmentation",
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed % (2**32))
        aug_points, aug_colors, aug_normals = (
            augment_points_mask3d_scannet_rgbn6(
                scene.points,
                scene.colors,
                use_colors=True,
                normals=scene.normals,
            )
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    if aug_colors is None:
        raise RuntimeError(f"{scene.scene_id}: RGB augmentation returned no colors")
    expected_shape = scene.points.shape
    if aug_points.shape != expected_shape or aug_colors.shape != expected_shape:
        raise RuntimeError(
            f"{scene.scene_id}: augmentation changed point/color cardinality"
        )
    if aug_normals.shape != expected_shape:
        raise RuntimeError(f"{scene.scene_id}: augmentation changed normal cardinality")
    if not (
        np.isfinite(aug_points).all()
        and np.isfinite(aug_colors).all()
        and np.isfinite(aug_normals).all()
    ):
        raise RuntimeError(f"{scene.scene_id}: augmentation produced non-finite values")
    features = build_input_features(
        aug_points,
        aug_colors,
        use_colors=True,
        use_normals=True,
        normals=aug_normals,
        color_mean=color_mean,
        color_std=color_std,
    ).astype(np.float32, copy=False)
    return SceneVisitInput(
        points=np.ascontiguousarray(aug_points, dtype=np.float32),
        features=np.ascontiguousarray(features, dtype=np.float32),
        augmentation_seed=int(augmentation_seed),
    )


class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self, in_channels: int = 72, hidden_dim: int = 128, out_dim: int = 128
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def zero_loss_touching_module_parameters(
    reference: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Return scalar zero while keeping a module in the synchronized graph."""
    loss = reference.sum() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            loss = loss + parameter.sum() * 0.0
    return loss


def remap_contrastive_batch_to_level(
    batch: ContrastiveTripletBatch,
    point_to_level: torch.Tensor,
) -> tuple[ContrastiveTripletBatch | None, dict[str, int]]:
    """Map point-indexed tuples to one sparse feature level.

    A coarse LitePT level can merge both positive endpoints or a negative into
    a token sampled as positive for the same proposal. Keeping such a tuple
    would manufacture a false/cosine-one negative, so those rows and colliding
    candidates are removed explicitly. Repeated mapped tokens are deduplicated
    in original sampling order, and rows with no surviving base negative are
    dropped without suppressing valid rows. The remaining rows are
    rectangularized to the largest common negative count and can therefore be
    passed to the existing criterion unchanged.
    """

    if point_to_level.ndim != 1:
        raise ValueError(
            f"point_to_level must have shape [N], got {tuple(point_to_level.shape)}"
        )
    if point_to_level.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"point_to_level must be integral, got {point_to_level.dtype}")

    point_to_level = point_to_level.long()
    positive_pairs_all = point_to_level[batch.positive_pairs.long()]
    negative_indices_all = point_to_level[batch.negative_indices.long()]
    feature_candidates_all = (
        point_to_level[batch.feature_candidate_indices.long()]
        if batch.feature_candidate_indices is not None
        else None
    )

    input_pairs = int(positive_pairs_all.shape[0])
    collapsed_mask = positive_pairs_all[:, 0] == positive_pairs_all[:, 1]
    collapsed_positive_pairs = int(collapsed_mask.sum().item())
    valid_row_indices = torch.where(~collapsed_mask)[0]

    empty_stats = {
        "input_pairs": input_pairs,
        "valid_pairs": int(valid_row_indices.numel()),
        "kept_pairs": 0,
        "collapsed_positive_pairs": collapsed_positive_pairs,
        "zero_negative_pairs": 0,
        "base_negatives_per_pair": 0,
        "base_negative_collisions": 0,
        "base_negative_duplicates": 0,
        "feature_candidate_pool": 0,
        "feature_hard_negatives": 0,
        "feature_candidate_collisions": 0,
        "feature_candidate_duplicates": 0,
    }
    if valid_row_indices.numel() == 0:
        return None, empty_stats

    positive_pairs = positive_pairs_all[valid_row_indices]
    proposal_labels = batch.proposal_labels[valid_row_indices]
    negative_indices = negative_indices_all[valid_row_indices]

    def proposal_positive_collision_mask(values: torch.Tensor) -> torch.Tensor:
        """Mark tokens sampled as positive anywhere for the row's proposal."""
        collisions = torch.zeros_like(values, dtype=torch.bool)
        for proposal_label in torch.unique(proposal_labels, sorted=True):
            rows = proposal_labels == proposal_label
            all_rows = batch.proposal_labels == proposal_label
            positive_tokens = torch.unique(
                positive_pairs_all[all_rows].reshape(-1), sorted=True
            )
            collisions[rows] = torch.isin(values[rows], positive_tokens)
        return collisions

    def stable_unique_positions(
        values: torch.Tensor,
        invalid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return first-occurrence positions, per-row counts and diagnostics."""
        if values.ndim != 2 or invalid.shape != values.shape:
            raise ValueError("Mapped negative tensors must be rectangular [P,M]")
        width = int(values.shape[1])
        if width == 0:
            positions = torch.empty_like(values, dtype=torch.long)
            counts = torch.zeros(
                values.shape[0], dtype=torch.long, device=values.device
            )
            return positions, counts, 0, 0

        sorted_values, sorted_positions = torch.sort(
            values, dim=1, stable=True
        )
        sorted_invalid = torch.gather(invalid, 1, sorted_positions)
        first_for_value = torch.ones_like(sorted_values, dtype=torch.bool)
        first_for_value[:, 1:] = (
            sorted_values[:, 1:] != sorted_values[:, :-1]
        )
        usable = first_for_value & ~sorted_invalid
        sentinel = torch.full_like(sorted_positions, width)
        first_positions = torch.where(usable, sorted_positions, sentinel)
        # Sorting the original positions restores the sampler's deterministic
        # order and avoids preferentially retaining low sparse-token IDs.
        stable_positions = torch.sort(first_positions, dim=1).values
        counts = (stable_positions < width).sum(dim=1)
        duplicates = int((~first_for_value).sum().item())
        collisions = int((first_for_value & sorted_invalid).sum().item())
        return stable_positions, counts, duplicates, collisions

    base_positions, base_counts, base_duplicates, base_collisions = (
        stable_unique_positions(
            negative_indices,
            proposal_positive_collision_mask(negative_indices),
        )
    )
    rows_with_negatives = base_counts > 0
    zero_negative_pairs = int((~rows_with_negatives).sum().item())
    if not bool(rows_with_negatives.any()):
        empty_stats.update(
            {
                "zero_negative_pairs": zero_negative_pairs,
                "base_negative_collisions": base_collisions,
                "base_negative_duplicates": base_duplicates,
            }
        )
        return None, empty_stats

    positive_pairs = positive_pairs[rows_with_negatives]
    proposal_labels = proposal_labels[rows_with_negatives]
    row_indices = valid_row_indices[rows_with_negatives]
    negative_indices = negative_indices[rows_with_negatives]
    base_positions = base_positions[rows_with_negatives]
    base_counts = base_counts[rows_with_negatives]
    base_count = int(base_counts.min().item())
    remapped_negatives = torch.gather(
        negative_indices, 1, base_positions[:, :base_count]
    )

    candidate_pool = 0
    feature_hard = 0
    remapped_candidates = None
    candidate_duplicates = 0
    candidate_collisions = 0
    requested_feature_hard = int(batch.num_feature_hard_negatives)
    if requested_feature_hard > 0 and feature_candidates_all is not None:
        feature_candidates = feature_candidates_all[row_indices]
        candidate_positions, candidate_counts, candidate_duplicates, candidate_collisions = (
            stable_unique_positions(
                feature_candidates,
                proposal_positive_collision_mask(feature_candidates),
            )
        )
        candidate_pool = int(candidate_counts.min().item())
        feature_hard = min(requested_feature_hard, candidate_pool)
        if feature_hard > 0:
            remapped_candidates = torch.gather(
                feature_candidates,
                1,
                candidate_positions[:, :candidate_pool],
            )

    remapped = ContrastiveTripletBatch(
        positive_pairs=positive_pairs,
        negative_indices=remapped_negatives,
        proposal_labels=batch.proposal_labels[row_indices],
        feature_candidate_indices=remapped_candidates,
        num_feature_hard_negatives=feature_hard,
    )
    return remapped, {
        "input_pairs": input_pairs,
        "valid_pairs": int(valid_row_indices.numel()),
        "kept_pairs": int(remapped.num_pairs),
        "collapsed_positive_pairs": collapsed_positive_pairs,
        "zero_negative_pairs": zero_negative_pairs,
        "base_negatives_per_pair": base_count,
        "base_negative_collisions": base_collisions,
        "base_negative_duplicates": base_duplicates,
        "feature_candidate_pool": candidate_pool,
        "feature_hard_negatives": feature_hard,
        "feature_candidate_collisions": candidate_collisions,
        "feature_candidate_duplicates": candidate_duplicates,
    }


def build_pure_token_contrastive_batch(
    batch: ContrastiveTripletBatch,
    point_to_level: torch.Tensor,
    level_xyz: torch.Tensor,
    *,
    level_name: str,
    include_proposal_diagnostics: bool = False,
) -> tuple[ContrastiveTripletBatch | None, dict[str, Any]]:
    """Sample one fixed-budget contrastive batch from strictly pure tokens.

    Point pairs cannot simply be mapped to a coarse hierarchy: a sparse token
    may contain both proposal and complement points, turning it into a false
    positive or false hard negative.  This routine instead aggregates the
    exact sparse proposal membership carried by ``ContrastiveTripletBatch``.
    A token is accepted as positive only when *all* source points mapped to it
    belong to the proposal, and as negative only when all source points are in
    the proposal's eligible complement.  Mixed and visibility-unknown tokens
    are ignored.  Every retained proposal is resampled to its original pair
    budget, avoiding the proposal-size bias caused by conditioning point pairs
    on coarse-level survival.
    """

    metadata = (
        batch.proposal_member_offsets,
        batch.proposal_member_indices,
        batch.pair_proposal_indices,
        batch.eligible_indices,
    )
    if any(value is None for value in metadata):
        raise ValueError(
            "Multi-scale pure-token sampling requires exact proposal membership"
        )
    assert batch.proposal_member_offsets is not None
    assert batch.proposal_member_indices is not None
    assert batch.pair_proposal_indices is not None
    assert batch.eligible_indices is not None
    if batch.multiscale_seed is None:
        raise ValueError("Multi-scale pure-token sampling requires a frozen seed")
    if point_to_level.ndim != 1 or point_to_level.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("point_to_level must be an integral [N] tensor")
    if level_xyz.ndim != 2 or level_xyz.shape[1] != 3:
        raise ValueError("level_xyz must have shape [V,3]")
    if point_to_level.device != level_xyz.device:
        raise ValueError("point_to_level and level_xyz must share a device")

    point_to_level = point_to_level.long()
    offsets = batch.proposal_member_offsets.long()
    members = batch.proposal_member_indices.long()
    pair_to_proposal = batch.pair_proposal_indices.long()
    eligible = torch.unique(batch.eligible_indices.long(), sorted=True)
    num_points = int(point_to_level.numel())
    num_tokens = int(level_xyz.shape[0])
    num_proposals = int(offsets.numel()) - 1
    if (
        num_tokens <= 0
        or num_proposals <= 0
        or offsets.ndim != 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(members.numel())
        or pair_to_proposal.shape != (batch.num_pairs,)
    ):
        raise ValueError("Malformed proposal-membership CSR contract")
    index_chunks = [members, eligible, pair_to_proposal]
    if any(tensor.device != point_to_level.device for tensor in index_chunks):
        raise ValueError("Multi-scale proposal metadata must share the feature device")
    if members.numel() == 0 or eligible.numel() == 0:
        raise ValueError("Proposal membership and eligibility must be non-empty")
    if (
        int(members.min()) < 0
        or int(members.max()) >= num_points
        or int(eligible.min()) < 0
        or int(eligible.max()) >= num_points
        or int(pair_to_proposal.min()) < 0
        or int(pair_to_proposal.max()) >= num_proposals
    ):
        raise ValueError("Proposal-membership metadata contains an invalid index")
    if not bool(torch.isin(members, eligible).all()):
        raise ValueError("Proposal members must be a subset of eligible points")
    if point_to_level.numel() and (
        int(point_to_level.min()) < 0 or int(point_to_level.max()) >= num_tokens
    ):
        raise ValueError("point_to_level contains an out-of-range token")

    all_counts = torch.bincount(point_to_level, minlength=num_tokens)
    eligible_counts = torch.bincount(
        point_to_level[eligible], minlength=num_tokens
    )
    generator = torch.Generator(device=point_to_level.device).manual_seed(
        stable_seed(
            int(batch.multiscale_seed),
            "pure-token-multiscale",
            str(level_name),
        )
    )

    requested_uniform = int(batch.num_uniform_negatives)
    requested_spatial = int(batch.num_spatial_hard_negatives)
    requested_feature_hard = int(batch.num_feature_hard_negatives)
    requested_spatial_pool = int(batch.spatial_candidate_pool)
    requested_feature_pool = int(batch.feature_candidate_pool)
    if requested_uniform < 0 or requested_spatial < 0:
        raise ValueError("Negative-category counts must be non-negative")
    if requested_uniform + requested_spatial <= 0:
        raise ValueError("Pure-token supervision requires base negatives")
    if requested_spatial > 0 and requested_spatial_pool <= 0:
        raise ValueError("Spatial hard negatives require a candidate pool")
    if requested_feature_hard > 0 and requested_feature_pool <= 0:
        raise ValueError("Feature hard negatives require a candidate pool")

    records: list[dict[str, torch.Tensor]] = []
    total_pure_positive = 0
    total_pure_negative = 0
    total_mixed_or_unknown = 0
    dropped_insufficient_positive = 0
    dropped_no_negative = 0
    input_pair_budget = 0
    retained_pair_budget = 0
    proposal_diagnostics: list[dict[str, int | str | bool]] = []

    with torch.no_grad():
        for proposal_index in range(num_proposals):
            pair_rows = torch.where(pair_to_proposal == proposal_index)[0]
            pair_budget = int(pair_rows.numel())
            if pair_budget <= 0:
                raise ValueError("Every sampled proposal occurrence needs pair rows")
            input_pair_budget += pair_budget
            start = int(offsets[proposal_index])
            end = int(offsets[proposal_index + 1])
            proposal_members = torch.unique(members[start:end], sorted=True)
            positive_counts = torch.bincount(
                point_to_level[proposal_members], minlength=num_tokens
            )

            # Strict means strict over every original scene point mapped to the
            # token, not only points visible in a 2D frame.  Visibility-unknown
            # support therefore cannot masquerade as pure proposal support.
            pure_positive = torch.where(
                (positive_counts > 0) & (positive_counts == all_counts)
            )[0]
            pure_negative = torch.where(
                (positive_counts == 0)
                & (eligible_counts > 0)
                & (eligible_counts == all_counts)
            )[0]
            mixed_or_unknown = torch.where(
                (all_counts > 0)
                & ~(
                    ((positive_counts > 0) & (positive_counts == all_counts))
                    | (
                        (positive_counts == 0)
                        & (eligible_counts > 0)
                        & (eligible_counts == all_counts)
                    )
                )
            )[0]
            total_pure_positive += int(pure_positive.numel())
            total_pure_negative += int(pure_negative.numel())
            total_mixed_or_unknown += int(mixed_or_unknown.numel())

            proposal_row: dict[str, int | str | bool] = {
                "proposal_index": proposal_index,
                "pair_budget": pair_budget,
                "member_point_count": int(proposal_members.numel()),
                "pure_positive_token_count": int(pure_positive.numel()),
                "pure_negative_token_count": int(pure_negative.numel()),
                "mixed_or_unknown_token_count": int(mixed_or_unknown.numel()),
                "occupied_token_count": int((all_counts > 0).sum()),
                "retained": False,
                "drop_reason": "",
            }
            if pure_positive.numel() < 2:
                dropped_insufficient_positive += 1
                proposal_row["drop_reason"] = "insufficient_pure_positive"
                proposal_diagnostics.append(proposal_row)
                continue
            if pure_negative.numel() == 0:
                dropped_no_negative += 1
                proposal_row["drop_reason"] = "no_pure_negative"
                proposal_diagnostics.append(proposal_row)
                continue

            positive_a_local = torch.randint(
                int(pure_positive.numel()),
                (pair_budget,),
                generator=generator,
                device=point_to_level.device,
            )
            positive_offset = torch.randint(
                int(pure_positive.numel()) - 1,
                (pair_budget,),
                generator=generator,
                device=point_to_level.device,
            )
            positive_b_local = (
                positive_a_local + 1 + positive_offset
            ) % int(pure_positive.numel())
            positive_pairs = torch.stack(
                [
                    pure_positive[positive_a_local],
                    pure_positive[positive_b_local],
                ],
                dim=1,
            )

            uniform = point_to_level.new_empty((pair_budget, 0))
            if requested_uniform > 0:
                uniform_draw = torch.randint(
                    int(pure_negative.numel()),
                    (pair_budget, requested_uniform),
                    generator=generator,
                    device=point_to_level.device,
                )
                uniform = pure_negative[uniform_draw]

            spatial = point_to_level.new_empty((pair_budget, 0))
            if requested_spatial > 0:
                spatial_pool_count = min(
                    requested_spatial_pool, int(pure_negative.numel())
                )
                spatial_pool = pure_negative[
                    torch.randperm(
                        int(pure_negative.numel()),
                        generator=generator,
                        device=point_to_level.device,
                    )[:spatial_pool_count]
                ]
                spatial_count = min(requested_spatial, spatial_pool_count)
                distances = torch.linalg.vector_norm(
                    level_xyz[spatial_pool][None, :, :]
                    - level_xyz[positive_pairs[:, 0]][:, None, :],
                    dim=-1,
                )
                nearest = torch.topk(
                    distances,
                    k=spatial_count,
                    dim=1,
                    largest=False,
                ).indices
                spatial = spatial_pool[nearest]

            candidates = point_to_level.new_empty((pair_budget, 0))
            if requested_feature_hard > 0:
                candidate_count = min(
                    requested_feature_pool, int(pure_negative.numel())
                )
                candidate_pool = pure_negative[
                    torch.randperm(
                        int(pure_negative.numel()),
                        generator=generator,
                        device=point_to_level.device,
                    )[:candidate_count]
                ]
                candidates = candidate_pool[None, :].expand(pair_budget, -1)

            proposal_label = batch.proposal_labels[pair_rows[0]].expand(
                pair_budget
            )
            records.append(
                {
                    "positive_pairs": positive_pairs,
                    "uniform": uniform,
                    "spatial": spatial,
                    "candidates": candidates,
                    "proposal_labels": proposal_label,
                }
            )
            proposal_row["retained"] = True
            proposal_diagnostics.append(proposal_row)
            retained_pair_budget += pair_budget

    base_stats: dict[str, int | float] = {
        "input_pairs": int(batch.num_pairs),
        "input_proposals": num_proposals,
        "valid_pairs": retained_pair_budget,
        "kept_pairs": retained_pair_budget,
        "kept_proposals": len(records),
        "dropped_proposals_insufficient_pure_positive": (
            dropped_insufficient_positive
        ),
        "dropped_proposals_no_pure_negative": dropped_no_negative,
        "pure_positive_tokens": total_pure_positive,
        "pure_negative_tokens": total_pure_negative,
        "mixed_or_unknown_tokens_ignored": total_mixed_or_unknown,
        "accepted_mixed_tokens": 0,
        "collapsed_positive_pairs": 0,
        "zero_negative_pairs": input_pair_budget - retained_pair_budget,
        "base_negative_collisions": 0,
        "base_negative_duplicates": -1,
        "feature_candidate_collisions": 0,
        "feature_candidate_duplicates": -1,
        "strict_token_purity": 1.0,
        "negative_diversity_measured": 0,
    }
    if include_proposal_diagnostics:
        base_stats["proposal_diagnostics"] = proposal_diagnostics
    if not records:
        base_stats.update(
            {
                "base_negatives_per_pair": 0,
                "uniform_negatives_per_pair": 0,
                "spatial_hard_negatives_per_pair": 0,
                "feature_candidate_pool": 0,
                "feature_hard_negatives": 0,
            }
        )
        return None, base_stats

    spatial_count = min(int(record["spatial"].shape[1]) for record in records)
    candidate_pool = min(
        int(record["candidates"].shape[1]) for record in records
    )
    feature_hard = min(requested_feature_hard, candidate_pool)
    negative_indices = torch.cat(
        [
            torch.cat(
                [
                    record["uniform"],
                    record["spatial"][:, :spatial_count],
                ],
                dim=1,
            )
            for record in records
        ],
        dim=0,
    )
    remapped = ContrastiveTripletBatch(
        positive_pairs=torch.cat(
            [record["positive_pairs"] for record in records], dim=0
        ),
        negative_indices=negative_indices,
        proposal_labels=torch.cat(
            [record["proposal_labels"] for record in records], dim=0
        ),
        feature_candidate_indices=(
            torch.cat(
                [
                    record["candidates"][:, :candidate_pool]
                    for record in records
                ],
                dim=0,
            )
            if feature_hard > 0
            else None
        ),
        num_feature_hard_negatives=feature_hard,
    )
    if include_proposal_diagnostics:
        sorted_base = torch.sort(negative_indices, dim=1).values
        base_unique = 1 + (
            sorted_base[:, 1:] != sorted_base[:, :-1]
        ).sum(dim=1)
        base_duplicates = int(
            (negative_indices.shape[1] - base_unique).sum()
        )
        base_stats.update(
            {
                "base_negative_duplicates": base_duplicates,
                "base_negative_duplicate_fraction": (
                    base_duplicates / float(negative_indices.numel())
                ),
                "base_negative_unique_per_pair_min": int(base_unique.min()),
                "base_negative_unique_per_pair_mean": float(
                    base_unique.float().mean()
                ),
                "negative_diversity_measured": 1,
            }
        )
        if feature_hard > 0:
            assert remapped.feature_candidate_indices is not None
            feature_candidates = remapped.feature_candidate_indices
            sorted_candidates = torch.sort(
                feature_candidates, dim=1
            ).values
            candidate_unique = 1 + (
                sorted_candidates[:, 1:] != sorted_candidates[:, :-1]
            ).sum(dim=1)
            candidate_duplicates = int(
                (feature_candidates.shape[1] - candidate_unique).sum()
            )
            combined = torch.cat(
                [negative_indices, feature_candidates], dim=1
            )
            sorted_combined = torch.sort(combined, dim=1).values
            combined_unique = 1 + (
                sorted_combined[:, 1:] != sorted_combined[:, :-1]
            ).sum(dim=1)
            overlap = base_unique + candidate_unique - combined_unique
            base_stats.update(
                {
                    "feature_candidate_duplicates": candidate_duplicates,
                    "feature_candidate_unique_per_pair_min": int(
                        candidate_unique.min()
                    ),
                    "feature_candidate_unique_per_pair_mean": float(
                        candidate_unique.float().mean()
                    ),
                    "feature_candidate_base_overlap_unique_per_pair_mean": float(
                        overlap.float().mean()
                    ),
                    "feature_candidate_base_overlap_unique_per_pair_max": int(
                        overlap.max()
                    ),
                }
            )
    base_stats.update(
        {
            "base_negatives_per_pair": int(negative_indices.shape[1]),
            "uniform_negatives_per_pair": requested_uniform,
            "spatial_hard_negatives_per_pair": spatial_count,
            "feature_candidate_pool": candidate_pool,
            "feature_hard_negatives": feature_hard,
        }
    )
    return remapped, base_stats


def normalize_hierarchy_frame_groups(
    *,
    batch: ContrastiveTripletBatch | None,
    source_cell: str | None,
    frame_groups: Sequence[Any] | None,
) -> list[MaskSupervisionGroup]:
    """Normalize legacy, tuple, and sampler-v2 frame-group inputs.

    Sampler-v2 objects are consumed structurally to avoid coupling this module
    to the sampler implementation: only ``.cell``, ``.batch``, and optional
    ``.frame_id`` are required.
    """

    if frame_groups is None:
        if batch is None:
            raise ValueError("Either batch or frame_groups must be provided")
        if not isinstance(batch, ContrastiveTripletBatch):
            raise TypeError("batch must be a ContrastiveTripletBatch")
        return [
            MaskSupervisionGroup(
                cell=str(source_cell) if source_cell is not None else "legacy",
                batch=batch,
                group_id=None,
            )
        ]
    if batch is not None:
        raise ValueError("Pass batch or frame_groups, not both")
    if source_cell is not None:
        raise ValueError("source_cell belongs to the single-batch API")
    normalized: list[MaskSupervisionGroup] = []
    for index, group in enumerate(frame_groups):
        if hasattr(group, "cell") and hasattr(group, "batch"):
            cell = str(group.cell)
            group_batch = group.batch
            group_id_value = getattr(group, "frame_id", None)
            if group_id_value is None:
                group_id_value = getattr(group, "group_id", None)
        elif isinstance(group, (tuple, list)) and len(group) in (2, 3):
            if len(group) == 2:
                cell, group_batch = group
                group_id_value = None
            else:
                cell, group_id_value, group_batch = group
            cell = str(cell)
        else:
            raise TypeError(
                "frame_groups entries must expose .cell/.batch or be "
                "(cell, batch)/(cell, group_id, batch) tuples"
            )
        if not isinstance(group_batch, ContrastiveTripletBatch):
            raise TypeError(
                f"frame_groups[{index}] does not contain a ContrastiveTripletBatch"
            )
        normalized.append(
            MaskSupervisionGroup(
                cell=cell,
                batch=group_batch,
                group_id=(
                    None if group_id_value is None else str(group_id_value)
                ),
            )
        )
    if not normalized:
        raise ValueError("frame_groups must not be empty")
    return normalized


def _cell_balanced_contrastive_results(
    records: Sequence[tuple[MaskSupervisionGroup, dict[str, Any]]],
) -> dict[str, Any]:
    """Pair-weight frame losses within cell, then equally average cells."""

    if not records:
        raise ValueError("At least one contrastive result is required")
    cells = tuple(dict.fromkeys(group.cell for group, _ in records))
    by_cell = {
        cell: [result for group, result in records if group.cell == cell]
        for cell in cells
    }
    cell_pair_counts = {
        cell: sum(int(result["num_positive_pairs"]) for result in results)
        for cell, results in by_cell.items()
    }
    if any(count <= 0 for count in cell_pair_counts.values()):
        raise ValueError("Every contrastive frame group must realize positive pairs")
    cell_loss_tensors = {}
    for cell, results in by_cell.items():
        denominator = float(cell_pair_counts[cell])
        zero = results[0]["loss_total"].new_zeros(())
        cell_loss_tensors[cell] = sum(
            (
                result["loss_total"] * float(result["num_positive_pairs"])
                for result in results
            ),
            zero,
        ) / denominator
    loss = torch.stack(list(cell_loss_tensors.values())).mean()
    average_keys = (
        "temperature",
        "positive_cosine_mean",
        "negative_cosine_mean",
        "cosine_gap",
        "triplet_ranking_accuracy",
        "hardest_negative_ranking_accuracy",
        "num_negatives_per_pair",
        "num_base_negatives_per_pair",
        "num_feature_hard_negatives_per_pair",
    )
    output: dict[str, Any] = {
        "loss_total": loss,
        "loss_contrastive": float(loss.detach()),
        "num_positive_pairs": sum(
            int(result["num_positive_pairs"]) for _, result in records
        ),
    }
    cells_metrics: dict[str, dict[str, Any]] = {}
    for cell, results in by_cell.items():
        cell_metrics: dict[str, Any] = {
            "group_count": len(results),
            "loss": float(cell_loss_tensors[cell].detach()),
            "num_positive_pairs": cell_pair_counts[cell],
            "loss_denominator_positive_pairs": cell_pair_counts[cell],
        }
        for key in average_keys:
            cell_metrics[key] = sum(
                float(result[key]) * int(result["num_positive_pairs"])
                for result in results
            ) / float(cell_pair_counts[cell])
        cells_metrics[cell] = cell_metrics
    for key in average_keys:
        output[key] = sum(float(cells_metrics[cell][key]) for cell in cells) / len(
            cells
        )
    output["frame_group_metrics"] = {
        "normalization": "pair_mean_within_cell_then_equal_mean_across_nonempty_cells",
        "group_count": len(records),
        "cell_count": len(cells),
        "cell_equalization_denominator": len(cells),
        "cells": cells_metrics,
        "zero_denominator": False,
    }
    return output


def _zero_contrastive_result(
    reference: torch.Tensor,
    criterion: nn.Module,
) -> dict[str, Any]:
    """Return the normal contrastive schema with a graph-connected zero loss.

    A V2 scene can legitimately lose every point-space pair after exact dec0
    token-purity and co-membership filtering.  That is an optimization-validity
    decision for the synchronized runner, not a rank-local exception.  Touching
    ``reference`` keeps the backbone/projector in the graph, while touching the
    criterion keeps its private learnable temperature safe with
    ``find_unused_parameters=False``.
    """

    loss = zero_loss_touching_module_parameters(reference, criterion)
    raw_temperature = getattr(criterion, "temperature", 0.0)
    if isinstance(raw_temperature, torch.Tensor):
        temperature = float(raw_temperature.detach())
    else:
        temperature = float(raw_temperature)
    return {
        "loss_total": loss,
        "loss_contrastive": 0.0,
        "temperature": temperature,
        "positive_cosine_mean": 0.0,
        "negative_cosine_mean": 0.0,
        "cosine_gap": 0.0,
        "triplet_ranking_accuracy": 0.0,
        "hardest_negative_ranking_accuracy": 0.0,
        "num_positive_pairs": 0,
        "num_negatives_per_pair": 0,
        "num_base_negatives_per_pair": 0,
        "num_feature_hard_negatives_per_pair": 0,
        "frame_group_metrics": {
            "normalization": (
                "pair_mean_within_cell_then_equal_mean_across_nonempty_cells"
            ),
            "group_count": 0,
            "cell_count": 0,
            "cell_equalization_denominator": 0,
            "cells": {},
            "zero_denominator": True,
        },
    }


def _token_pure_dec0_optimization_metrics(
    *,
    groups: Sequence[MaskSupervisionGroup],
    token_records: Sequence[tuple[MaskSupervisionGroup, Mapping[str, Any]]],
    group_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an exact, runner-facing V2 validity contract.

    Counts here are measured *after* point-to-dec0 remapping.  They must not be
    confused with the sampler's point-space pair yield.  Missing expected cells
    remain explicit zero-denominator rows so aggregation cannot silently erase
    a starved granularity.
    """

    expected_cells = tuple(TWO_D_CELLS)
    input_cells = tuple(dict.fromkeys(group.cell for group in groups))
    retained_cells = tuple(dict.fromkeys(group.cell for group, _ in token_records))
    unexpected_input_cells = tuple(
        cell for cell in input_cells if cell not in expected_cells
    )
    report_cells = expected_cells + tuple(
        cell for cell in unexpected_input_cells if cell not in expected_cells
    )
    cells: dict[str, dict[str, Any]] = {}
    for cell in report_cells:
        retained = [
            result for group, result in token_records if group.cell == cell
        ]
        denominator = sum(
            int(result["num_positive_pairs"]) for result in retained
        )
        cells[cell] = {
            "input_group_count": sum(group.cell == cell for group in groups),
            "retained_group_count": len(retained),
            "realized_positive_pairs": denominator,
            "loss_denominator_positive_pairs": denominator,
            "zero_denominator": denominator == 0,
            "optimization_valid": denominator > 0,
        }

    missing_input_cells = [cell for cell in expected_cells if cell not in input_cells]
    missing_retained_cells = [
        cell for cell in expected_cells if cells[cell]["zero_denominator"]
    ]
    exact_input_cell_set = (
        set(input_cells) == set(expected_cells) and not unexpected_input_cells
    )
    exact_retained_cell_set = (
        set(retained_cells) == set(expected_cells)
        and not any(cell not in expected_cells for cell in retained_cells)
    )
    expected_three_cell_valid = bool(
        exact_input_cell_set
        and exact_retained_cell_set
        and all(cells[cell]["optimization_valid"] for cell in expected_cells)
    )
    realized_by_cell = {
        cell: int(cells[cell]["realized_positive_pairs"])
        for cell in report_cells
    }
    denominators_by_cell = {
        cell: int(cells[cell]["loss_denominator_positive_pairs"])
        for cell in report_cells
    }
    return {
        "schema_version": "litept_token_pure_dec0_optimization/v2",
        "enabled": True,
        "optimization_valid": expected_three_cell_valid,
        "primary_loss_valid": bool(token_records),
        "expected_three_cell_valid": expected_three_cell_valid,
        "expected_input_cell_set_valid": exact_input_cell_set,
        "expected_retained_cell_set_valid": exact_retained_cell_set,
        "normalization": (
            "pair_mean_within_cell_then_equal_mean_nonempty_cells"
        ),
        "expected_cells": list(expected_cells),
        "input_cells": list(input_cells),
        "retained_cells": list(retained_cells),
        "missing_input_cells": missing_input_cells,
        "missing_retained_cells": missing_retained_cells,
        "unexpected_input_cells": list(unexpected_input_cells),
        "empty_cells": [
            cell for cell in report_cells if cells[cell]["zero_denominator"]
        ],
        "cell_equalization_denominator": len(retained_cells),
        "input_group_count": len(groups),
        "retained_group_count": len(token_records),
        "realized_positive_pairs": sum(realized_by_cell.values()),
        "loss_denominator_positive_pairs": sum(denominators_by_cell.values()),
        "realized_positive_pairs_by_cell": realized_by_cell,
        "loss_denominators_by_cell": denominators_by_cell,
        "zero_denominator": not bool(token_records),
        "cells": cells,
        "groups": [dict(metrics) for metrics in group_metrics],
    }


class LitePTContrastiveModel(nn.Module):
    def __init__(
        self,
        *,
        litept_root: Path,
        input_features: str = "rgbn6",
        representative_sampling: str = "first",
        token_pure_dec0: bool = False,
        multiscale_supervision: bool = False,
        multiscale_loss_weight: float = MULTISCALE_AUXILIARY_WEIGHT,
        multiscale_warmup_epochs: int = MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
        hierarchy_supervision: bool = False,
        hierarchy_mask_loss_weight: float = HIERARCHY_MASK_LOSS_WEIGHT,
        hierarchy_mask_stage_weights: Mapping[str, float] | None = None,
        hierarchy_mask_stage_weight_schedule: Mapping[str, Any] | None = None,
        hierarchy_query_projection_policy: str = "shared_dec0",
        hierarchy_vicreg_loss_weight: float = HIERARCHY_VICREG_LOSS_WEIGHT,
        hierarchy_warmup_fraction: float = HIERARCHY_WARMUP_FRACTION,
        hierarchy_variance_target: float = HIERARCHY_VARIANCE_TARGET,
        hierarchy_covariance_weight: float = HIERARCHY_COVARIANCE_WEIGHT,
        hierarchy_vicreg_max_tokens: int = HIERARCHY_VICREG_MAX_TOKENS,
    ) -> None:
        super().__init__()
        self.input_features = str(input_features)
        self.input_channels = input_channels_for(self.input_features)
        self.token_pure_dec0 = bool(token_pure_dec0)
        self.multiscale_supervision = bool(multiscale_supervision)
        self.multiscale_loss_weight = float(multiscale_loss_weight)
        self.multiscale_warmup_epochs = int(multiscale_warmup_epochs)
        self.hierarchy_supervision = bool(hierarchy_supervision)
        self.hierarchy_mask_loss_weight = float(hierarchy_mask_loss_weight)
        self.hierarchy_mask_stage_weights = (
            normalize_hierarchy_mask_stage_weights(
                hierarchy_mask_stage_weights
            )
        )
        self.hierarchy_mask_stage_weight_schedule = (
            normalize_hierarchy_mask_stage_weight_schedule(
                hierarchy_mask_stage_weight_schedule
            )
        )
        self.hierarchy_query_projection_policy = str(
            hierarchy_query_projection_policy
        )
        self.hierarchy_vicreg_loss_weight = float(hierarchy_vicreg_loss_weight)
        self.hierarchy_warmup_fraction = float(hierarchy_warmup_fraction)
        self.hierarchy_variance_target = float(hierarchy_variance_target)
        self.hierarchy_covariance_weight = float(hierarchy_covariance_weight)
        self.hierarchy_vicreg_max_tokens = int(hierarchy_vicreg_max_tokens)
        if self.multiscale_loss_weight < 0.0:
            raise ValueError("multiscale_loss_weight must be non-negative")
        if self.multiscale_warmup_epochs < 0:
            raise ValueError("multiscale_warmup_epochs must be non-negative")
        if self.multiscale_supervision and self.hierarchy_supervision:
            raise ValueError(
                "Legacy pure-token multiscale supervision and hierarchy "
                "supervision are mutually exclusive"
            )
        if self.hierarchy_mask_loss_weight < 0.0:
            raise ValueError("hierarchy_mask_loss_weight must be non-negative")
        if (
            self.hierarchy_query_projection_policy
            not in HIERARCHY_QUERY_PROJECTION_POLICIES
        ):
            raise ValueError(
                "Unknown hierarchy_query_projection_policy: "
                f"{self.hierarchy_query_projection_policy!r}"
            )
        if self.hierarchy_vicreg_loss_weight < 0.0:
            raise ValueError("hierarchy_vicreg_loss_weight must be non-negative")
        if not 0.0 <= self.hierarchy_warmup_fraction <= 1.0:
            raise ValueError("hierarchy_warmup_fraction must lie in [0,1]")
        if self.hierarchy_variance_target <= 0.0:
            raise ValueError("hierarchy_variance_target must be positive")
        if self.hierarchy_covariance_weight < 0.0:
            raise ValueError("hierarchy_covariance_weight must be non-negative")
        if self.hierarchy_vicreg_max_tokens < 2:
            raise ValueError("hierarchy_vicreg_max_tokens must be at least two")
        self.backbone = LitePTBackbone(
            litept_root=str(litept_root),
            in_channels=self.input_channels,
            grid_size=0.02,
            litept_variant="litept_s_star",
            multi_scale=self.multiscale_supervision,
            multi_scale_indices=(
                list(MULTISCALE_LEVEL_INDICES)
                if self.multiscale_supervision
                else None
            ),
            # Transfer contract shared with ScanNet LitePT -> Mask3D. This
            # deliberately replaces the wrapper's legacy mean reduction.
            voxel_reduce="representative",
            representative_sampling=representative_sampling,
            # Every V2 arm is judged on the same five native transfer stages.
            # Arm A keeps its legacy decoder-only auxiliary objective, but it
            # must still expose enc4 so final-dec0 and auxiliary gradient flow
            # through the shared backbone can be measured rather than marked
            # permanently unavailable by the evaluator.
            capture_hierarchy=(
                self.hierarchy_supervision or self.multiscale_supervision
            ),
            # Each update can use a different ScanNet scene. The wrapper's
            # legacy cache is only valid for repeated optimization of one scene.
            cache_training_voxelization=False,
        )
        self.projector = ContrastiveProjectionHead(72, 128, 128)
        self.criterion = PartFieldContrastiveCriterion(
            temperature=0.07,
            learnable_temperature=True,
            symmetric_feature_hard_mining=self.token_pure_dec0,
        )
        self.multiscale_projectors = nn.ModuleDict()
        self.multiscale_criteria = nn.ModuleDict()
        self.hierarchy_mask_supervisor: ProposalConditionedHierarchyLoss | None = None
        if self.multiscale_supervision:
            if set(MULTISCALE_LEVEL_LOSS_WEIGHTS) != set(MULTISCALE_LEVEL_NAMES):
                raise RuntimeError("Multi-scale stage-weight names drifted")
            if not math.isclose(
                sum(MULTISCALE_LEVEL_LOSS_WEIGHTS.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError("Multi-scale stage weights must sum to one")
            level_channels = self.backbone.multi_scale_channels
            expected_channels = (252, 144, 72)
            if level_channels != list(expected_channels):
                raise RuntimeError(
                    "Unexpected LitePT-S* multi-scale channels: "
                    f"{level_channels!r} != {list(expected_channels)!r}"
                )
            self.multiscale_projectors.update(
                {
                    name: ContrastiveProjectionHead(channels, 128, 128)
                    for name, channels in zip(
                        MULTISCALE_LEVEL_NAMES,
                        expected_channels,
                        strict=True,
                    )
                }
            )
            # Keep dec0's learnable temperature exactly isolated from the
            # auxiliary distributions.  These level-specific temperatures,
            # like their projection heads, are disposable at transfer time.
            self.multiscale_criteria.update(
                {
                    name: PartFieldContrastiveCriterion(
                        temperature=0.07,
                        learnable_temperature=True,
                        symmetric_feature_hard_mining=True,
                    )
                    for name in MULTISCALE_LEVEL_NAMES
                }
            )
        if self.hierarchy_supervision:
            self.hierarchy_mask_supervisor = ProposalConditionedHierarchyLoss(
                stage_channels=HIERARCHY_STAGE_CHANNELS,
                embedding_dim=128,
                temperature=0.07,
                query_projection_policy=(
                    self.hierarchy_query_projection_policy
                ),
            )

    @property
    def recipe_name(self) -> str:
        if self.hierarchy_supervision:
            return HIERARCHY_RECIPE_NAME
        if self.multiscale_supervision:
            return MULTISCALE_RECIPE_NAME
        return "final_dec0_only_v1"

    def multiscale_scale_for_epoch(self, epoch: int) -> float:
        """Return the scheduled auxiliary-loss multiplier for a scene visit."""
        if not self.multiscale_supervision:
            return 0.0
        if int(epoch) < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if self.multiscale_warmup_epochs == 0:
            ramp = 1.0
        else:
            ramp = min(1.0, float(epoch) / float(self.multiscale_warmup_epochs))
        return self.multiscale_loss_weight * ramp

    def hierarchy_ramp_for_step(
        self,
        *,
        supervision_step: int | None,
        supervision_total_steps: int | None,
        hierarchy_ramp_fraction: float | None,
    ) -> float:
        """Resolve an explicit ramp or a step-based first-5%-of-run ramp."""

        if not self.hierarchy_supervision:
            return 0.0
        if hierarchy_ramp_fraction is not None:
            value = float(hierarchy_ramp_fraction)
            if not 0.0 <= value <= 1.0:
                raise ValueError("hierarchy_ramp_fraction must lie in [0,1]")
            return value
        if supervision_step is None and supervision_total_steps is None:
            return 1.0
        if supervision_step is None or supervision_total_steps is None:
            raise ValueError(
                "supervision_step and supervision_total_steps must be supplied together"
            )
        step = int(supervision_step)
        total = int(supervision_total_steps)
        if step < 0 or total <= 0 or step > total:
            raise ValueError(
                f"Invalid hierarchy supervision progress step={step}, total={total}"
            )
        if self.hierarchy_warmup_fraction == 0.0:
            return 1.0
        ramp_steps = max(1, int(math.ceil(total * self.hierarchy_warmup_fraction)))
        return min(1.0, float(step) / float(ramp_steps))

    @staticmethod
    def _hierarchy_stage_features(
        output: LitePTBackboneOutput,
    ) -> dict[str, torch.Tensor]:
        expected = tuple(HIERARCHY_COARSE_TO_FINE)
        if output.hierarchy_stage_names != expected:
            raise RuntimeError(
                "LitePT hierarchy stage order drift: "
                f"{output.hierarchy_stage_names!r} != {expected!r}"
            )
        if len(output.hierarchy_tokens) != len(expected):
            raise RuntimeError("LitePT hierarchy token count drift")
        if len(output.hierarchy_parent_maps) != len(expected) - 1:
            raise RuntimeError("LitePT hierarchy parent-map count drift")
        stage_features = dict(
            zip(expected, output.hierarchy_tokens, strict=True)
        )
        for stage, expected_channels in HIERARCHY_STAGE_CHANNELS.items():
            features = stage_features[stage]
            if features.ndim != 2 or int(features.shape[1]) != expected_channels:
                raise RuntimeError(
                    f"LitePT hierarchy {stage} shape drift: {tuple(features.shape)}"
                )
        return stage_features

    def _encode_backbone(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> LitePTBackboneOutput:
        output = self.backbone(points, features)
        if not isinstance(output, LitePTBackboneOutput):
            raise TypeError("Contrastive pretraining requires a dense LitePT output")
        return output

    def encode_with_multiscale(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        *,
        token_space: bool = False,
    ) -> tuple[
        LitePTBackboneOutput,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Encode the final feature and optional disposable level projections."""
        backbone_output = self._encode_backbone(points, features)
        native = backbone_output.point_feat
        if native.shape[0] != points.shape[0]:
            raise RuntimeError(
                "LitePT per-point feature cardinality mismatch: "
                f"input_points={points.shape[0]} output_features={native.shape[0]}"
            )
        projector_input = (
            backbone_output.scene_tokens if token_space else native
        )
        projected = self.projector(projector_input)
        level_embeddings: dict[str, torch.Tensor] = {}
        if self.multiscale_supervision:
            if len(backbone_output.multi_scale_tokens) != len(
                MULTISCALE_LEVEL_NAMES
            ):
                raise RuntimeError("LitePT multi-scale output count drift")
            level_embeddings = {
                name: self.multiscale_projectors[name](tokens)
                for name, tokens in zip(
                    MULTISCALE_LEVEL_NAMES,
                    backbone_output.multi_scale_tokens,
                    strict=True,
                )
            }
        return backbone_output, native, projected, level_embeddings

    def encode(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, native, projected, _ = self.encode_with_multiscale(points, features)
        return native, projected

    def _legacy_multiscale_for_batch(
        self,
        *,
        backbone_output: LitePTBackboneOutput,
        level_embeddings_by_name: dict[str, torch.Tensor],
        batch: ContrastiveTripletBatch,
        return_proposal_diagnostics: bool,
    ) -> dict[str, Any]:
        """Evaluate the pre-existing pure-token objective for one group."""

        weighted_losses: list[torch.Tensor] = []
        unweighted_losses: dict[str, torch.Tensor] = {}
        stages: dict[str, dict[str, Any]] = {}
        for name, tokens, level_xyz, parent_map in zip(
            MULTISCALE_LEVEL_NAMES,
            backbone_output.multi_scale_tokens,
            backbone_output.multi_scale_xyz,
            backbone_output.finest_to_multi_scale,
            strict=True,
        ):
            level_embeddings = level_embeddings_by_name[name].float()
            point_to_level = parent_map[backbone_output.inverse_map.long()]
            level_batch, mapping_stats = build_pure_token_contrastive_batch(
                batch,
                point_to_level,
                level_xyz,
                level_name=name,
                include_proposal_diagnostics=return_proposal_diagnostics,
            )
            if level_batch is None:
                level_loss = zero_loss_touching_module_parameters(
                    level_embeddings, self.multiscale_criteria[name]
                )
                level_metrics = {
                    "loss": 0.0,
                    "hardest_negative_ranking_accuracy": 0.0,
                    "cosine_gap": 0.0,
                    "temperature": float(
                        self.multiscale_criteria[name].temperature.detach()
                    ),
                }
            else:
                level_result = self.multiscale_criteria[name](
                    level_embeddings, level_batch
                )
                level_loss = level_result["loss_total"]
                level_metrics = {
                    "loss": float(level_loss.detach()),
                    "hardest_negative_ranking_accuracy": float(
                        level_result["hardest_negative_ranking_accuracy"]
                    ),
                    "cosine_gap": float(level_result["cosine_gap"]),
                    "temperature": float(level_result["temperature"]),
                }
            stage_weight = float(MULTISCALE_LEVEL_LOSS_WEIGHTS[name])
            unweighted_losses[name] = level_loss
            weighted_losses.append(stage_weight * level_loss)
            stages[name] = {
                **level_metrics,
                **mapping_stats,
                "token_count": int(tokens.shape[0]),
                "stage_weight": stage_weight,
            }
        return {
            "loss_total": torch.stack(weighted_losses).sum(),
            "stage_loss_tensors": unweighted_losses,
            "stages": stages,
        }

    @staticmethod
    def _cell_balanced_legacy_multiscale(
        records: Sequence[tuple[MaskSupervisionGroup, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Valid-pair normalize within cells, then equally weight nonempty cells."""

        cells = tuple(dict.fromkeys(group.cell for group, _ in records))
        by_cell = {
            cell: [result for group, result in records if group.cell == cell]
            for cell in cells
        }

        reference = records[0][1]["loss_total"]
        stage_cell_losses: dict[str, dict[str, torch.Tensor]] = {
            stage: {} for stage in MULTISCALE_LEVEL_NAMES
        }
        stage_cell_denominators: dict[str, dict[str, int]] = {
            stage: {} for stage in MULTISCALE_LEVEL_NAMES
        }
        active_cells: list[str] = []
        for cell in cells:
            cell_has_pairs = False
            for stage in MULTISCALE_LEVEL_NAMES:
                denominator = sum(
                    int(result["stages"][stage]["kept_pairs"])
                    for result in by_cell[cell]
                )
                numerator = sum(
                    (
                        result["stage_loss_tensors"][stage]
                        * float(result["stages"][stage]["kept_pairs"])
                        for result in by_cell[cell]
                    ),
                    reference.new_zeros(()),
                )
                stage_cell_denominators[stage][cell] = denominator
                stage_cell_losses[stage][cell] = numerator / max(denominator, 1)
                cell_has_pairs = cell_has_pairs or denominator > 0
            if cell_has_pairs:
                active_cells.append(cell)
        if not active_cells:
            stage_tensors = {
                stage: reference * 0.0 for stage in MULTISCALE_LEVEL_NAMES
            }
        else:
            stage_tensors = {
                stage: torch.stack(
                    [stage_cell_losses[stage][cell] for cell in active_cells]
                ).mean()
                for stage in MULTISCALE_LEVEL_NAMES
            }
        stages: dict[str, dict[str, Any]] = {}
        sum_keys = {
            "input_pairs",
            "input_proposals",
            "valid_pairs",
            "kept_pairs",
            "kept_proposals",
            "dropped_proposals_insufficient_pure_positive",
            "dropped_proposals_no_pure_negative",
            "pure_positive_tokens",
            "pure_negative_tokens",
            "mixed_or_unknown_tokens_ignored",
            "zero_negative_pairs",
        }
        for stage in MULTISCALE_LEVEL_NAMES:
            stage_records = [
                record["stages"][stage] for _, record in records
            ]
            summary = dict(stage_records[0])
            for key, value in tuple(summary.items()):
                if key == "proposal_diagnostics":
                    summary.pop(key)
                elif isinstance(value, (int, float)):
                    if key in sum_keys:
                        summary[key] = sum(
                            float(record.get(key, 0.0)) for record in stage_records
                        )
                    else:
                        summary[key] = sum(
                            float(record.get(key, 0.0)) for record in stage_records
                        ) / len(stage_records)
            summary["loss"] = float(stage_tensors[stage].detach())
            summary["frame_group_count"] = len(records)
            summary["loss_denominator_valid_pairs_by_cell"] = {
                cell: stage_cell_denominators[stage][cell] for cell in cells
            }
            summary["cell_equalization_denominator"] = len(active_cells)
            stages[stage] = summary
        loss_total = sum(
            (
                float(MULTISCALE_LEVEL_LOSS_WEIGHTS[stage])
                * stage_tensors[stage]
                for stage in MULTISCALE_LEVEL_NAMES
            ),
            reference.new_zeros(()),
        )
        return {
            "loss_total": loss_total,
            "stage_loss_tensors": stage_tensors,
            "stages": stages,
            "normalization": (
                "valid_pair_mean_within_cell_then_equal_mean_nonempty_cells"
            ),
            "active_cells": active_cells,
            "empty_cells": [cell for cell in cells if cell not in active_cells],
        }

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        batch: ContrastiveTripletBatch | None = None,
        multiscale_loss_scale: float | None = None,
        return_loss_components: bool = False,
        return_multiscale_proposal_diagnostics: bool = False,
        *,
        source_cell: str | None = None,
        frame_groups: Sequence[Any] | None = None,
        supervision_step: int | None = None,
        supervision_total_steps: int | None = None,
        hierarchy_ramp_fraction: float | None = None,
    ) -> dict[str, Any]:
        groups = normalize_hierarchy_frame_groups(
            batch=batch,
            source_cell=source_cell,
            frame_groups=frame_groups,
        )
        (
            backbone_output,
            native,
            projected,
            level_embeddings_by_name,
        ) = self.encode_with_multiscale(
            points, features, token_space=self.token_pure_dec0
        )
        token_pure_metrics: dict[str, Any] | None = None
        if self.token_pure_dec0:
            token_records: list[
                tuple[MaskSupervisionGroup, dict[str, Any]]
            ] = []
            token_group_metrics: list[dict[str, Any]] = []
            for group in groups:
                token_batch, group_metrics = (
                    build_token_pure_dec0_contrastive_batch(
                        group.batch,
                        backbone_output.inverse_map,
                        backbone_output.scene_xyz,
                        max_positive_pairs_per_proposal=(
                            TOKEN_PURE_DEC0_MAX_PAIRS_PER_PROPOSAL
                        ),
                    )
                )
                token_group_metrics.append(
                    {
                        "cell": group.cell,
                        "group_id": group.group_id,
                        **group_metrics,
                    }
                )
                if token_batch is not None:
                    token_records.append(
                        (
                            group,
                            self.criterion(projected.float(), token_batch),
                        )
                    )
            final_result = (
                _cell_balanced_contrastive_results(token_records)
                if token_records
                else _zero_contrastive_result(projected.float(), self.criterion)
            )
            token_pure_metrics = _token_pure_dec0_optimization_metrics(
                groups=groups,
                token_records=token_records,
                group_metrics=token_group_metrics,
            )
        elif frame_groups is None:
            # Preserve the exact legacy arithmetic and result dictionary for
            # the single-batch path.
            final_result = self.criterion(projected.float(), groups[0].batch)
        else:
            final_result = _cell_balanced_contrastive_results(
                [
                    (group, self.criterion(projected.float(), group.batch))
                    for group in groups
                ]
            )
        result: dict[str, Any] = {
            **final_result,
            "loss_final_dec0": float(final_result["loss_total"].detach()),
            "loss_multiscale": 0.0,
            "multiscale_loss_scale": 0.0,
            "multiscale": {},
        }
        if token_pure_metrics is not None:
            result["token_pure_dec0"] = token_pure_metrics
            result["optimization_valid"] = bool(
                token_pure_metrics["optimization_valid"]
            )
        if self.multiscale_supervision:
            if len(backbone_output.multi_scale_tokens) != len(
                MULTISCALE_LEVEL_NAMES
            ):
                raise RuntimeError(
                    "LitePT multi-scale output count drift: "
                    f"{len(backbone_output.multi_scale_tokens)} != "
                    f"{len(MULTISCALE_LEVEL_NAMES)}"
                )
            if len(backbone_output.finest_to_multi_scale) != len(
                MULTISCALE_LEVEL_NAMES
            ):
                raise RuntimeError("LitePT multi-scale parent-map count drift")

            multiscale_records = [
                (
                    group,
                    self._legacy_multiscale_for_batch(
                        backbone_output=backbone_output,
                        level_embeddings_by_name=level_embeddings_by_name,
                        batch=group.batch,
                        return_proposal_diagnostics=(
                            return_multiscale_proposal_diagnostics
                        ),
                    ),
                )
                for group in groups
            ]
            if frame_groups is None:
                multiscale_result = multiscale_records[0][1]
            else:
                multiscale_result = self._cell_balanced_legacy_multiscale(
                    multiscale_records
                )
            effective_scale = (
                self.multiscale_loss_weight
                if multiscale_loss_scale is None
                else float(multiscale_loss_scale)
            )
            if not 0.0 <= effective_scale <= self.multiscale_loss_weight + 1e-8:
                raise ValueError(
                    "multiscale_loss_scale must lie in "
                    "[0, multiscale_loss_weight], "
                    f"got {effective_scale} with max {self.multiscale_loss_weight}"
                )
            result["multiscale"] = multiscale_result["stages"]
            for stage_metrics in result["multiscale"].values():
                stage_metrics["effective_loss_weight"] = (
                    effective_scale * float(stage_metrics["stage_weight"])
                )
            auxiliary_loss = multiscale_result["loss_total"]
            unweighted_auxiliary_mean = torch.stack(
                [
                    multiscale_result["stage_loss_tensors"][stage]
                    for stage in MULTISCALE_LEVEL_NAMES
                ]
            ).mean()
            result["loss_total"] = final_result["loss_total"] + (
                effective_scale * auxiliary_loss
            )
            result["loss_multiscale"] = float(auxiliary_loss.detach())
            result["loss_multiscale_unweighted_mean"] = float(
                unweighted_auxiliary_mean.detach()
            )
            result["multiscale_loss_scale"] = effective_scale
            result["multiscale_level_loss_weights"] = dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            )
            if frame_groups is not None:
                result["multiscale_normalization"] = multiscale_result[
                    "normalization"
                ]
                result["multiscale_active_cells"] = multiscale_result[
                    "active_cells"
                ]
                result["multiscale_empty_cells"] = multiscale_result[
                    "empty_cells"
                ]
            result["multiscale_valid_pairs"] = int(
                sum(
                    int(level["kept_pairs"])
                    for level in result["multiscale"].values()
                )
            )
            if return_loss_components:
                result["loss_component_tensors"] = {
                    "final_dec0": final_result["loss_total"],
                    "legacy_auxiliary_scaled": effective_scale
                    * multiscale_result["loss_total"],
                    **multiscale_result["stage_loss_tensors"],
                    **{
                        f"legacy_{stage}_scaled": effective_scale
                        * float(MULTISCALE_LEVEL_LOSS_WEIGHTS[stage])
                        * multiscale_result["stage_loss_tensors"][stage]
                        for stage in MULTISCALE_LEVEL_NAMES
                    },
                }
                result["hierarchy_junction_tensors"] = (
                    self._hierarchy_stage_features(backbone_output)
                )
                result["hierarchy_junction_availability"] = {
                    "available": list(HIERARCHY_COARSE_TO_FINE),
                    "unavailable": [],
                }
            return result

        if not self.hierarchy_supervision:
            return result

        stage_features = self._hierarchy_stage_features(backbone_output)
        point_maps = hierarchy_point_maps(
            backbone_output.inverse_map,
            backbone_output.hierarchy_parent_maps,
        )
        assert self.hierarchy_mask_supervisor is not None
        mask_result = self.hierarchy_mask_supervisor(
            stage_features=stage_features,
            point_maps=point_maps,
            groups=groups,
        )
        active_mask_stage_weights, mask_stage_schedule_telemetry = (
            hierarchy_mask_stage_weights_for_step(
                static_weights=self.hierarchy_mask_stage_weights,
                schedule=getattr(
                    self,
                    "hierarchy_mask_stage_weight_schedule",
                    None,
                ),
                supervision_step=supervision_step,
                supervision_total_steps=supervision_total_steps,
            )
        )
        weighted_mask_loss, weighted_mask_stage_tensors = (
            weighted_hierarchy_mask_loss(
                loss_total=mask_result["loss_total"],
                stage_loss_tensors=mask_result["stage_loss_tensors"],
                stage_weights=active_mask_stage_weights,
            )
        )
        vicreg_result = native_hierarchy_vicreg_loss(
            stage_features,
            variance_target=self.hierarchy_variance_target,
            covariance_weight=self.hierarchy_covariance_weight,
            max_tokens=self.hierarchy_vicreg_max_tokens,
        )
        ramp_fraction = self.hierarchy_ramp_for_step(
            supervision_step=supervision_step,
            supervision_total_steps=supervision_total_steps,
            hierarchy_ramp_fraction=hierarchy_ramp_fraction,
        )
        mask_scale = ramp_fraction * self.hierarchy_mask_loss_weight
        vicreg_scale = self.hierarchy_vicreg_loss_weight
        scaled_mask_loss = mask_scale * weighted_mask_loss
        scaled_vicreg_loss = vicreg_scale * vicreg_result["loss_total"]
        result["loss_total"] = (
            final_result["loss_total"]
            + scaled_mask_loss
            + scaled_vicreg_loss
        )

        hierarchy_stages: dict[str, dict[str, Any]] = {}
        for stage in HIERARCHY_COARSE_TO_FINE:
            mask_stage = mask_result["metrics"]["stages"].get(
                stage,
                {
                    "active": False,
                    "route_weight": 0.0,
                    "loss": 0.0,
                },
            )
            vicreg_stage = vicreg_result["metrics"]["stages"][stage]
            stage_weight = float(
                active_mask_stage_weights.get(stage, 1.0)
            )
            hierarchy_stages[stage] = {
                "token_count": int(stage_features[stage].shape[0]),
                "mask": mask_stage,
                "vicreg": vicreg_stage,
                "mask_loss": float(mask_stage.get("loss", 0.0)),
                "mask_weighted_loss": float(
                    weighted_mask_stage_tensors.get(
                        stage,
                        weighted_mask_loss * 0.0,
                    ).detach()
                ),
                "mask_stage_weight": stage_weight,
                "mask_effective_weight": mask_scale
                * stage_weight
                * float(mask_stage.get("effective_weight_realized", 0.0)),
                "vicreg_loss": float(vicreg_stage["loss"]),
                "vicreg_effective_weight": vicreg_scale
                * float(vicreg_stage["stage_weight"]),
            }
        source_cells = list(dict.fromkeys(group.cell for group in groups))
        mask_optimization_valid = bool(
            mask_result["metrics"]["optimization_valid"]
        )
        vicreg_stage_validity = {
            stage: bool(vicreg_result["metrics"]["stages"][stage]["valid"])
            for stage in HIERARCHY_COARSE_TO_FINE
        }
        vicreg_all_stages_valid = all(vicreg_stage_validity.values())
        vicreg_required = self.hierarchy_vicreg_loss_weight > 0.0
        hierarchy_optimization_valid = bool(
            mask_optimization_valid
            and (not vicreg_required or vicreg_all_stages_valid)
        )
        result["hierarchy_supervision"] = {
            "enabled": True,
            "recipe": HIERARCHY_RECIPE_NAME,
            "optimization_valid": hierarchy_optimization_valid,
            "mask_optimization_valid": mask_optimization_valid,
            "expected_three_cell_valid": bool(
                mask_result["metrics"]["expected_three_cell_valid"]
            ),
            "route_stage_valid": bool(
                mask_result["metrics"]["route_stage_valid"]
            ),
            "fold_denominators_by_cell": mask_result["metrics"][
                "fold_denominators_by_cell"
            ],
            "fold_denominators_by_route_stage": mask_result["metrics"][
                "fold_denominators_by_route_stage"
            ],
            "route_stage_validity": mask_result["metrics"][
                "route_stage_validity"
            ],
            "vicreg_required_for_optimization": vicreg_required,
            "vicreg_all_stages_valid": vicreg_all_stages_valid,
            "vicreg_stage_validity": vicreg_stage_validity,
            "ramp_fraction": ramp_fraction,
            "mask_loss_weight": self.hierarchy_mask_loss_weight,
            "mask_stage_weights": dict(active_mask_stage_weights),
            "query_projection_policy": (
                # Small synthetic/checkpoint-compatibility constructions used
                # by diagnostics predate this optional C8 field.  The
                # supervisor itself still defaults to the exact C3--C7 shared
                # policy, so report that default rather than making telemetry
                # introduce a new required attribute outside __init__.
                getattr(self, "hierarchy_query_projection_policy", "shared_dec0")
            ),
            "mask_stage_weight_schedule": {
                "configured": getattr(
                    self,
                    "hierarchy_mask_stage_weight_schedule",
                    None,
                ),
                **mask_stage_schedule_telemetry,
            },
            "mask_effective_loss_weight": mask_scale,
            "vicreg_loss_weight": vicreg_scale,
            "loss_mask": float(weighted_mask_loss.detach()),
            "loss_mask_unweighted": float(mask_result["loss_total"].detach()),
            "loss_mask_scaled": float(scaled_mask_loss.detach()),
            "loss_vicreg": float(vicreg_result["loss_total"].detach()),
            "loss_vicreg_scaled": float(scaled_vicreg_loss.detach()),
            "source_cells": source_cells,
            "frame_group_count": len(groups),
            "normalization": (
                "dec0_pair_mean_within_cell_equal_across_nonempty_cells;"
                "mask_realized_fold_mean_within_cell_equal_across_nonempty_cells"
            ),
            "stages": hierarchy_stages,
            "mask": mask_result["metrics"],
            "vicreg": vicreg_result["metrics"],
        }
        if token_pure_metrics is not None:
            result["optimization_valid"] = bool(
                token_pure_metrics["optimization_valid"]
                and hierarchy_optimization_valid
            )
        result["loss_hierarchy_mask"] = float(weighted_mask_loss.detach())
        result["loss_hierarchy_vicreg"] = float(
            vicreg_result["loss_total"].detach()
        )
        if return_loss_components:
            result["loss_component_tensors"] = {
                "final_dec0": final_result["loss_total"],
                "hierarchy_mask": weighted_mask_loss,
                "hierarchy_mask_scaled": scaled_mask_loss,
                "hierarchy_vicreg": vicreg_result["loss_total"],
                "hierarchy_vicreg_scaled": scaled_vicreg_loss,
                **{
                    f"hierarchy_mask_{stage}": loss
                    for stage, loss in mask_result[
                        "stage_loss_tensors"
                    ].items()
                },
                **{
                    f"hierarchy_mask_{stage}_weighted": loss
                    for stage, loss in weighted_mask_stage_tensors.items()
                },
                **{
                    f"hierarchy_vicreg_{stage}": loss
                    for stage, loss in vicreg_result[
                        "stage_loss_tensors"
                    ].items()
                },
            }
            result["hierarchy_junction_tensors"] = stage_features
        return result


def _resolve_artifact(manifest_path: Path, record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(strict=True)


def _load_clean_multicrop_scene_source(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any],
    input_features: str,
    coordinate_normalization_policy: str,
    verify_hashes: bool,
    color_mean: Sequence[float] | None,
    color_std: Sequence[float] | None,
) -> MultiCropSceneSource:
    """Load a hierarchy-free clean-pack adapter manifest fail-closed."""

    if input_features != "rgbn6":
        raise ValueError("clean-pack V2 sources require RGBN6")
    if (
        coordinate_normalization_policy
        != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
    ):
        raise ValueError(
            "clean-pack V2 sources require the canonical Structured3D-to-ScanNet "
            "coordinate policy"
        )
    required_checks = {
        "rgbn6_all_crops",
        "three_2d_cells_all_crops",
        "known_support_preserved",
        "no_hierarchy_or_gt",
        "single_variant_only",
        "artifact_hashes_recorded",
    }
    checks = manifest.get("checks")
    source_kind = str(manifest.get("source_kind", ""))
    if (
        manifest.get("dataset") != "structured3d"
        or source_kind not in {"current_pack", "clean_pack"}
        or manifest.get("source_cells") != list(TWO_D_CELLS)
        or manifest.get("crop_mode") != "sphere_50k"
        or manifest.get("crop_rotation_policy")
        != "deterministic_epoch_rotation_v1"
        or manifest.get("coordinate_normalization_policy")
        != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or not all(bool(checks[key]) for key in required_checks)
    ):
        raise ValueError(f"{manifest_path}: clean training-source contract drift")
    scene_id = str(manifest.get("scene_id", ""))
    if not scene_id.startswith("scene_"):
        raise ValueError(f"{manifest_path}: invalid Structured3D scene ID")
    crop_records = manifest.get("crops")
    if not isinstance(crop_records, list) or not 1 <= len(crop_records) <= 4:
        raise ValueError(f"{manifest_path}: expected one to four clean crops")

    def artifact(record: Mapping[str, Any], key: str) -> Path:
        value = record.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"{manifest_path}: crop artifact {key!r} is missing")
        path = Path(str(value.get("path", "")))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{manifest_path}: non-portable crop artifact {key!r}")
        resolved = (manifest_path.parent / path).resolve(strict=True)
        if int(value.get("size_bytes", -1)) != resolved.stat().st_size:
            raise ValueError(f"{manifest_path}: crop artifact size drift for {key}")
        if verify_hashes and str(value.get("sha256", "")) != sha256_file(resolved):
            raise ValueError(f"{manifest_path}: crop artifact hash drift for {key}")
        return resolved

    crops: list[SceneSource] = []
    seen_crop_ids: set[str] = set()
    variant_id = str(manifest.get("variant", {}).get("variant_id", ""))
    if not variant_id:
        raise ValueError(f"{manifest_path}: clean variant ID is missing")
    for crop_record in crop_records:
        if not isinstance(crop_record, Mapping):
            raise ValueError(f"{manifest_path}: invalid clean crop record")
        crop_id = str(crop_record.get("crop_id", ""))
        if not crop_id or crop_id in seen_crop_ids:
            raise ValueError(f"{manifest_path}: duplicate/empty clean crop ID")
        seen_crop_ids.add(crop_id)
        crop_manifest_path = artifact(crop_record, "crop_manifest")
        crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
        if (
            crop_manifest.get("schema_version") != CLEAN_CROP_PACK_SCHEMA
            or crop_manifest.get("scene_id") != scene_id
            or crop_manifest.get("crop_id") != crop_id
            or crop_manifest.get("variant") != manifest.get("variant")
            or crop_manifest.get("labels") is not None
            or crop_manifest.get("features", {}).get("channels") != "rgbn6"
        ):
            raise ValueError(f"{crop_manifest_path}: clean crop manifest drift")
        points_path = artifact(crop_record, "points")
        colors_path = artifact(crop_record, "colors")
        normals_path = artifact(crop_record, "normals")
        cache_path = artifact(crop_record, "projection_cache")
        points = np.asarray(np.load(points_path, allow_pickle=False), dtype=np.float32)
        colors = np.asarray(np.load(colors_path, allow_pickle=False), dtype=np.float32)
        normals = np.asarray(np.load(normals_path, allow_pickle=False), dtype=np.float32)
        n = int(points.shape[0])
        if (
            n <= 1
            or n > 50_000
            or points.shape != (n, 3)
            or colors.shape != (n, 3)
            or normals.shape != (n, 3)
        ):
            raise ValueError(f"{crop_manifest_path}: RGBN6 crop arrays are invalid")
        points, normals = canonicalize_structured3d_points_normals_to_scannet(
            points, normals
        )
        features = build_input_features(
            points,
            colors,
            use_colors=True,
            use_normals=True,
            normals=normals,
            color_mean=color_mean,
            color_std=color_std,
        ).astype(np.float32, copy=False)
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_schema = str(cache["schema_version"].item())
            declared_cache_schema = str(
                crop_record.get("projection_cache_schema", "")
            )
            if (
                cache_schema != declared_cache_schema
                or cache_schema
                not in {CLEAN_PROJECTION_CACHE_SCHEMA, RAW_CACHE_SCHEMA}
                or (
                    source_kind == "clean_pack"
                    and cache_schema != CLEAN_PROJECTION_CACHE_SCHEMA
                )
                or (
                    source_kind == "current_pack"
                    and cache_schema != RAW_CACHE_SCHEMA
                )
            ):
                raise ValueError(f"{cache_path}: projection-cache schema drift")
            if (
                cache_schema == CLEAN_PROJECTION_CACHE_SCHEMA
                and str(cache["variant_id"].item()) != variant_id
            ):
                raise ValueError(f"{cache_path}: clean projection variant drift")
            physical_ids = [
                str(value) for value in cache["physical_frame_ids"].tolist()
            ]
            granularities = [
                str(value) for value in cache["granularity_keys"].tolist()
            ]
            split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
            if cache_schema == CLEAN_PROJECTION_CACHE_SCHEMA:
                visible_offsets = np.asarray(
                    cache["visible_frame_offsets"], dtype=np.int64
                )
                visible_indices = np.asarray(
                    cache["visible_point_indices"], dtype=np.int64
                )
                known_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
                known_indices = np.asarray(
                    cache["eligible_point_indices"], dtype=np.int64
                )
            else:
                visible_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
                visible_indices = np.asarray(
                    cache["eligible_point_indices"], dtype=np.int64
                )
                known_offsets = None
                known_indices = None
            proposal_frame_offsets = np.asarray(
                cache["proposal_frame_offsets"], dtype=np.int64
            )
            proposal_point_offsets = np.asarray(
                cache["proposal_point_offsets"], dtype=np.int64
            )
            proposal_point_indices = np.asarray(
                cache["proposal_point_indices"], dtype=np.int64
            )
        num_frames = len(physical_ids)
        if (
            len(granularities) != num_frames
            or split_codes.shape != (num_frames,)
            or visible_offsets.shape != (num_frames + 1,)
            or proposal_frame_offsets.shape != (num_frames + 1,)
            or int(visible_offsets[-1]) != int(visible_indices.size)
            or int(proposal_frame_offsets[-1]) + 1
            != int(proposal_point_offsets.size)
            or int(proposal_point_offsets[-1])
            != int(proposal_point_indices.size)
        ):
            raise ValueError(f"{cache_path}: clean cache offsets drift")
        if cache_schema == CLEAN_PROJECTION_CACHE_SCHEMA and (
            known_offsets is None
            or known_indices is None
            or known_offsets.shape != (num_frames + 1,)
            or int(known_offsets[-1]) != int(known_indices.size)
        ):
            raise ValueError(f"{cache_path}: clean known-support offsets drift")
        raw_frames: list[RawFrame] = []
        trainable_cells: set[str] = set()
        for frame_index in range(num_frames):
            cell = f"2d/{granularities[frame_index]}"
            if cell not in TWO_D_CELLS:
                raise ValueError(f"{cache_path}: unexpected clean cell {cell!r}")
            visible = visible_indices[
                int(visible_offsets[frame_index]) : int(visible_offsets[frame_index + 1])
            ].copy()
            proposal_start = int(proposal_frame_offsets[frame_index])
            proposal_end = int(proposal_frame_offsets[frame_index + 1])
            member_start = int(proposal_point_offsets[proposal_start])
            member_end = int(proposal_point_offsets[proposal_end])
            members = proposal_point_indices[member_start:member_end].copy()
            offsets = (
                proposal_point_offsets[proposal_start : proposal_end + 1]
                - member_start
            ).copy()
            if known_offsets is not None and known_indices is not None:
                known = known_indices[
                    int(known_offsets[frame_index]) : int(
                        known_offsets[frame_index + 1]
                    )
                ]
            else:
                known = np.unique(members).astype(np.int64, copy=False)
            for values, label in ((visible, "visible"), (known, "known"), (members, "proposal")):
                if values.size and (int(values.min()) < 0 or int(values.max()) >= n):
                    raise ValueError(f"{cache_path}: {label} index outside crop")
            if np.setdiff1d(
                np.unique(known), np.unique(visible), assume_unique=True
            ).size:
                raise ValueError(f"{cache_path}: known support is not visible")
            if np.setdiff1d(
                np.unique(members), np.unique(known), assume_unique=True
            ).size:
                raise ValueError(f"{cache_path}: proposal lies outside known support")
            split = "train" if int(split_codes[frame_index]) == 0 else "heldout"
            if split == "train" and any(
                int(offsets[index + 1] - offsets[index]) >= 2
                and int(offsets[index + 1] - offsets[index]) < np.unique(visible).size
                for index in range(max(int(offsets.size) - 1, 0))
            ):
                trainable_cells.add(cell)
            raw_frames.append(
                RawFrame(
                    granularity=granularities[frame_index],
                    physical_frame_id=physical_ids[frame_index],
                    split=split,
                    visible_indices=np.unique(visible).astype(np.int64, copy=False),
                    proposal_offsets=offsets,
                    proposal_point_indices=members,
                )
            )
        if trainable_cells != set(TWO_D_CELLS):
            raise ValueError(
                f"{cache_path}: clean crop lacks trainable three-cell support: "
                f"{sorted(trainable_cells)}"
            )
        crops.append(
            SceneSource(
                scene_id=scene_id,
                manifest_path=manifest_path,
                manifest_sha256=sha256_file(manifest_path),
                points=points,
                features=features,
                labels_by_key={},
                raw_frames=tuple(raw_frames),
                colors=colors,
                normals=normals,
                crop_mode="sphere_50k",
                coordinate_normalization_policy=coordinate_normalization_policy,
                crop_id=crop_id,
                source_variant_id=variant_id,
            )
        )
    return MultiCropSceneSource(
        scene_id=scene_id,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        crops=tuple(crops),
        coordinate_normalization_policy=coordinate_normalization_policy,
    )


def load_scene_source(
    manifest_path: Path,
    *,
    input_features: str = "rgbn6",
    coordinate_normalization_policy: str = "native",
    verify_hashes: bool = True,
    color_mean: Sequence[float] | None = None,
    color_std: Sequence[float] | None = None,
) -> SceneSource | MultiCropSceneSource:
    input_features = str(input_features)
    coordinate_normalization_policy = str(coordinate_normalization_policy)
    if coordinate_normalization_policy not in {
        "native",
        STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    }:
        raise ValueError(
            "Unknown coordinate-normalization policy "
            f"{coordinate_normalization_policy!r}"
        )
    input_channels = input_channels_for(input_features)
    resolved_manifest = manifest_path.resolve(strict=True)
    initial_manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if initial_manifest.get("schema_version") == CLEAN_TRAINING_SOURCE_SCHEMA:
        return _load_clean_multicrop_scene_source(
            resolved_manifest,
            manifest=initial_manifest,
            input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            verify_hashes=verify_hashes,
            color_mean=color_mean,
            color_std=color_std,
        )
    audit = audit_scene_source_manifest(
        manifest_path,
        verify_hashes=verify_hashes,
    )
    manifest_path = audit.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    points = np.asarray(
        np.load(_resolve_artifact(manifest_path, artifacts["points"]), allow_pickle=False),
        dtype=np.float32,
    )
    colors = np.asarray(
        np.load(_resolve_artifact(manifest_path, artifacts["colors"]), allow_pickle=False),
        dtype=np.float32,
    )
    normals = None
    if input_features == "rgbn6":
        normals = np.asarray(
            np.load(
                _resolve_artifact(manifest_path, artifacts["normals"]),
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
    if (
        coordinate_normalization_policy
        == STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
    ):
        if input_features != "rgbn6" or normals is None:
            raise ValueError(
                f"{coordinate_normalization_policy} requires RGBN6 points and normals"
            )
        points, normals = canonicalize_structured3d_points_normals_to_scannet(
            points,
            normals,
        )
    features = build_input_features(
        points,
        colors,
        use_colors=True,
        use_normals=input_features == "rgbn6",
        normals=normals,
        color_mean=color_mean,
        color_std=color_std,
    ).astype(np.float32, copy=False)
    labels_by_key = {
        key: np.asarray(
            np.load(
                _resolve_artifact(manifest_path, artifacts[f"labels_{key}"]),
                allow_pickle=False,
            ),
            dtype=np.int64,
        )
        for key in GRANULARITY_KEYS
    }
    n = int(points.shape[0])
    if (
        points.shape != (n, 3)
        or colors.shape != (n, 3)
        or features.shape != (n, input_channels)
        or (normals is not None and normals.shape != (n, 3))
    ):
        raise ValueError(f"{manifest_path}: point/input shape drift")
    if any(labels.shape != (n,) for labels in labels_by_key.values()):
        raise ValueError(f"{manifest_path}: label shape drift")

    cache_path = _resolve_artifact(manifest_path, artifacts["raw_projection_cache"])
    with np.load(cache_path, allow_pickle=False) as cache:
        if str(cache["schema_version"].item()) != RAW_CACHE_SCHEMA:
            raise ValueError(f"{cache_path}: raw cache schema drift")
        crop_path = _resolve_artifact(manifest_path, artifacts["crop_indices"])
        if (
            verify_hashes
            and str(cache["crop_indices_sha256"].item()) != sha256_file(crop_path)
        ):
            raise ValueError(f"{cache_path}: crop hash drift")
        physical_ids = [str(value) for value in cache["physical_frame_ids"].tolist()]
        granularities = [str(value) for value in cache["granularity_keys"].tolist()]
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        frame_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
        visible_indices = np.asarray(cache["eligible_point_indices"], dtype=np.int64)
        proposal_frame_offsets = np.asarray(
            cache["proposal_frame_offsets"], dtype=np.int64
        )
        proposal_point_offsets = np.asarray(
            cache["proposal_point_offsets"], dtype=np.int64
        )
        proposal_point_indices = np.asarray(
            cache["proposal_point_indices"], dtype=np.int64
        )
    num_frames = len(physical_ids)
    if (
        len(granularities) != num_frames
        or split_codes.shape != (num_frames,)
        or frame_offsets.shape != (num_frames + 1,)
        or proposal_frame_offsets.shape != (num_frames + 1,)
        or int(frame_offsets[-1]) != int(visible_indices.size)
        or int(proposal_frame_offsets[-1]) + 1 != int(proposal_point_offsets.size)
        or int(proposal_point_offsets[-1]) != int(proposal_point_indices.size)
    ):
        raise ValueError(f"{cache_path}: compact-cache offset drift")

    raw_frames: list[RawFrame] = []
    for frame_index in range(num_frames):
        visible_start = int(frame_offsets[frame_index])
        visible_end = int(frame_offsets[frame_index + 1])
        proposal_start = int(proposal_frame_offsets[frame_index])
        proposal_end = int(proposal_frame_offsets[frame_index + 1])
        point_start = int(proposal_point_offsets[proposal_start])
        point_end = int(proposal_point_offsets[proposal_end])
        local_offsets = (
            proposal_point_offsets[proposal_start : proposal_end + 1] - point_start
        )
        visible = visible_indices[visible_start:visible_end].copy()
        members = proposal_point_indices[point_start:point_end].copy()
        if visible.size and (int(visible.min()) < 0 or int(visible.max()) >= n):
            raise ValueError(f"{cache_path}: visible index outside crop")
        if members.size and (int(members.min()) < 0 or int(members.max()) >= n):
            raise ValueError(f"{cache_path}: proposal index outside crop")
        raw_frames.append(
            RawFrame(
                granularity=granularities[frame_index],
                physical_frame_id=physical_ids[frame_index],
                split="train" if int(split_codes[frame_index]) == 0 else "heldout",
                visible_indices=visible,
                proposal_offsets=local_offsets.copy(),
                proposal_point_indices=members,
            )
        )
    expected_physical = len(audit.train_frames) + len(audit.heldout_frames)
    for key in GRANULARITY_KEYS:
        if len([frame for frame in raw_frames if frame.granularity == key]) != expected_physical:
            raise ValueError(
                f"{cache_path}: expected {expected_physical} virtual frames for {key}"
            )
        if len([frame for frame in raw_frames if frame.granularity == key and frame.split == "train"]) != len(audit.train_frames):
            raise ValueError(f"{cache_path}: train-frame cardinality drift for {key}")
        if len([frame for frame in raw_frames if frame.granularity == key and frame.split == "heldout"]) != len(audit.heldout_frames):
            raise ValueError(f"{cache_path}: heldout-frame cardinality drift for {key}")
    return SceneSource(
        scene_id=audit.scene_id,
        manifest_path=manifest_path,
        manifest_sha256=audit.manifest_sha256,
        points=points,
        features=features,
        labels_by_key=labels_by_key,
        raw_frames=tuple(raw_frames),
        colors=colors,
        normals=normals,
        crop_mode=audit.crop_mode,
        coordinate_normalization_policy=coordinate_normalization_policy,
    )


def _generator(device: torch.device, *parts: Any) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(stable_seed(*parts))


def sample_raw_batch(
    *,
    scene: SceneSource,
    frame: RawFrame,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed_parts: Iterable[Any],
    selected_proposal_indices: torch.Tensor | None = None,
) -> ContrastiveTripletBatch:
    generator = _generator(device, *seed_parts)
    return sample_overlapping_proposal_triplets(
        num_scene_points=int(scene.points.shape[0]),
        visible_indices=torch.from_numpy(frame.visible_indices).to(device),
        proposal_offsets=torch.from_numpy(frame.proposal_offsets).to(device),
        proposal_point_indices=torch.from_numpy(frame.proposal_point_indices).to(device),
        points=points,
        num_proposals=sampling.proposals_per_forward,
        positive_pairs_per_proposal=sampling.positive_pairs_per_proposal,
        num_uniform_negatives=sampling.uniform_negatives,
        num_spatial_hard_negatives=sampling.spatial_hard_negatives,
        spatial_candidate_pool=sampling.spatial_candidate_pool,
        num_feature_hard_negatives=sampling.feature_hard_negatives,
        feature_candidate_pool=sampling.feature_candidate_pool,
        selected_proposal_indices=selected_proposal_indices,
        generator=generator,
    )


def sample_3d_batch(
    *,
    labels: torch.Tensor,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed_parts: Iterable[Any],
    selected_proposal_labels: torch.Tensor | None = None,
) -> ContrastiveTripletBatch:
    generator = _generator(device, *seed_parts)
    return sample_partition_triplets(
        labels,
        eligible_mask=labels >= 0,
        points=points,
        num_proposals=sampling.proposals_per_forward,
        positive_pairs_per_proposal=sampling.positive_pairs_per_proposal,
        num_uniform_negatives=sampling.uniform_negatives,
        num_spatial_hard_negatives=sampling.spatial_hard_negatives,
        spatial_candidate_pool=sampling.spatial_candidate_pool,
        num_feature_hard_negatives=sampling.feature_hard_negatives,
        feature_candidate_pool=sampling.feature_candidate_pool,
        ignored_proposal_labels=(-1,),
        exclude_ignored_from_negatives=True,
        selected_proposal_labels=selected_proposal_labels,
        generator=generator,
    )


def coverage_queue_indices(
    *,
    population_size: int,
    start: int,
    count: int,
    seed_parts: Iterable[Any],
) -> list[int]:
    """Draw deterministic queue positions without replacement within each cycle."""

    population_size = int(population_size)
    start = int(start)
    count = int(count)
    if population_size <= 0:
        raise ValueError("coverage queue population_size must be positive")
    if start < 0 or count <= 0:
        raise ValueError("coverage queue start/count must be non-negative/positive")
    parts = tuple(seed_parts)
    permutations: dict[int, np.ndarray] = {}
    selected: list[int] = []
    for queue_position in range(start, start + count):
        cycle, within_cycle = divmod(queue_position, population_size)
        if cycle not in permutations:
            permutations[cycle] = np.random.default_rng(
                stable_seed(*parts, cycle)
            ).permutation(population_size)
        selected.append(int(permutations[cycle][within_cycle]))
    return selected


def _valid_raw_proposal_indices(frame: RawFrame) -> np.ndarray:
    sizes = frame.proposal_offsets[1:] - frame.proposal_offsets[:-1]
    return np.where(
        (sizes >= 2) & (sizes < int(frame.visible_indices.size))
    )[0].astype(np.int64, copy=False)


def _valid_partition_labels(labels: torch.Tensor) -> torch.Tensor:
    eligible = labels >= 0
    proposal_labels, counts = torch.unique(
        labels[eligible],
        sorted=True,
        return_counts=True,
    )
    eligible_count = int(eligible.sum())
    valid = (counts >= 2) & ((eligible_count - counts) >= 1)
    selected = proposal_labels[valid]
    if selected.numel() == 0:
        raise ValueError("No valid 3D proposal label for coverage sampling")
    return selected


def _coverage_raw_training_batch(
    *,
    scene: SceneSource,
    granularity: str,
    cell: str,
    epoch: int,
    occurrence: int,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
) -> tuple[ContrastiveTripletBatch, dict[str, Any]]:
    frames = tuple(
        frame
        for frame in scene.frames(granularity, "train")
        if _valid_raw_proposal_indices(frame).size > 0
    )
    if not frames:
        raise ValueError(
            f"{scene.scene_id}/{granularity}: no proposal-bearing train frame"
        )
    proposal_count = int(sampling.proposals_per_forward)
    queue_start = int(occurrence) * proposal_count
    frame_history = coverage_queue_indices(
        population_size=len(frames),
        start=0,
        count=queue_start + proposal_count,
        seed_parts=(scene.scene_id, granularity, "coverage-frame"),
    )
    current_frame_indices = frame_history[queue_start:]
    one_proposal = replace(sampling, proposals_per_forward=1)
    batches: list[ContrastiveTripletBatch] = []
    frame_ids: list[str] = []
    proposal_ids: list[str] = []
    for local_draw, frame_index in enumerate(current_frame_indices):
        frame = frames[frame_index]
        history_position = queue_start + local_draw
        prior_uses = sum(
            int(previous == frame_index)
            for previous in frame_history[:history_position]
        )
        valid_proposals = _valid_raw_proposal_indices(frame)
        if valid_proposals.size == 0:
            raise ValueError(
                f"{scene.scene_id}/{granularity}/{frame.physical_frame_id}: "
                "coverage frame has no valid proposal"
            )
        proposal_queue_index = coverage_queue_indices(
            population_size=int(valid_proposals.size),
            start=prior_uses,
            count=1,
            seed_parts=(
                scene.scene_id,
                granularity,
                frame.physical_frame_id,
                "coverage-proposal",
            ),
        )[0]
        proposal_index = int(valid_proposals[proposal_queue_index])
        batch = sample_raw_batch(
            scene=scene,
            frame=frame,
            points=points,
            device=device,
            sampling=one_proposal,
            seed_parts=(
                scene.scene_id,
                epoch,
                cell,
                frame.physical_frame_id,
                proposal_index,
                "coverage-train",
            ),
            selected_proposal_indices=torch.tensor(
                [proposal_index],
                dtype=torch.long,
                device=device,
            ),
        )
        # Coverage intentionally draws proposals from different RGB-D frames,
        # so there is no single eligible-point universe or seed for the
        # concatenated batch.  Preserve the historical single-scale coverage
        # path by dropping only the optional multi-scale CSR.  A multi-scale
        # run that selects coverage mode then fails explicitly in the strict
        # pure-token sampler instead of consuming invalid metadata.
        batches.append(
            replace(
                batch,
                proposal_member_offsets=None,
                proposal_member_indices=None,
                pair_proposal_indices=None,
                eligible_indices=None,
                multiscale_seed=None,
            )
        )
        frame_ids.append(frame.physical_frame_id)
        proposal_ids.append(f"{frame.physical_frame_id}:{proposal_index}")
    return concatenate_triplet_batches(batches), {
        "cell": cell,
        "family": "2d",
        "granularity": granularity,
        "frame_ids": frame_ids,
        "proposal_ids": proposal_ids,
    }


def training_batch(
    *,
    scene: SceneSource,
    epoch: int,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    repeat_g08_for_all_3d_slots: bool = False,
    sampling_mode: str = "random_with_replacement",
    return_metadata: bool = False,
    cells: Sequence[str] | None = None,
    cell_schedule: str = SOURCE_CELL_SCHEDULE_FIXED,
    schedule_seed: int = 42,
) -> (
    tuple[str, ContrastiveTripletBatch]
    | tuple[str, ContrastiveTripletBatch, dict[str, Any]]
):
    if sampling_mode not in TRAINING_SAMPLING_MODES:
        raise ValueError(
            f"Unknown training sampling mode {sampling_mode!r}; "
            f"expected one of {TRAINING_SAMPLING_MODES}"
        )
    active_cells = normalize_source_cells(cells)
    cell = source_cell_for_scene_epoch(
        scene.scene_id,
        epoch,
        cells=active_cells,
        schedule=cell_schedule,
        seed=schedule_seed,
    )
    family, granularity = cell.split("/", 1)
    occurrence = int(epoch) // len(active_cells)
    if family == "3d":
        label_key = "g08" if repeat_g08_for_all_3d_slots else granularity
        labels = torch.from_numpy(scene.labels_by_key[label_key]).to(device)
        selected_labels = None
        if sampling_mode in (
            "coverage_without_replacement",
            "hybrid_2d_random_3d_coverage",
        ):
            valid_labels = _valid_partition_labels(labels)
            selected_indices = coverage_queue_indices(
                population_size=int(valid_labels.numel()),
                start=occurrence * int(sampling.proposals_per_forward),
                count=int(sampling.proposals_per_forward),
                seed_parts=(
                    scene.scene_id,
                    cell,
                    label_key,
                    "coverage-3d-label",
                ),
            )
            selected_labels = valid_labels[
                torch.tensor(selected_indices, dtype=torch.long, device=device)
            ]
        batch = sample_3d_batch(
            labels=labels,
            points=points,
            device=device,
            sampling=sampling,
            seed_parts=(scene.scene_id, epoch, cell, label_key, "train"),
            selected_proposal_labels=selected_labels,
        )
        metadata = {
            "cell": cell,
            "family": "3d",
            "granularity": granularity,
            "frame_ids": [],
            "proposal_ids": [
                str(int(value))
                for value in (
                    selected_labels
                    if selected_labels is not None
                    else torch.unique(batch.proposal_labels, sorted=True)
                ).tolist()
            ],
        }
        return (cell, batch, metadata) if return_metadata else (cell, batch)
    if sampling_mode == "coverage_without_replacement":
        batch, metadata = _coverage_raw_training_batch(
            scene=scene,
            granularity=granularity,
            cell=cell,
            epoch=epoch,
            occurrence=occurrence,
            points=points,
            device=device,
            sampling=sampling,
        )
        return (cell, batch, metadata) if return_metadata else (cell, batch)
    frames = scene.frames(granularity, "train")
    if not frames:
        raise ValueError(
            f"{scene.scene_id}/{granularity}: no proposal-bearing train frame"
        )
    offset = stable_seed(scene.scene_id, granularity, "train-frame") % len(frames)
    frame = frames[(offset + occurrence) % len(frames)]
    batch = sample_raw_batch(
        scene=scene,
        frame=frame,
        points=points,
        device=device,
        sampling=sampling,
        seed_parts=(
            scene.scene_id,
            epoch,
            cell,
            frame.physical_frame_id,
            "train",
        ),
    )
    metadata = {
        "cell": cell,
        "family": "2d",
        "granularity": granularity,
        "frame_ids": [frame.physical_frame_id],
        "proposal_ids": [
            f"{frame.physical_frame_id}:{int(value)}"
            for value in torch.unique(batch.proposal_labels, sorted=True).tolist()
        ],
    }
    return (cell, batch, metadata) if return_metadata else (cell, batch)


def _holdout_batches(
    *,
    scene: SceneSource,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed: int,
    cells: Sequence[str] | None = None,
) -> list[tuple[str, str, ContrastiveTripletBatch]]:
    active_cells = set(normalize_source_cells(cells))
    batches: list[tuple[str, str, ContrastiveTripletBatch]] = []
    for key in GRANULARITY_KEYS:
        if f"3d/{key}" in active_cells:
            labels = torch.from_numpy(scene.labels_by_key[key]).to(device)
            batches.append(
                (
                    f"3d/{key}",
                    "3d",
                    sample_3d_batch(
                        labels=labels,
                        points=points,
                        device=device,
                        sampling=sampling,
                        seed_parts=(seed, scene.scene_id, "eval", "3d", key),
                    ),
                )
            )
        if f"2d/{key}" not in active_cells:
            continue
        for frame in scene.frames(key, "heldout"):
            batches.append(
                (
                    f"2d/{key}",
                    f"2d/{frame.physical_frame_id}",
                    sample_raw_batch(
                        scene=scene,
                        frame=frame,
                        points=points,
                        device=device,
                        sampling=sampling,
                        seed_parts=(
                            seed,
                            scene.scene_id,
                            "eval",
                            "2d",
                            key,
                            frame.physical_frame_id,
                        ),
                    ),
                )
            )
    return batches


def _token_pure_fixed_holdout_plan(
    *,
    scene: SceneSource,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed: int,
) -> Any:
    """Build one deterministic V2 heldout visit with exact frame locality.

    The V2 sampler, unlike the legacy holdout sampler above, carries the union
    of every g02/g05/g08 mask for a physical frame as its relation CSR and
    restricts negative eligibility to visible points with known relation
    membership.  Rotating the 4-way evaluation quota by scene balances the
    unavoidable extra proposal over a 3-cell panel without changing a saved
    pack between checkpoints.
    """

    frames_by_cell: dict[str, tuple[FrameProposalCatalog, ...]] = {}
    for cell in TWO_D_CELLS:
        granularity = cell.split("/", 1)[1]
        frames_by_cell[cell] = tuple(
            FrameProposalCatalog(
                cell=cell,
                frame_id=frame.physical_frame_id,
                visible_indices=torch.as_tensor(
                    frame.visible_indices, dtype=torch.long, device=device
                ),
                proposal_offsets=torch.as_tensor(
                    frame.proposal_offsets, dtype=torch.long, device=device
                ),
                proposal_point_indices=torch.as_tensor(
                    frame.proposal_point_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
            for frame in scene.frames(granularity, "heldout")
        )
    quota_rotation = int(
        stable_seed(seed, scene.scene_id, "fixed-eval-v2-quota-rotation")
        % len(TWO_D_CELLS)
    )
    total_quota = max(len(TWO_D_CELLS), int(sampling.proposals_per_forward))
    quotas = rotating_three_cell_quotas(
        quota_rotation,
        total_proposals=total_quota,
        cells=TWO_D_CELLS,
    )
    populations = {
        cell: sum(
            int(catalog.valid_proposal_indices().numel())
            for catalog in frames_by_cell[cell]
        )
        for cell in TWO_D_CELLS
    }
    if any(populations[cell] <= 0 for cell in TWO_D_CELLS):
        raise ValueError(
            f"{scene.scene_id}: V2 fixed-eval catalog lacks a granularity: "
            f"{populations}"
        )
    # Retry within the exact same deterministic fixed-eval key by restoring
    # each cell's state_after cursor.  ``quota_rotation`` affects only which
    # cell receives the extra fixed-pack proposal; it is not a training epoch
    # and must not demand a non-existent training-resume cursor.  The bound
    # visits every catalog entry at least once, so failure means no complete
    # safe three-cell slice exists rather than an unlucky first draw.
    maximum_attempts = max(
        math.ceil(populations[cell] / quotas[cell])
        for cell in TWO_D_CELLS
    )
    coverage_states: dict[str, DeterministicCoverageState | None] | None = None
    last_plan = None
    for attempt in range(maximum_attempts):
        try:
            plan = sample_multigranular_frame_group_plan(
                scene_id=scene.scene_id,
                epoch=0,
                seed=seed,
                points=points,
                frames_by_cell=frames_by_cell,
                total_proposal_quota=total_quota,
                cells=TWO_D_CELLS,
                positive_pairs_per_proposal=int(
                    sampling.positive_pairs_per_proposal
                ),
                num_uniform_negatives=int(sampling.uniform_negatives),
                num_spatial_hard_negatives=int(sampling.spatial_hard_negatives),
                spatial_candidate_pool=int(sampling.spatial_candidate_pool),
                num_feature_hard_negatives=int(sampling.feature_hard_negatives),
                feature_candidate_pool=int(sampling.feature_candidate_pool),
                require_negative_proposal_membership=True,
                require_hierarchy_routes=False,
                operation="multigranular-heldout-fixed-v2",
                coverage_states=coverage_states,
                quota_epoch=quota_rotation,
            )
        except NoRetainedFrameGroupsError as error:
            plan = error.plan
        last_plan = plan
        if {group.cell for group in plan.groups} == set(TWO_D_CELLS):
            annotated = {
                cell: {
                    **dict(metadata),
                    "fixed_eval_search_attempt": attempt,
                    "fixed_eval_search_maximum_attempts": maximum_attempts,
                }
                for cell, metadata in plan.coverage_by_cell.items()
            }
            return replace(plan, coverage_by_cell=annotated)
        coverage_states = {
            cell: (
                DeterministicCoverageState.from_dict(state)
                if isinstance(state, Mapping)
                else None
            )
            for cell, state in plan.state_after_by_cell().items()
        }
    raise ValueError(
        f"{scene.scene_id}: V2 fixed-eval queue exhausted without one safe "
        f"three-cell slice after {maximum_attempts} attempts: "
        f"{last_plan.audit_metadata() if last_plan is not None else populations}"
    )


def _expected_holdout_record_count(
    scene: SceneSource,
    cells: Sequence[str] | None = None,
) -> int:
    active_cells = set(normalize_source_cells(cells))
    count = 0
    for key in GRANULARITY_KEYS:
        if f"3d/{key}" in active_cells:
            count += 1
        if f"2d/{key}" in active_cells:
            count += len(scene.frames(key, "heldout"))
    return count


def _triplet_batch_record(
    batch: ContrastiveTripletBatch,
    *,
    token_pure_fixed_pack: bool = False,
    fixed_runtime_token_pair_rebuild: bool = False,
) -> dict[str, Any]:
    token_pair_payload_fields = (
        batch.token_positive_pair_offsets,
        batch.token_positive_pair_indices,
        batch.token_positive_pair_digest,
    )
    token_pair_payload_presence = tuple(
        value is not None for value in token_pair_payload_fields
    )
    if any(token_pair_payload_presence) and not all(token_pair_payload_presence):
        raise ValueError(
            "native-dec0 positive-pair payload fields must be supplied together"
        )
    if fixed_runtime_token_pair_rebuild:
        if not token_pure_fixed_pack:
            raise ValueError(
                "runtime token-pair rebuild is only valid for a token-pure fixed pack"
            )
        if not all(token_pair_payload_presence):
            raise ValueError(
                "fixed token-pair rebuild requires a validated planner payload"
            )
        if (
            batch.proposal_member_offsets is None
            or batch.pair_proposal_indices is None
        ):
            raise ValueError(
                "fixed token-pair rebuild requires proposal-occurrence metadata"
            )
        assert batch.token_positive_pair_offsets is not None
        assert batch.token_positive_pair_indices is not None
        assert batch.token_positive_pair_digest is not None
        validate_token_positive_pair_payload(
            batch.token_positive_pair_offsets,
            batch.token_positive_pair_indices,
            batch.token_positive_pair_digest,
            proposal_count=int(batch.proposal_member_offsets.numel()) - 1,
            algorithm_version=batch.algorithm_version,
            preselection_policy=SAMPLER_PRESELECTION_POLICY,
            proposal_labels=batch.proposal_labels,
            pair_proposal_indices=batch.pair_proposal_indices,
        )
    return {
        "token_pure_fixed_pack": bool(token_pure_fixed_pack),
        "positive_pairs": batch.positive_pairs.detach().cpu().long(),
        "negative_indices": batch.negative_indices.detach().cpu().long(),
        "proposal_labels": batch.proposal_labels.detach().cpu().long(),
        "feature_candidate_indices": (
            batch.feature_candidate_indices.detach().cpu().long()
            if batch.feature_candidate_indices is not None
            else None
        ),
        "deferred_token_loss": (
            batch.deferred_token_loss.metadata()
            if batch.deferred_token_loss is not None
            else None
        ),
        "num_feature_hard_negatives": int(batch.num_feature_hard_negatives),
        "proposal_member_offsets": (
            batch.proposal_member_offsets.detach().cpu().long()
            if batch.proposal_member_offsets is not None
            else None
        ),
        "proposal_member_indices": (
            batch.proposal_member_indices.detach().cpu().long()
            if batch.proposal_member_indices is not None
            else None
        ),
        "pair_proposal_indices": (
            batch.pair_proposal_indices.detach().cpu().long()
            if batch.pair_proposal_indices is not None
            else None
        ),
        "eligible_indices": (
            batch.eligible_indices.detach().cpu().long()
            if batch.eligible_indices is not None
            else None
        ),
        "relation_proposal_offsets": (
            batch.relation_proposal_offsets.detach().cpu().long()
            if batch.relation_proposal_offsets is not None
            else None
        ),
        "relation_proposal_member_indices": (
            batch.relation_proposal_member_indices.detach().cpu().long()
            if batch.relation_proposal_member_indices is not None
            else None
        ),
        "token_positive_pair_offsets": (
            batch.token_positive_pair_offsets.detach().cpu().long()
            if (
                batch.token_positive_pair_offsets is not None
                and not fixed_runtime_token_pair_rebuild
            )
            else None
        ),
        "token_positive_pair_indices": (
            batch.token_positive_pair_indices.detach().cpu().long()
            if (
                batch.token_positive_pair_indices is not None
                and not fixed_runtime_token_pair_rebuild
            )
            else None
        ),
        "token_positive_pair_digest": (
            None
            if fixed_runtime_token_pair_rebuild
            else batch.token_positive_pair_digest
        ),
        "native_dec0_pair_selection": (
            TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
            if fixed_runtime_token_pair_rebuild
            else None
        ),
        "algorithm_version": batch.algorithm_version,
        "require_negative_proposal_membership": bool(
            batch.require_negative_proposal_membership
        ),
        "multiscale_seed": batch.multiscale_seed,
        "num_uniform_negatives": int(batch.num_uniform_negatives),
        "num_spatial_hard_negatives": int(batch.num_spatial_hard_negatives),
        "spatial_candidate_pool": int(batch.spatial_candidate_pool),
        "feature_candidate_pool": int(batch.feature_candidate_pool),
    }


def _triplet_batch_from_record(
    record: dict[str, Any],
    *,
    device: torch.device,
    include_feature_hard: bool,
    sampling: SamplingConfig,
    num_scene_points: int,
) -> ContrastiveTripletBatch:
    token_pure_fixed_pack = bool(record.get("token_pure_fixed_pack", False))
    positive_pairs = record["positive_pairs"].to(device=device, dtype=torch.long)
    negative_indices = record["negative_indices"].to(
        device=device, dtype=torch.long
    )
    proposal_labels = record["proposal_labels"].to(
        device=device, dtype=torch.long
    )
    feature_candidates_cpu = record.get("feature_candidate_indices")
    feature_candidates = (
        feature_candidates_cpu.to(device=device, dtype=torch.long)
        if include_feature_hard and feature_candidates_cpu is not None
        else None
    )
    stored_feature_hard_count = int(record["num_feature_hard_negatives"])
    feature_hard_count = stored_feature_hard_count if include_feature_hard else 0
    if token_pure_fixed_pack and "require_negative_proposal_membership" not in record:
        raise ValueError(
            "Token-pure fixed eval requires an explicit negative-membership policy"
        )
    require_negative_proposal_membership = bool(
        record.get("require_negative_proposal_membership", False)
    )
    if token_pure_fixed_pack and not require_negative_proposal_membership:
        raise ValueError(
            "Token-pure fixed eval must restrict negatives to known relation members"
        )
    stored_feature_candidate_pool = int(
        record.get("feature_candidate_pool", sampling.feature_candidate_pool)
    )
    deferred_metadata = record.get("deferred_token_loss")
    deferred_token_loss: DeferredTokenLossPlan | None = None
    if deferred_metadata is not None:
        if not isinstance(deferred_metadata, Mapping):
            raise ValueError("Fixed-eval deferred token-loss metadata is malformed")
        try:
            deferred_token_loss = DeferredTokenLossPlan(
                requested_positive_pairs_per_proposal=int(
                    deferred_metadata["requested_positive_pairs_per_proposal"]
                ),
                requested_uniform_negatives_per_pair=int(
                    deferred_metadata["requested_uniform_negatives_per_pair"]
                ),
                requested_spatial_hard_negatives_per_pair=int(
                    deferred_metadata["requested_spatial_hard_negatives_per_pair"]
                ),
                requested_feature_hard_negatives_per_pair=int(
                    deferred_metadata["requested_feature_hard_negatives_per_pair"]
                ),
                effective_uniform_negatives_per_pair=int(
                    deferred_metadata["effective_uniform_negatives_per_pair"]
                ),
                effective_spatial_hard_negatives_per_pair=int(
                    deferred_metadata["effective_spatial_hard_negatives_per_pair"]
                ),
                effective_feature_candidate_pool_per_pair=int(
                    deferred_metadata["effective_feature_candidate_pool_per_pair"]
                ),
                effective_feature_hard_negatives_per_pair=int(
                    deferred_metadata["effective_feature_hard_negatives_per_pair"]
                ),
                point_negative_ids_materialized=bool(
                    deferred_metadata["point_negative_ids_materialized"]
                ),
                consumed_loss_space=str(deferred_metadata["consumed_loss_space"]),
                schema_version=str(deferred_metadata["schema_version"]),
                algorithm_version=str(deferred_metadata["algorithm_version"]),
                require_negative_proposal_membership=bool(
                    deferred_metadata["require_negative_proposal_membership"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Fixed-eval deferred token-loss metadata is invalid"
            ) from error
        if deferred_token_loss.metadata() != dict(deferred_metadata):
            raise ValueError("Fixed-eval deferred token-loss metadata drift")
        if (
            bool(deferred_token_loss.require_negative_proposal_membership)
            != require_negative_proposal_membership
        ):
            raise ValueError(
                "Fixed-eval deferred token-loss membership policy drift"
            )
        observed_deferred_widths = (
            int(record.get("num_uniform_negatives", sampling.uniform_negatives)),
            int(
                record.get(
                    "num_spatial_hard_negatives", sampling.spatial_hard_negatives
                )
            ),
            stored_feature_candidate_pool,
            stored_feature_hard_count,
        )
        expected_deferred_widths = (
            int(deferred_token_loss.effective_uniform_negatives_per_pair),
            int(deferred_token_loss.effective_spatial_hard_negatives_per_pair),
            int(deferred_token_loss.effective_feature_candidate_pool_per_pair),
            int(deferred_token_loss.effective_feature_hard_negatives_per_pair),
        )
        if observed_deferred_widths != expected_deferred_widths:
            raise ValueError("Fixed-eval deferred token-loss width drift")
    if (
        token_pure_fixed_pack
        and stored_feature_hard_count > 0
        and deferred_token_loss is None
        and feature_candidates_cpu is None
    ):
        raise ValueError(
            "Token-pure fixed eval needs deferred native-token feature candidates"
        )
    if not include_feature_hard:
        stored_feature_candidate_pool = 0
        if deferred_token_loss is not None:
            deferred_token_loss = replace(
                deferred_token_loss,
                requested_feature_hard_negatives_per_pair=0,
                effective_feature_candidate_pool_per_pair=0,
                effective_feature_hard_negatives_per_pair=0,
            )
    membership_keys = (
        "proposal_member_offsets",
        "proposal_member_indices",
        "pair_proposal_indices",
        "eligible_indices",
    )
    membership_presence = [record.get(key) is not None for key in membership_keys]
    if any(membership_presence) and not all(membership_presence):
        raise ValueError("Fixed-eval proposal-membership metadata is partial")
    proposal_member_offsets = (
        record["proposal_member_offsets"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    proposal_member_indices = (
        record["proposal_member_indices"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    pair_proposal_indices = (
        record["pair_proposal_indices"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    eligible_indices = (
        record["eligible_indices"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    relation_keys = (
        "relation_proposal_offsets",
        "relation_proposal_member_indices",
    )
    relation_presence = [record.get(key) is not None for key in relation_keys]
    if any(relation_presence) and not all(relation_presence):
        raise ValueError("Fixed-eval relation-proposal metadata is partial")
    relation_proposal_offsets = (
        record["relation_proposal_offsets"].to(
            device=device, dtype=torch.long
        )
        if all(relation_presence)
        else None
    )
    relation_proposal_member_indices = (
        record["relation_proposal_member_indices"].to(
            device=device, dtype=torch.long
        )
        if all(relation_presence)
        else None
    )
    token_pair_keys = (
        "token_positive_pair_offsets",
        "token_positive_pair_indices",
        "token_positive_pair_digest",
    )
    token_pair_presence = [record.get(key) is not None for key in token_pair_keys]
    if any(token_pair_presence) and not all(token_pair_presence):
        raise ValueError("Fixed-eval native-dec0 pair payload is partial")
    fixed_pair_selection = record.get("native_dec0_pair_selection")
    runtime_pair_rebuild = (
        fixed_pair_selection == TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
    )
    if token_pure_fixed_pack:
        if runtime_pair_rebuild:
            if any(token_pair_presence):
                raise ValueError(
                    "runtime fixed token-pair rebuild must not serialize backend token IDs"
                )
        elif not all(token_pair_presence):
            raise ValueError(
                "Token-pure fixed eval requires an exact payload or runtime "
                "native-token rebuild policy"
            )
    token_positive_pair_offsets = (
        record["token_positive_pair_offsets"].to(device=device, dtype=torch.long)
        if all(token_pair_presence)
        else None
    )
    token_positive_pair_indices = (
        record["token_positive_pair_indices"].to(device=device, dtype=torch.long)
        if all(token_pair_presence)
        else None
    )
    token_positive_pair_digest = (
        record["token_positive_pair_digest"] if all(token_pair_presence) else None
    )
    algorithm_version = record.get("algorithm_version")
    if token_pure_fixed_pack and (
        not all(membership_presence) or not all(relation_presence)
    ):
        raise ValueError(
            "Token-pure fixed eval requires selected and full relation CSR"
        )
    expected_pairs = (
        int(sampling.proposals_per_forward)
        * int(sampling.positive_pairs_per_proposal)
    )
    expected_base = int(sampling.uniform_negatives) + int(
        sampling.spatial_hard_negatives
    )
    if token_pure_fixed_pack:
        if positive_pairs.ndim != 2 or positive_pairs.shape[1] != 2:
            raise ValueError(
                "V2 fixed-eval positive pairs must have shape [P,2]"
            )
        if positive_pairs.shape[0] <= 0:
            raise ValueError("V2 fixed-eval batch contains no positive pair")
    elif positive_pairs.shape != (expected_pairs, 2):
        raise ValueError(
            f"Fixed-eval positive-pair shape drift: {tuple(positive_pairs.shape)}"
        )
    if token_pure_fixed_pack:
        if (
            negative_indices.ndim != 2
            or negative_indices.shape[0] != positive_pairs.shape[0]
        ):
            raise ValueError(
                "V2 fixed-eval negative rows do not match positive pairs"
            )
    elif negative_indices.shape != (expected_pairs, expected_base):
        raise ValueError(
            "Fixed-eval base-negative shape drift: "
            f"{tuple(negative_indices.shape)}"
        )
    expected_record_pairs = int(positive_pairs.shape[0])
    if proposal_labels.shape != (expected_record_pairs,):
        raise ValueError(
            f"Fixed-eval proposal-label shape drift: {tuple(proposal_labels.shape)}"
        )
    if include_feature_hard:
        if feature_candidates is None:
            if (
                not token_pure_fixed_pack
                or (feature_hard_count != 0 and deferred_token_loss is None)
            ):
                raise ValueError(
                    "Fixed-eval adaptive view lacks feature candidates"
                )
        else:
            if feature_candidates.ndim != 2:
                raise ValueError(
                    "Fixed-eval feature candidates must have shape [P,Q]"
                )
            expected_feature_shape = (
                (expected_record_pairs, int(feature_candidates.shape[1]))
                if token_pure_fixed_pack
                else (expected_pairs, int(sampling.feature_candidate_pool))
            )
            if feature_candidates.shape != expected_feature_shape:
                raise ValueError(
                    "Fixed-eval feature-candidate shape drift: "
                    f"{tuple(feature_candidates.shape)}"
                )
            if token_pure_fixed_pack:
                if (
                    feature_hard_count < 0
                    or feature_hard_count > int(feature_candidates.shape[1])
                    or feature_hard_count > int(sampling.feature_hard_negatives)
                ):
                    raise ValueError("V2 fixed-eval feature-hard count drift")
            elif feature_hard_count != int(sampling.feature_hard_negatives):
                raise ValueError("Fixed-eval feature-hard count drift")
    index_chunks = [positive_pairs.reshape(-1), negative_indices.reshape(-1)]
    if feature_candidates is not None:
        index_chunks.append(feature_candidates.reshape(-1))
    indices = torch.cat(index_chunks)
    if (
        indices.numel() == 0
        or int(indices.min()) < 0
        or int(indices.max()) >= int(num_scene_points)
    ):
        raise IndexError("Fixed-eval pack contains an out-of-scene point index")
    if proposal_member_offsets is not None:
        assert proposal_member_indices is not None
        assert pair_proposal_indices is not None
        assert eligible_indices is not None
        if (
            proposal_member_offsets.ndim != 1
            or proposal_member_offsets.numel() < 2
            or int(proposal_member_offsets[0]) != 0
            or bool((proposal_member_offsets[1:] < proposal_member_offsets[:-1]).any())
            or int(proposal_member_offsets[-1])
            != int(proposal_member_indices.numel())
            or pair_proposal_indices.shape != (expected_record_pairs,)
            or pair_proposal_indices.numel() == 0
            or proposal_member_indices.numel() == 0
            or int(proposal_member_indices.min()) < 0
            or int(proposal_member_indices.max()) >= int(num_scene_points)
            or eligible_indices.numel() == 0
            or int(eligible_indices.min()) < 0
            or int(eligible_indices.max()) >= int(num_scene_points)
            or int(pair_proposal_indices.min()) < 0
            or int(pair_proposal_indices.max())
            >= int(proposal_member_offsets.numel()) - 1
        ):
            raise ValueError("Fixed-eval proposal-membership contract drift")
    if all(token_pair_presence):
        assert token_positive_pair_offsets is not None
        assert token_positive_pair_indices is not None
        assert token_positive_pair_digest is not None
        if proposal_member_offsets is None or pair_proposal_indices is None:
            raise ValueError(
                "Native-dec0 pair payload requires proposal occurrence metadata"
            )
        validate_token_positive_pair_payload(
            token_positive_pair_offsets,
            token_positive_pair_indices,
            token_positive_pair_digest,
            proposal_count=int(proposal_member_offsets.numel()) - 1,
            algorithm_version=algorithm_version,
            preselection_policy=SAMPLER_PRESELECTION_POLICY,
            proposal_labels=proposal_labels,
            pair_proposal_indices=pair_proposal_indices,
        )
    if relation_proposal_offsets is not None:
        assert relation_proposal_member_indices is not None
        if (
            relation_proposal_offsets.ndim != 1
            or relation_proposal_offsets.numel() < 2
            or int(relation_proposal_offsets[0]) != 0
            or bool((relation_proposal_offsets[1:] < relation_proposal_offsets[:-1]).any())
            or int(relation_proposal_offsets[-1])
            != int(relation_proposal_member_indices.numel())
            or relation_proposal_member_indices.numel() == 0
            or int(relation_proposal_member_indices.min()) < 0
            or int(relation_proposal_member_indices.max())
            >= int(num_scene_points)
        ):
            raise ValueError("Fixed-eval full relation CSR contract drift")
    return ContrastiveTripletBatch(
        positive_pairs=positive_pairs,
        negative_indices=negative_indices,
        proposal_labels=proposal_labels,
        feature_candidate_indices=feature_candidates,
        num_feature_hard_negatives=feature_hard_count,
        proposal_member_offsets=proposal_member_offsets,
        proposal_member_indices=proposal_member_indices,
        pair_proposal_indices=pair_proposal_indices,
        eligible_indices=eligible_indices,
        relation_proposal_offsets=relation_proposal_offsets,
        relation_proposal_member_indices=(
            relation_proposal_member_indices
        ),
        multiscale_seed=(
            int(record["multiscale_seed"])
            if record.get("multiscale_seed") is not None
            else None
        ),
        num_uniform_negatives=int(
            record.get("num_uniform_negatives", sampling.uniform_negatives)
        ),
        num_spatial_hard_negatives=int(
            record.get(
                "num_spatial_hard_negatives",
                sampling.spatial_hard_negatives,
            )
        ),
        spatial_candidate_pool=int(
            record.get("spatial_candidate_pool", sampling.spatial_candidate_pool)
        ),
        feature_candidate_pool=int(
            stored_feature_candidate_pool
        ),
        algorithm_version=algorithm_version,
        deferred_token_loss=deferred_token_loss,
        require_negative_proposal_membership=require_negative_proposal_membership,
        token_positive_pair_offsets=token_positive_pair_offsets,
        token_positive_pair_indices=token_positive_pair_indices,
        token_positive_pair_digest=token_positive_pair_digest,
    )


def _validate_token_pure_fixed_record(
    *,
    scene: SceneSource,
    record: Mapping[str, Any],
    batch: ContrastiveTripletBatch,
) -> None:
    """Verify a saved V2 group against immutable same-frame source masks."""

    if not batch.require_negative_proposal_membership:
        raise ValueError(
            f"{scene.scene_id}: token-pure fixed record permits unknown negatives"
        )
    if (
        int(batch.num_feature_hard_negatives) > 0
        and batch.feature_candidate_indices is None
        and batch.deferred_token_loss is None
    ):
        raise ValueError(
            f"{scene.scene_id}: token-pure fixed record lacks deferred feature candidates"
        )

    cell = str(record.get("cell"))
    if cell not in TWO_D_CELLS:
        raise ValueError(
            f"{scene.scene_id}: token-pure fixed record has cell {cell!r}"
        )
    frame_id = str(record.get("frame_id", ""))
    if not frame_id or str(record.get("group")) != f"2d/{frame_id}":
        raise ValueError(
            f"{scene.scene_id}/{cell}: token-pure fixed frame identity drift"
        )
    granularity = cell.split("/", 1)[1]
    selected_frames = tuple(
        frame
        for frame in scene.frames(granularity, "heldout")
        if frame.physical_frame_id == frame_id
    )
    if len(selected_frames) != 1:
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: expected one heldout frame"
        )
    selected_frame = selected_frames[0]

    relation_members: list[torch.Tensor] = []
    relation_offsets = [0]
    for relation_cell in TWO_D_CELLS:
        relation_granularity = relation_cell.split("/", 1)[1]
        matches = tuple(
            frame
            for frame in scene.frames(relation_granularity, "heldout")
            if frame.physical_frame_id == frame_id
        )
        if len(matches) > 1:
            raise ValueError(
                f"{scene.scene_id}/{frame_id}: ambiguous same-frame relation "
                f"for {relation_cell}"
            )
        if not matches:
            # Source packs need not contain the same physical camera in every
            # granularity cell.  The sampler's complete relation is all masks
            # that are actually available for this physical frame; uncovered
            # points remain outside eligible_indices and therefore unknown.
            continue
        frame = matches[0]
        for proposal_index in range(int(frame.proposal_offsets.size) - 1):
            start = int(frame.proposal_offsets[proposal_index])
            end = int(frame.proposal_offsets[proposal_index + 1])
            members = torch.unique(
                torch.as_tensor(
                    frame.proposal_point_indices[start:end], dtype=torch.long
                ),
                sorted=True,
            )
            relation_members.append(members)
            relation_offsets.append(
                relation_offsets[-1] + int(members.numel())
            )
    expected_relation_offsets = torch.tensor(relation_offsets, dtype=torch.long)
    expected_relation_members = torch.cat(relation_members, dim=0)
    if batch.relation_proposal_offsets is None or (
        batch.relation_proposal_member_indices is None
    ):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: full relation CSR missing"
        )
    if not torch.equal(
        batch.relation_proposal_offsets.detach().cpu().long(),
        expected_relation_offsets,
    ) or not torch.equal(
        batch.relation_proposal_member_indices.detach().cpu().long(),
        expected_relation_members,
    ):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: full relation CSR drift"
        )

    visible = torch.unique(
        torch.as_tensor(selected_frame.visible_indices, dtype=torch.long),
        sorted=True,
    )
    known = visible[
        torch.isin(visible, torch.unique(expected_relation_members, sorted=True))
    ]
    if batch.eligible_indices is None or not torch.equal(
        batch.eligible_indices.detach().cpu().long(), known
    ):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: known-support eligibility drift"
        )

    offsets = batch.proposal_member_offsets
    members = batch.proposal_member_indices
    pair_proposals = batch.pair_proposal_indices
    if offsets is None or members is None or pair_proposals is None:
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: selected proposal CSR missing"
        )
    offsets = offsets.detach().cpu().long()
    members = members.detach().cpu().long()
    pair_proposals = pair_proposals.detach().cpu().long()
    labels = batch.proposal_labels.detach().cpu().long()
    payload_fields = (
        batch.token_positive_pair_offsets,
        batch.token_positive_pair_indices,
        batch.token_positive_pair_digest,
    )
    payload_presence = tuple(value is not None for value in payload_fields)
    pair_policy = record["batch"].get("native_dec0_pair_selection")
    if pair_policy != TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION:
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: fixed native-token pair policy drift"
        )
    if any(payload_presence):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: backend token IDs were serialized"
        )
    for occurrence in range(int(offsets.numel()) - 1):
        rows = torch.where(pair_proposals == occurrence)[0]
        if rows.numel() == 0:
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: selected occurrence has no row"
            )
        proposal_labels = torch.unique(labels[rows], sorted=True)
        if proposal_labels.numel() != 1:
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: occurrence label drift"
            )
        proposal_index = int(proposal_labels[0])
        if not 0 <= proposal_index < int(selected_frame.proposal_offsets.size) - 1:
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: proposal index drift"
            )
        source_start = int(selected_frame.proposal_offsets[proposal_index])
        source_end = int(selected_frame.proposal_offsets[proposal_index + 1])
        expected_members = torch.unique(
            torch.as_tensor(
                selected_frame.proposal_point_indices[source_start:source_end],
                dtype=torch.long,
            ),
            sorted=True,
        )
        stored_members = members[int(offsets[occurrence]) : int(offsets[occurrence + 1])]
        if not torch.equal(stored_members, expected_members):
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: selected proposal CSR drift"
            )


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_fixed_holdout_eval_pack(
    *,
    pack_dir: Path,
    scenes: list[SceneSource],
    sampling: SamplingConfig,
    sampling_profile: str,
    seed: int,
    verify_scene_hashes: bool = True,
    cells: Sequence[str] | None = None,
) -> dict[str, Any]:
    active_cells = normalize_source_cells(cells)
    token_pure_fixed_pack = tuple(active_cells) == TWO_D_CELLS
    coordinate_policies = {
        str(scene.coordinate_normalization_policy) for scene in scenes
    }
    if len(coordinate_policies) != 1:
        raise ValueError(
            "Fixed-eval scenes must share one coordinate-normalization policy: "
            f"{sorted(coordinate_policies)}"
        )
    coordinate_policy = next(iter(coordinate_policies))
    manifest_path = (pack_dir / "manifest.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FIXED_EVAL_PACK_SCHEMA:
        raise ValueError(f"Fixed-eval pack schema drift: {manifest_path}")
    if manifest.get("sampling_profile") != sampling_profile:
        raise ValueError(f"Fixed-eval sampling-profile drift: {manifest_path}")
    if manifest.get("sampling") != vars(sampling):
        raise ValueError(f"Fixed-eval sampling contract drift: {manifest_path}")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"Fixed-eval seed drift: {manifest_path}")
    if manifest.get("coordinate_normalization_policy", "native") != coordinate_policy:
        raise ValueError(
            f"Fixed-eval coordinate-normalization drift: {manifest_path}"
        )
    manifest_cells = manifest.get("source_cells")
    if manifest_cells is None:
        manifest_cells = list(SOURCE_CELLS)
    if list(manifest_cells) != list(active_cells):
        raise ValueError(
            f"Fixed-eval source-cell drift: expected {list(active_cells)}, "
            f"got {list(manifest_cells)}"
        )
    token_contract = manifest.get("token_pure_dec0_fixed_pack")
    if token_pure_fixed_pack:
        if not isinstance(token_contract, dict) or token_contract != {
            "schema_version": TOKEN_PURE_FIXED_EVAL_SCHEMA,
            "enabled": True,
            "source_cells": list(TWO_D_CELLS),
            "sampler": "coverage_multigranular_v2",
            "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
            "sampler_preselection_policy": SAMPLER_PRESELECTION_POLICY,
            "frame_local": True,
            "relation_csr": "same_physical_frame_all_g02_g05_g08_masks",
            "eligible_indices": "visible_intersection_known_relation_members",
            "hierarchy_route_eligibility": (
                "not_required_for_dec0_feature_evaluation"
            ),
            "native_token_feature_candidates": (
                TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY
            ),
            "fixed_comparable_feature_hard_negatives": 0,
            "native_dec0_pair_selection": TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION,
        }:
            raise ValueError(
                "V2 token-pure fixed-eval contract is absent or drifted; "
                "legacy dense packs are not token-safe"
            )
    expected_scenes = [
        {
            "scene_id": scene.scene_id,
            "source_manifest_sha256": scene.manifest_sha256,
        }
        for scene in scenes
    ]
    observed_scenes = [
        {
            "scene_id": row.get("scene_id"),
            "source_manifest_sha256": row.get("source_manifest_sha256"),
        }
        for row in manifest.get("scenes", [])
    ]
    if observed_scenes != expected_scenes:
        raise ValueError(f"Fixed-eval scene/source ordering drift: {manifest_path}")
    total_records = 0
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    for row in manifest["scenes"]:
        scene_id = str(row["scene_id"])
        scene = scene_by_id[scene_id]
        expected_file = f"{scene_id}.pt"
        if row.get("file") != expected_file:
            raise ValueError(
                f"Fixed-eval scene filename drift for {scene_id}: "
                f"{row.get('file')!r}"
            )
        scene_path = (pack_dir / row["file"]).resolve(strict=True)
        if verify_scene_hashes and sha256_file(scene_path) != row["sha256"]:
            raise ValueError(f"Fixed-eval scene artifact hash drift: {scene_path}")
        payload = torch.load(scene_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema_version") != FIXED_EVAL_SCENE_SCHEMA
            or payload.get("scene_id") != scene_id
            or payload.get("source_manifest_sha256")
            != scene.manifest_sha256
            or int(payload.get("num_scene_points", -1))
            != int(scene.points.shape[0])
            or payload.get("coordinate_normalization_policy", "native")
            != scene.coordinate_normalization_policy
            or bool(payload.get("token_pure_dec0_fixed_pack", False))
            != token_pure_fixed_pack
        ):
            raise ValueError(f"Fixed-eval scene payload drift: {scene_path}")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Fixed-eval scene records missing: {scene_path}")
        expected_record_count = (
            int(row["record_count"])
            if token_pure_fixed_pack
            else _expected_holdout_record_count(scene, cells=active_cells)
        )
        if (
            len(records) != expected_record_count
            or int(row["record_count"]) != expected_record_count
        ):
            raise ValueError(f"Fixed-eval scene record-count drift: {scene_path}")
        observed_cells: set[str] = set()
        for record in records:
            cell = str(record.get("cell"))
            if cell not in active_cells:
                raise ValueError(
                    f"Fixed-eval scene contains an unknown cell: {scene_path}"
                )
            observed_cells.add(cell)
            batch = _triplet_batch_from_record(
                record["batch"],
                device=torch.device("cpu"),
                include_feature_hard=True,
                sampling=sampling,
                num_scene_points=int(scene.points.shape[0]),
            )
            if token_pure_fixed_pack:
                if not bool(record["batch"].get("token_pure_fixed_pack", False)):
                    raise ValueError(
                        f"V2 fixed-eval record is not token-safe: {scene_path}"
                    )
                _validate_token_pure_fixed_record(
                    scene=scene,
                    record=record,
                    batch=batch,
                )
        if observed_cells != set(active_cells):
            raise ValueError(f"Fixed-eval scene cell coverage drift: {scene_path}")
        total_records += expected_record_count
    checks = manifest.get("checks", {})
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError(f"Fixed-eval pack checks failed: {manifest_path}")
    return {
        "schema_version": manifest["schema_version"],
        "path": str(pack_dir.resolve(strict=True)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sampling_profile": sampling_profile,
        "sampling": vars(sampling),
        "seed": int(seed),
        "coordinate_normalization_policy": coordinate_policy,
        "scene_count": len(expected_scenes),
        "record_count": total_records,
        "selection_view": "fixed_comparable",
        "fixed_negatives_per_pair": int(sampling.uniform_negatives)
        + int(sampling.spatial_hard_negatives),
        "adaptive_feature_hard_negatives_per_pair": int(
            sampling.feature_hard_negatives
        ),
        "token_pure_dec0_fixed_pack": token_contract,
        "passed": True,
    }


def prepare_fixed_holdout_eval_pack(
    *,
    pack_dir: Path,
    scenes: list[SceneSource],
    sampling: SamplingConfig,
    sampling_profile: str,
    seed: int,
    cells: Sequence[str] | None = None,
    precomputed_token_plans: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_cells = normalize_source_cells(cells)
    token_pure_fixed_pack = tuple(active_cells) == TWO_D_CELLS
    coordinate_policies = {
        str(scene.coordinate_normalization_policy) for scene in scenes
    }
    if len(coordinate_policies) != 1:
        raise ValueError(
            "Fixed-eval scenes must share one coordinate-normalization policy: "
            f"{sorted(coordinate_policies)}"
        )
    coordinate_policy = next(iter(coordinate_policies))
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.exists():
        return validate_fixed_holdout_eval_pack(
            pack_dir=pack_dir,
            scenes=scenes,
            sampling=sampling,
            sampling_profile=sampling_profile,
            seed=seed,
            cells=active_cells,
        )
    if pack_dir.exists() and any(pack_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite incomplete fixed-eval pack: {pack_dir}"
        )
    pack_dir.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, Any]] = []
    expected_pairs = (
        int(sampling.proposals_per_forward)
        * int(sampling.positive_pairs_per_proposal)
    )
    for scene in scenes:
        points = torch.from_numpy(scene.points)
        plan_audit: dict[str, Any] | None = None
        if token_pure_fixed_pack:
            plan = (
                precomputed_token_plans.get(scene.scene_id)
                if precomputed_token_plans is not None
                else None
            )
            if plan is None:
                plan = _token_pure_fixed_holdout_plan(
                    scene=scene,
                    points=points,
                    device=torch.device("cpu"),
                    sampling=sampling,
                    seed=seed,
                )
            if {group.cell for group in plan.groups} != set(TWO_D_CELLS):
                raise ValueError(
                    f"{scene.scene_id}: supplied V2 fixed-eval plan does not "
                    "cover all three source cells"
                )
            plan_audit = plan.audit_metadata()
            records = [
                {
                    "cell": group.cell,
                    "group": f"2d/{group.frame_id}",
                    "frame_id": group.frame_id,
                    "selected_proposal_indices": list(
                        group.selected_proposal_indices
                    ),
                    "coverage_metadata": dict(group.coverage_metadata),
                    "batch": _triplet_batch_record(
                        group.batch,
                        token_pure_fixed_pack=True,
                        fixed_runtime_token_pair_rebuild=True,
                    ),
                }
                for group in plan.groups
            ]
        else:
            batches = _holdout_batches(
                scene=scene,
                points=points,
                device=torch.device("cpu"),
                sampling=sampling,
                seed=seed,
                cells=active_cells,
            )
            records = [
                {
                    "cell": cell,
                    "group": group,
                    "batch": _triplet_batch_record(batch),
                }
                for cell, group, batch in batches
            ]
        cells_present = {str(record["cell"]) for record in records}
        if cells_present != set(active_cells):
            raise RuntimeError(
                f"{scene.scene_id}: fixed-eval cell coverage drift: {sorted(cells_present)}"
            )
        scene_path = pack_dir / f"{scene.scene_id}.pt"
        _atomic_torch_save(
            {
                "schema_version": FIXED_EVAL_SCENE_SCHEMA,
                "scene_id": scene.scene_id,
                "source_manifest_sha256": scene.manifest_sha256,
                "coordinate_normalization_policy": (
                    scene.coordinate_normalization_policy
                ),
                "num_scene_points": int(scene.points.shape[0]),
                "token_pure_dec0_fixed_pack": token_pure_fixed_pack,
                "token_pure_dec0_plan": plan_audit,
                "records": records,
            },
            scene_path,
        )
        scene_rows.append(
            {
                "scene_id": scene.scene_id,
                "source_manifest_sha256": scene.manifest_sha256,
                "file": scene_path.name,
                "sha256": sha256_file(scene_path),
                "record_count": len(records),
            }
        )
    manifest = {
        "schema_version": FIXED_EVAL_PACK_SCHEMA,
        "sampling_profile": sampling_profile,
        "sampling": vars(sampling),
        "seed": int(seed),
        "coordinate_normalization_policy": coordinate_policy,
        "source_cells": list(active_cells),
        "token_pure_dec0_fixed_pack": (
            {
                "schema_version": TOKEN_PURE_FIXED_EVAL_SCHEMA,
                "enabled": True,
                "source_cells": list(TWO_D_CELLS),
                "sampler": "coverage_multigranular_v2",
                "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
                "sampler_preselection_policy": SAMPLER_PRESELECTION_POLICY,
                "frame_local": True,
                "relation_csr": (
                    "same_physical_frame_all_g02_g05_g08_masks"
                ),
                "eligible_indices": (
                    "visible_intersection_known_relation_members"
                ),
                "hierarchy_route_eligibility": (
                    "not_required_for_dec0_feature_evaluation"
                ),
                "native_token_feature_candidates": (
                    TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY
                ),
                "fixed_comparable_feature_hard_negatives": 0,
                "native_dec0_pair_selection": (
                    TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
                ),
            }
            if token_pure_fixed_pack
            else None
        ),
        "views": {
            "fixed_comparable": {
                "negative_identity": "byte-identical_across_representations",
                "uniform_negatives": int(sampling.uniform_negatives),
                "spatial_hard_negatives": int(sampling.spatial_hard_negatives),
                "feature_hard_negatives": 0,
            },
            "adaptive_feature_hard": {
                "base_negative_identity": "byte-identical_across_representations",
                "feature_candidate_identity": "byte-identical_across_representations",
                "selected_feature_hard_identity": "representation_specific",
                "feature_hard_negatives": int(sampling.feature_hard_negatives),
            },
        },
        "scenes": scene_rows,
        "checks": {
            "scene_order_exact": len(scene_rows) == len(scenes),
            "all_active_cells_per_scene": True,
            **(
                {"all_six_cells_per_scene": True}
                if tuple(active_cells) == SOURCE_CELLS
                else {}
            ),
            **(
                {
                    "token_pure_three_cells_exact": True,
                    "frame_local_relation_csr_recorded": True,
                    "known_support_eligibility_recorded": True,
                }
                if token_pure_fixed_pack
                else {}
            ),
            "positive_pairs_per_record_exact": expected_pairs > 0,
            "base_negative_count_exact": (
                int(sampling.uniform_negatives)
                + int(sampling.spatial_hard_negatives)
                > 0
            ),
            "feature_candidate_pool_recorded": (
                int(sampling.feature_candidate_pool) > 0
            ),
        },
    }
    _atomic_json_write(manifest, manifest_path)
    return validate_fixed_holdout_eval_pack(
        pack_dir=pack_dir,
        scenes=scenes,
        sampling=sampling,
        sampling_profile=sampling_profile,
        seed=seed,
        cells=active_cells,
    )


def _fixed_scene_batch_views(
    *,
    pack_dir: Path,
    scene: SceneSource,
    device: torch.device,
    sampling: SamplingConfig,
) -> dict[str, list[tuple[str, str, ContrastiveTripletBatch]]]:
    scene_path = pack_dir / f"{scene.scene_id}.pt"
    payload = torch.load(scene_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != FIXED_EVAL_SCENE_SCHEMA
        or payload.get("scene_id") != scene.scene_id
        or payload.get("source_manifest_sha256") != scene.manifest_sha256
        or int(payload.get("num_scene_points", -1)) != int(scene.points.shape[0])
        or payload.get("coordinate_normalization_policy", "native")
        != scene.coordinate_normalization_policy
    ):
        raise ValueError(f"Fixed-eval scene payload drift: {scene_path}")
    output: dict[str, list[tuple[str, str, ContrastiveTripletBatch]]] = {
        "fixed_comparable": [],
        "adaptive_feature_hard": [],
    }
    for record in payload["records"]:
        for view, include_feature_hard in (
            ("fixed_comparable", False),
            ("adaptive_feature_hard", True),
        ):
            output[view].append(
                (
                    str(record["cell"]),
                    str(record["group"]),
                    _triplet_batch_from_record(
                        record["batch"],
                        device=device,
                        include_feature_hard=include_feature_hard,
                        sampling=sampling,
                        num_scene_points=int(scene.points.shape[0]),
                    ),
                )
            )
    return output


def feature_health(features: torch.Tensor, *, max_samples: int = 5000) -> dict[str, float]:
    with torch.no_grad():
        sample = features[:max_samples].float()
        centered = sample - sample.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False)
        covariance = centered.T @ centered / max(int(centered.shape[0]), 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        norms = torch.linalg.vector_norm(sample, dim=1)
        return {
            "feature_std_mean": float(std.mean()),
            "feature_std_min": float(std.min()),
            "effective_rank": float(entropy.exp()),
            "feature_norm_mean": float(norms.mean()),
            "feature_norm_std": float(norms.std(unbiased=False)),
            "feature_norm_min": float(norms.min()),
            "feature_norm_max": float(norms.max()),
        }


def evaluate_holdout(
    *,
    model: LitePTContrastiveModel,
    scenes: list[SceneSource],
    device: torch.device,
    sampling: SamplingConfig,
    seed: int,
    fixed_eval_pack: Path | None = None,
    fixed_eval_pack_report: dict[str, Any] | None = None,
    feature_spaces: tuple[str, ...] = ("native", "projected"),
    cells: Sequence[str] | None = None,
) -> dict[str, Any]:
    active_cells = normalize_source_cells(cells)
    if not feature_spaces or any(
        name not in ("native", "projected") for name in feature_spaces
    ):
        raise ValueError(f"Invalid evaluation feature spaces: {feature_spaces}")
    model.eval()
    metric_names = (
        "loss_total",
        "triplet_ranking_accuracy",
        "hardest_negative_ranking_accuracy",
        "cosine_gap",
    )
    records: list[dict[str, Any]] = []
    health_rows: list[dict[str, float]] = []
    multiscale_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    multiscale_mapping_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    multiscale_native_health: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    multiscale_projected_health: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    token_pure_mapping_records: list[dict[str, Any]] = []
    token_pure_evaluation = bool(
        fixed_eval_pack_report is not None
        and isinstance(
            fixed_eval_pack_report.get("token_pure_dec0_fixed_pack"), dict
        )
    )
    selection_view = (
        "fixed_comparable" if fixed_eval_pack is not None else "adaptive_feature_hard"
    )
    with torch.no_grad():
        for scene in scenes:
            forward_seed = stable_seed(seed, scene.scene_id, "eval-forward")
            random.seed(forward_seed)
            np.random.seed(forward_seed % (2**32))
            torch.manual_seed(forward_seed)
            points = torch.from_numpy(scene.points).to(device)
            features = torch.from_numpy(scene.features).to(device)
            (
                backbone_output,
                native,
                projected,
                level_embeddings_by_name,
            ) = model.encode_with_multiscale(
                points, features, token_space=token_pure_evaluation
            )
            native = (
                backbone_output.scene_tokens
                if token_pure_evaluation
                else native
            ).float()
            projected = projected.float()
            health_rows.append(feature_health(native))
            if model.multiscale_supervision:
                for name, tokens in zip(
                    MULTISCALE_LEVEL_NAMES,
                    backbone_output.multi_scale_tokens,
                    strict=True,
                ):
                    multiscale_native_health[name].append(
                        feature_health(tokens.float())
                    )
                    multiscale_projected_health[name].append(
                        feature_health(level_embeddings_by_name[name].float())
                    )
            if fixed_eval_pack is None:
                batch_views = {
                    "adaptive_feature_hard": _holdout_batches(
                        scene=scene,
                        points=points,
                        device=device,
                        sampling=sampling,
                        seed=seed,
                        cells=active_cells,
                    )
                }
            else:
                batch_views = _fixed_scene_batch_views(
                    pack_dir=fixed_eval_pack,
                    scene=scene,
                    device=device,
                    sampling=sampling,
                )
            for view, batches in batch_views.items():
                for cell, group, batch in batches:
                    evaluation_batch = batch
                    token_mapping: dict[str, Any] | None = None
                    if token_pure_evaluation:
                        evaluation_batch, token_mapping = (
                            build_token_pure_dec0_contrastive_batch(
                                batch,
                                backbone_output.inverse_map.long(),
                                backbone_output.scene_xyz,
                                max_positive_pairs_per_proposal=int(
                                    sampling.positive_pairs_per_proposal
                                ),
                            )
                        )
                        if view == "fixed_comparable":
                            token_pure_mapping_records.append(
                                {
                                    "scene_id": scene.scene_id,
                                    "cell": cell,
                                    "group": group,
                                    **token_mapping,
                                }
                            )
                        if evaluation_batch is None:
                            continue
                    embeddings_by_space = {
                        "native": native,
                        "projected": projected,
                    }
                    for feature_space in feature_spaces:
                        embeddings = embeddings_by_space[feature_space]
                        result = model.criterion(embeddings, evaluation_batch)
                        records.append(
                            {
                                "scene_id": scene.scene_id,
                                "cell": cell,
                                "group": group,
                                "view": view,
                                "feature_space": feature_space,
                                # Keep the denominator beside every metric:
                                # fixed V2 groups can contain different numbers
                                # of dec0-valid pairs after token remapping.
                                "positive_pair_denominator": int(
                                    evaluation_batch.num_pairs
                                ),
                                **{
                                    name: float(result[name])
                                    for name in metric_names
                                },
                            }
                        )
                    if model.multiscale_supervision:
                        for name, level_xyz, parent_map in zip(
                            MULTISCALE_LEVEL_NAMES,
                            backbone_output.multi_scale_xyz,
                            backbone_output.finest_to_multi_scale,
                            strict=True,
                        ):
                            level_batch, mapping_stats = (
                                build_pure_token_contrastive_batch(
                                    batch,
                                    parent_map[backbone_output.inverse_map.long()],
                                    level_xyz,
                                    level_name=name,
                                )
                            )
                            multiscale_mapping_records[name].append(
                                {
                                    key: float(value)
                                    for key, value in mapping_stats.items()
                                }
                            )
                            if level_batch is None:
                                continue
                            level_result = model.multiscale_criteria[name](
                                level_embeddings_by_name[name].float(),
                                level_batch,
                            )
                            multiscale_records[name].append(
                                {
                                    "loss": float(level_result["loss_total"]),
                                    "hardest_negative_ranking_accuracy": float(
                                        level_result[
                                            "hardest_negative_ranking_accuracy"
                                        ]
                                    ),
                                    "cosine_gap": float(level_result["cosine_gap"]),
                                    "temperature": float(
                                        level_result["temperature"]
                                    ),
                                }
                            )
            del (
                points,
                features,
                native,
                projected,
                backbone_output,
                level_embeddings_by_name,
            )

    view_metrics: dict[str, dict[str, Any]] = {}
    for view in sorted({str(row["view"]) for row in records}):
        cell_metrics: dict[str, dict[str, dict[str, float]]] = {}
        scene_cell_metrics: dict[
            str, dict[str, dict[str, dict[str, float]]]
        ] = {}
        macro: dict[str, dict[str, float]] = {}
        for feature_space in feature_spaces:
            cell_metrics[feature_space] = {}
            scene_cell_metrics[feature_space] = {
                scene.scene_id: {} for scene in scenes
            }
            for cell in active_cells:
                selected = [
                    row
                    for row in records
                    if (
                        row["view"] == view
                        and row["feature_space"] == feature_space
                        and row["cell"] == cell
                    )
                ]
                if not selected:
                    raise RuntimeError(
                        f"No holdout records for {view}/{feature_space}/{cell}"
                    )
                per_scene = {
                    scene.scene_id: [
                        row
                        for row in selected
                        if row["scene_id"] == scene.scene_id
                    ]
                    for scene in scenes
                }
                if any(not rows for rows in per_scene.values()) and not token_pure_evaluation:
                    missing = [
                        scene_id
                        for scene_id, rows in per_scene.items()
                        if not rows
                    ]
                    raise RuntimeError(
                        f"No holdout records for {view}/{feature_space}/{cell}: "
                        f"{missing}"
                    )
                for scene_id, rows in per_scene.items():
                    if not rows:
                        continue
                    scene_cell_metrics[feature_space][scene_id][cell] = {
                        name: float(np.mean([row[name] for row in rows]))
                        for name in metric_names
                    }
                cell_metrics[feature_space][cell] = {
                    name: float(
                        np.mean(
                            [
                                scene_cell_metrics[feature_space][scene.scene_id][cell][name]
                                for scene in scenes
                                if cell
                                in scene_cell_metrics[feature_space][scene.scene_id]
                            ]
                        )
                    )
                    for name in metric_names
                }
                cell_metrics[feature_space][cell][
                    "min_groups_per_scene"
                ] = float(min(len(rows) for rows in per_scene.values()))
                cell_metrics[feature_space][cell][
                    "max_groups_per_scene"
                ] = float(max(len(rows) for rows in per_scene.values()))
            hardest = {
                cell: cell_metrics[feature_space][cell][
                    "hardest_negative_ranking_accuracy"
                ]
                for cell in active_cells
            }
            two_d_values = [
                value
                for cell, value in hardest.items()
                if cell.startswith("2d/")
            ]
            three_d_values = [
                value
                for cell, value in hardest.items()
                if cell.startswith("3d/")
            ]
            family_macro: dict[str, float] = {
                "overall": float(np.mean(list(hardest.values()))),
            }
            if two_d_values:
                family_macro["2d"] = float(np.mean(two_d_values))
            if three_d_values:
                family_macro["3d"] = float(np.mean(three_d_values))
            macro[feature_space] = family_macro
        view_metrics[view] = {
            "native_selection_macro": macro["native"],
            "projected_diagnostic_macro": macro.get("projected"),
            "cell_metrics": cell_metrics,
            "scene_cell_metrics": scene_cell_metrics,
        }
    health = {
        key: float(
            np.min([row[key] for row in health_rows])
            if key in ("feature_std_min", "feature_norm_min")
            else np.max([row[key] for row in health_rows])
            if key == "feature_norm_max"
            else np.mean([row[key] for row in health_rows])
        )
        for key in health_rows[0]
    }
    model.train()
    cell_noun = (
        "six-cell"
        if tuple(active_cells) == SOURCE_CELLS
        else "2d-three-cell"
        if tuple(active_cells) == TWO_D_CELLS
        else f"{len(active_cells)}-cell"
    )
    selected_metrics = view_metrics[selection_view]
    fixed_dec0_cells = tuple(TWO_D_CELLS)
    fixed_dec0_cell_metrics: dict[str, dict[str, Any]] = {}
    for cell in fixed_dec0_cells:
        mapping_rows = [
            row for row in token_pure_mapping_records if row["cell"] == cell
        ]
        rows = [
            row
            for row in records
            if row["view"] == "fixed_comparable"
            and row["feature_space"] == "native"
            and row["cell"] == cell
        ]
        denominator = sum(int(row["positive_pair_denominator"]) for row in rows)
        requested_pairs = sum(
            sum(
                int(proposal.get("requested_token_pairs", 0))
                for proposal in row.get("proposals", ())
            )
            for row in mapping_rows
        )
        realized_pairs = sum(
            int(row.get("realized_positive_pairs", 0)) for row in mapping_rows
        )
        valid_groups = sum(
            int(int(row.get("realized_positive_pairs", 0)) > 0)
            for row in mapping_rows
        )
        attempted_groups = len(mapping_rows)
        valid_group_fraction = valid_groups / max(attempted_groups, 1)
        fixed_dec0_cell_metrics[cell] = {
            "cosine_gap": (
                float(
                    sum(
                        float(row["cosine_gap"])
                        * int(row["positive_pair_denominator"])
                        for row in rows
                    )
                    / denominator
                )
                if denominator
                else 0.0
            ),
            "denominator": int(denominator),
            "requested_positive_pairs": int(requested_pairs),
            "realized_positive_pairs": int(realized_pairs),
            "pair_yield_fraction": realized_pairs / max(requested_pairs, 1),
            "attempted_groups": attempted_groups,
            "valid_groups": valid_groups,
            "valid_group_fraction": valid_group_fraction,
            "valid": bool(
                token_pure_evaluation
                and rows
                and denominator > 0
                and attempted_groups > 0
                and valid_group_fraction >= 0.95
            ),
        }
    fixed_dec0_valid = all(
        fixed_dec0_cell_metrics[cell]["valid"] for cell in fixed_dec0_cells
    )
    fixed_dec0_macro_cosine_gap = (
        float(
            np.mean(
                [
                    fixed_dec0_cell_metrics[cell]["cosine_gap"]
                    for cell in fixed_dec0_cells
                ]
            )
        )
        if fixed_dec0_valid
        else 0.0
    )
    fixed_dec0_report = {
        "schema_version": "litept_token_pure_dec0_fixed_eval/v1",
        "exact_native_dec0_token_space": token_pure_evaluation,
        "expected_cells": list(fixed_dec0_cells),
        "valid": fixed_dec0_valid,
        "macro_cosine_gap": fixed_dec0_macro_cosine_gap,
        "denominator": int(
            sum(
                fixed_dec0_cell_metrics[cell]["denominator"]
                for cell in fixed_dec0_cells
            )
        ),
        "denominators_by_cell": {
            cell: fixed_dec0_cell_metrics[cell]["denominator"]
            for cell in fixed_dec0_cells
        },
        "cells": fixed_dec0_cell_metrics,
    }
    multiscale_diagnostics: dict[str, dict[str, Any]] = {}
    for name in MULTISCALE_LEVEL_NAMES:
        mapping_rows = multiscale_mapping_records[name]
        valid_rows = multiscale_records[name]
        if not mapping_rows:
            continue

        def aggregate_health(rows: list[dict[str, float]]) -> dict[str, float]:
            if not rows:
                return {}
            return {
                key: float(
                    np.min([row[key] for row in rows])
                    if key in ("feature_std_min", "feature_norm_min")
                    else np.max([row[key] for row in rows])
                    if key == "feature_norm_max"
                    else np.mean([row[key] for row in rows])
                )
                for key in rows[0]
            }

        multiscale_diagnostics[name] = {
            "attempted_batches": len(mapping_rows),
            "valid_batches": len(valid_rows),
            "valid_batch_fraction": len(valid_rows) / len(mapping_rows),
            "conditional_metrics": (
                {
                    key: float(np.mean([row[key] for row in valid_rows]))
                    for key in valid_rows[0]
                }
                if valid_rows
                else {}
            ),
            "mapping": {
                key: float(np.mean([row[key] for row in mapping_rows]))
                for key in mapping_rows[0]
            },
            "native_health": aggregate_health(multiscale_native_health[name]),
            "projected_health": aggregate_health(
                multiscale_projected_health[name]
            ),
        }
    return {
        "selection_view": selection_view,
        "selection_metric": (
            f"native 72D fixed-comparable {cell_noun} macro hardest-negative ranking"
            if selection_view == "fixed_comparable"
            else f"native 72D adaptive-hard {cell_noun} macro hardest-negative ranking"
        ),
        "native_selection_macro": selected_metrics["native_selection_macro"],
        "native_fixed_comparable_dec0_macro_cosine_gap": fixed_dec0_macro_cosine_gap,
        "native_fixed_comparable_dec0_macro_cosine_gap_valid": fixed_dec0_valid,
        "native_fixed_comparable_dec0_macro_cosine_gap_denominator": fixed_dec0_report[
            "denominator"
        ],
        "native_fixed_comparable_dec0": fixed_dec0_report,
        "projected_diagnostic_macro": selected_metrics.get(
            "projected_diagnostic_macro"
        ),
        "cell_metrics": selected_metrics["cell_metrics"],
        "scene_cell_metrics": selected_metrics["scene_cell_metrics"],
        "view_metrics": view_metrics,
        "native_health": health,
        "multiscale_diagnostics": multiscale_diagnostics,
        "fixed_eval_pack": fixed_eval_pack_report,
        "records": records,
    }


def optimizer_parameter_groups(
    model: LitePTContrastiveModel,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    names: dict[tuple[str, bool], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "backbone" if name.startswith("backbone.") else "head"
        no_decay = bool(
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "log_temperature" in name
        )
        key = (family, no_decay)
        groups.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    output: list[dict[str, Any]] = []
    report: dict[str, list[str]] = {}
    for family in ("backbone", "head"):
        for no_decay in (False, True):
            key = (family, no_decay)
            if key not in groups:
                continue
            lr = float(backbone_lr if family == "backbone" else head_lr)
            group_name = f"{family}/{'no_decay' if no_decay else 'decay'}"
            output.append(
                {
                    "params": groups[key],
                    "lr": lr,
                    "initial_lr": lr,
                    "weight_decay": 0.0 if no_decay else float(weight_decay),
                    "group_name": group_name,
                }
            )
            report[group_name] = names[key]
    return output, report


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in optimizer.param_groups:
        lr = float(group["initial_lr"]) * float(factor)
        group["lr"] = lr
        values[str(group["group_name"])] = lr
    return values
