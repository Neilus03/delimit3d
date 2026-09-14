"""Public PointTransformerV3 bridge for Delimit3D adaptation.

This module keeps the public Pointcept PT-v3m1-0 backbone intact and exposes
the same dense point-feature contract used by the Delimit3D contrastive
sampler.  Voxelization is deliberately deterministic: one lowest-index raw
point represents each 2 cm voxel, and the inverse map is returned alongside
the features.  The public segmentation head is not loaded or trained; a new
small projection head is used only for the Delimit3D objective.

The external PTv3 checkout is kept outside Git.  Its source revision and the
public checkpoint hash are recorded by the adaptation runner's provenance.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    FrameProposalCatalog,
    sample_multigranular_frame_group_plan,
)
from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion
from delimit3d.training.adaptation import (
    ContrastiveProjectionHead,
    SceneSource,
    remap_contrastive_batch_to_level,
    normalize_hierarchy_frame_groups,
    zero_loss_touching_module_parameters,
)


PTV3_SOURCE_COMMIT = "3229e9b7de1770c8ad17c316f8e349982de509f8"
PTV3_HF_REPO = "Pointcept/PointTransformerV3"
PTV3_HF_REVISION = "scannet-semseg-pt-v3m1-0-base"
PTV3_PUBLIC_CHECKPOINT_SHA256 = (
    "40206376ff2f83f48e4d1bc27d5c5d96be7c87c5d11eb45fe7be501959040e7f"
)
PTV3_INPUT_CHANNELS = 6
PTV3_DEFAULT_GRID_SIZE = 0.02
PTV3_DEFAULT_OUTPUT_DIM = 64

# These are the native PTv3 decoder stages at the same coarse-to-fine
# resolutions used by the LitePT auxiliary recipe.  The final dec0 objective
# remains the primary loss; each 2D mask granularity gets one scale-matched
# auxiliary objective.
PTV3_MULTISCALE_LEVEL_NAMES = ("dec3", "dec2", "dec1")
PTV3_MULTISCALE_LEVEL_LOSS_WEIGHTS = {
    "dec3": 0.20,
    "dec2": 0.30,
    "dec1": 0.50,
}
PTV3_MULTISCALE_ROUTE_BY_GRANULARITY = {
    "g02": "dec1",
    "g05": "dec2",
    "g08": "dec3",
}
PTV3_MULTISCALE_AUXILIARY_WEIGHT = 0.4
PTV3_MULTISCALE_WARMUP_UPDATES = 32

# These values mirror Pointcept's public ScanNet PT-v3m1-0-base config.  Keep
# them in one JSON-safe mapping so the resolved experiment records the exact
# architecture rather than relying on constructor defaults that may drift.
PTV3_DEFAULT_CONFIG: dict[str, Any] = {
    "in_channels": 6,
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (2, 2, 2, 6, 2),
    "enc_channels": (32, 64, 128, 256, 512),
    "enc_num_head": (2, 4, 8, 16, 32),
    "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    "dec_depths": (2, 2, 2, 2),
    "dec_channels": (64, 64, 128, 256),
    "dec_num_head": (4, 4, 8, 16),
    "dec_patch_size": (1024, 1024, 1024, 1024),
    "mlp_ratio": 4,
    "qkv_bias": True,
    "qk_scale": None,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "drop_path": 0.3,
    "pre_norm": True,
    "shuffle_orders": True,
    "enable_rpe": False,
    "enable_flash": True,
    "upcast_attention": False,
    "upcast_softmax": False,
    "cls_mode": False,
    "pdnorm_bn": False,
    "pdnorm_ln": False,
    "pdnorm_decouple": True,
    "pdnorm_adaptive": False,
    "pdnorm_affine": True,
    "pdnorm_conditions": ("ScanNet", "S3DIS", "Structured3D"),
}


@dataclass(frozen=True)
class PTv3FeatureOutput:
    """Token and dense raw-point features from one deterministic scene visit."""

    point_features: torch.Tensor  # [N_raw, 64]
    token_features: torch.Tensor  # [N_voxel, 64]
    raw_to_token: torch.Tensor  # [N_raw]
    token_xyz: torch.Tensor  # [N_voxel, 3]
    token_grid: torch.Tensor  # [N_voxel, 3]
    representative_indices: torch.Tensor  # [N_voxel]
    hierarchy_tokens: Mapping[str, torch.Tensor] = field(default_factory=dict)
    hierarchy_xyz: Mapping[str, torch.Tensor] = field(default_factory=dict)
    hierarchy_raw_maps: Mapping[str, torch.Tensor] = field(default_factory=dict)


def representative_first_voxelize(
    points: torch.Tensor,
    features: torch.Tensor,
    grid_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize one scene with stable lowest-raw-index representatives.

    The arithmetic follows the LitePT wrapper: floor coordinates divided by
    the grid size, subtract the per-scene minimum grid coordinate, and sort
    unique rows lexicographically.  A stable raw-index tie break makes the
    representative and inverse map reproducible across runs and devices.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {tuple(points.shape)}")
    if features.ndim != 2 or features.shape[0] != points.shape[0]:
        raise ValueError("features must be [N,C] aligned with points")
    if not np.isfinite(float(grid_size)) or float(grid_size) <= 0:
        raise ValueError("grid_size must be a positive finite value")
    n = int(points.shape[0])
    if n <= 0:
        raise ValueError("cannot voxelize an empty scene")

    # Use CPU NumPy only for the discrete selection.  It avoids backend-specific
    # tie ordering while the resulting maps are moved to the model device.
    points_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
    grid_np = np.floor(points_np / float(grid_size)).astype(np.int64, copy=False)
    grid_np -= grid_np.min(axis=0, keepdims=True)
    raw_idx = np.arange(n, dtype=np.int64)
    order = np.lexsort((raw_idx, grid_np[:, 2], grid_np[:, 1], grid_np[:, 0]))
    sorted_grid = grid_np[order]
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = np.any(sorted_grid[1:] != sorted_grid[:-1], axis=1)
    representative = order[starts]
    token_grid_np = grid_np[representative]
    raw_to_np = np.empty(n, dtype=np.int64)
    raw_to_np[order] = np.cumsum(starts, dtype=np.int64) - 1

    device = points.device
    representative_t = torch.as_tensor(
        representative, dtype=torch.long, device=device
    )
    raw_to_t = torch.as_tensor(raw_to_np, dtype=torch.long, device=device)
    token_grid_t = torch.as_tensor(token_grid_np, dtype=torch.int32, device=device)
    token_xyz = points[representative_t]
    token_features = features[representative_t]
    if int(raw_to_t.min().item()) != 0 or int(raw_to_t.max().item()) != len(representative) - 1:
        raise RuntimeError("representative-first voxel inverse map is not contiguous")
    return token_xyz, token_features, token_grid_t, raw_to_t, representative_t


def _import_external_ptv3(root: Path):
    """Import the detached official model while retaining relative imports."""

    root = Path(root).resolve(strict=True)
    if not (root / "model.py").is_file():
        raise FileNotFoundError(f"PTv3 source does not contain model.py: {root}")
    # The detached upstream repository intentionally has no package marker.
    # Adding a marker to the external artifact lets model.py resolve its
    # ``.serialization`` import without changing the vendored source itself.
    marker = root / "__init__.py"
    if not marker.exists():
        marker.touch()
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    package_name = root.name
    module = importlib.import_module(f"{package_name}.model")
    return module.PointTransformerV3


def _backbone_state_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    """Extract and normalize the backbone portion of a Pointcept checkpoint."""

    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        state = payload["state_dict"]
    elif isinstance(payload, Mapping):
        state = payload
    else:
        raise TypeError("PTv3 checkpoint must contain a state-dict mapping")
    result: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        name = str(raw_name)
        if name.startswith("module.backbone."):
            name = name[len("module.backbone.") :]
        elif name.startswith("backbone."):
            name = name[len("backbone.") :]
        else:
            # The public segmentation head and other runner state are not part
            # of the frozen PTv3 feature encoder.
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint tensor {raw_name!r} is not a Tensor")
        result[name] = value
    if not result:
        raise RuntimeError("No module.backbone.* tensors found in PTv3 checkpoint")
    return result


class PTv3Encoder(nn.Module):
    """Public ScanNet PTv3m1-0 backbone with a dense point mapping."""

    def __init__(
        self,
        *,
        ptv3_root: Path,
        checkpoint: Path,
        grid_size: float = PTV3_DEFAULT_GRID_SIZE,
        enable_flash: bool = True,
        shuffle_orders: bool = True,
        capture_hierarchy: bool = False,
    ) -> None:
        super().__init__()
        constructor = dict(PTV3_DEFAULT_CONFIG)
        constructor["enable_flash"] = bool(enable_flash)
        constructor["shuffle_orders"] = bool(shuffle_orders)
        model_cls = _import_external_ptv3(Path(ptv3_root))
        self.backbone = model_cls(**constructor)
        ckpt = Path(checkpoint).resolve(strict=True)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = _backbone_state_from_checkpoint(payload)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "PTv3 public checkpoint does not match the resolved architecture: "
                f"missing={list(missing)!r}, unexpected={list(unexpected)!r}"
            )
        self.grid_size = float(grid_size)
        self.input_channels = PTV3_INPUT_CHANNELS
        self.output_dim = PTV3_DEFAULT_OUTPUT_DIM
        self.capture_hierarchy = bool(capture_hierarchy)
        self._active_hierarchy_capture: dict[str, dict[str, torch.Tensor | None]] | None = None
        self._hierarchy_hook_handles: list[Any] = []
        if int(constructor["in_channels"]) != self.input_channels:
            raise RuntimeError("PTv3 input channel contract drifted from RGBN6")
        if self.capture_hierarchy:
            # PTv3's public implementation keeps the point hierarchy in named
            # enc/dec PointSequential modules.  Hooks let us expose those
            # tensors without modifying the external upstream checkout.
            for stage in ("enc0", "enc1", "enc2", "enc3", "enc4"):
                module = getattr(self.backbone.enc, stage, None)
                if module is None:
                    raise RuntimeError(f"PTv3 hierarchy stage missing: {stage}")
                self._hierarchy_hook_handles.append(
                    module.register_forward_hook(self._make_hierarchy_hook(stage))
                )
            for stage in ("dec3", "dec2", "dec1", "dec0"):
                module = getattr(self.backbone.dec, stage, None)
                if module is None:
                    raise RuntimeError(f"PTv3 hierarchy stage missing: {stage}")
                self._hierarchy_hook_handles.append(
                    module.register_forward_hook(self._make_hierarchy_hook(stage))
                )

    def _make_hierarchy_hook(self, stage: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._active_hierarchy_capture is None:
                return
            if not isinstance(output, Mapping) or "feat" not in output:
                raise RuntimeError(f"PTv3 stage {stage} did not return a Point")
            feature = output["feat"]
            coord = output.get("coord")
            if not isinstance(feature, torch.Tensor) or not isinstance(coord, torch.Tensor):
                raise RuntimeError(f"PTv3 stage {stage} has invalid feature/coord tensors")
            # Clone keeps the intermediate tensor stable when a later
            # unpooling step mutates the Point object.  clone() preserves the
            # autograd graph, which is required for auxiliary supervision.
            self._active_hierarchy_capture[stage] = {
                "feat": feature.clone(),
                "coord": coord.clone(),
                "pooling_inverse": (
                    output.get("pooling_inverse").clone()
                    if isinstance(output.get("pooling_inverse"), torch.Tensor)
                    else None
                ),
            }

        return hook

    @staticmethod
    def _hierarchy_maps(
        *,
        raw_to_token: torch.Tensor,
        captured: Mapping[str, Mapping[str, torch.Tensor | None]],
    ) -> dict[str, torch.Tensor]:
        """Compose PTv3 pooling inverses into raw-point-to-stage maps."""

        n_tokens = int(raw_to_token.max().item()) + 1
        raw_stage_maps: dict[str, torch.Tensor] = {
            "enc0": raw_to_token,
        }
        # ``token_map`` always has one entry per enc0 token.  Each pooling
        # inverse maps the previous stage's token index into the next stage,
        # so it can be composed repeatedly before expanding back to raw points.
        token_map = torch.arange(
            n_tokens, device=raw_to_token.device, dtype=torch.long
        )
        for index in range(1, 5):
            stage = f"enc{index}"
            inverse = captured[stage].get("pooling_inverse")
            if not isinstance(inverse, torch.Tensor):
                raise RuntimeError(f"PTv3 stage {stage} has no pooling inverse")
            inverse = inverse.long()
            if inverse.ndim != 1 or int(inverse.shape[0]) < int(token_map.max().item()) + 1:
                raise RuntimeError(
                    f"PTv3 pooling inverse shape drift at {stage}: "
                    f"{tuple(inverse.shape)} vs token_map_max={int(token_map.max().item())}"
                )
            token_map = inverse[token_map]
            raw_stage_maps[stage] = token_map[raw_to_token]
        # Decoder stage dec{s} returns to the token set of encoder stage s.
        return {
            "dec0": raw_stage_maps["enc0"],
            "dec1": raw_stage_maps["enc1"],
            "dec2": raw_stage_maps["enc2"],
            "dec3": raw_stage_maps["enc3"],
        }

    def forward(
        self,
        *,
        points: torch.Tensor,
        features: torch.Tensor,
        seed: int,
    ) -> PTv3FeatureOutput:
        if points.device != features.device:
            raise ValueError("points and features must be on the same device")
        if features.shape[1] != self.input_channels:
            raise ValueError(
                f"PTv3 expects RGBN6 ({self.input_channels}), got {features.shape[1]}"
            )
        # PTv3 shuffles serialization orders.  Seed each visit explicitly so
        # the adaptation stream is reproducible, including its random order.
        torch.random.manual_seed(int(seed))
        if points.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        token_xyz, token_input, token_grid, raw_to_token, representatives = (
            representative_first_voxelize(points, features, self.grid_size)
        )
        n_tokens = int(token_xyz.shape[0])
        data = {
            "coord": token_xyz,
            "grid_coord": token_grid,
            "feat": token_input,
            "offset": torch.tensor(
                [n_tokens], dtype=torch.long, device=points.device
            ),
            "grid_size": float(self.grid_size),
        }
        if self.capture_hierarchy:
            self._active_hierarchy_capture = {}
        try:
            point = self.backbone(data)
        finally:
            captured = self._active_hierarchy_capture
            self._active_hierarchy_capture = None
        token_features = point.feat
        if token_features.ndim != 2 or int(token_features.shape[0]) != n_tokens:
            raise RuntimeError(
                "PTv3 decoder did not return one feature per input token: "
                f"{tuple(token_features.shape)} vs {n_tokens}"
            )
        point_features = token_features[raw_to_token]
        if int(point_features.shape[0]) != int(points.shape[0]):
            raise RuntimeError("PTv3 raw-point feature cardinality mismatch")
        hierarchy_tokens: dict[str, torch.Tensor] = {}
        hierarchy_xyz: dict[str, torch.Tensor] = {}
        hierarchy_raw_maps: dict[str, torch.Tensor] = {}
        if self.capture_hierarchy:
            if captured is None:
                raise RuntimeError("PTv3 hierarchy capture returned no data")
            required = set(("enc0", "enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1", "dec0"))
            if not required.issubset(captured):
                raise RuntimeError(
                    "PTv3 hierarchy capture incomplete: "
                    f"missing={sorted(required.difference(captured))}"
                )
            stage_maps = self._hierarchy_maps(
                raw_to_token=raw_to_token,
                captured=captured,
            )
            for stage in ("dec3", "dec2", "dec1", "dec0"):
                feat = captured[stage].get("feat")
                coord = captured[stage].get("coord")
                raw_map = stage_maps[stage]
                if not isinstance(feat, torch.Tensor) or not isinstance(coord, torch.Tensor):
                    raise RuntimeError(f"PTv3 stage {stage} tensors are unavailable")
                if int(raw_map.max().item()) >= int(feat.shape[0]):
                    raise RuntimeError(f"PTv3 stage {stage} raw map exceeds token count")
                hierarchy_tokens[stage] = feat
                hierarchy_xyz[stage] = coord
                hierarchy_raw_maps[stage] = raw_map
        return PTv3FeatureOutput(
            point_features=point_features,
            token_features=token_features,
            raw_to_token=raw_to_token,
            token_xyz=point.coord,
            token_grid=token_grid,
            representative_indices=representatives,
            hierarchy_tokens=hierarchy_tokens,
            hierarchy_xyz=hierarchy_xyz,
            hierarchy_raw_maps=hierarchy_raw_maps,
        )


def _catalogs_for_scene(
    *, scene: SceneSource, device: torch.device, cells: Sequence[str]
) -> dict[str, tuple[FrameProposalCatalog, ...]]:
    catalogs: dict[str, tuple[FrameProposalCatalog, ...]] = {}
    for cell in cells:
        if not str(cell).startswith("2d/"):
            raise ValueError(f"PTv3 Delimit3D adaptation accepts 2D cells, got {cell!r}")
        granularity = str(cell).split("/", 1)[1]
        catalogs[str(cell)] = tuple(
            FrameProposalCatalog(
                cell=str(cell),
                frame_id=frame.physical_frame_id,
                visible_indices=torch.as_tensor(
                    frame.visible_indices, dtype=torch.long, device=device
                ),
                proposal_offsets=torch.as_tensor(
                    frame.proposal_offsets, dtype=torch.long, device=device
                ),
                proposal_point_indices=torch.as_tensor(
                    frame.proposal_point_indices, dtype=torch.long, device=device
                ),
            )
            for frame in scene.frames(granularity, "train")
        )
    return catalogs


def build_ptv3_v2_plan(
    *,
    scene: SceneSource,
    epoch: int,
    seed: int,
    points: torch.Tensor,
    cells: Sequence[str],
    sampling: Any,
    coverage_states: Mapping[str, DeterministicCoverageState | None] | None = None,
):
    """Build the audited three-granularity Delimit3D proposal plan."""

    cell_tuple = tuple(str(cell) for cell in cells)
    return sample_multigranular_frame_group_plan(
        scene_id=scene.scene_id,
        epoch=int(epoch),
        seed=int(seed),
        points=points,
        frames_by_cell=_catalogs_for_scene(
            scene=scene, device=points.device, cells=cell_tuple
        ),
        total_proposal_quota=int(sampling.proposals_per_forward),
        cells=cell_tuple,
        positive_pairs_per_proposal=int(sampling.positive_pairs_per_proposal),
        num_uniform_negatives=int(sampling.uniform_negatives),
        num_spatial_hard_negatives=int(sampling.spatial_hard_negatives),
        spatial_candidate_pool=int(sampling.spatial_candidate_pool),
        num_feature_hard_negatives=int(sampling.feature_hard_negatives),
        feature_candidate_pool=int(sampling.feature_candidate_pool),
        require_negative_proposal_membership=True,
        require_hierarchy_routes=False,
        defer_token_loss=False,
        operation="ptv3-rgbn6-pretraining-v2",
        coverage_states=coverage_states,
    )


class PTv3ContrastiveModel(nn.Module):
    """Public PTv3 plus the disposable Delimit3D projection/criterion."""

    def __init__(
        self,
        *,
        ptv3_root: Path,
        checkpoint: Path,
        grid_size: float = PTV3_DEFAULT_GRID_SIZE,
        projection_dim: int = 128,
        projection_hidden_dim: int = 128,
        enable_flash: bool = True,
        shuffle_orders: bool = True,
        multiscale_supervision: bool = False,
        multiscale_loss_weight: float = PTV3_MULTISCALE_AUXILIARY_WEIGHT,
        multiscale_warmup_updates: int = PTV3_MULTISCALE_WARMUP_UPDATES,
    ) -> None:
        super().__init__()
        self.encoder = PTv3Encoder(
            ptv3_root=ptv3_root,
            checkpoint=checkpoint,
            grid_size=grid_size,
            enable_flash=enable_flash,
            shuffle_orders=shuffle_orders,
            capture_hierarchy=multiscale_supervision,
        )
        self.multiscale_supervision = bool(multiscale_supervision)
        self.multiscale_loss_weight = float(multiscale_loss_weight)
        self.multiscale_warmup_updates = int(multiscale_warmup_updates)
        if self.multiscale_loss_weight < 0.0:
            raise ValueError("multiscale_loss_weight must be non-negative")
        if self.multiscale_warmup_updates < 0:
            raise ValueError("multiscale_warmup_updates must be non-negative")
        self.projector = ContrastiveProjectionHead(
            self.encoder.output_dim, projection_hidden_dim, projection_dim
        )
        self.criterion = PartFieldContrastiveCriterion(
            temperature=0.07, learnable_temperature=True
        )
        self.multiscale_projectors = nn.ModuleDict()
        self.multiscale_criteria = nn.ModuleDict()
        if self.multiscale_supervision:
            stage_channels = {"dec3": 256, "dec2": 128, "dec1": 64}
            self.multiscale_projectors.update(
                {
                    stage: ContrastiveProjectionHead(
                        stage_channels[stage], projection_hidden_dim, projection_dim
                    )
                    for stage in PTV3_MULTISCALE_LEVEL_NAMES
                }
            )
            self.multiscale_criteria.update(
                {
                    stage: PartFieldContrastiveCriterion(
                        temperature=0.07,
                        learnable_temperature=True,
                        symmetric_feature_hard_mining=False,
                    )
                    for stage in PTV3_MULTISCALE_LEVEL_NAMES
                }
            )

    @property
    def feature_dim(self) -> int:
        return int(self.encoder.output_dim)

    def forward(
        self,
        *,
        points: torch.Tensor,
        features: torch.Tensor,
        batch: Any,
        seed: int,
        frame_groups: Sequence[Any] | None = None,
        supervision_step: int | None = None,
        supervision_total_steps: int | None = None,
    ) -> dict[str, Any]:
        encoded = self.encoder(points=points, features=features, seed=int(seed))
        groups = normalize_hierarchy_frame_groups(
            batch=batch, source_cell=None, frame_groups=frame_groups
        )
        projected = self.projector(encoded.point_features)
        results = [
            self.criterion(projected.float(), group.batch) for group in groups
        ]
        if not results:
            raise RuntimeError("PTv3 Delimit3D plan retained no contrastive groups")
        final_loss = torch.stack([item["loss_total"] for item in results]).mean()
        auxiliary_loss = final_loss.new_zeros(())
        auxiliary_records: dict[str, list[dict[str, Any]]] = {
            stage: [] for stage in PTV3_MULTISCALE_LEVEL_NAMES
        }
        auxiliary_tensors: list[torch.Tensor] = []
        if self.multiscale_supervision:
            for group in groups:
                granularity = str(group.cell).split("/", 1)[-1]
                stage = PTV3_MULTISCALE_ROUTE_BY_GRANULARITY.get(granularity)
                if stage is None:
                    raise RuntimeError(
                        f"No PTv3 multiscale route for source cell {group.cell!r}"
                    )
                stage_features = encoded.hierarchy_tokens.get(stage)
                stage_xyz = encoded.hierarchy_xyz.get(stage)
                stage_map = encoded.hierarchy_raw_maps.get(stage)
                if stage_features is None or stage_xyz is None or stage_map is None:
                    raise RuntimeError(f"PTv3 hierarchy output missing stage {stage}")
                stage_batch, mapping_stats = remap_contrastive_batch_to_level(
                    group.batch,
                    stage_map,
                )
                if stage_batch is None:
                    stage_loss = zero_loss_touching_module_parameters(
                        stage_features, self.multiscale_criteria[stage]
                    )
                    stage_metrics = {
                        "loss": 0.0,
                        "cosine_gap": 0.0,
                        "temperature": float(
                            self.multiscale_criteria[stage].temperature.detach()
                        ),
                        **mapping_stats,
                    }
                else:
                    stage_projected = self.multiscale_projectors[stage](
                        stage_features
                    )
                    stage_result = self.multiscale_criteria[stage](
                        stage_projected.float(), stage_batch
                    )
                    stage_loss = stage_result["loss_total"]
                    stage_metrics = {
                        "loss": float(stage_loss.detach()),
                        "cosine_gap": float(stage_result["cosine_gap"]),
                        "temperature": float(stage_result["temperature"]),
                        **mapping_stats,
                    }
                auxiliary_tensors.append(stage_loss)
                auxiliary_records[stage].append(stage_metrics)
            if auxiliary_tensors:
                auxiliary_loss = torch.stack(auxiliary_tensors).mean()
            step = int(supervision_step or 0)
            total = int(supervision_total_steps or 0)
            if self.multiscale_warmup_updates == 0:
                ramp = 1.0
            else:
                ramp = min(1.0, max(0.0, float(step) / float(self.multiscale_warmup_updates)))
            if total and step > total:
                raise ValueError("supervision_step exceeds supervision_total_steps")
            auxiliary_scale = self.multiscale_loss_weight * ramp
        else:
            auxiliary_scale = 0.0
        loss = final_loss + auxiliary_scale * auxiliary_loss
        stage_summary = {
            stage: {
                "loss": sum(item["loss"] for item in records) / max(len(records), 1),
                "cosine_gap": sum(item["cosine_gap"] for item in records) / max(len(records), 1),
                "temperature": sum(item["temperature"] for item in records) / max(len(records), 1),
                "group_count": len(records),
            }
            for stage, records in auxiliary_records.items()
        }
        result: dict[str, Any] = {
            "loss_total": loss,
            "loss_contrastive": float(loss.detach()),
            "loss_final_dec0": float(final_loss.detach()),
            "loss_multiscale": float(auxiliary_loss.detach()),
            "multiscale_loss_scale": float(auxiliary_scale),
            "multiscale": stage_summary,
            "temperature": float(self.criterion.temperature.detach()),
            "cosine_gap": float(
                sum(float(item["cosine_gap"]) for item in results) / len(results)
            ),
            "positive_cosine_mean": float(
                sum(float(item["positive_cosine_mean"]) for item in results)
                / len(results)
            ),
            "negative_cosine_mean": float(
                sum(float(item["negative_cosine_mean"]) for item in results)
                / len(results)
            ),
            "triplet_ranking_accuracy": float(
                sum(float(item["triplet_ranking_accuracy"]) for item in results)
                / len(results)
            ),
            "hardest_negative_ranking_accuracy": float(
                sum(
                    float(item["hardest_negative_ranking_accuracy"])
                    for item in results
                )
                / len(results)
            ),
            "group_count": len(results),
            "feature_dim": self.feature_dim,
            "raw_point_count": int(encoded.point_features.shape[0]),
            "final_token_count": int(encoded.token_features.shape[0]),
            "raw_to_token": encoded.raw_to_token,
            "token_xyz": encoded.token_xyz,
            "token_grid": encoded.token_grid,
            "representative_indices": encoded.representative_indices,
            "point_features": encoded.point_features,
        }
        return result
