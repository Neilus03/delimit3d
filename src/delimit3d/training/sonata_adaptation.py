"""Sonata encoder bridge for Delimit3D short 2D-pseudomask adaptation.

The bridge deliberately keeps the official Sonata/PTv3 input contract (9 channels:
centered coordinates, RGB, and normals) and exposes one dense feature per raw
scene point by composing Sonata's voxel and hierarchical pooling maps.  It is
separate from the validated LitePT runner so the latter remains unchanged.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
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
    build_scene_visit_input,
    normalize_hierarchy_frame_groups,
)


SONATA_SOURCE_COMMIT = "18c09ff8d713494f78a8213792262b910977a65d"
SONATA_HF_REPO = "facebook/sonata"
SONATA_INPUT_CHANNELS = 9
SONATA_DEFAULT_GRID_SIZE = 0.02
SONATA_DEFAULT_OUTPUT_DIM = 512


@dataclass(frozen=True)
class SonataFeatureOutput:
    """Features and deterministic maps for one raw scene visit."""

    point_features: torch.Tensor  # [N_raw, C]
    token_features: torch.Tensor  # [N_final, C]
    raw_to_token: torch.Tensor  # [N_raw]
    token_xyz: torch.Tensor  # [N_final, 3]
    sampled_xyz: torch.Tensor  # [N_grid, 3]


def _catalogs_for_scene(
    *, scene: SceneSource, device: torch.device, cells: Sequence[str]
) -> dict[str, tuple[FrameProposalCatalog, ...]]:
    catalogs: dict[str, tuple[FrameProposalCatalog, ...]] = {}
    for cell in cells:
        if not str(cell).startswith("2d/"):
            raise ValueError(f"Sonata 2D adaptation only accepts 2D cells, got {cell!r}")
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


def build_sonata_v2_plan(
    *,
    scene: SceneSource,
    epoch: int,
    seed: int,
    points: torch.Tensor,
    cells: Sequence[str],
    sampling: Any,
    coverage_states: Mapping[str, DeterministicCoverageState | None] | None = None,
):
    """Build the same deterministic three-granularity plan used by LitePT V2."""
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
        operation="sonata-rgbn6-pretraining-v2",
        coverage_states=coverage_states,
    )


def _compose_raw_to_final(point: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw->final map, final xyz, and sampled xyz from a Sonata Point."""
    # The final encoder Point carries the nested pooling trace.  Sonata keeps
    # the raw GridSample inverse on the earliest parent, not on the deepest
    # Point returned by the encoder.
    deepest = point
    inverses: list[torch.Tensor] = []
    while "pooling_parent" in deepest.keys():
        if "pooling_inverse" not in deepest.keys():
            raise RuntimeError("Sonata pooling trace is incomplete")
        inverses.append(deepest.pooling_inverse.long())
        deepest = deepest.pooling_parent
    if "inverse" not in deepest.keys():
        raise RuntimeError("Sonata output lost the raw-point inverse map")
    raw_to_sampled = deepest.inverse.long()
    # The nested trace is final->...->2cm.  Apply maps from 2cm upwards.
    sampled_to_final = torch.arange(
        deepest.feat.shape[0], device=deepest.feat.device, dtype=torch.long
    )
    for inverse in reversed(inverses):
        sampled_to_final = inverse[sampled_to_final]
    raw_to_final = sampled_to_final[raw_to_sampled]
    return raw_to_final, point.coord, deepest.coord


class SonataEncoder(nn.Module):
    """Official Sonata checkpoint plus deterministic dense point projection."""

    def __init__(
        self,
        *,
        sonata_root: Path,
        checkpoint: Path,
        grid_size: float = SONATA_DEFAULT_GRID_SIZE,
        enable_flash: bool = False,
        patch_size: int = 1024,
    ) -> None:
        super().__init__()
        root = Path(sonata_root).resolve(strict=True)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from sonata.model import load  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised on compute nodes
            raise RuntimeError(
                "Unable to import the external Sonata source; check SONATA_ROOT "
                "and its compute-node dependencies"
            ) from exc
        ckpt = Path(checkpoint).resolve(strict=True)
        custom_config = {
            "enable_flash": bool(enable_flash),
            "enc_patch_size": [int(patch_size)] * 5,
        }
        self.backbone = load(str(ckpt), custom_config=custom_config)
        self.grid_size = float(grid_size)
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        self.output_dim = int(getattr(self.backbone, "enc_channels", [SONATA_DEFAULT_OUTPUT_DIM])[-1])
        if int(getattr(self.backbone, "in_channels", -1)) != SONATA_INPUT_CHANNELS:
            raise RuntimeError(
                "Sonata checkpoint input contract drifted: expected 9 channels "
                f"(coord+RGB+normal), got {getattr(self.backbone, 'in_channels', None)!r}"
            )

    @staticmethod
    def _transform_config(grid_size: float) -> list[dict[str, Any]]:
        # This is the official default pipeline with the grid size exposed as a
        # resolved config value.  Train mode is deterministic once NumPy/Python
        # RNGs are seeded immediately before each call.
        return [
            {"type": "CenterShift", "apply_z": True},
            {
                "type": "GridSample",
                "grid_size": float(grid_size),
                "hash_type": "fnv",
                "mode": "train",
                "return_grid_coord": True,
                "return_inverse": True,
            },
            {"type": "NormalizeColor"},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "grid_coord", "color", "inverse"),
                "feat_keys": ("coord", "color", "normal"),
            },
        ]

    def _prepare(
        self,
        *,
        points: torch.Tensor,
        colors: np.ndarray,
        normals: np.ndarray,
        seed: int,
    ) -> dict[str, Any]:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be [N,3], got {tuple(points.shape)}")
        n = int(points.shape[0])
        colors_np = np.asarray(colors, dtype=np.float32)
        normals_np = np.asarray(normals, dtype=np.float32)
        points_np = points.detach().cpu().numpy().astype(np.float32, copy=True)
        if colors_np.shape != (n, 3) or normals_np.shape != (n, 3):
            raise ValueError("Sonata requires raw RGB and normal arrays aligned to points")
        # Sonata's NormalizeColor expects 0..255 input.  Existing source packs
        # store either that convention or 0..1; convert the latter explicitly.
        if float(np.nanmax(colors_np)) <= 1.0 + 1e-5:
            colors_np = colors_np * 255.0
        data: dict[str, Any] = {
            "coord": points_np,
            "color": colors_np,
            "normal": normals_np,
        }
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        # Import lazily so CPU unit tests do not require spconv/timm.
        from sonata.transform import Compose  # type: ignore

        data = Compose(self._transform_config(self.grid_size))(data)
        return data

    def forward(
        self,
        *,
        points: torch.Tensor,
        colors: np.ndarray,
        normals: np.ndarray,
        seed: int,
    ) -> SonataFeatureOutput:
        data = self._prepare(points=points, colors=colors, normals=normals, seed=seed)
        for key, value in list(data.items()):
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device=points.device, non_blocking=True)
        point = self.backbone(data)
        raw_to_final, final_xyz, sampled_xyz = _compose_raw_to_final(point)
        token_features = point.feat
        point_features = token_features[raw_to_final]
        if int(point_features.shape[0]) != int(points.shape[0]):
            raise RuntimeError("Sonata raw-point feature cardinality mismatch")
        return SonataFeatureOutput(
            point_features=point_features,
            token_features=token_features,
            raw_to_token=raw_to_final,
            token_xyz=final_xyz,
            sampled_xyz=sampled_xyz,
        )


class SonataContrastiveModel(nn.Module):
    """Sonata backbone + disposable contrastive projection head."""

    def __init__(
        self,
        *,
        sonata_root: Path,
        checkpoint: Path,
        grid_size: float = SONATA_DEFAULT_GRID_SIZE,
        projection_dim: int = 128,
        projection_hidden_dim: int = 128,
        enable_flash: bool = False,
        patch_size: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = SonataEncoder(
            sonata_root=sonata_root,
            checkpoint=checkpoint,
            grid_size=grid_size,
            enable_flash=enable_flash,
            patch_size=patch_size,
        )
        self.projector = ContrastiveProjectionHead(
            self.encoder.output_dim, projection_hidden_dim, projection_dim
        )
        self.criterion = PartFieldContrastiveCriterion(
            temperature=0.07, learnable_temperature=True
        )

    @property
    def feature_dim(self) -> int:
        return int(self.encoder.output_dim)

    def forward(
        self,
        *,
        points: torch.Tensor,
        colors: np.ndarray,
        normals: np.ndarray,
        batch: Any,
        seed: int,
        frame_groups: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        encoded = self.encoder(
            points=points, colors=colors, normals=normals, seed=int(seed)
        )
        projected = self.projector(encoded.point_features)
        groups = normalize_hierarchy_frame_groups(
            batch=batch, source_cell=None, frame_groups=frame_groups
        )
        if frame_groups is None:
            result = self.criterion(projected.float(), groups[0].batch)
        else:
            results = [self.criterion(projected.float(), group.batch) for group in groups]
            loss = torch.stack([item["loss_total"] for item in results]).mean()
            result = {
                "loss_total": loss,
                "loss_contrastive": float(loss.detach()),
                "temperature": float(self.criterion.temperature.detach()),
                "cosine_gap": float(
                    sum(float(item["cosine_gap"]) for item in results) / len(results)
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
            }
        result.update(
            {
                "feature_dim": self.feature_dim,
                "raw_point_count": int(encoded.point_features.shape[0]),
                "final_token_count": int(encoded.token_features.shape[0]),
                "raw_to_token": encoded.raw_to_token,
                "token_xyz": encoded.token_xyz,
                "point_features": encoded.point_features,
            }
        )
        return result
