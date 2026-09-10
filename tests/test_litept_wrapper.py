from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from delimit3d.models.litept_wrapper import LitePTBackbone


class _IdentityLitePT(nn.Module):
    def forward(self, point_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
        return SimpleNamespace(
            feat=point_dict["feat"],
            grid_coord=point_dict["grid_coord"],
            offset=point_dict["offset"],
        )


def _minimal_backbone(*, cache_training_voxelization: bool) -> LitePTBackbone:
    backbone = LitePTBackbone.__new__(LitePTBackbone)
    nn.Module.__init__(backbone)
    backbone.in_channels = 6
    backbone.grid_size = 0.02
    backbone.voxel_reduce = "mean"
    backbone.representative_sampling = "random"
    backbone.cache_training_voxelization = cache_training_voxelization
    backbone.feature_pyramid_mode = None
    backbone._cached_voxelization = None
    backbone._multi_scale = False
    backbone._captured = {}
    backbone._multi_scale_channels = []
    backbone._multi_scale_divisors = []
    backbone._multi_scale_indices = None
    backbone._fpn_captured = {}
    backbone.model = _IdentityLitePT()
    backbone.train()
    return backbone


def test_disabled_training_cache_revoxelizes_different_scenes() -> None:
    backbone = _minimal_backbone(cache_training_voxelization=False)

    first = backbone(torch.zeros(3, 3), torch.zeros(3, 6))
    second = backbone(
        torch.arange(15, dtype=torch.float32).reshape(5, 3),
        torch.ones(5, 6),
    )

    assert first.point_feat.shape[0] == 3
    assert second.point_feat.shape[0] == 5
    assert backbone._cached_voxelization is None


def test_legacy_training_cache_remains_available_for_single_scene_optimizers() -> None:
    backbone = _minimal_backbone(cache_training_voxelization=True)

    backbone(torch.zeros(3, 3), torch.zeros(3, 6))
    assert backbone._cached_voxelization is not None
