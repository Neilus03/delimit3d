"""Thin wrapper around the standalone LitePT encoder-decoder.

Downstream code receives a structured :class:`LitePTBackboneOutput`::

    bb = backbone(coord, feat)
    bb.point_feat    # [N, 72] dense per-point features
    bb.scene_tokens  # [V, 72] sparse voxel features
    bb.point_xyz     # [N, 3]  original coordinates
    bb.scene_xyz     # [V, 3]  voxel centroids
    bb.inverse_map   # [N]     point → voxel index

By default the backbone is **LitePT-S*** (ScanNet instance-seg architecture from
``configs/scannet/insseg-litept-small-v1m2.py``): deeper decoder
``dec_depths=(2, 2, 2, 2)`` vs LitePT-S where ``dec_depths=(0, 0, 0, 0)``. Use
``litept_variant="litept_s"`` only for checkpoints trained with the shallow decoder.

Everything LitePT-specific (voxelization, Point dict, offset, grid_coord,
sparse tensors) is hidden inside this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# LitePT-S* — mirrors LitePT/configs/scannet/insseg-litept-small-v1m2.py ``backbone`` dict
# (excluding ``type`` and ``in_channels``, which come from the adaptation config).
LITEPT_S_STAR_KWARGS: dict[str, Any] = {
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (2, 2, 2, 6, 2),
    "enc_channels": (36, 72, 144, 252, 504),
    "enc_num_head": (2, 4, 8, 14, 28),
    "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    "enc_conv": (True, True, True, False, False),
    "enc_attn": (False, False, False, True, True),
    "enc_rope_freq": (100.0, 100.0, 100.0, 100.0, 100.0),
    "dec_depths": (2, 2, 2, 2),
    "dec_channels": (72, 72, 144, 252),
    "dec_num_head": (4, 4, 8, 14),
    "dec_patch_size": (1024, 1024, 1024, 1024),
    "dec_conv": (True, True, True, False),
    "dec_attn": (False, False, False, True),
    "dec_rope_freq": (100.0, 100.0, 100.0, 100.0),
    "mlp_ratio": 4,
    "qkv_bias": True,
    "qk_scale": None,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "drop_path": 0.3,
    "pre_norm": True,
    "shuffle_orders": True,
    "enc_mode": False,
}


def _litept_constructor_kwargs(variant: str) -> dict[str, Any]:
    if variant == "litept_s_star":
        return dict(LITEPT_S_STAR_KWARGS)
    if variant == "litept_s":
        # ``litept.model.LitePT`` defaults match semantic-seg LitePT-S (no decoder blocks).
        return {}
    raise ValueError(
        f"Unknown litept_variant {variant!r}; expected 'litept_s_star' or 'litept_s'."
    )


@dataclass
class LitePTBackboneOutput:
    """Structured output from the LitePT backbone.

    Carries both the dense per-point features needed for final mask
    dot-products and the sparse voxel tokens needed for efficient
    Transformer cross-attention in the decoder.

    When ``multi_scale`` is enabled, ``multi_scale_tokens`` and
    ``multi_scale_xyz`` contain per-decoder-substage features in
    **coarse → fine** order (e.g. 4 entries for LitePT-S*).
    """

    point_feat: torch.Tensor    # [N, C] dense per-point features
    point_xyz: torch.Tensor     # [N, 3] original point coordinates
    scene_tokens: torch.Tensor  # [V, C] sparse voxel features (finest scale)
    scene_xyz: torch.Tensor     # [V, 3] voxel centroids (finest scale)
    inverse_map: torch.Tensor   # [N]    maps points → voxel token index
    point_offsets: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    scene_token_offsets: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))

    multi_scale_tokens: list[torch.Tensor] = field(default_factory=list)
    multi_scale_xyz: list[torch.Tensor] = field(default_factory=list)
    multi_scale_offsets: list[torch.Tensor] = field(default_factory=list)
    finest_to_multi_scale: list[torch.Tensor] = field(default_factory=list)

    # Optional transfer-facing native hierarchy.  Unlike ``multi_scale_*``,
    # this includes enc4 and dec0 and carries exact adjacent child-to-parent
    # maps in dec0->dec1->dec2->dec3->enc4 order.
    hierarchy_stage_names: tuple[str, ...] = ()
    hierarchy_tokens: tuple[torch.Tensor, ...] = ()
    hierarchy_xyz: tuple[torch.Tensor, ...] = ()
    hierarchy_grids: tuple[torch.Tensor, ...] = ()
    hierarchy_offsets: tuple[torch.Tensor, ...] = ()
    hierarchy_parent_maps: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class LitePTPyramidLevel:
    """One named LitePT feature level in a batched sparse pyramid."""

    name: str
    stride: int
    channels: int
    features: torch.Tensor
    grid: torch.Tensor
    xyz: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class LitePTFeaturePyramid:
    """Faithful tensor contract between LitePT-S* and Mask3D.

    ``memory_levels`` is ordered coarse to fine. The native ``mask3d_fpn``
    contract uses ``enc4`` (stride 16), ``dec3`` (8), ``dec2`` (4), and
    ``dec1`` (2). The separate ``mask3d_decoder_only_fpn`` contract replaces
    the native enc4 features with exact dec3 means on the enc4 grid, while
    keeping the same four spatial levels. ``parent_maps`` is ordered fine to
    coarse and contains exact adjacent child-to-parent indices for 1->2,
    2->4, 4->8, and 8->16. Indices are global within the concatenated batch;
    offsets make every scene boundary explicit.
    """

    finest_features: torch.Tensor
    finest_grid: torch.Tensor
    finest_xyz: torch.Tensor
    finest_offsets: torch.Tensor
    memory_levels: tuple[LitePTPyramidLevel, ...]
    point_xyz: torch.Tensor
    point_offsets: torch.Tensor
    original_to_finest: torch.Tensor
    representative_indices: torch.Tensor
    parent_maps: tuple[torch.Tensor, ...]

    @property
    def finest_channels(self) -> int:
        return int(self.finest_features.shape[-1])

    @property
    def memory_channels(self) -> tuple[int, ...]:
        return tuple(level.channels for level in self.memory_levels)

    @property
    def memory_strides(self) -> tuple[int, ...]:
        return tuple(level.stride for level in self.memory_levels)


def _mean_pool_features_by_parent(
    features: torch.Tensor,
    parent_map: torch.Tensor,
    parent_count: int,
) -> torch.Tensor:
    """Mean-pool child features onto an exact sparse parent grid."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    mapping = parent_map.to(device=features.device, dtype=torch.long).flatten()
    if mapping.shape != (features.shape[0],):
        raise ValueError(
            "parent_map must contain one index per child feature: "
            f"{tuple(mapping.shape)} != {(features.shape[0],)}"
        )
    count = int(parent_count)
    if count <= 0:
        raise ValueError(f"parent_count must be positive, got {count}")
    if mapping.numel() and (
        int(mapping.min().item()) < 0 or int(mapping.max().item()) >= count
    ):
        raise ValueError("parent_map contains an out-of-range parent index")

    pooled = features.new_zeros((count, features.shape[1]))
    pooled.index_add_(0, mapping, features)
    counts = features.new_zeros(count)
    counts.index_add_(0, mapping, features.new_ones(mapping.shape[0]))
    if bool((counts == 0).any().item()):
        raise RuntimeError("LitePT parent grid contains a token with no finest descendants")
    return pooled / counts.unsqueeze(1)


class LitePTBackbone(nn.Module):
    """Wraps :class:`litept.model.LitePT` for single- or multi-scene use.

    Parameters
    ----------
    litept_root:
        Absolute path to the cloned LitePT repository.  The wrapper adds it
        to ``sys.path`` so that ``from litept.model import LitePT`` works.
    in_channels:
        Number of per-point input feature channels (must match the actual
        feature tensor passed to :meth:`forward`).
    grid_size:
        Voxel edge length in meters for grid sampling.
    cache_training_voxelization:
        Reuse the first single-scene training voxelization. This is only valid
        for workflows that repeatedly optimize the same scene. Multi-scene
        training must disable it.
    """

    def __init__(
        self,
        litept_root: str,
        in_channels: int,
        grid_size: float = 0.02,
        litept_variant: str = "litept_s_star",
        litept_kwargs: dict[str, Any] | None = None,
        multi_scale: bool = False,
        multi_scale_indices: list[int] | None = None,
        cache_training_voxelization: bool = True,
        feature_pyramid_mode: str | None = None,
        voxel_reduce: str = "mean",
        representative_sampling: str = "random",
        capture_hierarchy: bool = False,
    ) -> None:
        super().__init__()
        self.litept_root = Path(litept_root).resolve()
        self.in_channels = int(in_channels)
        self.grid_size = float(grid_size)
        if litept_variant not in ("litept_s_star", "litept_s"):
            raise ValueError(
                f"litept_variant must be 'litept_s_star' or 'litept_s', got {litept_variant!r}"
            )
        self.litept_variant = litept_variant

        if not self.litept_root.exists():
            raise FileNotFoundError(
                f"LitePT root does not exist: {self.litept_root}"
            )

        litept_root_str = str(self.litept_root)
        if litept_root_str not in sys.path:
            sys.path.insert(0, litept_root_str)

        from litept.model import LitePT

        ctor: dict[str, Any] = _litept_constructor_kwargs(litept_variant)
        if litept_kwargs:
            ctor.update(litept_kwargs)
        self.model = LitePT(in_channels=self.in_channels, **ctor)

        # LitePT-S and LitePT-S* both use dec_channels[0] = 72 at the finest output.
        self.out_channels: int = 72
        self.cache_training_voxelization = bool(cache_training_voxelization)
        self.feature_pyramid_mode = feature_pyramid_mode
        self.capture_hierarchy = bool(capture_hierarchy)
        supported_pyramid_modes = {
            None,
            "mask3d_fpn",
            "mask3d_dec0_pooled_fpn",
            "mask3d_decoder_only_fpn",
        }
        if self.feature_pyramid_mode not in supported_pyramid_modes:
            raise ValueError(
                "feature_pyramid_mode must be None, 'mask3d_fpn', "
                "'mask3d_dec0_pooled_fpn', or 'mask3d_decoder_only_fpn', got "
                f"{self.feature_pyramid_mode!r}"
            )
        self.voxel_reduce = str(voxel_reduce)
        if self.voxel_reduce not in ("mean", "representative"):
            raise ValueError(
                "voxel_reduce must be 'mean' or 'representative', got "
                f"{self.voxel_reduce!r}"
            )
        self.representative_sampling = str(representative_sampling).strip().lower()
        if self.representative_sampling not in {"random", "first"}:
            raise ValueError(
                "representative_sampling must be 'random' or 'first', got "
                f"{representative_sampling!r}"
            )
        if self.feature_pyramid_mode in {
            "mask3d_fpn",
            "mask3d_dec0_pooled_fpn",
            "mask3d_decoder_only_fpn",
        }:
            if self.litept_variant != "litept_s_star":
                raise ValueError(
                    f"{self.feature_pyramid_mode} requires "
                    "litept_variant='litept_s_star'"
                )
            if self.voxel_reduce != "representative":
                raise ValueError(
                    f"{self.feature_pyramid_mode} requires "
                    "voxel_reduce='representative' so "
                    "Structured3D pretraining and ScanNet transfer use the same input contract"
                )
            if multi_scale:
                raise ValueError(
                    f"{self.feature_pyramid_mode} is a separate output contract; "
                    "do not also set multi_scale=true"
                )
        if self.capture_hierarchy:
            if self.litept_variant != "litept_s_star":
                raise ValueError("capture_hierarchy requires litept_variant='litept_s_star'")
            if self.feature_pyramid_mode is not None:
                raise ValueError(
                    "capture_hierarchy augments LitePTBackboneOutput and cannot be "
                    "combined with a feature_pyramid_mode return contract"
                )

        self._cached_voxelization: (
            tuple[
                dict[str, torch.Tensor],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None

        #  multi-scale decoder feature capture
        self._multi_scale = bool(multi_scale)
        self._captured: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._multi_scale_channels: list[int] = []
        self._multi_scale_divisors: list[int] = []
        self._multi_scale_indices: list[int] | None = None
        self._fpn_captured: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

        if self._multi_scale and hasattr(self.model, "dec"):
            dec_channels_cfg = list(ctor.get("dec_channels", (72, 72, 144, 252)))
            # dec_channels_cfg is indexed s=0 (finest) .. s=3 (coarsest).
            # self.model.dec runs in forward order dec3→dec0 (coarsest→finest),
            # so reversed gives coarse→fine channel widths. Hook indices i are in
            # that same coarse→fine order (i ascending).
            all_channels_coarse_to_fine = list(reversed(dec_channels_cfg))
            num_dec_stages = len(self.model.dec)

            if multi_scale_indices is None:
                selected = list(range(num_dec_stages))
            else:
                selected = sorted({int(i) for i in multi_scale_indices})
                for i in selected:
                    if i < 0 or i >= num_dec_stages:
                        raise ValueError(
                            f"multi_scale_indices contains out-of-range index {i}; "
                            f"expected [0, {num_dec_stages - 1}]"
                        )

            self._multi_scale_indices = selected
            self._multi_scale_channels = [
                all_channels_coarse_to_fine[i] for i in selected
            ]
            strides = list(ctor.get("stride", (2, 2, 2, 2)))
            self._multi_scale_divisors = []
            for i in selected:
                encoder_stage = num_dec_stages - 1 - i
                divisor = 1
                for stride in strides[:encoder_stage]:
                    divisor *= int(stride)
                self._multi_scale_divisors.append(divisor)

            for i in selected:
                self.model.dec[i].register_forward_hook(self._make_hook(i))

        if self.capture_hierarchy or self.feature_pyramid_mode in {
            "mask3d_fpn",
            "mask3d_dec0_pooled_fpn",
            "mask3d_decoder_only_fpn",
        }:
            if not hasattr(self.model, "enc") or not hasattr(self.model, "dec"):
                raise RuntimeError("LitePT-S* does not expose enc/dec stages")
            if len(self.model.enc) != 5 or len(self.model.dec) != 4:
                raise RuntimeError(
                    "Native hierarchy capture requires five LitePT encoder and "
                    "four decoder stages"
                )
            self.model.enc[4].register_forward_hook(self._make_fpn_hook("enc4"))
            for index, name in enumerate(("dec3", "dec2", "dec1", "dec0")):
                self.model.dec[index].register_forward_hook(self._make_fpn_hook(name))

    def _make_hook(self, scale_idx: int):
        """Capture features and integer grids from a LitePT decoder substage."""
        def hook(module, input, output):
            offset = getattr(
                output,
                "offset",
                torch.tensor([output.feat.shape[0]], device=output.feat.device, dtype=torch.long),
            )
            self._captured[scale_idx] = (
                output.feat,
                output.coord,
                offset,
                output.grid_coord,
            )
        return hook

    def _make_fpn_hook(self, name: str):
        """Capture a named LitePT stage without adding state-dict entries."""
        def hook(module, inputs, output):
            del module, inputs
            offset = getattr(
                output,
                "offset",
                torch.tensor(
                    [output.feat.shape[0]],
                    device=output.feat.device,
                    dtype=torch.long,
                ),
            )
            self._fpn_captured[name] = (
                output.feat,
                output.grid_coord,
                output.coord,
                offset,
            )

        return hook

    @staticmethod
    def _grid_parent_map(
        fine_grid: torch.Tensor,
        fine_offsets: torch.Tensor,
        coarse_grid: torch.Tensor,
        coarse_offsets: torch.Tensor,
        divisor: int,
    ) -> torch.Tensor:
        """Map finest voxels to an exact LitePT pooling ancestor.

        LitePT's ``GridPooling`` uses integer division followed by a unique
        operation independently per batch item. Repeating that key operation
        here recovers the pooling inverse without depending on LitePT internals.
        """
        if fine_grid.shape[0] == 0:
            return torch.zeros(0, device=fine_grid.device, dtype=torch.long)

        def scene_ids(offsets: torch.Tensor) -> torch.Tensor:
            starts = torch.cat([offsets.new_zeros(1), offsets[:-1]])
            counts = offsets - starts
            return torch.repeat_interleave(
                torch.arange(offsets.numel(), device=offsets.device), counts
            )

        fine_key = torch.cat(
            [scene_ids(fine_offsets)[:, None], fine_grid.long() // int(divisor)],
            dim=1,
        )
        coarse_key = torch.cat(
            [scene_ids(coarse_offsets)[:, None], coarse_grid.long()], dim=1
        )
        all_keys = torch.cat([coarse_key, fine_key], dim=0)
        _, inverse = torch.unique(all_keys, dim=0, sorted=True, return_inverse=True)
        coarse_inverse = inverse[: coarse_key.shape[0]]
        fine_inverse = inverse[coarse_key.shape[0] :]
        lookup = torch.full(
            (int(inverse.max().item()) + 1,),
            -1,
            device=fine_grid.device,
            dtype=torch.long,
        )
        lookup[coarse_inverse] = torch.arange(
            coarse_key.shape[0], device=fine_grid.device
        )
        mapping = lookup[fine_inverse]
        if bool((mapping < 0).any().item()):
            raise RuntimeError(
                "Could not recover LitePT finest-to-level pooling mapping"
            )
        return mapping

    @property
    def multi_scale_channels(self) -> list[int] | None:
        """Channel widths of each captured decoder scale (coarse → fine), or None."""
        if self._multi_scale and self._multi_scale_channels:
            return list(self._multi_scale_channels)
        return None

    # ------------------------------------------------------------------ #

    def _normalize_inputs(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Squeeze batch dim if present and validate shapes."""
        if coord.ndim == 3:
            assert coord.shape[0] == 1, (
                f"Only batch_size=1 supported, got coord shape {tuple(coord.shape)}"
            )
            coord = coord[0]

        if feat.ndim == 3:
            assert feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got feat shape {tuple(feat.shape)}"
            )
            feat = feat[0]

        assert coord.ndim == 2 and coord.shape[1] == 3, (
            f"coord must be [N, 3], got {tuple(coord.shape)}"
        )
        assert feat.ndim == 2 and feat.shape[1] == self.in_channels, (
            f"feat must be [N, {self.in_channels}], got {tuple(feat.shape)}"
        )
        assert coord.shape[0] == feat.shape[0], (
            f"coord has {coord.shape[0]} points but feat has {feat.shape[0]}"
        )

        return coord.contiguous().float(), feat.contiguous().float()

    # ------------------------------------------------------------------ #

    def _normalize_offsets(
        self,
        point_offsets: torch.Tensor | None,
        *,
        num_points: int,
        device: torch.device,
    ) -> torch.Tensor:
        if point_offsets is None:
            return torch.tensor([num_points], device=device, dtype=torch.long)

        offsets = point_offsets.to(device=device, dtype=torch.long).flatten()
        if offsets.numel() == 0:
            raise ValueError("point_offsets must contain at least one scene")
        if int(offsets[-1].item()) != num_points:
            raise ValueError(
                f"point_offsets[-1]={int(offsets[-1].item())} but coord has {num_points} points"
            )
        prev = 0
        for end in offsets.tolist():
            if end <= prev:
                raise ValueError(
                    f"point_offsets must be strictly increasing, got {offsets.tolist()}"
                )
            prev = end
        return offsets

    def _voxelize_single(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grid-sample to one voxel token and retain its source-point index.

        Returns
        -------
        point_dict : dict
            LitePT input dict with mean-pooled or representative values.
        inverse : Tensor [N]
            Maps each original point to its voxel index.
        scene_xyz : Tensor [V, 3]
            Coordinates supplied to LitePT for each voxel.
        representative_indices : Tensor [V]
            One source point per sorted voxel. Train mode samples uniformly,
            matching LitePT ``GridSample(mode='train')``; eval uses the first
            source point deterministically.
        """
        grid_coord = torch.floor(coord / self.grid_size).to(torch.int64)
        grid_coord = grid_coord - grid_coord.min(dim=0).values

        unique_grid, inverse = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True,
        )

        V = unique_grid.shape[0]

        sorted_source = torch.argsort(inverse, stable=True)
        counts = torch.bincount(inverse, minlength=V)
        starts = torch.cumsum(counts, dim=0) - counts
        representative_indices = sorted_source[starts]

        if self.voxel_reduce == "representative":
            if self.training and self.representative_sampling == "random":
                random_offsets = torch.floor(
                    torch.rand(V, device=coord.device) * counts.to(torch.float32)
                ).long()
                representative_indices = sorted_source[starts + random_offsets]
            scene_xyz = coord[representative_indices]
            scene_feat = feat[representative_indices]
        else:
            voxel_counts = torch.bincount(inverse, minlength=V).float().clamp(min=1.0)

            scene_xyz = torch.zeros(V, 3, device=coord.device, dtype=coord.dtype)
            scene_xyz.index_add_(0, inverse, coord)
            scene_xyz = scene_xyz / voxel_counts[:, None]

            scene_feat = torch.zeros(
                V, feat.shape[1], device=feat.device, dtype=feat.dtype,
            )
            scene_feat.index_add_(0, inverse, feat)
            scene_feat = scene_feat / voxel_counts[:, None]

        point_dict = {
            "coord": scene_xyz,
            "grid_coord": unique_grid.int(),
            "feat": scene_feat,
            "offset": torch.tensor([V], device=coord.device, dtype=torch.long),
        }

        return point_dict, inverse, scene_xyz, representative_indices

    # ------------------------------------------------------------------ #

    def _voxelize_batched(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_offsets: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Voxelize a concatenated multi-scene batch while preserving boundaries."""
        inverse = torch.empty(coord.shape[0], device=coord.device, dtype=torch.long)
        grid_parts: list[torch.Tensor] = []
        xyz_parts: list[torch.Tensor] = []
        feat_parts: list[torch.Tensor] = []
        representative_parts: list[torch.Tensor] = []
        voxel_offsets: list[int] = []

        start = 0
        voxel_base = 0
        for end in point_offsets.tolist():
            scene_coord = coord[start:end]
            scene_feat = feat[start:end]
            point_dict, scene_inverse, scene_xyz, scene_representatives = (
                self._voxelize_single(scene_coord, scene_feat)
            )

            grid_parts.append(point_dict["grid_coord"])
            xyz_parts.append(scene_xyz)
            feat_parts.append(point_dict["feat"])
            representative_parts.append(scene_representatives + start)
            inverse[start:end] = scene_inverse + voxel_base

            voxel_base += int(scene_xyz.shape[0])
            voxel_offsets.append(voxel_base)
            start = end

        scene_xyz = torch.cat(xyz_parts, dim=0)
        point_dict = {
            "coord": scene_xyz,
            "grid_coord": torch.cat(grid_parts, dim=0),
            "feat": torch.cat(feat_parts, dim=0),
            "offset": torch.tensor(voxel_offsets, device=coord.device, dtype=torch.long),
        }
        return (
            point_dict,
            inverse,
            scene_xyz,
            point_dict["offset"],
            torch.cat(representative_parts, dim=0),
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_offsets: torch.Tensor | None = None,
    ) -> LitePTBackboneOutput | LitePTFeaturePyramid:
        """Run LitePT and return structured backbone output.

        Parameters
        ----------
        coord : (N, 3) or (1, N, 3) float
        feat  : (N, C) or (1, N, C) float

        Returns
        -------
        :class:`LitePTBackboneOutput` with dense per-point features,
        sparse voxel tokens, coordinates, and inverse mapping.
        When ``multi_scale`` is enabled, also includes per-scale
        decoder features in coarse → fine order.
        """
        coord, feat = self._normalize_inputs(coord, feat)
        point_offsets_t = self._normalize_offsets(
            point_offsets,
            num_points=coord.shape[0],
            device=coord.device,
        )
        batched = point_offsets_t.numel() > 1

        if (
            self.cache_training_voxelization
            and self._cached_voxelization is not None
            and self.training
            and not batched
        ):
            (
                point_dict,
                inverse,
                scene_xyz,
                scene_token_offsets,
                representative_indices,
            ) = self._cached_voxelization
        else:
            if batched:
                (
                    point_dict,
                    inverse,
                    scene_xyz,
                    scene_token_offsets,
                    representative_indices,
                ) = self._voxelize_batched(coord, feat, point_offsets_t)
            else:
                point_dict, inverse, scene_xyz, representative_indices = (
                    self._voxelize_single(coord, feat)
                )
                scene_token_offsets = point_dict["offset"]
                if self.cache_training_voxelization and self.training:
                    self._cached_voxelization = (
                        point_dict,
                        inverse,
                        scene_xyz,
                        scene_token_offsets,
                        representative_indices,
                    )

        self._captured.clear()
        self._fpn_captured.clear()

        out = self.model(point_dict)

        scene_tokens = out.feat          # [V, C]
        point_feat = scene_tokens[inverse]  # [N, C]
        scene_token_offsets = getattr(out, "offset", scene_token_offsets)

        if self.feature_pyramid_mode in {
            "mask3d_fpn",
            "mask3d_dec0_pooled_fpn",
            "mask3d_decoder_only_fpn",
        }:
            expected = ("enc4", "dec3", "dec2", "dec1", "dec0")
            missing = [name for name in expected if name not in self._fpn_captured]
            if missing:
                raise RuntimeError(f"LitePT FPN hooks did not capture stages: {missing}")

            stage_channels = {
                "enc4": 504,
                "dec3": 252,
                "dec2": 144,
                "dec1": 72,
                "dec0": 72,
            }
            stage_strides = {
                "enc4": 16,
                "dec3": 8,
                "dec2": 4,
                "dec1": 2,
                "dec0": 1,
            }
            for name in expected:
                features, _grid, _xyz, _offsets = self._fpn_captured[name]
                if int(features.shape[-1]) != stage_channels[name]:
                    raise RuntimeError(
                        f"LitePT {name} channels={features.shape[-1]}, "
                        f"expected {stage_channels[name]}"
                    )

            hierarchy_fine_to_coarse = ("dec0", "dec1", "dec2", "dec3", "enc4")
            parent_maps: list[torch.Tensor] = []
            for child_name, parent_name in zip(
                hierarchy_fine_to_coarse[:-1], hierarchy_fine_to_coarse[1:]
            ):
                _, child_grid, _, child_offsets = self._fpn_captured[child_name]
                _, parent_grid, _, parent_offsets = self._fpn_captured[parent_name]
                parent_maps.append(
                    self._grid_parent_map(
                        child_grid,
                        child_offsets,
                        parent_grid,
                        parent_offsets,
                        2,
                    )
                )

            finest_features, finest_grid, finest_xyz, finest_offsets = (
                self._fpn_captured["dec0"]
            )
            memory_stage_names = ("enc4", "dec3", "dec2", "dec1")
            if self.feature_pyramid_mode == "mask3d_dec0_pooled_fpn":
                pooled_by_stage: dict[str, torch.Tensor] = {"dec0": finest_features}
                for mapping, child_name, parent_name in zip(
                    parent_maps,
                    hierarchy_fine_to_coarse[:-1],
                    hierarchy_fine_to_coarse[1:],
                ):
                    parent_features = self._fpn_captured[parent_name][0]
                    pooled_by_stage[parent_name] = _mean_pool_features_by_parent(
                        pooled_by_stage[child_name],
                        mapping,
                        int(parent_features.shape[0]),
                    )
                memory_levels = tuple(
                    LitePTPyramidLevel(
                        name=f"dec0_pool_{name}",
                        stride=stage_strides[name],
                        channels=72,
                        features=pooled_by_stage[name],
                        grid=self._fpn_captured[name][1],
                        xyz=self._fpn_captured[name][2],
                        offsets=self._fpn_captured[name][3],
                    )
                    for name in memory_stage_names
                )
            elif self.feature_pyramid_mode == "mask3d_decoder_only_fpn":
                # Downstream ablation: never give Mask3D native enc4 features.
                # The coarse memory still lives on the exact enc4 grid, but is a
                # differentiable dec3 descendant mean through LitePT's native
                # dec3->enc4 parent map. This preserves the four-level decoder
                # geometry without treating a weak enc4 representation as memory.
                dec3_features = self._fpn_captured["dec3"][0]
                enc4_features = self._fpn_captured["enc4"][0]
                dec3_on_enc4 = _mean_pool_features_by_parent(
                    dec3_features,
                    parent_maps[3],
                    int(enc4_features.shape[0]),
                )
                memory_levels = (
                    LitePTPyramidLevel(
                        name="dec3_pool_enc4",
                        stride=stage_strides["enc4"],
                        channels=stage_channels["dec3"],
                        features=dec3_on_enc4,
                        grid=self._fpn_captured["enc4"][1],
                        xyz=self._fpn_captured["enc4"][2],
                        offsets=self._fpn_captured["enc4"][3],
                    ),
                    *(
                        LitePTPyramidLevel(
                            name=name,
                            stride=stage_strides[name],
                            channels=stage_channels[name],
                            features=self._fpn_captured[name][0],
                            grid=self._fpn_captured[name][1],
                            xyz=self._fpn_captured[name][2],
                            offsets=self._fpn_captured[name][3],
                        )
                        for name in ("dec3", "dec2", "dec1")
                    ),
                )
            else:
                memory_levels = tuple(
                    LitePTPyramidLevel(
                        name=name,
                        stride=stage_strides[name],
                        channels=stage_channels[name],
                        features=self._fpn_captured[name][0],
                        grid=self._fpn_captured[name][1],
                        xyz=self._fpn_captured[name][2],
                        offsets=self._fpn_captured[name][3],
                    )
                    for name in memory_stage_names
                )
            return LitePTFeaturePyramid(
                finest_features=finest_features,
                finest_grid=finest_grid,
                finest_xyz=finest_xyz,
                finest_offsets=finest_offsets,
                memory_levels=memory_levels,
                point_xyz=coord,
                point_offsets=point_offsets_t,
                original_to_finest=inverse,
                representative_indices=representative_indices,
                parent_maps=tuple(parent_maps),
            )

        hierarchy_stage_names: tuple[str, ...] = ()
        hierarchy_tokens: tuple[torch.Tensor, ...] = ()
        hierarchy_xyz: tuple[torch.Tensor, ...] = ()
        hierarchy_grids: tuple[torch.Tensor, ...] = ()
        hierarchy_offsets: tuple[torch.Tensor, ...] = ()
        hierarchy_parent_maps: tuple[torch.Tensor, ...] = ()
        if self.capture_hierarchy:
            hierarchy_stage_names = ("enc4", "dec3", "dec2", "dec1", "dec0")
            missing = [
                name
                for name in hierarchy_stage_names
                if name not in self._fpn_captured
            ]
            if missing:
                raise RuntimeError(
                    f"LitePT hierarchy hooks did not capture stages: {missing}"
                )
            expected_channels = {
                "enc4": 504,
                "dec3": 252,
                "dec2": 144,
                "dec1": 72,
                "dec0": 72,
            }
            for name in hierarchy_stage_names:
                features = self._fpn_captured[name][0]
                if features.ndim != 2 or int(features.shape[1]) != expected_channels[name]:
                    raise RuntimeError(
                        f"LitePT {name} hierarchy channels={tuple(features.shape)}, "
                        f"expected [N,{expected_channels[name]}]"
                    )
            hierarchy_tokens = tuple(
                self._fpn_captured[name][0] for name in hierarchy_stage_names
            )
            hierarchy_grids = tuple(
                self._fpn_captured[name][1] for name in hierarchy_stage_names
            )
            hierarchy_xyz = tuple(
                self._fpn_captured[name][2] for name in hierarchy_stage_names
            )
            hierarchy_offsets = tuple(
                self._fpn_captured[name][3] for name in hierarchy_stage_names
            )
            fine_to_coarse = ("dec0", "dec1", "dec2", "dec3", "enc4")
            adjacent: list[torch.Tensor] = []
            for child_name, parent_name in zip(
                fine_to_coarse[:-1], fine_to_coarse[1:], strict=True
            ):
                _, child_grid, _, child_offsets = self._fpn_captured[child_name]
                _, parent_grid, _, parent_offsets = self._fpn_captured[parent_name]
                adjacent.append(
                    self._grid_parent_map(
                        child_grid,
                        child_offsets,
                        parent_grid,
                        parent_offsets,
                        2,
                    )
                )
            hierarchy_parent_maps = tuple(adjacent)

        ms_tokens: list[torch.Tensor] = []
        ms_xyz: list[torch.Tensor] = []
        ms_offsets: list[torch.Tensor] = []
        ms_grids: list[torch.Tensor] = []
        if self._multi_scale and self._captured:
            for i in sorted(self._captured.keys()):
                cap_feat, cap_coord, cap_offset, cap_grid = self._captured[i]
                ms_tokens.append(cap_feat)
                ms_xyz.append(cap_coord)
                ms_offsets.append(cap_offset)
                ms_grids.append(cap_grid)

        finest_grid = out.grid_coord
        finest_to_multi_scale = [
            self._grid_parent_map(
                finest_grid,
                scene_token_offsets,
                level_grid,
                level_offsets,
                divisor,
            )
            for level_grid, level_offsets, divisor in zip(
                ms_grids, ms_offsets, self._multi_scale_divisors
            )
        ]

        return LitePTBackboneOutput(
            point_feat=point_feat,
            point_xyz=coord,
            scene_tokens=scene_tokens,
            scene_xyz=scene_xyz,
            inverse_map=inverse,
            point_offsets=point_offsets_t,
            scene_token_offsets=scene_token_offsets,
            multi_scale_tokens=ms_tokens,
            multi_scale_xyz=ms_xyz,
            multi_scale_offsets=ms_offsets,
            finest_to_multi_scale=finest_to_multi_scale,
            hierarchy_stage_names=hierarchy_stage_names,
            hierarchy_tokens=hierarchy_tokens,
            hierarchy_xyz=hierarchy_xyz,
            hierarchy_grids=hierarchy_grids,
            hierarchy_offsets=hierarchy_offsets,
            hierarchy_parent_maps=hierarchy_parent_maps,
        )
