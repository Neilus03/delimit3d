"""Strict LitePT checkpoint loading and native-feature extraction.

This module is intentionally small and model-facing only.  It keeps the
representation evaluation independent from plotting and from the historical
script layout.  Checkpoints are loaded into the pinned LitePT wrapper,
with a strict 72-D native feature contract and no optimizer/projector state.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.models.litept_wrapper import LitePTBackbone


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_state(payload: Any, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: checkpoint payload is not a mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise KeyError(f"{path}: no state_dict, model_state_dict, or model")


def _map_to_wrapper_key(
    key: str,
    *,
    expected_keys: set[str] | None = None,
) -> str | None:
    key = str(key).removeprefix("module.")
    for prefix in ("backbone.model.", "backbone.", "model."):
        if key.startswith(prefix):
            return "model." + key.removeprefix(prefix)
    candidate = "model." + key
    if expected_keys is None or candidate in expected_keys:
        return candidate
    return None


def load_backbone(
    checkpoint_path: Path,
    *,
    litept_root: Path,
    in_channels: int = 6,
    device: torch.device | str = "cpu",
) -> LitePTBackbone:
    """Load one checkpoint with the canonical LitePT-S* RGBN6 contract."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    model = LitePTBackbone(
        litept_root=str(Path(litept_root).expanduser().resolve(strict=True)),
        in_channels=int(in_channels),
        grid_size=0.02,
        litept_variant="litept_s_star",
        multi_scale=False,
        voxel_reduce="representative",
        representative_sampling="first",
        cache_training_voxelization=False,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = _payload_state(payload, checkpoint_path)
    expected = model.state_dict()
    expected_keys = set(expected)
    mapped: dict[str, torch.Tensor] = {}
    for source_key, tensor in source.items():
        if not torch.is_tensor(tensor):
            continue
        target_key = _map_to_wrapper_key(str(source_key), expected_keys=expected_keys)
        if target_key is not None:
            mapped[target_key] = tensor.detach().cpu()
    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    shape_mismatch = sorted(
        key for key in set(expected) & set(mapped)
        if tuple(expected[key].shape) != tuple(mapped[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"{checkpoint_path}: strict backbone mismatch "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"shape={len(shape_mismatch)}"
        )
    model.load_state_dict(mapped, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def center_shift(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the established center-XY/ground-Z scene coordinate contract."""

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty [N,3] array")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    shift = np.array(
        [(minimum[0] + maximum[0]) / 2.0, (minimum[1] + maximum[1]) / 2.0, minimum[2]],
        dtype=np.float32,
    )
    return (points - shift).astype(np.float32, copy=False), shift


@torch.no_grad()
def infer_features(
    model: LitePTBackbone,
    *,
    points: np.ndarray,
    features: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    """Return native 72-D point features without a projection head."""

    points = np.asarray(points, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if features.ndim != 2 or features.shape[0] != points.shape[0]:
        raise ValueError("features must have shape [N,C] aligned with points")
    coord = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    feat = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    output = model(coord, feat).point_feat.detach().float().cpu().numpy()
    if output.shape != (points.shape[0], 72):
        raise RuntimeError(f"Unexpected native feature shape {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("LitePT native features contain non-finite values")
    return output.astype(np.float32, copy=False)


def configure_attention_backend(device: torch.device, requested: str = "auto") -> str:
    """Patch the optional FlashAttention call for pre-Ampere inference."""

    if device.type != "cuda":
        raise RuntimeError("attention backend selection requires CUDA")
    major, _minor = torch.cuda.get_device_capability(device)
    if requested == "flash" or (requested == "auto" and major >= 8):
        if major < 8:
            raise RuntimeError("FlashAttention requires Ampere or newer")
        return "flash"

    def _sdpa_varlen_qkvpacked(
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen: int,
        dropout_p: float,
        softmax_scale: float | None,
    ) -> torch.Tensor:
        del max_seqlen
        outputs: list[torch.Tensor] = []
        boundaries = cu_seqlens.detach().cpu().tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            sequence = qkv[int(start):int(end)]
            q = sequence[:, 0].permute(1, 0, 2).unsqueeze(0).float()
            k = sequence[:, 1].permute(1, 0, 2).unsqueeze(0).float()
            v = sequence[:, 2].permute(1, 0, 2).unsqueeze(0).float()
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=False, scale=softmax_scale
            )
            outputs.append(attended.squeeze(0).permute(1, 0, 2).to(qkv.dtype))
        if not outputs:
            return qkv.new_empty((0, qkv.shape[2], qkv.shape[3]))
        return torch.cat(outputs, dim=0)

    import flash_attn  # type: ignore
    flash_attn.flash_attn_varlen_qkvpacked_func = _sdpa_varlen_qkvpacked
    return "sdpa"


__all__ = [
    "build_input_features",
    "center_shift",
    "configure_attention_backend",
    "infer_features",
    "load_backbone",
    "sha256_file",
    "_map_to_wrapper_key",
    "_payload_state",
]
