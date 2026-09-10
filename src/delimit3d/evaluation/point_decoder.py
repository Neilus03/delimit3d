"""A small, frozen-backbone point-conditioned object decoder.

The experiment in :mod:`scripts.evaluation.run_point_decoder` deliberately
keeps this module independent from Mask3D and PointGroup.  A decoder receives
one point feature as the click and scores every scene point using only the
frozen native LitePT feature and the clicked feature.  This makes the
comparison a direct test of whether the two encoders expose a useful
point-conditioned object signal to the *same* newly initialized readout.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PointConditionedObjectDecoder(nn.Module):
    """Shared MLP for class-agnostic point-prompted object masks.

    The input is the concatenation of four feature relations:

    ``normalize(point)``, ``normalize(query)``, their difference, and their
    elementwise product.  Coordinates, colors, labels, and a learned encoder
    branch are intentionally absent.  Both encoder arms therefore expose the
    same 72-D feature contract to exactly the same decoder architecture.
    """

    def __init__(
        self,
        feature_dim: int = 72,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        input_dim = 4 * self.feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.GELU(),
            nn.Linear(self.bottleneck_dim, 1),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        query_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return foreground logits for points and one clicked query.

        ``point_features`` may be ``[N, D]`` or ``[B, N, D]``.  A query may
        be ``[D]`` or ``[B, D]``; broadcasting across points is supported.
        """

        if point_features.ndim not in (2, 3):
            raise ValueError("point_features must have shape [N,D] or [B,N,D]")
        if point_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"point_features last dimension must be {self.feature_dim}, "
                f"got {point_features.shape[-1]}"
            )
        if query_feature.ndim not in (1, 2):
            raise ValueError("query_feature must have shape [D] or [B,D]")
        if query_feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"query_feature last dimension must be {self.feature_dim}, "
                f"got {query_feature.shape[-1]}"
            )

        point_unit = F.normalize(point_features, dim=-1, eps=1e-8)
        query_unit = F.normalize(query_feature, dim=-1, eps=1e-8)
        if point_unit.ndim == 2:
            if query_unit.ndim == 2:
                if query_unit.shape[0] != 1:
                    raise ValueError(
                        "a [N,D] point tensor accepts one query or a [D] query"
                    )
                query_unit = query_unit[0]
            query_unit = query_unit.unsqueeze(0).expand(point_unit.shape[0], -1)
        else:
            if query_unit.ndim == 1:
                query_unit = query_unit.view(1, 1, -1).expand(
                    point_unit.shape[0], point_unit.shape[1], -1
                )
            elif query_unit.ndim == 2:
                if query_unit.shape[0] != point_unit.shape[0]:
                    raise ValueError("batch dimensions of point and query features differ")
                query_unit = query_unit.unsqueeze(1).expand(
                    -1, point_unit.shape[1], -1
                )
            else:  # pragma: no cover - guarded above, kept for type checkers
                raise ValueError("invalid query rank")

        relation = torch.cat(
            [point_unit, query_unit, point_unit - query_unit, point_unit * query_unit],
            dim=-1,
        )
        return self.network(relation).squeeze(-1)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state dict with names, dtypes, shapes, and raw bytes."""

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_initialization(
    path: str | Path,
    *,
    seed: int,
    feature_dim: int = 72,
    hidden_dim: int = 128,
    bottleneck_dim: int = 64,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and save one immutable decoder initialization."""

    output = Path(path).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite decoder initialization: {output}")
    torch.manual_seed(int(seed))
    decoder = PointConditionedObjectDecoder(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
    )
    state = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
    tensor_hash = state_sha256(state)
    payload = {
        "schema": "delimit3d_point_decoder_initialization/v1",
        "seed": int(seed),
        "feature_dim": int(feature_dim),
        "hidden_dim": int(hidden_dim),
        "bottleneck_dim": int(bottleneck_dim),
        "state_dict": state,
        "tensor_state_sha256": tensor_hash,
        "metadata": dict(metadata or {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "path": str(output),
        "schema": payload["schema"],
        "seed": int(seed),
        "feature_dim": int(feature_dim),
        "hidden_dim": int(hidden_dim),
        "bottleneck_dim": int(bottleneck_dim),
        "tensor_state_sha256": tensor_hash,
        "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def load_initialization(
    path: str | Path,
    *,
    expected_feature_dim: int = 72,
    expected_hidden_dim: int = 128,
    expected_bottleneck_dim: int = 64,
) -> tuple[PointConditionedObjectDecoder, dict[str, Any]]:
    """Load and strictly verify an immutable decoder initialization."""

    path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.1 compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: decoder initialization is not a mapping")
    if payload.get("schema") != "delimit3d_point_decoder_initialization/v1":
        raise ValueError(f"{path}: unexpected decoder initialization schema")
    dimensions = {
        "feature_dim": expected_feature_dim,
        "hidden_dim": expected_hidden_dim,
        "bottleneck_dim": expected_bottleneck_dim,
    }
    for key, expected in dimensions.items():
        if int(payload.get(key, -1)) != int(expected):
            raise ValueError(
                f"{path}: {key}={payload.get(key)!r}, expected {int(expected)}"
            )
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{path}: missing decoder state_dict")
    observed_hash = state_sha256(state)
    if observed_hash != payload.get("tensor_state_sha256"):
        raise ValueError(f"{path}: decoder initialization tensor hash mismatch")
    decoder = PointConditionedObjectDecoder(**dimensions)
    decoder.load_state_dict(state, strict=True)
    report = {
        "path": str(path),
        "seed": int(payload.get("seed", -1)),
        "tensor_state_sha256": observed_hash,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": dict(payload.get("metadata") or {}),
    }
    return decoder, report


def balanced_feature_rows(
    unit_features: np.ndarray,
    target_indices: np.ndarray,
    query_index: int,
    *,
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic, balanced point/query pairs for one click."""

    features = np.asarray(unit_features, dtype=np.float32)
    target = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[1] <= 0:
        raise ValueError("unit_features must have shape [N,D]")
    if target.size == 0:
        raise ValueError("target_indices must be non-empty")
    n = int(features.shape[0])
    query_index = int(query_index)
    if not 0 <= query_index < n:
        raise IndexError(query_index)
    positive_pool = target[target != query_index]
    positive_pool = np.unique(positive_pool)
    positive_pool = positive_pool[(positive_pool >= 0) & (positive_pool < n)]
    negative_mask = np.ones(n, dtype=bool)
    negative_mask[target[(target >= 0) & (target < n)]] = False
    negative_mask[query_index] = False
    negative_pool = np.flatnonzero(negative_mask)
    if len(positive_pool) == 0 or len(negative_pool) == 0:
        raise ValueError("both positive and negative sampling pools are required")
    count = int(samples_per_class)
    if count <= 0:
        raise ValueError("samples_per_class must be positive")
    pos = rng.choice(positive_pool, size=count, replace=len(positive_pool) < count)
    neg = rng.choice(negative_pool, size=count, replace=len(negative_pool) < count)
    indices = np.concatenate([pos, neg]).astype(np.int64, copy=False)
    labels = np.concatenate(
        [np.ones(count, dtype=np.float32), np.zeros(count, dtype=np.float32)]
    )
    return indices, labels


__all__ = [
    "PointConditionedObjectDecoder",
    "balanced_feature_rows",
    "load_initialization",
    "save_initialization",
    "state_sha256",
]
