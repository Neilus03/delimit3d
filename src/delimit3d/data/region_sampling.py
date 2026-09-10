"""Point-style region sampling for training (numpy-first).

SphereCrop matches the common Pointcept pattern: pick a random point as the sphere
center, then keep the *nearest* ``point_max`` points (partial sort).

**GridSample** is optional second-phase downsampling: one random point per occupied
voxel. When combined with LitePT (which voxelizes again inside the backbone), you
risk *double* grid quantization—profile end-to-end cost and supervision alignment
before enabling this in the training loop.

The implementation lives at the top-level ``delimit3d`` package so lightweight
audits can use it without importing the complete dataset/training stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def sphere_crop_indices(
    coords: np.ndarray,
    *,
    rng: np.random.Generator,
    point_max: int,
) -> np.ndarray:
    """Indices of up to ``point_max`` points closest to a random center.

    Parameters
    ----------
    coords:
        (N, 3) float point coordinates.
    rng:
        NumPy random generator (seeded by caller for reproducibility).
    point_max:
        Maximum number of points to keep (must be >= 1).

    Returns
    -------
    (K,) int64 indices into ``coords``, where ``K == min(N, point_max)``, sorted
    by increasing distance to the chosen center (deterministic tie-break among
    the kept set).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    n = int(coords.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pm = int(point_max)
    if pm < 1:
        raise ValueError("point_max must be >= 1")

    if n <= pm:
        return np.arange(n, dtype=np.int64)

    center_idx = int(rng.integers(0, n))
    center = coords[center_idx]
    d2 = np.sum((coords - center) ** 2, axis=1, dtype=np.float64)
    part = np.argpartition(d2, pm - 1)[:pm]
    order = np.argsort(d2[part])
    return part[order].astype(np.int64)


def grid_sample_indices(
    coords: np.ndarray,
    *,
    rng: np.random.Generator,
    grid_size: float,
) -> np.ndarray:
    """Train-style voxel pooling: retain one random point per occupied voxel.

    This is intended as an *optional* second-phase downsampler. LitePT already
    voxelizes inside the backbone; stacking this unconditionally can duplicate work
    or misalign supervision—use only after profiling (see module docstring).

    Parameters
    ----------
    coords:
        (N, 3) float coordinates in the same units as ``grid_size``.
    rng:
        NumPy random generator.
    grid_size:
        Voxel edge length (> 0).

    Returns
    -------
    (K,) int64 with ``K <= N``: one index per occupied voxel, chosen uniformly at
    random among points falling in that voxel.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    n = int(coords.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    gs = float(grid_size)
    if not (gs > 0.0):
        raise ValueError("grid_size must be > 0")

    voxel_idx = np.floor(coords / gs).astype(np.int64)
    # Random visitation order: first point in shuffled order that hits a voxel wins.
    order = rng.permutation(n)
    seen: set[tuple[int, int, int]] = set()
    picked: list[int] = []
    for i in order.tolist():
        vx = int(voxel_idx[i, 0])
        vy = int(voxel_idx[i, 1])
        vz = int(voxel_idx[i, 2])
        key = (vx, vy, vz)
        if key in seen:
            continue
        seen.add(key)
        picked.append(i)
    return np.asarray(picked, dtype=np.int64)


def sphere_crop_indices_multi_center(
    coords: np.ndarray,
    *,
    rng: np.random.Generator,
    point_max: int,
    num_fragments: int,
) -> list[np.ndarray]:
    """Several independent sphere crops (different random centers each).

    Used for optional fragment-based full-scene evaluation / merging.
    """
    out: list[np.ndarray] = []
    for _ in range(int(num_fragments)):
        out.append(sphere_crop_indices(coords, rng=rng, point_max=point_max))
    return out
