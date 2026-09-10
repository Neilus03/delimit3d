"""Point cloud augmentations aligned with LitePT ScanNet-style training.

Ports the geometric + chromatic stack used in LitePT configs such as
``configs/scannet/insseg-litept-small-v1m2.py`` (upstream:
https://github.com/prs-eth/LitePT).

Applied in ``__getitem__`` so each epoch sees fresh random transforms.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    from scipy import interpolate as _interp
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover
    _interp = None
    _ndimage = None

_HAS_SCIPY = _interp is not None and _ndimage is not None


# Structured3D's official files use a right-handed +Z-up frame.  CHORUS's
# exporter stores them after the explicit cyclic permutation below, so the
# compact source packs are right-handed +Y-up.  V2 canonicalizes those arrays
# to the ScanNet +Z-up model frame with a second *explicit* proper rotation.
# Keep both matrices literal: the runtime contract must never derive one with
# ``inv``/``pinv`` (and thereby make an axis/sign convention implicit).
STRUCTURED3D_OFFICIAL_Z_UP_TO_LOCAL_Y_UP = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float32,
)
STRUCTURED3D_LOCAL_Y_UP_TO_SCANNET_Z_UP = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY = (
    "structured3d_local_y_up_to_scannet_z_up_v1"
)

# These are public contract constants, not mutable augmentation state.
STRUCTURED3D_OFFICIAL_Z_UP_TO_LOCAL_Y_UP.setflags(write=False)
STRUCTURED3D_LOCAL_Y_UP_TO_SCANNET_Z_UP.setflags(write=False)


def _rotation_matrix(angle: float, axis: str) -> np.ndarray:
    rot_cos, rot_sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
    if axis == "y":
        return np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
    if axis == "z":
        return np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
    raise NotImplementedError(axis)


def _normalize_normals_inplace(normals: np.ndarray) -> None:
    """Normalize finite normal vectors while preserving all-zero sentinels."""
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must have shape (N, 3), got {normals.shape}")
    if not np.isfinite(normals).all():
        raise ValueError("normals contain non-finite values")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(
        normals,
        lengths,
        out=normals,
        where=lengths > np.finfo(np.float32).eps,
    )


def canonicalize_structured3d_points_normals_to_scannet(
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate stored Structured3D RGBN6 arrays into ScanNet's +Z-up frame.

    The same determinant-``+1`` matrix is applied to points and normals before
    any train augmentation or unaugmented evaluation.  Row-vector arrays use
    ``x_out = x_in @ R.T``; point order/cardinality is unchanged.  Because this
    is a rigid rotation, normals use the same matrix and are unit-normalized
    only to remove storage roundoff.
    """

    coord = np.asarray(points, dtype=np.float32)
    nrm = np.asarray(normals, dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {coord.shape}")
    if nrm.shape != coord.shape:
        raise ValueError(
            "normals must match Structured3D point shape: "
            f"points={coord.shape} normals={nrm.shape}"
        )
    if not np.isfinite(coord).all():
        raise ValueError("Structured3D points contain non-finite values")
    if not np.isfinite(nrm).all():
        raise ValueError("Structured3D normals contain non-finite values")

    canonical_points = coord @ STRUCTURED3D_LOCAL_Y_UP_TO_SCANNET_Z_UP.T
    canonical_normals = nrm @ STRUCTURED3D_LOCAL_Y_UP_TO_SCANNET_Z_UP.T
    canonical_points = canonical_points.astype(np.float32, copy=False)
    canonical_normals = canonical_normals.astype(np.float32, copy=False)
    _normalize_normals_inplace(canonical_normals)
    return canonical_points, canonical_normals


def _random_rotate(
    coord: np.ndarray,
    angle: tuple[float, float],
    axis: str,
    center: np.ndarray | list[float] | None,
    p: float,
    normals: np.ndarray | None = None,
) -> None:
    if random.random() > p:
        return
    ang = np.random.uniform(angle[0], angle[1]) * np.pi
    rot_t = _rotation_matrix(ang, axis)
    if center is None:
        x_min, y_min, z_min = coord.min(axis=0)
        x_max, y_max, z_max = coord.max(axis=0)
        c = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2],
            dtype=np.float32,
        )
    else:
        c = np.asarray(center, dtype=np.float32)
    coord -= c
    coord[:] = coord @ rot_t.T
    coord += c
    if normals is not None:
        normals[:] = normals @ rot_t.T


def _random_scale(coord: np.ndarray, scale: tuple[float, float]) -> None:
    s = float(np.random.uniform(scale[0], scale[1]))
    coord *= s


def _random_anisotropic_scale(
    coord: np.ndarray,
    scale: tuple[float, float],
    p: float,
    normals: np.ndarray | None = None,
) -> None:
    """Apply Mask3D Scale3d and the inverse-transpose normal transform."""
    if random.random() > p:
        return
    factors = np.random.uniform(scale[0], scale[1], size=(1, 3)).astype(
        np.float32
    )
    coord *= factors
    if normals is not None:
        # For x' = Sx, normals transform as S^{-T}n.  S is diagonal here.
        normals /= factors
        _normalize_normals_inplace(normals)


def _random_flip(
    coord: np.ndarray,
    p: float,
    normals: np.ndarray | None = None,
) -> None:
    if np.random.rand() < p:
        coord[:, 0] *= -1.0
        if normals is not None:
            normals[:, 0] *= -1.0
    if np.random.rand() < p:
        coord[:, 1] *= -1.0
        if normals is not None:
            normals[:, 1] *= -1.0


def _random_flip_mask3d(
    coord: np.ndarray,
    p: float,
    normals: np.ndarray | None = None,
) -> None:
    """Reflect x/y around the scene maximum, matching Mask3D's dataset code."""
    for axis in (0, 1):
        if random.random() >= p:
            continue
        maximum = float(np.max(coord[:, axis]))
        coord[:, axis] = maximum - coord[:, axis]
        if normals is not None:
            normals[:, axis] *= -1.0


def _random_jitter(coord: np.ndarray, sigma: float, clip: float) -> None:
    jitter = np.clip(
        sigma * np.random.randn(*coord.shape).astype(np.float32),
        -clip,
        clip,
    )
    coord += jitter


def _elastic_distortion(
    coord: np.ndarray,
    distortion_params: list[list[float]],
) -> None:
    if not _HAS_SCIPY or not distortion_params:
        return
    if random.random() > 0.95:
        return

    blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3.0
    blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3.0
    blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3.0
    coords_min = coord.min(0)

    for granularity, magnitude in distortion_params:
        noise_dim = ((coord - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        for _ in range(2):
            noise = _ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = _ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = _ndimage.convolve(noise, blurz, mode="constant", cval=0)

        ax = [
            np.linspace(d_min, d_max, d, dtype=np.float32)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = _interp.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coord[:] = coord + interp(coord) * magnitude


def _chromatic_jitter_float01(color: np.ndarray, p: float, std: float) -> None:
    """LitePT uses 0–255 colors with ``std * 255`` noise; here ``color`` is float 0–1."""
    if color is None or color.size == 0:
        return
    if np.random.rand() >= p:
        return
    noise = np.random.randn(color.shape[0], 3).astype(np.float32) * std
    color[:] = np.clip(color + noise, 0.0, 1.0)


def augment_points_litept_scannet(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    use_colors: bool = True,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Copy inputs, apply LitePT ScanNet-style train augmentations, return augmented arrays.

    Order matches ``configs/scannet/insseg-litept-small-v1m2.py``: z/x/y rotate,
    scale, flip, jitter, elastic, chromatic jitter.

    When *normals* is provided, the same **rigid** rotations and axis flips are
    applied to normals. Uniform scale leaves directions unchanged. **Jitter and
    elastic** warp positions only (v1 approximation — normals not updated).
    """
    coord = np.asarray(points, dtype=np.float32).copy()
    color: np.ndarray | None
    if colors is not None and use_colors:
        color = np.asarray(colors, dtype=np.float32).copy()
        if color.max() > 1.0:
            color = color / 255.0
    else:
        color = None

    nrm: np.ndarray | None
    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float32).copy()
    else:
        nrm = None

    _random_rotate(coord, (-1.0, 1.0), "z", [0.0, 0.0, 0.0], p=0.5, normals=nrm)
    _random_rotate(coord, (-1.0 / 64.0, 1.0 / 64.0), "x", None, p=0.5, normals=nrm)
    _random_rotate(coord, (-1.0 / 64.0, 1.0 / 64.0), "y", None, p=0.5, normals=nrm)
    _random_scale(coord, (0.9, 1.1))
    _random_flip(coord, p=0.5, normals=nrm)
    _random_jitter(coord, sigma=0.005, clip=0.02)

    if _HAS_SCIPY:
        _elastic_distortion(coord, [[0.2, 0.4], [0.8, 1.6]])
    if color is not None:
        _chromatic_jitter_float01(color, p=0.95, std=0.05)

    return coord, color, nrm


def _mask3d_color_augment_float01(color: np.ndarray) -> None:
    """Apply Mask3D's RandomBrightnessContrast and RGBShift distributions."""
    if color is None or color.size == 0:
        return

    if random.random() < 0.5:
        brightness = float(np.random.uniform(-0.2, 0.2))
        contrast = float(np.random.uniform(-0.2, 0.2))
        color[:] = np.clip(color * (1.0 + contrast) + brightness, 0.0, 1.0)

    if random.random() < 0.5:
        shift = np.random.randint(-20, 21, size=(1, 3)).astype(np.float32) / 255.0
        color[:] = np.clip(color + shift, 0.0, 1.0)


def augment_points_mask3d_scannet(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    use_colors: bool = True,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply the released Mask3D ScanNet training transform stack.

    The implementation follows ``datasets/semseg.py`` plus the released
    ``volumentations_aug.yaml`` and ``albumentations_aug.yaml``: centering and
    random translation, x/y reflection, elastic distortion, anisotropic
    scaling, z/y/x rotations, then color jitter.  Inputs are copied and point
    order is preserved so GT instance and segment arrays remain aligned.
    """
    coord = np.asarray(points, dtype=np.float32).copy()
    color: np.ndarray | None
    if colors is not None and use_colors:
        color = np.asarray(colors, dtype=np.float32).copy()
        if color.size and color.max() > 1.0:
            color = color / 255.0
    else:
        color = None

    nrm = (
        np.asarray(normals, dtype=np.float32).copy()
        if normals is not None
        else None
    )

    if coord.shape[0] > 0:
        # This is the centering/translation performed by the released
        # SemanticSegmentationDataset before the configured transforms.
        coord -= coord.mean(axis=0, keepdims=True)
        coord += np.random.uniform(coord.min(axis=0), coord.max(axis=0)) / 2.0

        _random_flip_mask3d(coord, p=0.5, normals=nrm)
        if _HAS_SCIPY:
            _elastic_distortion(coord, [[0.2, 0.4], [0.8, 1.6]])

        # volumentations_aug.yaml
        _random_anisotropic_scale(coord, (0.9, 1.1), p=0.5)
        _random_rotate(coord, (-1.0, 1.0), "z", [0.0, 0.0, 0.0], p=0.5, normals=nrm)
        _random_rotate(
            coord,
            (-1.0 / 24.0, 1.0 / 24.0),
            "y",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )
        _random_rotate(
            coord,
            (-1.0 / 24.0, 1.0 / 24.0),
            "x",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )

    if color is not None:
        _mask3d_color_augment_float01(color)
    return coord, color, nrm


def augment_points_mask3d_scannet_rgbn6(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    use_colors: bool = True,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Apply the normal-aware Mask3D ScanNet RGBN6 transform contract.

    This profile retains Mask3D centering/translation, x/y reflection,
    anisotropic scale, z/y/x rotations, and color augmentation.  Elastic
    distortion is intentionally disabled because updating normals under its
    spatially varying deformation would require a local Jacobian or per-epoch
    normal recomputation.  Point order remains unchanged.
    """
    coord = np.asarray(points, dtype=np.float32).copy()
    color: np.ndarray | None
    if colors is not None and use_colors:
        color = np.asarray(colors, dtype=np.float32).copy()
        if color.size and color.max() > 1.0:
            color = color / 255.0
    else:
        color = None

    if normals is None:
        raise ValueError("mask3d_scannet_rgbn6 requires per-point normals")
    nrm = np.asarray(normals, dtype=np.float32).copy()
    if nrm.shape != coord.shape:
        raise ValueError(
            "normals must match point shape for mask3d_scannet_rgbn6: "
            f"points={coord.shape} normals={nrm.shape}"
        )
    _normalize_normals_inplace(nrm)

    if coord.shape[0] > 0:
        coord -= coord.mean(axis=0, keepdims=True)
        coord += np.random.uniform(coord.min(axis=0), coord.max(axis=0)) / 2.0

        _random_flip_mask3d(coord, p=0.5, normals=nrm)

        # Deliberately no elastic distortion in this RGBN6 profile.
        _random_anisotropic_scale(
            coord,
            (0.9, 1.1),
            p=0.5,
            normals=nrm,
        )
        _random_rotate(
            coord,
            (-1.0, 1.0),
            "z",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )
        _random_rotate(
            coord,
            (-1.0 / 24.0, 1.0 / 24.0),
            "y",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )
        _random_rotate(
            coord,
            (-1.0 / 24.0, 1.0 / 24.0),
            "x",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )
        _normalize_normals_inplace(nrm)

    if color is not None:
        _mask3d_color_augment_float01(color)
    return coord, color, nrm


LITEP_SCANNET_DEFAULTS: dict[str, Any] = {
    "description": "LitePT insseg-litept-small-v1m2 train transforms (geom + chromatic)",
}

MASK3D_SCANNET_DEFAULTS: dict[str, Any] = {
    "description": "Released Mask3D ScanNet train transforms",
}

MASK3D_SCANNET_RGBN6_DEFAULTS: dict[str, Any] = {
    "description": (
        "Normal-aware Mask3D ScanNet transforms without elastic distortion"
    ),
    "elastic_distortion": False,
    "normal_transform": (
        "inverse-transpose for anisotropic scale; rigid for flips/rotations"
    ),
}
