from __future__ import annotations

import numpy as np

from delimit3d.source.common.types import VisibilityConfig


def compute_visible_points(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    valid_indices: np.ndarray,
    depth_map_m: np.ndarray,
    visibility_cfg: VisibilityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth_map_m.shape

    finite_mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    u = u[finite_mask]
    v = v[finite_mask]
    z = z[finite_mask]
    valid_indices = valid_indices[finite_mask]

    u_int = u.astype(np.int32)
    v_int = v.astype(np.int32)

    valid_uv = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    u_int = u_int[valid_uv]
    v_int = v_int[valid_uv]
    z = z[valid_uv]
    original_indices = valid_indices[valid_uv]

    z_depth = depth_map_m[v_int, u_int]
    is_visible = (
        (z_depth > visibility_cfg.min_depth_m)
        & np.isfinite(z_depth)
        & (np.abs(z - z_depth) < visibility_cfg.z_tolerance_m)
    )

    return original_indices[is_visible], u_int[is_visible], v_int[is_visible]