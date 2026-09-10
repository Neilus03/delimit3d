from __future__ import annotations

import numpy as np


def project_points_to_image(
    points_3d: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_points = points_3d.shape[0]
    world_to_cam = np.linalg.inv(pose_c2w)

    points_h = np.hstack([points_3d, np.ones((num_points, 1), dtype=points_3d.dtype)])
    points_cam = (world_to_cam @ points_h.T).T

    valid_z = points_cam[:, 2] > 0
    valid_indices = np.where(valid_z)[0]

    cam_xyz = points_cam[valid_z, :3]
    points_2d_h = (intrinsics @ cam_xyz.T).T

    u = points_2d_h[:, 0] / points_2d_h[:, 2]
    v = points_2d_h[:, 1] / points_2d_h[:, 2]
    z = cam_xyz[:, 2]

    return u, v, z, valid_indices