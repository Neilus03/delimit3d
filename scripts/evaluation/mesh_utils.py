"""Deterministic point-to-surface utilities for interactive-segmentation figures.

Mesh outputs are descriptive visualizations. They never feed back into metrics.
The mesh is built from the exact raw PLY points selected for a close-up and is
colored from the saved point-level prediction at a declared click budget.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Iterable

import numpy as np


def deterministic_subset(points: np.ndarray, max_points: int = 12000) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if len(points) <= int(max_points):
        return np.arange(len(points), dtype=np.int64)
    # Farthest-in-index stride is stable across runs and avoids RNG state.
    return np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)


def mesh_from_points(
    points: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    max_points: int = 12000,
    knn: int = 30,
    radii: Iterable[float] = (0.04, 0.08, 0.16),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices, triangular faces, and retained source indices.

    Open3D ball-pivoting is preferred because it preserves local surfaces.
    A SciPy convex hull is a deterministic visual fallback when the point set
    is too small or Open3D cannot construct triangles.
    """
    points = np.asarray(points, dtype=np.float32)
    keep = deterministic_subset(points, max_points)
    vertices = points[keep]
    if len(vertices) < 4:
        return vertices, np.empty((0, 3), dtype=np.int64), keep
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices.astype(np.float64)))
        if normals is not None:
            values = np.asarray(normals, dtype=np.float32)[keep]
            if values.shape == vertices.shape and np.isfinite(values).all():
                cloud.normals = o3d.utility.Vector3dVector(values.astype(np.float64))
        if not cloud.has_normals():
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(
                    knn=min(int(knn), max(3, len(vertices) - 1))
                )
            )
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector([float(value) for value in radii]),
        )
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        if faces.ndim == 2 and faces.shape[1] == 3 and len(faces):
            return vertices, faces, keep
    except Exception:
        pass
    try:
        from scipy.spatial import ConvexHull

        faces = np.asarray(ConvexHull(vertices).simplices, dtype=np.int64)
        if faces.ndim == 2 and faces.shape[1] == 3 and len(faces):
            return vertices, faces, keep
    except Exception:
        pass
    return vertices, np.empty((0, 3), dtype=np.int64), keep


def labels_to_colors(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    local_id: int,
) -> np.ndarray:
    """Color points as TP/FP/FN/background for one requested object."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must be aligned")
    out = np.empty((len(predicted), 3), dtype=np.float32)
    tp = (predicted == int(local_id)) & (target == int(local_id))
    fp = (predicted == int(local_id)) & ~tp
    fn = (target == int(local_id)) & ~tp
    out[:] = (0.72, 0.74, 0.78)
    out[tp] = (0.05, 0.62, 0.46)
    out[fp] = (0.92, 0.24, 0.20)
    out[fn] = (0.98, 0.63, 0.12)
    return out


def write_mesh_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be [M,3]")
    if colors is None:
        rgb = np.full((len(vertices), 3), 190, dtype=np.uint8)
    else:
        rgb = np.clip(np.asarray(colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        if rgb.shape != (len(vertices), 3):
            raise ValueError("colors must align with vertices")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for point, color in zip(vertices, rgb, strict=True):
            handle.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def face_colors(vertex_colors: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return vertex_colors[faces].mean(axis=1)


def write_mesh_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
