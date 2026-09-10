"""Write compact label-coloured PLY files for source-preparation audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def labels_to_vertex_colors(
    labels: np.ndarray,
    unlabeled_color: tuple[int, int, int] = (80, 80, 80),
    seed: int = 42,
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    colors = np.empty((labels.shape[0], 3), dtype=np.uint8)
    colors[:] = np.array(unlabeled_color, dtype=np.uint8)
    valid_mask = labels >= 0
    if np.any(valid_mask):
        max_label = int(labels[valid_mask].max())
        rng = np.random.default_rng(seed)
        palette = rng.integers(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        colors[valid_mask] = palette[labels[valid_mask]]
    return colors


def save_labeled_mesh_ply(
    source_ply_path: Path,
    labels: np.ndarray,
    out_path: Path,
    unlabeled_color: tuple[int, int, int] = (80, 80, 80),
) -> None:
    source_ply_path = Path(source_ply_path)
    out_path = Path(out_path)
    plydata = PlyData.read(str(source_ply_path))
    if "vertex" not in plydata:
        raise RuntimeError(f"PLY file has no vertex element: {source_ply_path}")
    vertex_data = plydata["vertex"].data
    n_vertices = len(vertex_data)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != n_vertices:
        raise ValueError(
            f"Label length mismatch for {source_ply_path}: "
            f"got {labels.shape[0]}, expected {n_vertices}"
        )
    vertex_colors = labels_to_vertex_colors(labels, unlabeled_color=unlabeled_color)
    out_vertices = np.empty(
        n_vertices,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    out_vertices["x"] = np.asarray(vertex_data["x"], dtype=np.float32)
    out_vertices["y"] = np.asarray(vertex_data["y"], dtype=np.float32)
    out_vertices["z"] = np.asarray(vertex_data["z"], dtype=np.float32)
    out_vertices["red"] = vertex_colors[:, 0]
    out_vertices["green"] = vertex_colors[:, 1]
    out_vertices["blue"] = vertex_colors[:, 2]
    elements = [PlyElement.describe(out_vertices, "vertex")]
    face_element = next((el for el in plydata.elements if el.name == "face"), None)
    if face_element is not None and len(face_element.data) > 0:
        elements.append(PlyElement.describe(face_element.data, "face"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=False).write(str(out_path))
