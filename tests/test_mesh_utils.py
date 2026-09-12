from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))
from mesh_utils import labels_to_colors, deterministic_subset, write_mesh_ply


def test_deterministic_subset_is_stable():
    points = np.arange(300, dtype=np.float32).reshape(100, 3)
    first = deterministic_subset(points, 17)
    second = deterministic_subset(points, 17)
    assert np.array_equal(first, second)
    assert len(first) == 17
    assert first[0] == 0 and first[-1] == 99


def test_labels_to_colors_marks_tp_fp_fn():
    predicted = np.array([1, 0, 1, 2], dtype=np.int64)
    target = np.array([1, 1, 0, 0], dtype=np.int64)
    colors = labels_to_colors(predicted, target, local_id=1)
    assert tuple(colors[0]) == (0.05, 0.62, 0.46)
    assert tuple(colors[1]) == (0.98, 0.63, 0.12)
    assert tuple(colors[2]) == (0.92, 0.24, 0.20)


def test_write_mesh_ply_is_readable(tmp_path):
    path = tmp_path / "mesh.ply"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    digest = write_mesh_ply(path, vertices, faces)
    text = path.read_text()
    assert len(digest) == 64
    assert "element vertex 3" in text
    assert "element face 1" in text
    assert text.endswith("3 0 1 2\n")
