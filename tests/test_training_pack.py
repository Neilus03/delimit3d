from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.data.training_pack import load_training_pack_scene_multi


def _save_linked_array(source_dir: Path, pack_dir: Path, name: str, array: np.ndarray) -> None:
    source_path = source_dir / name
    np.save(source_path, array)
    (pack_dir / name).symlink_to(source_path)


def test_partfield_training_pack_contract_with_rgb_normals_and_extra_meta(tmp_path) -> None:
    scene_dir = tmp_path / "scene0000_00"
    pack_dir = scene_dir / "training_pack"
    source_dir = tmp_path / "arrays"
    pack_dir.mkdir(parents=True)
    source_dir.mkdir()

    n = 5
    points = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    colors = np.full((n, 3), 128, dtype=np.uint8)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
    valid = np.ones(n, dtype=np.uint8)
    labels = {
        "g0.2": np.array([0, 0, 1, 1, 2], dtype=np.int32),
        "g0.5": np.array([0, 0, 0, 0, 1], dtype=np.int32),
        "g0.8": np.array([0, 0, 0, 0, 0], dtype=np.int32),
    }

    _save_linked_array(source_dir, pack_dir, "points.npy", points)
    _save_linked_array(source_dir, pack_dir, "colors.npy", colors)
    _save_linked_array(source_dir, pack_dir, "normals.npy", normals)
    _save_linked_array(source_dir, pack_dir, "valid_points.npy", valid)
    _save_linked_array(source_dir, pack_dir, "seen_points.npy", valid)
    _save_linked_array(source_dir, pack_dir, "supervision_mask.npy", valid)
    for g_key, arr in labels.items():
        _save_linked_array(source_dir, pack_dir, f"labels_{g_key}.npy", arr)

    (pack_dir / "partfield_hierarchy_meta.json").write_text(
        json.dumps({"extra": "ignored"}),
        encoding="utf-8",
    )
    meta = {
        "pack_version": "1.0",
        "scene_id": "scene0000_00",
        "num_points": n,
        "granularities": [0.2, 0.5, 0.8],
        "label_files": {
            "g0.2": "labels_g0.2.npy",
            "g0.5": "labels_g0.5.npy",
            "g0.8": "labels_g0.8.npy",
        },
        "optional_files_present": {
            "colors.npy": True,
            "normals.npy": True,
        },
    }
    (pack_dir / "scene_meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )

    scene = load_training_pack_scene_multi(scene_dir, ("g02", "g05", "g08"))

    assert scene.scene_id == "scene0000_00"
    assert scene.points.shape == (n, 3)
    assert scene.colors is not None and scene.colors.shape == (n, 3)
    assert scene.normals is not None and scene.normals.shape == (n, 3)
    assert sorted(scene.labels_by_granularity) == ["g02", "g05", "g08"]
    assert (scene.training_pack_dir / "partfield_hierarchy_meta.json").exists()

    features = build_input_features(
        scene.points,
        scene.colors,
        use_colors=True,
        use_normals=True,
        normals=scene.normals,
        append_xyz=False,
    )
    assert features.shape == (n, 6)
    assert np.allclose(features[:, :3], 128.0 / 255.0)
    assert np.allclose(features[:, 3:], normals)
