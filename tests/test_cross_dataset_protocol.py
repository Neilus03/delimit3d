from pathlib import Path

import numpy as np
import pytest

from delimit3d.evaluation import cross_dataset_protocol as p


def _write_ply(path: Path, labels: list[int]) -> None:
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(labels)
    values = np.zeros(
        count,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("R", "u1"),
            ("G", "u1"),
            ("B", "u1"),
            ("label", "i4"),
        ],
    )
    values["x"] = np.arange(count, dtype=np.float32)
    values["y"] = np.arange(count, dtype=np.float32) * 0.1
    values["z"] = 0.25
    values["R"] = 10
    values["G"] = 20
    values["B"] = 30
    values["label"] = labels
    PlyData([PlyElement.describe(values, "vertex")], text=True).write(path)


def _trace(count=2):
    return {
        "scene": "sceneA",
        "object_count": count,
        "object_ids": list(range(10, 10 + count)),
        "states": [
            {"total_clicks": total, "mean_iou": min(1.0, total / float(20 * count))}
            for total in range(20 * count + 1)
        ],
    }


def test_official_multi_list_preserves_local_to_raw_mapping(tmp_path):
    path = tmp_path / "val_list.json"
    path.write_text(
        '{"sceneB_obj_1":{"clicks":{"1":[]},"obj":{"1":37003}},'
        '"sceneA_obj_2":{"clicks":{"1":[],"2":[],"0":[]},'
        '"obj":{"1":28.0,"2":21.0}}}'
    )
    result = p.read_official_multi_list(path)
    assert list(result) == ["sceneA_obj_2", "sceneB_obj_1"]
    assert result["sceneA_obj_2"]["scene"] == "sceneA"
    assert result["sceneA_obj_2"]["object_count"] == 2
    assert result["sceneA_obj_2"]["obj"] == {"1": 28, "2": 21}
    assert result["sceneA_obj_2"]["clicks"] == {"0": [], "1": [], "2": []}


def test_official_multi_list_rejects_noncontiguous_local_labels(tmp_path):
    path = tmp_path / "val_list.json"
    path.write_text('{"sceneA_obj_2":{"obj":{"1":28,"3":21},"clicks":{}}}')
    with pytest.raises(ValueError, match="exactly local labels"):
        p.read_official_multi_list(path)


def test_single_list_retains_zero_based_crop_identity_and_class(tmp_path):
    ids = tmp_path / "object_ids.npy"
    np.save(ids, np.asarray([["sceneA", "0"], ["sceneB", "17"]]))
    classes = tmp_path / "object_classes.txt"
    classes.write_text("chair\ncar\n")
    result = p.read_official_single_list(ids, classes)
    assert result[0]["scene"] == "sceneA"
    assert result[0]["instance"] == 0
    assert result[0]["object_id"] == "0"
    assert result[0]["semantic_name"] == "chair"
    assert result[1]["instance"] == 17


def test_build_and_load_records_preserve_raw_labels_and_binary_crop(tmp_path):
    official = tmp_path / "official"
    _write_ply(official / "scans" / "sceneA.ply", [10, 10, 20, 20, 0])
    _write_ply(
        official / "single" / "crops" / "sceneA" / "sceneA_crop_0.ply",
        [0, 1, 0, 1],
    )
    (official / "val_list.json").write_text(
        '{"sceneA_obj_2":{"clicks":{"0":[],"1":[],"2":[]},'
        '"obj":{"1":10,"2":20}}}'
    )
    np.save(official / "single" / "object_ids.npy", np.asarray([["sceneA", "0"]]))
    (official / "single" / "object_classes.txt").write_text("chair\n")
    normals = tmp_path / "normals"
    normals.mkdir()
    np.save(normals / "sceneA.npy", np.tile([0, 0, 1], (5, 1)).astype(np.float32))
    np.save(
        normals / "sceneA_crop_0.npy",
        np.tile([0, 1, 0], (4, 1)).astype(np.float32),
    )

    records = p.build_official_records(
        official,
        dataset="S3DIS",
        normal_root=normals,
        require_normals=True,
    )
    assert records["dataset"] == "S3DIS"
    assert records["scenes"][0]["official_object_ids"] == [10, 20]
    assert records["MO"][0]["local_to_official"] == {"1": 10, "2": 20}
    assert records["SO"][0]["instance"] == 0
    assert records["SO"][0]["data_role"] == "single_crop"
    assert records["SO"][0]["normal_path"].endswith("sceneA_crop_0.npy")

    _, _, _, labels, objects, masks = p.load_scene_with_labels(records["MO"][0])
    np.testing.assert_array_equal(labels, [10, 10, 20, 20, 0])
    assert [item["instance"] for item in objects] == [10, 20]
    np.testing.assert_array_equal(masks[10], [0, 1])
    np.testing.assert_array_equal(masks[20], [2, 3])

    _, _, normals_loaded, labels_crop, objects_crop, masks_crop = p.load_scene_with_labels(
        records["SO"][0]
    )
    np.testing.assert_array_equal(labels_crop, [0, 1, 0, 1])
    np.testing.assert_array_equal(normals_loaded[:, 1], [1, 1, 1, 1])
    assert [item["instance"] for item in objects_crop] == [0]
    np.testing.assert_array_equal(masks_crop[0], [1, 3])


def test_flat_normal_path_and_hash_drift_are_checked(tmp_path):
    normal = tmp_path / "scene.npy"
    np.save(normal, np.ones((3, 3), dtype=np.float32))
    path = p.find_aligned_normal_path(tmp_path, "scene")
    assert path == normal
    expected = p.file_sha256(normal)
    np.testing.assert_array_equal(
        p.load_aligned_normals(tmp_path, "scene", 3, expected_sha256=expected),
        np.ones((3, 3), dtype=np.float32),
    )
    normal.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        p.load_aligned_normals(tmp_path, "scene", 3, expected_sha256=expected)


def test_initial_clicks_are_deterministic_and_one_per_object():
    xyz = np.asarray(
        [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [1, 0, 0], [1.1, 0, 0]],
        dtype=np.float32,
    )
    target = np.asarray([1, 1, 0, 2, 2])
    first = p.initial_clicks(target, xyz, 77)
    second = p.initial_clicks(target, xyz, 77)
    assert first == second
    assert set(first[0]) == {"0", "1", "2"}
    assert all(len(first[0][label]) == 1 for label in ("1", "2"))


def test_metrics_cover_twenty_clicks_and_official_csv_accounting():
    trace = _trace()
    metrics = p.metrics_from_trace(trace)
    assert tuple(key for key in metrics if key.startswith("IoU")) == tuple(
        f"IoU@{click}" for click in p.CLICKS
    )
    assert metrics["IoU@20"] == 1.0
    assert metrics["NoC@80"] == 16.0
    mo = p.official_lines(3, trace, "MO")
    assert mo[0] == "3 A 2 0.0 0.0\n"
    assert mo[2] == "3 A 2 1.0 0.05\n"
    assert mo[-1] == "3 A 2 20.0 1.0\n"
    so = p.official_lines(4, {"scene": "sceneA", "object_ids": [0], "object_count": 1, "states": _trace(1)["states"]}, "SO")
    assert so[0] == "4 A 0 0 0.0\n"
    assert so[-1] == "4 A 0 20 1.0\n"


def test_paired_bootstrap_is_exactly_paired_and_has_all_metrics():
    metric_names = [f"IoU@{value}" for value in p.CLICKS] + [
        f"NoC@{value}" for value in p.THRESHOLDS
    ]
    public = {
        "sceneA": {name: 0.2 for name in metric_names},
        "sceneB": {name: 0.3 for name in metric_names},
    }
    adapted = {
        key: {name: value + 0.1 for name, value in row.items()}
        for key, row in public.items()
    }
    result = p.paired_bootstrap(public, adapted, 100, 3)
    assert set(result) == set(metric_names)
    assert result["IoU@20"]["delta"] == pytest.approx(0.1)
    assert result["IoU@20"]["ci95_lower"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="paired identities"):
        p.paired_bootstrap(public, {"sceneA": adapted["sceneA"]}, 100, 3)


def test_so_csv_prefers_original_object_index_over_binary_target_label():
    trace = {
        "scene": "0000000002_0000000385",
        "object_count": 1,
        "object_ids": [1],
        "official_object_index": 0,
        "states": [{"total_clicks": 0, "mean_iou": 0.0}],
    }
    assert p.official_lines(0, trace, "SO") == ["0 0000000002_0000000385 0 0 0.0\n"]


def test_official_lines_use_original_scene_for_single_crop():
    trace = {
        "scene": "base__crop_0",
        "official_scene": "base",
        "object_count": 1,
        "object_ids": [1],
        "official_object_index": 0,
        "states": [{"total_clicks": 0, "mean_iou": 0.0}],
    }
    assert p.official_lines(0, trace, "SO") == ["0 base 0 0 0.0\n"]
