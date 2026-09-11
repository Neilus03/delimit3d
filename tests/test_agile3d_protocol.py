from __future__ import annotations

import numpy as np

from delimit3d.evaluation.agile3d_protocol import (
    append_clicks,
    build_token_targets,
    choose_non_overlapping_group,
    dense_click_reference,
    enforce_click_labels,
    kdtree_click_reference,
    metric_at_threshold,
    paired_scene_bootstrap,
    raw_object_ious,
    simulated_corrections,
)


def test_kdtree_matches_dense_click_selection() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    target = np.asarray([1, 1, 0, 0], dtype=np.int64)
    prediction = np.asarray([0, 0, 1, 0], dtype=np.int64)
    dense = dense_click_reference(prediction, target, xyz)
    tree = kdtree_click_reference(prediction, target, xyz)
    assert [
        (row["cluster_id"], row["center_index"], row["target_label"]) for row in dense
    ] == [
        (row["cluster_id"], row["center_index"], row["target_label"]) for row in tree
    ]


def test_group_selection_is_deterministic_and_nonoverlapping() -> None:
    masks = {1: np.asarray([0, 1]), 2: np.asarray([2, 3]), 3: np.asarray([4, 5])}
    first, rejected_first = choose_non_overlapping_group(
        [1, 2, 3], masks, 2, rng=np.random.default_rng(9)
    )
    second, rejected_second = choose_non_overlapping_group(
        [1, 2, 3], masks, 2, rng=np.random.default_rng(9)
    )
    assert first == second
    assert rejected_first == rejected_second


def test_token_targets_and_click_enforcement() -> None:
    representatives = np.asarray([0, 2, 4, 7], dtype=np.int64)
    masks = {1: np.asarray([0, 4]), 2: np.asarray([2, 7])}
    targets = build_token_targets(representatives, masks, [1, 2])
    np.testing.assert_array_equal(targets[1], [0, 2])
    np.testing.assert_array_equal(targets[2], [1, 3])
    prediction = enforce_click_labels(np.zeros(4, dtype=np.int64), {"1": [2], "0": [3]})
    np.testing.assert_array_equal(prediction, [0, 0, 1, 0])


def test_simulator_appends_signed_clicks() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    target = np.asarray([1, 1, 0, 0], dtype=np.int64)
    prediction = np.asarray([1, 0, 0, 0], dtype=np.int64)
    clicks = {"0": [], "1": [0]}
    times = {"0": [], "1": [0]}
    new, new_times, events = simulated_corrections(
        prediction, target, xyz, clicks, times, training=False
    )
    assert events
    assert new
    clicks, times = append_clicks(clicks, times, new, new_times)
    assert sum(len(value) for value in clicks.values()) == 2


def test_raw_iou_and_bootstrap() -> None:
    prediction = np.asarray([0, 1, 1, 0], dtype=np.int64)
    target = np.asarray([0, 1, 0, 0], dtype=np.int64)
    inverse = np.asarray([0, 1, 2, 3], dtype=np.int64)
    assert raw_object_ious(prediction, target, inverse, 1)["1"] == 0.5
    state = [
        {"total_clicks": 1, "mean_iou": 0.2},
        {"total_clicks": 2, "mean_iou": 0.4},
    ]
    assert metric_at_threshold(state, 2)["mean_iou"] == 0.4
    result = paired_scene_bootstrap(
        {"a": 0.1, "b": 0.2},
        {"a": 0.2, "b": 0.1},
        "iou@1",
        samples=100,
        seed=3,
    )
    assert result["observed_difference"] == 0.0

