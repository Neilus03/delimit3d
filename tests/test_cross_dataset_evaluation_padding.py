from pathlib import Path

import numpy as np

from scripts.evaluation.run_cross_dataset_matched import _pad_trace_to_budget
from delimit3d.evaluation import cross_dataset_protocol as protocol


def test_pad_trace_to_budget_adds_exact_states_and_is_idempotent(tmp_path: Path):
    prediction_path = tmp_path / "trace.npz"
    np.savez_compressed(
        prediction_path,
        state_000=np.asarray([1, 0, 0], dtype=np.int16),
        state_001=np.asarray([1, 1, 0], dtype=np.int16),
    )
    trace = {
        "scene": "sceneA",
        "object_count": 1,
        "object_ids": [0],
        "states": [
            {
                "state": 0,
                "total_clicks": 1,
                "clicks_per_object": 1.0,
                "clicks": {"0": [], "1": [0]},
                "click_times": {"0": [], "1": [0]},
                "object_ious": {"1": 0.5},
                "mean_iou": 0.5,
            },
            {
                "state": 1,
                "total_clicks": 2,
                "clicks_per_object": 2.0,
                "clicks": {"0": [], "1": [0, 1]},
                "click_times": {"0": [], "1": [0, 1]},
                "object_ious": {"1": 1.0},
                "mean_iou": 1.0,
            },
        ],
    }
    config = {"protocol": {"click_budget": 20}}
    repaired, metadata = _pad_trace_to_budget(trace, prediction_path, config)
    assert metadata is not None
    assert metadata["from_total_clicks"] == 2
    assert metadata["to_total_clicks"] == 20
    assert [state["total_clicks"] for state in repaired["states"]] == list(range(1, 21))
    assert all(state["perfect_prediction_padded"] for state in repaired["states"][2:])
    with np.load(prediction_path) as archive:
        assert archive.files[-1] == "state_019"
        np.testing.assert_array_equal(archive["state_019"], archive["state_001"])
    metrics = protocol.metrics_from_trace(repaired)
    assert metrics["IoU@20"] == 1.0
    before = prediction_path.read_bytes()
    same, metadata_again = _pad_trace_to_budget(repaired, prediction_path, config)
    assert metadata_again is None
    assert same["states"] == repaired["states"]
    assert prediction_path.read_bytes() == before
