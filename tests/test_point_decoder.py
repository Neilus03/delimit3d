from __future__ import annotations

import numpy as np
import torch

from delimit3d.evaluation.point_decoder import (
    PointConditionedObjectDecoder,
    balanced_feature_rows,
    save_initialization,
    state_sha256,
)


def test_point_decoder_shapes_and_batch_broadcasting() -> None:
    torch.manual_seed(7)
    decoder = PointConditionedObjectDecoder(feature_dim=4, hidden_dim=8, bottleneck_dim=4)
    points = torch.randn(5, 4)
    query = torch.randn(4)
    assert decoder(points, query).shape == (5,)
    batched = torch.randn(2, 5, 4)
    queries = torch.randn(2, 4)
    assert decoder(batched, queries).shape == (2, 5)


def test_balanced_rows_are_deterministic_and_exclude_click() -> None:
    features = np.arange(60, dtype=np.float32).reshape(15, 4)
    target = np.array([1, 2, 3, 4, 5, 6])
    first_i, first_y = balanced_feature_rows(
        features, target, 2, samples_per_class=20, rng=np.random.default_rng(42)
    )
    second_i, second_y = balanced_feature_rows(
        features, target, 2, samples_per_class=20, rng=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(first_i, second_i)
    np.testing.assert_array_equal(first_y, second_y)
    assert not np.any(first_i == 2)
    assert int(first_y.sum()) == 20
    assert int((first_y == 0).sum()) == 20
    assert np.all(np.isin(first_i[first_y == 1], target))
    assert not np.any(np.isin(first_i[first_y == 0], target))


def test_initialization_hash_is_reproducible(tmp_path) -> None:
    path = tmp_path / "decoder.pt"
    report = save_initialization(
        path,
        seed=3,
        feature_dim=4,
        hidden_dim=8,
        bottleneck_dim=4,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert report["tensor_state_sha256"] == state_sha256(payload["state_dict"])
    assert payload["schema"] == "delimit3d_point_decoder_initialization/v1"
