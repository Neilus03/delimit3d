from __future__ import annotations

import torch
import pytest

from delimit3d.models.litept_wrapper import LitePTBackbone


def test_prequantized_input_preserves_token_order_and_scene_offsets() -> None:
    backbone = object.__new__(LitePTBackbone)
    coord = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [1.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    features = torch.randn(3, 3)
    grid = torch.tensor([[0, 0, 0], [1, 0, 0], [-2, 4, 1]], dtype=torch.int32)
    offsets = torch.tensor([2, 3], dtype=torch.long)

    point_dict, inverse, scene_xyz, scene_offsets, representatives = (
        backbone._prepare_prequantized(coord, features, grid, offsets)
    )

    assert point_dict["feat"] is features
    assert point_dict["coord"] is coord
    assert point_dict["offset"].tolist() == [2, 3]
    assert inverse.tolist() == [0, 1, 2]
    assert scene_xyz is coord
    assert scene_offsets.tolist() == [2, 3]
    assert representatives.tolist() == [0, 1, 2]
    assert point_dict["grid_coord"].tolist() == [[0, 0, 0], [1, 0, 0], [0, 0, 0]]


def test_prequantized_input_rejects_duplicate_tokens() -> None:
    backbone = object.__new__(LitePTBackbone)
    coord = torch.zeros(2, 3)
    features = torch.zeros(2, 3)
    grid = torch.zeros(2, 3, dtype=torch.int32)
    offsets = torch.tensor([2], dtype=torch.long)
    with pytest.raises(ValueError, match="duplicate tokens"):
        backbone._prepare_prequantized(coord, features, grid, offsets)
