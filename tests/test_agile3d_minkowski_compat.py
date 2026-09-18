import numpy as np
import torch

from delimit3d.evaluation.agile3d_minkowski_compat import (
    _batched_coordinates,
    _sparse_quantize,
)


def test_sparse_quantize_returns_aligned_maps() -> None:
    coordinates = np.array(
        [[0.01, 0.01, 0.01], [0.04, 0.01, 0.01], [0.06, 0.01, 0.01]],
        dtype=np.float32,
    )
    quantized, unique_index, inverse = _sparse_quantize(
        coordinates=coordinates,
        quantization_size=0.05,
        return_index=True,
        return_inverse=True,
    )

    assert quantized.tolist() == [[0, 0, 0], [1, 0, 0]]
    assert unique_index.tolist() == [0, 2]
    assert inverse.tolist() == [0, 0, 1]


def test_batched_coordinates_prepends_scene_index() -> None:
    result = _batched_coordinates(
        [
            np.array([[1, 2, 3]], dtype=np.int32),
            np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int32),
        ]
    )

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32
    assert result.tolist() == [[0, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]]
