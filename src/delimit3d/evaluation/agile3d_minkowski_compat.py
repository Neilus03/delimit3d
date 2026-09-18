"""Minimal MinkowskiEngine utility compatibility for the AGILE3D dataset.

The released AGILE3D dataset imports MinkowskiEngine for two utility
functions even when its sparse-convolution backbone is replaced: quantizing
coordinates and adding a batch column.  Euler's LitePT environment does not
ship the compiled MinkowskiEngine extension, so this module supplies only
those data utilities.  It is deliberately not a sparse-convolution backend.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import torch


def _sparse_quantize(
    *,
    coordinates: np.ndarray,
    quantization_size: float | np.ndarray | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
    **kwargs: Any,
) -> Any:
    """Match the AGILE3D dataset's MinkowskiEngine utility call.

    The released dataset passes NumPy coordinates and requests the quantized
    coordinates plus the representative and inverse maps.  ``np.unique``
    gives the same lexicographically ordered unique-coordinate representation
    used by the utility contract, while the maps keep features, labels, and
    inverse expansion aligned.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported sparse_quantize arguments: {unsupported}")
    values = np.asarray(coordinates)
    if values.ndim != 2:
        raise ValueError(f"coordinates must be a rank-2 array, got {values.shape}")

    if quantization_size is None:
        quantized = values
    else:
        size = np.asarray(quantization_size, dtype=np.float64)
        if size.ndim == 0:
            size = np.full(values.shape[1], float(size), dtype=np.float64)
        if size.shape != (values.shape[1],) or np.any(size <= 0):
            raise ValueError(
                "quantization_size must be a positive scalar or one value per coordinate dimension"
            )
        quantized = np.floor(values / size).astype(np.int32, copy=False)

    unique_coords, unique_index, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    result: list[Any] = [unique_coords]
    if return_index:
        result.append(unique_index.astype(np.int64, copy=False))
    if return_inverse:
        result.append(inverse.astype(np.int64, copy=False))
    if len(result) == 1:
        return result[0]
    return tuple(result)


def _batched_coordinates(coordinates: list[np.ndarray] | tuple[np.ndarray, ...]) -> torch.Tensor:
    """Prepend the scene index in the format expected by AGILE3D collation."""
    arrays = [np.asarray(value, dtype=np.int32) for value in coordinates]
    if not arrays:
        return torch.empty((0, 0), dtype=torch.int32)
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("each coordinate array must be rank 2")
    width = arrays[0].shape[1]
    if any(value.shape[1] != width for value in arrays):
        raise ValueError("all coordinate arrays must have the same width")
    batched = np.concatenate(
        [
            np.concatenate(
                [np.full((value.shape[0], 1), batch_id, dtype=np.int32), value],
                axis=1,
            )
            for batch_id, value in enumerate(arrays)
        ],
        axis=0,
    )
    return torch.from_numpy(batched)


def install_minkowski_engine_compat() -> str:
    """Keep native MinkowskiEngine when present, otherwise install the shim."""
    try:
        import MinkowskiEngine  # noqa: F401

        return "native"
    except ModuleNotFoundError as error:
        if error.name != "MinkowskiEngine":
            raise

    module = types.ModuleType("MinkowskiEngine")
    utils = types.SimpleNamespace(
        sparse_quantize=_sparse_quantize,
        batched_coordinates=_batched_coordinates,
    )
    module.utils = utils  # type: ignore[attr-defined]
    module.__version__ = "agile3d-utility-compat-v1"
    sys.modules["MinkowskiEngine"] = module
    return "numpy_utility_compat_v1"
