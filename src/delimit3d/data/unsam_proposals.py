"""Deterministic serialization for overlapping UnSAM proposals.

The legacy teacher writes one integer label per pixel, which destroys
proposal overlap.  The contrastive-pretraining data path needs both views:

* the complete ordered proposal set for the raw-overlap fidelity experiment;
* the legacy area-ordered first-mask-wins partition as a paired control.

Masks are bit-packed before ``np.savez_compressed`` so storage scales with
image area rather than an uncompressed ``K x H x W`` boolean tensor.

This module intentionally lives at the top-level ``delimit3d`` package so source
preparation does not import the complete training-data package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "unsam_overlapping_proposals/v1"


@dataclass(frozen=True)
class UnSAMProposalSet:
    """One frame's deterministically ordered overlapping proposals."""

    masks: np.ndarray  # bool [K, H, W]
    areas: np.ndarray  # int64 [K], recomputed from masks
    reported_areas: np.ndarray  # float32 [K], teacher metadata
    predicted_ious: np.ndarray  # float32 [K]
    stability_scores: np.ndarray  # float32 [K]
    bboxes: np.ndarray  # float32 [K, 4], XYWH
    source_indices: np.ndarray  # int32 [K], position in teacher output

    def __post_init__(self) -> None:
        if self.masks.ndim != 3 or self.masks.dtype != np.bool_:
            raise ValueError(
                f"masks must be bool [K,H,W], got {self.masks.dtype} {self.masks.shape}"
            )
        count = int(self.masks.shape[0])
        expected_vectors = {
            "areas": (self.areas, (count,)),
            "reported_areas": (self.reported_areas, (count,)),
            "predicted_ious": (self.predicted_ious, (count,)),
            "stability_scores": (self.stability_scores, (count,)),
            "source_indices": (self.source_indices, (count,)),
            "bboxes": (self.bboxes, (count, 4)),
        }
        for name, (value, shape) in expected_vectors.items():
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
        recomputed = self.masks.reshape(count, -1).sum(axis=1, dtype=np.int64)
        if not np.array_equal(self.areas.astype(np.int64, copy=False), recomputed):
            raise ValueError("areas do not match the serialized masks")

    @property
    def image_shape(self) -> tuple[int, int]:
        return int(self.masks.shape[1]), int(self.masks.shape[2])

    @property
    def num_proposals(self) -> int:
        return int(self.masks.shape[0])


def proposals_from_teacher_output(
    masks_data: Sequence[Mapping[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> UnSAMProposalSet:
    """Validate and deterministically order raw teacher proposals.

    The ordering intentionally matches ``UnSAMv2Teacher.run_on_frames``:
    descending reported area with stable teacher-output order for ties.
    Empty masks and entries without a segmentation are omitted because they
    cannot create a legacy label or a valid contrastive proposal.
    """

    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    validated: list[dict[str, Any]] = []
    for source_index, mask_dict in enumerate(masks_data):
        segmentation = mask_dict.get("segmentation")
        if segmentation is None:
            continue
        mask = np.asarray(segmentation, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"Proposal {source_index} shape {mask.shape} != {(height, width)}"
            )
        actual_area = int(mask.sum(dtype=np.int64))
        if actual_area <= 0:
            continue
        reported_area = float(mask_dict.get("area", actual_area))
        bbox = np.asarray(mask_dict.get("bbox", [np.nan] * 4), dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f"Proposal {source_index} bbox shape {bbox.shape} != (4,)")
        validated.append(
            {
                "mask": mask,
                "actual_area": actual_area,
                "reported_area": reported_area,
                "predicted_iou": float(mask_dict.get("predicted_iou", np.nan)),
                "stability_score": float(mask_dict.get("stability_score", np.nan)),
                "bbox": bbox,
                "source_index": source_index,
            }
        )

    validated.sort(key=lambda item: -item["reported_area"])
    if validated:
        masks = np.stack([item["mask"] for item in validated]).astype(bool, copy=False)
    else:
        masks = np.zeros((0, height, width), dtype=bool)

    return UnSAMProposalSet(
        masks=masks,
        areas=np.asarray([item["actual_area"] for item in validated], dtype=np.int64),
        reported_areas=np.asarray(
            [item["reported_area"] for item in validated], dtype=np.float32
        ),
        predicted_ious=np.asarray(
            [item["predicted_iou"] for item in validated], dtype=np.float32
        ),
        stability_scores=np.asarray(
            [item["stability_score"] for item in validated], dtype=np.float32
        ),
        bboxes=np.asarray([item["bbox"] for item in validated], dtype=np.float32).reshape(
            -1, 4
        ),
        source_indices=np.asarray(
            [item["source_index"] for item in validated], dtype=np.int32
        ),
    )


def flatten_proposals(
    proposal_set: UnSAMProposalSet,
    *,
    mask_dtype: np.dtype | str = np.int32,
) -> np.ndarray:
    """Reproduce the legacy first-mask-wins partition from ordered proposals."""

    dtype = np.dtype(mask_dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"mask_dtype must be integral, got {dtype}")
    flat = np.zeros(proposal_set.image_shape, dtype=dtype)
    local_mask_id = 1
    for mask in proposal_set.masks:
        fill = mask & (flat == 0)
        if not bool(fill.any()):
            continue
        if local_mask_id > np.iinfo(dtype).max:
            raise RuntimeError(
                f"Too many flattened masks for dtype={dtype}: {local_mask_id}"
            )
        flat[fill] = local_mask_id
        local_mask_id += 1
    return flat


def proposal_overlap_statistics(proposal_set: UnSAMProposalSet) -> dict[str, Any]:
    """Return compact overlap/flattening diagnostics for one frame."""

    if proposal_set.num_proposals == 0:
        return {
            "num_raw_proposals": 0,
            "num_flat_proposals": 0,
            "raw_mask_pixels": 0,
            "union_pixels": 0,
            "overlap_pixels": 0,
            "overlap_fraction_of_union": 0.0,
        }
    raw_mask_pixels = int(proposal_set.areas.sum(dtype=np.int64))
    union_pixels = int(proposal_set.masks.any(axis=0).sum(dtype=np.int64))
    overlap_pixels = raw_mask_pixels - union_pixels
    flat = flatten_proposals(proposal_set)
    return {
        "num_raw_proposals": proposal_set.num_proposals,
        "num_flat_proposals": int(flat.max()) if flat.size else 0,
        "raw_mask_pixels": raw_mask_pixels,
        "union_pixels": union_pixels,
        "overlap_pixels": overlap_pixels,
        "overlap_fraction_of_union": overlap_pixels / max(union_pixels, 1),
    }


def save_proposal_set(path: Path, proposal_set: UnSAMProposalSet) -> None:
    """Atomically save a bit-packed proposal file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = proposal_set.masks.shape
    num_pixels = int(height * width)
    packed = np.packbits(
        proposal_set.masks.reshape(count, num_pixels), axis=1, bitorder="little"
    )
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            np.savez_compressed(
                tmp,
                schema_version=np.asarray(SCHEMA_VERSION),
                mask_shape=np.asarray([height, width], dtype=np.int32),
                num_pixels=np.asarray(num_pixels, dtype=np.int64),
                packed_masks=packed,
                areas=proposal_set.areas.astype(np.int64, copy=False),
                reported_areas=proposal_set.reported_areas.astype(np.float32, copy=False),
                predicted_ious=proposal_set.predicted_ious.astype(np.float32, copy=False),
                stability_scores=proposal_set.stability_scores.astype(np.float32, copy=False),
                bboxes=proposal_set.bboxes.astype(np.float32, copy=False),
                source_indices=proposal_set.source_indices.astype(np.int32, copy=False),
            )
            tmp.flush()
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(path)


def load_proposal_set(path: Path) -> UnSAMProposalSet:
    """Load and validate one bit-packed proposal file."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        schema = str(payload["schema_version"].item())
        if schema != SCHEMA_VERSION:
            raise ValueError(f"Unsupported proposal schema {schema!r} in {path}")
        height, width = (int(value) for value in payload["mask_shape"].tolist())
        num_pixels = int(payload["num_pixels"].item())
        if num_pixels != height * width:
            raise ValueError(
                f"num_pixels={num_pixels} does not match mask_shape={(height, width)}"
            )
        packed = np.asarray(payload["packed_masks"], dtype=np.uint8)
        if packed.ndim != 2:
            raise ValueError(f"packed_masks must be 2D, got {packed.shape}")
        masks = np.unpackbits(packed, axis=1, count=num_pixels, bitorder="little")
        masks = masks.reshape(packed.shape[0], height, width).astype(bool, copy=False)
        proposal_set = UnSAMProposalSet(
            masks=masks,
            areas=np.asarray(payload["areas"], dtype=np.int64),
            reported_areas=np.asarray(payload["reported_areas"], dtype=np.float32),
            predicted_ious=np.asarray(payload["predicted_ious"], dtype=np.float32),
            stability_scores=np.asarray(payload["stability_scores"], dtype=np.float32),
            bboxes=np.asarray(payload["bboxes"], dtype=np.float32),
            source_indices=np.asarray(payload["source_indices"], dtype=np.int32),
        )
    return proposal_set
