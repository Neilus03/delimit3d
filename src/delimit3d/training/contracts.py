#!/usr/bin/env python3
"""Frozen contracts for scratch LitePT GT3 + UnSAM3 pretraining.

This module is intentionally independent of CUDA and the UnSAM/PartField model
implementations.  It owns the parts of the experiment that must be cheap to
unit-test and impossible to silently drift:

* the official ScanNet train/validation and 1,180/21 upstream split;
* the six equally exposed source cells;
* the per-scene epoch-to-cell schedule;
* the deterministic 12/4 RGB-D frame split; the Structured3D 90/10
  train/held-out split; and
* strict validation of compact per-scene source manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "litept_gt3_unsam3_scratch_v1"
HOLDOUT_SALT = "/litept_gt3_unsam3_holdout_v1"
SCENE_SOURCE_SCHEMA = "litept_gt3_unsam3_scene_source/v1"
DATASET_MANIFEST_SCHEMA = "litept_gt3_unsam3_dataset/v1"
RAW_CACHE_SCHEMA = "litept_gt3_unsam3_raw_projection_cache/v1"

GRANULARITY_KEYS = ("g02", "g05", "g08")
SOURCE_FAMILIES = ("2d", "3d")
SOURCE_CELLS = tuple(
    f"{family}/{granularity}"
    for family in SOURCE_FAMILIES
    for granularity in GRANULARITY_KEYS
)
TWO_D_CELLS = tuple(f"2d/{granularity}" for granularity in GRANULARITY_KEYS)
SOURCE_FAMILY_2D3D = "2d3d"
SOURCE_FAMILY_2D = "2d"
SOURCE_FAMILY_CHOICES = (SOURCE_FAMILY_2D3D, SOURCE_FAMILY_2D)
SOURCE_CELL_SCHEDULE_FIXED = "fixed_rotation_v1"
SOURCE_CELL_SCHEDULE_BALANCED_RANDOM = "balanced_random_permutation_v1"
SOURCE_CELL_SCHEDULE_CHOICES = (
    SOURCE_CELL_SCHEDULE_FIXED,
    SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
)
LOGICAL_BATCH_EVAL_EPOCHS = (
    0,
    18,
    36,
    84,
    168,
    252,
    336,
    504,
    672,
    756,
    840,
    924,
    1008,
)

OFFICIAL_TRAIN_SCENES = 1201
OFFICIAL_VAL_SCENES = 312
UPSTREAM_HOLDOUT_SCENES = 21
OPTIMIZATION_SCENES = 1180
WORLD_SIZE = 4
SCENES_PER_RANK = 295
FULL_EPOCHS = 336
TWO_D_EPOCHS = 168
MAX_SCHEDULE_EPOCHS = 1008
PROPOSALS_PER_FORWARD = 4
UPDATES_PER_EPOCH = SCENES_PER_RANK
FULL_UPDATES = FULL_EPOCHS * UPDATES_PER_EPOCH
EXPECTED_SCENE_FORWARDS = FULL_EPOCHS * OPTIMIZATION_SCENES
EXPECTED_PROPOSALS = EXPECTED_SCENE_FORWARDS * PROPOSALS_PER_FORWARD
EXPECTED_PROPOSALS_PER_CELL = EXPECTED_PROPOSALS // len(SOURCE_CELLS)
EXPECTED_PROPOSALS_PER_SCENE_CELL = (
    FULL_EPOCHS // len(SOURCE_CELLS) * PROPOSALS_PER_FORWARD
)

PHYSICAL_FRAMES_PER_SCENE = 16
HELDOUT_FRAME_POSITIONS = (3, 7, 11, 15)
TRAIN_FRAMES_PER_SCENE = 12
HELDOUT_FRAMES_PER_SCENE = 4

# The compact-source tensor/cache contract is shared by ScanNet and
# Structured3D.  Dataset-specific split construction remains outside this
# module, but the per-scene validator must accept both stable ID forms.
_SCENE_ID_RE = re.compile(r"^(?:scene\d{4}_\d{2}|scene_\d{5})$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("/".join(str(part) for part in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def lr_factor(
    *,
    update: int,
    total_updates: int,
    warmup_updates: int,
    min_factor: float = 0.01,
) -> float:
    """Linear warmup followed by cosine decay to ``min_factor``."""

    if not 0 <= int(update) <= int(total_updates):
        raise ValueError("update outside schedule")
    if not 0 < int(warmup_updates) < int(total_updates):
        raise ValueError("warmup_updates must be in (0, total_updates)")
    if int(update) <= int(warmup_updates):
        return max(int(update), 1) / float(warmup_updates)
    progress = (int(update) - int(warmup_updates)) / float(
        int(total_updates) - int(warmup_updates)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_factor + (1.0 - min_factor) * cosine)


def extension_lr_factor(
    *,
    update: int,
    start_update: int,
    total_updates: int,
    warmup_updates: int,
    min_factor: float = 0.01,
) -> float:
    """Warm-restart a completed checkpoint, then cosine-decay to the same floor.

    ``start_update`` is the completed source checkpoint.  Its terminal learning
    rate is ``min_factor`` times each optimizer group's frozen ``initial_lr``.
    The added phase warms continuously from that floor to the original peak,
    then decays back to the floor at ``total_updates``.
    """

    if not 0 <= int(start_update) <= int(update) <= int(total_updates):
        raise ValueError("update outside extension schedule")
    extension_updates = int(total_updates) - int(start_update)
    if not 0 < int(warmup_updates) < extension_updates:
        raise ValueError("warmup_updates must be in (0, extension_updates)")
    if not 0.0 < float(min_factor) <= 1.0:
        raise ValueError("min_factor must be in (0, 1]")
    elapsed = int(update) - int(start_update)
    if elapsed <= int(warmup_updates):
        progress = elapsed / float(warmup_updates)
        return float(min_factor + (1.0 - min_factor) * progress)
    progress = (elapsed - int(warmup_updates)) / float(
        extension_updates - int(warmup_updates)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_factor + (1.0 - min_factor) * cosine)


def normalize_initialization_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the additive initialization-kind field across schema versions."""

    report = dict(value)
    public_checkpoint_used = report.get("public_checkpoint_used")
    if not isinstance(public_checkpoint_used, bool):
        raise ValueError("Initialization report lacks a boolean public-checkpoint flag")
    expected_kind = (
        "public_litept_backbone" if public_checkpoint_used else "scratch"
    )
    observed_kind = report.get("initialization_kind")
    if observed_kind is None:
        report["initialization_kind"] = expected_kind
    elif observed_kind != expected_kind:
        raise ValueError(
            "Initialization kind disagrees with the public-checkpoint flag"
        )
    return report


def load_scene_ids(path: Path) -> list[str]:
    """Load a comment-tolerant scene list and reject ambiguous entries."""

    rows = [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    scene_ids = [row for row in rows if row]
    invalid = [scene_id for scene_id in scene_ids if not _SCENE_ID_RE.fullmatch(scene_id)]
    if invalid:
        raise ValueError(f"Invalid ScanNet scene IDs in {path}: {invalid[:5]}")
    duplicates = sorted(
        scene_id for scene_id in set(scene_ids) if scene_ids.count(scene_id) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate ScanNet scene IDs in {path}: {duplicates[:5]}")
    return scene_ids


def upstream_split(
    train_scene_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return optimization and checkpoint-selection scenes.

    The 21 smallest salted SHA256 values form the holdout.  Optimization order
    remains the official split-file order so materialization is stable without
    changing the existing data traversal convention.
    """

    if len(train_scene_ids) != OFFICIAL_TRAIN_SCENES:
        raise ValueError(
            f"Expected {OFFICIAL_TRAIN_SCENES} official train scenes, "
            f"got {len(train_scene_ids)}"
        )
    if len(set(train_scene_ids)) != len(train_scene_ids):
        raise ValueError("Official train scene list contains duplicates")
    ranked = sorted(
        train_scene_ids,
        key=lambda scene_id: (
            sha256_bytes(f"{scene_id}{HOLDOUT_SALT}".encode("utf-8")),
            scene_id,
        ),
    )
    holdout_set = set(ranked[:UPSTREAM_HOLDOUT_SCENES])
    optimization = [scene_id for scene_id in train_scene_ids if scene_id not in holdout_set]
    holdout = sorted(holdout_set)
    if len(optimization) != OPTIMIZATION_SCENES or len(holdout) != UPSTREAM_HOLDOUT_SCENES:
        raise AssertionError("Upstream split cardinality drift")
    return optimization, holdout


def split_digest(scene_ids: Sequence[str]) -> str:
    """Hash the exact ordered scene sequence, including a terminal newline."""

    return sha256_bytes(("".join(f"{scene_id}\n" for scene_id in scene_ids)).encode("utf-8"))


def frame_split(frame_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split exactly 16 ordered physical frames into deterministic 12/4 sets."""

    if len(frame_ids) != PHYSICAL_FRAMES_PER_SCENE:
        raise ValueError(
            f"Expected {PHYSICAL_FRAMES_PER_SCENE} physical frames, got {len(frame_ids)}"
        )
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Physical frame IDs must be unique")
    heldout_positions = set(HELDOUT_FRAME_POSITIONS)
    train = [
        frame_id for index, frame_id in enumerate(frame_ids) if index not in heldout_positions
    ]
    heldout = [
        frame_id for index, frame_id in enumerate(frame_ids) if index in heldout_positions
    ]
    if len(train) != TRAIN_FRAMES_PER_SCENE or len(heldout) != HELDOUT_FRAMES_PER_SCENE:
        raise AssertionError("Physical-frame split cardinality drift")
    return train, heldout


def frame_split_all_available(frame_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Deterministically reserve roughly 10% of any nonempty frame set.

    Unlike :func:`frame_split`, this is for Structured3D's variable number of
    archive frames.  For scenes with at least two frames, the validation
    (internally called ``heldout``) count is ``ceil(10% * N)`` with a minimum
    of one, and the reserved frames are spread deterministically through the
    ordered frame list.  A one-frame scene has no non-overlapping held-out
    view; callers must record that exceptional case explicitly.
    """

    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Frame IDs must be nonempty and unique")
    if len(frame_ids) == 1:
        return list(frame_ids), []
    heldout_count = max(1, math.ceil(len(frame_ids) / 10))
    heldout_positions = {
        ((index + 1) * len(frame_ids) - 1) // heldout_count
        for index in range(heldout_count)
    }
    heldout = [
        frame_id
        for index, frame_id in enumerate(frame_ids)
        if index in heldout_positions
    ]
    heldout_set = set(heldout)
    train = [frame_id for frame_id in frame_ids if frame_id not in heldout_set]
    if not train:
        raise AssertionError("All-frames split unexpectedly has no training frame")
    return train, heldout


def source_cells_for_family(family: str) -> tuple[str, ...]:
    """Return the active source cells for one pretext arm."""

    if family == SOURCE_FAMILY_2D3D:
        return SOURCE_CELLS
    if family == SOURCE_FAMILY_2D:
        return TWO_D_CELLS
    raise ValueError(
        f"Unknown source family {family!r}; expected one of {SOURCE_FAMILY_CHOICES}"
    )


def normalize_source_cells(cells: Sequence[str] | None) -> tuple[str, ...]:
    if cells is None:
        return SOURCE_CELLS
    normalized = tuple(str(cell) for cell in cells)
    if not normalized:
        raise ValueError("source cells must be non-empty")
    unknown = [cell for cell in normalized if cell not in SOURCE_CELLS]
    if unknown:
        raise ValueError(f"Unknown source cells: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate source cells: {normalized}")
    return normalized


def source_cell_for_scene_epoch(
    scene_id: str,
    epoch: int,
    cells: Sequence[str] | None = None,
    *,
    schedule: str = SOURCE_CELL_SCHEDULE_FIXED,
    seed: int = 42,
) -> str:
    """Map a scene/epoch visit to one cell with exact cycle balance.

    ``fixed_rotation_v1`` preserves the original experiment contract.  The
    balanced-random schedule deterministically permutes the active cells for
    every complete per-scene cycle.  Its order can vary across cycles without
    ever changing the exposure count.
    """

    if not _SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"Invalid ScanNet scene ID: {scene_id}")
    if not 0 <= int(epoch) < MAX_SCHEDULE_EPOCHS:
        raise ValueError(
            f"Epoch must be in [0, {MAX_SCHEDULE_EPOCHS}), got {epoch}"
        )
    active = normalize_source_cells(cells)
    if schedule not in SOURCE_CELL_SCHEDULE_CHOICES:
        raise ValueError(
            f"Unknown source-cell schedule {schedule!r}; "
            f"expected one of {SOURCE_CELL_SCHEDULE_CHOICES}"
        )
    active_key = ",".join(active)
    if schedule == SOURCE_CELL_SCHEDULE_FIXED:
        digest_key = (
            f"{scene_id}/{EXPERIMENT_ID}/cell"
            if active == SOURCE_CELLS
            else f"{scene_id}/{EXPERIMENT_ID}/cell/{active_key}"
        )
        stable_offset = int(sha256_bytes(digest_key.encode("utf-8"))[:16], 16)
        return active[(stable_offset + int(epoch)) % len(active)]

    cycle, position = divmod(int(epoch), len(active))
    ordered = tuple(
        sorted(
            active,
            key=lambda cell: (
                sha256_bytes(
                    (
                        f"{scene_id}/{EXPERIMENT_ID}/balanced-cell/"
                        f"{active_key}/{int(seed)}/{cycle}/{cell}"
                    ).encode("utf-8")
                ),
                cell,
            ),
        )
    )
    return ordered[position]


def exposure_counts(
    scene_ids: Sequence[str],
    epochs: int = FULL_EPOCHS,
    cells: Sequence[str] | None = None,
    *,
    proposals_per_forward: int = PROPOSALS_PER_FORWARD,
    cell_schedule: str = SOURCE_CELL_SCHEDULE_FIXED,
    seed: int = 42,
) -> dict[str, int]:
    """Count proposal exposure for an ordered set of complete scene epochs."""

    if not 0 <= int(epochs) <= MAX_SCHEDULE_EPOCHS:
        raise ValueError(f"epochs must be in [0, {MAX_SCHEDULE_EPOCHS}]")
    if int(proposals_per_forward) <= 0:
        raise ValueError("proposals_per_forward must be positive")
    active = normalize_source_cells(cells)
    counts = {cell: 0 for cell in active}
    for epoch in range(int(epochs)):
        for scene_id in scene_ids:
            cell = source_cell_for_scene_epoch(
                scene_id,
                epoch,
                cells=active,
                schedule=cell_schedule,
                seed=seed,
            )
            counts[cell] += int(proposals_per_forward)
    return counts


@dataclass(frozen=True)
class SceneSourceAudit:
    scene_id: str
    manifest_path: Path
    manifest_sha256: str
    crop_mode: str
    crop_points: int
    train_frames: tuple[str, ...]
    heldout_frames: tuple[str, ...]
    trainable_proposals: Mapping[str, int]
    heldout_trainable_proposals: Mapping[str, int]


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context} is missing keys: {missing}")


def audit_scene_source_manifest(
    manifest_path: Path,
    *,
    expected_scene_id: str | None = None,
    verify_hashes: bool = True,
) -> SceneSourceAudit:
    """Validate one compact source pack without silently accepting drift."""

    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCENE_SOURCE_SCHEMA:
        raise ValueError(
            f"{manifest_path}: schema {manifest.get('schema_version')!r} "
            f"!= {SCENE_SOURCE_SCHEMA!r}"
        )
    scene_id = str(manifest.get("scene_id", ""))
    if not _SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"{manifest_path}: invalid scene_id {scene_id!r}")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"{manifest_path}: scene_id {scene_id!r} != expected {expected_scene_id!r}"
        )
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"{manifest_path}: experiment_id drift")
    if manifest.get("source_cells") != list(SOURCE_CELLS):
        raise ValueError(f"{manifest_path}: six-cell ordering drift")

    crop = manifest.get("crop", {})
    _require_keys(crop, ("indices", "num_points"), f"{manifest_path}: crop")
    crop_points = int(crop["num_points"])
    crop_mode = str(crop.get("mode", "sphere_50k"))
    if crop_mode == "full_scene":
        if str(manifest.get("dataset")) != "structured3d":
            raise ValueError(f"{manifest_path}: full_scene crop is Structured3D-only")
        max_crop_points = 10_000_000
    elif crop_mode == "sphere_50k":
        max_crop_points = 50_000
    else:
        raise ValueError(f"{manifest_path}: unknown crop mode {crop_mode!r}")
    if crop_points <= 1 or crop_points > max_crop_points:
        raise ValueError(
            f"{manifest_path}: invalid crop size {crop_points} for mode {crop_mode}"
        )

    frames = manifest.get("frames", {})
    _require_keys(frames, ("physical", "train", "heldout"), f"{manifest_path}: frames")
    physical = tuple(str(frame_id) for frame_id in frames["physical"])
    selection_mode = str(frames.get("selection_mode", "selected_16"))
    if selection_mode == "selected_16":
        expected_train, expected_heldout = frame_split(physical)
    elif selection_mode == "all_complete_valid_pose":
        expected_train, expected_heldout = frame_split_all_available(physical)
    else:
        raise ValueError(f"{manifest_path}: unknown frame selection mode {selection_mode!r}")
    if tuple(frames["train"]) != tuple(expected_train):
        raise ValueError(f"{manifest_path}: train-frame split drift")
    if tuple(frames["heldout"]) != tuple(expected_heldout):
        raise ValueError(f"{manifest_path}: heldout-frame split drift")

    artifacts = manifest.get("artifacts", {})
    required_artifacts = (
        "points",
        "colors",
        "normals",
        "crop_indices",
        "labels_g02",
        "labels_g05",
        "labels_g08",
        "raw_projection_cache",
    )
    _require_keys(artifacts, required_artifacts, f"{manifest_path}: artifacts")
    base_dir = manifest_path.parent
    for key in required_artifacts:
        record = artifacts[key]
        _require_keys(record, ("path", "sha256", "size_bytes"), f"artifact {key}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve(strict=True)
        if int(record["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"{manifest_path}: size mismatch for {key}")
        if verify_hashes and record["sha256"] != sha256_file(path):
            raise ValueError(f"{manifest_path}: SHA256 mismatch for {key}")

    counts = manifest.get("counts", {})
    trainable = counts.get("trainable_proposals", {})
    heldout_trainable = counts.get("heldout_trainable_proposals", {})
    if set(trainable) != set(SOURCE_CELLS):
        raise ValueError(f"{manifest_path}: trainable proposal cells drift")
    if set(heldout_trainable) != set(SOURCE_CELLS):
        raise ValueError(f"{manifest_path}: heldout proposal cells drift")
    if any(int(trainable[cell]) <= 0 for cell in SOURCE_CELLS):
        raise ValueError(f"{manifest_path}: at least one train cell has no proposal")
    if expected_heldout and any(int(heldout_trainable[cell]) <= 0 for cell in SOURCE_CELLS):
        raise ValueError(f"{manifest_path}: at least one heldout cell has no proposal")

    checks = manifest.get("checks", {})
    required_checks = {
        "points_aligned",
        "crop_indices_valid",
        "g08_exact_gt",
        "g02_nested_in_g05",
        "g05_nested_in_g08",
        "raw_overlap_preserved",
        "all_six_train_cells_nonempty",
        "all_six_heldout_cells_nonempty",
    }
    if set(checks) != required_checks or not all(checks.values()):
        raise ValueError(f"{manifest_path}: source checks failed or drifted: {checks}")

    return SceneSourceAudit(
        scene_id=scene_id,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        crop_mode=crop_mode,
        crop_points=crop_points,
        train_frames=tuple(expected_train),
        heldout_frames=tuple(expected_heldout),
        trainable_proposals={cell: int(trainable[cell]) for cell in SOURCE_CELLS},
        heldout_trainable_proposals={
            cell: int(heldout_trainable[cell]) for cell in SOURCE_CELLS
        },
    )
