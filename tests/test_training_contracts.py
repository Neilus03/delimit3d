from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from delimit3d.training.contracts import (
    EXPECTED_PROPOSALS_PER_CELL,
    EXPECTED_PROPOSALS_PER_SCENE_CELL,
    FULL_EPOCHS,
    GRANULARITY_KEYS,
    MAX_SCHEDULE_EPOCHS,
    PROPOSALS_PER_FORWARD,
    SCENE_SOURCE_SCHEMA,
    SOURCE_CELLS,
    SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
    TWO_D_CELLS,
    TWO_D_EPOCHS,
    audit_scene_source_manifest,
    extension_lr_factor,
    exposure_counts,
    frame_split,
    frame_split_all_available,
    lr_factor,
    normalize_initialization_report,
    source_cell_for_scene_epoch,
    source_cells_for_family,
    stable_seed,
    upstream_split,
)


def _scene_ids(count: int) -> list[str]:
    return [f"scene{index:04d}_00" for index in range(count)]


def test_upstream_split_is_deterministic_and_order_preserving() -> None:
    official = _scene_ids(1201)
    train_a, holdout_a = upstream_split(official)
    train_b, holdout_b = upstream_split(official)
    assert train_a == train_b
    assert holdout_a == holdout_b
    assert len(train_a) == 1180
    assert len(holdout_a) == 21
    assert set(train_a).isdisjoint(holdout_a)
    assert train_a == [scene_id for scene_id in official if scene_id not in holdout_a]


def test_every_scene_has_exact_full_run_cell_exposure() -> None:
    scene_id = "scene0000_00"
    cells = [source_cell_for_scene_epoch(scene_id, epoch) for epoch in range(FULL_EPOCHS)]
    assert {cell: cells.count(cell) * 4 for cell in SOURCE_CELLS} == {
        cell: EXPECTED_PROPOSALS_PER_SCENE_CELL for cell in SOURCE_CELLS
    }


def test_full_exposure_matches_frozen_contract() -> None:
    counts = exposure_counts(_scene_ids(1180))
    assert counts == {cell: EXPECTED_PROPOSALS_PER_CELL for cell in SOURCE_CELLS}


def test_three_times_schedule_extends_the_same_balanced_cycle() -> None:
    scene_ids = _scene_ids(1180)
    counts = exposure_counts(scene_ids, epochs=MAX_SCHEDULE_EPOCHS)
    assert MAX_SCHEDULE_EPOCHS == 3 * FULL_EPOCHS
    assert counts == {
        cell: 3 * EXPECTED_PROPOSALS_PER_CELL for cell in SOURCE_CELLS
    }
    assert source_cell_for_scene_epoch(
        "scene0000_00", FULL_EPOCHS
    ) == source_cell_for_scene_epoch("scene0000_00", 0)


def test_two_d_family_has_equal_168_epoch_exposure() -> None:
    assert TWO_D_EPOCHS == 168
    assert TWO_D_EPOCHS == FULL_EPOCHS // 2
    assert source_cells_for_family("2d") == TWO_D_CELLS
    assert source_cells_for_family("2d3d") == SOURCE_CELLS
    scene_id = "scene0000_00"
    six_cell_zero = source_cell_for_scene_epoch(scene_id, 0)
    two_d_zero = source_cell_for_scene_epoch(scene_id, 0, cells=TWO_D_CELLS)
    assert six_cell_zero == source_cell_for_scene_epoch(scene_id, 0, cells=SOURCE_CELLS)
    assert two_d_zero in TWO_D_CELLS
    cells = [
        source_cell_for_scene_epoch(scene_id, epoch, cells=TWO_D_CELLS)
        for epoch in range(TWO_D_EPOCHS)
    ]
    assert set(cells) == set(TWO_D_CELLS)
    assert {cell: cells.count(cell) for cell in TWO_D_CELLS} == {
        cell: TWO_D_EPOCHS // len(TWO_D_CELLS) for cell in TWO_D_CELLS
    }
    counts = exposure_counts(_scene_ids(1180), epochs=TWO_D_EPOCHS, cells=TWO_D_CELLS)
    expected = (
        1180 * TWO_D_EPOCHS * PROPOSALS_PER_FORWARD // len(TWO_D_CELLS)
    )
    assert counts == {cell: expected for cell in TWO_D_CELLS}


def test_balanced_random_two_d_schedule_is_deterministic_and_cycle_exact() -> None:
    kwargs = {
        "cells": TWO_D_CELLS,
        "schedule": SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
        "seed": 42,
    }
    visits = [
        source_cell_for_scene_epoch("scene_00000", epoch, **kwargs)
        for epoch in range(18)
    ]
    repeated = [
        source_cell_for_scene_epoch("scene_00000", epoch, **kwargs)
        for epoch in range(18)
    ]
    cycles = [tuple(visits[start : start + 3]) for start in range(0, 18, 3)]
    assert visits == repeated
    assert all(set(cycle) == set(TWO_D_CELLS) for cycle in cycles)
    assert len(set(cycles)) > 1


def test_rgbn6_matched_schedule_has_exact_sixteen_mask_exposure() -> None:
    scene_ids = ["scene_00000", "scene_00001"]
    counts = exposure_counts(
        scene_ids,
        epochs=FULL_EPOCHS,
        cells=TWO_D_CELLS,
        proposals_per_forward=16,
        cell_schedule=SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
        seed=42,
    )
    expected = len(scene_ids) * FULL_EPOCHS * 16 // len(TWO_D_CELLS)
    assert counts == {cell: expected for cell in TWO_D_CELLS}


def test_frame_split_is_exact_12_4() -> None:
    frames = [str(index) for index in range(16)]
    train, heldout = frame_split(frames)
    assert train == ["0", "1", "2", "4", "5", "6", "8", "9", "10", "12", "13", "14"]
    assert heldout == ["3", "7", "11", "15"]


@pytest.mark.parametrize(
    ("count", "expected_heldout"),
    ((1, 0), (2, 1), (9, 1), (10, 1), (11, 2), (16, 2), (100, 10)),
)
def test_all_available_frame_split_is_deterministic_90_10(
    count: int, expected_heldout: int
) -> None:
    frames = [str(index) for index in range(count)]
    train, heldout = frame_split_all_available(frames)
    assert len(heldout) == expected_heldout
    assert len(train) + len(heldout) == count
    assert set(train).isdisjoint(heldout)
    assert train + heldout != []


def test_scene_manifest_fails_closed_on_bad_source_check(tmp_path: Path) -> None:
    artifacts = {}
    for name in (
        "points",
        "colors",
        "normals",
        "crop_indices",
        "labels_g02",
        "labels_g05",
        "labels_g08",
        "raw_projection_cache",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        import hashlib

        artifacts[name] = {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    checks = {
        "points_aligned": True,
        "crop_indices_valid": True,
        "g08_exact_gt": True,
        "g02_nested_in_g05": True,
        "g05_nested_in_g08": True,
        "raw_overlap_preserved": True,
        "all_six_train_cells_nonempty": True,
        "all_six_heldout_cells_nonempty": False,
    }
    manifest = {
        "schema_version": SCENE_SOURCE_SCHEMA,
        "experiment_id": "litept_gt3_unsam3_scratch_v1",
        "scene_id": "scene0000_00",
        "source_cells": list(SOURCE_CELLS),
        "crop": {"indices": "crop_indices.bin", "num_points": 100},
        "frames": {
            "physical": [str(index) for index in range(16)],
            "train": frame_split([str(index) for index in range(16)])[0],
            "heldout": frame_split([str(index) for index in range(16)])[1],
        },
        "artifacts": artifacts,
        "counts": {
            "trainable_proposals": {cell: 1 for cell in SOURCE_CELLS},
            "heldout_trainable_proposals": {cell: 1 for cell in SOURCE_CELLS},
        },
        "checks": checks,
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source checks failed"):
        audit_scene_source_manifest(manifest_path)


def test_frozen_granularity_order() -> None:
    assert GRANULARITY_KEYS == ("g02", "g05", "g08")


def test_lr_schedule_hits_warmup_peak_and_cosine_floor() -> None:
    assert lr_factor(update=1, total_updates=99_120, warmup_updates=5_015) == 1 / 5_015
    assert lr_factor(update=5_015, total_updates=99_120, warmup_updates=5_015) == 1.0
    assert math.isclose(
        lr_factor(update=99_120, total_updates=99_120, warmup_updates=5_015),
        0.01,
    )


def test_extension_lr_schedule_is_continuous_and_hits_new_cosine_floor() -> None:
    start_update = 99_120
    total_updates = 297_360
    warmup_updates = 5_015
    assert extension_lr_factor(
        update=start_update,
        start_update=start_update,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
    ) == 0.01
    assert extension_lr_factor(
        update=start_update + warmup_updates,
        start_update=start_update,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
    ) == 1.0
    assert math.isclose(
        extension_lr_factor(
            update=total_updates,
            start_update=start_update,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        ),
        0.01,
    )
    first = extension_lr_factor(
        update=start_update + 1,
        start_update=start_update,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
    )
    assert 0.01 < first < 0.011


def test_initialization_report_normalizes_additive_kind_field() -> None:
    legacy = {
        "path": "/tmp/initial.pt",
        "sha256": "abc",
        "seed": 42,
        "strict": True,
        "public_checkpoint_used": False,
    }
    current = {**legacy, "initialization_kind": "scratch"}
    assert normalize_initialization_report(legacy) == current
    assert normalize_initialization_report(current) == current
    with pytest.raises(ValueError, match="disagrees"):
        normalize_initialization_report(
            {**legacy, "initialization_kind": "public_litept_backbone"}
        )


def test_stable_seed_is_repeatable_and_order_sensitive() -> None:
    assert stable_seed("scene0000_00", 3, "2d/g02") == stable_seed(
        "scene0000_00", 3, "2d/g02"
    )
    assert stable_seed("scene0000_00", 3, "2d/g02") != stable_seed(
        "scene0000_00", 3, "3d/g02"
    )
