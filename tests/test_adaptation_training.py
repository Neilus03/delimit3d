from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delimit3d.training.adaptation import (
    RawFrame,
    SamplingConfig,
    SceneSource,
    TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
    _fixed_scene_batch_views,
    _token_pure_fixed_holdout_plan,
    _triplet_batch_from_record,
    _triplet_batch_record,
    augmentation_contract_for_profile,
    build_pure_token_contrastive_batch,
    build_scene_visit_input,
    coverage_queue_indices,
    prepare_fixed_holdout_eval_pack,
    remap_contrastive_batch_to_level,
    sampling_config_for_profile,
    training_batch,
    validate_fixed_holdout_eval_pack,
    zero_loss_touching_module_parameters,
)
from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    LOGICAL_BATCH_EVAL_EPOCHS,
    SOURCE_CELLS,
    SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
    TWO_D_CELLS,
    exposure_counts,
    sha256_file,
)
from delimit3d.data.partfield_contrastive import (
    SAMPLER_ALGORITHM_VERSION,
    ContrastiveTripletBatch,
    concatenate_triplet_batches,
    sample_partition_triplets,
    token_positive_pair_payload_digest,
)
from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    MultigranularFrameGroupPlan,
)
from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion
from delimit3d.training.runner import (
    DEFAULT_EVAL_EPOCHS,
    _add_flat_objective_gradients,
    _add_manual_gradients,
    _clip_gradient_diagnostics,
    _content_addressed_report,
    _flat_objective_gradients,
    _global_shared_gradient_report,
    _install_global_pcgrad,
    _load_logical_scene_group,
    _manual_objective_gradients,
    _native_token_pure_fixed_pack_cosine_gap,
    _semantic_config_sha256,
    _telemetry_auxiliary_budget,
    _summarize_hierarchy_records,
    _summarize_token_pure_records,
    _update_for_completed_epoch,
    _updates_for_scene_visit_duration,
)
import delimit3d.training.runner as ddp_runner
from delimit3d.cli.select_checkpoint import (
    main as select_checkpoint,
)


def test_native_token_pure_fixed_pack_gap_fails_closed() -> None:
    cells = {
        cell: {"valid": True, "denominator": 4, "cosine_gap": 0.1}
        for cell in ("2d/g02", "2d/g05", "2d/g08")
    }
    evaluation = {
        "selection_view": "fixed_comparable",
        "fixed_eval_pack": {"token_pure_dec0_fixed_pack": {"enabled": True}},
        "native_fixed_comparable_dec0": {
            "valid": True,
            "expected_cells": ["2d/g02", "2d/g05", "2d/g08"],
            "cells": cells,
            "macro_cosine_gap": 0.1,
        },
    }
    assert _native_token_pure_fixed_pack_cosine_gap(evaluation) == pytest.approx(0.1)
    cells["2d/g08"]["denominator"] = 0
    with pytest.raises(ValueError, match="lacks exact valid"):
        _native_token_pure_fixed_pack_cosine_gap(evaluation)


def test_auxiliary_gradient_budget_uses_conservative_objective_stage_maxima() -> None:
    records = [
        {
            "objectives": {
                "final_dec0": {"stages": {}},
                "mask": {
                    "stages": {
                        "enc4": {
                            "norm_ratio_valid": True,
                            "grad_norm_ratio_vs_final_dec0": 0.20,
                        },
                        "dec3": {
                            "norm_ratio_valid": True,
                            "grad_norm_ratio_vs_final_dec0": 0.30,
                        },
                    }
                },
                "vicreg": {
                    "stages": {
                        "enc4": {
                            "norm_ratio_valid": True,
                            "grad_norm_ratio_vs_final_dec0": 0.10,
                        }
                    }
                },
            }
        }
    ]
    report = _telemetry_auxiliary_budget(records)
    assert report["conservative_sum_aux_norm_ratio_max"] == pytest.approx(0.4)
    assert report["per_objective_max_ratio"] == pytest.approx(
        {"mask": 0.3, "vicreg": 0.1}
    )
    assert report["passed"] is True


def test_global_pcgrad_projects_only_enabled_conflicting_auxiliary(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    components = {
        "final_dec0": parameter[0],
        "auxiliary": -parameter[0] + parameter[1],
    }
    gradients = _manual_objective_gradients(
        components=components,
        parameters=[parameter],
        scale=1.0,
    )
    accumulated = _add_manual_gradients(None, gradients)
    monkeypatch.setattr(ddp_runner.dist, "all_reduce", lambda *_args, **_kwargs: None)
    report = _install_global_pcgrad(
        accumulated=accumulated,
        parameters=[parameter],
        projected_objectives={"auxiliary"},
        world_size=1,
    )
    assert report["objectives"]["auxiliary"]["applied"] is True
    torch.testing.assert_close(parameter.grad, torch.tensor([1.0, 1.0]))


def test_flat_global_shared_gradient_projects_backbone_and_reports_budget(
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    components = {
        "final_dec0": parameter[0],
        "auxiliary": -parameter[0] + parameter[1],
    }
    accumulated = _add_flat_objective_gradients(
        None,
        _flat_objective_gradients(
            components=components,
            parameters=[parameter],
            scale=1.0,
        ),
    )
    monkeypatch.setattr(ddp_runner.dist, "all_reduce", lambda *_args, **_kwargs: None)
    report = _global_shared_gradient_report(
        accumulated=accumulated,
        shared_parameters=[parameter],
        projected_objectives={"auxiliary"},
        world_size=1,
        install_projected_gradient=True,
    )
    auxiliary = report["objectives"]["auxiliary"]
    assert auxiliary["global_shared_cosine_before"] < 0.0
    assert auxiliary["global_shared_cosine_after"] == pytest.approx(0.0)
    assert report["auxiliary_budget"]["passed_before"] is False
    torch.testing.assert_close(parameter.grad, torch.tensor([1.0, 1.0]))


def test_explicit_delivery_reducers_retain_cells_denominators_and_worst_groups() -> None:
    token_summary = _summarize_token_pure_records(
        [
            {
                "optimization_valid": True,
                "retained_cells": ["2d/g02", "2d/g05", "2d/g08"],
                "groups": [
                    {
                        "cell": "2d/g02",
                        "group_id": "f0",
                        "input_proposals": 2,
                        "retained_proposals": 1,
                        "input_point_pairs": 12,
                        "realized_positive_pairs": 4,
                        "zero_denominator": False,
                        "proposals": [
                            {
                                "requested_token_pairs": 8,
                                "pure_positive_survival_fraction": 0.5,
                                "safe_negative_tokens_min": 3,
                                "drop_reason": None,
                            },
                            {
                                "requested_token_pairs": 4,
                                "pure_positive_survival_fraction": 0.25,
                                "drop_reason": "no_pure_negative_token",
                            },
                        ],
                    }
                ],
            }
        ],
        cells=("2d/g02", "2d/g05", "2d/g08"),
    )
    assert token_summary["exact_three_cell_fraction"] == 1.0
    assert token_summary["per_cell"]["2d/g02"]["requested_token_pairs"] == 12
    assert token_summary["per_cell"]["2d/g02"][
        "minimum_pure_positive_survival_fraction"
    ] == pytest.approx(0.25)
    assert token_summary["worst_groups"][0]["group_id"] == "f0"

    hierarchy_summary = _summarize_hierarchy_records(
        [
            {
                "optimization_valid": True,
                "mask": {
                    "cells": {
                        "g02": {
                            "group_count": 1,
                            "loss_denominator_valid_folds": 2,
                            "route": [
                                {
                                    "routed_dec1_proposals": 1,
                                    "routed_dec2_proposals": 2,
                                }
                            ],
                        }
                    }
                },
                "stages": {
                    "enc4": {
                        "token_count": 16,
                        "mask": {
                            "loss_denominator_valid_folds": 2,
                            "proposals_attempted": 2,
                            "proposals_valid": 1,
                            "folds_attempted": 4,
                            "folds_valid": 2,
                        },
                        "vicreg": {
                            "valid": True,
                            "std_min": 0.2,
                            "participation_rank": 8.0,
                            "local_sample_count": 16,
                            "global_sample_count": 32,
                            "discarded_token_count": 0,
                        },
                    }
                },
            }
        ],
        cells=("2d/g02", "2d/g05", "2d/g08"),
    )
    assert hierarchy_summary["per_cell"]["g02"][
        "valid_fold_denominator"
    ] == 2
    assert hierarchy_summary["per_stage"]["enc4"][
        "mask_fold_survival_fraction"
    ] == pytest.approx(0.5)


def test_content_addressed_resume_report_ignores_extraction_paths() -> None:
    first = {
        "path": "/tmp/allocation-a/initial.pt",
        "sha256": "abc",
        "nested": {
            "snapshot_root": "/tmp/allocation-a/code",
            "manifest_sha256": "def",
        },
    }
    second = {
        "path": "/tmp/allocation-b/initial.pt",
        "sha256": "abc",
        "nested": {
            "snapshot_root": "/tmp/allocation-b/code",
            "manifest_sha256": "def",
        },
    }
    assert _content_addressed_report(first) == _content_addressed_report(second)
    changed = {**second, "sha256": "changed"}
    assert _content_addressed_report(first) != _content_addressed_report(changed)


def test_semantic_digest_ignores_operational_pause_but_not_training_horizon() -> None:
    paused = {
        "total_updates": 82040,
        "pause_after_update": 20510,
        "execution_stop_update": 20510,
        "executed_scene_visits_per_rank": 123060,
        "sampling": {"mode": "coverage_multigranular_v2"},
    }
    continued = {
        **paused,
        "pause_after_update": None,
        "execution_stop_update": 82040,
        "executed_scene_visits_per_rank": 492240,
    }
    assert _semantic_config_sha256(paused) == _semantic_config_sha256(continued)
    changed_horizon = {**continued, "total_updates": 82041}
    assert _semantic_config_sha256(paused) != _semantic_config_sha256(
        changed_horizon
    )


def test_semantic_config_digest_ignores_attempt_lineage_but_not_recipe() -> None:
    first = {
        "dataset": "structured3d",
        "grad_clip_norm": 1.0,
        "execution_provenance": {
            "path": "/tmp/job-a/provenance.json",
            "sha256": "attempt-a",
            "schema_version": (
                "structured3d_rgbn6_fullscene_execution_provenance/v1"
            ),
            "trust_root": {"code_archive_sha256": "code"},
            "passed": True,
        },
        "resume_parent": None,
        "resume_into_fresh_attempt": False,
    }
    second = {
        **first,
        "execution_provenance": {
            **first["execution_provenance"],
            "path": "/tmp/job-b/provenance.json",
            "sha256": "attempt-b",
        },
        "resume_parent": {"sha256": "checkpoint"},
        "resume_into_fresh_attempt": True,
        "lineage_parent_checkpoints": [{"sha256": "checkpoint"}],
    }
    assert _semantic_config_sha256(first) == _semantic_config_sha256(second)
    changed = {**second, "grad_clip_norm": 0.5}
    assert _semantic_config_sha256(first) != _semantic_config_sha256(changed)


def _synthetic_scene(scene_id: str = "scene0000_00") -> SceneSource:
    num_points = 30
    points = np.stack(
        [
            np.linspace(0.0, 1.0, num_points, dtype=np.float32),
            np.linspace(1.0, 2.0, num_points, dtype=np.float32),
            np.linspace(2.0, 3.0, num_points, dtype=np.float32),
        ],
        axis=1,
    )
    labels = np.repeat(np.arange(3, dtype=np.int64), 10)
    visible = np.arange(24, dtype=np.int64)
    raw_frames = tuple(
        RawFrame(
            granularity=granularity,
            physical_frame_id=f"{granularity}-heldout",
            split="heldout",
            visible_indices=visible.copy(),
            proposal_offsets=np.asarray([0, 8, 16], dtype=np.int64),
            proposal_point_indices=np.arange(16, dtype=np.int64),
        )
        for granularity in ("g02", "g05", "g08")
    )
    return SceneSource(
        scene_id=scene_id,
        manifest_path=Path(f"/synthetic/{scene_id}/source_manifest.json"),
        manifest_sha256=f"sha256-{scene_id}",
        points=points,
        features=np.zeros((num_points, 6), dtype=np.float32),
        labels_by_key={
            "g02": labels.copy(),
            "g05": labels.copy(),
            "g08": labels.copy(),
        },
        raw_frames=raw_frames,
    )


def _small_sampling() -> SamplingConfig:
    return SamplingConfig(
        proposals_per_forward=2,
        positive_pairs_per_proposal=3,
        uniform_negatives=4,
        spatial_hard_negatives=2,
        spatial_candidate_pool=8,
        feature_hard_negatives=2,
        feature_candidate_pool=8,
    )


def _fullscene_rgbn6_scene(scene_id: str = "scene_00000") -> SceneSource:
    scene = _synthetic_scene(scene_id)
    colors = np.linspace(
        0.05,
        0.95,
        scene.points.shape[0] * 3,
        dtype=np.float32,
    ).reshape(-1, 3)
    normals = np.tile(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        (scene.points.shape[0], 1),
    )
    return replace(
        scene,
        colors=colors,
        normals=normals,
        crop_mode="full_scene",
    )


def test_multiscale_remap_drops_collapsed_pairs_and_negative_collisions() -> None:
    batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        negative_indices=torch.tensor(
            [[2, 3, 4], [0, 4, 5]], dtype=torch.long
        ),
        proposal_labels=torch.tensor([7, 7], dtype=torch.long),
        feature_candidate_indices=torch.tensor(
            [[2, 3, 4, 5], [0, 4, 5, 6]], dtype=torch.long
        ),
        num_feature_hard_negatives=2,
    )
    # The first positive pair collapses at the coarse level.  The second pair
    # remains distinct, and its negatives/candidates must not contain either
    # endpoint after mapping.
    point_to_level = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6], dtype=torch.long)

    remapped, stats = remap_contrastive_batch_to_level(batch, point_to_level)

    assert remapped is not None
    assert stats == {
        "input_pairs": 2,
        "valid_pairs": 1,
        "kept_pairs": 1,
        "collapsed_positive_pairs": 1,
        "zero_negative_pairs": 0,
        "base_negatives_per_pair": 2,
        "base_negative_collisions": 1,
        "base_negative_duplicates": 0,
        "feature_candidate_pool": 3,
        "feature_hard_negatives": 2,
        "feature_candidate_collisions": 1,
        "feature_candidate_duplicates": 0,
    }
    assert torch.equal(remapped.positive_pairs, torch.tensor([[1, 2]]))
    assert torch.equal(remapped.negative_indices, torch.tensor([[3, 4]]))
    assert torch.equal(
        remapped.feature_candidate_indices, torch.tensor([[3, 4, 5]])
    )
    assert remapped.num_feature_hard_negatives == 2
    assert not bool(
        torch.isin(remapped.negative_indices, remapped.positive_pairs).any()
    )


def test_multiscale_remap_returns_none_when_all_positive_pairs_collapse() -> None:
    batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        negative_indices=torch.tensor([[4, 5], [4, 5]], dtype=torch.long),
        proposal_labels=torch.tensor([0, 1], dtype=torch.long),
    )
    remapped, stats = remap_contrastive_batch_to_level(
        batch,
        torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long),
    )

    assert remapped is None
    assert stats["input_pairs"] == 2
    assert stats["valid_pairs"] == 0
    assert stats["collapsed_positive_pairs"] == 2


def test_multiscale_remap_drops_only_zero_negative_rows_and_keeps_order() -> None:
    batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        negative_indices=torch.tensor(
            [[4, 5, 4], [6, 4, 6]], dtype=torch.long
        ),
        proposal_labels=torch.tensor([10, 11], dtype=torch.long),
    )
    # Row zero has only endpoint collisions.  Row one has mapped negatives in
    # first-occurrence order [5,4], which must not be sorted by token ID.
    point_to_level = torch.tensor([0, 1, 2, 3, 1, 0, 5], dtype=torch.long)

    remapped, stats = remap_contrastive_batch_to_level(batch, point_to_level)

    assert remapped is not None
    assert stats["valid_pairs"] == 2
    assert stats["zero_negative_pairs"] == 1
    assert stats["kept_pairs"] == 1
    assert torch.equal(remapped.positive_pairs, torch.tensor([[2, 3]]))
    assert torch.equal(remapped.negative_indices, torch.tensor([[5, 1]]))


def test_pure_token_sampler_uses_fixed_budget_and_excludes_mixed_tokens() -> None:
    batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        negative_indices=torch.tensor([[4, 6], [5, 7]], dtype=torch.long),
        proposal_labels=torch.tensor([7, 7], dtype=torch.long),
        feature_candidate_indices=torch.tensor(
            [[4, 5, 6], [4, 5, 6]], dtype=torch.long
        ),
        num_feature_hard_negatives=1,
        proposal_member_offsets=torch.tensor([0, 4], dtype=torch.long),
        proposal_member_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        pair_proposal_indices=torch.tensor([0, 0], dtype=torch.long),
        eligible_indices=torch.arange(8, dtype=torch.long),
        multiscale_seed=123,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=1,
        spatial_candidate_pool=3,
        feature_candidate_pool=3,
    )
    point_to_level = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])
    level_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    level_batch, stats = build_pure_token_contrastive_batch(
        batch,
        point_to_level,
        level_xyz,
        level_name="dec1",
        include_proposal_diagnostics=True,
    )

    assert level_batch is not None
    assert level_batch.num_pairs == 2
    assert level_batch.negative_indices.shape == (2, 3)
    assert level_batch.feature_candidate_indices is not None
    assert level_batch.feature_candidate_indices.shape == (2, 3)
    assert bool(torch.isin(level_batch.positive_pairs, torch.tensor([0, 1])).all())
    assert bool(
        torch.isin(level_batch.negative_indices, torch.tensor([2, 3, 4])).all()
    )
    assert not bool(
        torch.isin(level_batch.negative_indices, level_batch.positive_pairs).any()
    )
    assert stats["kept_proposals"] == 1
    assert stats["kept_pairs"] == 2
    assert stats["accepted_mixed_tokens"] == 0
    assert stats["strict_token_purity"] == 1.0
    assert stats["negative_diversity_measured"] == 1
    assert int(stats["base_negative_unique_per_pair_min"]) >= 1
    assert stats["proposal_diagnostics"] == [
        {
            "proposal_index": 0,
            "pair_budget": 2,
            "member_point_count": 4,
            "pure_positive_token_count": 2,
            "pure_negative_token_count": 3,
            "mixed_or_unknown_token_count": 0,
            "occupied_token_count": 5,
            "retained": True,
            "drop_reason": "",
        }
    ]


def test_pure_token_sampler_rejects_mixed_positive_support() -> None:
    batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 2]], dtype=torch.long),
        negative_indices=torch.tensor([[1, 3]], dtype=torch.long),
        proposal_labels=torch.tensor([5], dtype=torch.long),
        proposal_member_offsets=torch.tensor([0, 2], dtype=torch.long),
        proposal_member_indices=torch.tensor([0, 2], dtype=torch.long),
        pair_proposal_indices=torch.tensor([0], dtype=torch.long),
        eligible_indices=torch.arange(4, dtype=torch.long),
        multiscale_seed=9,
        num_uniform_negatives=1,
    )
    # Both positive-support tokens also contain a complement point.
    point_to_level = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    level_xyz = torch.zeros((2, 3), dtype=torch.float32)

    level_batch, stats = build_pure_token_contrastive_batch(
        batch,
        point_to_level,
        level_xyz,
        level_name="dec3",
    )

    assert level_batch is None
    assert stats["dropped_proposals_insufficient_pure_positive"] == 1
    assert stats["mixed_or_unknown_tokens_ignored"] == 2
    assert stats["accepted_mixed_tokens"] == 0


def test_pure_token_auxiliary_criterion_is_endpoint_symmetric() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    common = {
        "negative_indices": torch.tensor([[2]], dtype=torch.long),
        "proposal_labels": torch.tensor([3], dtype=torch.long),
        "feature_candidate_indices": torch.tensor([[3, 4]], dtype=torch.long),
        "num_feature_hard_negatives": 1,
    }
    forward_batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        **common,
    )
    swapped_batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[1, 0]], dtype=torch.long),
        **common,
    )
    criterion = PartFieldContrastiveCriterion(
        learnable_temperature=False,
        symmetric_feature_hard_mining=True,
    )

    forward_loss = criterion(embeddings, forward_batch)["loss_total"]
    swapped_loss = criterion(embeddings, swapped_batch)["loss_total"]

    assert float(forward_loss) == pytest.approx(float(swapped_loss), abs=1.0e-7)


def test_pure_token_membership_csr_survives_batch_concatenation() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    generator = torch.Generator().manual_seed(77)
    batch = sample_partition_triplets(
        labels,
        num_proposals=2,
        positive_pairs_per_proposal=2,
        num_uniform_negatives=1,
        selected_proposal_labels=torch.tensor([0, 1], dtype=torch.long),
        generator=generator,
    )

    assert torch.equal(
        batch.proposal_member_offsets,
        torch.tensor([0, 2, 4], dtype=torch.long),
    )
    assert torch.equal(
        batch.proposal_member_indices,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    assert torch.equal(
        batch.pair_proposal_indices,
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    assert torch.equal(batch.eligible_indices, torch.arange(6))

    combined = concatenate_triplet_batches([batch, batch])

    assert torch.equal(
        combined.proposal_member_offsets,
        torch.tensor([0, 2, 4, 6, 8], dtype=torch.long),
    )
    assert torch.equal(
        combined.pair_proposal_indices,
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long),
    )
    assert combined.multiscale_seed == 77


def test_pure_token_empty_level_keeps_auxiliary_temperature_in_graph() -> None:
    reference = torch.randn(4, 3, requires_grad=True)
    criterion = PartFieldContrastiveCriterion(learnable_temperature=True)

    loss = zero_loss_touching_module_parameters(reference, criterion)
    loss.backward()

    assert reference.grad is not None
    assert criterion.log_temperature.grad is not None
    assert float(criterion.log_temperature.grad) == 0.0


def test_logical_scene_group_preserves_requested_order_and_skips_rehash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_preload_sources(**kwargs: object) -> list[SimpleNamespace]:
        scene_id = str(kwargs["scene_ids"][0])  # type: ignore[index]
        verify_hashes = bool(kwargs["verify_hashes"])
        calls.append((scene_id, verify_hashes))
        return [SimpleNamespace(scene_id=scene_id)]

    monkeypatch.setattr(ddp_runner, "_preload_sources", fake_preload_sources)
    scene_ids = ["scene_00000", "scene_00001", "scene_00002"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        loaded = _load_logical_scene_group(
            scene_indices=[2, 0, 1],
            scene_ids=scene_ids,
            source_root=tmp_path,
            expected_manifest_hashes={scene_id: scene_id for scene_id in scene_ids},
            input_features="rgbn6",
            coordinate_normalization_policy="native",
            executor=executor,
        )

    assert [scene.scene_id for scene in loaded] == [
        "scene_00002",
        "scene_00000",
        "scene_00001",
    ]
    assert sorted(calls) == sorted((scene_id, False) for scene_id in scene_ids)


def test_fullscene_rgbn6_augmentation_is_deterministic_and_immutable() -> None:
    scene = _fullscene_rgbn6_scene()
    base_points = scene.points.copy()
    base_colors = scene.colors.copy() if scene.colors is not None else None
    base_normals = scene.normals.copy() if scene.normals is not None else None
    profile = TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE

    random.seed(17)
    np.random.seed(19)
    first = build_scene_visit_input(
        scene=scene,
        epoch=4,
        global_seed=42,
        input_features="rgbn6",
        augmentation_profile=profile,
    )
    random.seed(999)
    np.random.seed(1001)
    repeated = build_scene_visit_input(
        scene=scene,
        epoch=4,
        global_seed=42,
        input_features="rgbn6",
        augmentation_profile=profile,
    )
    next_epoch = build_scene_visit_input(
        scene=scene,
        epoch=5,
        global_seed=42,
        input_features="rgbn6",
        augmentation_profile=profile,
    )

    assert first.augmentation_seed == repeated.augmentation_seed
    assert first.augmentation_seed != next_epoch.augmentation_seed
    np.testing.assert_array_equal(first.points, repeated.points)
    np.testing.assert_array_equal(first.features, repeated.features)
    assert first.points.shape == scene.points.shape
    assert first.features.shape == (scene.points.shape[0], 6)
    assert not np.array_equal(first.points, next_epoch.points)
    np.testing.assert_array_equal(scene.points, base_points)
    np.testing.assert_array_equal(scene.colors, base_colors)
    np.testing.assert_array_equal(scene.normals, base_normals)
    np.testing.assert_allclose(
        np.linalg.norm(first.features[:, 3:6], axis=1),
        np.ones(scene.points.shape[0]),
        rtol=0.0,
        atol=1e-6,
    )


def test_fullscene_rgbn6_augmentation_contract_removes_no_points() -> None:
    contract = augmentation_contract_for_profile(
        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE
    )
    assert contract["point_removal"] is False
    assert contract["point_order_preserved"] is True
    assert contract["geometry"]["elastic_distortion"] is False
    assert contract["geometry"]["coordinate_jitter"] is False
    assert "sphere_crop" in contract["excluded"]
    assert "point_dropout" in contract["excluded"]


def test_fullscene_rgbn6_augmentation_rejects_fixed_crop() -> None:
    scene = replace(_fullscene_rgbn6_scene(), crop_mode="sphere_50k")
    with pytest.raises(ValueError, match="requires crop.mode='full_scene'"):
        build_scene_visit_input(
            scene=scene,
            epoch=0,
            global_seed=42,
            input_features="rgbn6",
            augmentation_profile=TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
        )


def test_sphere50k_rgbn6_augmentation_uses_every_stored_point() -> None:
    scene = replace(_fullscene_rgbn6_scene(), crop_mode="sphere_50k")
    result = build_scene_visit_input(
        scene=scene,
        epoch=7,
        global_seed=42,
        input_features="rgbn6",
        augmentation_profile=TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
    )
    contract = augmentation_contract_for_profile(
        TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K
    )

    assert result.points.shape == scene.points.shape
    assert result.features.shape == (scene.points.shape[0], 6)
    assert contract["required_crop_mode"] == "sphere_50k"
    assert contract["point_removal"] is False
    assert contract["point_order_preserved"] is True
    assert "additional_sphere_crop" in contract["excluded"]
    assert "point_subsampling" in contract["excluded"]


def test_sphere50k_rgbn6_augmentation_rejects_full_scene() -> None:
    with pytest.raises(ValueError, match="requires crop.mode='sphere_50k'"):
        build_scene_visit_input(
            scene=_fullscene_rgbn6_scene(),
            epoch=0,
            global_seed=42,
            input_features="rgbn6",
            augmentation_profile=TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
        )


def _coverage_scene(scene_id: str = "scene0000_00") -> SceneSource:
    num_points = 30
    points = np.stack(
        [
            np.linspace(0.0, 1.0, num_points, dtype=np.float32),
            np.linspace(1.0, 2.0, num_points, dtype=np.float32),
            np.linspace(2.0, 3.0, num_points, dtype=np.float32),
        ],
        axis=1,
    )
    labels = np.repeat(np.arange(15, dtype=np.int64), 2)
    raw_frames = tuple(
        RawFrame(
            granularity=granularity,
            physical_frame_id=f"frame-{frame_index:02d}",
            split="train",
            visible_indices=np.arange(num_points, dtype=np.int64),
            proposal_offsets=np.asarray([0, 4, 8, 12, 16], dtype=np.int64),
            proposal_point_indices=np.arange(16, dtype=np.int64),
        )
        for granularity in ("g02", "g05", "g08")
        for frame_index in range(12)
    )
    return SceneSource(
        scene_id=scene_id,
        manifest_path=Path(f"/synthetic/{scene_id}/source_manifest.json"),
        manifest_sha256=f"sha256-{scene_id}",
        points=points,
        features=np.zeros((num_points, 6), dtype=np.float32),
        labels_by_key={
            "g02": labels.copy(),
            "g05": labels.copy(),
            "g08": labels.copy(),
        },
        raw_frames=raw_frames,
    )


def test_coverage_queue_exhausts_each_cycle_before_repeating() -> None:
    first = coverage_queue_indices(
        population_size=5,
        start=0,
        count=10,
        seed_parts=("scene", "cell"),
    )
    repeated = coverage_queue_indices(
        population_size=5,
        start=0,
        count=10,
        seed_parts=("scene", "cell"),
    )
    assert first == repeated
    assert len(set(first[:5])) == 5
    assert len(set(first[5:])) == 5


def test_coverage_training_uses_twelve_unique_frames_and_labels_at_same_compute() -> None:
    scene = _coverage_scene()
    points = torch.from_numpy(scene.points)
    sampling = SamplingConfig(
        proposals_per_forward=4,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=0,
    )
    seen: dict[str, set[str]] = {cell: set() for cell in SOURCE_CELLS}
    first_pass: list[tuple[str, tuple[str, ...]]] = []
    for epoch in range(18):
        cell, batch, metadata = training_batch(
            scene=scene,
            epoch=epoch,
            points=points,
            device=torch.device("cpu"),
            sampling=sampling,
            sampling_mode="coverage_without_replacement",
            return_metadata=True,
        )
        assert batch.num_pairs == 4
        assert len(metadata["proposal_ids"]) == 4
        assert len(set(metadata["proposal_ids"])) == 4
        if cell.startswith("2d/"):
            assert len(metadata["frame_ids"]) == 4
            assert len(set(metadata["frame_ids"])) == 4
            seen[cell].update(metadata["frame_ids"])
        else:
            seen[cell].update(metadata["proposal_ids"])
        first_pass.append((cell, tuple(metadata["proposal_ids"])))

    assert all(len(values) == 12 for values in seen.values())

    repeated_pass: list[tuple[str, tuple[str, ...]]] = []
    for epoch in range(18):
        cell, _, metadata = training_batch(
            scene=scene,
            epoch=epoch,
            points=points,
            device=torch.device("cpu"),
            sampling=sampling,
            sampling_mode="coverage_without_replacement",
            return_metadata=True,
        )
        repeated_pass.append((cell, tuple(metadata["proposal_ids"])))
    assert repeated_pass == first_pass


def test_two_d_training_batch_never_selects_3d_cells() -> None:
    scene = _coverage_scene()
    points = torch.from_numpy(scene.points)
    sampling = SamplingConfig(
        proposals_per_forward=2,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=0,
    )
    seen: set[str] = set()
    for epoch in range(12):
        cell, batch = training_batch(
            scene=scene,
            epoch=epoch,
            points=points,
            device=torch.device("cpu"),
            sampling=sampling,
            cells=TWO_D_CELLS,
        )
        assert cell in TWO_D_CELLS
        assert not cell.startswith("3d/")
        assert batch.num_pairs == 2
        seen.add(cell)
    assert seen == set(TWO_D_CELLS)


def test_hybrid_keeps_previous_2d_sampling_and_uses_3d_coverage_at_p8() -> None:
    scene = _coverage_scene()
    points = torch.from_numpy(scene.points)
    sampling = SamplingConfig(
        proposals_per_forward=8,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=0,
    )
    three_d_sequences: dict[str, list[str]] = {
        cell: [] for cell in SOURCE_CELLS if cell.startswith("3d/")
    }
    for epoch in range(18):
        cell, hybrid_batch, metadata = training_batch(
            scene=scene,
            epoch=epoch,
            points=points,
            device=torch.device("cpu"),
            sampling=sampling,
            sampling_mode="hybrid_2d_random_3d_coverage",
            return_metadata=True,
        )
        assert hybrid_batch.num_pairs == 8
        if cell.startswith("2d/"):
            _, previous_batch, previous_metadata = training_batch(
                scene=scene,
                epoch=epoch,
                points=points,
                device=torch.device("cpu"),
                sampling=sampling,
                sampling_mode="random_with_replacement",
                return_metadata=True,
            )
            assert metadata == previous_metadata
            assert len(metadata["frame_ids"]) == 1
            assert torch.equal(
                hybrid_batch.positive_pairs,
                previous_batch.positive_pairs,
            )
            assert torch.equal(
                hybrid_batch.negative_indices,
                previous_batch.negative_indices,
            )
        else:
            assert metadata["frame_ids"] == []
            assert len(metadata["proposal_ids"]) == 8
            three_d_sequences[cell].extend(metadata["proposal_ids"])

    # Each 3D cell has 15 valid labels. The first 15 draws exhaust all of them
    # before the deterministic queue starts its next shuffled cycle.
    for sequence in three_d_sequences.values():
        assert len(sequence) == 24
        assert len(set(sequence[:15])) == 15


def test_pf_hnm_768_profile_has_exact_negative_budget() -> None:
    sampling = sampling_config_for_profile("pf_hnm_768_v1")
    assert sampling.proposals_per_forward == 4
    assert sampling.positive_pairs_per_proposal == 64
    assert sampling.uniform_negatives == 256
    assert sampling.spatial_hard_negatives == 256
    assert sampling.spatial_candidate_pool == 1024
    assert sampling.feature_hard_negatives == 256
    assert sampling.feature_candidate_pool == 1024
    assert (
        sampling.uniform_negatives
        + sampling.spatial_hard_negatives
        + sampling.feature_hard_negatives
        == 768
    )


def test_matched_profile_uses_sixteen_proposals_per_scene() -> None:
    sampling = sampling_config_for_profile("pf_hnm_512_p16_v1")
    assert sampling.proposals_per_forward == 16
    assert sampling.positive_pairs_per_proposal == 64
    assert sampling.uniform_negatives == 256
    assert sampling.spatial_hard_negatives == 128
    assert sampling.feature_hard_negatives == 128


def test_training_batch_uses_balanced_random_granularity_cycles() -> None:
    scene = _coverage_scene(scene_id="scene_00000")
    points = torch.from_numpy(scene.points)
    sampling = SamplingConfig(
        proposals_per_forward=2,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=0,
    )
    observed = []
    for epoch in range(9):
        cell, batch = training_batch(
            scene=scene,
            epoch=epoch,
            points=points,
            device=torch.device("cpu"),
            sampling=sampling,
            cells=TWO_D_CELLS,
            cell_schedule=SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
            schedule_seed=42,
        )
        observed.append(cell)
        assert batch.num_pairs == 2
    assert all(
        set(observed[start : start + 3]) == set(TWO_D_CELLS)
        for start in range(0, 9, 3)
    )


def test_logical_update_accounting_carries_complete_scene_visits() -> None:
    assert tuple(
        epoch for epoch in LOGICAL_BATCH_EVAL_EPOCHS if epoch <= 336
    ) == (0, 18, 36, 84, 168, 252, 336)
    assert _update_for_completed_epoch(
        epoch=336,
        scene_visits_per_epoch_per_rank=1465,
        scenes_per_rank_per_update=6,
    ) == 82_040
    assert _updates_for_scene_visit_duration(
        epochs=17,
        scene_visits_per_epoch_per_rank=1465,
        scenes_per_rank_per_update=6,
    ) == 4_151
    with pytest.raises(ValueError, match="does not align"):
        _update_for_completed_epoch(
            epoch=17,
            scene_visits_per_epoch_per_rank=1465,
            scenes_per_rank_per_update=6,
        )


def test_clean1008_schedule_and_gradient_clip_telemetry() -> None:
    assert DEFAULT_EVAL_EPOCHS == (
        0,
        17,
        34,
        85,
        169,
        252,
        336,
        504,
        672,
        756,
        840,
        924,
        1008,
    )
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    diagnostics = _clip_gradient_diagnostics([parameter], max_norm=1.0)
    assert diagnostics["pre_clip_norm"] == pytest.approx(5.0)
    assert diagnostics["clip_return_norm"] == pytest.approx(5.0)
    assert diagnostics["post_clip_norm"] == pytest.approx(1.0)
    assert diagnostics["clip_coefficient"] == pytest.approx(0.2)
    assert diagnostics["was_clipped"] == 1.0


def test_fixed_eval_pack_reuses_base_negatives_and_keeps_adaptive_view(
    tmp_path: Path,
) -> None:
    scene = _synthetic_scene()
    sampling = _small_sampling()
    pack_dir = tmp_path / "fixed"
    report = prepare_fixed_holdout_eval_pack(
        pack_dir=pack_dir,
        scenes=[scene],
        sampling=sampling,
        sampling_profile="unit_test_profile",
        seed=42,
    )
    assert report["passed"] is True
    assert report["scene_count"] == 1
    assert report["record_count"] == 6
    assert report["fixed_negatives_per_pair"] == 6
    assert report["adaptive_feature_hard_negatives_per_pair"] == 2
    assert report["coordinate_normalization_policy"] == "native"

    views = _fixed_scene_batch_views(
        pack_dir=pack_dir,
        scene=scene,
        device=torch.device("cpu"),
        sampling=sampling,
    )
    assert set(views) == {"fixed_comparable", "adaptive_feature_hard"}
    assert len(views["fixed_comparable"]) == 6
    assert len(views["adaptive_feature_hard"]) == 6
    for fixed, adaptive in zip(
        views["fixed_comparable"],
        views["adaptive_feature_hard"],
        strict=True,
    ):
        assert fixed[:2] == adaptive[:2]
        fixed_batch = fixed[2]
        adaptive_batch = adaptive[2]
        assert torch.equal(
            fixed_batch.positive_pairs,
            adaptive_batch.positive_pairs,
        )
        assert torch.equal(
            fixed_batch.negative_indices,
            adaptive_batch.negative_indices,
        )
        assert fixed_batch.feature_candidate_indices is None
        assert fixed_batch.num_feature_hard_negatives == 0
        assert adaptive_batch.feature_candidate_indices is not None
        assert adaptive_batch.num_feature_hard_negatives == 2

    repeated = prepare_fixed_holdout_eval_pack(
        pack_dir=pack_dir,
        scenes=[scene],
        sampling=sampling,
        sampling_profile="unit_test_profile",
        seed=42,
    )
    assert repeated["manifest_sha256"] == report["manifest_sha256"]

    canonical_scene = replace(
        scene,
        coordinate_normalization_policy=(
            "structured3d_local_y_up_to_scannet_z_up_v1"
        ),
    )
    with pytest.raises(ValueError, match="coordinate-normalization drift"):
        validate_fixed_holdout_eval_pack(
            pack_dir=pack_dir,
            scenes=[canonical_scene],
            sampling=sampling,
            sampling_profile="unit_test_profile",
            seed=42,
        )


def test_fixed_eval_pack_fails_closed_on_artifact_drift(tmp_path: Path) -> None:
    scene = _synthetic_scene()
    sampling = _small_sampling()
    pack_dir = tmp_path / "fixed"
    prepare_fixed_holdout_eval_pack(
        pack_dir=pack_dir,
        scenes=[scene],
        sampling=sampling,
        sampling_profile="unit_test_profile",
        seed=42,
    )
    scene_path = pack_dir / f"{scene.scene_id}.pt"
    scene_path.write_bytes(scene_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="artifact hash drift"):
        validate_fixed_holdout_eval_pack(
            pack_dir=pack_dir,
            scenes=[scene],
            sampling=sampling,
            sampling_profile="unit_test_profile",
            seed=42,
        )


def test_token_pure_fixed_eval_advances_queue_until_all_cells_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _synthetic_scene()
    sampling = _small_sampling()
    calls: list[object] = []
    state = DeterministicCoverageState(
        seed=42,
        epoch=0,
        scene_id=scene.scene_id,
        operation="unit/fixed",
        population_size=2,
        population_fingerprint="fixture",
        cursor=1,
    ).to_dict()

    def fake_plan(**kwargs: object) -> MultigranularFrameGroupPlan:
        calls.append(kwargs.get("coverage_states"))
        retained = TWO_D_CELLS[1:] if len(calls) == 1 else TWO_D_CELLS
        return MultigranularFrameGroupPlan(
            groups=tuple(SimpleNamespace(cell=cell) for cell in retained),
            quota_by_cell={cell: 1 for cell in TWO_D_CELLS},
            coverage_by_cell={
                cell: {"state_after": dict(state)} for cell in TWO_D_CELLS
            },
        )

    monkeypatch.setattr(
        "scripts.partfield_contrastive_gt3_unsam3_training."
        "sample_multigranular_frame_group_plan",
        fake_plan,
    )
    plan = _token_pure_fixed_holdout_plan(
        scene=scene,
        points=torch.from_numpy(scene.points),
        device=torch.device("cpu"),
        sampling=sampling,
        seed=42,
    )
    assert len(calls) == 2
    assert calls[0] is None
    assert isinstance(calls[1], dict)
    assert all(
        row["fixed_eval_search_attempt"] == 1
        for row in plan.coverage_by_cell.values()
    )


def test_token_pure_fixed_eval_rotates_quota_without_training_epoch_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _synthetic_scene()
    sampling = _small_sampling()
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs: object) -> MultigranularFrameGroupPlan:
        calls.append(dict(kwargs))
        return MultigranularFrameGroupPlan(
            groups=tuple(SimpleNamespace(cell=cell) for cell in TWO_D_CELLS),
            quota_by_cell={"2d/g02": 1, "2d/g05": 2, "2d/g08": 1},
            coverage_by_cell={cell: {} for cell in TWO_D_CELLS},
        )

    monkeypatch.setattr(
        "scripts.partfield_contrastive_gt3_unsam3_training.stable_seed",
        lambda *_parts: 1,
    )
    monkeypatch.setattr(
        "scripts.partfield_contrastive_gt3_unsam3_training."
        "sample_multigranular_frame_group_plan",
        fake_plan,
    )
    _token_pure_fixed_holdout_plan(
        scene=scene,
        points=torch.from_numpy(scene.points),
        device=torch.device("cpu"),
        sampling=sampling,
        seed=42,
    )

    assert len(calls) == 1
    assert calls[0]["epoch"] == 0
    assert calls[0]["quota_epoch"] == 1
    assert calls[0]["coverage_states"] is None
    assert calls[0]["require_hierarchy_routes"] is False


def test_fixed_eval_relation_csr_roundtrip_and_monotonic_validation() -> None:
    sampling = SamplingConfig(
        proposals_per_forward=1,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=0,
    )
    positive_pairs = torch.tensor([[0, 1]])
    proposal_labels = torch.tensor([0])
    pair_proposal_indices = torch.tensor([0])
    token_positive_pair_offsets = torch.tensor([0, 1])
    token_positive_pair_indices = torch.tensor([[0, 1]])
    batch = ContrastiveTripletBatch(
        positive_pairs=positive_pairs,
        negative_indices=torch.tensor([[2]]),
        proposal_labels=proposal_labels,
        proposal_member_offsets=torch.tensor([0, 2]),
        proposal_member_indices=torch.tensor([0, 1]),
        pair_proposal_indices=pair_proposal_indices,
        eligible_indices=torch.tensor([0, 1, 2]),
        relation_proposal_offsets=torch.tensor([0, 2, 3]),
        relation_proposal_member_indices=torch.tensor([0, 1, 2]),
        algorithm_version=SAMPLER_ALGORITHM_VERSION,
        require_negative_proposal_membership=True,
        token_positive_pair_offsets=token_positive_pair_offsets,
        token_positive_pair_indices=token_positive_pair_indices,
        token_positive_pair_digest=token_positive_pair_payload_digest(
            token_positive_pair_offsets,
            token_positive_pair_indices,
            proposal_labels=proposal_labels,
            pair_proposal_indices=pair_proposal_indices,
        ),
    )
    record = _triplet_batch_record(batch, token_pure_fixed_pack=True)
    restored = _triplet_batch_from_record(
        record,
        device=torch.device("cpu"),
        include_feature_hard=False,
        sampling=sampling,
        num_scene_points=3,
    )
    assert torch.equal(restored.relation_proposal_offsets, torch.tensor([0, 2, 3]))
    assert torch.equal(
        restored.relation_proposal_member_indices, torch.tensor([0, 1, 2])
    )
    record["relation_proposal_offsets"] = torch.tensor([0, 3, 2])
    with pytest.raises(ValueError, match="full relation CSR contract drift"):
        _triplet_batch_from_record(
            record,
            device=torch.device("cpu"),
            include_feature_hard=False,
            sampling=sampling,
            num_scene_points=3,
        )


def test_token_pure_fixed_eval_v5_rebuild_policy_never_serializes_backend_tokens() -> None:
    """V5 records selected proposal CSR, not CPU/backend-specific token IDs."""
    sampling = SamplingConfig(
        proposals_per_forward=1,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=0,
    )
    proposal_labels = torch.tensor([0])
    pair_proposal_indices = torch.tensor([0])
    token_positive_pair_offsets = torch.tensor([0, 1])
    token_positive_pair_indices = torch.tensor([[0, 1]])
    batch = ContrastiveTripletBatch(
        positive_pairs=torch.tensor([[0, 1]]),
        negative_indices=torch.tensor([[2]]),
        proposal_labels=proposal_labels,
        proposal_member_offsets=torch.tensor([0, 2]),
        proposal_member_indices=torch.tensor([0, 1]),
        pair_proposal_indices=pair_proposal_indices,
        eligible_indices=torch.tensor([0, 1, 2]),
        relation_proposal_offsets=torch.tensor([0, 2, 3]),
        relation_proposal_member_indices=torch.tensor([0, 1, 2]),
        algorithm_version=SAMPLER_ALGORITHM_VERSION,
        require_negative_proposal_membership=True,
        token_positive_pair_offsets=token_positive_pair_offsets,
        token_positive_pair_indices=token_positive_pair_indices,
        token_positive_pair_digest=token_positive_pair_payload_digest(
            token_positive_pair_offsets,
            token_positive_pair_indices,
            proposal_labels=proposal_labels,
            pair_proposal_indices=pair_proposal_indices,
        ),
    )

    record = _triplet_batch_record(
        batch,
        token_pure_fixed_pack=True,
        fixed_runtime_token_pair_rebuild=True,
    )
    assert record["native_dec0_pair_selection"] == TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
    assert record["token_positive_pair_offsets"] is None
    assert record["token_positive_pair_indices"] is None
    assert record["token_positive_pair_digest"] is None

    restored = _triplet_batch_from_record(
        record,
        device=torch.device("cpu"),
        include_feature_hard=False,
        sampling=sampling,
        num_scene_points=3,
    )
    assert restored.token_positive_pair_offsets is None
    assert restored.token_positive_pair_indices is None
    assert restored.token_positive_pair_digest is None

    record["token_positive_pair_offsets"] = token_positive_pair_offsets
    record["token_positive_pair_indices"] = token_positive_pair_indices
    record["token_positive_pair_digest"] = batch.token_positive_pair_digest
    with pytest.raises(ValueError, match="must not serialize backend token IDs"):
        _triplet_batch_from_record(
            record,
            device=torch.device("cpu"),
            include_feature_hard=False,
            sampling=sampling,
            num_scene_points=3,
        )


def test_token_pure_fixed_eval_validates_available_same_frame_relations(
    tmp_path: Path,
) -> None:
    scene = _synthetic_scene()
    # V2 routes g08 through enc4.  Separate every source point by more than
    # the four-level 32 cm voxel span so this compact fixture contains both a
    # pure positive and a pure negative enc4 token for each g08 proposal.
    coordinate = np.arange(scene.points.shape[0], dtype=np.float32) * 0.5
    scene = replace(
        scene,
        points=np.stack((coordinate, coordinate + 1.0, coordinate + 2.0), axis=1),
    )
    sampling = SamplingConfig(
        proposals_per_forward=1,
        positive_pairs_per_proposal=1,
        uniform_negatives=1,
        spatial_hard_negatives=0,
        feature_hard_negatives=1,
        feature_candidate_pool=1,
    )
    report = prepare_fixed_holdout_eval_pack(
        pack_dir=tmp_path / "token_pure_fixed",
        scenes=[scene],
        sampling=sampling,
        sampling_profile="unit_token_pure_profile",
        seed=42,
        cells=TWO_D_CELLS,
    )
    assert report["passed"] is True
    assert report["record_count"] == 3
    views = _fixed_scene_batch_views(
        pack_dir=tmp_path / "token_pure_fixed",
        scene=scene,
        device=torch.device("cpu"),
        sampling=sampling,
    )
    for _cell, _group, batch in views["fixed_comparable"]:
        assert batch.require_negative_proposal_membership is True
        assert batch.num_feature_hard_negatives == 0
        assert batch.feature_candidate_indices is None
        assert batch.feature_candidate_pool == 0
        assert batch.deferred_token_loss is not None
    for _cell, _group, batch in views["adaptive_feature_hard"]:
        assert batch.require_negative_proposal_membership is True
        assert batch.num_feature_hard_negatives == 1
        assert batch.feature_candidate_indices is None
        assert batch.deferred_token_loss is not None


def test_clean1008_selector_consumes_fixed_view_and_code_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    train_split = tmp_path / "train.txt"
    train_split.write_text(
        "\n".join(f"scene{index:04d}_00" for index in range(1180)) + "\n",
        encoding="utf-8",
    )
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text('{"passed": true}\n', encoding="utf-8")
    fixed_manifest = tmp_path / "fixed_manifest.json"
    fixed_manifest.write_text('{"passed": true}\n', encoding="utf-8")
    code_manifest = tmp_path / "code_manifest.json"
    code_manifest.write_text('{"passed": true}\n', encoding="utf-8")
    fixed_report = {
        "passed": True,
        "manifest_path": str(fixed_manifest),
        "manifest_sha256": sha256_file(fixed_manifest),
    }
    code_report = {
        "passed": True,
        "path": str(code_manifest),
        "sha256": sha256_file(code_manifest),
    }
    resolved = {
        "three_d_mode": "gt3",
        "sampling_profile": "pf_hnm_768_v1",
        "cache_training_voxelization": False,
        "voxelization_mode": "per_forward_single_scene_recompute",
        "fixed_eval_pack": fixed_report,
        "code_manifest": code_report,
        "initialization": {"sha256": "synthetic-initialization"},
        "source_dataset_audit": {
            "passed": True,
            "path": str(source_audit),
            "sha256": sha256_file(source_audit),
        },
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved),
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "passed": True,
                "epochs": 1008,
                "updates": 297_360,
            }
        ),
        encoding="utf-8",
    )
    eval_epochs = DEFAULT_EVAL_EPOCHS
    rows = []
    selected_evaluation = None
    for epoch in eval_epochs:
        improvement = 0.03 if epoch == 17 else 0.01 if epoch > 0 else 0.0
        evaluation = {
            "kind": "eval",
            "epoch": epoch,
            "update": epoch * 295,
            "selection_view": "fixed_comparable",
            "selection_metric": "native fixed-comparable macro",
            "fixed_eval_pack": fixed_report,
            "native_selection_macro": {
                "overall": 0.10 + improvement,
                "2d": 0.10 + improvement,
                "3d": 0.10 + improvement,
            },
            "native_health": {
                "effective_rank": 3.0,
                "feature_std_min": 0.01,
            },
        }
        rows.append(evaluation)
        if epoch == 17:
            selected_evaluation = evaluation
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert selected_evaluation is not None
    scene_ids = [
        line
        for line in train_split.read_text(encoding="utf-8").splitlines()
        if line
    ]
    torch.save(
        {
            "epoch": 17,
            "update": 17 * 295,
            "experiment_id": EXPERIMENT_ID,
            "public_checkpoint_used": False,
            "evaluation": selected_evaluation,
            "global_exposure_counts": exposure_counts(scene_ids, epochs=17),
        },
        checkpoints / "epoch_0017.pt",
    )
    output = run_dir / "checkpoint_selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--run-dir",
            str(run_dir),
            "--train-split",
            str(train_split),
            "--output",
            str(output),
            "--expected-epochs",
            "1008",
            "--expected-three-d-mode",
            "gt3",
            "--expected-sampling-profile",
            "pf_hnm_768_v1",
        ],
    )
    select_checkpoint()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_epoch"] == 17
    assert report["selection_view"] == "fixed_comparable"
    assert report["sampling_profile"] == "pf_hnm_768_v1"
    assert report["upstream_gate_passed"] is True


def test_selector_accepts_complete_rgb3_2d_only_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    source_root = tmp_path / "source"
    source_root.mkdir()
    train_split = tmp_path / "train.txt"
    scene_ids = [f"scene{index:04d}_00" for index in range(6)]
    train_split.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")
    split_metadata = {
        "dataset": "structured3d",
        "source_root": str(source_root),
    }
    train_split.with_name(train_split.name + ".json").write_text(
        json.dumps(split_metadata),
        encoding="utf-8",
    )
    resolved = {
        "dataset": "structured3d",
        "source_family": "2d",
        "source_cells": list(TWO_D_CELLS),
        "three_d_mode": None,
        "input_features": "rgb3",
        "input_channels": 3,
        "voxel_reduce": "representative",
        "sampling_profile": "pf_hnm_512_v1",
        "cache_training_voxelization": False,
        "voxelization_mode": "per_forward_single_scene_recompute",
        "fixed_eval_pack": None,
        "code_manifest": None,
        "initialization": {"sha256": "synthetic-rgb3-initialization"},
        "source_dataset_audit": None,
        "updates_per_epoch": 3,
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved),
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "passed": True,
                "epochs": 336,
                "updates": 1008,
                "checks": {
                    "source_cells_exact": True,
                    "source_exposure_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )
    eval_epochs = (0, 17, 34, 85, 169, 252, 336)
    rows = []
    selected_evaluation = None
    for epoch in eval_epochs:
        improvement = 0.08 if epoch == 34 else 0.03 if epoch > 0 else 0.0
        evaluation = {
            "kind": "eval",
            "epoch": epoch,
            "update": epoch * 3,
            "selection_view": "adaptive_feature_hard",
            "selection_metric": "native 2d-three-cell macro",
            "native_selection_macro": {
                "overall": 0.10 + improvement,
                "2d": 0.10 + improvement,
            },
            "native_health": {
                "effective_rank": 3.0,
                "feature_std_min": 0.01,
            },
        }
        rows.append(evaluation)
        if epoch == 34:
            selected_evaluation = evaluation
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert selected_evaluation is not None
    torch.save(
        {
            "epoch": 34,
            "update": 102,
            "experiment_id": EXPERIMENT_ID,
            "public_checkpoint_used": False,
            "evaluation": selected_evaluation,
            "resolved_config": {
                "source_family": "2d",
                "source_cells": list(TWO_D_CELLS),
                "input_features": "rgb3",
                "input_channels": 3,
                "voxel_reduce": "representative",
            },
            "global_exposure_counts": exposure_counts(
                scene_ids,
                epochs=34,
                cells=TWO_D_CELLS,
            ),
        },
        checkpoints / "epoch_0034.pt",
    )
    output = run_dir / "checkpoint_selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--run-dir",
            str(run_dir),
            "--train-split",
            str(train_split),
            "--output",
            str(output),
            "--expected-epochs",
            "336",
            "--expected-source-family",
            "2d",
            "--expected-input-features",
            "rgb3",
            "--expected-sampling-profile",
            "pf_hnm_512_v1",
        ],
    )
    select_checkpoint()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_epoch"] == 34
    assert report["source_family"] == "2d"
    assert report["source_cells"] == list(TWO_D_CELLS)
    assert report["three_d_mode"] is None
    assert report["input_features"] == "rgb3"
    assert report["input_channels"] == 3
    assert report["voxel_reduce"] == "representative"
    assert report["improvements"] == pytest.approx(
        {"overall": 0.08, "2d": 0.08}
    )
    assert report["gate_checks"]["2d_macro_improves_0.02"] is True
    assert "3d_macro_improves_0.02" not in report["gate_checks"]
    assert report["upstream_gate_passed"] is True


def test_selector_enforces_matched_rgbn6_logical_batch_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    source_root = tmp_path / "source"
    source_root.mkdir()
    train_split = tmp_path / "train.txt"
    scene_ids = [f"scene_{index:05d}" for index in range(6)]
    train_split.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")
    train_split.with_name(train_split.name + ".json").write_text(
        json.dumps(
            {
                "dataset": "structured3d",
                "source_root": str(source_root),
            }
        ),
        encoding="utf-8",
    )
    sampling = sampling_config_for_profile("pf_hnm_512_p16_v1")
    resolved = {
        "dataset": "structured3d",
        "world_size": 2,
        "seed": 42,
        "source_family": "2d",
        "source_cells": list(TWO_D_CELLS),
        "source_cell_schedule": SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
        "source_cell_schedule_seed": 42,
        "three_d_mode": None,
        "input_features": "rgbn6",
        "input_channels": 6,
        "voxel_reduce": "representative",
        "sampling_profile": "pf_hnm_512_p16_v1",
        "sampling": vars(sampling),
        "evaluation_sampling_profile": "pf_hnm_512_v1",
        "evaluation_sampling": vars(
            sampling_config_for_profile("pf_hnm_512_v1")
        ),
        "evaluation_proposals_per_forward": 4,
        "scene_visits_per_epoch_per_rank": 3,
        "scenes_per_rank_per_update": 6,
        "scenes_per_global_update": 12,
        "proposals_per_rank_per_update": 96,
        "proposals_per_global_update": 192,
        "cache_training_voxelization": False,
        "voxelization_mode": "per_forward_single_scene_recompute",
        "fixed_eval_pack": None,
        "code_manifest": None,
        "initialization": {"sha256": "synthetic-rgbn6-initialization"},
        "source_dataset_audit": None,
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved),
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "passed": True,
                "epochs": 336,
                "updates": 168,
                "checks": {
                    "source_cells_exact": True,
                    "source_exposure_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = []
    selected_evaluation = None
    for epoch in (
        value for value in LOGICAL_BATCH_EVAL_EPOCHS if value <= 336
    ):
        improvement = 0.08 if epoch == 36 else 0.03 if epoch > 0 else 0.0
        evaluation = {
            "kind": "eval",
            "epoch": epoch,
            "update": epoch * 3 // 6,
            "selection_view": "adaptive_feature_hard",
            "selection_metric": "native 2d-three-cell macro",
            "native_selection_macro": {
                "overall": 0.10 + improvement,
                "2d": 0.10 + improvement,
            },
            "native_health": {
                "effective_rank": 3.0,
                "feature_std_min": 0.01,
            },
        }
        rows.append(evaluation)
        if epoch == 36:
            selected_evaluation = evaluation
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert selected_evaluation is not None
    checkpoint_resolved = {
        "source_family": "2d",
        "source_cells": list(TWO_D_CELLS),
        "source_cell_schedule": SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
        "input_features": "rgbn6",
        "input_channels": 6,
        "voxel_reduce": "representative",
        "scenes_per_rank_per_update": 6,
        "sampling": vars(sampling),
        "evaluation_sampling_profile": "pf_hnm_512_v1",
        "evaluation_sampling": vars(
            sampling_config_for_profile("pf_hnm_512_v1")
        ),
    }
    torch.save(
        {
            "epoch": 36,
            "update": 18,
            "experiment_id": EXPERIMENT_ID,
            "public_checkpoint_used": False,
            "evaluation": selected_evaluation,
            "resolved_config": checkpoint_resolved,
            "global_exposure_counts": exposure_counts(
                scene_ids,
                epochs=36,
                cells=TWO_D_CELLS,
                proposals_per_forward=16,
                cell_schedule=SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
                seed=42,
            ),
        },
        checkpoints / "epoch_0036.pt",
    )

    output = run_dir / "checkpoint_selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--run-dir",
            str(run_dir),
            "--train-split",
            str(train_split),
            "--output",
            str(output),
            "--expected-epochs",
            "336",
            "--expected-source-family",
            "2d",
            "--expected-input-features",
            "rgbn6",
            "--expected-sampling-profile",
            "pf_hnm_512_p16_v1",
            "--expected-evaluation-sampling-profile",
            "pf_hnm_512_v1",
            "--expected-scenes-per-rank-per-update",
            "6",
            "--expected-proposals-per-forward",
            "16",
            "--expected-evaluation-proposals-per-forward",
            "4",
            "--expected-source-cell-schedule",
            SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
            "--expected-source-cell-schedule-seed",
            "42",
        ],
    )
    select_checkpoint()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_epoch"] == 36
    assert report["input_features"] == "rgbn6"
    assert report["input_channels"] == 6
    assert report["scenes_per_rank_per_update"] == 6
    assert report["scenes_per_global_update"] == 12
    assert report["proposals_per_forward"] == 16
    assert report["evaluation_sampling_profile"] == "pf_hnm_512_v1"
    assert report["evaluation_proposals_per_forward"] == 4
    assert report["proposals_per_global_update"] == 192
    assert (
        report["source_cell_schedule"]
        == SOURCE_CELL_SCHEDULE_BALANCED_RANDOM
    )
    assert report["upstream_gate_passed"] is True
    _flat_objective_gradients,
    _global_shared_gradient_report,
