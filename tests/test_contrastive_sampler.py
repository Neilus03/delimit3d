from __future__ import annotations

import dataclasses
from dataclasses import replace
import json

import pytest
import torch

import delimit3d.data.contrastive_sampler_v2 as sampler_v2
import delimit3d.data.partfield_contrastive as sampler

from delimit3d.data.contrastive_sampler_v2 import (
    DEFAULT_2D_CELLS,
    DeterministicCoverageState,
    FrameProposalCatalog,
    NoRetainedFrameGroupsError,
    rotating_three_cell_quotas,
    sample_multigranular_frame_group_plan,
    sample_multigranular_frame_groups,
    derive_litept_hierarchy_ancestry,
)
from delimit3d.data.partfield_contrastive import (
    SAMPLER_ALGORITHM_VERSION,
    concatenate_triplet_batches,
    build_relation_membership_index,
    co_membership_safe_negative_mask,
    co_membership_safe_negative_mask_reference,
    co_membership_safe_negative_pair_counts,
    sample_adaptive_unique_negative_rows,
    sample_safe_overlapping_proposal_triplets,
    sample_unique_positive_pairs,
    token_positive_pair_payload_digest,
)
from delimit3d.losses.litept_hierarchy_supervision import (
    build_token_pure_dec0_contrastive_batch,
)
from delimit3d.models.litept_wrapper import LitePTBackbone


def _generator(seed: int = 7) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def test_unique_positive_pairs_are_deterministic_and_adapt_to_pair_population() -> None:
    values = torch.tensor([5, 2, 3, 2], dtype=torch.long)
    first = sample_unique_positive_pairs(values, 20, generator=_generator())
    second = sample_unique_positive_pairs(values, 20, generator=_generator())

    assert torch.equal(first, second)
    assert first.shape == (3, 2)
    assert bool((first[:, 0] < first[:, 1]).all())
    assert torch.unique(first, dim=0).shape[0] == first.shape[0]


def test_adaptive_unique_negative_rows_filters_and_never_repeats() -> None:
    candidates = torch.tensor([1, 1, 2, 3, 4], dtype=torch.long)
    valid = torch.tensor(
        [[True, True, True, True, False], [False, False, True, False, True]]
    )
    excluded = torch.tensor([[3], [2]], dtype=torch.long)
    selection = sample_adaptive_unique_negative_rows(
        candidates,
        4,
        num_rows=2,
        valid_mask=valid,
        excluded_indices=excluded,
        generator=_generator(),
    )

    assert selection.selected_per_row == 1
    assert selection.indices.shape == (2, 1)
    assert int(selection.indices[0, 0]) in {1, 2}
    assert int(selection.indices[1, 0]) == 4
    assert int(selection.indices[0, 0]) != 3
    assert int(selection.indices[1, 0]) != 2
    assert selection.metadata()["adaptive_truncation"]
    assert selection.metadata()["rows_truncated"] == 2


def test_co_membership_filter_can_combine_cross_granularity_masks() -> None:
    # Proposals can be interpreted as g02={0,1}, g05={0,2}, g08={3,4}.
    offsets = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    members = torch.tensor([0, 1, 0, 2, 3, 4], dtype=torch.long)
    candidates = torch.tensor([[1, 2, 3, 5], [0, 2, 4, 5]], dtype=torch.long)
    safe = co_membership_safe_negative_mask(
        torch.tensor([0, 3]),
        candidates,
        proposal_offsets=offsets,
        proposal_point_indices=members,
    )

    assert safe.tolist() == [
        [False, False, True, True],
        [True, True, False, True],
    ]


def test_packed_relation_index_matches_reference_past_63_proposals() -> None:
    relation_sets = [[] for _ in range(130)]
    relation_sets[0] = [0, 1]
    relation_sets[62] = [0, 5]
    relation_sets[63] = [0, 6, 6]  # duplicate edge exercises canonicalization
    relation_sets[64] = [2, 3]
    relation_sets[65] = [0, 4]
    relation_sets[129] = [2, 7]
    offsets = [0]
    members = []
    for values in relation_sets:
        members.extend(values)
        offsets.append(len(members))
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    members_tensor = torch.tensor(members, dtype=torch.long)
    anchors = torch.tensor([0, 2, 0], dtype=torch.long)
    candidates = torch.tensor(
        [[1, 2, 4, 5, 6, 7], [3, 7, 0, 4, 5, 6], [7, 1, 2, 3, 4, 5]],
        dtype=torch.long,
    )
    packed = build_relation_membership_index(
        offsets_tensor, members_tensor, point_capacity=8
    )
    assert packed.point_words.shape == (8, 3)
    expected = co_membership_safe_negative_mask_reference(
        anchors,
        candidates,
        proposal_offsets=offsets_tensor,
        proposal_point_indices=members_tensor,
        require_candidate_membership=True,
    )
    actual = co_membership_safe_negative_mask(
        anchors,
        candidates,
        proposal_offsets=offsets_tensor,
        proposal_point_indices=members_tensor,
        require_candidate_membership=True,
        membership_index=packed,
    )
    assert torch.equal(actual, expected)
    assert actual.tolist() == [
        [False, True, False, False, False, True],
        [False, False, True, True, True, True],
        [True, False, True, True, False, False],
    ]


@pytest.mark.parametrize("require_membership", [False, True])
def test_pair_safe_counts_match_two_reference_masks_past_word_boundary(
    require_membership: bool,
) -> None:
    relation_sets = [[] for _ in range(130)]
    relation_sets[0] = [0, 1]
    relation_sets[62] = [0, 5]
    relation_sets[63] = [0, 6, 6]
    relation_sets[64] = [2, 3]
    relation_sets[65] = [0, 4]
    relation_sets[129] = [2, 7]
    offsets = [0]
    members = []
    for values in relation_sets:
        members.extend(values)
        offsets.append(len(members))
    offsets = torch.tensor(offsets, dtype=torch.long)
    members = torch.tensor(members, dtype=torch.long)
    anchors = torch.tensor([[0, 2], [0, 3], [2, 0]], dtype=torch.long)
    candidates = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    packed = build_relation_membership_index(offsets, members, point_capacity=9)
    expected = (
        co_membership_safe_negative_mask(
            anchors[:, 0], candidates,
            proposal_offsets=offsets,
            proposal_point_indices=members,
            require_candidate_membership=require_membership,
            membership_index=packed,
        )
        & co_membership_safe_negative_mask(
            anchors[:, 1], candidates,
            proposal_offsets=offsets,
            proposal_point_indices=members,
            require_candidate_membership=require_membership,
            membership_index=packed,
        )
    ).sum(dim=1)
    actual = co_membership_safe_negative_pair_counts(
        anchors,
        candidates,
        proposal_offsets=offsets,
        proposal_point_indices=members,
        require_candidate_membership=require_membership,
        membership_index=packed,
    )
    assert torch.equal(actual, expected)


def test_packed_row_pair_intersections_match_reference_across_blocks() -> None:
    generator = torch.Generator(device="cpu").manual_seed(23)
    for row_count, column_count in ((2, 1), (7, 62), (65, 63), (129, 130)):
        rows = torch.randint(
            0,
            2,
            (row_count, column_count),
            generator=generator,
            dtype=torch.int64,
        ).bool()
        expected = all(
            bool((rows[left] & rows[right]).any())
            for left in range(row_count - 1)
            for right in range(left + 1, row_count)
        )
        actual = sampler_v2._all_row_pair_intersections_nonempty(
            rows,
            block_size=7,
        )
        assert actual is expected

    assert not sampler_v2._all_row_pair_intersections_nonempty(
        torch.eye(65, 65, dtype=torch.bool),
        block_size=64,
    )
    assert sampler_v2._all_row_pair_intersections_nonempty(
        torch.ones((65, 130), dtype=torch.bool),
        block_size=64,
    )


@pytest.mark.parametrize("require_membership", [False, True])
def test_relation_signature_pair_check_matches_reference_and_cached_counts(
    require_membership: bool,
) -> None:
    # Packed rows intentionally include duplicate candidate signatures and a
    # zero signature, exercising both the grouped-catalog path and the
    # membership-required filtering without materializing a P-by-Q mask.
    anchors = torch.tensor(
        [[1, 0], [2, 0], [3, 0], [4, 0]], dtype=torch.int64
    )
    candidates = torch.tensor(
        [[1, 0], [1, 0], [2, 0], [4, 0], [8, 0], [0, 0]],
        dtype=torch.int64,
    )
    eligible = candidates.ne(0).any(dim=-1) if require_membership else torch.ones(
        candidates.shape[0], dtype=torch.bool
    )
    safe = (anchors[:, None, :] & candidates[None, :, :]).eq(0).all(dim=-1)
    safe &= eligible[None, :]
    expected = all(
        bool((safe[left] & safe[right]).any())
        for left in range(safe.shape[0] - 1)
        for right in range(left + 1, safe.shape[0])
    )
    actual = sampler_v2._all_anchor_pairs_have_safe_negative_signature(
        anchors,
        candidates,
        require_candidate_membership=require_membership,
    )
    signatures, inverse, counts = torch.unique(
        candidates,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    cached_actual = sampler_v2._all_anchor_pairs_have_safe_negative_signature(
        anchors,
        signatures,
        require_candidate_membership=require_membership,
        candidate_signature_counts=counts,
    )
    assert actual is expected
    assert cached_actual is expected
    assert inverse.numel() == candidates.shape[0]


@pytest.mark.parametrize("require_membership", [False, True])
def test_sampled_relation_signature_check_is_exact_for_the_sampled_pair_set(
    require_membership: bool,
) -> None:
    relation_words = torch.tensor(
        [[1, 0], [2, 0], [3, 0], [4, 0], [8, 0]], dtype=torch.int64
    )
    candidates = relation_words[[0, 1, 2, 3, 4, 4]]
    pairs = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 4]], dtype=torch.long)
    eligible = candidates.ne(0).any(dim=-1) if require_membership else torch.ones(
        candidates.shape[0], dtype=torch.bool
    )
    pair_words = relation_words[pairs[:, 0]] | relation_words[pairs[:, 1]]
    safe = (pair_words[:, None, :] & candidates[None, :, :]).eq(0).all(dim=-1)
    safe &= eligible[None, :]
    expected = bool(safe.any(dim=1).all())
    actual = sampler_v2._sampled_anchor_pairs_have_safe_negative_signature(
        pairs,
        relation_words,
        candidates,
        require_candidate_membership=require_membership,
    )
    signatures, _inverse, counts = torch.unique(
        candidates,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    cached_actual = sampler_v2._sampled_anchor_pairs_have_safe_negative_signature(
        pairs,
        relation_words,
        signatures,
        require_candidate_membership=require_membership,
        candidate_signature_counts=counts,
    )
    assert actual is expected
    assert cached_actual is expected


def test_relation_metadata_preserves_duplicate_overlap_csr_while_support_is_unique() -> None:
    """Raw relation CSR is provenance; only its support set may be deduplicated."""

    visible = torch.arange(6, dtype=torch.long)
    proposal_offsets = torch.tensor([0, 2], dtype=torch.long)
    proposal_members = torch.tensor([0, 1], dtype=torch.long)
    relation_offsets = torch.tensor([0, 4, 7, 10], dtype=torch.long)
    relation_members = torch.tensor(
        [0, 1, 1, 2, 1, 2, 3, 3, 4, 4], dtype=torch.long
    )

    batch, metadata = sample_safe_overlapping_proposal_triplets(
        num_scene_points=6,
        visible_indices=visible,
        proposal_offsets=proposal_offsets,
        proposal_point_indices=proposal_members,
        selected_proposal_indices=torch.tensor([0], dtype=torch.long),
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        relation_proposal_offsets=relation_offsets,
        relation_proposal_point_indices=relation_members,
        generator=_generator(101),
    )

    assert batch.relation_proposal_offsets is not None
    assert batch.relation_proposal_member_indices is not None
    assert torch.equal(batch.relation_proposal_offsets, relation_offsets)
    assert torch.equal(batch.relation_proposal_member_indices, relation_members)
    assert batch.relation_proposal_member_indices.numel() == relation_members.numel()
    assert metadata["relation_proposal_count"] == 3
    assert metadata["known_count"] == int(torch.unique(relation_members).numel())
    assert metadata["known_count"] < int(relation_members.numel())


def test_safe_sampler_adapts_k_and_keeps_all_negative_families_disjoint() -> None:
    points = torch.stack(
        [torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.zeros(10)],
        dim=1,
    )
    visible = torch.arange(10, dtype=torch.long)
    offsets = torch.tensor([0, 3, 6], dtype=torch.long)
    members = torch.tensor([0, 1, 2, 2, 3, 4], dtype=torch.long)
    batch, metadata = sample_safe_overlapping_proposal_triplets(
        num_scene_points=10,
        visible_indices=visible,
        proposal_offsets=offsets,
        proposal_point_indices=members,
        selected_proposal_indices=torch.tensor([0], dtype=torch.long),
        points=points,
        positive_pairs_per_proposal=20,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=1,
        spatial_candidate_pool=6,
        num_feature_hard_negatives=2,
        feature_candidate_pool=6,
        generator=_generator(11),
    )

    assert batch.num_pairs == 3  # exactly 3 choose 2 unique positive pairs
    assert torch.unique(batch.positive_pairs, dim=0).shape[0] == batch.num_pairs
    all_families = torch.cat(
        [batch.negative_indices, batch.feature_candidate_indices], dim=1
    )
    assert all(
        torch.unique(row).numel() == row.numel() for row in all_families
    )
    safe_a = co_membership_safe_negative_mask(
        batch.positive_pairs[:, 0],
        all_families,
        proposal_offsets=offsets,
        proposal_point_indices=members,
    )
    safe_b = co_membership_safe_negative_mask(
        batch.positive_pairs[:, 1],
        all_families,
        proposal_offsets=offsets,
        proposal_point_indices=members,
    )
    assert bool((safe_a & safe_b).all())
    assert metadata["negative_ids_unique_within_and_across_families"]
    assert metadata["uniform"]["selected_per_pair"] <= 2


def _deferred_fixture() -> dict[str, torch.Tensor]:
    return {
        "visible": torch.arange(16, dtype=torch.long),
        "offsets": torch.tensor([0, 4, 8, 12], dtype=torch.long),
        "members": torch.arange(12, dtype=torch.long),
        "points": torch.stack(
            [torch.arange(16, dtype=torch.float32), torch.zeros(16), torch.zeros(16)],
            dim=1,
        ),
    }


def _sample_deferred_fixture(*, defer_token_loss: bool) -> tuple[object, dict]:
    fixture = _deferred_fixture()
    return sample_safe_overlapping_proposal_triplets(
        num_scene_points=16,
        visible_indices=fixture["visible"],
        proposal_offsets=fixture["offsets"],
        proposal_point_indices=fixture["members"],
        selected_proposal_indices=torch.tensor([0, 1], dtype=torch.long),
        points=fixture["points"],
        positive_pairs_per_proposal=5,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=1,
        spatial_candidate_pool=4,
        num_feature_hard_negatives=2,
        feature_candidate_pool=5,
        defer_token_loss=defer_token_loss,
        generator=_generator(101),
    )


def test_deferred_and_materialized_paths_share_pairs_and_effective_widths(
    monkeypatch,
) -> None:
    materialized, materialized_metadata = _sample_deferred_fixture(
        defer_token_loss=False
    )

    def fail_point_sampler(*args, **kwargs):
        del args, kwargs
        raise AssertionError("point negative sampler must be skipped in deferred mode")

    monkeypatch.setattr(sampler, "sample_adaptive_unique_negative_rows", fail_point_sampler)
    monkeypatch.setattr(sampler, "_sample_adaptive_spatial_negative_rows", fail_point_sampler)
    deferred, deferred_metadata = _sample_deferred_fixture(defer_token_loss=True)

    assert deferred.algorithm_version == SAMPLER_ALGORITHM_VERSION
    assert torch.equal(deferred.positive_pairs, materialized.positive_pairs)
    assert torch.equal(deferred.pair_proposal_indices, materialized.pair_proposal_indices)
    assert torch.equal(deferred.proposal_member_offsets, materialized.proposal_member_offsets)
    assert torch.equal(deferred.proposal_member_indices, materialized.proposal_member_indices)
    assert deferred.negative_indices.shape == (deferred.num_pairs, 0)
    assert deferred.feature_candidate_indices is None
    assert deferred.deferred_token_loss is not None
    assert deferred.deferred_token_loss.metadata()["algorithm_version"] == (
        SAMPLER_ALGORITHM_VERSION
    )
    assert deferred.deferred_token_loss.metadata()[
        "require_negative_proposal_membership"
    ] is False
    for field in (
        "num_uniform_negatives",
        "num_spatial_hard_negatives",
        "feature_candidate_pool",
        "num_feature_hard_negatives",
    ):
        assert getattr(deferred, field) == getattr(materialized, field)
    assert materialized_metadata["point_negative_ids_materialized"] is True
    assert materialized_metadata["point_negative_sampling_skipped"] is False
    assert deferred_metadata["point_negative_ids_materialized"] is False
    assert deferred_metadata["point_negative_sampling_skipped"] is True


def test_native_token_payload_bypasses_all_raw_pair_and_negative_construction(
    monkeypatch,
) -> None:
    """Native dec0 rows retain raw CSR provenance without rebuilding point tuples."""

    visible = torch.arange(8, dtype=torch.long)
    proposal_offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    proposal_members = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    relation_offsets = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    relation_members = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    selected = torch.tensor([0, 1], dtype=torch.long)
    payload_offsets = torch.tensor([0, 1, 2], dtype=torch.long)
    payload_pairs = torch.tensor([[10, 11], [12, 13]], dtype=torch.long)
    pair_proposal_indices = torch.tensor([0, 1], dtype=torch.long)
    payload_labels = selected[pair_proposal_indices]
    payload_digest = token_positive_pair_payload_digest(
        payload_offsets,
        payload_pairs,
        proposal_labels=payload_labels,
        pair_proposal_indices=pair_proposal_indices,
    )

    def fail_raw_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native payload must bypass raw point construction")

    for name in (
        "build_relation_membership_index",
        "_unique_proposal_member_counts",
        "sample_unique_positive_pairs",
        "co_membership_safe_negative_pair_counts",
        "sample_adaptive_unique_negative_rows",
        "_sample_adaptive_spatial_negative_rows",
    ):
        monkeypatch.setattr(sampler, name, fail_raw_path)

    batch, metadata = sample_safe_overlapping_proposal_triplets(
        num_scene_points=8,
        visible_indices=visible,
        proposal_offsets=proposal_offsets,
        proposal_point_indices=proposal_members,
        selected_proposal_indices=selected,
        positive_pairs_per_proposal=2,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        relation_proposal_offsets=relation_offsets,
        relation_proposal_point_indices=relation_members,
        defer_token_loss=True,
        generator=_generator(909),
        token_positive_pair_offsets=payload_offsets,
        token_positive_pair_indices=payload_pairs,
        token_positive_pair_digest=payload_digest,
    )

    assert torch.equal(batch.positive_pairs, payload_pairs)
    assert torch.equal(batch.token_positive_pair_offsets, payload_offsets)
    assert torch.equal(batch.token_positive_pair_indices, payload_pairs)
    assert torch.equal(batch.proposal_member_offsets, proposal_offsets)
    assert torch.equal(batch.proposal_member_indices, proposal_members)
    assert torch.equal(batch.pair_proposal_indices, pair_proposal_indices)
    assert torch.equal(batch.proposal_labels, payload_labels)
    assert torch.equal(batch.relation_proposal_offsets, relation_offsets)
    assert torch.equal(batch.relation_proposal_member_indices, relation_members)
    assert batch.negative_indices.shape == (2, 0)
    assert batch.feature_candidate_indices is None
    assert metadata["token_positive_pair_payload"] is True
    assert metadata["point_negative_sampling_skipped"] is True


def test_deferred_and_materialized_rebuild_the_same_token_objective() -> None:
    materialized, _ = _sample_deferred_fixture(defer_token_loss=False)
    deferred, _ = _sample_deferred_fixture(defer_token_loss=True)
    point_to_token = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9],
        dtype=torch.long,
    )
    token_xyz = torch.stack(
        [
            torch.arange(10, dtype=torch.float32),
            torch.zeros(10),
            torch.zeros(10),
        ],
        dim=1,
    )
    materialized_tokens, materialized_metrics = (
        build_token_pure_dec0_contrastive_batch(
            materialized, point_to_token, token_xyz
        )
    )
    deferred_tokens, deferred_metrics = build_token_pure_dec0_contrastive_batch(
        deferred, point_to_token, token_xyz
    )
    assert materialized_tokens is not None
    assert deferred_tokens is not None
    assert torch.equal(
        materialized_tokens.positive_pairs, deferred_tokens.positive_pairs
    )
    assert torch.equal(
        materialized_tokens.negative_indices, deferred_tokens.negative_indices
    )
    assert torch.equal(
        materialized_tokens.feature_candidate_indices,
        deferred_tokens.feature_candidate_indices,
    )
    assert materialized_metrics["realized_positive_pairs"] == deferred_metrics[
        "realized_positive_pairs"
    ]
    assert materialized_metrics["zero_denominator"] is False
    assert deferred_metrics["zero_denominator"] is False


def test_v2_per_proposal_positive_stream_is_independent_of_other_negative_draws() -> None:
    fixture = _deferred_fixture()
    common = dict(
        num_scene_points=16,
        visible_indices=fixture["visible"],
        proposal_offsets=fixture["offsets"],
        proposal_point_indices=fixture["members"],
        points=fixture["points"],
        positive_pairs_per_proposal=5,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=1,
        spatial_candidate_pool=4,
        num_feature_hard_negatives=2,
        feature_candidate_pool=5,
    )
    only_second, _ = sample_safe_overlapping_proposal_triplets(
        **common,
        selected_proposal_indices=torch.tensor([1]),
        defer_token_loss=True,
        generator=_generator(303),
    )
    both, _ = sample_safe_overlapping_proposal_triplets(
        **common,
        selected_proposal_indices=torch.tensor([0, 1]),
        defer_token_loss=True,
        generator=_generator(303),
    )
    assert both.pair_proposal_indices is not None
    second_rows = both.pair_proposal_indices == 1
    assert torch.equal(only_second.positive_pairs, both.positive_pairs[second_rows])


def test_v2_explicit_generator_advances_one_root_per_call() -> None:
    fixture = _deferred_fixture()
    kwargs = dict(
        num_scene_points=16,
        visible_indices=fixture["visible"],
        proposal_offsets=fixture["offsets"],
        proposal_point_indices=fixture["members"],
        selected_proposal_indices=torch.tensor([1]),
        positive_pairs_per_proposal=5,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        defer_token_loss=True,
    )
    generator = _generator(307)
    first, first_metadata = sample_safe_overlapping_proposal_triplets(
        **kwargs, generator=generator
    )
    second, second_metadata = sample_safe_overlapping_proposal_triplets(
        **kwargs, generator=generator
    )
    assert first_metadata["root_seed"] != second_metadata["root_seed"]
    assert not torch.equal(first.positive_pairs, second.positive_pairs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_v2_cpu_generator_is_replayable_for_cuda_tensors() -> None:
    device = torch.device("cuda")
    fixture = _deferred_fixture()
    kwargs = dict(
        num_scene_points=16,
        visible_indices=fixture["visible"].to(device),
        proposal_offsets=fixture["offsets"].to(device),
        proposal_point_indices=fixture["members"].to(device),
        selected_proposal_indices=torch.tensor([1], device=device),
        positive_pairs_per_proposal=5,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        defer_token_loss=True,
    )
    first, first_metadata = sample_safe_overlapping_proposal_triplets(
        **kwargs, generator=_generator(311)
    )
    second, second_metadata = sample_safe_overlapping_proposal_triplets(
        **kwargs, generator=_generator(311)
    )
    assert first_metadata["root_seed"] == second_metadata["root_seed"]
    assert torch.equal(first.positive_pairs, second.positive_pairs)


def test_concatenate_preserves_deferred_plan_and_rejects_contract_drift() -> None:
    first, _ = _sample_deferred_fixture(defer_token_loss=True)
    fixture = _deferred_fixture()
    second, _ = sample_safe_overlapping_proposal_triplets(
        num_scene_points=16,
        visible_indices=fixture["visible"],
        proposal_offsets=fixture["offsets"],
        proposal_point_indices=fixture["members"],
        selected_proposal_indices=torch.tensor([2], dtype=torch.long),
        points=fixture["points"],
        positive_pairs_per_proposal=5,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=1,
        spatial_candidate_pool=4,
        num_feature_hard_negatives=2,
        feature_candidate_pool=5,
        defer_token_loss=True,
        generator=_generator(101),
    )
    combined = concatenate_triplet_batches([first, second])
    assert combined.deferred_token_loss == first.deferred_token_loss
    assert combined.algorithm_version == SAMPLER_ALGORITHM_VERSION
    assert combined.pair_proposal_indices is not None
    assert combined.pair_proposal_indices.min().item() == 0
    assert combined.pair_proposal_indices.max().item() == 2
    assert combined.proposal_member_offsets.tolist() == [0, 4, 8, 12]

    drifted = replace(
        second,
        num_uniform_negatives=1,
        deferred_token_loss=replace(
            second.deferred_token_loss,
            requested_uniform_negatives_per_pair=1,
        ),
    )
    with pytest.raises(ValueError, match="negative-category|deferred"):
        concatenate_triplet_batches([first, drifted])


def test_membership_required_batch_exposes_known_support_and_full_relation_csr() -> None:
    points = torch.stack(
        [torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.zeros(10)],
        dim=1,
    )
    offsets = torch.tensor([0, 3, 6], dtype=torch.long)
    members = torch.tensor([0, 1, 2, 2, 3, 4], dtype=torch.long)
    batch, metadata = sample_safe_overlapping_proposal_triplets(
        num_scene_points=10,
        visible_indices=torch.arange(10, dtype=torch.long),
        proposal_offsets=offsets,
        proposal_point_indices=members,
        selected_proposal_indices=torch.tensor([0], dtype=torch.long),
        points=points,
        positive_pairs_per_proposal=20,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
        generator=_generator(19),
    )

    assert batch.eligible_indices is not None
    assert batch.eligible_indices.tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(batch.relation_proposal_offsets, offsets)
    assert torch.equal(batch.relation_proposal_member_indices, members)
    assert batch.pair_proposal_indices is not None
    assert batch.pair_proposal_indices.tolist() == [0]
    assert metadata["visible_count"] == 10
    assert metadata["known_count"] == 5
    assert metadata["unknown_count"] == 5
    assert metadata["eligible_count"] == 5


def test_multigranular_relation_csr_is_built_once_per_physical_frame(monkeypatch) -> None:
    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (
            _catalog(cell, "frame-a", 0),
            _catalog(cell, "frame-b", 6),
        )
        for cell in DEFAULT_2D_CELLS
    }
    calls: list[tuple[str, ...]] = []
    original = sampler_v2._combined_relation_membership

    def spy(catalogs):
        calls.append(tuple(catalog.frame_id for catalog in catalogs))
        return original(catalogs)

    monkeypatch.setattr(sampler_v2, "_combined_relation_membership", spy)
    timing: dict[str, object] = {}
    plan = sample_multigranular_frame_group_plan(
        scene_id="scene_lazy_relations",
        epoch=0,
        seed=9,
        points=points,
        frames_by_cell=frames,
        total_proposal_quota=1,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        timing=timing,
    )
    inspected_frame_ids = {
        row["frame_id"]
        for coverage in plan.coverage_by_cell.values()
        for row in coverage["preselection_inspected_entries"]
    }
    assert len(calls) == len(inspected_frame_ids)
    assert {call[0] for call in calls} == inspected_frame_ids
    assert all(set(call) == {call[0]} for call in calls)
    assert timing["preselection_frame_support_count"] == len(calls)
    assert "frame-a" in {group.frame_id for group in plan.groups} or "frame-b" in {
        group.frame_id for group in plan.groups
    }


def test_catalog_preselection_contexts_are_built_only_for_inspected_positions(
    monkeypatch,
) -> None:
    """A quota stop must not construct contexts for the untouched catalog tail."""

    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (
            _catalog(cell, f"{cell}-frame-a", 0),
            _catalog(cell, f"{cell}-frame-b", 6),
        )
        for cell in DEFAULT_2D_CELLS
    }
    calls: list[tuple[str, str]] = []
    original = sampler_v2._catalog_preselection_context

    def spy(**kwargs):
        catalog = kwargs["catalog"]
        calls.append((catalog.cell, catalog.frame_id))
        return original(**kwargs)

    monkeypatch.setattr(sampler_v2, "_catalog_preselection_context", spy)
    timing: dict[str, object] = {}
    plan = sample_multigranular_frame_group_plan(
        scene_id="scene_lazy_catalog_contexts",
        epoch=0,
        seed=42,
        points=points,
        frames_by_cell=frames,
        total_proposal_quota=1,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
        timing=timing,
    )

    inspected_catalogs = {
        (cell, row["frame_id"])
        for cell, coverage in plan.coverage_by_cell.items()
        for row in coverage["preselection_inspected_entries"]
    }
    assert plan.quota_by_cell == {
        "2d/g02": 1,
        "2d/g05": 0,
        "2d/g08": 0,
    }
    assert len(inspected_catalogs) == 1
    assert set(calls) == inspected_catalogs
    assert timing["preselection_catalog_count"] == len(calls)
    assert timing["preselection_evidence_calls"] == sum(
        coverage["preselection_inspected_count"]
        for coverage in plan.coverage_by_cell.values()
    )
    assert all(
        not coverage["preselection_inspected_entries"]
        for cell, coverage in plan.coverage_by_cell.items()
        if cell != "2d/g02"
    )


def test_audit_can_materialize_the_complete_preselection_population() -> None:
    """The offline denominator must not make the production scan eager."""

    points = torch.stack(
        [torch.arange(48, dtype=torch.float32) / 10.0, torch.zeros(48), torch.zeros(48)],
        dim=1,
    )
    proposal_members = torch.arange(24, dtype=torch.long)
    catalog = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="audit-full-evidence",
        visible_indices=torch.arange(48, dtype=torch.long),
        proposal_offsets=torch.arange(0, 25, 2, dtype=torch.long),
        proposal_point_indices=proposal_members,
    )
    common = dict(
        scene_id="scene_audit_full_evidence",
        epoch=0,
        seed=17,
        points=points,
        frames_by_cell={"2d/g02": (catalog,), "2d/g05": (), "2d/g08": ()},
        total_proposal_quota=1,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
    )
    lazy = sample_multigranular_frame_group_plan(**common)
    exhaustive = sample_multigranular_frame_group_plan(
        **common,
        materialize_full_preselection_evidence=True,
    )

    lazy_row = lazy.coverage_by_cell["2d/g02"]
    exhaustive_row = exhaustive.coverage_by_cell["2d/g02"]
    assert lazy_row["raw_population"] == 12
    assert len(lazy_row["preselection_raw_eligibility"]) < 12
    assert lazy_row["preselection_evidence_complete"] is False
    assert len(exhaustive_row["preselection_raw_eligibility"]) == 12
    assert exhaustive_row["preselection_evidence_complete"] is True
    assert exhaustive_row["representable_population_complete"] is True
    assert exhaustive_row["representable_population_count"] == sum(
        row["eligible"] for row in exhaustive_row["preselection_raw_eligibility"]
    )
    assert lazy_row["selected_catalog_positions"] == exhaustive_row[
        "selected_catalog_positions"
    ]
    assert lazy_row["state_after"] == exhaustive_row["state_after"]


def test_fixed_quota_rotation_can_be_decoupled_from_coverage_epoch() -> None:
    """A fixed pack may rotate quotas without fabricating a train cursor."""

    points = torch.stack(
        [torch.arange(24, dtype=torch.float32) / 10.0, torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (_catalog(cell, f"{cell}-fixed", 0),)
        for cell in DEFAULT_2D_CELLS
    }
    plan = sample_multigranular_frame_group_plan(
        scene_id="scene_fixed_quota_rotation",
        epoch=0,
        quota_epoch=1,
        seed=19,
        points=points,
        frames_by_cell=frames,
        total_proposal_quota=4,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
        operation="unit-fixed-quota",
    )

    assert plan.quota_by_cell == {"2d/g02": 1, "2d/g05": 2, "2d/g08": 1}
    for coverage in plan.coverage_by_cell.values():
        assert coverage["quota_epoch"] == 1
        assert coverage["state_before"]["epoch"] == 0


def test_malformed_frame_catalog_and_coverage_state_are_rejected() -> None:
    points = torch.zeros((24, 3))
    valid_frames = {
        cell: (_catalog(cell, "frame-a", 0),) for cell in DEFAULT_2D_CELLS
    }
    malformed = replace(
        valid_frames["2d/g02"][0],
        proposal_offsets=torch.tensor([1, 2, 3, 4]),
    )
    malformed_frames = {**valid_frames, "2d/g02": (malformed,)}
    with pytest.raises(ValueError, match="offsets"):
        sample_multigranular_frame_group_plan(
            scene_id="scene_bad_catalog",
            epoch=0,
            seed=1,
            points=points,
            frames_by_cell=malformed_frames,
            total_proposal_quota=1,
            positive_pairs_per_proposal=1,
            num_uniform_negatives=1,
            num_spatial_hard_negatives=0,
            num_feature_hard_negatives=0,
        )
    with pytest.raises(TypeError, match="coverage state"):
        sample_multigranular_frame_group_plan(
            scene_id="scene_bad_state",
            epoch=0,
            seed=1,
            points=points,
            frames_by_cell=valid_frames,
            total_proposal_quota=1,
            positive_pairs_per_proposal=1,
            num_uniform_negatives=1,
            num_spatial_hard_negatives=0,
            num_feature_hard_negatives=0,
            coverage_states={"2d/g02": {"cursor": 0}},
        )


def test_coverage_state_serializes_and_crosses_boundary_without_duplicates() -> None:
    state = DeterministicCoverageState(
        seed=42,
        epoch=3,
        scene_id="scene_00001",
        operation="train/2d/g02",
        population_size=5,
        population_fingerprint="catalog-sha",
        cursor=4,
    )
    selected, after, metadata = state.draw_unique(4)
    restored = DeterministicCoverageState.from_json(after.to_json())

    assert len(selected) == len(set(selected)) == 4
    assert metadata["crossed_cycle_boundary"]
    assert metadata["duplicate_count"] == 0
    assert restored == after
    assert after.cursor == 8

    exhausted, _, exhausted_metadata = state.draw_unique(8)
    assert len(exhausted) == 5
    assert exhausted_metadata["adaptive_truncation"]


def test_coverage_scan_resume_matches_one_uninterrupted_scan() -> None:
    state = DeterministicCoverageState(
        seed=47,
        epoch=0,
        scene_id="scene_scan_resume",
        operation="multigranular-train-v2/2d/g02",
        population_size=7,
        population_fingerprint="scan-fingerprint",
    )
    permutation = state._permutation().tolist()
    eligible = (permutation[1], permutation[3], permutation[5])

    uninterrupted, uninterrupted_after, uninterrupted_metadata = (
        state.scan_eligible(2, eligible_positions=eligible)
    )
    first, after_first, first_metadata = state.scan_eligible(
        1, eligible_positions=eligible
    )
    second, resumed_after, second_metadata = after_first.scan_eligible(
        1, eligible_positions=eligible
    )

    assert first + second == uninterrupted
    assert resumed_after.cursor == uninterrupted_after.cursor
    assert first_metadata["cursor_after"] == second_metadata["cursor_before"]
    assert uninterrupted_metadata["inspected_count"] == 4
    assert uninterrupted_metadata["skipped_positions"] == [
        permutation[0],
        permutation[2],
    ]
    assert uninterrupted_metadata["duplicate_count"] == 0


def test_lazy_coverage_scan_matches_eager_prefix_and_reports_lower_bound() -> None:
    state = DeterministicCoverageState(
        seed=51,
        epoch=0,
        scene_id="scene_lazy_scan",
        operation="multigranular-train-v2/2d/g02",
        population_size=9,
        population_fingerprint="lazy-fingerprint",
    )
    permutation = state._permutation().tolist()
    eligible = {permutation[1], permutation[3], permutation[7]}
    visited: list[int] = []

    selected, after, metadata = state.scan_eligible_lazy(
        2,
        evaluate_position=lambda position: (
            visited.append(int(position)) or int(position) in eligible
        ),
    )
    eager_selected, eager_after, eager_metadata = state.scan_eligible(
        2,
        eligible_positions=tuple(eligible),
    )

    assert selected == eager_selected
    assert after.cursor == eager_after.cursor
    assert metadata["inspected_positions"] == eager_metadata["inspected_positions"]
    assert visited == metadata["inspected_positions"]
    assert metadata["selected_count"] == 2
    assert metadata["representable_population_complete"] is False
    # The third eligible row is deliberately beyond the quota stop point.
    assert metadata["eligible_population_size"] == 2
    assert eager_metadata["eligible_population_size"] == 3
    assert metadata["duplicate_count"] == 0


def test_lazy_coverage_scan_resume_is_cursor_parity_and_cache_friendly() -> None:
    state = DeterministicCoverageState(
        seed=52,
        epoch=0,
        scene_id="scene_lazy_resume",
        operation="multigranular-train-v2/2d/g05",
        population_size=8,
        population_fingerprint="lazy-resume-fingerprint",
    )
    permutation = state._permutation().tolist()
    eligible = {permutation[1], permutation[4], permutation[6]}
    cache: dict[int, bool] = {}
    evaluations: list[int] = []

    def evaluate(position: int) -> bool:
        position = int(position)
        if position not in cache:
            evaluations.append(position)
            cache[position] = position in eligible
        return cache[position]

    uninterrupted, uninterrupted_after, uninterrupted_metadata = (
        state.scan_eligible_lazy(2, evaluate_position=evaluate)
    )
    first, after_first, first_metadata = state.scan_eligible_lazy(
        1, evaluate_position=evaluate
    )
    second, resumed_after, second_metadata = after_first.scan_eligible_lazy(
        1, evaluate_position=evaluate
    )

    assert first + second == uninterrupted
    assert resumed_after.cursor == uninterrupted_after.cursor
    assert first_metadata["cursor_after"] == second_metadata["cursor_before"]
    # The first call and its resumed split cover the same raw prefix.  The
    # shared cache must therefore see each inspected position exactly once.
    assert evaluations == uninterrupted_metadata["inspected_positions"]
    assert len(evaluations) == len(set(evaluations))
    assert first_metadata["representable_population_complete"] is False
    assert second_metadata["representable_population_complete"] is False


def test_empty_valid_catalog_marks_representability_complete() -> None:
    points = torch.stack(
        [torch.arange(8, dtype=torch.float32), torch.zeros(8), torch.zeros(8)],
        dim=1,
    )
    frames = {cell: () for cell in DEFAULT_2D_CELLS}
    with pytest.raises(NoRetainedFrameGroupsError) as caught:
        sample_multigranular_frame_group_plan(
            scene_id="scene_empty_catalog",
            epoch=0,
            seed=9,
            points=points,
            frames_by_cell=frames,
            total_proposal_quota=1,
            positive_pairs_per_proposal=1,
            num_uniform_negatives=1,
            num_spatial_hard_negatives=0,
            num_feature_hard_negatives=0,
            require_negative_proposal_membership=True,
        )

    plan = caught.value.plan
    assert plan.groups == ()
    assert all(
        coverage["representable_population_complete"] is True
        for coverage in plan.coverage_by_cell.values()
    )


def test_ancestry_matches_wrapper_voxel_and_pooling_maps() -> None:
    points = torch.tensor(
        [
            [-0.111, 0.021, 0.0],
            [-0.108, 0.021, 0.0],
            [-0.071, 0.039, 0.0],
            [0.010, 0.041, 0.0],
            [0.051, 0.079, 0.0],
            [0.052, 0.079, 0.0],
            [0.131, 0.121, 0.0],
        ],
        dtype=torch.float32,
    )
    ancestry = derive_litept_hierarchy_ancestry(points)
    representatives = ancestry.representative_indices.tolist()
    assert 0 in representatives and 1 not in representatives
    assert 4 in representatives and 5 not in representatives
    assert len(representatives) == int(ancestry.stage_grids["dec0"].shape[0])

    stage_names = ("dec0", "dec1", "dec2", "dec3", "enc4")
    for index, parent_map in enumerate(ancestry.parent_maps):
        child = stage_names[index]
        parent = stage_names[index + 1]
        expected = LitePTBackbone._grid_parent_map(
            ancestry.stage_grids[child],
            torch.tensor([ancestry.stage_grids[child].shape[0]]),
            ancestry.stage_grids[parent],
            torch.tensor([ancestry.stage_grids[parent].shape[0]]),
            2,
        )
        assert torch.equal(parent_map, expected)
        assert torch.equal(
            ancestry.point_to_stage[parent],
            expected[ancestry.point_to_stage[child]],
        )


def _catalog(cell: str, frame_id: str, start: int) -> FrameProposalCatalog:
    # Three disjoint two-point masks and a shared 24-point visible universe.
    members = torch.tensor(
        [start, start + 1, start + 2, start + 3, start + 4, start + 5],
        dtype=torch.long,
    )
    return FrameProposalCatalog(
        cell=cell,
        frame_id=frame_id,
        visible_indices=torch.arange(24, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.long),
        proposal_point_indices=members,
    )


def _unsafe_single_catalog(cell: str, frame_id: str) -> FrameProposalCatalog:
    return FrameProposalCatalog(
        cell=cell,
        frame_id=frame_id,
        visible_indices=torch.arange(24, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 2], dtype=torch.long),
        proposal_point_indices=torch.tensor([0, 1], dtype=torch.long),
    )


def test_preselection_uses_candidate_visibility_for_stage_support() -> None:
    # Points 2 and 3 occupy distinct dec0 voxels but the same dec1 parent.
    # A sibling catalog sees point 3; the candidate catalog does not.  The
    # sibling must not make the candidate's partially visible dec1 negative
    # token look fully known.
    points = torch.tensor(
        [[0.08, 0.0, 0.0], [0.12, 0.0, 0.0], [0.00, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=torch.float32,
    )
    candidate = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="visibility-frame",
        visible_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        proposal_offsets=torch.tensor([0, 2], dtype=torch.long),
        proposal_point_indices=torch.tensor([0, 1], dtype=torch.long),
    )
    sibling = FrameProposalCatalog(
        cell="2d/g05",
        frame_id="visibility-frame",
        visible_indices=torch.arange(4, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 2], dtype=torch.long),
        proposal_point_indices=torch.tensor([2, 3], dtype=torch.long),
    )
    ancestry = derive_litept_hierarchy_ancestry(points)
    support = sampler_v2._frame_relation_stage_support(
        (candidate, sibling), ancestry, num_points=4
    )
    evidence = sampler_v2._proposal_preselection_evidence(
        catalog=candidate,
        proposal_index=0,
        granularity="g02",
        ancestry=ancestry,
        frame_support=support,
        require_negative_proposal_membership=False,
    )

    assert evidence["stage_support"]["dec1"]["positive_support"] is True
    assert evidence["stage_support"]["dec1"]["negative_support"] is False
    assert evidence["relation_safe_negative_exists"] is True
    assert evidence["eligible"] is False
    assert "missing_valid_dec1_fallback_route" in evidence["reason"]

    dec0_only_evidence = sampler_v2._proposal_preselection_evidence(
        catalog=candidate,
        proposal_index=0,
        granularity="g02",
        ancestry=ancestry,
        frame_support=support,
        require_negative_proposal_membership=False,
        require_hierarchy_routes=False,
    )
    assert dec0_only_evidence["hierarchy_route_eligibility_required"] is False
    assert dec0_only_evidence["eligible"] is True
    assert dec0_only_evidence["reason"] is None


def test_preselection_requires_every_pure_token_pair_to_have_safe_negative() -> None:
    points = torch.stack(
        [torch.arange(4, dtype=torch.float32), torch.zeros(4), torch.zeros(4)],
        dim=1,
    )
    catalog = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="one-viable-pair",
        visible_indices=torch.arange(4, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 3, 5], dtype=torch.long),
        proposal_point_indices=torch.tensor([0, 1, 2, 2, 3], dtype=torch.long),
    )
    ancestry = derive_litept_hierarchy_ancestry(points)
    support = sampler_v2._frame_relation_stage_support(
        (catalog,), ancestry, num_points=4
    )
    evidence = sampler_v2._proposal_preselection_evidence(
        catalog=catalog,
        proposal_index=0,
        granularity="g02",
        ancestry=ancestry,
        frame_support=support,
        require_negative_proposal_membership=False,
    )

    # Token 3 is safe for the (0,1) pair, but the relation [2,3] makes it
    # unsafe for every pair containing token 2.  Existence of one viable pair
    # must not allow a random/tokenized draw to make the group disappear.
    assert evidence["pure_dec0_token_count"] == 3
    assert evidence["relation_safe_negative_exists"] is False
    assert evidence["eligible"] is False
    assert "no_relation_safe_negative_for_two_pure_dec0_endpoints" in evidence[
        "reason"
    ]


def test_preselection_checks_pairs_within_the_same_endpoint_chunk() -> None:
    points = torch.stack(
        [torch.arange(5, dtype=torch.float32), torch.zeros(5), torch.zeros(5)],
        dim=1,
    )
    catalog = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="same-chunk-unsafe-pair",
        visible_indices=torch.arange(5, dtype=torch.long),
        # Proposal 0 is selected.  Proposal 1 makes negative 3 ambiguous for
        # endpoint 1; proposal 2 makes negative 4 ambiguous for endpoint 2.
        # Thus (0,1) can use 4 and (0,2) can use 3, while (1,2) has no shared
        # safe negative.  All three positives lie in the same 256-token chunk.
        proposal_offsets=torch.tensor([0, 3, 5, 7], dtype=torch.long),
        proposal_point_indices=torch.tensor(
            [0, 1, 2, 1, 3, 2, 4], dtype=torch.long
        ),
    )
    ancestry = derive_litept_hierarchy_ancestry(points)
    support = sampler_v2._frame_relation_stage_support(
        (catalog,), ancestry, num_points=5
    )
    evidence = sampler_v2._proposal_preselection_evidence(
        catalog=catalog,
        proposal_index=0,
        granularity="g02",
        ancestry=ancestry,
        frame_support=support,
        require_negative_proposal_membership=False,
    )

    assert evidence["pure_dec0_token_count"] == 3
    assert evidence["relation_safe_negative_exists"] is False
    assert evidence["eligible"] is False


@pytest.mark.parametrize("require_negative_membership", [False, True])
def test_catalog_preselection_context_matches_uncached_evidence(
    require_negative_membership: bool,
) -> None:
    points = torch.stack(
        [torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.zeros(10)],
        dim=1,
    )
    catalog = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="cached-preselection",
        visible_indices=torch.arange(10, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 4, 7], dtype=torch.long),
        proposal_point_indices=torch.tensor(
            [0, 1, 1, 2, 2, 3, 3], dtype=torch.long
        ),
    )
    ancestry = derive_litept_hierarchy_ancestry(points)
    support = sampler_v2._frame_relation_stage_support(
        (catalog,), ancestry, num_points=10
    )
    context = sampler_v2._catalog_preselection_context(
        catalog=catalog,
        ancestry=ancestry,
        frame_support=support,
        num_points=10,
    )

    assert context["unique_proposal_offsets"].tolist() == [0, 3, 5]
    assert context["unique_proposal_members"].tolist() == [0, 1, 2, 2, 3]
    for proposal_index in range(2):
        uncached = sampler_v2._proposal_preselection_evidence(
            catalog=catalog,
            proposal_index=proposal_index,
            granularity="g02",
            ancestry=ancestry,
            frame_support=support,
            require_negative_proposal_membership=require_negative_membership,
        )
        cached = sampler_v2._proposal_preselection_evidence(
            catalog=catalog,
            proposal_index=proposal_index,
            granularity="g02",
            ancestry=ancestry,
            frame_support=support,
            require_negative_proposal_membership=require_negative_membership,
            catalog_context=context,
        )
        assert cached == uncached


def test_epoch_after_ancestry_preselection_requires_persisted_state() -> None:
    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (_catalog(cell, "persisted-frame", 0),)
        for cell in DEFAULT_2D_CELLS
    }
    with pytest.raises(ValueError, match="persisted coverage state"):
        sample_multigranular_frame_group_plan(
            scene_id="scene_state_required",
            epoch=1,
            seed=42,
            points=points,
            frames_by_cell=frames,
            total_proposal_quota=1,
            positive_pairs_per_proposal=1,
            num_uniform_negatives=1,
            num_spatial_hard_negatives=0,
            num_feature_hard_negatives=0,
        )


def test_preselection_skips_all_co_contained_row_and_replaces_from_later_safe_row() -> None:
    points = torch.stack(
        [torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.zeros(10)],
        dim=1,
    )
    bad = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="shared-relation-frame",
        visible_indices=torch.arange(6, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 4], dtype=torch.long),
        proposal_point_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    safe = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="shared-relation-frame",
        visible_indices=torch.arange(10, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 4], dtype=torch.long),
        proposal_point_indices=torch.tensor([6, 7, 8, 9], dtype=torch.long),
    )
    all_contained_relation = FrameProposalCatalog(
        cell="2d/g05",
        frame_id="shared-relation-frame",
        visible_indices=torch.arange(6, dtype=torch.long),
        proposal_offsets=torch.tensor([0, 6], dtype=torch.long),
        proposal_point_indices=torch.arange(6, dtype=torch.long),
    )
    frames = {
        "2d/g02": (bad, safe),
        "2d/g05": (all_contained_relation,),
        "2d/g08": (),
    }
    raw_entries = [(0, 0, "shared-relation-frame"), (1, 0, "shared-relation-frame")]
    fingerprint = sampler_v2._catalog_fingerprint(raw_entries)
    seed = None
    for candidate_seed in range(128):
        candidate_state = DeterministicCoverageState(
            seed=candidate_seed,
            epoch=0,
            scene_id="scene_relation_replace",
            operation="multigranular-train-v2/2d/g02",
            population_size=2,
            population_fingerprint=fingerprint,
        )
        if candidate_state._permutation().tolist() == [0, 1]:
            seed = candidate_seed
            break
    assert seed is not None

    plan = sample_multigranular_frame_group_plan(
        scene_id="scene_relation_replace",
        epoch=0,
        seed=seed,
        points=points,
        frames_by_cell=frames,
        total_proposal_quota=16,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
    )
    coverage = plan.coverage_by_cell["2d/g02"]
    assert coverage["representable_population"] == 1
    assert coverage["realized_count"] == 1
    assert coverage["preselection_skipped_count"] == 1
    assert coverage["preselection_skipped_entries"][0]["frame_position"] == 0
    assert coverage["preselection_skipped_entries"][0][
        "relation_safe_negative_exists"
    ] is False
    assert coverage["selected_entries"] == [
        {
            "frame_position": 1,
            "proposal_index": 0,
            "frame_id": "shared-relation-frame",
        }
    ]
    assert [group.frame_id for group in plan.groups] == [
        "shared-relation-frame"
    ]


def test_multigranular_groups_rotate_quotas_and_remain_frame_local() -> None:
    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (
            _catalog(cell, "frame-a", 0),
            _catalog(cell, "frame-b", 6),
        )
        for cell in DEFAULT_2D_CELLS
    }
    kwargs = dict(
        scene_id="scene_00002",
        epoch=0,
        seed=42,
        points=points,
        frames_by_cell=frames,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=1,
        spatial_candidate_pool=6,
        num_feature_hard_negatives=1,
        feature_candidate_pool=4,
        require_negative_proposal_membership=True,
    )
    plan = sample_multigranular_frame_group_plan(**kwargs)
    first = plan.groups
    second = sample_multigranular_frame_groups(**kwargs)

    assert rotating_three_cell_quotas(0) == {
        "2d/g02": 6,
        "2d/g05": 5,
        "2d/g08": 5,
    }
    assert rotating_three_cell_quotas(1) == {
        "2d/g02": 5,
        "2d/g05": 6,
        "2d/g08": 5,
    }
    assert [group.cell for group in first] == sorted(
        [group.cell for group in first], key=DEFAULT_2D_CELLS.index
    )
    assert sum(len(group.selected_proposal_indices) for group in first) == 16
    assert plan.requested_proposal_count == 16
    assert plan.realized_proposal_count == 16
    assert plan.contrastive_retained_proposal_count == 16
    assert not plan.audit_metadata()["adaptive_truncation"]
    assert json.loads(json.dumps(plan.state_after_by_cell()))["2d/g02"][
        "cursor"
    ] == 6
    assert len({group.frame_id for group in first if group.cell == "2d/g02"}) == 2
    assert [
        (group.cell, group.frame_id, group.selected_proposal_indices)
        for group in first
    ] == [
        (group.cell, group.frame_id, group.selected_proposal_indices)
        for group in second
    ]
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.batch.positive_pairs, right.batch.positive_pairs)
        assert torch.equal(left.batch.negative_indices, right.batch.negative_indices)
        assert len(left.selected_proposal_indices) == len(
            set(left.selected_proposal_indices)
        )
        assert left.coverage_metadata["duplicate_count"] == 0
        assert left.batch.relation_proposal_offsets is not None
        assert int(left.batch.relation_proposal_offsets.numel()) - 1 == 9
        assert left.batch.eligible_indices is not None
        assert int(left.batch.eligible_indices.numel()) == 6
        assert left.batch.token_positive_pair_offsets is not None
        assert left.batch.token_positive_pair_indices is not None
        assert left.batch.token_positive_pair_digest
        assert int(left.batch.token_positive_pair_offsets.numel()) == (
            len(left.selected_proposal_indices) + 1
        )
        assert int(left.batch.token_positive_pair_indices.shape[0]) == (
            left.batch.num_pairs
        )
        assert left.coverage_metadata["sampler"][
            "token_positive_pair_payload"
        ] is True


def test_scene_plan_reports_empty_and_exhausted_cells_without_padding() -> None:
    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        "2d/g02": (_catalog("2d/g02", "frame-a", 0),),
        "2d/g05": (),
        "2d/g08": (
            _catalog("2d/g08", "frame-a", 6),
            _catalog("2d/g08", "frame-b", 12),
        ),
    }
    plan = sample_multigranular_frame_group_plan(
        scene_id="scene_exhausted",
        epoch=0,
        seed=42,
        points=points,
        frames_by_cell=frames,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
    )

    assert plan.requested_proposal_count == 16
    assert plan.realized_proposal_count == 8
    assert plan.contrastive_retained_proposal_count == 8
    assert sum(
        len(group.selected_proposal_indices) for group in plan.groups
    ) == 8
    assert plan.coverage_by_cell["2d/g02"]["requested_count"] == 6
    assert plan.coverage_by_cell["2d/g02"]["realized_count"] == 3
    assert plan.coverage_by_cell["2d/g02"]["adaptive_truncation"]
    empty = plan.coverage_by_cell["2d/g05"]
    assert empty["requested_count"] == 5
    assert empty["realized_count"] == 0
    assert empty["state_after"] is None
    assert empty["no_group_reason"] == "empty_valid_proposal_catalog"
    assert empty["contrastive_retained_count"] == 0
    assert plan.coverage_by_cell["2d/g08"]["realized_count"] == 5
    assert plan.audit_metadata()["adaptive_truncation"]


def test_scene_plan_records_unsafe_frame_drop_and_continues_other_groups() -> None:
    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (
            _unsafe_single_catalog(cell, "frame-unsafe"),
            _catalog(cell, "frame-safe", 6),
        )
        for cell in DEFAULT_2D_CELLS
    }
    plan = sample_multigranular_frame_group_plan(
        scene_id="scene_partial_survival",
        epoch=0,
        seed=42,
        points=points,
        frames_by_cell=frames,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
    )

    assert plan.groups
    assert {group.frame_id for group in plan.groups} == {"frame-safe"}
    assert plan.contrastive_retained_proposal_count == 9
    for cell in DEFAULT_2D_CELLS:
        coverage = plan.coverage_by_cell[cell]
        assert coverage["attempted_frame_group_count"] == 1
        assert coverage["frame_group_count"] == 1
        assert coverage["dropped_frame_group_count"] == 0
        assert coverage["raw_population"] == 4
        assert coverage["representable_population_count"] == 3
        ineligible = [
            row
            for row in coverage["preselection_raw_eligibility"]
            if not row["eligible"]
        ]
        assert len(ineligible) == 1
        assert ineligible[0]["frame_id"] == "frame-unsafe"
        assert ineligible[0]["reason"]


def test_scene_plan_raises_typed_error_with_accounting_when_all_groups_drop() -> None:
    points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    )
    frames = {
        cell: (_unsafe_single_catalog(cell, "frame-unsafe"),)
        for cell in DEFAULT_2D_CELLS
    }
    with pytest.raises(NoRetainedFrameGroupsError) as caught:
        sample_multigranular_frame_group_plan(
            scene_id="scene_zero_survival",
            epoch=0,
            seed=42,
            points=points,
            frames_by_cell=frames,
            positive_pairs_per_proposal=1,
            num_uniform_negatives=1,
            num_spatial_hard_negatives=0,
            num_feature_hard_negatives=0,
            require_negative_proposal_membership=True,
        )

    plan = caught.value.plan
    assert plan.groups == ()
    assert plan.realized_proposal_count == 0
    assert plan.contrastive_retained_proposal_count == 0
    assert sum(
        row["preselection_skipped_count"]
        for row in plan.coverage_by_cell.values()
    ) == 3
    assert all(
        row["no_group_reason"] == "no_preselection_representable_proposal"
        for row in plan.coverage_by_cell.values()
    )


def _assert_v2_plans_equal(left, right) -> None:
    assert left.quota_by_cell == right.quota_by_cell
    assert left.coverage_by_cell == right.coverage_by_cell
    assert len(left.groups) == len(right.groups)
    for left_group, right_group in zip(left.groups, right.groups, strict=True):
        assert left_group.cell == right_group.cell
        assert left_group.frame_id == right_group.frame_id
        assert left_group.selected_proposal_indices == right_group.selected_proposal_indices
        assert left_group.coverage_metadata == right_group.coverage_metadata
        left_batch = left_group.batch
        right_batch = right_group.batch
        for field in dataclasses.fields(left_batch):
            left_value = getattr(left_batch, field.name)
            right_value = getattr(right_batch, field.name)
            if isinstance(left_value, torch.Tensor):
                assert isinstance(right_value, torch.Tensor)
                assert torch.equal(left_value, right_value), field.name
            else:
                assert left_value == right_value, field.name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_planning_matches_cuda_planning_for_controlled_and_adversarial_masks() -> None:
    """CPU planning must be a pure placement optimization, not a new sampler."""

    controlled_points = torch.stack(
        [torch.arange(24, dtype=torch.float32), torch.zeros(24), torch.zeros(24)],
        dim=1,
    ).cuda()
    controlled_frames = {
        cell: (_catalog(cell, "controlled-frame", 0),)
        for cell in DEFAULT_2D_CELLS
    }
    controlled_frames = {
        cell: tuple(
            replace(
                catalog,
                visible_indices=catalog.visible_indices.cuda(),
                proposal_offsets=catalog.proposal_offsets.cuda(),
                proposal_point_indices=catalog.proposal_point_indices.cuda(),
            )
            for catalog in catalogs
        )
        for cell, catalogs in controlled_frames.items()
    }
    common = dict(
        scene_id="planning_equivalence_controlled",
        epoch=0,
        seed=42,
        points=controlled_points,
        frames_by_cell=controlled_frames,
        total_proposal_quota=4,
        positive_pairs_per_proposal=2,
        num_uniform_negatives=2,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=True,
    )
    old = sample_multigranular_frame_group_plan(
        **common, planning_device=controlled_points.device
    )
    timing: dict[str, object] = {}
    new = sample_multigranular_frame_group_plan(
        **common, planning_device="cpu", timing=timing
    )
    _assert_v2_plans_equal(old, new)
    assert timing["triplet_calls"] == len(new.groups)
    assert timing["ancestry_ms"] >= 0.0
    assert timing["planning_transfer_ms"] >= 0.0
    assert timing["preselection_setup_ms"] >= 0.0
    assert timing["preselection_evidence_ms"] >= 0.0
    assert timing["preselection_evidence_calls"] == timing["preselection_proposal_count"]
    assert timing["preselection_eligible_count"] + timing["preselection_rejected_count"] == timing["preselection_proposal_count"]
    assert timing["triplet_sampling_total_ms"] >= 0.0
    assert timing["planning_device"] == "cpu"
    assert timing["cpu_integer_planning"] is True
    assert timing["deferred_safety_device"] == "cpu"
    assert timing["relation_context_count"] == 1
    assert timing["catalog_context_count"] == 3

    # Recreate the deterministic call state for a cold-vs-warm measurement.
    # The test intentionally checks every repeat, so a faster placement must
    # not alter proposal order, relation evidence, or any batch metadata.
    warm_plans = []
    warm_timings = []
    for _ in range(3):
        repeat_timing: dict[str, object] = {}
        repeat = sample_multigranular_frame_group_plan(
            **common, planning_device="cpu", timing=repeat_timing
        )
        _assert_v2_plans_equal(new, repeat)
        warm_plans.append(repeat)
        warm_timings.append(repeat_timing)
    assert len(warm_plans) == 3
    assert all(
        float(repeat_timing["triplet_sampling_total_ms"]) >= 0.0
        for repeat_timing in warm_timings
    )

    # The relation graph deliberately contains the same-chunk unsafe pair
    # exercised by the preselection regression.  CPU placement must preserve
    # the fail-closed selection and every serialized evidence field.
    adversarial_points = torch.stack(
        [torch.arange(5, dtype=torch.float32), torch.zeros(5), torch.zeros(5)],
        dim=1,
    ).cuda()
    adversarial_catalog = FrameProposalCatalog(
        cell="2d/g02",
        frame_id="adversarial-frame",
        visible_indices=torch.arange(5, dtype=torch.long).cuda(),
        proposal_offsets=torch.tensor([0, 3, 5, 7], dtype=torch.long).cuda(),
        proposal_point_indices=torch.tensor(
            [0, 1, 2, 1, 3, 2, 4], dtype=torch.long
        ).cuda(),
    )
    safe_catalogs = {
        cell: (replace(adversarial_catalog, cell=cell),)
        for cell in DEFAULT_2D_CELLS
    }
    adversarial_common = dict(
        scene_id="planning_equivalence_adversarial",
        epoch=0,
        seed=17,
        points=adversarial_points,
        frames_by_cell=safe_catalogs,
        total_proposal_quota=1,
        positive_pairs_per_proposal=1,
        num_uniform_negatives=1,
        num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0,
        require_negative_proposal_membership=False,
    )
    with pytest.raises(NoRetainedFrameGroupsError) as old_error:
        sample_multigranular_frame_group_plan(
            **adversarial_common, planning_device=adversarial_points.device
        )
    with pytest.raises(NoRetainedFrameGroupsError) as new_error:
        sample_multigranular_frame_group_plan(
            **adversarial_common, planning_device="cpu"
        )
    _assert_v2_plans_equal(old_error.value.plan, new_error.value.plan)
