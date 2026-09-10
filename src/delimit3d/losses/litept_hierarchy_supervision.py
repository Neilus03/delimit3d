"""Transfer-aligned supervision for the native LitePT feature hierarchy.

The mask objective is proposal-conditioned: a query pooled from one half of a
proposal's finest tokens predicts the held-out half at a routed hierarchy
stage.  Targets remain soft when pooling mixes proposal and complement points,
and visibility-unknown support only reduces confidence instead of becoming a
false negative.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from delimit3d.data.partfield_contrastive import (
    SAMPLER_ALGORITHM_VERSION,
    SAMPLER_PRESELECTION_POLICY,
    SAMPLER_SEED_SCHEME,
    TOKEN_POSITIVE_PAIR_PAYLOAD_SCHEMA,
    ContrastiveTripletBatch,
    build_relation_membership_index,
    co_membership_safe_negative_mask,
    sample_unique_positive_pairs,
    validate_token_positive_pair_payload,
)


HIERARCHY_FINE_TO_COARSE = ("dec0", "dec1", "dec2", "dec3", "enc4")
HIERARCHY_COARSE_TO_FINE = tuple(reversed(HIERARCHY_FINE_TO_COARSE))
HIERARCHY_STAGE_CHANNELS: dict[str, int] = {
    "enc4": 504,
    "dec3": 252,
    "dec2": 144,
    "dec1": 72,
    "dec0": 72,
}
# ``shared_dec0`` is the original C3--C7 behaviour: one detached-dec0 query
# projection has to serve every native target stage.  C8 isolates that
# disposable head per routed target stage while preserving the same detached
# dec0 tokens, folds, keys, targets, and backbone-facing loss.  The policy is
# deliberately explicit because it changes auxiliary-head optimization, not
# the transfer backbone interface.
HIERARCHY_QUERY_PROJECTION_POLICIES = (
    "shared_dec0",
    "per_stage_dec0",
)
_INTEGRAL_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

# g02 chooses exactly one of its two stages per proposal.  It uses dec2 only
# when the coarse target has enough known positive and negative support;
# otherwise it falls back to dec1.  The other granularities supervise both
# listed stages.  Realized fold losses are pooled within each granularity and
# the non-empty granularities are then given equal weight.
MASK_ROUTE_STAGES: dict[str, tuple[str, ...]] = {
    "g02": ("dec1", "dec2"),
    "g05": ("dec2", "dec3"),
    "g08": ("dec3", "enc4"),
}

# Keep the approved cumulative routed-survival threshold visible in the loss
# telemetry.  Visit-level trainability still uses assignment and non-zero
# fold denominators; the health audit applies this threshold over the agreed
# cumulative window rather than discarding an otherwise valid microbatch.
MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION = 0.90


@dataclass(frozen=True)
class MaskSupervisionGroup:
    """One frame-local proposal group and its explicit source cell."""

    cell: str
    batch: ContrastiveTripletBatch
    group_id: str | None = None


@dataclass(frozen=True)
class _TokenRebuildBase:
    """Scene/token computations shared by frame groups on one encoded scene."""

    relation_token_offsets: torch.Tensor
    relation_token_members: torch.Tensor
    relation_index: Any
    all_counts: torch.Tensor
    eligible_counts: torch.Tensor
    fully_known: torch.Tensor
    fully_known_tokens: torch.Tensor


_TOKEN_REBUILD_CACHE_LIMIT = 4
_TOKEN_REBUILD_CACHE: OrderedDict[
    tuple[Any, ...],
    tuple[tuple[weakref.ReferenceType[Any], ...], _TokenRebuildBase],
] = OrderedDict()


def _token_rebuild_cache_key(*values: torch.Tensor) -> tuple[Any, ...]:
    """Identify immutable tensor storage without retaining scene inputs."""

    return tuple(
        item
        for value in values
        for item in (
            id(value),
            int(value.data_ptr()),
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            str(value.device),
            int(getattr(value, "_version", 0)),
        )
    )


def _cached_token_rebuild_base(
    *,
    point_to_token: torch.Tensor,
    eligible: torch.Tensor,
    relation_offsets: torch.Tensor,
    relation_members: torch.Tensor,
    token_count: int,
) -> _TokenRebuildBase | None:
    key = _token_rebuild_cache_key(
        point_to_token, eligible, relation_offsets, relation_members
    ) + (int(token_count),)
    entry = _TOKEN_REBUILD_CACHE.get(key)
    if entry is None:
        return None
    references, base = entry
    if any(
        reference() is not value
        for reference, value in zip(
            references,
            (point_to_token, eligible, relation_offsets, relation_members),
            strict=True,
        )
    ):
        _TOKEN_REBUILD_CACHE.pop(key, None)
        return None
    _TOKEN_REBUILD_CACHE.move_to_end(key)
    return base


def _store_token_rebuild_base(
    *,
    point_to_token: torch.Tensor,
    eligible: torch.Tensor,
    relation_offsets: torch.Tensor,
    relation_members: torch.Tensor,
    token_count: int,
    base: _TokenRebuildBase,
) -> None:
    key = _token_rebuild_cache_key(
        point_to_token, eligible, relation_offsets, relation_members
    ) + (int(token_count),)
    _TOKEN_REBUILD_CACHE[key] = (
        (
            weakref.ref(point_to_token),
            weakref.ref(eligible),
            weakref.ref(relation_offsets),
            weakref.ref(relation_members),
        ),
        base,
    )
    _TOKEN_REBUILD_CACHE.move_to_end(key)
    while len(_TOKEN_REBUILD_CACHE) > _TOKEN_REBUILD_CACHE_LIMIT:
        _TOKEN_REBUILD_CACHE.popitem(last=False)


def hierarchy_point_maps(
    original_to_finest: torch.Tensor,
    adjacent_parent_maps: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compose exact point-to-stage maps from adjacent LitePT parent maps."""

    if original_to_finest.ndim != 1:
        raise ValueError("original_to_finest must be one-dimensional")
    if original_to_finest.dtype not in _INTEGRAL_DTYPES:
        raise TypeError("original_to_finest must be integral")
    if len(adjacent_parent_maps) != len(HIERARCHY_FINE_TO_COARSE) - 1:
        raise ValueError(
            "Expected four adjacent maps dec0->dec1->dec2->dec3->enc4, got "
            f"{len(adjacent_parent_maps)}"
        )
    current = original_to_finest.long()
    if current.numel() and int(current.min().item()) < 0:
        raise ValueError("original_to_finest contains negatives")
    output = {"dec0": current}
    child_count = (
        int(current.max().item()) + 1 if current.numel() else 0
    )
    for child, parent, mapping in zip(
        HIERARCHY_FINE_TO_COARSE[:-1],
        HIERARCHY_FINE_TO_COARSE[1:],
        adjacent_parent_maps,
        strict=True,
    ):
        if mapping.dtype not in _INTEGRAL_DTYPES:
            raise TypeError(f"Adjacent map {child}->{parent} must be integral")
        mapping = mapping.to(device=current.device, dtype=torch.long).flatten()
        if mapping.numel() != child_count:
            raise ValueError(
                f"Adjacent map {child}->{parent} has {mapping.numel()} rows; "
                f"expected {child_count}"
            )
        if mapping.numel() and int(mapping.min().item()) < 0:
            raise ValueError(f"Adjacent map {child}->{parent} contains negatives")
        if current.numel() and int(current.max().item()) >= mapping.numel():
            raise ValueError(f"Point map indexes outside {child}->{parent}")
        current = mapping[current]
        output[parent] = current
        child_count = int(mapping.max().item()) + 1 if mapping.numel() else 0
    return output


def _granularity(cell: str) -> str:
    value = str(cell).strip()
    parts = value.split("/", 1)
    granularity = parts[-1]
    if len(parts) == 2 and parts[0] != "2d":
        raise ValueError(
            "Proposal-conditioned mask hierarchy currently supports 2D cells, "
            f"got {cell!r}"
        )
    if granularity not in MASK_ROUTE_STAGES:
        raise ValueError(
            f"Unknown mask granularity {cell!r}; expected 2d/g02, 2d/g05, or 2d/g08"
        )
    return granularity


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _token_proposal_stream(
    *,
    root_seed: int,
    proposal_index: int,
    family: str,
    device: torch.device,
) -> torch.Generator:
    """Create a token-space stream independent of other families/proposals."""

    return torch.Generator(device=device).manual_seed(
        _stable_seed(
            SAMPLER_ALGORITHM_VERSION,
            int(root_seed),
            int(proposal_index),
            str(family),
        )
    )


def _zero_touch(reference: torch.Tensor, module: nn.Module) -> torch.Tensor:
    loss = reference.sum() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            loss = loss + parameter.sum() * 0.0
    return loss


def _validate_membership(
    batch: ContrastiveTripletBatch,
    *,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = (
        batch.proposal_member_offsets,
        batch.proposal_member_indices,
        batch.eligible_indices,
    )
    if any(value is None for value in fields):
        raise ValueError(
            "Hierarchy mask supervision requires proposal membership CSR and eligible_indices"
        )
    offsets, members, eligible = fields
    assert offsets is not None and members is not None and eligible is not None
    if not (offsets.device == members.device == eligible.device):
        raise ValueError("proposal membership tensors must share one device")
    offsets = offsets.long().flatten()
    members = members.long().flatten()
    eligible = eligible.long().flatten()
    if offsets.numel() < 2 or int(offsets[0].item()) != 0:
        raise ValueError("proposal_member_offsets must start at zero and contain a proposal")
    if bool((offsets < 0).any().item()):
        raise ValueError("proposal_member_offsets must be non-negative")
    if int(offsets[-1].item()) != members.numel():
        raise ValueError("proposal_member_offsets do not span proposal_member_indices")
    if bool((offsets[1:] <= offsets[:-1]).any().item()):
        raise ValueError("Every proposal occurrence must contain at least one member")
    for name, indices in (("members", members), ("eligible", eligible)):
        if indices.numel() == 0:
            raise ValueError(f"{name} indices must be non-empty")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_points:
            raise ValueError(f"{name} indices are outside the scene")
    if torch.unique(eligible).numel() != eligible.numel():
        raise ValueError("eligible_indices must not contain duplicates")
    eligible_mask = torch.zeros(
        num_points, dtype=torch.bool, device=eligible.device
    )
    eligible_mask[eligible] = True
    if not bool(eligible_mask[members].all().item()):
        raise ValueError("Proposal members must be a subset of eligible points")
    return offsets, members, eligible


def _counts_at_sorted_tokens(
    point_indices: torch.Tensor,
    point_to_stage: torch.Tensor,
    sorted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Count selected points into an already-sorted sparse token universe."""

    result = torch.zeros(
        sorted_tokens.numel(), device=point_to_stage.device, dtype=torch.float32
    )
    if point_indices.numel() == 0:
        return result
    token_ids, counts = torch.unique(
        point_to_stage[point_indices], sorted=True, return_counts=True
    )
    positions = torch.searchsorted(sorted_tokens, token_ids)
    if (
        positions.numel()
        and (
            int(positions.max().item()) >= sorted_tokens.numel()
            or not bool(torch.equal(sorted_tokens[positions], token_ids))
        )
    ):
        raise RuntimeError("Proposal token was absent from the eligible token universe")
    result[positions] = counts.float()
    return result


def _sorted_row_exclusion_mask(
    candidate_tokens: torch.Tensor,
    excluded_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Return a row-wise exclusion mask without repeated ``torch.isin`` calls.

    Token candidates are already a sorted, unique universe in the V2 rebuild.
    Sorting the small per-row exclusion lists once lets ``searchsorted`` test
    every row in one tensor operation; this keeps the exact membership result
    while avoiding one synchronizing ``isin`` kernel per proposal/pair row.
    """

    if excluded_indices is None or excluded_indices.shape[1] == 0:
        return candidate_tokens.new_zeros(
            (int(excluded_indices.shape[0]) if excluded_indices is not None else 0,
             int(candidate_tokens.numel())),
            dtype=torch.bool,
        )
    rows = int(excluded_indices.shape[0])
    candidates = candidate_tokens.reshape(1, -1).expand(rows, -1).contiguous()
    sorted_excluded = excluded_indices.long().sort(dim=1).values
    positions = torch.searchsorted(sorted_excluded, candidates)
    in_range = positions < int(sorted_excluded.shape[1])
    safe_positions = positions.clamp_max(int(sorted_excluded.shape[1]) - 1)
    return in_range & (
        sorted_excluded.gather(1, safe_positions) == candidates
    )


def _sample_sorted_unique_values(
    values: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample from sorted unique values with the sampler's exact RNG contract."""

    population_size = int(values.numel())
    count = min(int(count), population_size)
    if count == 0:
        return values.new_empty((0,), dtype=torch.long)
    # Match ``sample_unique_without_replacement`` and its integer-range helper:
    # the candidate universe in normal scenes is below this bounded threshold,
    # while the rejection branch preserves the same stream for very large
    # token universes.
    if population_size <= 1_000_000 or count * 4 >= population_size:
        draw = torch.randperm(
            population_size,
            generator=generator,
            device=values.device,
        )[:count]
        return values[draw].long()
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < count:
        remaining = count - len(selected)
        draws = torch.randint(
            population_size,
            (max(16, remaining * 2),),
            generator=generator,
            device=values.device,
        )
        for value in draws.tolist():
            integer = int(value)
            if integer not in seen:
                seen.add(integer)
                selected.append(integer)
                if len(selected) == count:
                    break
    indices = torch.tensor(selected, dtype=torch.long, device=values.device)
    return values[indices].long()


def _sample_sorted_unique_negative_rows(
    *,
    candidate_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    requested_per_row: int,
    excluded_indices: torch.Tensor | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rectangular rows from a sorted token universe.

    This is the narrow V2 rebuild specialization of
    ``sample_adaptive_unique_negative_rows``.  ``candidate_tokens`` and each
    masked row are sorted and unique, so the generic helper's per-row
    ``unique``/``isin`` work can be replaced by one vectorized exclusion pass.
    Random draws remain row-ordered and use the same full-permutation versus
    rejection branch as the shared sampler.
    """

    if candidate_tokens.ndim != 1 or valid_mask.ndim != 2:
        raise ValueError("sorted token rows require [N] candidates and [P,N] mask")
    if valid_mask.shape[1] != candidate_tokens.numel():
        raise ValueError("sorted token row mask does not match candidates")
    rows = int(valid_mask.shape[0])
    excluded_mask = _sorted_row_exclusion_mask(
        candidate_tokens, excluded_indices
    )
    if excluded_indices is None or excluded_indices.shape[1] == 0:
        row_mask = valid_mask.bool()
    else:
        row_mask = valid_mask.bool() & ~excluded_mask
    available = row_mask.sum(dim=1).long()
    selected_count = min(
        int(requested_per_row),
        int(available.min().item()) if available.numel() else 0,
    )
    selected_rows = [
        _sample_sorted_unique_values(
            candidate_tokens[row_mask[row]],
            selected_count,
            generator=generator,
        )
        for row in range(rows)
    ]
    selected = (
        torch.stack(selected_rows, dim=0)
        if selected_rows
        else candidate_tokens.new_empty((0, selected_count), dtype=torch.long)
    )
    return selected.long(), available


def _pure_dec0_query_partition(
    *,
    proposal_members: torch.Tensor,
    point_to_dec0: torch.Tensor,
    dec0_all_counts: torch.Tensor,
    dec0_eligible_counts: torch.Tensor,
    group_seed: int,
    proposal_index: int,
) -> dict[str, torch.Tensor]:
    """Build the one deterministic, pure-token two-fold proposal partition."""

    dec0_ids, dec0_inverse, dec0_counts = torch.unique(
        point_to_dec0[proposal_members],
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    fully_known = dec0_eligible_counts[dec0_ids] == dec0_all_counts[dec0_ids]
    pure_positive = fully_known & (dec0_counts == dec0_all_counts[dec0_ids])
    pure_indices = torch.nonzero(pure_positive, as_tuple=False).flatten()
    pure_dec0_ids = dec0_ids[pure_indices]
    pure_dec0_counts = dec0_counts[pure_indices]
    token_fold = torch.empty(
        pure_dec0_ids.numel(), dtype=torch.long, device=pure_dec0_ids.device
    )
    if pure_dec0_ids.numel():
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed(group_seed, proposal_index, "query-fold")
        )
        permutation = torch.randperm(
            pure_dec0_ids.numel(), generator=generator
        ).to(device=pure_dec0_ids.device)
        token_fold.zero_()
        token_fold[permutation] = torch.arange(
            pure_dec0_ids.numel(), device=pure_dec0_ids.device
        ) % 2
    touched_token_fold = torch.full(
        (dec0_ids.numel(),),
        -1,
        dtype=torch.long,
        device=dec0_ids.device,
    )
    touched_token_fold[pure_indices] = token_fold
    return {
        "dec0_ids": dec0_ids,
        "fully_known": fully_known,
        "pure_dec0_ids": pure_dec0_ids,
        "pure_dec0_counts": pure_dec0_counts,
        "token_fold": token_fold,
        "member_fold": touched_token_fold[dec0_inverse],
    }


def _tokenize_membership_csr(
    offsets: torch.Tensor,
    members: torch.Tensor,
    point_to_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map point-membership CSR to unique token-membership CSR."""

    offsets = offsets.long().flatten()
    members = members.long().flatten()
    if offsets.device != members.device or members.device != point_to_token.device:
        raise ValueError("tokenization membership tensors must share one device")
    if offsets.numel() < 1 or int(offsets[0].item()) != 0:
        raise ValueError("membership offsets must start at zero")
    if bool((offsets < 0).any().item()):
        raise ValueError("membership offsets must be non-negative")
    if int(offsets[-1].item()) != members.numel():
        raise ValueError("membership offsets do not span members")
    if bool((offsets[1:] < offsets[:-1]).any().item()):
        raise ValueError("membership offsets must be non-decreasing")
    if members.numel() and (
        int(members.min().item()) < 0
        or int(members.max().item()) >= point_to_token.numel()
    ):
        raise ValueError("membership point index is outside point_to_token")
    proposal_count = int(offsets.numel()) - 1
    if proposal_count == 0 or members.numel() == 0:
        return (
            torch.zeros(
                proposal_count + 1,
                dtype=torch.long,
                device=point_to_token.device,
            ),
            point_to_token.new_empty((0,), dtype=torch.long),
        )
    sizes = offsets[1:] - offsets[:-1]
    proposal_ids = torch.repeat_interleave(
        torch.arange(
            proposal_count,
            dtype=torch.long,
            device=point_to_token.device,
        ),
        sizes,
    )
    token_ids = point_to_token[members]
    token_capacity = int(point_to_token.max().item()) + 1
    keys = proposal_ids * token_capacity + token_ids
    keys = torch.unique(keys, sorted=True)
    unique_proposals = torch.div(keys, token_capacity, rounding_mode="floor")
    unique_tokens = keys.remainder(token_capacity)
    counts = torch.bincount(unique_proposals, minlength=proposal_count)
    token_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=point_to_token.device),
            counts.cumsum(dim=0),
        ]
    )
    return token_offsets, unique_tokens.long()


def _sample_spatial_token_rows(
    *,
    candidate_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_tokens: torch.Tensor,
    token_xyz: torch.Tensor,
    requested_per_row: int,
    candidate_pool: int,
    excluded_indices: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select row-wise unique nearest tokens after a bounded random pool."""

    rows: list[torch.Tensor] = []
    available: list[int] = []
    requested = int(requested_per_row)
    excluded_mask = _sorted_row_exclusion_mask(
        candidate_tokens, excluded_indices
    )
    for row in range(int(anchor_tokens.numel())):
        values = candidate_tokens[valid_mask[row].bool()].long()
        if excluded_indices.shape[1]:
            values = values[~excluded_mask[row][valid_mask[row].bool()]]
        available.append(int(values.numel()))
        if int(candidate_pool) > 0 and values.numel() > int(candidate_pool):
            draw = torch.randperm(
                int(values.numel()),
                device=values.device,
                generator=generator,
            )[: int(candidate_pool)]
            values = values[draw]
        if values.numel():
            distances = torch.linalg.vector_norm(
                token_xyz[values] - token_xyz[anchor_tokens[row]], dim=-1
            )
            values = values[
                torch.argsort(distances, stable=True)
            ]
        rows.append(values)
    width = min(
        requested,
        min((int(values.numel()) for values in rows), default=0),
    )
    selected = (
        torch.stack([values[:width] for values in rows], dim=0)
        if rows
        else candidate_tokens.new_empty((0, width), dtype=torch.long)
    )
    return selected, torch.tensor(
        available, dtype=torch.long, device=candidate_tokens.device
    )


def build_token_pure_dec0_contrastive_batch(
    batch: ContrastiveTripletBatch,
    point_to_token: torch.Tensor,
    token_xyz: torch.Tensor,
    *,
    max_positive_pairs_per_proposal: int = 64,
) -> tuple[ContrastiveTripletBatch | None, dict[str, Any]]:
    """Rebuild a v2 point batch in exact, pure native-dec0 token space.

    Positive tokens contain only known points from their selected proposal.
    Negative tokens are fully known and contain no point from that proposal.
    Full same-frame, cross-granularity relation membership then removes every
    candidate sharing any teacher mask with either positive endpoint.
    """

    if point_to_token.ndim != 1:
        raise ValueError("point_to_token must be one-dimensional")
    if point_to_token.dtype not in _INTEGRAL_DTYPES:
        raise TypeError("point_to_token must be integral")
    point_to_token = point_to_token.long()
    if point_to_token.numel() == 0 or int(max_positive_pairs_per_proposal) <= 0:
        raise ValueError("point_to_token and max positive-pair budget must be non-empty")
    if token_xyz.ndim != 2 or token_xyz.shape[1] != 3:
        raise ValueError("token_xyz must be [tokens,3]")
    if token_xyz.device != point_to_token.device:
        raise ValueError("point_to_token and token_xyz must share one device")
    token_count = int(token_xyz.shape[0])
    if int(point_to_token.min().item()) < 0 or int(point_to_token.max().item()) >= token_count:
        raise ValueError("point_to_token indexes outside token_xyz")
    offsets, members, eligible = _validate_membership(
        batch, num_points=int(point_to_token.numel())
    )
    if eligible.device != point_to_token.device:
        raise ValueError("point-to-token and membership tensors must share one device")
    pair_to_proposal = batch.pair_proposal_indices
    if pair_to_proposal is None:
        raise ValueError("token_pure_dec0 requires pair_proposal_indices")
    pair_to_proposal = pair_to_proposal.long().flatten()
    proposal_count = int(offsets.numel()) - 1
    if pair_to_proposal.shape != (batch.num_pairs,) or (
        pair_to_proposal.numel()
        and (
            int(pair_to_proposal.min().item()) < 0
            or int(pair_to_proposal.max().item()) >= proposal_count
        )
    ):
        raise ValueError("pair_proposal_indices do not match selected proposal CSR")
    if batch.negative_indices.ndim != 2 or batch.negative_indices.shape[0] != batch.num_pairs:
        raise ValueError("negative_indices do not match positive pair rows")
    if any(
        int(value) < 0
        for value in (
            batch.num_uniform_negatives,
            batch.num_spatial_hard_negatives,
            batch.num_feature_hard_negatives,
            batch.spatial_candidate_pool,
            batch.feature_candidate_pool,
        )
    ):
        raise ValueError("token rebuild counts and candidate pools must be non-negative")
    if (
        int(batch.num_spatial_hard_negatives) > 0
        and int(batch.spatial_candidate_pool) <= 0
    ):
        raise ValueError("token rebuild requires a positive spatial candidate pool")
    if int(batch.num_feature_hard_negatives) > int(batch.feature_candidate_pool):
        raise ValueError(
            "token rebuild feature candidate pool is smaller than hard-negative count"
        )
    deferred_plan = batch.deferred_token_loss
    require_negative_proposal_membership = bool(
        getattr(batch, "require_negative_proposal_membership", False)
    )
    if batch.algorithm_version not in (None, SAMPLER_ALGORITHM_VERSION):
        raise ValueError(
            f"Unsupported sampler algorithm version {batch.algorithm_version!r}"
        )
    if deferred_plan is not None:
        if batch.algorithm_version != SAMPLER_ALGORITHM_VERSION:
            raise ValueError(
                "Deferred token loss requires the V2 sampler algorithm stamp"
            )
        if batch.negative_indices.shape != (batch.num_pairs, 0):
            raise ValueError("Deferred token loss must not materialize point negatives")
        if batch.feature_candidate_indices is not None:
            raise ValueError(
                "Deferred token loss must not materialize point feature candidates"
            )
        expected_widths = (
            int(deferred_plan.effective_uniform_negatives_per_pair),
            int(deferred_plan.effective_spatial_hard_negatives_per_pair),
            int(deferred_plan.effective_feature_candidate_pool_per_pair),
            int(deferred_plan.effective_feature_hard_negatives_per_pair),
        )
        observed_widths = (
            int(batch.num_uniform_negatives),
            int(batch.num_spatial_hard_negatives),
            int(batch.feature_candidate_pool),
            int(batch.num_feature_hard_negatives),
        )
        if observed_widths != expected_widths:
            raise ValueError("Deferred token-loss plan disagrees with batch widths")
        if deferred_plan.algorithm_version != SAMPLER_ALGORITHM_VERSION:
            raise ValueError("Deferred token-loss plan algorithm version drift")
        if bool(
            getattr(deferred_plan, "require_negative_proposal_membership", False)
        ) != require_negative_proposal_membership:
            raise ValueError(
                "Deferred token-loss plan disagrees about negative membership policy"
            )
    token_pair_payload_fields = (
        batch.token_positive_pair_offsets,
        batch.token_positive_pair_indices,
        batch.token_positive_pair_digest,
    )
    token_pair_payload_presence = tuple(
        value is not None for value in token_pair_payload_fields
    )
    if any(token_pair_payload_presence) and not all(token_pair_payload_presence):
        raise ValueError(
            "native-dec0 positive-pair payload fields must be supplied together"
        )
    token_pair_payload = all(token_pair_payload_presence)
    if token_pair_payload:
        assert batch.token_positive_pair_offsets is not None
        assert batch.token_positive_pair_indices is not None
        assert batch.token_positive_pair_digest is not None
        validate_token_positive_pair_payload(
            batch.token_positive_pair_offsets,
            batch.token_positive_pair_indices,
            batch.token_positive_pair_digest,
            proposal_count=proposal_count,
            token_count=token_count,
            algorithm_version=batch.algorithm_version,
            preselection_policy=SAMPLER_PRESELECTION_POLICY,
            proposal_labels=batch.proposal_labels,
            pair_proposal_indices=pair_to_proposal,
        )
    relation_offsets = batch.relation_proposal_offsets
    relation_members = batch.relation_proposal_member_indices
    if relation_offsets is None or relation_members is None:
        raise ValueError("token_pure_dec0 requires full relation-proposal CSR")
    cached_base = _cached_token_rebuild_base(
        point_to_token=point_to_token,
        eligible=eligible,
        relation_offsets=relation_offsets,
        relation_members=relation_members,
        token_count=token_count,
    )
    if cached_base is None:
        relation_token_offsets, relation_token_members = (
            _tokenize_membership_csr(
                relation_offsets,
                relation_members,
                point_to_token,
            )
        )
        relation_index = build_relation_membership_index(
            relation_token_offsets,
            relation_token_members,
            point_capacity=token_count,
        )
        all_counts = torch.bincount(point_to_token, minlength=token_count)
        eligible_counts = torch.bincount(
            point_to_token[eligible], minlength=token_count
        )
        fully_known = (all_counts > 0) & (eligible_counts == all_counts)
        fully_known_tokens = torch.where(fully_known)[0].long()
        cached_base = _TokenRebuildBase(
            relation_token_offsets=relation_token_offsets,
            relation_token_members=relation_token_members,
            relation_index=relation_index,
            all_counts=all_counts,
            eligible_counts=eligible_counts,
            fully_known=fully_known,
            fully_known_tokens=fully_known_tokens,
        )
        _store_token_rebuild_base(
            point_to_token=point_to_token,
            eligible=eligible,
            relation_offsets=relation_offsets,
            relation_members=relation_members,
            token_count=token_count,
            base=cached_base,
        )
    relation_token_offsets = cached_base.relation_token_offsets
    relation_token_members = cached_base.relation_token_members
    relation_index = cached_base.relation_index
    all_counts = cached_base.all_counts
    eligible_counts = cached_base.eligible_counts
    fully_known = cached_base.fully_known
    fully_known_tokens = cached_base.fully_known_tokens
    seed = int(batch.multiscale_seed or 0)
    records: list[dict[str, Any]] = []
    proposal_metrics: list[dict[str, Any]] = []
    for proposal_index in range(proposal_count):
        start = int(offsets[proposal_index].item())
        end = int(offsets[proposal_index + 1].item())
        proposal_points = torch.unique(members[start:end], sorted=True)
        proposal_counts = torch.bincount(
            point_to_token[proposal_points], minlength=token_count
        )
        touched_tokens = torch.where(proposal_counts > 0)[0].long()
        pure_positive_tokens = torch.where(
            fully_known & (proposal_counts == all_counts) & (proposal_counts > 0)
        )[0].long()
        pure_negative_tokens = fully_known_tokens[
            proposal_counts[fully_known_tokens] == 0
        ]
        input_pair_rows = torch.where(pair_to_proposal == proposal_index)[0]
        if token_pair_payload:
            assert batch.token_positive_pair_offsets is not None
            assert batch.token_positive_pair_indices is not None
            pair_start = int(
                batch.token_positive_pair_offsets[proposal_index].item()
            )
            pair_end = int(
                batch.token_positive_pair_offsets[proposal_index + 1].item()
            )
            pairs = batch.token_positive_pair_indices[pair_start:pair_end].long()
            if int(pairs.shape[0]) > int(max_positive_pairs_per_proposal):
                raise ValueError(
                    "native-dec0 positive-pair payload exceeds requested pair budget"
                )
            if pairs.numel() and not bool(
                torch.isin(pairs, pure_positive_tokens).all().item()
            ):
                raise ValueError(
                    "native-dec0 positive-pair payload contains a non-pure token"
                )
            requested_pairs = int(pairs.shape[0])
            positive_pair_source = "native_dec0_payload"
        else:
            requested_pairs = min(
                int(max_positive_pairs_per_proposal), int(input_pair_rows.numel())
            )
            pairs = sample_unique_positive_pairs(
                pure_positive_tokens,
                requested_pairs,
                generator=_token_proposal_stream(
                    root_seed=seed,
                    proposal_index=proposal_index,
                    family="token_positive_pairs",
                    device=point_to_token.device,
                ),
            )
            positive_pair_source = "token_rebuild_resampled"
        base_metrics: dict[str, Any] = {
            "proposal_index": proposal_index,
            "input_point_pairs": int(input_pair_rows.numel()),
            "requested_token_pairs": requested_pairs,
            "positive_pair_source": positive_pair_source,
            "touched_token_count": int(touched_tokens.numel()),
            "fully_known_touched_token_count": int(
                fully_known[touched_tokens].sum().item()
            ),
            "pure_positive_token_count": int(pure_positive_tokens.numel()),
            "pure_negative_token_count": int(pure_negative_tokens.numel()),
            "pure_positive_survival_fraction": (
                int(pure_positive_tokens.numel()) / max(int(touched_tokens.numel()), 1)
            ),
        }
        if pairs.numel() == 0 or pure_negative_tokens.numel() == 0:
            proposal_metrics.append(
                {
                    **base_metrics,
                    "realized_token_pairs": 0,
                    "drop_reason": (
                        "no_native_dec0_positive_pair"
                        if token_pair_payload and pairs.numel() == 0
                        else "fewer_than_two_pure_positive_tokens"
                        if pure_positive_tokens.numel() < 2
                        else "no_pure_negative_token"
                    ),
                }
            )
            continue

        safe_endpoints = co_membership_safe_negative_mask(
            pairs.transpose(0, 1).reshape(-1),
            pure_negative_tokens,
            proposal_offsets=relation_token_offsets,
            proposal_point_indices=relation_token_members,
            require_candidate_membership=require_negative_proposal_membership,
            membership_index=relation_index,
        )
        safe = safe_endpoints[: pairs.shape[0]] & safe_endpoints[pairs.shape[0] :]
        viable = safe.any(dim=1)
        pairs = pairs[viable]
        safe = safe[viable]
        if pairs.shape[0] == 0:
            proposal_metrics.append(
                {
                    **base_metrics,
                    "realized_token_pairs": 0,
                    "drop_reason": "no_co_membership_safe_negative",
                }
            )
            continue

        uniform_indices, _uniform_available = _sample_sorted_unique_negative_rows(
            candidate_tokens=pure_negative_tokens,
            valid_mask=safe,
            requested_per_row=int(batch.num_uniform_negatives),
            excluded_indices=None,
            generator=_token_proposal_stream(
                root_seed=seed,
                proposal_index=proposal_index,
                family="token_uniform",
                device=point_to_token.device,
            ),
        )
        uniform_width = int(uniform_indices.shape[1])
        spatial, spatial_available = _sample_spatial_token_rows(
            candidate_tokens=pure_negative_tokens,
            valid_mask=safe,
            anchor_tokens=pairs[:, 0],
            token_xyz=token_xyz,
            requested_per_row=int(batch.num_spatial_hard_negatives),
            candidate_pool=int(batch.spatial_candidate_pool),
            excluded_indices=uniform_indices,
            generator=_token_proposal_stream(
                root_seed=seed,
                proposal_index=proposal_index,
                family="token_spatial",
                device=point_to_token.device,
            ),
        )
        excluded = torch.cat([uniform_indices, spatial], dim=1)
        feature_indices, _feature_available = _sample_sorted_unique_negative_rows(
            candidate_tokens=pure_negative_tokens,
            valid_mask=safe,
            requested_per_row=int(batch.feature_candidate_pool),
            excluded_indices=excluded,
            generator=_token_proposal_stream(
                root_seed=seed,
                proposal_index=proposal_index,
                family="token_feature",
                device=point_to_token.device,
            ),
        )
        feature_hard = min(
            int(batch.num_feature_hard_negatives),
            int(feature_indices.shape[1]),
        )
        if uniform_width + spatial.shape[1] + feature_hard == 0:
            proposal_metrics.append(
                {
                    **base_metrics,
                    "realized_token_pairs": 0,
                    "drop_reason": "adaptive_token_negative_width_is_zero",
                }
            )
            continue
        label = (
            int(batch.proposal_labels[input_pair_rows[0]].item())
            if input_pair_rows.numel()
            else proposal_index
        )
        records.append(
            {
                "proposal_index": proposal_index,
                "pairs": pairs,
                "uniform": uniform_indices,
                "spatial": spatial,
                "feature": feature_indices,
                "feature_hard": feature_hard,
                "label": label,
                "members": pure_positive_tokens,
                "safe_available": safe.sum(dim=1).long(),
                "spatial_available": spatial_available,
            }
        )
        proposal_metrics.append(
            {
                **base_metrics,
                "realized_token_pairs": int(pairs.shape[0]),
                "safe_negative_tokens_min": int(safe.sum(dim=1).min().item()),
                "drop_reason": None,
            }
        )

    base_metrics: dict[str, Any] = {
        "mode": "token_pure_dec0",
        "algorithm_version": SAMPLER_ALGORITHM_VERSION,
        "seed_scheme": SAMPLER_SEED_SCHEME,
        "root_seed": seed,
        "native_dec0_token_positive_pair_payload": (
            {
                "schema_version": TOKEN_POSITIVE_PAIR_PAYLOAD_SCHEMA,
                "digest": batch.token_positive_pair_digest,
                "pair_count": int(batch.token_positive_pair_indices.shape[0]),
                "proposal_count": proposal_count,
                "consumed": True,
            }
            if token_pair_payload
            else None
        ),
        "input_point_pairs": int(batch.num_pairs),
        "input_proposals": proposal_count,
        "token_count": token_count,
        "fully_known_token_count": int(fully_known.sum().item()),
        "relation_proposal_count": int(relation_token_offsets.numel()) - 1,
        "max_positive_pairs_per_proposal": int(max_positive_pairs_per_proposal),
        "deferred_token_loss": (
            deferred_plan.metadata() if deferred_plan is not None else None
        ),
        "require_negative_proposal_membership": bool(
            require_negative_proposal_membership
        ),
        "proposals": proposal_metrics,
    }
    if not records:
        return None, {
            **base_metrics,
            "retained_proposals": 0,
            "realized_positive_pairs": 0,
            "zero_denominator": True,
        }

    uniform_width = min(int(record["uniform"].shape[1]) for record in records)
    spatial_width = min(int(record["spatial"].shape[1]) for record in records)
    feature_width = min(int(record["feature"].shape[1]) for record in records)
    feature_hard = min(
        int(batch.num_feature_hard_negatives),
        feature_width,
        min(int(record["feature_hard"]) for record in records),
    )
    if uniform_width + spatial_width + feature_hard == 0:
        return None, {
            **base_metrics,
            "retained_proposals": 0,
            "realized_positive_pairs": 0,
            "zero_denominator": True,
        }

    pair_chunks: list[torch.Tensor] = []
    negative_chunks: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    pair_proposals: list[torch.Tensor] = []
    member_chunks: list[torch.Tensor] = []
    member_offsets = [0]
    for occurrence, record in enumerate(records):
        pairs = record["pairs"]
        pair_chunks.append(pairs)
        negative_chunks.append(
            torch.cat(
                [
                    record["uniform"][:, :uniform_width],
                    record["spatial"][:, :spatial_width],
                ],
                dim=1,
            )
        )
        if feature_hard:
            feature_chunks.append(record["feature"][:, :feature_width])
        labels.append(
            torch.full(
                (pairs.shape[0],),
                int(record["label"]),
                dtype=torch.long,
                device=pairs.device,
            )
        )
        pair_proposals.append(
            torch.full(
                (pairs.shape[0],),
                occurrence,
                dtype=torch.long,
                device=pairs.device,
            )
        )
        member_chunks.append(record["members"])
        member_offsets.append(
            member_offsets[-1] + int(record["members"].numel())
        )
    positive_pairs = torch.cat(pair_chunks, dim=0).long()
    negative_indices = torch.cat(negative_chunks, dim=0).long()
    feature_candidates = (
        torch.cat(feature_chunks, dim=0).long() if feature_hard else None
    )
    all_negative_families = [negative_indices]
    if feature_candidates is not None:
        all_negative_families.append(feature_candidates)
    combined_negatives = torch.cat(all_negative_families, dim=1)
    unique_negative_width = torch.tensor(
        [torch.unique(row).numel() for row in combined_negatives],
        dtype=torch.long,
        device=combined_negatives.device,
    )
    if not bool((unique_negative_width == combined_negatives.shape[1]).all().item()):
        raise RuntimeError("Token negative families are not unique and disjoint")
    if bool(torch.isin(positive_pairs, combined_negatives).any().item()):
        # torch.isin here is deliberately conservative across rows; row-wise
        # endpoint validation below provides the exact assertion.
        for row in range(int(positive_pairs.shape[0])):
            if bool(
                torch.isin(positive_pairs[row], combined_negatives[row]).any().item()
            ):
                raise RuntimeError("Token negative row contains a positive endpoint")

    output = ContrastiveTripletBatch(
        positive_pairs=positive_pairs,
        negative_indices=negative_indices,
        proposal_labels=torch.cat(labels, dim=0),
        feature_candidate_indices=feature_candidates,
        num_feature_hard_negatives=feature_hard,
        proposal_member_offsets=torch.tensor(
            member_offsets, dtype=torch.long, device=point_to_token.device
        ),
        proposal_member_indices=torch.cat(member_chunks, dim=0).long(),
        pair_proposal_indices=torch.cat(pair_proposals, dim=0).long(),
        eligible_indices=fully_known_tokens,
        relation_proposal_offsets=relation_token_offsets,
        relation_proposal_member_indices=relation_token_members,
        multiscale_seed=seed,
        num_uniform_negatives=uniform_width,
        num_spatial_hard_negatives=spatial_width,
        spatial_candidate_pool=int(batch.spatial_candidate_pool),
        feature_candidate_pool=feature_width,
        algorithm_version=SAMPLER_ALGORITHM_VERSION,
        require_negative_proposal_membership=(
            require_negative_proposal_membership
        ),
    )
    return output, {
        **base_metrics,
        "retained_proposals": len(records),
        "realized_positive_pairs": int(output.num_pairs),
        "uniform_negatives_per_pair": uniform_width,
        "spatial_hard_negatives_per_pair": spatial_width,
        "feature_candidate_tokens_per_pair": feature_width,
        "feature_hard_negatives_per_endpoint": feature_hard,
        "feature_hard_mining": "symmetric_endpoint_specific",
        "positive_pairs_unique_unordered_per_proposal": True,
        "positive_pairs_collapsed": 0,
        "negative_ids_unique_disjoint_per_row": True,
        "teacher_co_membership_filtered_after_tokenization": True,
        "zero_denominator": False,
    }


def _weighted_soft_mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Balanced soft BCE plus Dice; return None if either class has no mass."""

    positive_mass = (confidence * target).sum()
    negative_mass = (confidence * (1.0 - target)).sum()
    base_metrics = {
        "positive_mass": float(positive_mass.detach()),
        "negative_mass": float(negative_mass.detach()),
        "confidence_sum": float(confidence.sum().detach()),
    }
    if float(positive_mass.detach()) <= eps or float(negative_mass.detach()) <= eps:
        return None, base_metrics

    balance = 0.5 * confidence * (
        target / positive_mass.clamp_min(eps)
        + (1.0 - target) / negative_mass.clamp_min(eps)
    )
    bce = (
        balance
        * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    ).sum()
    probability = logits.sigmoid()
    intersection = (confidence * probability * target).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (
        (confidence * probability).sum() + positive_mass + eps
    )
    loss = 0.5 * (bce + dice)
    with torch.no_grad():
        hard = probability >= 0.5
        hard_target = target >= 0.5
        intersection_hard = (confidence * (hard & hard_target)).sum()
        union_hard = (confidence * (hard | hard_target)).sum()
        iou = intersection_hard / union_hard.clamp_min(eps)
        soft_fraction = ((target > eps) & (target < 1.0 - eps)).float().mean()
    return loss, {
        **base_metrics,
        "bce_loss": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "hard_iou": float(iou.detach()),
        "soft_target_fraction": float(soft_fraction.detach()),
        "confidence_mean": float(confidence.mean().detach()),
        "logit_mean": float(logits.detach().mean()),
    }


def _weighted_soft_mask_loss_batched(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate a rectangular batch of balanced soft mask losses.

    Every row is one proposal fold and columns are the shared eligible-token
    universe.  Invalid columns carry zero confidence, so the formulas match
    ``_weighted_soft_mask_loss`` while one BCE/Dice graph replaces dozens of
    tiny per-fold graphs.  ``valid_mask`` remains explicit because a token can
    have no supervised mass for one held-out fold.
    """

    if logits.ndim != 2 or target.shape != logits.shape or confidence.shape != logits.shape:
        raise ValueError("Batched mask-loss tensors must share shape [folds,tokens]")
    if valid_mask.shape != logits.shape:
        raise ValueError("Batched mask-loss valid_mask must match logits")
    valid = valid_mask.bool()
    confidence = confidence * valid
    positive_mass = (confidence * target).sum(dim=1)
    negative_mass = (confidence * (1.0 - target)).sum(dim=1)
    fold_valid = (positive_mass > float(eps)) & (negative_mass > float(eps))
    safe_positive = positive_mass.clamp_min(float(eps))
    safe_negative = negative_mass.clamp_min(float(eps))
    balance = 0.5 * confidence * (
        target / safe_positive[:, None]
        + (1.0 - target) / safe_negative[:, None]
    )
    bce = (
        balance
        * F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
    ).sum(dim=1)
    probability = logits.sigmoid()
    intersection = (confidence * probability * target).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + float(eps)) / (
        (confidence * probability).sum(dim=1) + positive_mass + float(eps)
    )
    losses = 0.5 * (bce + dice)
    hard = probability >= 0.5
    hard_target = target >= 0.5
    intersection_hard = (
        confidence * (hard & hard_target)
    ).sum(dim=1)
    union_hard = (confidence * (hard | hard_target)).sum(dim=1)
    iou = intersection_hard / union_hard.clamp_min(float(eps))
    valid_count = valid.sum(dim=1).clamp_min(1).to(logits.dtype)
    soft_fraction = (
        ((target > float(eps)) & (target < 1.0 - float(eps)) & valid)
        .sum(dim=1)
        .to(logits.dtype)
        / valid_count
    )
    metrics = {
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "confidence_sum": confidence.sum(dim=1),
        "bce_loss": bce,
        "dice_loss": dice,
        "hard_iou": iou,
        "soft_target_fraction": soft_fraction,
        "confidence_mean": confidence.sum(dim=1) / valid_count,
        "logit_mean": (logits * valid).sum(dim=1) / valid_count,
    }
    return losses, fold_valid, metrics


def _g02_post_voxel_representability_filter(
    proposal_routes: Mapping[str, Sequence[int]],
    route_metrics: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Filter unrepresentable g02 routes without inventing replacement labels.

    The raw frame sampler's selected proposals remain the accounting universe.
    At the post-voxel boundary, a proposal is trainable only when its exact
    membership has at least two pure dec0 query tokens.  The current batch
    contract does not identify unselected same-frame proposals independently
    of the cross-cell relation CSR, so silently substituting one would change
    the g02 source-cell semantics.  We therefore use the smallest safe policy:
    retain representable selected proposals, report every filtered raw index,
    and fail closed when no selected candidate remains.
    """

    rows = route_metrics.get("proposals")
    if not isinstance(rows, Sequence):
        raise ValueError("g02 route metrics must contain proposal rows")
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or "proposal_index" not in row:
            raise ValueError("Malformed g02 route proposal metrics")
        proposal_index = int(row["proposal_index"])
        if proposal_index in by_index:
            raise ValueError("Duplicate g02 route proposal metric")
        by_index[proposal_index] = row

    filtered: dict[str, tuple[int, ...]] = {}
    filtered_indices: list[int] = []
    effective_indices: list[int] = []
    raw_indices: list[int] = []
    for stage, indices in proposal_routes.items():
        stage_effective: list[int] = []
        for value in indices:
            proposal_index = int(value)
            if proposal_index not in by_index:
                raise ValueError(
                    "g02 route index is absent from proposal metrics"
                )
            raw_indices.append(proposal_index)
            if bool(by_index[proposal_index].get("post_voxel_representable", False)):
                stage_effective.append(proposal_index)
                effective_indices.append(proposal_index)
            else:
                filtered_indices.append(proposal_index)
        filtered[stage] = tuple(stage_effective)

    if len(raw_indices) != len(set(raw_indices)):
        raise ValueError("g02 proposal routes must assign each proposal once")
    if set(raw_indices) != set(by_index):
        raise ValueError("g02 proposal metrics and routes disagree")
    filtered_indices = sorted(filtered_indices)
    effective_indices = sorted(effective_indices)
    return filtered, {
        "policy": (
            "drop_selected_g02_proposals_with_fewer_than_two_pure_"
            "dec0_query_tokens"
        ),
        "minimum_pure_dec0_query_tokens": 2,
        "raw_proposal_count": len(raw_indices),
        "effective_proposal_count": len(effective_indices),
        "filtered_proposal_count": len(filtered_indices),
        "raw_proposal_indices": sorted(raw_indices),
        "effective_proposal_indices": effective_indices,
        "filtered_proposal_indices": filtered_indices,
        "replacement_count": 0,
        "replacement_available": False,
        "failed_closed": not bool(effective_indices),
    }


@dataclass(frozen=True)
class _PreparedStageMembership:
    """Integer membership/count tensors shared by one group's routed stages."""

    eligible_tokens: torch.Tensor
    eligible_counts: torch.Tensor
    all_counts: torch.Tensor
    full_counts_by_proposal: tuple[torch.Tensor, ...]
    pure_fold_counts_by_proposal: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _PreparedHierarchyScene:
    """Map/count tensors independent of a frame group's proposal CSR."""

    point_to_dec0: torch.Tensor
    dec0_all_counts: torch.Tensor
    stage_all_counts: Mapping[str, torch.Tensor]
    stage_token_counts: Mapping[str, int]


@dataclass(frozen=True)
class _PreparedHierarchyMembership:
    """Reusable token/query bookkeeping for one frame-local group."""

    offsets: torch.Tensor
    members: torch.Tensor
    eligible: torch.Tensor
    point_to_dec0: torch.Tensor
    dec0_all_counts: torch.Tensor
    dec0_eligible_counts: torch.Tensor
    proposal_members: tuple[torch.Tensor, ...]
    partitions: tuple[dict[str, torch.Tensor], ...]
    stages: Mapping[str, _PreparedStageMembership]


def _prepare_hierarchy_scene(
    *,
    point_maps: Mapping[str, torch.Tensor],
    dec0_token_count: int,
    stage_token_counts: Mapping[str, int],
) -> _PreparedHierarchyScene:
    """Build map-derived counts once for all frame groups in one scene."""

    point_to_dec0 = point_maps["dec0"]
    num_points = int(point_to_dec0.numel())
    if point_to_dec0.shape != (num_points,):
        raise ValueError("Point map for dec0 has the wrong shape")
    if point_to_dec0.numel() and int(point_to_dec0.max().item()) >= int(
        dec0_token_count
    ):
        raise ValueError("Point map for dec0 indexes outside query features")
    dec0_all_counts = torch.bincount(
        point_to_dec0, minlength=int(dec0_token_count)
    )
    resolved_stage_counts = {
        stage: int(stage_token_counts[stage])
        for stage in HIERARCHY_COARSE_TO_FINE
        if stage != "dec0"
    }
    stage_all_counts: dict[str, torch.Tensor] = {}
    for stage, stage_token_count in resolved_stage_counts.items():
        point_to_stage = point_maps[stage]
        if point_to_stage.shape != (num_points,):
            raise ValueError(f"Point map for {stage} has the wrong shape")
        if point_to_stage.numel() and int(point_to_stage.max().item()) >= stage_token_count:
            raise ValueError(f"Point map for {stage} indexes outside stage features")
        stage_all_counts[stage] = torch.bincount(
            point_to_stage, minlength=stage_token_count
        ).float()
    return _PreparedHierarchyScene(
        point_to_dec0=point_to_dec0,
        dec0_all_counts=dec0_all_counts,
        stage_all_counts=stage_all_counts,
        stage_token_counts=resolved_stage_counts,
    )


def _prepare_hierarchy_membership(
    *,
    point_maps: Mapping[str, torch.Tensor],
    batch: ContrastiveTripletBatch,
    group_seed: int,
    dec0_token_count: int,
    stage_token_counts: Mapping[str, int] | None = None,
    stage_names: Sequence[str] | None = None,
    scene: _PreparedHierarchyScene | None = None,
) -> _PreparedHierarchyMembership:
    """Prepare proposal/query counts once before routed stage evaluation.

    The previous implementation rebuilt the same proposal uniques, dec0
    purity partitions, and point-to-stage count vectors once per routed stage.
    These tensors are integer-only and independent of learned features, so
    preparing them once preserves all sampling/denominator semantics while
    removing repeated full-scene scans from both routing and mask loss.
    """

    num_points = int(point_maps["dec0"].numel())
    offsets, members, eligible = _validate_membership(
        batch, num_points=num_points
    )
    point_to_dec0 = point_maps["dec0"]
    if point_to_dec0.shape != (num_points,):
        raise ValueError("Point map for dec0 has the wrong shape")
    if point_to_dec0.numel() and int(point_to_dec0.max().item()) >= int(
        dec0_token_count
    ):
        raise ValueError("Point map for dec0 indexes outside query features")

    prepared_stages = (
        tuple(stage for stage in HIERARCHY_COARSE_TO_FINE if stage != "dec0")
        if stage_names is None
        else tuple(str(stage) for stage in stage_names)
    )
    if len(set(prepared_stages)) != len(prepared_stages):
        raise ValueError("stage_names must not contain duplicates")
    if any(
        stage not in HIERARCHY_COARSE_TO_FINE or stage == "dec0"
        for stage in prepared_stages
    ):
        raise ValueError("stage_names must identify native non-dec0 stages")
    if scene is None:
        if stage_token_counts is None:
            stage_token_counts = {
                stage: (
                    int(point_maps[stage].max().item()) + 1
                    if point_maps[stage].numel()
                    else 0
                )
                for stage in prepared_stages
            }
        resolved_stage_counts = {
            stage: int(stage_token_counts[stage]) for stage in prepared_stages
        }
        scene = _prepare_hierarchy_scene(
            point_maps=point_maps,
            dec0_token_count=dec0_token_count,
            stage_token_counts=resolved_stage_counts,
        )
    else:
        if scene.point_to_dec0 is not point_to_dec0:
            raise ValueError("Prepared hierarchy scene does not match dec0 map")
        resolved_stage_counts = {
            stage: int(scene.stage_token_counts[stage])
            for stage in prepared_stages
        }

    dec0_all_counts = scene.dec0_all_counts
    dec0_eligible_counts = torch.bincount(
        point_to_dec0[eligible], minlength=int(dec0_token_count)
    )
    proposal_members: list[torch.Tensor] = []
    partitions: list[dict[str, torch.Tensor]] = []
    proposal_count = int(offsets.numel()) - 1
    for proposal_index in range(proposal_count):
        start = int(offsets[proposal_index].item())
        end = int(offsets[proposal_index + 1].item())
        current_members = torch.unique(members[start:end], sorted=True)
        proposal_members.append(current_members)
        partitions.append(
            _pure_dec0_query_partition(
                proposal_members=current_members,
                point_to_dec0=point_to_dec0,
                dec0_all_counts=dec0_all_counts,
                dec0_eligible_counts=dec0_eligible_counts,
                group_seed=group_seed,
                proposal_index=proposal_index,
            )
        )

    stage_membership: dict[str, _PreparedStageMembership] = {}
    for stage in prepared_stages:
        point_to_stage = point_maps[stage]
        stage_token_count = int(resolved_stage_counts[stage])
        if point_to_stage.shape != (num_points,):
            raise ValueError(f"Point map for {stage} has the wrong shape")
        if point_to_stage.numel() and int(point_to_stage.max().item()) >= stage_token_count:
            raise ValueError(f"Point map for {stage} indexes outside stage features")
        eligible_tokens, eligible_counts = torch.unique(
            point_to_stage[eligible], sorted=True, return_counts=True
        )
        all_counts = scene.stage_all_counts[stage]
        full_counts_by_proposal: list[torch.Tensor] = []
        pure_fold_counts_by_proposal: list[torch.Tensor] = []
        for current_members, partition in zip(
            proposal_members, partitions, strict=True
        ):
            full_counts_by_proposal.append(
                _counts_at_sorted_tokens(
                    current_members, point_to_stage, eligible_tokens
                )
            )
            pure_mask = partition["member_fold"] >= 0
            pure_stage_tokens = point_to_stage[current_members[pure_mask]]
            pure_folds = partition["member_fold"][pure_mask]
            fold_keys = pure_folds * stage_token_count + pure_stage_tokens
            fold_counts = torch.bincount(
                fold_keys, minlength=2 * stage_token_count
            ).reshape(2, stage_token_count)
            pure_fold_counts_by_proposal.append(
                fold_counts[:, eligible_tokens].float()
            )
        stage_membership[stage] = _PreparedStageMembership(
            eligible_tokens=eligible_tokens,
            eligible_counts=eligible_counts,
            all_counts=all_counts,
            full_counts_by_proposal=tuple(full_counts_by_proposal),
            pure_fold_counts_by_proposal=tuple(pure_fold_counts_by_proposal),
        )
    return _PreparedHierarchyMembership(
        offsets=offsets,
        members=members,
        eligible=eligible,
        point_to_dec0=point_to_dec0,
        dec0_all_counts=dec0_all_counts,
        dec0_eligible_counts=dec0_eligible_counts,
        proposal_members=tuple(proposal_members),
        partitions=tuple(partitions),
        stages=stage_membership,
    )


class ProposalConditionedHierarchyLoss(nn.Module):
    """Two-fold proposal queries predicting soft occupancy at routed stages."""

    def __init__(
        self,
        *,
        stage_channels: Mapping[str, int] = HIERARCHY_STAGE_CHANNELS,
        embedding_dim: int = 128,
        temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
        query_projection_policy: str = "shared_dec0",
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        channels = {name: int(value) for name, value in stage_channels.items()}
        if set(channels) != set(HIERARCHY_FINE_TO_COARSE):
            raise ValueError("stage_channels must define enc4, dec3, dec2, dec1, dec0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0 < min_temperature <= temperature <= max_temperature:
            raise ValueError("Invalid hierarchy mask temperature bounds")
        query_projection_policy = str(query_projection_policy)
        if query_projection_policy not in HIERARCHY_QUERY_PROJECTION_POLICIES:
            raise ValueError(
                "Unknown hierarchy query-projection policy: "
                f"{query_projection_policy!r}"
            )
        self.stage_channels = channels
        self.embedding_dim = int(embedding_dim)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.query_projection_policy = query_projection_policy
        self.eps = float(eps)
        if self.query_projection_policy == "shared_dec0":
            # Preserve the C3--C7 module names/state-dict layout exactly.
            self.query_norm = nn.LayerNorm(channels["dec0"])
            self.query_projection = nn.Linear(
                channels["dec0"], self.embedding_dim, bias=False
            )
            self.stage_query_norms = None
            self.stage_query_projections = None
        else:
            # Do not instantiate an unused shared head: V2 uses DDP with
            # find_unused_parameters=False.  ``_zero_touch`` below still
            # connects a dynamically unrouted stage head on every rank.
            self.query_norm = None
            self.query_projection = None
            self.stage_query_norms = nn.ModuleDict(
                {
                    stage: nn.LayerNorm(channels["dec0"])
                    for stage in HIERARCHY_COARSE_TO_FINE
                    if stage != "dec0"
                }
            )
            self.stage_query_projections = nn.ModuleDict(
                {
                    stage: nn.Linear(
                        channels["dec0"], self.embedding_dim, bias=False
                    )
                    for stage in HIERARCHY_COARSE_TO_FINE
                    if stage != "dec0"
                }
            )
        self.key_norms = nn.ModuleDict(
            {
                stage: nn.LayerNorm(channels[stage])
                for stage in HIERARCHY_COARSE_TO_FINE
                if stage != "dec0"
            }
        )
        self.key_projections = nn.ModuleDict(
            {
                stage: nn.Linear(channels[stage], self.embedding_dim, bias=False)
                for stage in HIERARCHY_COARSE_TO_FINE
                if stage != "dec0"
            }
        )
        initial = math.log(float(temperature))
        self.log_temperatures = nn.ParameterDict(
            {
                stage: nn.Parameter(torch.tensor(initial, dtype=torch.float32))
                for stage in HIERARCHY_COARSE_TO_FINE
                if stage != "dec0"
            }
        )

    def temperature(self, stage: str) -> torch.Tensor:
        return self.log_temperatures[stage].exp().clamp(
            self.min_temperature, self.max_temperature
        )

    def _queries_by_stage(
        self, dec0_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project detached dec0 query tokens under the selected policy.

        Keeping dec0 detached is intentional: the hierarchy objective must
        train routed native stages rather than turn dec0 into a shortcut.  In
        the per-stage policy only the disposable query-head coordinates change;
        folds and token targets remain bit-for-bit the same as shared_dec0.
        """

        detached = dec0_features.detach()
        routed_stages = tuple(
            stage for stage in HIERARCHY_COARSE_TO_FINE if stage != "dec0"
        )
        if self.query_projection_policy == "shared_dec0":
            assert self.query_norm is not None
            assert self.query_projection is not None
            query = F.normalize(
                self.query_projection(self.query_norm(detached)),
                dim=-1,
                eps=self.eps,
            )
            return {stage: query for stage in routed_stages}

        assert self.stage_query_norms is not None
        assert self.stage_query_projections is not None
        return {
            stage: F.normalize(
                self.stage_query_projections[stage](
                    self.stage_query_norms[stage](detached)
                ),
                dim=-1,
                eps=self.eps,
            )
            for stage in routed_stages
        }

    def _one_group_stage(
        self,
        *,
        query_tokens: torch.Tensor,
        stage_keys: torch.Tensor,
        point_maps: Mapping[str, torch.Tensor],
        batch: ContrastiveTripletBatch,
        stage: str,
        group_seed: int,
        proposal_indices: Sequence[int] | None = None,
        prepared: _PreparedHierarchyMembership | None = None,
    ) -> dict[str, Any]:
        if prepared is None:
            prepared = _prepare_hierarchy_membership(
                point_maps=point_maps,
                batch=batch,
                group_seed=group_seed,
                dec0_token_count=int(query_tokens.shape[0]),
                stage_token_counts={
                    name: (
                        int(stage_keys.shape[0])
                        if name == stage
                        else (
                            int(point_maps[name].max().item()) + 1
                            if point_maps[name].numel()
                            else 0
                        )
                    )
                    for name in HIERARCHY_FINE_TO_COARSE
                },
                stage_names=(stage,),
            )
        stage_prepared = prepared.stages[stage]
        eligible_tokens = stage_prepared.eligible_tokens
        eligible_counts = stage_prepared.eligible_counts
        all_counts = stage_prepared.all_counts
        keys = stage_keys[eligible_tokens]
        available_proposals = len(prepared.proposal_members)
        if proposal_indices is None:
            selected_proposals = tuple(range(available_proposals))
        else:
            selected_proposals = tuple(int(index) for index in proposal_indices)
            if len(set(selected_proposals)) != len(selected_proposals):
                raise ValueError("proposal_indices must not contain duplicates")
            if selected_proposals and (
                min(selected_proposals) < 0
                or max(selected_proposals) >= available_proposals
            ):
                raise IndexError("proposal_indices contain an out-of-range proposal")
        attempted_proposals = len(selected_proposals)
        attempted_folds = 2 * attempted_proposals
        loss_sum = _zero_touch(stage_keys, self)
        valid_folds = 0
        valid_proposals = 0
        metric_sums: dict[str, float] = {}
        target_token_count = 0
        query_token_count = 0
        query_tokens_touched = 0
        query_tokens_fully_known = 0
        query_tokens_pure_positive = 0
        invalid_query_proposals = 0
        fold_support_indices: list[torch.Tensor] = []
        fold_support_weights: list[torch.Tensor] = []
        fold_targets: list[torch.Tensor] = []
        fold_confidences: list[torch.Tensor] = []
        fold_valid_masks: list[torch.Tensor] = []
        fold_records: list[tuple[int, int]] = []
        temperature = self.temperature(stage).to(
            device=keys.device, dtype=keys.dtype
        )

        for proposal_index in selected_proposals:
            proposal_members = prepared.proposal_members[proposal_index]
            partition = prepared.partitions[proposal_index]
            dec0_ids = partition["dec0_ids"]
            fully_known = partition["fully_known"]
            pure_dec0_ids = partition["pure_dec0_ids"]
            pure_dec0_counts = partition["pure_dec0_counts"]
            token_fold = partition["token_fold"]
            member_fold = partition["member_fold"]
            query_tokens_touched += int(dec0_ids.numel())
            query_tokens_fully_known += int(fully_known.sum().item())
            query_tokens_pure_positive += int(pure_dec0_ids.numel())
            if pure_dec0_ids.numel() < 2:
                invalid_query_proposals += 1
                continue
            full_counts = stage_prepared.full_counts_by_proposal[proposal_index]
            negative_counts = eligible_counts.float() - full_counts
            if bool((negative_counts < 0).any().item()):
                raise RuntimeError("Proposal membership exceeds eligible support")

            for fold in (0, 1):
                support_token_mask = token_fold == fold
                heldout_member_mask = member_fold == (1 - fold)
                if not bool(support_token_mask.any().item()) or not bool(
                    heldout_member_mask.any().item()
                ):
                    continue
                weights = pure_dec0_counts[support_token_mask].to(query_tokens.dtype)
                heldout_counts = stage_prepared.pure_fold_counts_by_proposal[
                    proposal_index
                ][1 - fold]
                supervised_counts = heldout_counts + negative_counts
                valid = supervised_counts > 0
                if not bool(valid.any().item()):
                    continue
                target = torch.where(
                    valid,
                    heldout_counts / supervised_counts.clamp_min(1.0),
                    torch.zeros_like(supervised_counts),
                )
                confidence = torch.where(
                    valid,
                    supervised_counts
                    / all_counts[eligible_tokens].clamp_min(1.0),
                    torch.zeros_like(supervised_counts),
                )
                fold_support_indices.append(
                    pure_dec0_ids[support_token_mask]
                )
                fold_support_weights.append(weights)
                fold_targets.append(target)
                fold_confidences.append(confidence)
                fold_valid_masks.append(valid)
                fold_records.append(
                    (proposal_index, int(support_token_mask.sum().item()))
                )
        if fold_support_indices:
            max_support = max(
                int(indices.numel()) for indices in fold_support_indices
            )
            support_indices = torch.stack(
                [
                    F.pad(indices, (0, max_support - int(indices.numel())))
                    for indices in fold_support_indices
                ],
                dim=0,
            )
            support_weights = torch.stack(
                [
                    F.pad(weights, (0, max_support - int(weights.numel())))
                    for weights in fold_support_weights
                ],
                dim=0,
            ).to(query_tokens.dtype)
            queries = query_tokens[support_indices]
            queries = (queries * support_weights.unsqueeze(-1)).sum(dim=1)
            queries = queries / support_weights.sum(dim=1).clamp_min(1.0).unsqueeze(
                -1
            )
            queries = F.normalize(queries, dim=1, eps=self.eps)
            targets = torch.stack(fold_targets, dim=0)
            confidences = torch.stack(fold_confidences, dim=0)
            valid_masks = torch.stack(fold_valid_masks, dim=0)
            logits = torch.mm(keys, queries.transpose(0, 1)).transpose(0, 1)
            logits = logits / temperature
            fold_losses, fold_is_valid, fold_metrics = (
                _weighted_soft_mask_loss_batched(
                    logits,
                    targets,
                    confidences,
                    valid_masks,
                    eps=self.eps,
                )
            )
            valid_indices = torch.nonzero(
                fold_is_valid, as_tuple=False
            ).flatten().tolist()
            if valid_indices:
                loss_sum = loss_sum + fold_losses[fold_is_valid].sum()
            valid_proposal_indices: set[int] = set()
            metric_names = tuple(fold_metrics)
            for index in valid_indices:
                proposal_index, support_count = fold_records[index]
                valid_proposal_indices.add(proposal_index)
                valid_folds += 1
                target_token_count += int(valid_masks[index].sum().item())
                query_token_count += support_count
                for key in metric_names:
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(
                        fold_metrics[key][index].detach()
                    )
            valid_proposals = len(valid_proposal_indices)

        # Frame groups are aggregated later with this explicit realized-fold
        # denominator.  Empty records retain a differentiable zero so every
        # routed head remains safe under DDP's unused-parameter checks.
        loss = loss_sum / max(valid_folds, 1)
        metrics: dict[str, float | int] = {
            "loss": float(loss.detach()),
            "loss_numerator": float(loss_sum.detach()),
            "loss_denominator_valid_folds": valid_folds,
            "proposals_attempted": attempted_proposals,
            "proposals_valid": valid_proposals,
            "proposal_valid_fraction": valid_proposals / max(attempted_proposals, 1),
            "folds_attempted": attempted_folds,
            "folds_valid": valid_folds,
            "fold_valid_fraction": valid_folds / max(attempted_folds, 1),
            "target_token_count": target_token_count,
            "query_token_count": query_token_count,
            "query_tokens_touched": query_tokens_touched,
            "query_tokens_fully_known": query_tokens_fully_known,
            "query_tokens_pure_positive": query_tokens_pure_positive,
            "query_pure_survival_fraction": (
                query_tokens_pure_positive / max(query_tokens_touched, 1)
            ),
            "proposals_invalid_fewer_than_two_pure_query_tokens": (
                invalid_query_proposals
            ),
            "eligible_token_count": int(eligible_tokens.numel()),
            "temperature": float(self.temperature(stage).detach()),
        }
        for key, value in metric_sums.items():
            metrics[key] = value / max(valid_folds, 1)
        return {
            "loss_total": loss,
            "loss_numerator": loss_sum,
            "loss_denominator": valid_folds,
            "metrics": metrics,
        }

    def _g02_proposal_routes(
        self,
        *,
        point_maps: Mapping[str, torch.Tensor],
        batch: ContrastiveTripletBatch,
        group_seed: int,
        prepared: _PreparedHierarchyMembership | None = None,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        """Choose dec2 only when both deterministic folds have coarse support."""

        if prepared is None:
            prepared = _prepare_hierarchy_membership(
                point_maps=point_maps,
                batch=batch,
                group_seed=group_seed,
                dec0_token_count=(
                    int(point_maps["dec0"].max().item()) + 1
                    if point_maps["dec0"].numel()
                    else 0
                ),
                stage_names=("dec2",),
            )
        dec2_prepared = prepared.stages["dec2"]
        eligible_tokens = dec2_prepared.eligible_tokens
        eligible_counts = dec2_prepared.eligible_counts
        routed: dict[str, list[int]] = {"dec1": [], "dec2": []}
        support: list[dict[str, Any]] = []
        for proposal_index, partition in enumerate(prepared.partitions):
            pure_fold_counts = dec2_prepared.pure_fold_counts_by_proposal[
                proposal_index
            ]
            positive_known_tokens_by_fold = [
                int((pure_fold_counts[fold] > 0).sum().item())
                for fold in (0, 1)
            ]
            positive_counts = dec2_prepared.full_counts_by_proposal[
                proposal_index
            ]
            negative_counts = eligible_counts.float() - positive_counts
            negative_known_mass = float(negative_counts.sum().item())
            stage = (
                "dec2"
                if min(positive_known_tokens_by_fold, default=0) >= 2
                and negative_known_mass > self.eps
                else "dec1"
            )
            routed[stage].append(proposal_index)
            support.append(
                {
                    "proposal_index": proposal_index,
                    "stage": stage,
                    "pure_query_dec0_tokens": int(
                        partition["pure_dec0_ids"].numel()
                    ),
                    "touched_dec0_tokens": int(partition["dec0_ids"].numel()),
                    "fully_known_dec0_tokens": int(
                        partition["fully_known"].sum().item()
                    ),
                    "positive_known_dec2_tokens_by_fold": (
                        positive_known_tokens_by_fold
                    ),
                    "minimum_positive_known_dec2_tokens_across_folds": min(
                        positive_known_tokens_by_fold, default=0
                    ),
                    "negative_known_dec2_mass": negative_known_mass,
                    # This is the post-voxel representability predicate used
                    # by the conservative g02 filter below.  It is derived
                    # only from exact proposal membership and point->dec0
                    # remapping; mixed/unknown tokens are never substituted.
                    "post_voxel_representable": bool(
                        partition["pure_dec0_ids"].numel() >= 2
                    ),
                    "post_voxel_filter_reason": (
                        None
                        if partition["pure_dec0_ids"].numel() >= 2
                        else "fewer_than_two_pure_dec0_query_tokens"
                    ),
                }
            )
        return {stage: tuple(indices) for stage, indices in routed.items()}, {
            "policy": (
                "dec2_if_each_deterministic_query_fold_has_at_least_2_"
                "positive_known_dec2_tokens_and_negative_known_dec2_mass>0_"
                "else_dec1"
            ),
            "minimum_positive_known_dec2_tokens_per_fold": 2,
            "requires_positive_known_negative_mass": True,
            "routed_dec1_proposals": len(routed["dec1"]),
            "routed_dec2_proposals": len(routed["dec2"]),
            "proposals": support,
        }
    def forward(
        self,
        *,
        stage_features: Mapping[str, torch.Tensor],
        point_maps: Mapping[str, torch.Tensor],
        groups: Sequence[MaskSupervisionGroup],
    ) -> dict[str, Any]:
        if not groups:
            raise ValueError("At least one hierarchy mask supervision group is required")
        if set(stage_features) != set(HIERARCHY_FINE_TO_COARSE):
            raise ValueError("stage_features must contain all five native LitePT stages")
        if set(point_maps) != set(HIERARCHY_FINE_TO_COARSE):
            raise ValueError("point_maps must contain all five native LitePT stages")
        for stage, features in stage_features.items():
            if features.ndim != 2 or features.shape[1] != self.stage_channels[stage]:
                raise ValueError(
                    f"{stage} features must be [N,{self.stage_channels[stage]}], "
                    f"got {tuple(features.shape)}"
                )

        # Stop the query shortcut at dec0: the mask objective trains the query
        # heads, key heads, and routed native stages, but cannot improve dec0
        # merely by pulling it toward its keys.  C8 may use a separate
        # disposable query coordinate system for each target stage.
        query_tokens_by_stage = self._queries_by_stage(stage_features["dec0"])
        stage_keys = {
            stage: F.normalize(
                self.key_projections[stage](self.key_norms[stage](stage_features[stage])),
                dim=-1,
                eps=self.eps,
            )
            for stage in HIERARCHY_COARSE_TO_FINE
            if stage != "dec0"
        }

        granularities = [_granularity(group.cell) for group in groups]
        input_granularities = tuple(dict.fromkeys(granularities))
        edge_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
        route_metrics: dict[str, list[dict[str, Any]]] = {
            granularity: [] for granularity in input_granularities
        }
        scene = _prepare_hierarchy_scene(
            point_maps=point_maps,
            dec0_token_count=int(stage_features["dec0"].shape[0]),
            stage_token_counts={
                stage: int(stage_features[stage].shape[0])
                for stage in HIERARCHY_FINE_TO_COARSE
                if stage != "dec0"
            },
        )
        for group_index, (group, granularity) in enumerate(
            zip(groups, granularities, strict=True)
        ):
            group_seed = int(group.batch.multiscale_seed or 0)
            routed_group_seed = _stable_seed(
                group_seed, group_index, group.group_id or "", granularity
            )
            prepared = _prepare_hierarchy_membership(
                point_maps=point_maps,
                batch=group.batch,
                group_seed=routed_group_seed,
                dec0_token_count=int(stage_features["dec0"].shape[0]),
                stage_token_counts={
                    stage: int(stage_features[stage].shape[0])
                    for stage in HIERARCHY_FINE_TO_COARSE
                },
                stage_names=MASK_ROUTE_STAGES[granularity],
                scene=scene,
            )
            if granularity == "g02":
                raw_proposal_routes, group_route_metrics = self._g02_proposal_routes(
                    point_maps=point_maps,
                    batch=group.batch,
                    group_seed=routed_group_seed,
                    prepared=prepared,
                )
                proposal_routes, representability_metrics = (
                    _g02_post_voxel_representability_filter(
                        raw_proposal_routes, group_route_metrics
                    )
                )
                group_route_metrics = {
                    **group_route_metrics,
                    "post_voxel_representability": representability_metrics,
                }
                route_metrics[granularity].append(group_route_metrics)
            else:
                proposal_count = int(
                    group.batch.proposal_member_offsets.numel() - 1
                )
                proposal_routes = {
                    stage: tuple(range(proposal_count))
                    for stage in MASK_ROUTE_STAGES[granularity]
                }
                raw_proposal_routes = proposal_routes
            for stage in MASK_ROUTE_STAGES[granularity]:
                record = self._one_group_stage(
                    query_tokens=query_tokens_by_stage[stage],
                    stage_keys=stage_keys[stage],
                    point_maps=point_maps,
                    batch=group.batch,
                    stage=stage,
                    group_seed=routed_group_seed,
                    proposal_indices=proposal_routes[stage],
                    prepared=prepared,
                )
                if granularity == "g02":
                    # ``proposals_*`` remain raw selected-route accounting for
                    # the cumulative survival gate.  Effective counters expose
                    # the post-voxel filtered route actually sent through the
                    # loss, so filtering cannot masquerade as raw survival.
                    record_metrics = record["metrics"]
                    effective_attempted = int(
                        record_metrics.get("proposals_attempted", 0)
                    )
                    effective_valid = int(record_metrics.get("proposals_valid", 0))
                    raw_attempted = len(raw_proposal_routes[stage])
                    filtered_rows = [
                        row
                        for row in group_route_metrics["proposals"]
                        if row["stage"] == stage
                        and not bool(row.get("post_voxel_representable", False))
                    ]
                    # Keep post-voxel purity diagnostics for filtered raw
                    # proposals even though they no longer enter the loss
                    # loop.  This preserves the raw survival explanation.
                    record_metrics["query_tokens_touched"] = int(
                        record_metrics.get("query_tokens_touched", 0)
                    ) + sum(
                        int(row.get("touched_dec0_tokens", 0))
                        for row in filtered_rows
                    )
                    record_metrics["query_tokens_fully_known"] = int(
                        record_metrics.get("query_tokens_fully_known", 0)
                    ) + sum(
                        int(row.get("fully_known_dec0_tokens", 0))
                        for row in filtered_rows
                    )
                    record_metrics["query_tokens_pure_positive"] = int(
                        record_metrics.get("query_tokens_pure_positive", 0)
                    ) + sum(
                        int(row.get("pure_query_dec0_tokens", 0))
                        for row in filtered_rows
                    )
                    # The filtered rows are restored to the raw diagnostic
                    # totals above; recompute the ratio from those totals so
                    # it cannot accidentally describe only the effective
                    # (post-filter) route.
                    record_metrics["query_pure_survival_fraction"] = (
                        record_metrics["query_tokens_pure_positive"]
                        / max(record_metrics["query_tokens_touched"], 1)
                    )
                    record_metrics[
                        "proposals_invalid_fewer_than_two_pure_query_tokens"
                    ] = int(
                        record_metrics.get(
                            "proposals_invalid_fewer_than_two_pure_query_tokens",
                            0,
                        )
                    ) + len(filtered_rows)
                    record_metrics["effective_proposals_attempted"] = (
                        effective_attempted
                    )
                    record_metrics["effective_proposals_valid"] = effective_valid
                    record_metrics["post_voxel_filtered_proposals"] = (
                        raw_attempted - effective_attempted
                    )
                    record_metrics["post_voxel_replacement_count"] = 0
                    record_metrics["proposals_attempted"] = raw_attempted
                    record_metrics["proposals_valid"] = effective_valid
                    effective_folds_attempted = int(
                        record_metrics.get("folds_attempted", 0)
                    )
                    record_metrics["effective_folds_attempted"] = (
                        effective_folds_attempted
                    )
                    record_metrics["folds_attempted"] = 2 * raw_attempted
                    record_metrics["proposal_valid_fraction"] = (
                        effective_valid / max(raw_attempted, 1)
                    )
                edge_records.setdefault((granularity, stage), []).append(record)

        total = _zero_touch(stage_features["dec0"], self)
        stage_tensors = {
            stage: stage_features[stage].sum() * 0.0
            for stage in HIERARCHY_COARSE_TO_FINE
            if stage != "dec0"
        }
        stages: dict[str, dict[str, Any]] = {
            stage: {
                "active": False,
                "effective_weight_realized": 0.0,
                "loss": 0.0,
                "contribution_to_mask_total": 0.0,
                "group_count": 0,
                "proposals_attempted": 0,
                "proposals_valid": 0,
                "effective_proposals_attempted": 0,
                "effective_proposals_valid": 0,
                "post_voxel_filtered_proposals": 0,
                "post_voxel_replacement_count": 0,
                "effective_folds_attempted": 0,
                "folds_attempted": 0,
                "folds_valid": 0,
                "target_token_count": 0,
                "query_token_count": 0,
                "query_tokens_touched": 0,
                "query_tokens_fully_known": 0,
                "query_tokens_pure_positive": 0,
                "proposals_invalid_fewer_than_two_pure_query_tokens": 0,
            }
            for stage in HIERARCHY_COARSE_TO_FINE
            if stage != "dec0"
        }
        cells: dict[str, dict[str, Any]] = {}
        averaged_metric_names = (
            "bce_loss",
            "dice_loss",
            "hard_iou",
            "soft_target_fraction",
            "confidence_mean",
            "positive_mass",
            "negative_mass",
            "confidence_sum",
            "logit_mean",
        )
        counter_names = (
            "proposals_attempted",
            "proposals_valid",
            "effective_proposals_attempted",
            "effective_proposals_valid",
            "post_voxel_filtered_proposals",
            "post_voxel_replacement_count",
            "folds_attempted",
            "effective_folds_attempted",
            "folds_valid",
            "target_token_count",
            "query_token_count",
            "query_tokens_touched",
            "query_tokens_fully_known",
            "query_tokens_pure_positive",
            "proposals_invalid_fewer_than_two_pure_query_tokens",
        )
        cell_numerators: dict[str, torch.Tensor] = {}
        cell_denominators: dict[str, int] = {}
        for granularity in input_granularities:
            cell_records = [
                record
                for (cell, _), records in edge_records.items()
                if cell == granularity
                for record in records
            ]
            zero = stage_features["dec0"].sum() * 0.0
            numerator = sum(
                (record["loss_numerator"] for record in cell_records), zero
            )
            denominator = sum(
                int(record["loss_denominator"]) for record in cell_records
            )
            cell_numerators[granularity] = numerator
            cell_denominators[granularity] = denominator
            cells[granularity] = {
                "active": denominator > 0,
                "loss": float((numerator / max(denominator, 1)).detach()),
                "loss_numerator": float(numerator.detach()),
                "loss_denominator_valid_folds": denominator,
                "group_count": granularities.count(granularity),
                "route": route_metrics[granularity],
                "stages": {},
            }

        nonempty_granularities = tuple(
            granularity
            for granularity in input_granularities
            if cell_denominators[granularity] > 0
        )
        cell_equalization_denominator = len(nonempty_granularities)
        if cell_equalization_denominator:
            total = sum(
                (
                    cell_numerators[granularity]
                    / float(cell_denominators[granularity])
                    for granularity in nonempty_granularities
                ),
                total,
            ) / float(cell_equalization_denominator)

        for (granularity, stage), records in edge_records.items():
            edge_numerator = sum(
                (record["loss_numerator"] for record in records),
                stage_features[stage].sum() * 0.0,
            )
            edge_denominator = sum(
                int(record["loss_denominator"]) for record in records
            )
            cell_denominator = cell_denominators[granularity]
            effective_weight = (
                edge_denominator
                / float(cell_denominator * cell_equalization_denominator)
                if cell_denominator > 0 and cell_equalization_denominator > 0
                else 0.0
            )
            contribution = (
                edge_numerator
                / float(cell_denominator * cell_equalization_denominator)
                if effective_weight > 0.0
                else edge_numerator * 0.0
            )
            stage_tensors[stage] = stage_tensors[stage] + contribution
            summary: dict[str, Any] = {
                "active": edge_denominator > 0,
                "loss": float(
                    (edge_numerator / max(edge_denominator, 1)).detach()
                ),
                "loss_numerator": float(edge_numerator.detach()),
                "loss_denominator_valid_folds": edge_denominator,
                "effective_weight_realized": effective_weight,
                "contribution_to_mask_total": float(contribution.detach()),
                "group_count": len(records),
                "temperature": float(self.temperature(stage).detach()),
            }
            for key in counter_names:
                summary[key] = int(
                    sum(int(record["metrics"].get(key, 0)) for record in records)
                )
            summary["proposal_valid_fraction"] = summary["proposals_valid"] / max(
                summary["proposals_attempted"], 1
            )
            summary["routed_survival_fraction"] = summary[
                "proposal_valid_fraction"
            ]
            summary["minimum_routed_survival_fraction"] = (
                MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
            )
            summary["routed_survival_gate_passed"] = (
                summary["proposals_attempted"] == 0
                or summary["routed_survival_fraction"]
                >= MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
            )
            summary["fold_valid_fraction"] = summary["folds_valid"] / max(
                summary["folds_attempted"], 1
            )
            for key in averaged_metric_names:
                weighted_values = [
                    (
                        float(record["metrics"][key]),
                        int(record["loss_denominator"]),
                    )
                    for record in records
                    if key in record["metrics"]
                    and int(record["loss_denominator"]) > 0
                ]
                metric_denominator = sum(weight for _, weight in weighted_values)
                if metric_denominator:
                    summary[key] = sum(
                        value * weight for value, weight in weighted_values
                    ) / metric_denominator
            cells[granularity]["stages"][stage] = summary

            stage_summary = stages[stage]
            stage_summary["active"] = (
                bool(stage_summary["active"]) or edge_denominator > 0
            )
            stage_summary["effective_weight_realized"] += effective_weight
            stage_summary["contribution_to_mask_total"] += float(
                contribution.detach()
            )
            stage_summary["group_count"] += len(records)
            for key in counter_names:
                stage_summary[key] += int(summary[key])

        for stage, summary in stages.items():
            stage_records = [
                record
                for (cell, edge_stage), records in edge_records.items()
                if edge_stage == stage
                for record in records
            ]
            stage_denominator = sum(
                int(record["loss_denominator"]) for record in stage_records
            )
            stage_numerator = sum(
                (
                    float(record["metrics"]["loss_numerator"])
                    for record in stage_records
                ),
                0.0,
            )
            summary["loss"] = stage_numerator / max(stage_denominator, 1)
            summary["loss_numerator"] = stage_numerator
            summary["loss_denominator_valid_folds"] = stage_denominator
            summary["proposal_valid_fraction"] = summary["proposals_valid"] / max(
                summary["proposals_attempted"], 1
            )
            summary["routed_survival_fraction"] = summary[
                "proposal_valid_fraction"
            ]
            summary["minimum_routed_survival_fraction"] = (
                MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
            )
            summary["routed_survival_gate_passed"] = (
                summary["proposals_attempted"] == 0
                or summary["routed_survival_fraction"]
                >= MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
            )
            summary["fold_valid_fraction"] = summary["folds_valid"] / max(
                summary["folds_attempted"], 1
            )
            summary["temperature"] = float(self.temperature(stage).detach())

        expected_granularities = tuple(MASK_ROUTE_STAGES)
        input_group_counts = {
            granularity: granularities.count(granularity)
            for granularity in expected_granularities
        }
        fold_denominators_by_cell = {
            granularity: int(cell_denominators.get(granularity, 0))
            for granularity in expected_granularities
        }
        fold_denominators_by_route_stage: dict[str, dict[str, int]] = {}
        route_stage_validity: dict[str, dict[str, Any]] = {}
        for granularity in expected_granularities:
            cell_groups = [
                group
                for group, observed in zip(groups, granularities, strict=True)
                if observed == granularity
            ]
            cell_proposal_count = sum(
                int(group.batch.proposal_member_offsets.numel()) - 1
                for group in cell_groups
            )
            stage_rows: dict[str, dict[str, Any]] = {}
            assigned_proposals = 0
            for stage in MASK_ROUTE_STAGES[granularity]:
                records = edge_records.get((granularity, stage), [])
                proposals_attempted = sum(
                    int(record["metrics"]["proposals_attempted"])
                    for record in records
                )
                proposals_valid = sum(
                    int(record["metrics"]["proposals_valid"])
                    for record in records
                )
                effective_proposals_attempted = sum(
                    int(
                        record["metrics"].get(
                            "effective_proposals_attempted",
                            record["metrics"].get("proposals_attempted", 0),
                        )
                    )
                    for record in records
                )
                effective_proposals_valid = sum(
                    int(
                        record["metrics"].get(
                            "effective_proposals_valid",
                            record["metrics"].get("proposals_valid", 0),
                        )
                    )
                    for record in records
                )
                post_voxel_filtered_proposals = sum(
                    int(record["metrics"].get("post_voxel_filtered_proposals", 0))
                    for record in records
                )
                post_voxel_replacement_count = sum(
                    int(record["metrics"].get("post_voxel_replacement_count", 0))
                    for record in records
                )
                folds_attempted = sum(
                    int(record["metrics"]["folds_attempted"])
                    for record in records
                )
                folds_valid = sum(
                    int(record["loss_denominator"])
                    for record in records
                )
                assigned_proposals += proposals_attempted
                route_required = (
                    proposals_attempted > 0
                    if granularity == "g02"
                    else bool(cell_groups)
                )
                assignment_valid = (
                    proposals_attempted <= cell_proposal_count
                    if granularity == "g02"
                    else proposals_attempted == cell_proposal_count
                )
                denominator_valid = (not route_required) or folds_valid > 0
                proposal_survival_fraction = proposals_valid / max(
                    proposals_attempted, 1
                )
                routed_survival_gate_passed = (
                    (
                        proposals_attempted == 0
                        and not route_required
                    )
                    or (
                        proposals_attempted > 0
                        and proposal_survival_fraction
                        >= MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
                    )
                )
                stage_rows[stage] = {
                    "route_required": route_required,
                    "route_active": proposals_attempted > 0,
                    "assignment_valid": assignment_valid,
                    "stage_aware_representability_valid": (
                        denominator_valid and routed_survival_gate_passed
                    ),
                    "optimization_valid": (
                        assignment_valid
                        and denominator_valid
                    ),
                    "proposals_attempted": proposals_attempted,
                    "proposals_valid": proposals_valid,
                    "effective_proposals_attempted": (
                        effective_proposals_attempted
                    ),
                    "effective_proposals_valid": effective_proposals_valid,
                    "post_voxel_filtered_proposals": (
                        post_voxel_filtered_proposals
                    ),
                    "post_voxel_replacement_count": (
                        post_voxel_replacement_count
                    ),
                    "proposals_unrepresentable": (
                        proposals_attempted - proposals_valid
                    ),
                    "routed_survival_fraction": proposal_survival_fraction,
                    "minimum_routed_survival_fraction": (
                        MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
                    ),
                    "routed_survival_gate_passed": routed_survival_gate_passed,
                    "all_routed_proposals_have_valid_fold": (
                        proposals_attempted == 0
                        or proposals_valid == proposals_attempted
                    ),
                    "folds_attempted": folds_attempted,
                    "folds_valid": folds_valid,
                    "loss_denominator_valid_folds": folds_valid,
                    "zero_denominator": folds_valid == 0,
                }
            if granularity == "g02":
                route_assignment_valid = assigned_proposals == cell_proposal_count
            else:
                route_assignment_valid = all(
                    row["assignment_valid"] for row in stage_rows.values()
                )
            route_stages_valid = all(
                row["optimization_valid"] for row in stage_rows.values()
            )
            cell_valid = bool(
                cell_groups
                and cell_proposal_count > 0
                and fold_denominators_by_cell[granularity] > 0
                and route_assignment_valid
                and route_stages_valid
            )
            fold_denominators_by_route_stage[granularity] = {
                stage: int(row["loss_denominator_valid_folds"])
                for stage, row in stage_rows.items()
            }
            route_stage_validity[granularity] = {
                "optimization_valid": cell_valid,
                "stage_aware_representability_valid": all(
                    row["stage_aware_representability_valid"]
                    for row in stage_rows.values()
                ),
                "routed_survival_gate_passed": all(
                    row["routed_survival_gate_passed"]
                    for row in stage_rows.values()
                ),
                "minimum_routed_proposal_survival_fraction": (
                    MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
                ),
                "input_group_count": len(cell_groups),
                "input_proposal_count": cell_proposal_count,
                "assigned_proposal_count": assigned_proposals,
                "route_assignment_valid": route_assignment_valid,
                "loss_denominator_valid_folds": (
                    fold_denominators_by_cell[granularity]
                ),
                "zero_denominator": (
                    fold_denominators_by_cell[granularity] == 0
                ),
                "stages": stage_rows,
            }

        expected_input_set_valid = set(input_granularities) == set(
            expected_granularities
        )
        expected_three_cell_valid = bool(
            expected_input_set_valid
            and all(
                route_stage_validity[granularity]["optimization_valid"]
                for granularity in expected_granularities
            )
        )

        return {
            "loss_total": total,
            "stage_loss_tensors": stage_tensors,
            "metrics": {
                "schema_version": "litept_hierarchy_mask_optimization/v2",
                "query_projection_policy": self.query_projection_policy,
                "loss": float(total.detach()),
                "optimization_valid": expected_three_cell_valid,
                "expected_three_cell_valid": expected_three_cell_valid,
                "expected_input_granularity_set_valid": expected_input_set_valid,
                "route_stage_valid": all(
                    route_stage_validity[granularity]["optimization_valid"]
                    for granularity in expected_granularities
                ),
                "expected_granularities": list(expected_granularities),
                "input_granularities": list(input_granularities),
                "missing_input_granularities": [
                    granularity
                    for granularity in expected_granularities
                    if granularity not in input_granularities
                ],
                "input_group_counts_by_granularity": input_group_counts,
                "active_granularities": list(nonempty_granularities),
                "empty_granularities": [
                    granularity
                    for granularity in expected_granularities
                    if fold_denominators_by_cell[granularity] == 0
                ],
                "frame_group_count": len(groups),
                "cell_equalization_denominator": cell_equalization_denominator,
                "loss_denominator_valid_folds": sum(
                    fold_denominators_by_cell.values()
                ),
                "fold_denominators_by_cell": fold_denominators_by_cell,
                "fold_denominators_by_route_stage": (
                    fold_denominators_by_route_stage
                ),
                "minimum_routed_proposal_survival_fraction": (
                    MIN_ROUTED_PROPOSAL_SURVIVAL_FRACTION
                ),
                "route_stage_validity": route_stage_validity,
                "normalization": (
                    "sum_fold_losses/sum_valid_folds_within_granularity_then_"
                    "equal_mean_nonempty_granularities"
                ),
                "zero_denominator_policy": (
                    "differentiable_zero_edge; exclude_empty_granularity_from_"
                    "cell_equalization_denominator"
                ),
                "stages": stages,
                "cells": cells,
            },
        }


def native_variance_covariance_loss(
    features: torch.Tensor,
    *,
    variance_target: float = 1.0,
    covariance_weight: float = 0.04,
    max_tokens: int = 4096,
    min_covariance_tokens: int = 4,
    eps: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Distributed VICReg-style guard directly on native FP32 features.

    Each rank contributes the same number of deterministic, evenly spaced
    tokens: ``min(max_tokens, global minimum local token count)``.  With one
    scene per rank this is scene-balanced rather than large-scene weighted.
    Uneven counts and all-zero rank failures are synchronized before the
    differentiable all-gather, and metrics expose all sampling/discard counts.
    """

    if features.ndim != 2:
        raise ValueError("features must be [tokens, channels]")
    if variance_target <= 0 or covariance_weight < 0:
        raise ValueError("Invalid variance/covariance weights")
    if max_tokens < 2 or min_covariance_tokens < 2:
        raise ValueError("Token thresholds must be at least two")
    token_count, channels = features.shape
    zero = features.sum() * 0.0
    if channels < 1:
        raise ValueError("features must contain at least one channel")
    world_size = 1
    rank = 0
    per_rank_token_counts = [int(token_count)]
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_count = torch.tensor(
            [int(token_count)], dtype=torch.long, device=features.device
        )
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        per_rank_token_counts = [int(value.item()) for value in gathered_counts]
    balanced_count = min(
        int(max_tokens), min(per_rank_token_counts, default=0)
    )
    if balanced_count == 0:
        local_sample = features[:0]
    elif token_count > balanced_count:
        indices = torch.div(
            torch.arange(balanced_count, device=features.device) * token_count,
            balanced_count,
            rounding_mode="floor",
        )
        local_sample = features[indices]
    else:
        local_sample = features

    sample = local_sample
    zero_rank = any(count == 0 for count in per_rank_token_counts)
    if world_size > 1 and not zero_rank:
        from torch.distributed.nn.functional import (
            all_gather as differentiable_all_gather,
        )

        sample = torch.cat(
            list(differentiable_all_gather(local_sample)), dim=0
        )

    global_sample_count = int(sample.shape[0])
    per_rank_sample_counts = (
        [balanced_count for _ in range(world_size)]
        if not zero_rank
        else [0 for _ in range(world_size)]
    )
    common_metrics: dict[str, Any] = {
        "token_count": int(token_count),
        "local_token_count": int(token_count),
        "sampled_token_count": int(local_sample.shape[0]),
        "local_sampled_token_count": int(local_sample.shape[0]),
        "global_sampled_token_count": global_sample_count,
        "distributed": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "per_rank_token_counts": per_rank_token_counts,
        "per_rank_sampled_token_counts": per_rank_sample_counts,
        "per_rank_discarded_token_counts": [
            count - balanced_count if not zero_rank else count
            for count in per_rank_token_counts
        ],
        "balanced_tokens_per_rank": balanced_count if not zero_rank else 0,
        "zero_token_rank_detected": zero_rank,
        "global_sampling_semantics": (
            "equal_per_rank_min_global_count_cap_then_global_scene_balanced_stats"
        ),
    }
    if zero_rank or global_sample_count < 2:
        return zero, {
            **common_metrics,
            "valid": False,
            "invalid_reason": (
                "zero_token_rank" if zero_rank else "fewer_than_two_global_tokens"
            ),
            "covariance_valid": False,
            "loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
            "component_std_mean": 0.0,
            "component_std_min": 0.0,
            "participation_rank": 0.0,
        }

    # No per-token normalization or disposable projection: radial/amplitude
    # and low-rank collapse must remain visible to this native-stage guard.
    sample = sample.float()
    centered = sample - sample.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(sample.shape[0] - 1)
    diagonal = covariance.diagonal()
    component_std = torch.sqrt(diagonal.clamp_min(0.0) + eps)
    variance_loss = F.relu(float(variance_target) - component_std).mean()
    covariance_valid = int(sample.shape[0]) >= int(min_covariance_tokens)
    if covariance_valid and channels > 1:
        off_diagonal_square_sum = covariance.square().sum() - diagonal.square().sum()
        covariance_loss = off_diagonal_square_sum / float(channels * (channels - 1))
    else:
        covariance_loss = zero
    loss = variance_loss + float(covariance_weight) * covariance_loss
    with torch.no_grad():
        trace = diagonal.clamp_min(0.0).sum()
        frobenius_square = covariance.square().sum()
        participation_rank = trace.square() / frobenius_square.clamp_min(eps)
        if channels > 1:
            off_diagonal_abs_mean = (
                covariance.abs().sum() - diagonal.abs().sum()
            ) / float(channels * (channels - 1))
        else:
            off_diagonal_abs_mean = covariance.new_zeros(())
    return loss, {
        **common_metrics,
        "valid": True,
        "covariance_valid": bool(covariance_valid),
        "loss": float(loss.detach()),
        "variance_loss": float(variance_loss.detach()),
        "covariance_loss": float(covariance_loss.detach()),
        "component_std_mean": float(component_std.detach().mean()),
        "component_std_min": float(component_std.detach().min()),
        "covariance_offdiag_abs_mean": float(off_diagonal_abs_mean.detach()),
        "participation_rank": float(participation_rank.detach()),
    }


def _finalize_packed_native_vicreg_stage(
    sample: torch.Tensor,
    zero: torch.Tensor,
    *,
    common_metrics: Mapping[str, Any],
    zero_rank: bool,
    variance_target: float,
    covariance_weight: float,
    min_covariance_tokens: int,
    eps: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute one stage's unchanged FP32 VICReg statistics after packing."""

    global_sample_count = int(sample.shape[0])
    if zero_rank or global_sample_count < 2:
        return zero, {
            **common_metrics,
            "valid": False,
            "invalid_reason": (
                "zero_token_rank" if zero_rank else "fewer_than_two_global_tokens"
            ),
            "covariance_valid": False,
            "loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
            "component_std_mean": 0.0,
            "component_std_min": 0.0,
            "participation_rank": 0.0,
        }

    # Match native_variance_covariance_loss exactly after transport: all
    # statistics remain FP32 and no per-token normalization hides collapse.
    sample = sample.float()
    centered = sample - sample.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(sample.shape[0] - 1)
    diagonal = covariance.diagonal()
    component_std = torch.sqrt(diagonal.clamp_min(0.0) + eps)
    variance_loss = F.relu(float(variance_target) - component_std).mean()
    covariance_valid = int(sample.shape[0]) >= int(min_covariance_tokens)
    if covariance_valid and sample.shape[1] > 1:
        off_diagonal_square_sum = covariance.square().sum() - diagonal.square().sum()
        covariance_loss = off_diagonal_square_sum / float(
            sample.shape[1] * (sample.shape[1] - 1)
        )
    else:
        covariance_loss = zero
    loss = variance_loss + float(covariance_weight) * covariance_loss
    with torch.no_grad():
        trace = diagonal.clamp_min(0.0).sum()
        frobenius_square = covariance.square().sum()
        participation_rank = trace.square() / frobenius_square.clamp_min(eps)
        if sample.shape[1] > 1:
            off_diagonal_abs_mean = (
                covariance.abs().sum() - diagonal.abs().sum()
            ) / float(sample.shape[1] * (sample.shape[1] - 1))
        else:
            off_diagonal_abs_mean = covariance.new_zeros(())
    return loss, {
        **common_metrics,
        "valid": True,
        "covariance_valid": bool(covariance_valid),
        "loss": float(loss.detach()),
        "variance_loss": float(variance_loss.detach()),
        "covariance_loss": float(covariance_loss.detach()),
        "component_std_mean": float(component_std.detach().mean()),
        "component_std_min": float(component_std.detach().min()),
        "covariance_offdiag_abs_mean": float(off_diagonal_abs_mean.detach()),
        "participation_rank": float(participation_rank.detach()),
    }


def _packed_native_vicreg_losses(
    stage_features: Mapping[str, torch.Tensor],
    *,
    variance_target: float,
    covariance_weight: float,
    max_tokens: int,
    min_covariance_tokens: int = 4,
    eps: float = 1.0e-4,
) -> tuple[dict[str, tuple[torch.Tensor, dict[str, Any]]], dict[str, Any]]:
    """Compute all five native VICReg stages with one packed collective.

    One count all-gather exchanges the local size of each stage.  The
    resulting per-stage global-minimum caps make every rank's flattened
    five-stage vector the same length.  One differentiable all-gather then
    transports that vector; slices are reconstructed before applying the
    standalone FP32 statistics.  A zero-count rank invalidates only the
    affected stage, preserving the prior fail-closed behavior while allowing
    other stages to retain gradients.
    """

    if set(stage_features) != set(HIERARCHY_FINE_TO_COARSE):
        raise ValueError("stage_features must contain all five native LitePT stages")
    if variance_target <= 0 or covariance_weight < 0:
        raise ValueError("Invalid variance/covariance weights")
    if int(max_tokens) < 2 or int(min_covariance_tokens) < 2:
        raise ValueError("Token thresholds must be at least two")
    stages = tuple(HIERARCHY_COARSE_TO_FINE)
    first = stage_features[stages[0]]
    if first.ndim != 2 or first.shape[1] < 1:
        raise ValueError("native stage features must be [tokens, channels]")
    device = first.device
    for stage in stages:
        features = stage_features[stage]
        if features.ndim != 2 or features.shape[1] < 1:
            raise ValueError(
                f"{stage} native features must be [tokens, channels], "
                f"got {tuple(features.shape)}"
            )
        if features.device != device:
            raise ValueError("all native stage features must share one device")

    world_size = 1
    rank = 0
    local_counts = [int(stage_features[stage].shape[0]) for stage in stages]
    per_rank_counts = [local_counts]
    count_collective_performed = False
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_count_tensor = torch.tensor(
            local_counts, dtype=torch.long, device=device
        )
        gathered_counts = [
            torch.zeros_like(local_count_tensor) for _ in range(world_size)
        ]
        dist.all_gather(gathered_counts, local_count_tensor)
        per_rank_counts = [
            [int(value) for value in tensor.tolist()]
            for tensor in gathered_counts
        ]
        count_collective_performed = True

    caps = {
        stage: min(
            int(max_tokens),
            min(per_rank[stage_index] for per_rank in per_rank_counts),
        )
        for stage_index, stage in enumerate(stages)
    }

    local_samples: dict[str, torch.Tensor] = {}
    segment_offsets: dict[str, tuple[int, int]] = {}
    segment_offset = 0
    packed_segments: list[torch.Tensor] = []
    for stage in stages:
        features = stage_features[stage]
        cap = int(caps[stage])
        if cap == 0:
            local_samples[stage] = features[:0]
            continue
        token_count = int(features.shape[0])
        if token_count > cap:
            indices = torch.div(
                torch.arange(cap, device=device) * token_count,
                cap,
                rounding_mode="floor",
            )
            local_sample = features[indices]
        else:
            local_sample = features
        local_samples[stage] = local_sample
        flattened = local_sample.float().reshape(-1)
        segment_length = int(flattened.numel())
        segment_offsets[stage] = (segment_offset, segment_length)
        segment_offset += segment_length
        packed_segments.append(flattened)

    packed_local = (
        torch.cat(packed_segments, dim=0)
        if packed_segments
        else first.new_empty((0,), dtype=torch.float32)
    )
    packed_length = int(packed_local.numel())
    differentiable_collective_performed = False
    if world_size > 1 and packed_length > 0:
        from torch.distributed.nn.functional import (
            all_gather as differentiable_all_gather,
        )

        packed_global = torch.cat(
            list(differentiable_all_gather(packed_local)), dim=0
        )
        differentiable_collective_performed = True
    else:
        packed_global = packed_local

    output: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
    for stage_index, stage in enumerate(stages):
        features = stage_features[stage]
        cap = int(caps[stage])
        zero = features.sum() * 0.0
        zero_rank = any(
            int(per_rank[stage_index]) == 0 for per_rank in per_rank_counts
        )
        if cap == 0 or packed_length == 0:
            sample = features[:0]
        elif world_size == 1:
            offset, length = segment_offsets[stage]
            sample = packed_global[offset : offset + length].reshape(
                cap, int(features.shape[1])
            )
        else:
            offset, length = segment_offsets[stage]
            rank_segments = [
                packed_global[
                    rank_index * packed_length + offset :
                    rank_index * packed_length + offset + length
                ].reshape(cap, int(features.shape[1]))
                for rank_index in range(world_size)
            ]
            sample = torch.cat(rank_segments, dim=0)

        common_metrics: dict[str, Any] = {
            "token_count": int(features.shape[0]),
            "local_token_count": int(features.shape[0]),
            "sampled_token_count": int(local_samples[stage].shape[0]),
            "local_sampled_token_count": int(local_samples[stage].shape[0]),
            "global_sampled_token_count": int(sample.shape[0]),
            "distributed": world_size > 1,
            "world_size": world_size,
            "rank": rank,
            "per_rank_token_counts": [
                int(per_rank[stage_index]) for per_rank in per_rank_counts
            ],
            "per_rank_sampled_token_counts": (
                [cap for _ in range(world_size)]
                if not zero_rank
                else [0 for _ in range(world_size)]
            ),
            "per_rank_discarded_token_counts": [
                int(per_rank[stage_index]) - cap
                if not zero_rank
                else int(per_rank[stage_index])
                for per_rank in per_rank_counts
            ],
            "balanced_tokens_per_rank": cap if not zero_rank else 0,
            "zero_token_rank_detected": zero_rank,
            "global_sampling_semantics": (
                "equal_per_rank_min_global_count_cap_then_global_"
                "scene_balanced_stats_packed_five_stage"
            ),
            "packed_collective": True,
            "packed_stage_index": stage_index,
            "packed_stage_cap": cap,
            "packed_local_vector_length": packed_length,
            "count_all_gather_performed": count_collective_performed,
            "differentiable_all_gather_performed": (
                differentiable_collective_performed
            ),
        }
        output[stage] = _finalize_packed_native_vicreg_stage(
            sample,
            zero,
            common_metrics=common_metrics,
            zero_rank=zero_rank,
            variance_target=float(variance_target),
            covariance_weight=float(covariance_weight),
            min_covariance_tokens=int(min_covariance_tokens),
            eps=float(eps),
        )

    return output, {
        "schema_version": "native_vicreg_packed_collective/v1",
        "world_size": world_size,
        "rank": rank,
        "stages": list(stages),
        "per_rank_token_counts": per_rank_counts,
        "caps_by_stage": {stage: int(caps[stage]) for stage in stages},
        "packed_local_vector_length": packed_length,
        "count_all_gather_count": int(count_collective_performed),
        "differentiable_all_gather_count": int(
            differentiable_collective_performed
        ),
        "zero_token_stages": [
            stage
            for stage_index, stage in enumerate(stages)
            if any(int(per_rank[stage_index]) == 0 for per_rank in per_rank_counts)
        ],
        "semantics": (
            "one_all_gather_of_five_counts_then_one_differentiable_"
            "all_gather_of_rank_equal_stage_segments"
        ),
    }


def native_hierarchy_vicreg_loss(
    stage_features: Mapping[str, torch.Tensor],
    *,
    variance_target: float = 1.0,
    covariance_weight: float = 0.04,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Equal-weight variance/covariance guard over all five native stages."""

    stage_weight = 1.0 / len(HIERARCHY_FINE_TO_COARSE)
    packed_results, collective_metrics = _packed_native_vicreg_losses(
        stage_features,
        variance_target=variance_target,
        covariance_weight=covariance_weight,
        max_tokens=max_tokens,
    )
    total = sum(
        (features.sum() * 0.0 for features in stage_features.values()),
        next(iter(stage_features.values())).new_zeros(()),
    )
    stage_tensors: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for stage in HIERARCHY_COARSE_TO_FINE:
        stage_loss, stage_metrics = packed_results[stage]
        weighted = stage_weight * stage_loss
        total = total + weighted
        stage_tensors[stage] = weighted
        metrics[stage] = {**stage_metrics, "stage_weight": stage_weight}
    return {
        "loss_total": total,
        "stage_loss_tensors": stage_tensors,
        "metrics": {
            "loss": float(total.detach()),
            "variance_target": float(variance_target),
            "covariance_weight": float(covariance_weight),
            "max_tokens": int(max_tokens),
            "collective": collective_metrics,
            "stages": metrics,
        },
    }


def hierarchy_gradient_telemetry(
    *,
    loss_components: Mapping[str, torch.Tensor],
    junctions: Mapping[str, torch.Tensor],
    reference_objective: str = "final_dec0",
    max_elements_per_junction: int = 16384,
) -> dict[str, Any]:
    """Measure objective gradients at native stages without consuming the graph.

    The helper uses ``autograd.grad(..., retain_graph=True)`` and never writes
    ``.grad``.  Norms and cosine products use a deterministic bounded subset of
    each junction gradient; the normal training backward remains valid.
    """

    if reference_objective not in loss_components:
        raise ValueError(f"Missing reference objective {reference_objective!r}")
    if int(max_elements_per_junction) <= 0:
        raise ValueError("max_elements_per_junction must be positive")
    if not junctions:
        raise ValueError("At least one named junction is required")
    names = tuple(junctions)
    tensors = tuple(junctions[name] for name in names)
    if any(not tensor.requires_grad for tensor in tensors):
        raise ValueError("Every gradient-telemetry junction must require gradients")

    sampled_gradients: dict[str, dict[str, torch.Tensor | None]] = {}
    output: dict[str, Any] = {
        "schema": "litept_hierarchy_gradient_telemetry/v1",
        "reference_objective": reference_objective,
        "max_elements_per_junction": int(max_elements_per_junction),
        "objectives": {},
    }
    for objective, loss in loss_components.items():
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError(f"Objective {objective!r} must be a scalar tensor")
        gradients = torch.autograd.grad(
            loss,
            tensors,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        sampled_gradients[objective] = {}
        stage_metrics: dict[str, Any] = {}
        for name, gradient in zip(names, gradients, strict=True):
            if gradient is None:
                sampled = None
                norm = 0.0
                finite = True
                sampled_elements = 0
            else:
                flat = gradient.detach().float().reshape(-1)
                if flat.numel() > int(max_elements_per_junction):
                    indices = torch.div(
                        torch.arange(
                            int(max_elements_per_junction), device=flat.device
                        )
                        * flat.numel(),
                        int(max_elements_per_junction),
                        rounding_mode="floor",
                    )
                    sampled = flat[indices]
                else:
                    sampled = flat
                norm = float(torch.linalg.vector_norm(sampled))
                finite = bool(torch.isfinite(sampled).all().item())
                sampled_elements = int(sampled.numel())
            sampled_gradients[objective][name] = sampled
            stage_metrics[name] = {
                "grad_norm": norm,
                "finite": finite,
                "sampled_elements": sampled_elements,
                "cosine_vs_final_dec0": 0.0,
                "cosine_valid": False,
                "grad_norm_ratio_vs_final_dec0": 0.0,
                "norm_ratio_valid": False,
            }
        output["objectives"][objective] = {"stages": stage_metrics}

    reference = sampled_gradients[reference_objective]
    for objective, gradients in sampled_gradients.items():
        for name in names:
            gradient = gradients[name]
            reference_gradient = reference[name]
            metrics = output["objectives"][objective]["stages"][name]
            if gradient is None or reference_gradient is None:
                continue
            gradient_norm = torch.linalg.vector_norm(gradient)
            reference_norm = torch.linalg.vector_norm(reference_gradient)
            if float(gradient_norm) > 0.0 and float(reference_norm) > 0.0:
                cosine = torch.dot(gradient, reference_gradient) / (
                    gradient_norm * reference_norm
                )
                metrics["cosine_vs_final_dec0"] = float(cosine)
                metrics["cosine_valid"] = True
                metrics["grad_norm_ratio_vs_final_dec0"] = float(
                    gradient_norm / reference_norm
                )
                metrics["norm_ratio_valid"] = True
    return output
