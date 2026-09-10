"""Deterministic frame-local sampling for multigranular 2D-mask pretraining.

The v2 API deliberately returns an ordered sequence of frame-local batches.
Visibility and proposal-membership CSR are never concatenated across physical
frames, while one caller-owned backbone forward can still serve every group.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import torch

from delimit3d.data.partfield_contrastive import (
    SAMPLER_ALGORITHM_VERSION,
    SAMPLER_SEED_SCHEME,
    ContrastiveTripletBatch,
    build_relation_membership_index,
    co_membership_safe_negative_mask,
    _unique_proposal_member_counts,
    sample_safe_overlapping_proposal_triplets,
    sample_unique_positive_pairs,
    SAMPLER_PRESELECTION_POLICY,
    token_positive_pair_payload_digest,
    validate_token_positive_pair_payload,
)


COVERAGE_STATE_SCHEMA = "partfield_deterministic_coverage_state/v1"
DEFAULT_2D_CELLS = ("2d/g02", "2d/g05", "2d/g08")
PRESELECTION_POLICY = SAMPLER_PRESELECTION_POLICY


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("/".join(str(part) for part in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _preselection_positive_pair_generator(
    *,
    seed: int,
    scene_id: str,
    operation: str,
    cell: str,
    frame_id: str,
    proposal_index: int,
    device: torch.device,
) -> torch.Generator:
    """Build the versioned proposal-local representative pair stream."""

    return torch.Generator(device=device).manual_seed(
        _stable_seed(
            PRESELECTION_POLICY,
            int(seed),
            scene_id,
            operation,
            cell,
            frame_id,
            int(proposal_index),
            "positive-token-pairs",
        )
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DeterministicCoverageState:
    """Serializable cursor over one deterministic circular permutation.

    The identity records ``seed/epoch/scene/operation`` explicitly.  The
    permutation itself is stable across epochs; callers advance ``cursor`` by
    the realized quotas from preceding epochs to guarantee catalog coverage
    rather than drawing a fresh random subset every epoch.
    """

    seed: int
    epoch: int
    scene_id: str
    operation: str
    population_size: int
    population_fingerprint: str
    cursor: int = 0
    schema_version: str = COVERAGE_STATE_SCHEMA
    algorithm_version: str = SAMPLER_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COVERAGE_STATE_SCHEMA:
            raise ValueError(f"unsupported coverage schema {self.schema_version!r}")
        if int(self.seed) < 0 or int(self.epoch) < 0 or int(self.cursor) < 0:
            raise ValueError("seed, epoch, and cursor must be non-negative")
        if int(self.population_size) <= 0:
            raise ValueError("population_size must be positive")
        if not self.scene_id or not self.operation or not self.population_fingerprint:
            raise ValueError("scene, operation, and population fingerprint are required")
        if self.algorithm_version != SAMPLER_ALGORITHM_VERSION:
            raise ValueError(
                f"unsupported sampler algorithm version {self.algorithm_version!r}"
            )

    @property
    def key(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "seed": int(self.seed),
                "epoch": int(self.epoch),
                "scene_id": self.scene_id,
                "operation": self.operation,
                "population_size": int(self.population_size),
                "population_fingerprint": self.population_fingerprint,
                "algorithm_version": self.algorithm_version,
            }
        )

    def draw_unique(
        self, requested_count: int
    ) -> tuple[tuple[int, ...], "DeterministicCoverageState", dict[str, Any]]:
        """Return an adaptive, duplicate-free circular slice and next state."""

        requested_count = int(requested_count)
        if requested_count < 0:
            raise ValueError("requested_count must be non-negative")
        selected_count = min(requested_count, int(self.population_size))
        permutation = self._permutation()
        positions = (
            int(self.cursor)
            + torch.arange(selected_count, dtype=torch.long, device="cpu")
        ) % int(self.population_size)
        selected = tuple(int(value) for value in permutation[positions].tolist())
        next_state = replace(self, cursor=int(self.cursor) + selected_count)
        metadata = {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "seed_scheme": SAMPLER_SEED_SCHEME,
            "state_key": self.key,
            "requested_count": requested_count,
            "selected_count": selected_count,
            "adaptive_truncation": selected_count < requested_count,
            "population_size": int(self.population_size),
            "population_fingerprint": self.population_fingerprint,
            "cursor_before": int(self.cursor),
            "cursor_after": int(next_state.cursor),
            "cycle_before": int(self.cursor) // int(self.population_size),
            "cycle_after": (
                (int(next_state.cursor) - 1) // int(self.population_size)
                if selected_count
                else int(self.cursor) // int(self.population_size)
            ),
            "crossed_cycle_boundary": bool(
                selected_count
                and int(self.cursor) // int(self.population_size)
                != (int(next_state.cursor) - 1) // int(self.population_size)
            ),
            "duplicate_count": len(selected) - len(set(selected)),
        }
        return selected, next_state, metadata

    def _permutation(self) -> torch.Tensor:
        """Recreate the immutable raw catalog permutation for this state."""

        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed(
                self.schema_version,
                self.seed,
                self.scene_id,
                self.operation,
                self.population_size,
                self.population_fingerprint,
                "coverage-permutation",
            )
        )
        return torch.randperm(
            int(self.population_size), generator=generator, device="cpu"
        )

    def scan_eligible(
        self,
        requested_count: int,
        *,
        eligible_positions: Sequence[int],
    ) -> tuple[tuple[int, ...], "DeterministicCoverageState", dict[str, Any]]:
        """Scan one raw permutation cycle until eligible entries fill a quota.

        Unlike :meth:`draw_unique`, the cursor advances by every inspected raw
        entry, including entries rejected by a deterministic pre-selection
        predicate.  At most one population cycle is inspected.  If fewer
        eligible entries exist than requested, all eligible entries are
        returned and the caller can report adaptive truncation or fail closed.
        """

        requested_count = int(requested_count)
        if requested_count < 0:
            raise ValueError("requested_count must be non-negative")
        population_size = int(self.population_size)
        eligible = tuple(int(value) for value in eligible_positions)
        if any(value < 0 or value >= population_size for value in eligible):
            raise ValueError("eligible coverage positions are out of range")
        if len(set(eligible)) != len(eligible):
            raise ValueError("eligible coverage positions must be unique")

        target = min(requested_count, population_size, len(eligible))
        permutation = self._permutation()
        selected: list[int] = []
        inspected: list[int] = []
        eligible_set = set(eligible)
        # A zero quota intentionally consumes no raw coverage state.  Any
        # positive quota, including an empty eligible population, scans at
        # most one complete cycle so the cursor records exact inspection.
        max_inspected = 0 if requested_count == 0 else population_size
        for offset in range(max_inspected):
            position = int(
                permutation[(int(self.cursor) + offset) % population_size].item()
            )
            inspected.append(position)
            if position in eligible_set and len(selected) < target:
                selected.append(position)
                if len(selected) == target:
                    break

        inspected_count = len(inspected)
        next_state = replace(
            self,
            cursor=int(self.cursor) + inspected_count,
        )
        skipped = [position for position in inspected if position not in selected]
        metadata = {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "seed_scheme": SAMPLER_SEED_SCHEME,
            "state_key": self.key,
            "requested_count": requested_count,
            "selected_count": len(selected),
            "eligible_population_size": len(eligible),
            # ``scan_eligible`` receives the complete raw predicate, so its
            # representability count is exhaustive even when the quota is
            # filled before the permutation cycle is consumed.
            "representable_population_complete": True,
            "adaptive_truncation": len(selected) < requested_count,
            "population_size": population_size,
            "population_fingerprint": self.population_fingerprint,
            "cursor_before": int(self.cursor),
            "cursor_after": int(next_state.cursor),
            "inspected_count": inspected_count,
            "inspected_positions": list(inspected),
            "selected_positions": list(selected),
            "skipped_positions": skipped,
            "cycle_before": int(self.cursor) // population_size,
            "cycle_after": (
                (int(next_state.cursor) - 1) // population_size
                if inspected_count
                else int(self.cursor) // population_size
            ),
            "crossed_cycle_boundary": bool(
                inspected_count
                and int(self.cursor) // population_size
                != (int(next_state.cursor) - 1) // population_size
            ),
            "duplicate_count": len(inspected) - len(set(inspected)),
        }
        return tuple(selected), next_state, metadata

    def scan_eligible_lazy(
        self,
        requested_count: int,
        *,
        evaluate_position: Callable[[int], bool],
    ) -> tuple[tuple[int, ...], "DeterministicCoverageState", dict[str, Any]]:
        """Scan the raw permutation while evaluating positions on demand.

        ``evaluate_position`` is called in exactly the immutable permutation
        order, once per inspected raw position, until the requested quota is
        filled or one complete population cycle has been consumed.  The
        callback is deliberately position-based so callers can cache the
        resulting evidence by raw catalog position and replay a saved cursor
        without changing the coverage contract.

        The returned ``eligible_population_size`` is a lower bound unless
        ``representable_population_complete`` is true.  This is intentional:
        the hot path must not exhaustively evaluate the remaining catalog just
        to report an offline-audit statistic.
        """

        requested_count = int(requested_count)
        if requested_count < 0:
            raise ValueError("requested_count must be non-negative")
        if not callable(evaluate_position):
            raise TypeError("evaluate_position must be callable")

        population_size = int(self.population_size)
        target = min(requested_count, population_size)
        permutation = self._permutation()
        selected: list[int] = []
        inspected: list[int] = []
        eligible_count = 0

        # A zero quota intentionally consumes no raw coverage state.  Any
        # positive quota, including a catalog with no eligible rows, inspects
        # at most one complete cycle and therefore preserves exact cursor and
        # exhaustion accounting.
        max_inspected = 0 if requested_count == 0 else population_size
        for offset in range(max_inspected):
            position = int(
                permutation[(int(self.cursor) + offset) % population_size].item()
            )
            inspected.append(position)
            if bool(evaluate_position(position)):
                eligible_count += 1
                selected.append(position)
                if len(selected) == target:
                    break

        inspected_count = len(inspected)
        next_state = replace(
            self,
            cursor=int(self.cursor) + inspected_count,
        )
        selected_set = set(selected)
        skipped = [position for position in inspected if position not in selected_set]
        complete = bool(
            requested_count > 0 and inspected_count >= population_size
        )
        metadata = {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "seed_scheme": SAMPLER_SEED_SCHEME,
            "state_key": self.key,
            "requested_count": requested_count,
            "selected_count": len(selected),
            # This is an inspected lower bound unless the explicit completion
            # flag below says that the full cycle was exhausted.
            "eligible_population_size": eligible_count,
            "representable_population_complete": complete,
            "adaptive_truncation": len(selected) < requested_count,
            "population_size": population_size,
            "population_fingerprint": self.population_fingerprint,
            "cursor_before": int(self.cursor),
            "cursor_after": int(next_state.cursor),
            "inspected_count": inspected_count,
            "inspected_positions": list(inspected),
            "selected_positions": list(selected),
            "skipped_positions": skipped,
            "cycle_before": int(self.cursor) // population_size,
            "cycle_after": (
                (int(next_state.cursor) - 1) // population_size
                if inspected_count
                else int(self.cursor) // population_size
            ),
            "crossed_cycle_boundary": bool(
                inspected_count
                and int(self.cursor) // population_size
                != (int(next_state.cursor) - 1) // population_size
            ),
            "duplicate_count": len(inspected) - len(set(inspected)),
        }
        return tuple(selected), next_state, metadata

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def for_epoch(self, epoch: int) -> "DeterministicCoverageState":
        """Rebind a saved cursor to the epoch in which it will be consumed.

        Cursors are the cross-epoch replay state; the epoch is part of the
        state identity so that an old record cannot silently be used as if it
        came from a different schedule.  A resumable checkpoint stores the
        state after the preceding epoch, therefore callers explicitly rebind
        that record before the first visit of the resumed epoch.  The sampler
        still validates the cursor against the deterministic epoch-derived
        minimum and the catalog fingerprint.
        """

        epoch = int(epoch)
        if epoch < int(self.epoch):
            raise ValueError(
                "coverage state cannot be rebound to an earlier epoch"
            )
        return replace(self, epoch=epoch)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeterministicCoverageState":
        expected = {
            "seed",
            "epoch",
            "scene_id",
            "operation",
            "population_size",
            "population_fingerprint",
            "cursor",
            "schema_version",
        }
        allowed = expected | {"algorithm_version"}
        if not expected.issubset(value) or not set(value).issubset(allowed):
            raise ValueError(
                "coverage state fields differ: "
                f"missing={sorted(expected - set(value))} "
                f"extra={sorted(set(value) - expected)}"
            )
        decoded = dict(value)
        decoded.setdefault("algorithm_version", SAMPLER_ALGORITHM_VERSION)
        return cls(**decoded)

    @classmethod
    def from_json(cls, value: str) -> "DeterministicCoverageState":
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("coverage state JSON must encode an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class FrameProposalCatalog:
    """One cell/frame proposal universe using scene-point indices."""

    cell: str
    frame_id: str
    visible_indices: torch.Tensor
    proposal_offsets: torch.Tensor
    proposal_point_indices: torch.Tensor

    def valid_proposal_indices(self) -> torch.Tensor:
        if not self.cell.startswith("2d/g") or not self.frame_id:
            raise ValueError("a 2D source cell and physical frame ID are required")
        if self.visible_indices.ndim != 1 or self.proposal_offsets.ndim != 1:
            raise ValueError("visible indices and proposal offsets must be vectors")
        if self.proposal_point_indices.ndim != 1:
            raise ValueError("proposal point indices must be a vector")
        integral_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if any(
            tensor.dtype not in integral_dtypes
            for tensor in (
                self.visible_indices,
                self.proposal_offsets,
                self.proposal_point_indices,
            )
        ):
            raise TypeError("frame proposal indices and offsets must be integral")
        devices = {
            self.visible_indices.device,
            self.proposal_offsets.device,
            self.proposal_point_indices.device,
        }
        if len(devices) != 1:
            raise ValueError("frame proposal tensors must share one device")
        if self.proposal_offsets.numel() == 0 or int(self.proposal_offsets[0]) != 0:
            raise ValueError("proposal offsets must start at zero")
        if bool((self.proposal_offsets < 0).any()):
            raise ValueError("proposal offsets must be non-negative")
        if int(self.proposal_offsets[-1]) != int(self.proposal_point_indices.numel()):
            raise ValueError("proposal offsets do not span proposal memberships")
        if bool((self.proposal_offsets[1:] < self.proposal_offsets[:-1]).any()):
            raise ValueError("proposal offsets must be non-decreasing")
        if self.visible_indices.numel() and bool((self.visible_indices < 0).any()):
            raise ValueError("visible point indices must be non-negative")
        if self.proposal_point_indices.numel() and bool(
            (self.proposal_point_indices < 0).any()
        ):
            raise ValueError("proposal point indices must be non-negative")
        visible = torch.unique(self.visible_indices.long(), sorted=True)
        if self.proposal_point_indices.numel() and not bool(
            torch.isin(self.proposal_point_indices.long(), visible).all()
        ):
            raise ValueError("proposal membership must be a subset of visible points")
        unique_counts = _unique_proposal_member_counts(
            self.proposal_offsets,
            self.proposal_point_indices,
        )
        return torch.where(
            (unique_counts >= 2) & (unique_counts < int(visible.numel()))
        )[0].long()


@dataclass(frozen=True)
class FrameLocalContrastiveGroup:
    """One independently sampled visibility universe within a scene visit."""

    cell: str
    frame_id: str
    batch: ContrastiveTripletBatch
    selected_proposal_indices: tuple[int, ...]
    coverage_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MultigranularFrameGroupPlan:
    """Scene-level result retaining accounting for cells with no group.

    ``coverage_by_cell`` is ordered by ``quota_by_cell`` and is fully JSON
    serializable.  In particular, an empty valid catalog is represented by a
    zero realized count and ``state_before/state_after=None`` rather than by a
    padded or duplicated proposal selection.
    """

    groups: tuple[FrameLocalContrastiveGroup, ...]
    quota_by_cell: Mapping[str, int]
    coverage_by_cell: Mapping[str, Mapping[str, Any]]

    @property
    def requested_proposal_count(self) -> int:
        return sum(int(value) for value in self.quota_by_cell.values())

    @property
    def realized_proposal_count(self) -> int:
        return sum(
            int(metadata["realized_count"])
            for metadata in self.coverage_by_cell.values()
        )

    @property
    def contrastive_retained_proposal_count(self) -> int:
        return sum(
            int(metadata["contrastive_retained_count"])
            for metadata in self.coverage_by_cell.values()
        )

    def state_after_by_cell(self) -> dict[str, Mapping[str, Any] | None]:
        """Return the serializable post-draw cursor record for provenance."""

        return {
            cell: metadata["state_after"]
            for cell, metadata in self.coverage_by_cell.items()
        }

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "algorithm_version": SAMPLER_ALGORITHM_VERSION,
            "seed_scheme": SAMPLER_SEED_SCHEME,
            "quota_by_cell": dict(self.quota_by_cell),
            "requested_proposal_count": self.requested_proposal_count,
            "realized_proposal_count": self.realized_proposal_count,
            "contrastive_retained_proposal_count": (
                self.contrastive_retained_proposal_count
            ),
            "adaptive_truncation": (
                self.realized_proposal_count < self.requested_proposal_count
            ),
            "coverage_by_cell": {
                cell: dict(metadata)
                for cell, metadata in self.coverage_by_cell.items()
            },
        }


class NoRetainedFrameGroupsError(RuntimeError):
    """Raised when a scene produced accounting but no trainable frame group."""

    def __init__(self, plan: MultigranularFrameGroupPlan) -> None:
        self.plan = plan
        super().__init__(
            "scene plan retained zero frame groups; inspect exception.plan.audit_metadata()"
        )


def rotating_three_cell_quotas(
    epoch: int,
    *,
    total_proposals: int = 16,
    cells: Sequence[str] = DEFAULT_2D_CELLS,
) -> dict[str, int]:
    """Allocate a long-run-balanced rotating quota (16 becomes 6/5/5)."""

    if int(epoch) < 0 or int(total_proposals) <= 0:
        raise ValueError("epoch must be non-negative and total_proposals positive")
    ordered = tuple(str(cell) for cell in cells)
    if len(ordered) != 3 or len(set(ordered)) != 3:
        raise ValueError("exactly three unique cells are required")
    base, remainder = divmod(int(total_proposals), len(ordered))
    output = {cell: base for cell in ordered}
    start = int(epoch) % len(ordered)
    for offset in range(remainder):
        output[ordered[(start + offset) % len(ordered)]] += 1
    return output


def _catalog_fingerprint(entries: Sequence[tuple[int, int, str]]) -> str:
    return _canonical_sha256(
        [
            {"frame_position": frame, "proposal_index": proposal, "frame_id": frame_id}
            for frame, proposal, frame_id in entries
        ]
    )


LITEPT_HIERARCHY_STAGES = ("dec0", "dec1", "dec2", "dec3", "enc4")
_STATIC_ROUTE_STAGES = {
    "g02": ("dec1",),
    "g05": ("dec2", "dec3"),
    "g08": ("dec3", "enc4"),
}


@dataclass(frozen=True)
class LitePTHierarchyAncestry:
    """Deterministic point-to-native-stage maps for one augmented scene.

    The construction mirrors :class:`LitePTBackbone`: finest coordinates are
    ``floor(points / grid_size)`` after a per-scene minimum shift, and every
    coarser grid is the sorted unique set of ``child_grid // 2``.  The
    ``representative_indices`` field is the first source point in each sorted
    finest voxel, matching the wrapper's representative-first/eval rule.
    """

    point_to_stage: Mapping[str, torch.Tensor]
    stage_grids: Mapping[str, torch.Tensor]
    parent_maps: tuple[torch.Tensor, ...]
    representative_indices: torch.Tensor


def derive_litept_hierarchy_ancestry(
    points: torch.Tensor,
    *,
    grid_size: float = 0.02,
) -> LitePTHierarchyAncestry:
    """Derive exact LitePT voxel and pooling ancestry without a backbone.

    This helper is intentionally scene-local.  The wrapper normalizes input
    coordinates to float32 before voxelization, so the same conversion is
    applied here.  One-scene planners therefore use the per-scene minimum
    shift directly; batched callers should invoke this once per scene.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if points.numel() == 0:
        raise ValueError("points must contain at least one scene point")
    if float(grid_size) <= 0.0:
        raise ValueError("grid_size must be positive")
    coord = points.float().contiguous()
    grid = torch.floor(coord / float(grid_size)).to(torch.int64)
    grid = grid - grid.min(dim=0).values
    finest_grid, point_to_finest = torch.unique(
        grid,
        dim=0,
        sorted=True,
        return_inverse=True,
    )

    sorted_source = torch.argsort(point_to_finest, stable=True)
    counts = torch.bincount(
        point_to_finest,
        minlength=int(finest_grid.shape[0]),
    )
    starts = torch.cumsum(counts, dim=0) - counts
    representative_indices = sorted_source[starts]

    point_to_stage: dict[str, torch.Tensor] = {
        "dec0": point_to_finest.long()
    }
    stage_grids: dict[str, torch.Tensor] = {"dec0": finest_grid}
    parent_maps: list[torch.Tensor] = []
    child_grid = finest_grid
    child_point_map = point_to_finest.long()
    for child, parent in zip(
        LITEPT_HIERARCHY_STAGES[:-1],
        LITEPT_HIERARCHY_STAGES[1:],
        strict=True,
    ):
        parent_grid, child_to_parent = torch.unique(
            child_grid // 2,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        child_to_parent = child_to_parent.long()
        parent_maps.append(child_to_parent)
        child_point_map = child_to_parent[child_point_map]
        point_to_stage[parent] = child_point_map
        stage_grids[parent] = parent_grid
        child_grid = parent_grid

    return LitePTHierarchyAncestry(
        point_to_stage=point_to_stage,
        stage_grids=stage_grids,
        parent_maps=tuple(parent_maps),
        representative_indices=representative_indices.long(),
    )


def _frame_relation_stage_support(
    catalogs: Sequence[FrameProposalCatalog],
    ancestry: LitePTHierarchyAncestry,
    *,
    num_points: int,
) -> dict[str, Any]:
    """Count fully-known positive/negative tokens for one physical frame."""

    device = ancestry.representative_indices.device
    relation_chunks = [
        catalog.proposal_point_indices.long()
        for catalog in catalogs
        if catalog.proposal_point_indices.numel()
    ]
    relation_points = (
        torch.unique(torch.cat(relation_chunks, dim=0), sorted=True)
        if relation_chunks
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    if relation_points.numel() and (
        int(relation_points.min()) < 0
        or int(relation_points.max()) >= int(num_points)
    ):
        raise ValueError("frame relation membership indexes outside points")
    relation_known = torch.zeros(
        int(num_points), dtype=torch.bool, device=device
    )
    relation_known[relation_points] = True
    relation_offsets, relation_members = _combined_relation_membership(catalogs)
    relation_index = build_relation_membership_index(
        relation_offsets,
        relation_members,
        point_capacity=int(num_points),
    )

    # This tokenized CSR is invariant across every proposal in the physical
    # frame.  The previous implementation rebuilt it (and its packed index)
    # once per raw proposal during ancestry-aware preselection.  Keep it in
    # the frame support record so the exact same integer representation is
    # reused by every evidence query.
    relation_token_offsets, relation_token_members = _relation_tokens_csr(
        ancestry.point_to_stage["dec0"],
        relation_offsets,
        relation_members,
        token_capacity=int(ancestry.stage_grids["dec0"].shape[0]),
    )
    relation_token_index = build_relation_membership_index(
        relation_token_offsets,
        relation_token_members,
        point_capacity=int(ancestry.stage_grids["dec0"].shape[0]),
    )

    all_counts: dict[str, torch.Tensor] = {}
    known_counts: dict[str, torch.Tensor] = {}
    for stage in LITEPT_HIERARCHY_STAGES:
        point_map = ancestry.point_to_stage[stage]
        token_count = int(ancestry.stage_grids[stage].shape[0])
        all_counts[stage] = torch.bincount(
            point_map,
            minlength=token_count,
        )
        known_counts[stage] = torch.bincount(
            point_map[relation_known],
            minlength=token_count,
        )
    return {
        "relation_known_points": relation_points,
        "relation_known_mask": relation_known,
        "relation_offsets": relation_offsets,
        "relation_members": relation_members,
        "relation_index": relation_index,
        "relation_token_offsets": relation_token_offsets,
        "relation_token_members": relation_token_members,
        "relation_token_index": relation_token_index,
        "all_counts": all_counts,
        "known_counts": known_counts,
    }


def _relation_tokens_csr(
    point_to_token: torch.Tensor,
    relation_offsets: torch.Tensor,
    relation_members: torch.Tensor,
    *,
    token_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map relation membership CSR to token CSR once, preserving sort order."""

    proposal_count = int(relation_offsets.numel()) - 1
    if proposal_count <= 0 or relation_members.numel() == 0:
        return (
            torch.zeros(1, dtype=torch.long, device=relation_offsets.device),
            torch.empty((0,), dtype=torch.long, device=relation_offsets.device),
        )
    sizes = relation_offsets[1:] - relation_offsets[:-1]
    proposal_ids = torch.repeat_interleave(
        torch.arange(
            proposal_count,
            dtype=torch.long,
            device=relation_offsets.device,
        ),
        sizes,
    )
    relation_tokens = point_to_token[relation_members.long()]
    keys = torch.unique(
        proposal_ids * int(token_capacity) + relation_tokens,
        sorted=True,
    )
    unique_proposals = torch.div(
        keys, int(token_capacity), rounding_mode="floor"
    )
    token_members = keys.remainder(int(token_capacity)).long()
    counts = torch.bincount(unique_proposals, minlength=proposal_count)
    token_offsets = torch.cat(
        [
            torch.zeros(
                1,
                dtype=torch.long,
                device=relation_offsets.device,
            ),
            counts.cumsum(dim=0),
        ]
    )
    return token_offsets, token_members


def _all_anchor_pairs_have_safe_negative_signature(
    anchor_words: torch.Tensor,
    candidate_words: torch.Tensor,
    *,
    require_candidate_membership: bool,
    candidate_signature_counts: torch.Tensor | None = None,
    block_size: int = 64,
) -> bool:
    """Check the all-pairs safety predicate without a ``[P,Q]`` mask.

    ``co_membership_safe_negative_mask`` is the exact row-level oracle, but
    materializing its ``[P,Q]`` result for every preselection proposal is the
    dominant real-scene cost.  Safety depends only on the packed relation
    signature of a candidate, so equal candidate signatures can be grouped
    once and weighted by their multiplicity.  The all-pairs result is then
    determined by weighted blocked counts and an exact signature query for
    the cardinality-hard pairs.

    The pure-positive and pure-negative token sets supplied by the caller are
    disjoint by construction.  Consequently the ``candidate != anchor`` term
    in the row-level oracle is vacuous here; zero signatures are retained only
    when candidate membership is not required, exactly matching that oracle.
    """

    if anchor_words.ndim != 2 or candidate_words.ndim != 2:
        raise ValueError("relation signatures must have shape [N,W]")
    if anchor_words.dtype != torch.int64 or candidate_words.dtype != torch.int64:
        raise TypeError("relation signatures must be int64")
    if anchor_words.device != candidate_words.device:
        raise ValueError("relation signatures must share one device")
    anchor_count, word_count = (int(value) for value in anchor_words.shape)
    candidate_count = int(candidate_words.shape[0])
    if int(candidate_words.shape[1]) != word_count:
        raise ValueError("anchor and candidate signatures disagree on word count")
    if anchor_count < 2:
        return True
    if candidate_count == 0:
        return False
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")

    if candidate_signature_counts is None:
        if require_candidate_membership:
            candidate_words = candidate_words[candidate_words.ne(0).any(dim=-1)]
            candidate_count = int(candidate_words.shape[0])
            if candidate_count == 0:
                return False
        signatures, signature_counts = torch.unique(
            candidate_words,
            dim=0,
            sorted=True,
            return_counts=True,
        )
        signature_counts = signature_counts.long()
    else:
        if candidate_signature_counts.ndim != 1:
            raise ValueError("candidate signature counts must have shape [U]")
        if candidate_signature_counts.device != candidate_words.device:
            raise ValueError("signature counts and signatures must share one device")
        if int(candidate_signature_counts.numel()) != candidate_count:
            raise ValueError("signature counts disagree with signature rows")
        if candidate_signature_counts.numel() and bool(
            (candidate_signature_counts < 0).any().item()
        ):
            raise ValueError("candidate signature counts must be non-negative")
        signatures = candidate_words
        signature_counts = candidate_signature_counts.long()
        present = signature_counts > 0
        signatures = signatures[present]
        signature_counts = signature_counts[present]
        if require_candidate_membership:
            nonempty = signatures.ne(0).any(dim=-1)
            signatures = signatures[nonempty]
            signature_counts = signature_counts[nonempty]
        candidate_count = int(signature_counts.sum().item())
        if candidate_count == 0:
            return False
    signature_count = int(signatures.shape[0])
    if signature_count == 0:
        return False

    # A zero relation signature is safe for every anchor.  It is a global
    # witness whenever membership is optional and avoids any pair work.
    if bool(signatures.eq(0).all(dim=1).any().item()):
        return True

    # Choose bounded two-dimensional tiles.  The overlap tensor is boolean
    # and has shape [anchor_tile, signature_tile, word_count]; cap it so a
    # large scene never allocates a P-by-Q temporary.
    max_overlap_elements = 2_000_000
    signature_tile = min(signature_count, 4096)
    anchor_tile = max(
        1,
        min(
            anchor_count,
            max_overlap_elements // max(1, signature_tile * word_count),
        ),
    )
    safe_counts = torch.zeros(
        anchor_count,
        dtype=torch.long,
        device=anchor_words.device,
    )
    common_signature = torch.ones(
        signature_count,
        dtype=torch.bool,
        device=anchor_words.device,
    )
    for anchor_start in range(0, anchor_count, anchor_tile):
        anchor_end = min(anchor_count, anchor_start + anchor_tile)
        anchor_block = anchor_words[anchor_start:anchor_end]
        blocked_count = torch.zeros(
            anchor_end - anchor_start,
            dtype=torch.long,
            device=anchor_words.device,
        )
        for signature_start in range(0, signature_count, signature_tile):
            signature_end = min(signature_count, signature_start + signature_tile)
            overlap = (
                anchor_block[:, None, :]
                & signatures[signature_start:signature_end][None, :, :]
            ).ne(0).any(dim=-1)
            safe = ~overlap
            blocked_count += (~safe).to(dtype=torch.long).mul(
                signature_counts[signature_start:signature_end]
            ).sum(dim=1)
            common_signature[signature_start:signature_end] &= safe.all(dim=0)
        safe_counts[anchor_start:anchor_end] = candidate_count - blocked_count

    # ``b_i+b_j<Q`` is an exact sufficient condition for every pair, where
    # b_i is the number of blocked candidates.  Checking the two largest
    # complements is enough to accept the entire family without pair lists.
    if bool((safe_counts == 0).any().item()):
        return False
    if bool(common_signature.any().item()):
        return True
    blocked_counts = candidate_count - safe_counts
    if bool(torch.topk(blocked_counts, k=2).values.sum().item() < candidate_count):
        return True

    # Generate exactly the cardinality-hard unordered pairs in sorted order.
    order = torch.argsort(safe_counts, stable=True)
    sorted_counts = safe_counts[order]
    sorted_positions = torch.arange(anchor_count, device=anchor_words.device)
    pair_ends = torch.searchsorted(
        sorted_counts,
        candidate_count - sorted_counts,
        right=True,
    )
    pair_counts = (pair_ends - sorted_positions - 1).clamp_min(0)
    pair_total = int(pair_counts.sum().item())
    if pair_total == 0:
        return True
    pair_starts = pair_counts.cumsum(dim=0) - pair_counts
    flat_pair_positions = torch.arange(pair_total, device=anchor_words.device)
    repeated_starts = torch.repeat_interleave(pair_starts, pair_counts)
    repeated_left = torch.repeat_interleave(sorted_positions + 1, pair_counts)
    right_sorted = flat_pair_positions - repeated_starts + repeated_left
    left_sorted = torch.repeat_interleave(sorted_positions, pair_counts)
    left_indices = order[left_sorted]
    right_indices = order[right_sorted]

    pair_batch_size = max(1024, int(block_size) * 64)
    for pair_start in range(0, pair_total, pair_batch_size):
        pair_end = min(pair_total, pair_start + pair_batch_size)
        pair_words = (
            anchor_words[left_indices[pair_start:pair_end]]
            | anchor_words[right_indices[pair_start:pair_end]]
        )
        pair_safe = torch.zeros(
            pair_end - pair_start,
            dtype=torch.bool,
            device=anchor_words.device,
        )
        pair_tile = max(
            1,
            min(
                signature_count,
                max_overlap_elements // max(1, pair_words.shape[0] * word_count),
            ),
        )
        for signature_start in range(0, signature_count, pair_tile):
            signature_end = min(signature_count, signature_start + pair_tile)
            overlap = (
                pair_words[:, None, :]
                & signatures[signature_start:signature_end][None, :, :]
            ).ne(0).any(dim=-1)
            pair_safe |= (~overlap).any(dim=-1)
            if bool(pair_safe.all().item()):
                break
        if not bool(pair_safe.all().item()):
            return False
    return True


def _sampled_anchor_pairs_have_safe_negative_signature(
    pair_tokens: torch.Tensor,
    relation_words: torch.Tensor,
    candidate_words: torch.Tensor,
    *,
    require_candidate_membership: bool,
    candidate_signature_counts: torch.Tensor | None = None,
) -> bool:
    """Check at most 64 sampled token pairs against grouped candidates.

    This is the versioned preselection policy's representative check.  The
    final point/token rebuild remains authoritative and may apply its own
    strict survival gate; this helper only avoids scanning every theoretical
    pure-token pair during catalog coverage.
    """

    if pair_tokens.ndim != 2 or int(pair_tokens.shape[1]) != 2:
        raise ValueError("sampled token pairs must have shape [P,2]")
    if pair_tokens.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("sampled token pairs must be integral")
    if relation_words.ndim != 2 or candidate_words.ndim != 2:
        raise ValueError("relation signatures must have shape [N,W]")
    if relation_words.dtype != torch.int64 or candidate_words.dtype != torch.int64:
        raise TypeError("relation signatures must be int64")
    if not (
        pair_tokens.device == relation_words.device == candidate_words.device
    ):
        raise ValueError("sampled-pair inputs must share one device")
    if pair_tokens.numel() == 0:
        return False
    if int(pair_tokens.min()) < 0 or int(pair_tokens.max()) >= int(
        relation_words.shape[0]
    ):
        raise ValueError("sampled token pair indexes outside relation signatures")
    if int(candidate_words.shape[1]) != int(relation_words.shape[1]):
        raise ValueError("anchor and candidate signatures disagree on word count")

    if candidate_signature_counts is None:
        if require_candidate_membership:
            candidate_words = candidate_words[candidate_words.ne(0).any(dim=-1)]
        signatures, _counts = torch.unique(
            candidate_words,
            dim=0,
            sorted=True,
            return_counts=True,
        )
    else:
        if candidate_signature_counts.ndim != 1:
            raise ValueError("candidate signature counts must have shape [U]")
        if candidate_signature_counts.device != candidate_words.device:
            raise ValueError("signature counts and signatures must share one device")
        if int(candidate_signature_counts.numel()) != int(candidate_words.shape[0]):
            raise ValueError("signature counts disagree with signature rows")
        counts = candidate_signature_counts.long()
        present = counts > 0
        signatures = candidate_words[present]
        counts = counts[present]
        if require_candidate_membership:
            nonempty = signatures.ne(0).any(dim=-1)
            signatures = signatures[nonempty]
            counts = counts[nonempty]
    if int(signatures.shape[0]) == 0:
        return False

    # Candidate multiplicities do not affect existence.  Keep the query
    # bounded by the sampled-pair budget and the number of unique signatures.
    pair_words = (
        relation_words[pair_tokens[:, 0].long()]
        | relation_words[pair_tokens[:, 1].long()]
    )
    pair_safe = torch.zeros(
        int(pair_tokens.shape[0]), dtype=torch.bool, device=pair_tokens.device
    )
    max_elements = 2_000_000
    word_count = max(1, int(relation_words.shape[1]))
    signature_tile = max(
        1,
        min(
            int(signatures.shape[0]),
            max_elements // max(1, int(pair_words.shape[0]) * word_count),
        ),
    )
    for start in range(0, int(signatures.shape[0]), signature_tile):
        end = min(int(signatures.shape[0]), start + signature_tile)
        overlap = (
            pair_words[:, None, :]
            & signatures[start:end][None, :, :]
        ).ne(0).any(dim=-1)
        pair_safe |= (~overlap).any(dim=-1)
        if bool(pair_safe.all().item()):
            return True
    return bool(pair_safe.all().item())


def _all_pure_dec0_pairs_have_relation_safe_negative(
    *,
    members: torch.Tensor,
    visible: torch.Tensor,
    ancestry: LitePTHierarchyAncestry,
    frame_support: Mapping[str, Any],
    require_negative_proposal_membership: bool,
    objective_known_counts: torch.Tensor | None = None,
    precomputed_present_tokens: torch.Tensor | None = None,
    precomputed_present_point_counts: torch.Tensor | None = None,
    precomputed_full_known_tokens: torch.Tensor | None = None,
    precomputed_full_known_mask: torch.Tensor | None = None,
    precomputed_negative_signatures: torch.Tensor | None = None,
    precomputed_negative_signature_counts: torch.Tensor | None = None,
    sampled_positive_pair_count: int | None = None,
    sampled_positive_pair_generator: torch.Generator | None = None,
    sampled_positive_pair_output: list[torch.Tensor] | None = None,
) -> bool:
    """Require every possible pure-token pair to retain a safe negative.

    In the legacy exact mode, the deferred token rebuild draws a deterministic
    subset of unordered pure-dec0 token pairs, so merely finding one viable
    pair is insufficient.  The versioned preselection mode supplies a
    deterministic capped pair stream and checks exactly those representative
    pairs; the final token rebuild remains authoritative for its selected
    proposal occurrence.
    """

    if int(members.numel()) < 2:
        return False
    point_to_dec0 = ancestry.point_to_stage["dec0"]
    all_counts = frame_support["all_counts"]["dec0"]
    if objective_known_counts is None:
        # ``frame_support`` is the union relation over every same-frame cell,
        # but the downstream sampler can only construct negatives from this
        # catalog's own visibility universe.  Keep the exact same
        # intersection here; using the frame union alone would accept a
        # proposal whose only nominal negatives live outside the candidate
        # frame's visible points.
        candidate_known_mask = torch.zeros_like(
            frame_support["relation_known_mask"], dtype=torch.bool
        )
        candidate_known_mask[visible] = frame_support["relation_known_mask"][visible]
        # With the default point-space eligibility policy, the token rebuild
        # sees every candidate-visible point as known.  Membership-required
        # batches use the stricter visible∩relation-known mask.
        objective_known_mask = (
            candidate_known_mask
            if require_negative_proposal_membership
            else candidate_known_mask.clone()
        )
        if not require_negative_proposal_membership:
            objective_known_mask.zero_()
            objective_known_mask[visible] = True
        known_counts = torch.bincount(
            point_to_dec0[objective_known_mask],
            minlength=int(all_counts.numel()),
        )
    else:
        known_counts = objective_known_counts
    if precomputed_full_known_mask is None:
        fully_known = (all_counts > 0) & (known_counts == all_counts)
    else:
        if precomputed_full_known_mask.ndim != 1:
            raise ValueError("precomputed full-known mask must be one-dimensional")
        if int(precomputed_full_known_mask.numel()) != int(all_counts.numel()):
            raise ValueError("precomputed full-known mask disagrees with dec0 tokens")
        fully_known = precomputed_full_known_mask
    cached_negative_signatures = (
        precomputed_negative_signatures is not None
        and precomputed_negative_signature_counts is not None
    )
    if cached_negative_signatures and (
        precomputed_present_tokens is not None
        and precomputed_present_point_counts is not None
    ):
        # ``fully_known`` is already a token-indexed mask.  Indexing it by the
        # proposal's present tokens is O(P); constructing the equivalent
        # ``torch.isin(full_known_tokens, present_tokens)`` scans the entire
        # catalog for every proposal and was the real CPU hot spot.
        present_tokens = precomputed_present_tokens.long()
        present_point_counts = precomputed_present_point_counts.long()
        pure_tokens = present_tokens[
            fully_known[present_tokens]
            & (present_point_counts == all_counts[present_tokens])
        ]
        # The cached signature multiplicities represent exactly
        # ``full_known - present``; no candidate token ID list is needed.
        pure_negative_tokens = None
    elif (
        precomputed_present_tokens is not None
        and precomputed_present_point_counts is not None
        and precomputed_full_known_tokens is not None
    ):
        present_tokens = precomputed_present_tokens.long()
        present_point_counts = precomputed_present_point_counts.long()
        pure_tokens = present_tokens[
            fully_known[present_tokens]
            & (present_point_counts == all_counts[present_tokens])
        ]
        pure_negative_tokens = precomputed_full_known_tokens[
            ~torch.isin(precomputed_full_known_tokens, present_tokens)
        ]
    else:
        proposal_counts = torch.bincount(
            point_to_dec0[members],
            minlength=int(all_counts.numel()),
        )
        pure_tokens = torch.where(
            fully_known & (proposal_counts == all_counts) & (proposal_counts > 0)
        )[0].long()
        pure_negative_tokens = torch.where(
            fully_known & (proposal_counts == 0)
        )[0].long()
    if pure_tokens.numel() < 2:
        return False
    if pure_negative_tokens is not None and pure_negative_tokens.numel() == 0:
        return False

    # Tokenize the same full relation CSR used by the later token rebuild.
    # Point-level safety is weaker when one dec0 token has multiple points: a
    # different point in that token can still share a teacher proposal.
    relation_index = frame_support["relation_token_index"]
    if precomputed_negative_signatures is not None:
        if precomputed_negative_signature_counts is None:
            raise ValueError("precomputed negative signature counts are required")
        candidate_signatures = precomputed_negative_signatures
        candidate_signature_counts = precomputed_negative_signature_counts
    else:
        candidate_signatures = relation_index.point_words[pure_negative_tokens]
        candidate_signature_counts = None
    if sampled_positive_pair_count is not None:
        if int(sampled_positive_pair_count) <= 0:
            return False
        if sampled_positive_pair_generator is None:
            raise ValueError("sampled positive-pair generator is required")
        sampled_pairs = sample_unique_positive_pairs(
            pure_tokens,
            min(int(sampled_positive_pair_count), 64),
            generator=sampled_positive_pair_generator,
        )
        if sampled_positive_pair_output is not None:
            sampled_positive_pair_output.append(sampled_pairs.detach())
        return _sampled_anchor_pairs_have_safe_negative_signature(
            sampled_pairs,
            relation_index.point_words,
            candidate_signatures,
            require_candidate_membership=require_negative_proposal_membership,
            candidate_signature_counts=candidate_signature_counts,
        )
    return _all_anchor_pairs_have_safe_negative_signature(
        relation_index.point_words[pure_tokens],
        candidate_signatures,
        require_candidate_membership=require_negative_proposal_membership,
        candidate_signature_counts=candidate_signature_counts,
    )


def _all_row_pair_intersections_nonempty(
    rows: torch.Tensor,
    *,
    block_size: int = 64,
) -> bool:
    """Check every unordered row pair for a shared true column.

    The preselection predicate needs ``(rows[i] & rows[j]).any()`` for every
    ``i < j``.  A dense ``[N,N,W]`` broadcast is prohibitively expensive for
    real scenes.  First use exact witnesses and the cardinality bound
    ``|A|+|B| > Q => A∩B != ∅`` to discard pairs that cannot fail.  Only the
    remaining pairs are checked in bounded batches using the same signed-safe
    63-bit packing as the relation index.  The result is exact; the reduction
    only changes which pairs need to be materialized.
    """

    if rows.ndim != 2:
        raise ValueError("rows must have shape [N,Q]")
    if rows.dtype != torch.bool:
        raise TypeError("rows must be boolean")
    row_count, column_count = (int(value) for value in rows.shape)
    if row_count < 2:
        return True
    if column_count == 0:
        return False
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")

    # A pair whose row cardinalities sum to more than the universe size also
    # has a guaranteed intersection.  The two largest complements therefore
    # give an O(N) exact acceptance test before any pair indexing.  This is the
    # common path for sparse relation graphs with many safe negatives.
    safe_counts = rows.sum(dim=1, dtype=torch.long)
    if bool((safe_counts == 0).any().item()):
        return False
    if bool(rows.all(dim=0).any().item()):
        return True
    unsafe_counts = column_count - safe_counts
    if bool(torch.topk(unsafe_counts, k=2).values.sum().item() < column_count):
        return True

    # Otherwise work in sorted cardinality order so the exact hard-pair list
    # can be generated without a Python loop over rows.
    order = torch.argsort(safe_counts, stable=True)
    sorted_counts = safe_counts[order]
    sorted_positions = torch.arange(row_count, device=rows.device)
    pair_ends = torch.searchsorted(
        sorted_counts,
        column_count - sorted_counts,
        right=True,
    )
    pair_counts = (pair_ends - sorted_positions - 1).clamp_min(0)
    pair_total = int(pair_counts.sum().item())
    if pair_total == 0:
        return True

    # The hard pairs are the sorted positions i<j with
    # safe_counts[i] + safe_counts[j] <= column_count.  Construct their
    # indices by arithmetic on one flattened range, avoiding per-row Python
    # reductions and avoiding an O(N^2) boolean matrix.
    pair_starts = pair_counts.cumsum(dim=0) - pair_counts
    flat_pair_positions = torch.arange(pair_total, device=rows.device)
    repeated_starts = torch.repeat_interleave(pair_starts, pair_counts)
    repeated_left = torch.repeat_interleave(
        sorted_positions + 1,
        pair_counts,
    )
    right_sorted = flat_pair_positions - repeated_starts + repeated_left
    left_sorted = torch.repeat_interleave(sorted_positions, pair_counts)
    left_indices = order[left_sorted]
    right_indices = order[right_sorted]

    # The smallest-cardinality pair is the most likely failing witness.  Test
    # it directly before allocating the packed hard-row representation.
    if not bool((rows[left_indices[0]] & rows[right_indices[0]]).any().item()):
        return False

    # Pack only rows participating in a hard pair.  Keeping the pair stream
    # separate bounds temporary memory even when a pathological catalog has
    # many hard pairs.
    hard_indices = torch.unique(
        torch.cat((left_indices, right_indices)),
        sorted=True,
    )
    hard_rows = rows[hard_indices]
    hard_count = int(hard_indices.numel())
    hard_lookup = torch.full(
        (row_count,),
        -1,
        dtype=torch.long,
        device=rows.device,
    )
    hard_lookup[hard_indices] = torch.arange(
        hard_count,
        dtype=torch.long,
        device=rows.device,
    )
    left_indices = hard_lookup[left_indices]
    right_indices = hard_lookup[right_indices]

    words = (column_count + 62) // 63
    padded_columns = words * 63 - column_count
    packed_rows = hard_rows.to(dtype=torch.int64)
    if padded_columns:
        packed_rows = torch.cat(
            [
                packed_rows,
                torch.zeros(
                    (hard_count, padded_columns),
                    dtype=torch.int64,
                    device=rows.device,
                ),
            ],
            dim=1,
        )
    packed_rows = packed_rows.reshape(hard_count, words, 63)
    shifts = torch.arange(
        63,
        dtype=torch.int64,
        device=rows.device,
    )
    packed_rows = (packed_rows << shifts).sum(dim=2)

    # Check a stream of pairs rather than a dense pair-by-pair broadcast.
    # ``block_size`` remains the public tuning knob; scale it to a modest
    # number of pairs while keeping the old positive-value validation.
    pair_batch_size = max(1024, int(block_size) * 64)
    for pair_start in range(0, pair_total, pair_batch_size):
        pair_end = min(pair_total, pair_start + pair_batch_size)
        pair_has_shared_column = (
            packed_rows[left_indices[pair_start:pair_end]]
            & packed_rows[right_indices[pair_start:pair_end]]
        ).any(dim=-1)
        if not bool(pair_has_shared_column.all().item()):
            return False
    return True


def _unique_catalog_proposal_members(
    catalog: FrameProposalCatalog,
    *,
    point_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one canonical unique-member CSR for every catalog proposal."""

    offsets = catalog.proposal_offsets.long()
    members = catalog.proposal_point_indices.long()
    proposal_count = int(offsets.numel()) - 1
    if proposal_count < 0:
        raise ValueError("proposal offsets must contain at least one entry")
    if proposal_count == 0 or members.numel() == 0:
        return (
            torch.zeros(
                proposal_count + 1,
                dtype=torch.long,
                device=offsets.device,
            ),
            torch.empty((0,), dtype=torch.long, device=offsets.device),
        )
    if int(point_capacity) <= int(members.max()):
        raise ValueError("point_capacity does not cover catalog members")
    sizes = offsets[1:] - offsets[:-1]
    proposal_ids = torch.repeat_interleave(
        torch.arange(
            proposal_count,
            dtype=torch.long,
            device=offsets.device,
        ),
        sizes,
    )
    keys = torch.unique(
        proposal_ids * int(point_capacity) + members,
        sorted=True,
    )
    unique_proposals = torch.div(
        keys,
        int(point_capacity),
        rounding_mode="floor",
    )
    unique_members = keys.remainder(int(point_capacity)).long()
    unique_counts = torch.bincount(
        unique_proposals,
        minlength=proposal_count,
    )
    unique_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=offsets.device),
            unique_counts.cumsum(dim=0),
        ]
    )
    return unique_offsets, unique_members


def _catalog_preselection_context(
    *,
    catalog: FrameProposalCatalog,
    ancestry: LitePTHierarchyAncestry,
    frame_support: Mapping[str, Any],
    num_points: int,
) -> dict[str, Any]:
    """Cache proposal-invariant planning tensors for one catalog."""

    visible = torch.unique(catalog.visible_indices.long(), sorted=True)
    relation_known = frame_support["relation_known_mask"]
    visible_relation_known = relation_known[visible]
    known_points = visible[visible_relation_known]
    known_counts: dict[str, torch.Tensor] = {}
    visible_counts: dict[str, torch.Tensor] = {}
    for stage in LITEPT_HIERARCHY_STAGES:
        point_map = ancestry.point_to_stage[stage]
        token_count = int(frame_support["all_counts"][stage].numel())
        known_counts[stage] = torch.bincount(
            point_map[known_points],
            minlength=token_count,
        )
        visible_counts[stage] = torch.bincount(
            point_map[visible],
            minlength=token_count,
        )
    unique_offsets, unique_members = _unique_catalog_proposal_members(
        catalog,
        point_capacity=int(num_points),
    )
    proposal_count = int(unique_offsets.numel()) - 1
    unique_sizes = unique_offsets[1:] - unique_offsets[:-1]
    proposal_ids = torch.repeat_interleave(
        torch.arange(
            proposal_count,
            dtype=torch.long,
            device=unique_offsets.device,
        ),
        unique_sizes,
    )
    proposal_stage_positive_counts: dict[str, torch.Tensor] = {}
    proposal_stage_negative_counts: dict[str, torch.Tensor] = {}
    dec0_present_tokens = torch.empty(
        (0,), dtype=torch.long, device=unique_offsets.device
    )
    dec0_present_point_counts = torch.empty(
        (0,), dtype=torch.long, device=unique_offsets.device
    )
    dec0_present_offsets = torch.zeros(
        proposal_count + 1,
        dtype=torch.long,
        device=unique_offsets.device,
    )
    for stage in LITEPT_HIERARCHY_STAGES:
        all_counts = frame_support["all_counts"][stage]
        token_count = int(all_counts.numel())
        if unique_members.numel() == 0:
            proposal_stage_positive_counts[stage] = torch.zeros(
                proposal_count,
                dtype=torch.long,
                device=unique_offsets.device,
            )
            proposal_stage_negative_counts[stage] = torch.full(
                (proposal_count,),
                int(((all_counts > 0) & (
                    known_counts[stage] == all_counts
                )).sum()),
                dtype=torch.long,
                device=unique_offsets.device,
            )
            continue
        token_ids = ancestry.point_to_stage[stage][unique_members]
        token_keys = proposal_ids * token_count + token_ids
        sorted_keys = torch.sort(token_keys).values
        unique_token_keys, point_counts = torch.unique_consecutive(
            sorted_keys,
            return_counts=True,
        )
        unique_proposals = torch.div(
            unique_token_keys,
            token_count,
            rounding_mode="floor",
        )
        unique_tokens = unique_token_keys.remainder(token_count).long()
        fully_known = (all_counts > 0) & (
            known_counts[stage] == all_counts
        )
        positive_rows = fully_known[unique_tokens] & (
            point_counts == all_counts[unique_tokens]
        )
        proposal_stage_positive_counts[stage] = torch.bincount(
            unique_proposals[positive_rows],
            minlength=proposal_count,
        ).long()
        present_fully_known_rows = fully_known[unique_tokens]
        present_fully_known_counts = torch.bincount(
            unique_proposals[present_fully_known_rows],
            minlength=proposal_count,
        ).long()
        proposal_stage_negative_counts[stage] = (
            int(fully_known.sum()) - present_fully_known_counts
        ).long()
        if stage == "dec0":
            dec0_present_tokens = unique_tokens
            dec0_present_point_counts = point_counts
            dec0_present_offsets = torch.cat(
                [
                    torch.zeros(
                        1,
                        dtype=torch.long,
                        device=unique_offsets.device,
                    ),
                    torch.bincount(
                        unique_proposals,
                        minlength=proposal_count,
                    ).cumsum(dim=0),
                ]
            )
    dec0_all_counts = frame_support["all_counts"]["dec0"]
    dec0_full_known_tokens = {
        "relation": torch.where(
            (dec0_all_counts > 0)
            & (known_counts["dec0"] == dec0_all_counts)
        )[0].long(),
        "visible": torch.where(
            (dec0_all_counts > 0)
            & (visible_counts["dec0"] == dec0_all_counts)
        )[0].long(),
    }
    # Candidate safety depends on relation signatures, not on the proposal
    # that is currently being inspected.  Group the frame's full-known token
    # signatures once; each proposal only subtracts its present token counts
    # from this compact catalog instead of rebuilding a [P,Q] mask.
    relation_token_words = frame_support["relation_token_index"].point_words
    negative_signature_cache: dict[str, dict[str, torch.Tensor]] = {}
    for mode, full_known_tokens in dec0_full_known_tokens.items():
        full_words = relation_token_words[full_known_tokens]
        if full_words.numel() == 0:
            negative_signature_cache[mode] = {
                "signatures": torch.empty(
                    (0, int(relation_token_words.shape[1])),
                    dtype=torch.int64,
                    device=relation_token_words.device,
                ),
                "full_counts": torch.empty(
                    (0,), dtype=torch.long, device=relation_token_words.device
                ),
                "token_signature_ids": torch.full(
                    (int(relation_token_words.shape[0]),),
                    -1,
                    dtype=torch.long,
                    device=relation_token_words.device,
                ),
            }
            continue
        signatures, inverse = torch.unique(
            full_words,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        token_signature_ids = torch.full(
            (int(relation_token_words.shape[0]),),
            -1,
            dtype=torch.long,
            device=relation_token_words.device,
        )
        token_signature_ids[full_known_tokens] = inverse.long()
        negative_signature_cache[mode] = {
            "signatures": signatures,
            "full_counts": torch.bincount(
                inverse.long(), minlength=int(signatures.shape[0])
            ).long(),
            "token_signature_ids": token_signature_ids,
        }
    return {
        "visible": visible,
        "relation_known_counts": known_counts,
        "visible_counts": visible_counts,
        "unique_proposal_offsets": unique_offsets,
        "unique_proposal_members": unique_members,
        "proposal_stage_positive_counts": proposal_stage_positive_counts,
        "proposal_stage_negative_counts": proposal_stage_negative_counts,
        "dec0_present_offsets": dec0_present_offsets,
        "dec0_present_tokens": dec0_present_tokens,
        "dec0_present_point_counts": dec0_present_point_counts,
        "dec0_full_known_tokens": dec0_full_known_tokens,
        "negative_signature_cache": negative_signature_cache,
    }


def _proposal_preselection_evidence(
    *,
    catalog: FrameProposalCatalog,
    proposal_index: int,
    granularity: str,
    ancestry: LitePTHierarchyAncestry,
    frame_support: Mapping[str, Any],
    require_negative_proposal_membership: bool,
    require_hierarchy_routes: bool = True,
    catalog_context: Mapping[str, Any] | None = None,
    sampled_positive_pair_count: int | None = None,
    sampled_positive_pair_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Return label-free representability evidence for one raw catalog row."""

    if catalog_context is None:
        start = int(catalog.proposal_offsets[proposal_index].item())
        end = int(catalog.proposal_offsets[proposal_index + 1].item())
        members = torch.unique(
            catalog.proposal_point_indices[start:end].long(), sorted=True
        )
        visible = torch.unique(catalog.visible_indices.long(), sorted=True)
    else:
        unique_offsets = catalog_context["unique_proposal_offsets"]
        unique_members = catalog_context["unique_proposal_members"]
        start = int(unique_offsets[proposal_index])
        end = int(unique_offsets[proposal_index + 1])
        members = unique_members[start:end]
        visible = catalog_context["visible"]
        dec0_present_offsets = catalog_context["dec0_present_offsets"]
        dec0_present_start = int(dec0_present_offsets[proposal_index])
        dec0_present_end = int(dec0_present_offsets[proposal_index + 1])
    precomputed_negative_signatures = None
    precomputed_negative_signature_counts = None
    precomputed_full_known_mask = None
    if catalog_context is not None:
        signature_mode = (
            "relation" if require_negative_proposal_membership else "visible"
        )
        signature_cache = catalog_context.get("negative_signature_cache")
        if signature_cache is not None:
            signature_info = signature_cache[signature_mode]
            precomputed_negative_signatures = signature_info["signatures"]
            precomputed_full_known_mask = signature_info["token_signature_ids"] >= 0
            present_signature_ids = signature_info["token_signature_ids"][
                catalog_context["dec0_present_tokens"][
                    dec0_present_start:dec0_present_end
                ]
            ]
            present_signature_ids = present_signature_ids[present_signature_ids >= 0]
            present_signature_counts = torch.bincount(
                present_signature_ids,
                minlength=int(precomputed_negative_signatures.shape[0]),
            ).long()
            precomputed_negative_signature_counts = (
                signature_info["full_counts"] - present_signature_counts
            )
    point_to_stage = ancestry.point_to_stage
    if catalog_context is None:
        candidate_known_mask = frame_support["relation_known_mask"].clone()
        candidate_known_mask.zero_()
        candidate_known_mask[visible] = frame_support["relation_known_mask"][visible]
    stage_support: dict[str, dict[str, Any]] = {}
    for stage in LITEPT_HIERARCHY_STAGES:
        all_counts = frame_support["all_counts"][stage]
        # Relation-known support from another granularity is useful only when
        # the candidate catalog can actually see that point.  This mirrors
        # the later sampler's ``visible ∩ relation`` universe and prevents a
        # frame with a narrower visibility mask from claiming fully-known
        # tokens using support visible only in a sibling cell.
        if catalog_context is None:
            known_counts = torch.bincount(
                point_to_stage[stage][candidate_known_mask],
                minlength=int(all_counts.numel()),
            )
            proposal_counts = torch.bincount(
                point_to_stage[stage][members],
                minlength=int(all_counts.numel()),
            )
            fully_known = (all_counts > 0) & (known_counts == all_counts)
            pure_positive = fully_known & (proposal_counts == all_counts) & (
                proposal_counts > 0
            )
            pure_negative = fully_known & (proposal_counts == 0)
            positive_token_count = int(pure_positive.sum().item())
            negative_token_count = int(pure_negative.sum().item())
        else:
            known_counts = catalog_context["relation_known_counts"][stage]
            positive_token_count = int(
                catalog_context["proposal_stage_positive_counts"][stage][
                    proposal_index
                ]
            )
            negative_token_count = int(
                catalog_context["proposal_stage_negative_counts"][stage][
                    proposal_index
                ]
            )
        stage_support[stage] = {
            "positive_token_count": positive_token_count,
            "negative_token_count": negative_token_count,
            "positive_support": positive_token_count > 0,
            "negative_support": negative_token_count > 0,
            "valid": positive_token_count > 0 and negative_token_count > 0,
        }

    pure_dec0_count = int(
        stage_support["dec0"]["positive_token_count"]
    )
    reasons: list[str] = []
    if pure_dec0_count < 2:
        reasons.append("fewer_than_two_fully_known_pure_dec0_tokens")
    if require_hierarchy_routes:
        for stage in _STATIC_ROUTE_STAGES[granularity]:
            if not bool(stage_support[stage]["valid"]):
                reasons.append(f"missing_positive_or_negative_{stage}_support")
    dec2_valid = bool(stage_support["dec2"]["valid"])
    dec1_valid = bool(stage_support["dec1"]["valid"])
    if require_hierarchy_routes and granularity == "g02" and not dec1_valid:
        reasons.append("missing_valid_dec1_fallback_route")
    # A proposal that already lacks two pure dec0 positives or any negative
    # support cannot satisfy the relation predicate.  Avoid rebuilding its
    # token membership query; this is an exact fail-closed shortcut and keeps
    # obviously invalid catalog rows out of the hot safety path.
    sampled_positive_pairs: list[torch.Tensor] = []
    if (
        pure_dec0_count < 2
        or not bool(stage_support["dec0"]["negative_support"])
    ):
        relation_safe = False
    else:
        relation_safe = _all_pure_dec0_pairs_have_relation_safe_negative(
            members=members,
            visible=visible,
            ancestry=ancestry,
            frame_support=frame_support,
            require_negative_proposal_membership=require_negative_proposal_membership,
            objective_known_counts=(
                catalog_context["relation_known_counts"]["dec0"]
                if require_negative_proposal_membership and catalog_context is not None
                else catalog_context["visible_counts"]["dec0"]
                if catalog_context is not None
                else None
            ),
            precomputed_present_tokens=(
                catalog_context["dec0_present_tokens"][
                    dec0_present_start:dec0_present_end
                ]
                if catalog_context is not None
                else None
            ),
            precomputed_present_point_counts=(
                catalog_context["dec0_present_point_counts"][
                    dec0_present_start:dec0_present_end
                ]
                if catalog_context is not None
                else None
            ),
            precomputed_full_known_tokens=(
                catalog_context["dec0_full_known_tokens"][
                    "relation" if require_negative_proposal_membership else "visible"
                ]
                if catalog_context is not None
                else None
            ),
            precomputed_full_known_mask=precomputed_full_known_mask,
            precomputed_negative_signatures=precomputed_negative_signatures,
            precomputed_negative_signature_counts=precomputed_negative_signature_counts,
            sampled_positive_pair_count=sampled_positive_pair_count,
            sampled_positive_pair_generator=sampled_positive_pair_generator,
            sampled_positive_pair_output=sampled_positive_pairs,
        )
    if not relation_safe:
        reasons.append("no_relation_safe_negative_for_two_pure_dec0_endpoints")
    preferred_route_stage = (
        "dec2" if granularity == "g02" and dec2_valid else "dec1"
    )
    return {
        "proposal_member_count": int(members.numel()),
        "pure_dec0_token_count": pure_dec0_count,
        "fully_known_pure_dec0_token_count": pure_dec0_count,
        "stage_support": stage_support,
        "hierarchy_route_eligibility_required": bool(require_hierarchy_routes),
        "preferred_route_stage": preferred_route_stage,
        "relation_safe_positive_pair_exists": bool(relation_safe),
        "relation_safe_negative_exists": bool(relation_safe),
        # Kept out of the JSON-facing evidence row by the planner.  The
        # planner retains this native-dec0 pair tensor by raw catalog position
        # and transports it only for selected frame-group occurrences.
        "token_positive_pairs": (
            sampled_positive_pairs[0]
            if sampled_positive_pairs
            else None
        ),
        "eligible": not reasons,
        "reason": None if not reasons else ";".join(reasons),
    }


def _coverage_minimum_cursor_before_epoch(
    *,
    epoch: int,
    cell: str,
    population_size: int,
    total_proposals: int,
    cells: Sequence[str],
) -> int:
    """Return a lower-bound cursor, not the ancestry-aware exact cursor.

    Pre-selection may inspect rejected raw entries, so the exact cursor after
    epoch zero is data-dependent.  The planner only uses this quota-derived
    value as a sanity lower bound; every epoch after zero requires the
    persisted post-scan state.
    """

    return sum(
        min(
            rotating_three_cell_quotas(
                previous,
                total_proposals=total_proposals,
                cells=cells,
            )[cell],
            population_size,
        )
        for previous in range(int(epoch))
    )


def _combined_relation_membership(
    catalogs: Sequence[FrameProposalCatalog],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not catalogs:
        raise ValueError("at least one relation catalog is required")
    device = catalogs[0].proposal_offsets.device
    proposal_ids: list[torch.Tensor] = []
    member_chunks: list[torch.Tensor] = []
    proposal_base = 0
    point_capacity = 0
    for catalog in catalogs:
        if (
            catalog.proposal_offsets.device != device
            or catalog.proposal_point_indices.device != device
        ):
            raise ValueError("relation catalogs must share one device")
        offsets = catalog.proposal_offsets.long()
        members = catalog.proposal_point_indices.long()
        proposal_count = int(offsets.numel()) - 1
        if members.numel():
            point_capacity = max(point_capacity, int(members.max().item()) + 1)
        if proposal_count:
            sizes = offsets[1:] - offsets[:-1]
            proposal_ids.append(
                torch.repeat_interleave(
                    torch.arange(
                        proposal_count,
                        dtype=torch.long,
                        device=device,
                    )
                    + proposal_base,
                    sizes,
                )
            )
            member_chunks.append(members)
        proposal_base += proposal_count
    if proposal_base == 0:
        return (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    capacity = max(point_capacity, 1)
    all_proposal_ids = torch.cat(proposal_ids, dim=0)
    all_members = torch.cat(member_chunks, dim=0)
    keys = torch.unique(
        all_proposal_ids * capacity + all_members,
        sorted=True,
    )
    unique_proposals = torch.div(keys, capacity, rounding_mode="floor")
    unique_members = keys.remainder(capacity).long()
    counts = torch.bincount(unique_proposals, minlength=proposal_base)
    relation_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            counts.cumsum(dim=0),
        ]
    )
    return (
        relation_offsets,
        unique_members,
    )


def sample_multigranular_frame_group_plan(
    *,
    scene_id: str,
    epoch: int,
    seed: int,
    points: torch.Tensor,
    frames_by_cell: Mapping[str, Sequence[FrameProposalCatalog]],
    total_proposal_quota: int = 16,
    cells: Sequence[str] = DEFAULT_2D_CELLS,
    positive_pairs_per_proposal: int = 64,
    num_uniform_negatives: int = 256,
    num_spatial_hard_negatives: int = 128,
    spatial_candidate_pool: int = 1024,
    num_feature_hard_negatives: int = 128,
    feature_candidate_pool: int = 1024,
    require_negative_proposal_membership: bool = False,
    require_hierarchy_routes: bool = True,
    operation: str = "multigranular-train-v2",
    coverage_states: Mapping[str, DeterministicCoverageState | None] | None = None,
    quota_epoch: int | None = None,
    planning_device: str | torch.device | None = None,
    timing: dict[str, Any] | None = None,
    materialize_full_preselection_evidence: bool = False,
) -> MultigranularFrameGroupPlan:
    """Build deterministic, ordered, frame-local groups for one scene visit.

    Proposal coverage is computed over every valid ``(frame, proposal)`` entry
    in a cell.  The returned order is cell order followed by first occurrence
    of a selected frame in that cell's coverage slice.  Same-frame masks from
    all cells form the co-membership relation graph used to cancel ambiguous
    negatives.

    ``quota_epoch`` is normally the same as ``epoch``.  Fixed evaluation uses
    a stable per-scene quota rotation while retaining a fresh, epoch-zero
    coverage queue; separating those concepts avoids pretending that an
    unrelated training cursor exists for that fixed pack.

    ``require_hierarchy_routes`` remains enabled for every training visit: it
    prefilters g02/g05/g08 rows against the routed query-mask stages.  The
    fixed dec0 feature evaluator disables only that extra criterion because it
    never computes a hierarchy-mask loss; its native token pairs and
    ambiguity-safe negatives remain identical V2 contract checks.

    ``materialize_full_preselection_evidence`` is an audit-only switch.  The
    training path evaluates the ancestry predicate lazily only until its quota
    is filled; an offline coverage audit needs the complete raw population to
    establish the representable denominator.  Enabling this switch fills the
    same deterministic per-position cache after selection without changing the
    selected rows, cursor, or any training semantics.
    """

    ordered_cells = tuple(str(cell) for cell in cells)
    if set(frames_by_cell) != set(ordered_cells):
        raise ValueError("frames_by_cell must contain exactly the requested cells")
    if (
        not scene_id
        or int(epoch) < 0
        or int(seed) < 0
        or not operation
        or (quota_epoch is not None and int(quota_epoch) < 0)
    ):
        raise ValueError("scene, non-negative epoch/seed, and operation are required")
    if coverage_states is not None and not set(coverage_states).issubset(
        set(ordered_cells)
    ):
        raise ValueError("coverage_states contains an unknown cell")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    def phase_start() -> float | None:
        if timing is None:
            return None
        if points.device.type == "cuda":
            torch.cuda.synchronize(points.device)
        return time.perf_counter()

    def phase_end(name: str, start: float | None) -> None:
        if timing is None or start is None:
            return
        if points.device.type == "cuda":
            torch.cuda.synchronize(points.device)
        timing[name] = 1000.0 * (time.perf_counter() - start)

    ancestry_start = phase_start()
    # Keep voxelization itself on the caller's device.  This preserves the
    # exact float32 floor/unique ancestry used by the LitePT wrapper.  Integer
    # planning below is then moved to CPU, where the many scalar checks do not
    # become hundreds of CUDA launches and host synchronizations.
    ancestry = derive_litept_hierarchy_ancestry(points)
    phase_end("ancestry_ms", ancestry_start)

    requested_planning_device = (
        torch.device(planning_device)
        if planning_device is not None
        else torch.device("cpu") if points.device.type == "cuda" else points.device
    )
    if requested_planning_device.type == "cuda" and requested_planning_device.index is None:
        requested_planning_device = points.device
    if timing is not None:
        timing["planning_device"] = str(requested_planning_device)
        timing["cpu_integer_planning"] = bool(
            requested_planning_device.type == "cpu"
            and points.device.type == "cuda"
        )
    if requested_planning_device.type == "cuda" and requested_planning_device != points.device:
        raise ValueError("planning_device must match points when it is CUDA")
    planning_transfer_start = phase_start()
    planning_ancestry = ancestry
    planning_frames_by_cell: Mapping[str, Sequence[FrameProposalCatalog]] = frames_by_cell
    if requested_planning_device != points.device:
        if requested_planning_device.type != "cpu":
            raise ValueError("only CPU planning is supported for a CUDA scene")
        planning_ancestry = LitePTHierarchyAncestry(
            point_to_stage={
                stage: values.detach().cpu()
                for stage, values in ancestry.point_to_stage.items()
            },
            stage_grids={
                stage: values.detach().cpu()
                for stage, values in ancestry.stage_grids.items()
            },
            parent_maps=tuple(value.detach().cpu() for value in ancestry.parent_maps),
            representative_indices=ancestry.representative_indices.detach().cpu(),
        )
        planning_frames_by_cell = {
            cell: tuple(
                FrameProposalCatalog(
                    cell=catalog.cell,
                    frame_id=catalog.frame_id,
                    visible_indices=catalog.visible_indices.detach().cpu(),
                    proposal_offsets=catalog.proposal_offsets.detach().cpu(),
                    proposal_point_indices=catalog.proposal_point_indices.detach().cpu(),
                )
                for catalog in catalogs
            )
            for cell, catalogs in frames_by_cell.items()
        }
    phase_end("planning_transfer_ms", planning_transfer_start)
    resolved_quota_epoch = int(epoch) if quota_epoch is None else int(quota_epoch)
    quotas = rotating_three_cell_quotas(
        resolved_quota_epoch,
        total_proposals=int(total_proposal_quota),
        cells=ordered_cells,
    )
    preselection_start = phase_start()
    catalogs_by_frame: dict[str, list[FrameProposalCatalog]] = {}
    valid_indices_by_cell: dict[str, tuple[torch.Tensor, ...]] = {}
    for cell in ordered_cells:
        source_frames = tuple(frames_by_cell[cell])
        frames = tuple(planning_frames_by_cell[cell])
        valid_for_cell: list[torch.Tensor] = []
        for source_catalog, catalog in zip(source_frames, frames, strict=True):
            if catalog.cell != cell:
                raise ValueError(
                    f"frame catalog {catalog.frame_id} has cell {catalog.cell!r}"
                )
            if source_catalog.visible_indices.device != points.device:
                raise ValueError("frame catalogs and points must share one device")
            valid_indices = catalog.valid_proposal_indices()
            for tensor_name, tensor in (
                ("visible_indices", catalog.visible_indices),
                ("proposal_point_indices", catalog.proposal_point_indices),
            ):
                if tensor.numel() and int(tensor.max()) >= int(points.shape[0]):
                    raise ValueError(
                        f"{cell}/{catalog.frame_id}: {tensor_name} indexes outside points"
                    )
            valid_for_cell.append(valid_indices)
            catalogs_by_frame.setdefault(catalog.frame_id, []).append(catalog)
        valid_indices_by_cell[cell] = tuple(valid_for_cell)
    # Same-frame relation support is another potentially large planning
    # structure.  It is needed by preselection only after a queue position for
    # that physical frame is inspected, and selected triplet sampling reuses
    # the same record below.  Keep construction lazy alongside catalog
    # preselection contexts; catalog validation above still covers every
    # supplied catalog before any scan begins.
    frame_support_by_id: dict[str, dict[str, Any]] = {}

    def get_frame_support(frame_id: str) -> dict[str, Any]:
        support = frame_support_by_id.get(frame_id)
        if support is None:
            support = _frame_relation_stage_support(
                catalogs_by_frame[frame_id],
                planning_ancestry,
                num_points=int(points.shape[0]),
            )
            frame_support_by_id[frame_id] = support
            if timing is not None:
                timing["preselection_frame_support_count"] = len(
                    frame_support_by_id
                )
        return support
    # Catalog contexts can be sizeable on real scenes (unique proposal CSR,
    # stage support counts, and grouped negative signatures).  Keep the cache
    # empty until the deterministic coverage scan actually inspects a raw
    # catalog position; a catalog whose queue positions are never visited must
    # not pay this setup cost.
    catalog_preselection_cache: dict[tuple[str, int], dict[str, Any]] = {}
    phase_end("preselection_setup_ms", preselection_start)
    # ``frame_support_by_id`` already owns the exact same-frame relation CSR
    # needed for ancestry-aware preselection.  Reuse it for selected training
    # groups rather than rebuilding the potentially large relation a second
    # time.  All granularities sharing one physical frame therefore consume
    # one canonical CSR construction.

    groups: list[FrameLocalContrastiveGroup] = []
    coverage_by_cell: dict[str, Mapping[str, Any]] = {}
    triplet_start = phase_start()
    device_relation_cache: dict[str, dict[str, torch.Tensor | Any]] = {}
    device_catalog_cache: dict[tuple[str, int], dict[str, torch.Tensor | Any]] = {}
    if timing is not None:
        # These counters isolate the catalog-wide preselection scan from the
        # later selected-group triplet work.  They are diagnostics only and
        # never enter coverage or resume metadata.
        timing["preselection_evidence_ms"] = 0.0
        timing["preselection_evidence_calls"] = 0
        timing["preselection_proposal_count"] = 0
        timing["preselection_eligible_count"] = 0
        timing["preselection_rejected_count"] = 0
        timing["preselection_catalog_count"] = 0
        timing["preselection_frame_support_count"] = 0
    for cell in ordered_cells:
        planning_frames = tuple(planning_frames_by_cell[cell])
        source_frames = tuple(frames_by_cell[cell])
        entries: list[tuple[int, int, str]] = []
        entry_catalogs: list[FrameProposalCatalog] = []
        for frame_position, frame in enumerate(planning_frames):
            for proposal_index in valid_indices_by_cell[cell][frame_position].tolist():
                entries.append((frame_position, int(proposal_index), frame.frame_id))
                entry_catalogs.append(frame)
        entry_position_by_key = {
            (int(frame_position), int(proposal_index)): position
            for position, (frame_position, proposal_index, _frame_id) in enumerate(
                entries
            )
        }
        fingerprint = _catalog_fingerprint(entries)
        if not entries:
            if coverage_states is not None and coverage_states.get(cell) is not None:
                raise ValueError(
                    f"{scene_id}/{cell}: saved coverage state has no current population"
                )
            coverage_by_cell[cell] = {
                "schema_version": COVERAGE_STATE_SCHEMA,
                "algorithm_version": SAMPLER_ALGORITHM_VERSION,
                "seed_scheme": SAMPLER_SEED_SCHEME,
                "state_key": None,
                "requested_count": int(quotas[cell]),
                "quota_epoch": resolved_quota_epoch,
                "selected_count": 0,
                "realized_count": 0,
                "adaptive_truncation": bool(quotas[cell]),
                "population_size": 0,
                "raw_population": 0,
                "population_fingerprint": fingerprint,
                "cursor_before": 0,
                "cursor_after": 0,
                "cycle_before": None,
                "cycle_after": None,
                "crossed_cycle_boundary": False,
                "duplicate_count": 0,
                "selected_catalog_positions": [],
                "selected_entries": [],
                "representable_population": 0,
                "representable_population_count": 0,
                "representable_population_complete": True,
                "preselection_inspected_count": 0,
                "preselection_inspected_entries": [],
                "preselection_skipped_entries": [],
                "preselection_raw_eligibility": [],
                "preselection_evidence_complete": True,
                "preselection_policy": PRESELECTION_POLICY,
                "hierarchy_route_eligibility_required": bool(
                    require_hierarchy_routes
                ),
                "state_before": None,
                "state_after": None,
                "no_group_reason": "empty_valid_proposal_catalog",
                "frame_group_count": 0,
                "attempted_frame_group_count": 0,
                "dropped_frame_group_count": 0,
                "dropped_frame_groups": [],
                "contrastive_retained_count": 0,
                "contrastive_dropped_count": 0,
            }
            continue
        saved_state = (
            coverage_states.get(cell)
            if coverage_states is not None
            else None
        )
        if int(epoch) > 0 and saved_state is None:
            raise ValueError(
                f"{scene_id}/{cell}: persisted coverage state is required "
                "after ancestry-aware preselection"
            )
        cursor = _coverage_minimum_cursor_before_epoch(
            epoch=int(epoch),
            cell=cell,
            population_size=len(entries),
            total_proposals=int(total_proposal_quota),
            cells=ordered_cells,
        )
        default_state = DeterministicCoverageState(
            seed=int(seed),
            epoch=int(epoch),
            scene_id=scene_id,
            operation=f"{operation}/{cell}",
            population_size=len(entries),
            population_fingerprint=fingerprint,
            cursor=cursor,
        )
        state = saved_state
        if state is None:
            state = default_state
        elif not isinstance(state, DeterministicCoverageState):
            raise TypeError(
                f"{scene_id}/{cell}: coverage state must be "
                "DeterministicCoverageState or None"
            )
        elif state.key != default_state.key or state.cursor < default_state.cursor:
            raise ValueError(
                f"{scene_id}/{cell}: saved coverage state identity/cursor mismatch"
            )

        # Evaluate representability lazily in the immutable raw permutation
        # order.  The cache is keyed by raw catalog position (rather than by
        # inspection order), so a resumed cursor and any repeated local query
        # observe byte-identical evidence without re-running the predicate.
        entry_evidence_cache: dict[int, dict[str, Any]] = {}
        entry_token_pair_cache: dict[int, torch.Tensor] = {}
        evidence_start = time.perf_counter() if timing is not None else None

        def evaluate_position(position: int) -> bool:
            position = int(position)
            cached = entry_evidence_cache.get(position)
            if cached is not None:
                return bool(cached["eligible"])
            frame_position, proposal_index, frame_id = entries[position]
            catalog = entry_catalogs[position]
            frame_support = get_frame_support(frame_id)
            catalog_key = (cell, int(frame_position))
            catalog_context = catalog_preselection_cache.get(catalog_key)
            if catalog_context is None:
                catalog_context = _catalog_preselection_context(
                    catalog=catalog,
                    ancestry=planning_ancestry,
                    frame_support=frame_support,
                    num_points=int(points.shape[0]),
                )
                catalog_preselection_cache[catalog_key] = catalog_context
                if timing is not None:
                    timing["preselection_catalog_count"] = len(
                        catalog_preselection_cache
                    )
            evidence = _proposal_preselection_evidence(
                catalog=catalog,
                proposal_index=int(proposal_index),
                granularity=cell.rsplit("/", 1)[-1],
                ancestry=planning_ancestry,
                frame_support=frame_support,
                require_negative_proposal_membership=(
                    require_negative_proposal_membership
                ),
                require_hierarchy_routes=require_hierarchy_routes,
                catalog_context=catalog_context,
                sampled_positive_pair_count=min(
                    int(positive_pairs_per_proposal), 64
                ),
                sampled_positive_pair_generator=_preselection_positive_pair_generator(
                    seed=int(seed),
                    scene_id=scene_id,
                    operation=operation,
                    cell=cell,
                    frame_id=frame_id,
                    proposal_index=int(proposal_index),
                    device=planning_ancestry.point_to_stage["dec0"].device,
                ),
            )
            token_positive_pairs = evidence.get("token_positive_pairs")
            if token_positive_pairs is not None:
                if (
                    token_positive_pairs.ndim != 2
                    or int(token_positive_pairs.shape[1]) != 2
                ):
                    raise ValueError(
                        "preselection native-dec0 pairs must have shape [P,2]"
                    )
                entry_token_pair_cache[position] = token_positive_pairs.detach()
            entry_evidence_cache[position] = {
                "catalog_position": position,
                "frame_position": int(frame_position),
                "proposal_index": int(proposal_index),
                "frame_id": frame_id,
                "eligible": bool(evidence["eligible"]),
                "reason": evidence["reason"],
                "proposal_member_count": int(
                    evidence["proposal_member_count"]
                ),
                "pure_dec0_token_count": int(
                    evidence["pure_dec0_token_count"]
                ),
                "fully_known_pure_dec0_token_count": int(
                    evidence["fully_known_pure_dec0_token_count"]
                ),
                "preferred_route_stage": evidence["preferred_route_stage"],
                "relation_safe_positive_pair_exists": bool(
                    evidence["relation_safe_positive_pair_exists"]
                ),
                "relation_safe_negative_exists": bool(
                    evidence["relation_safe_negative_exists"]
                ),
                "token_positive_pair_count": (
                    int(token_positive_pairs.shape[0])
                    if token_positive_pairs is not None
                    else 0
                ),
                "stage_support": evidence["stage_support"],
            }
            return bool(entry_evidence_cache[position]["eligible"])

        selected_positions, next_state, coverage = state.scan_eligible_lazy(
            quotas[cell],
            evaluate_position=evaluate_position,
        )
        inspected_entries = [
            entry_evidence_cache[int(position)]
            for position in coverage["inspected_positions"]
        ]
        if materialize_full_preselection_evidence:
            # Coverage auditing must know the exact representable denominator,
            # but this intentionally never runs on the production hot path.
            # ``evaluate_position`` is cache-backed, so positions inspected by
            # the actual queue are neither recomputed nor allowed to alter its
            # cursor/state transition.
            for position in range(len(entries)):
                evaluate_position(position)
            entry_evidence = [entry_evidence_cache[position] for position in range(len(entries))]
        else:
            entry_evidence = inspected_entries
        representable_population = sum(
            bool(entry["eligible"]) for entry in entry_evidence
        )
        if timing is not None and evidence_start is not None:
            timing["preselection_evidence_ms"] = float(
                timing["preselection_evidence_ms"]
            ) + 1000.0 * (time.perf_counter() - evidence_start)
            timing["preselection_evidence_calls"] = int(
                timing["preselection_evidence_calls"]
            ) + len(entry_evidence)
            timing["preselection_proposal_count"] = int(
                timing["preselection_proposal_count"]
            ) + len(entry_evidence)
            timing["preselection_eligible_count"] = int(
                timing["preselection_eligible_count"]
            ) + representable_population
            timing["preselection_rejected_count"] = int(
                timing["preselection_rejected_count"]
            ) + len(entry_evidence) - representable_population
        selected_entries = [entries[position] for position in selected_positions]
        skipped_entries = [
            entry
            for entry in inspected_entries
            if not bool(entry["eligible"])
        ]
        coverage = {
            **coverage,
            "realized_count": int(len(selected_entries)),
            "raw_population": int(len(entries)),
            "representable_population": int(representable_population),
            "representable_population_count": int(representable_population),
            "representable_population_complete": bool(
                coverage["representable_population_complete"]
                or materialize_full_preselection_evidence
            ),
            "selected_catalog_positions": list(selected_positions),
            "selected_entries": [
                {
                    "frame_position": int(frame_position),
                    "proposal_index": int(proposal_index),
                    "frame_id": frame_id,
                }
                for frame_position, proposal_index, frame_id in selected_entries
            ],
            "preselection_policy": PRESELECTION_POLICY,
            "hierarchy_route_eligibility_required": bool(
                require_hierarchy_routes
            ),
            "quota_epoch": resolved_quota_epoch,
            "preselection_inspected_count": int(len(inspected_entries)),
            "preselection_inspected_entries": inspected_entries,
            "preselection_skipped_entries": skipped_entries,
            "preselection_raw_eligibility": entry_evidence,
            "preselection_evidence_complete": bool(
                coverage["representable_population_complete"]
                or materialize_full_preselection_evidence
            ),
            "preselection_skipped_count": int(len(skipped_entries)),
            "state_before": state.to_dict(),
            "state_after": next_state.to_dict(),
            "no_group_reason": (
                None
                if selected_entries
                else "no_preselection_representable_proposal"
            ),
        }
        coverage_by_cell[cell] = coverage

        # Preserve first selected-frame occurrence rather than sorting frame IDs;
        # this makes group order part of the deterministic coverage contract.
        selected_by_frame: dict[int, list[int]] = {}
        for frame_position, proposal_index, _ in selected_entries:
            selected_by_frame.setdefault(frame_position, []).append(proposal_index)
        first_cell_group = len(groups)
        dropped_frame_groups: list[dict[str, Any]] = []
        for frame_position, proposal_indices in selected_by_frame.items():
            frame = source_frames[frame_position]
            planning_frame = planning_frames[frame_position]
            frame_id = frame.frame_id
            token_pair_chunks: list[torch.Tensor] = []
            token_pair_offsets = [0]
            for proposal_index in proposal_indices:
                raw_position = entry_position_by_key[
                    (int(frame_position), int(proposal_index))
                ]
                token_pairs = entry_token_pair_cache.get(raw_position)
                if token_pairs is None:
                    raise RuntimeError(
                        "selected eligible proposal has no cached native-dec0 pairs"
                    )
                token_pairs = token_pairs.to(device=points.device, dtype=torch.long)
                token_pair_chunks.append(token_pairs)
                token_pair_offsets.append(
                    token_pair_offsets[-1] + int(token_pairs.shape[0])
                )
            token_pair_offsets_tensor = torch.tensor(
                token_pair_offsets,
                dtype=torch.long,
                device=points.device,
            )
            token_pair_indices_tensor = (
                torch.cat(token_pair_chunks, dim=0)
                if token_pair_chunks
                else torch.empty(
                    (0, 2), dtype=torch.long, device=points.device
                )
            )
            token_pair_proposal_indices = torch.repeat_interleave(
                torch.arange(
                    len(proposal_indices),
                    dtype=torch.long,
                    device=points.device,
                ),
                token_pair_offsets_tensor[1:] - token_pair_offsets_tensor[:-1],
            )
            token_pair_labels = torch.tensor(
                proposal_indices,
                dtype=torch.long,
                device=points.device,
            )[token_pair_proposal_indices]
            token_pair_digest = token_positive_pair_payload_digest(
                token_pair_offsets_tensor,
                token_pair_indices_tensor,
                algorithm_version=SAMPLER_ALGORITHM_VERSION,
                preselection_policy=PRESELECTION_POLICY,
                proposal_labels=token_pair_labels,
                pair_proposal_indices=token_pair_proposal_indices,
            )
            validate_token_positive_pair_payload(
                token_pair_offsets_tensor,
                token_pair_indices_tensor,
                token_pair_digest,
                proposal_count=len(proposal_indices),
                algorithm_version=SAMPLER_ALGORITHM_VERSION,
                preselection_policy=PRESELECTION_POLICY,
                proposal_labels=token_pair_labels,
                pair_proposal_indices=token_pair_proposal_indices,
            )
            if frame_id not in device_relation_cache:
                support = get_frame_support(frame_id)
                relation_offsets = support["relation_offsets"].to(points.device)
                relation_members = support["relation_members"].to(points.device)
                relation_index = build_relation_membership_index(
                    relation_offsets,
                    relation_members,
                    point_capacity=int(points.shape[0]),
                )
                relation_points = torch.unique(relation_members.long(), sorted=True)
                device_relation_cache[frame_id] = {
                    "relation_offsets": relation_offsets,
                    "relation_members": relation_members,
                    "relation_points": relation_points,
                    "relation_index": relation_index,
                    "safety_relation_offsets": support["relation_offsets"],
                    "safety_relation_points": support["relation_members"],
                    "safety_relation_index": support["relation_index"],
                }
            catalog_key = (cell, int(frame_position))
            if catalog_key not in device_catalog_cache:
                relation_context = device_relation_cache[frame_id]
                visible = torch.unique(frame.visible_indices.long(), sorted=True)
                known_indices = visible[
                    torch.isin(visible, relation_context["relation_points"])
                ]
                device_catalog_cache[catalog_key] = {
                    **relation_context,
                    "visible": visible,
                    "known": known_indices,
                    "valid": planning_frame.valid_proposal_indices().to(points.device),
                    "safety_visible": torch.unique(
                        planning_frame.visible_indices.long(), sorted=True
                    ),
                }
            prepared = device_catalog_cache[catalog_key]
            generator = torch.Generator(device=points.device).manual_seed(
                _stable_seed(
                    seed,
                    epoch,
                    scene_id,
                    operation,
                    cell,
                    frame.frame_id,
                    ",".join(str(value) for value in proposal_indices),
                    "frame-local-triplets",
                )
            )
            selected_tensor = torch.tensor(
                proposal_indices, dtype=torch.long, device=points.device
            )
            try:
                call_start = phase_start()
                batch, sampler_metadata = sample_safe_overlapping_proposal_triplets(
                    num_scene_points=int(points.shape[0]),
                    visible_indices=frame.visible_indices,
                    proposal_offsets=frame.proposal_offsets,
                    proposal_point_indices=frame.proposal_point_indices,
                    selected_proposal_indices=selected_tensor,
                    points=points,
                    positive_pairs_per_proposal=int(positive_pairs_per_proposal),
                    num_uniform_negatives=int(num_uniform_negatives),
                    num_spatial_hard_negatives=int(num_spatial_hard_negatives),
                    spatial_candidate_pool=int(spatial_candidate_pool),
                    num_feature_hard_negatives=int(num_feature_hard_negatives),
                    feature_candidate_pool=int(feature_candidate_pool),
                    relation_proposal_offsets=prepared["relation_offsets"],
                    relation_proposal_point_indices=prepared["relation_members"],
                    require_negative_proposal_membership=bool(
                        require_negative_proposal_membership
                    ),
                    defer_token_loss=True,
                    generator=generator,
                    prepared_visible_indices=prepared["visible"],
                    prepared_relation_points=prepared["relation_points"],
                    prepared_relation_index=prepared["relation_index"],
                    prepared_known_indices=prepared["known"],
                    prepared_valid_proposal_indices=prepared["valid"],
                    prepared_safety_visible_indices=prepared["safety_visible"],
                    prepared_safety_relation_index=prepared[
                        "safety_relation_index"
                    ],
                    prepared_safety_relation_offsets=prepared[
                        "safety_relation_offsets"
                    ],
                    prepared_safety_relation_points=prepared[
                        "safety_relation_points"
                    ],
                    deferred_safety_device=(
                        "cpu"
                        if requested_planning_device.type == "cpu"
                        else None
                    ),
                    token_positive_pair_offsets=token_pair_offsets_tensor,
                    token_positive_pair_indices=token_pair_indices_tensor,
                    token_positive_pair_digest=token_pair_digest,
                )
                if timing is not None and call_start is not None:
                    if points.device.type == "cuda":
                        torch.cuda.synchronize(points.device)
                    timing["triplet_calls"] = int(timing.get("triplet_calls", 0)) + 1
                    timing["triplet_sampling_ms"] = float(
                        timing.get("triplet_sampling_ms", 0.0)
                    ) + 1000.0 * (time.perf_counter() - call_start)
            except ValueError as error:
                if str(error) != "no selected proposal retained a safe contrastive row":
                    raise
                dropped_frame_groups.append(
                    {
                        "cell": cell,
                        "frame_id": frame.frame_id,
                        "frame_position": int(frame_position),
                        "selected_proposal_indices": [
                            int(value) for value in proposal_indices
                        ],
                        "selected_proposal_count": len(proposal_indices),
                        "reason": "no_selected_proposal_retained_safe_contrastive_row",
                    }
                )
                continue
            groups.append(
                FrameLocalContrastiveGroup(
                    cell=cell,
                    frame_id=frame.frame_id,
                    batch=batch,
                    selected_proposal_indices=tuple(proposal_indices),
                    coverage_metadata={
                        **coverage,
                        "quota_by_cell": quotas,
                        "group_selected_proposal_indices": list(proposal_indices),
                        "group_selected_count": len(proposal_indices),
                        "sampler": sampler_metadata,
                    },
                )
            )
        retained_count = sum(
            len(group.coverage_metadata["sampler"]["retained_proposal_indices"])
            for group in groups[first_cell_group:]
        )
        coverage_by_cell[cell] = {
            **coverage,
            "frame_group_count": len(groups) - first_cell_group,
            "attempted_frame_group_count": len(selected_by_frame),
            "dropped_frame_group_count": len(dropped_frame_groups),
            "dropped_frame_groups": dropped_frame_groups,
            "contrastive_retained_count": int(retained_count),
            "contrastive_dropped_count": int(len(selected_entries) - retained_count),
        }
    phase_end("triplet_sampling_total_ms", triplet_start)
    if timing is not None:
        timing["relation_context_count"] = len(device_relation_cache)
        timing["catalog_context_count"] = len(device_catalog_cache)
        timing["deferred_safety_device"] = (
            "cpu"
            if requested_planning_device.type == "cpu"
            else str(points.device)
        )
    plan = MultigranularFrameGroupPlan(
        groups=tuple(groups),
        quota_by_cell=quotas,
        coverage_by_cell=coverage_by_cell,
    )
    if not plan.groups:
        raise NoRetainedFrameGroupsError(plan)
    return plan


def sample_multigranular_frame_groups(
    *,
    scene_id: str,
    epoch: int,
    seed: int,
    points: torch.Tensor,
    frames_by_cell: Mapping[str, Sequence[FrameProposalCatalog]],
    total_proposal_quota: int = 16,
    cells: Sequence[str] = DEFAULT_2D_CELLS,
    positive_pairs_per_proposal: int = 64,
    num_uniform_negatives: int = 256,
    num_spatial_hard_negatives: int = 128,
    spatial_candidate_pool: int = 1024,
    num_feature_hard_negatives: int = 128,
    feature_candidate_pool: int = 1024,
    require_negative_proposal_membership: bool = False,
    require_hierarchy_routes: bool = True,
    operation: str = "multigranular-train-v2",
    coverage_states: Mapping[str, DeterministicCoverageState | None] | None = None,
    quota_epoch: int | None = None,
    planning_device: str | torch.device | None = None,
    timing: dict[str, Any] | None = None,
    materialize_full_preselection_evidence: bool = False,
) -> tuple[FrameLocalContrastiveGroup, ...]:
    """Compatibility wrapper returning only ordered frame-local groups.

    New integrations should call :func:`sample_multigranular_frame_group_plan`
    so hard-gate accounting is retained when a cell has no valid proposal.
    """

    return sample_multigranular_frame_group_plan(
        scene_id=scene_id,
        epoch=epoch,
        seed=seed,
        points=points,
        frames_by_cell=frames_by_cell,
        total_proposal_quota=total_proposal_quota,
        cells=cells,
        positive_pairs_per_proposal=positive_pairs_per_proposal,
        num_uniform_negatives=num_uniform_negatives,
        num_spatial_hard_negatives=num_spatial_hard_negatives,
        spatial_candidate_pool=spatial_candidate_pool,
        num_feature_hard_negatives=num_feature_hard_negatives,
        feature_candidate_pool=feature_candidate_pool,
        require_negative_proposal_membership=require_negative_proposal_membership,
        require_hierarchy_routes=require_hierarchy_routes,
        operation=operation,
        coverage_states=coverage_states,
        quota_epoch=quota_epoch,
        planning_device=planning_device,
        timing=timing,
        materialize_full_preselection_evidence=materialize_full_preselection_evidence,
    ).groups


__all__ = [
    "COVERAGE_STATE_SCHEMA",
    "DEFAULT_2D_CELLS",
    "DeterministicCoverageState",
    "FrameLocalContrastiveGroup",
    "FrameProposalCatalog",
    "LITEPT_HIERARCHY_STAGES",
    "LitePTHierarchyAncestry",
    "MultigranularFrameGroupPlan",
    "NoRetainedFrameGroupsError",
    "PRESELECTION_POLICY",
    "derive_litept_hierarchy_ancestry",
    "rotating_three_cell_quotas",
    "sample_multigranular_frame_group_plan",
    "sample_multigranular_frame_groups",
]
