"""Proposal sampling for PartField-style contrastive pretraining.

The sampler deliberately knows nothing about the source of a proposal.  A 2D
UnSAM proposal supplies labels and an eligibility mask restricted to points
visible in one RGB-D frame.  A 3D PartField proposal supplies labels and an
eligibility mask restricted to supervised points in the scene crop.  Keeping
that contract shared is important for matched 2D/3D/joint experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import torch


# This version is deliberately part of the serialized sampler metadata.  A
# change to the proposal/negative stream contract must never be mistaken for a
# replay-equivalent implementation when a checkpoint is resumed.
SAMPLER_ALGORITHM_VERSION = "partfield_contrastive_v2/preselection_ancestry_v2"
SAMPLER_SEED_SCHEME = (
    "sha256(preselection_policy,seed,scene_id,operation,cell,frame_id,"
    "proposal_index,positive-token-pairs)+"
    "sha256(root_seed,algorithm_version,proposal_index,operation,proposal-family)+"
    "coverage_permutation_scan"
)
# This is kept beside the algorithm stamp so native-dec0 pair producers and
# consumers cannot silently disagree about which ancestry-aware preselection
# predicate produced an exact pair payload.
SAMPLER_PRESELECTION_POLICY = (
    "litept_voxel_ancestry_relation_safe_sampled_pairs_v2"
)
TOKEN_POSITIVE_PAIR_PAYLOAD_SCHEMA = (
    "partfield_native_dec0_token_positive_pairs/v1"
)

_TOKEN_POSITIVE_PAIR_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


def token_positive_pair_payload_digest(
    token_positive_pair_offsets: torch.Tensor,
    token_positive_pair_indices: torch.Tensor,
    *,
    algorithm_version: str = SAMPLER_ALGORITHM_VERSION,
    preselection_policy: str = SAMPLER_PRESELECTION_POLICY,
    proposal_labels: torch.Tensor | None = None,
    pair_proposal_indices: torch.Tensor | None = None,
) -> str:
    """Hash the versioned native-dec0 positive-pair CSR payload.

    The digest intentionally canonicalizes both integer tensors to CPU int64
    lists before hashing.  This keeps the handoff digest independent of the
    producer device and tensor storage layout while binding the payload
    schema, sampler policy, offsets, pair IDs, and retained raw proposal
    occurrence order exactly.  ``proposal_labels`` is the pair-row label
    tensor and ``pair_proposal_indices`` maps those rows to the CSR
    occurrences; together they preserve the raw proposal labels even when
    multiple pair rows belong to one occurrence.
    """

    if not torch.is_tensor(token_positive_pair_offsets) or not torch.is_tensor(
        token_positive_pair_indices
    ):
        raise TypeError("native-dec0 pair payload fields must be tensors")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise TypeError("native-dec0 pair payload requires an algorithm version")
    if not isinstance(preselection_policy, str) or not preselection_policy:
        raise TypeError("native-dec0 pair payload requires a preselection policy")
    metadata: dict[str, Any] = {
        "sampler_algorithm_version": algorithm_version,
        "sampler_preselection_policy": preselection_policy,
    }
    if (proposal_labels is None) != (pair_proposal_indices is None):
        raise ValueError(
            "proposal_labels and pair_proposal_indices must be supplied together"
        )
    if proposal_labels is not None and pair_proposal_indices is not None:
        if not torch.is_tensor(proposal_labels) or not torch.is_tensor(
            pair_proposal_indices
        ):
            raise TypeError(
                "proposal labels and pair-to-occurrence map must be tensors"
            )
        if proposal_labels.ndim != 1 or pair_proposal_indices.ndim != 1:
            raise ValueError(
                "proposal labels and pair-to-occurrence map must be vectors"
            )
        if proposal_labels.shape != pair_proposal_indices.shape:
            raise ValueError(
                "proposal labels and pair-to-occurrence map must have one row per pair"
            )
        if proposal_labels.dtype not in _TOKEN_POSITIVE_PAIR_INTEGER_DTYPES:
            raise TypeError("proposal_labels must be integral")
        if pair_proposal_indices.dtype not in _TOKEN_POSITIVE_PAIR_INTEGER_DTYPES:
            raise TypeError("pair_proposal_indices must be integral")
        if proposal_labels.device != pair_proposal_indices.device:
            raise ValueError(
                "proposal labels and pair-to-occurrence map must share one device"
            )
        offsets = token_positive_pair_offsets.detach().long()
        pair_map = pair_proposal_indices.detach().long()
        if pair_map.numel() and (
            int(pair_map.min()) < 0
            or int(pair_map.max()) >= int(offsets.numel()) - 1
        ):
            raise ValueError("pair-to-occurrence map indexes outside pair CSR")
        occurrence_labels: list[int] = []
        labels = proposal_labels.detach().long()
        for occurrence in range(int(offsets.numel()) - 1):
            rows = torch.where(pair_map == occurrence)[0]
            if rows.numel() == 0:
                raise ValueError(
                    "each retained proposal occurrence must have at least one pair row"
                )
            unique_labels = torch.unique(labels[rows], sorted=True)
            if unique_labels.numel() != 1:
                raise ValueError("proposal labels drift within one occurrence")
            occurrence_labels.append(int(unique_labels[0]))
        metadata.update(
            {
                "retained_raw_proposal_labels": occurrence_labels,
                "pair_proposal_indices": pair_map.cpu().tolist(),
                "proposal_labels": labels.cpu().tolist(),
            }
        )
    canonical = {
        "schema_version": TOKEN_POSITIVE_PAIR_PAYLOAD_SCHEMA,
        **metadata,
        "token_positive_pair_offsets": token_positive_pair_offsets.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
        .tolist(),
        "token_positive_pair_indices": token_positive_pair_indices.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1, 2)
        .tolist(),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_token_positive_pair_payload(
    token_positive_pair_offsets: torch.Tensor,
    token_positive_pair_indices: torch.Tensor,
    token_positive_pair_digest: str,
    *,
    proposal_count: int | None = None,
    token_count: int | None = None,
    algorithm_version: str = SAMPLER_ALGORITHM_VERSION,
    preselection_policy: str = SAMPLER_PRESELECTION_POLICY,
    proposal_labels: torch.Tensor | None = None,
    pair_proposal_indices: torch.Tensor | None = None,
) -> None:
    """Fail closed on malformed or stale exact native-dec0 pair payloads."""

    if algorithm_version != SAMPLER_ALGORITHM_VERSION:
        raise ValueError(
            "native-dec0 pair payload algorithm version is stale or unsupported"
        )
    if preselection_policy != SAMPLER_PRESELECTION_POLICY:
        raise ValueError(
            "native-dec0 pair payload preselection policy is stale or unsupported"
        )
    if proposal_labels is None or pair_proposal_indices is None:
        raise ValueError(
            "native-dec0 pair payload digest must bind proposal labels and occurrence order"
        )
    if not torch.is_tensor(token_positive_pair_offsets) or not torch.is_tensor(
        token_positive_pair_indices
    ):
        raise TypeError("native-dec0 pair payload fields must be tensors")
    offsets = token_positive_pair_offsets
    pairs = token_positive_pair_indices
    if offsets.ndim != 1 or offsets.numel() < 1:
        raise ValueError("token_positive_pair_offsets must be [U+1]")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("token_positive_pair_indices must be [T,2]")
    if offsets.dtype not in _TOKEN_POSITIVE_PAIR_INTEGER_DTYPES:
        raise TypeError(
            "token_positive_pair_offsets must be integral, "
            f"got {offsets.dtype}"
        )
    if pairs.dtype not in _TOKEN_POSITIVE_PAIR_INTEGER_DTYPES:
        raise TypeError(
            "token_positive_pair_indices must be integral, "
            f"got {pairs.dtype}"
        )
    if offsets.device != pairs.device:
        raise ValueError("native-dec0 pair payload tensors must share one device")
    offsets = offsets.long()
    pairs = pairs.long()
    if int(offsets[0]) != 0:
        raise ValueError("token_positive_pair_offsets must start at zero")
    if bool((offsets < 0).any()):
        raise ValueError("token_positive_pair_offsets must be non-negative")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("token_positive_pair_offsets must be non-decreasing")
    if int(offsets[-1]) != int(pairs.shape[0]):
        raise ValueError(
            "token_positive_pair_offsets do not span token_positive_pair_indices"
        )
    if proposal_count is not None and int(offsets.numel()) != int(proposal_count) + 1:
        raise ValueError(
            "token_positive_pair_offsets proposal count disagrees with batch"
        )
    if pairs.numel():
        if int(pairs.min()) < 0:
            raise ValueError("token_positive_pair_indices must be non-negative")
        if bool((pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError("native-dec0 positive pair endpoints must differ")
        if token_count is not None and int(pairs.max()) >= int(token_count):
            raise ValueError("token_positive_pair_indices exceed native-dec0 token count")
        # Exact positive pairs are unordered and must not repeat within one
        # proposal occurrence.  The same token pair may legitimately recur
        # for a different overlapping proposal occurrence.
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            rows = pairs[int(start) : int(end)]
            if rows.shape[0] <= 1:
                continue
            canonical_rows = torch.sort(rows, dim=1).values
            if torch.unique(canonical_rows, dim=0).shape[0] != rows.shape[0]:
                raise ValueError(
                    "native-dec0 unordered positive pair repeated within proposal"
                )
    if not isinstance(token_positive_pair_digest, str):
        raise TypeError("token_positive_pair_digest must be a SHA-256 string")
    if len(token_positive_pair_digest) != 64:
        raise ValueError("token_positive_pair_digest must be 64 hex characters")
    try:
        int(token_positive_pair_digest, 16)
    except ValueError as exc:
        raise ValueError("token_positive_pair_digest must be hexadecimal") from exc
    expected_digest = token_positive_pair_payload_digest(
        offsets,
        pairs,
        algorithm_version=algorithm_version,
        preselection_policy=preselection_policy,
        proposal_labels=proposal_labels,
        pair_proposal_indices=pair_proposal_indices,
    )
    if token_positive_pair_digest != expected_digest:
        raise ValueError("native-dec0 positive-pair payload digest mismatch")


def _stable_seed(*parts: object) -> int:
    """Return a stable signed-64 seed independent of Python hash randomization."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _generator_seed(generator: torch.Generator | None) -> int:
    return int(generator.initial_seed()) if generator is not None else 0


def _sample_root_seed(
    generator: torch.Generator | None,
    *,
    device: torch.device,
) -> int:
    """Resolve and consume one root draw before deriving independent streams.

    Consuming one scalar from an explicit generator preserves the usual
    call-to-call advancement contract.  The derived root, rather than the
    generator's mutable state, is then recorded in metadata so selecting a
    subset of proposals cannot perturb another proposal's positive stream.
    """

    # A generator must drive a tensor on its own device.  Draw the root on
    # that device when a caller supplied an explicit CPU generator for a CUDA
    # scene; only the resulting integer crosses devices.
    draw_device = torch.device(device)
    if generator is not None:
        try:
            draw_device = torch.device(generator.device)
        except AttributeError:  # pragma: no cover - old Torch compatibility
            draw_device = torch.device("cpu")
    # ``generator=None`` means the caller opted into Torch's default RNG.  A
    # single scalar draw preserves that convention, after which all V2 draws
    # use versioned per-proposal streams and can be skipped independently.
    raw = torch.empty((), dtype=torch.int64, device=draw_device).random_(
        generator=generator
    )
    return int(raw.item()) & ((1 << 63) - 1)


def _device_generator(
    *,
    device: torch.device,
    seed: int,
) -> torch.Generator:
    """Create a generator supported by the destination device.

    Torch does not allow a CPU generator to drive CUDA random kernels.  V2
    derives every stream from an integer seed first, so rebuilding the stream
    on the destination device is deterministic and avoids a CPU/CUDA mismatch.
    """

    return torch.Generator(device=device).manual_seed(int(seed))


def _generator_for_device(
    generator: torch.Generator | None,
    *,
    device: torch.device,
) -> torch.Generator | None:
    """Adapt a caller-owned generator to the tensor/device being sampled.

    The historical API accepts a generator without requiring callers to
    rebuild it for CUDA.  Torch's random kernels do require matching devices,
    so preserve the caller's initial seed when a CPU generator is used for a
    CUDA tensor (or vice versa).  Same-device generators are returned
    unchanged, preserving the legacy random stream exactly.
    """

    if generator is None:
        return None
    try:
        generator_device = torch.device(generator.device)
    except AttributeError:  # pragma: no cover - old Torch compatibility
        generator_device = torch.device("cpu")
    target = torch.device(device)
    if generator_device == target:
        return generator
    return _device_generator(device=target, seed=_generator_seed(generator))


def _proposal_stream_seed(
    root_seed: int,
    proposal_index: int,
    operation: str,
) -> int:
    return _stable_seed(
        SAMPLER_ALGORITHM_VERSION,
        int(root_seed),
        int(proposal_index),
        str(operation),
    )


def _proposal_streams(
    *,
    root_seed: int,
    proposal_index: int,
    device: torch.device,
) -> dict[str, torch.Generator]:
    """Build independent per-proposal streams for every sampling family."""

    proposal_seed = _proposal_stream_seed(
        int(root_seed), int(proposal_index), "proposal"
    )
    return {
        family: _device_generator(
            device=torch.device(device),
            seed=_stable_seed(
                SAMPLER_ALGORITHM_VERSION,
                proposal_seed,
                family,
            ),
        )
        for family in ("positive_pairs", "uniform", "spatial", "feature")
    }


@dataclass(frozen=True)
class DeferredTokenLossPlan:
    """Point-space feasibility record for a loss consumed in token space.

    V2 rebuilds every positive pair and negative family after the LitePT dec0
    voxel map is known.  Materializing point-space negatives before that map is
    therefore pure overhead.  This record makes the deferral explicit while
    retaining the effective point-space widths used by the token rebuild and
    the audit trail.
    """

    requested_positive_pairs_per_proposal: int
    requested_uniform_negatives_per_pair: int
    requested_spatial_hard_negatives_per_pair: int
    requested_feature_hard_negatives_per_pair: int
    effective_uniform_negatives_per_pair: int
    effective_spatial_hard_negatives_per_pair: int
    effective_feature_candidate_pool_per_pair: int
    effective_feature_hard_negatives_per_pair: int
    point_negative_ids_materialized: bool = False
    consumed_loss_space: str = "native_dec0_tokens"
    schema_version: str = "partfield_deferred_token_loss_plan/v1"
    algorithm_version: str = SAMPLER_ALGORITHM_VERSION
    require_negative_proposal_membership: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.requested_positive_pairs_per_proposal,
            self.requested_uniform_negatives_per_pair,
            self.requested_spatial_hard_negatives_per_pair,
            self.requested_feature_hard_negatives_per_pair,
            self.effective_uniform_negatives_per_pair,
            self.effective_spatial_hard_negatives_per_pair,
            self.effective_feature_candidate_pool_per_pair,
            self.effective_feature_hard_negatives_per_pair,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("Deferred token-loss counts must be non-negative")
        if self.point_negative_ids_materialized:
            raise ValueError("A deferred token-loss plan cannot materialize point negatives")
        if self.consumed_loss_space != "native_dec0_tokens":
            raise ValueError("Deferred token loss must be consumed in native dec0 space")
        if self.schema_version != "partfield_deferred_token_loss_plan/v1":
            raise ValueError("Unsupported deferred token-loss plan schema")
        if self.algorithm_version != SAMPLER_ALGORITHM_VERSION:
            raise ValueError("Unsupported deferred token-loss algorithm version")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "consumed_loss_space": self.consumed_loss_space,
            "point_negative_ids_materialized": self.point_negative_ids_materialized,
            "require_negative_proposal_membership": bool(
                getattr(self, "require_negative_proposal_membership", False)
            ),
            "requested_positive_pairs_per_proposal": int(
                self.requested_positive_pairs_per_proposal
            ),
            "requested_uniform_negatives_per_pair": int(
                self.requested_uniform_negatives_per_pair
            ),
            "requested_spatial_hard_negatives_per_pair": int(
                self.requested_spatial_hard_negatives_per_pair
            ),
            "requested_feature_hard_negatives_per_pair": int(
                self.requested_feature_hard_negatives_per_pair
            ),
            "effective_uniform_negatives_per_pair": int(
                self.effective_uniform_negatives_per_pair
            ),
            "effective_spatial_hard_negatives_per_pair": int(
                self.effective_spatial_hard_negatives_per_pair
            ),
            "effective_feature_candidate_pool_per_pair": int(
                self.effective_feature_candidate_pool_per_pair
            ),
            "effective_feature_hard_negatives_per_pair": int(
                self.effective_feature_hard_negatives_per_pair
            ),
        }


@dataclass(frozen=True)
class ContrastiveTripletBatch:
    """Indices for a batch of positive pairs and per-pair negatives."""

    positive_pairs: torch.Tensor  # [P, 2]
    negative_indices: torch.Tensor  # [P, M_random+M_geometry]
    proposal_labels: torch.Tensor  # [P]
    feature_candidate_indices: torch.Tensor | None = None  # [P, Q]
    num_feature_hard_negatives: int = 0
    # Optional exact proposal-membership contract used only by native
    # multi-scale deep supervision.  Membership is CSR over the proposal
    # *occurrences* sampled in this batch; pair_proposal_indices maps each
    # point-level pair row back to one occurrence.  eligible_indices defines
    # the proposal-complement universe (frame-visible points for 2D, valid
    # partition points for 3D).  Keeping this metadata sparse avoids a dense
    # [proposal, scene-point] mask for multi-million-point full scenes.
    proposal_member_offsets: torch.Tensor | None = None  # [U + 1]
    proposal_member_indices: torch.Tensor | None = None  # [sum members]
    pair_proposal_indices: torch.Tensor | None = None  # [P], values in [0,U)
    eligible_indices: torch.Tensor | None = None  # [E]
    # Optional full same-frame teacher relation across every 2D granularity.
    # Unlike proposal_member_* (only selected occurrences), this CSR is used
    # to rebuild false-negative-safe rows after point-to-token remapping.
    relation_proposal_offsets: torch.Tensor | None = None  # [R + 1]
    relation_proposal_member_indices: torch.Tensor | None = None  # [sum members]
    multiscale_seed: int | None = None
    num_uniform_negatives: int = 0
    num_spatial_hard_negatives: int = 0
    spatial_candidate_pool: int = 0
    feature_candidate_pool: int = 0
    deferred_token_loss: DeferredTokenLossPlan | None = None
    # ``None`` identifies the historical with-replacement path.  V2 batches
    # carry the algorithm stamp so fixed-eval records and checkpoint metadata
    # can distinguish replay contracts without changing legacy defaults.
    algorithm_version: str | None = None
    # Preserve whether the sampler intentionally restricts negatives to points
    # carrying at least one teacher-proposal membership.  ``False`` is the
    # historical visible-complement default and must survive token rebuild.
    require_negative_proposal_membership: bool = False
    # Optional exact native-dec0 positive-pair payload.  The CSR is ordered by
    # retained raw-proposal occurrence, not by token ID; the digest binds that
    # order, labels, sampler algorithm, and preselection policy.
    token_positive_pair_offsets: torch.Tensor | None = None  # [U + 1]
    token_positive_pair_indices: torch.Tensor | None = None  # [T, 2]
    token_positive_pair_digest: str | None = None

    def __post_init__(self) -> None:
        payload_fields = (
            self.token_positive_pair_offsets,
            self.token_positive_pair_indices,
            self.token_positive_pair_digest,
        )
        payload_presence = tuple(value is not None for value in payload_fields)
        if any(payload_presence) and not all(payload_presence):
            raise ValueError(
                "native-dec0 positive-pair payload fields must be supplied together"
            )
        if not all(payload_presence):
            return
        if self.algorithm_version != SAMPLER_ALGORITHM_VERSION:
            raise ValueError(
                "native-dec0 positive-pair payload requires the current sampler algorithm"
            )
        if self.proposal_member_offsets is None or self.pair_proposal_indices is None:
            raise ValueError(
                "native-dec0 positive-pair payload requires proposal occurrence metadata"
            )
        assert self.token_positive_pair_offsets is not None
        assert self.token_positive_pair_indices is not None
        assert self.token_positive_pair_digest is not None
        validate_token_positive_pair_payload(
            self.token_positive_pair_offsets,
            self.token_positive_pair_indices,
            self.token_positive_pair_digest,
            proposal_count=int(self.proposal_member_offsets.numel()) - 1,
            algorithm_version=self.algorithm_version,
            proposal_labels=self.proposal_labels,
            pair_proposal_indices=self.pair_proposal_indices,
        )
        payload_device = self.token_positive_pair_offsets.device
        if any(
            tensor.device != payload_device
            for tensor in (
                self.token_positive_pair_indices,
                self.proposal_member_offsets,
                self.pair_proposal_indices,
                self.proposal_labels,
            )
        ):
            raise ValueError(
                "native-dec0 pair payload and proposal occurrence metadata must share one device"
            )

    @property
    def num_pairs(self) -> int:
        return int(self.positive_pairs.shape[0])

    @property
    def num_negatives(self) -> int:
        return int(self.negative_indices.shape[1]) + int(
            self.num_feature_hard_negatives
        )

    @property
    def num_base_negatives(self) -> int:
        return int(self.negative_indices.shape[1])


@dataclass(frozen=True)
class AdaptiveNegativeSelection:
    """Rectangular row-wise sampling result with an explicit adaptive width.

    ``indices`` never repeats an index within a row.  When one or more rows
    contain fewer unique candidates than requested, every row is truncated to
    the smallest available width so the result remains a dense tensor suitable
    for the contrastive criterion.
    """

    indices: torch.Tensor
    requested_per_row: int
    selected_per_row: int
    available_per_row: torch.Tensor

    def metadata(self) -> dict[str, Any]:
        available = self.available_per_row
        return {
            "requested_per_row": int(self.requested_per_row),
            "selected_per_row": int(self.selected_per_row),
            "adaptive_truncation": bool(
                self.selected_per_row < self.requested_per_row
            ),
            "minimum_available": (
                int(available.min()) if available.numel() else 0
            ),
            "maximum_available": (
                int(available.max()) if available.numel() else 0
            ),
            "rows_truncated": int(
                (available < int(self.requested_per_row)).sum()
            ),
        }


def concatenate_triplet_batches(
    batches: list[ContrastiveTripletBatch],
) -> ContrastiveTripletBatch:
    """Concatenate batches that share the same negative-mining contract."""

    if not batches:
        raise ValueError("At least one triplet batch is required")
    feature_hard_counts = {int(batch.num_feature_hard_negatives) for batch in batches}
    if len(feature_hard_counts) != 1:
        raise ValueError("Triplet batches use different feature-hard-negative counts")
    candidate_presence = {batch.feature_candidate_indices is not None for batch in batches}
    if len(candidate_presence) != 1:
        raise ValueError("Triplet batches disagree about feature candidate presence")
    membership_presence: set[bool] = set()
    for batch in batches:
        fields = (
            batch.proposal_member_offsets,
            batch.proposal_member_indices,
            batch.pair_proposal_indices,
            batch.eligible_indices,
        )
        present = tuple(value is not None for value in fields)
        if any(present) and not all(present):
            raise ValueError(
                "Triplet batch contains partial proposal-membership metadata"
            )
        membership_presence.add(all(present))
    if len(membership_presence) != 1:
        raise ValueError("Triplet batches disagree about proposal-membership metadata")
    relation_presence: set[bool] = set()
    for batch in batches:
        present = (
            batch.relation_proposal_offsets is not None,
            batch.relation_proposal_member_indices is not None,
        )
        if any(present) and not all(present):
            raise ValueError("Triplet batch contains partial relation-membership CSR")
        relation_presence.add(all(present))
    if len(relation_presence) != 1:
        raise ValueError("Triplet batches disagree about relation-membership metadata")
    token_pair_presence: set[bool] = set()
    for batch in batches:
        fields = (
            batch.token_positive_pair_offsets,
            batch.token_positive_pair_indices,
            batch.token_positive_pair_digest,
        )
        present = tuple(value is not None for value in fields)
        if any(present) and not all(present):
            raise ValueError(
                "Triplet batch contains partial native-dec0 positive-pair payload"
            )
        token_pair_presence.add(all(present))
    if len(token_pair_presence) != 1:
        raise ValueError(
            "Triplet batches disagree about native-dec0 positive-pair payload"
        )
    metadata_counts = {
        (
            int(batch.num_uniform_negatives),
            int(batch.num_spatial_hard_negatives),
            int(batch.spatial_candidate_pool),
            int(batch.feature_candidate_pool),
        )
        for batch in batches
    }
    if len(metadata_counts) != 1:
        raise ValueError("Triplet batches use different negative-category contracts")
    deferred_plans = {batch.deferred_token_loss for batch in batches}
    if len(deferred_plans) != 1:
        raise ValueError("Triplet batches disagree about deferred token-loss plans")
    deferred_token_loss = next(iter(deferred_plans))
    algorithm_versions = {batch.algorithm_version for batch in batches}
    if len(algorithm_versions) != 1:
        raise ValueError("Triplet batches disagree about sampler algorithm version")
    algorithm_version = next(iter(algorithm_versions))
    if algorithm_version is not None and algorithm_version != SAMPLER_ALGORITHM_VERSION:
        raise ValueError(f"Unsupported sampler algorithm version {algorithm_version!r}")
    if deferred_token_loss is not None and algorithm_version != SAMPLER_ALGORITHM_VERSION:
        raise ValueError("Deferred token loss requires the V2 sampler algorithm")
    relation_membership_requirements = {
        bool(getattr(batch, "require_negative_proposal_membership", False))
        for batch in batches
    }
    if len(relation_membership_requirements) != 1:
        raise ValueError(
            "Triplet batches disagree about negative proposal-membership policy"
        )
    require_negative_proposal_membership = next(
        iter(relation_membership_requirements)
    )

    proposal_member_offsets = None
    proposal_member_indices = None
    pair_proposal_indices = None
    eligible_indices = None
    multiscale_seed = None
    relation_proposal_offsets = None
    relation_proposal_member_indices = None
    token_positive_pair_offsets = None
    token_positive_pair_indices = None
    token_positive_pair_digest = None
    if True in membership_presence:
        reference_eligible = batches[0].eligible_indices
        assert reference_eligible is not None
        if any(
            batch.eligible_indices is None
            or not torch.equal(batch.eligible_indices, reference_eligible)
            for batch in batches[1:]
        ):
            raise ValueError(
                "Concatenated proposals do not share one eligible-point universe"
            )
        eligible_indices = reference_eligible
        member_chunks: list[torch.Tensor] = []
        offset_values = [0]
        pair_chunks: list[torch.Tensor] = []
        proposal_base = 0
        member_base = 0
        for batch in batches:
            assert batch.proposal_member_offsets is not None
            assert batch.proposal_member_indices is not None
            assert batch.pair_proposal_indices is not None
            assert batch.eligible_indices is not None
            local_offsets = batch.proposal_member_offsets.long()
            local_pair_map = batch.pair_proposal_indices.long()
            local_proposals = int(local_offsets.numel()) - 1
            if (
                local_proposals <= 0
                or local_offsets.ndim != 1
                or int(local_offsets[0]) != 0
                or int(local_offsets[-1])
                != int(batch.proposal_member_indices.numel())
                or bool((local_offsets[1:] < local_offsets[:-1]).any())
                or bool((local_offsets[1:] == local_offsets[:-1]).any())
                or local_pair_map.shape != (batch.num_pairs,)
                or (
                    local_pair_map.numel() > 0
                    and int(local_pair_map.min()) < 0
                )
                or (
                    local_pair_map.numel() > 0
                    and int(local_pair_map.max()) >= local_proposals
                )
                or bool(
                    (
                        torch.bincount(
                            local_pair_map,
                            minlength=local_proposals,
                        )
                        == 0
                    ).any()
                )
                or batch.multiscale_seed is None
            ):
                raise ValueError("Malformed proposal-membership CSR contract")
            metadata_device = batch.eligible_indices.device
            if any(
                value.device != metadata_device
                for value in (
                    batch.proposal_member_offsets,
                    batch.proposal_member_indices,
                    batch.pair_proposal_indices,
                )
            ):
                raise ValueError(
                    "Proposal-membership CSR tensors must share one device"
                )
            member_chunks.append(batch.proposal_member_indices)
            offset_values.extend(
                (local_offsets[1:] + member_base).tolist()
            )
            pair_chunks.append(local_pair_map + proposal_base)
            proposal_base += int(local_offsets.numel()) - 1
            member_base += int(batch.proposal_member_indices.numel())
        proposal_member_offsets = torch.tensor(
            offset_values,
            dtype=torch.long,
            device=reference_eligible.device,
        )
        proposal_member_indices = torch.cat(member_chunks, dim=0)
        pair_proposal_indices = torch.cat(pair_chunks, dim=0)
        seeds = {batch.multiscale_seed for batch in batches}
        if len(seeds) != 1 or None in seeds:
            raise ValueError("Concatenated proposals disagree about multi-scale seed")
        multiscale_seed = int(seeds.pop())
    if True in relation_presence:
        relation_proposal_offsets = batches[0].relation_proposal_offsets
        relation_proposal_member_indices = (
            batches[0].relation_proposal_member_indices
        )
        assert relation_proposal_offsets is not None
        assert relation_proposal_member_indices is not None
        _validate_proposal_membership_csr(
            relation_proposal_offsets,
            relation_proposal_member_indices,
        )
        if any(
            batch.relation_proposal_offsets is None
            or batch.relation_proposal_member_indices is None
            or not torch.equal(
                batch.relation_proposal_offsets,
                relation_proposal_offsets,
            )
            or not torch.equal(
                batch.relation_proposal_member_indices,
                relation_proposal_member_indices,
            )
            for batch in batches[1:]
        ):
            raise ValueError(
                "Concatenated batches do not share one relation-membership CSR"
            )

    if True in token_pair_presence:
        pair_chunks: list[torch.Tensor] = []
        offset_values = [0]
        pair_base = 0
        for batch in batches:
            assert batch.token_positive_pair_offsets is not None
            assert batch.token_positive_pair_indices is not None
            local_offsets = batch.token_positive_pair_offsets.long()
            local_pairs = batch.token_positive_pair_indices.long()
            if local_offsets.device != local_pairs.device:
                raise ValueError(
                    "native-dec0 pair payload tensors must share one device"
                )
            pair_chunks.append(local_pairs)
            offset_values.extend(
                (local_offsets[1:] + int(pair_base)).tolist()
            )
            pair_base += int(local_pairs.shape[0])
        payload_device = pair_chunks[0].device
        token_positive_pair_offsets = torch.tensor(
            offset_values,
            dtype=torch.long,
            device=payload_device,
        )
        token_positive_pair_indices = torch.cat(pair_chunks, dim=0)

    (
        num_uniform_negatives,
        num_spatial_hard_negatives,
        spatial_candidate_pool,
        feature_candidate_pool,
    ) = metadata_counts.pop()
    concatenated_positive_pairs = torch.cat(
        [batch.positive_pairs for batch in batches], dim=0
    )
    concatenated_negative_indices = torch.cat(
        [batch.negative_indices for batch in batches], dim=0
    )
    concatenated_proposal_labels = torch.cat(
        [batch.proposal_labels for batch in batches], dim=0
    )
    if True in token_pair_presence:
        assert token_positive_pair_offsets is not None
        assert token_positive_pair_indices is not None
        assert pair_proposal_indices is not None
        token_positive_pair_digest = token_positive_pair_payload_digest(
            token_positive_pair_offsets,
            token_positive_pair_indices,
            algorithm_version=algorithm_version,
            proposal_labels=concatenated_proposal_labels,
            pair_proposal_indices=pair_proposal_indices,
        )
    return ContrastiveTripletBatch(
        positive_pairs=concatenated_positive_pairs,
        negative_indices=concatenated_negative_indices,
        proposal_labels=concatenated_proposal_labels,
        feature_candidate_indices=(
            torch.cat(
                [
                    batch.feature_candidate_indices
                    for batch in batches
                    if batch.feature_candidate_indices is not None
                ],
                dim=0,
            )
            if True in candidate_presence
            else None
        ),
        num_feature_hard_negatives=feature_hard_counts.pop(),
        proposal_member_offsets=proposal_member_offsets,
        proposal_member_indices=proposal_member_indices,
        pair_proposal_indices=pair_proposal_indices,
        eligible_indices=eligible_indices,
        relation_proposal_offsets=relation_proposal_offsets,
        relation_proposal_member_indices=relation_proposal_member_indices,
        multiscale_seed=multiscale_seed,
        num_uniform_negatives=num_uniform_negatives,
        num_spatial_hard_negatives=num_spatial_hard_negatives,
        spatial_candidate_pool=spatial_candidate_pool,
        feature_candidate_pool=feature_candidate_pool,
        deferred_token_loss=deferred_token_loss,
        algorithm_version=algorithm_version,
        require_negative_proposal_membership=require_negative_proposal_membership,
        token_positive_pair_offsets=token_positive_pair_offsets,
        token_positive_pair_indices=token_positive_pair_indices,
        token_positive_pair_digest=token_positive_pair_digest,
    )


def sample_overlapping_proposal_triplets(
    *,
    num_scene_points: int,
    visible_indices: torch.Tensor,
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    points: torch.Tensor | None = None,
    num_proposals: int = 4,
    positive_pairs_per_proposal: int = 64,
    num_uniform_negatives: int = 256,
    num_spatial_hard_negatives: int = 0,
    spatial_candidate_pool: int = 1024,
    num_feature_hard_negatives: int = 0,
    feature_candidate_pool: int = 1024,
    selected_proposal_indices: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ContrastiveTripletBatch:
    """Sample triplets from an overlapping raw proposal set.

    A point may belong to any number of proposals. For a selected proposal,
    positives are its member points and negatives are visible points outside
    that proposal; membership in another overlapping proposal is irrelevant.
    """

    if visible_indices.ndim != 1 or proposal_offsets.ndim != 1:
        raise ValueError("visible_indices and proposal_offsets must be one-dimensional")
    if proposal_point_indices.ndim != 1:
        raise ValueError("proposal_point_indices must be one-dimensional")
    if proposal_offsets.numel() < 1 or int(proposal_offsets[0]) != 0:
        raise ValueError("proposal_offsets must start at zero")
    if int(proposal_offsets[-1]) != int(proposal_point_indices.numel()):
        raise ValueError("proposal_offsets do not span proposal_point_indices")
    if visible_indices.device != proposal_offsets.device or visible_indices.device != proposal_point_indices.device:
        raise ValueError("Raw proposal tensors must be on the same device")
    if int(num_scene_points) <= 0:
        raise ValueError("num_scene_points must be positive")

    visible_mask = torch.zeros(
        int(num_scene_points), dtype=torch.bool, device=visible_indices.device
    )
    visible_mask[visible_indices.long()] = True
    # Cache construction guarantees that every proposal membership is a subset
    # of the frame-visible set. Counts therefore establish both >=2 positives
    # and >=1 complement negative without scanning a dense scene mask for every
    # proposal on every training step.
    proposal_sizes = proposal_offsets[1:] - proposal_offsets[:-1]
    valid_tensor = torch.where(
        (proposal_sizes >= 2) & (proposal_sizes < int(visible_indices.numel()))
    )[0].long()
    if valid_tensor.numel() == 0:
        raise ValueError("No raw proposal has at least two positives and one visible negative")
    sampling_generator = _generator_for_device(
        generator, device=visible_indices.device
    )
    if selected_proposal_indices is None:
        selected = valid_tensor[
            torch.randint(
                int(valid_tensor.numel()),
                (int(num_proposals),),
                generator=sampling_generator,
                device=visible_indices.device,
            )
        ]
    else:
        if (
            selected_proposal_indices.ndim != 1
            or int(selected_proposal_indices.numel()) != int(num_proposals)
        ):
            raise ValueError(
                "selected_proposal_indices must contain exactly num_proposals entries"
            )
        if selected_proposal_indices.device != visible_indices.device:
            raise ValueError(
                "selected_proposal_indices must share the raw proposal device"
            )
        selected = selected_proposal_indices.long()
        if not bool(torch.isin(selected, valid_tensor).all()):
            raise ValueError(
                "selected_proposal_indices contains an invalid raw proposal"
            )
    batches: list[ContrastiveTripletBatch] = []
    for proposal_index_tensor in selected:
        proposal_index = int(proposal_index_tensor)
        start = int(proposal_offsets[proposal_index])
        end = int(proposal_offsets[proposal_index + 1])
        members = proposal_point_indices[start:end].long()
        labels = torch.full(
            (int(num_scene_points),), -1, dtype=torch.long, device=visible_indices.device
        )
        labels[visible_indices.long()] = 0
        labels[members] = 1
        batch = sample_partition_triplets(
            labels,
            eligible_mask=visible_mask,
            points=points,
            num_proposals=1,
            positive_pairs_per_proposal=int(positive_pairs_per_proposal),
            num_uniform_negatives=int(num_uniform_negatives),
            num_spatial_hard_negatives=int(num_spatial_hard_negatives),
            spatial_candidate_pool=int(spatial_candidate_pool),
            num_feature_hard_negatives=int(num_feature_hard_negatives),
            feature_candidate_pool=int(feature_candidate_pool),
            ignored_proposal_labels=(-1, 0),
            exclude_ignored_from_negatives=False,
            generator=sampling_generator,
        )
        batches.append(
            ContrastiveTripletBatch(
                positive_pairs=batch.positive_pairs,
                negative_indices=batch.negative_indices,
                proposal_labels=torch.full_like(
                    batch.proposal_labels, proposal_index
                ),
                feature_candidate_indices=batch.feature_candidate_indices,
                num_feature_hard_negatives=batch.num_feature_hard_negatives,
                proposal_member_offsets=batch.proposal_member_offsets,
                proposal_member_indices=batch.proposal_member_indices,
                pair_proposal_indices=batch.pair_proposal_indices,
                eligible_indices=batch.eligible_indices,
                multiscale_seed=batch.multiscale_seed,
                num_uniform_negatives=batch.num_uniform_negatives,
                num_spatial_hard_negatives=batch.num_spatial_hard_negatives,
                spatial_candidate_pool=batch.spatial_candidate_pool,
                feature_candidate_pool=batch.feature_candidate_pool,
                relation_proposal_offsets=batch.relation_proposal_offsets,
                relation_proposal_member_indices=(
                    batch.relation_proposal_member_indices
                ),
                deferred_token_loss=batch.deferred_token_loss,
                algorithm_version=batch.algorithm_version,
                require_negative_proposal_membership=(
                    getattr(batch, "require_negative_proposal_membership", False)
                ),
            )
        )
    return concatenate_triplet_batches(batches)


def _sample_with_replacement(
    values: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if count == 0:
        return values.new_empty((0,), dtype=torch.long)
    if values.numel() == 0:
        raise ValueError("Cannot sample from an empty tensor")
    draw = torch.randint(
        int(values.numel()),
        (int(count),),
        generator=_generator_for_device(generator, device=values.device),
        device=values.device,
    )
    return values[draw]


def _ignored_mask(labels: torch.Tensor, ignored_labels: Iterable[int]) -> torch.Tensor:
    ignored = torch.zeros_like(labels, dtype=torch.bool)
    for value in ignored_labels:
        ignored |= labels == int(value)
    return ignored


def _sample_unique_pool(
    values: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw a shared candidate pool without replacement when possible."""

    if count <= 0:
        raise ValueError("Candidate-pool count must be positive")
    unique_count = min(int(count), int(values.numel()))
    if unique_count == int(values.numel()):
        return values
    permutation = torch.randperm(
        int(values.numel()),
        generator=_generator_for_device(generator, device=values.device),
        device=values.device,
    )
    return values[permutation[:unique_count]]


def _require_integral_vector(values: torch.Tensor, *, name: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"{name} must be integral, got {values.dtype}")


def _sample_unique_integer_range(
    population_size: int,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample unique integer IDs without allocating a huge full permutation."""

    population_size = int(population_size)
    count = int(count)
    if population_size < 0 or count < 0:
        raise ValueError("population_size and count must be non-negative")
    count = min(count, population_size)
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    # A caller may keep one CPU generator for CPU and CUDA paths.  Random
    # kernels require a generator on their own device, but preserving the
    # integer seed keeps the stream deterministic across this API boundary.
    rng = _generator_for_device(generator, device=torch.device(device))

    # A full permutation is faster and simpler when its allocation is bounded.
    # Positive-pair populations can otherwise be O(N^2), so use deterministic
    # batched rejection sampling for the sparse-count regime.
    if population_size <= 1_000_000 or count * 4 >= population_size:
        return torch.randperm(
            population_size,
            generator=rng,
            device=device,
        )[:count]

    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < count:
        remaining = count - len(selected)
        draws = torch.randint(
            population_size,
            (max(16, remaining * 2),),
            generator=rng,
            device=device,
        )
        for value in draws.tolist():
            integer = int(value)
            if integer not in seen:
                seen.add(integer)
                selected.append(integer)
                if len(selected) == count:
                    break
    return torch.tensor(selected, dtype=torch.long, device=device)


def sample_unique_without_replacement(
    values: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw at most ``count`` distinct values with deterministic Torch RNG.

    Duplicate input values are collapsed before sampling.  The returned length
    is therefore ``min(count, number_of_unique_values)`` rather than silently
    repeating a small candidate pool.
    """

    _require_integral_vector(values, name="values")
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    unique = torch.unique(values.long(), sorted=True)
    draw = _sample_unique_integer_range(
        int(unique.numel()),
        int(count),
        device=values.device,
        generator=generator,
    )
    return unique[draw]


def sample_unique_positive_pairs(
    positive_indices: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample unique unordered positive pairs without self-pairs.

    The result adapts to at most ``N choose 2`` rows.  Sampling pair IDs rather
    than materializing ``torch.combinations`` keeps large masks inexpensive.
    """

    _require_integral_vector(positive_indices, name="positive_indices")
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    points = torch.unique(positive_indices.long(), sorted=True)
    point_count = int(points.numel())
    if point_count < 2 or int(count) == 0:
        return points.new_empty((0, 2))
    pair_population = point_count * (point_count - 1) // 2
    pair_ids = _sample_unique_integer_range(
        pair_population,
        int(count),
        device=points.device,
        generator=generator,
    )
    rows = torch.arange(point_count, dtype=torch.long, device=points.device)
    row_starts = rows * (2 * point_count - rows - 1) // 2
    left = torch.searchsorted(row_starts[1:], pair_ids, right=True)
    right = left + 1 + pair_ids - row_starts[left]
    return torch.stack([points[left], points[right]], dim=1)


def _validate_proposal_membership_csr(
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
) -> None:
    _require_integral_vector(proposal_offsets, name="proposal_offsets")
    _require_integral_vector(
        proposal_point_indices, name="proposal_point_indices"
    )
    if proposal_offsets.device != proposal_point_indices.device:
        raise ValueError("proposal membership tensors must share one device")
    if proposal_offsets.numel() == 0 or int(proposal_offsets[0]) != 0:
        raise ValueError("proposal_offsets must start at zero")
    if proposal_offsets.numel() and bool((proposal_offsets < 0).any()):
        raise ValueError("proposal_offsets must be non-negative")
    if int(proposal_offsets[-1]) != int(proposal_point_indices.numel()):
        raise ValueError("proposal_offsets do not span proposal_point_indices")
    if bool((proposal_offsets[1:] < proposal_offsets[:-1]).any()):
        raise ValueError("proposal_offsets must be non-decreasing")
    if proposal_point_indices.numel() and bool((proposal_point_indices < 0).any()):
        raise ValueError("proposal_point_indices must be non-negative")


def _unique_proposal_member_counts(
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    *,
    point_capacity: int | None = None,
) -> torch.Tensor:
    """Count distinct member IDs per CSR proposal without a dense mask."""

    proposal_offsets = proposal_offsets.long()
    proposal_count = int(proposal_offsets.numel()) - 1
    counts = torch.zeros(
        proposal_count, dtype=torch.long, device=proposal_offsets.device
    )
    if proposal_count == 0 or proposal_point_indices.numel() == 0:
        return counts
    members = proposal_point_indices.long()
    capacity = (
        int(point_capacity)
        if point_capacity is not None
        else int(members.max().item()) + 1
    )
    if capacity <= 0:
        raise ValueError("point_capacity must be positive when members are present")
    sizes = proposal_offsets[1:] - proposal_offsets[:-1]
    proposal_ids = torch.repeat_interleave(
        torch.arange(proposal_count, dtype=torch.long, device=proposal_offsets.device),
        sizes,
    )
    unique_keys = torch.unique(proposal_ids * capacity + members, sorted=True)
    unique_proposals = torch.div(
        unique_keys, capacity, rounding_mode="floor"
    )
    counts = torch.bincount(unique_proposals, minlength=proposal_count)
    return counts.long()


@dataclass(frozen=True)
class RelationMembershipIndex:
    """Packed point-to-proposal membership reused across safety queries.

    Each signed int64 word uses 63 bits, avoiding the sign bit while reducing
    co-membership to vectorized bitwise intersections.  Duplicate CSR edges
    are canonicalized before packing, so integer addition is exactly bitwise
    OR for every point/word location.
    """

    point_words: torch.Tensor
    proposal_count: int
    bits_per_word: int = 63

    def __post_init__(self) -> None:
        if self.point_words.ndim != 2 or self.point_words.dtype != torch.int64:
            raise ValueError("point_words must be an int64 [points,words] tensor")
        if int(self.proposal_count) < 0 or int(self.bits_per_word) != 63:
            raise ValueError("Invalid packed relation-membership contract")
        expected_words = max(1, (int(self.proposal_count) + 62) // 63)
        if int(self.point_words.shape[1]) != expected_words:
            raise ValueError(
                "point_words has the wrong number of 63-bit proposal words"
            )

    @property
    def point_capacity(self) -> int:
        return int(self.point_words.shape[0])

    def contains(self, point_indices: torch.Tensor) -> torch.Tensor:
        if point_indices.numel() == 0:
            return torch.zeros_like(point_indices, dtype=torch.bool)
        values = point_indices.long()
        if int(values.min()) < 0 or int(values.max()) >= self.point_capacity:
            raise ValueError("Point index is outside the relation-membership index")
        return self.point_words[values].ne(0).any(dim=-1)


def build_relation_membership_index(
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    *,
    point_capacity: int | None = None,
) -> RelationMembershipIndex:
    """Pack a proposal CSR into a reusable exact point-membership index."""

    _validate_proposal_membership_csr(
        proposal_offsets, proposal_point_indices
    )
    offsets = proposal_offsets.long()
    members = proposal_point_indices.long()
    proposal_count = int(offsets.numel()) - 1
    inferred_capacity = int(members.max()) + 1 if members.numel() else 0
    capacity = inferred_capacity if point_capacity is None else int(point_capacity)
    if capacity < inferred_capacity or capacity < 0:
        raise ValueError("point_capacity does not cover proposal membership")
    word_count = max(1, (proposal_count + 62) // 63)
    packed = torch.zeros(
        (capacity, word_count),
        dtype=torch.int64,
        device=offsets.device,
    )
    if proposal_count == 0 or members.numel() == 0:
        return RelationMembershipIndex(packed, proposal_count)

    sizes = offsets[1:] - offsets[:-1]
    proposal_ids = torch.repeat_interleave(
        torch.arange(proposal_count, dtype=torch.long, device=offsets.device),
        sizes,
    )
    # Canonicalize duplicate membership entries before summing disjoint bits.
    edge_keys = members * max(proposal_count, 1) + proposal_ids
    edge_keys = torch.unique(edge_keys, sorted=False)
    edge_members = torch.div(edge_keys, max(proposal_count, 1), rounding_mode="floor")
    edge_proposals = edge_keys.remainder(max(proposal_count, 1))
    word_ids = torch.div(edge_proposals, 63, rounding_mode="floor")
    bit_ids = edge_proposals.remainder(63)
    bit_values = torch.ones_like(bit_ids, dtype=torch.int64) << bit_ids
    flat_locations = edge_members * word_count + word_ids
    packed.view(-1).index_add_(0, flat_locations, bit_values)
    return RelationMembershipIndex(packed, proposal_count)


def co_membership_safe_negative_mask_reference(
    anchor_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    require_candidate_membership: bool = False,
) -> torch.Tensor:
    """Small, transparent reference implementation for packed-index audits.

    This intentionally uses Python sets and is not used in training.  Keeping
    it next to the packed implementation makes a CPU regression check possible
    for the 63-bit word boundary and gives future changes an executable oracle.
    """

    _require_integral_vector(anchor_indices, name="anchor_indices")
    _validate_proposal_membership_csr(proposal_offsets, proposal_point_indices)
    if candidate_indices.ndim not in (1, 2):
        raise ValueError("candidate_indices must have shape [Q] or [P,Q]")
    if candidate_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("candidate_indices must be integral")
    if candidate_indices.device != anchor_indices.device:
        raise ValueError("anchors and candidates must share one device")
    if proposal_offsets.device != anchor_indices.device:
        raise ValueError("anchors and proposal membership must share one device")
    rows = int(anchor_indices.numel())
    if candidate_indices.ndim == 2 and candidate_indices.shape[0] != rows:
        raise ValueError("row-specific candidates must match the anchor count")
    candidates = (
        candidate_indices[None, :].expand(rows, -1)
        if candidate_indices.ndim == 1
        else candidate_indices
    ).long()
    if rows == 0:
        return torch.empty_like(candidates, dtype=torch.bool)
    if int(anchor_indices.min()) < 0:
        raise ValueError("anchor_indices must be non-negative")
    if candidates.numel() and int(candidates.min()) < 0:
        raise ValueError("candidate_indices must be non-negative")
    relation_sets: list[set[int]] = []
    offsets = proposal_offsets.long().tolist()
    members = proposal_point_indices.long().tolist()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        relation_sets.append(set(int(value) for value in members[start:end]))
    output = torch.zeros_like(candidates, dtype=torch.bool)
    for row in range(rows):
        anchor = int(anchor_indices[row])
        candidate_membership = [
            any(anchor in relation and candidate in relation for relation in relation_sets)
            for candidate in candidates[row].tolist()
        ]
        output[row] = torch.tensor(
            [
                candidate != anchor
                and not shares
                and (
                    not require_candidate_membership
                    or any(candidate in relation for relation in relation_sets)
                )
                for candidate, shares in zip(
                    candidates[row].tolist(), candidate_membership, strict=True
                )
            ],
            dtype=torch.bool,
            device=candidates.device,
        )
    return output


def co_membership_safe_negative_mask(
    anchor_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    require_candidate_membership: bool = False,
    membership_index: RelationMembershipIndex | None = None,
) -> torch.Tensor:
    """Mark candidates that share no teacher proposal with their anchor.

    ``candidate_indices`` may be one shared ``[Q]`` vector or a row-specific
    ``[P,Q]`` matrix.  The proposal CSR may combine masks from several
    granularities as long as all point IDs use the same frame-local scene
    indexing.  No dense point-by-proposal matrix is constructed.
    """

    _require_integral_vector(anchor_indices, name="anchor_indices")
    if candidate_indices.ndim not in (1, 2):
        raise ValueError("candidate_indices must have shape [Q] or [P,Q]")
    if candidate_indices.device != anchor_indices.device:
        raise ValueError("anchors and candidates must share one device")
    if proposal_offsets.device != anchor_indices.device:
        raise ValueError("anchors and proposal membership must share one device")
    if candidate_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("candidate_indices must be integral")
    if anchor_indices.numel() and int(anchor_indices.min()) < 0:
        raise ValueError("anchor_indices must be non-negative")
    if candidate_indices.numel() and int(candidate_indices.min()) < 0:
        raise ValueError("candidate_indices must be non-negative")
    pair_count = int(anchor_indices.numel())
    if candidate_indices.ndim == 2 and candidate_indices.shape[0] != pair_count:
        raise ValueError("row-specific candidates must match the anchor count")
    if membership_index is None:
        candidate_max = (
            int(candidate_indices.max()) + 1 if candidate_indices.numel() else 0
        )
        anchor_max = int(anchor_indices.max()) + 1 if anchor_indices.numel() else 0
        relation_max = (
            int(proposal_point_indices.max()) + 1
            if proposal_point_indices.numel()
            else 0
        )
        membership_index = build_relation_membership_index(
            proposal_offsets,
            proposal_point_indices,
            point_capacity=max(candidate_max, anchor_max, relation_max),
        )
    elif membership_index.point_words.device != anchor_indices.device:
        raise ValueError("membership index and query tensors must share one device")
    if membership_index.proposal_count != int(proposal_offsets.numel()) - 1:
        raise ValueError("membership index and proposal CSR disagree on proposal count")
    candidates = (
        candidate_indices[None, :].expand(pair_count, -1)
        if candidate_indices.ndim == 1
        else candidate_indices
    ).long()
    anchors = anchor_indices.long()
    if int(anchors.numel()) and int(anchors.max()) >= membership_index.point_capacity:
        raise ValueError("anchor index is outside the relation-membership index")
    if int(candidates.numel()) and int(candidates.max()) >= membership_index.point_capacity:
        raise ValueError("candidate index is outside the relation-membership index")
    safe = candidates != anchors[:, None]
    if pair_count == 0 or candidates.shape[1] == 0:
        return safe

    anchor_words = membership_index.point_words[anchors]
    candidate_words = membership_index.point_words[candidates]
    shared = torch.zeros_like(candidates, dtype=torch.bool)
    for word in range(int(anchor_words.shape[1])):
        shared |= (
            anchor_words[:, word, None]
            & candidate_words[..., word]
        ).ne(0)
    safe &= ~shared
    if require_candidate_membership:
        safe &= candidate_words.ne(0).any(dim=-1)
    return safe


def co_membership_safe_negative_pair_counts(
    anchor_pairs: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    require_candidate_membership: bool = False,
    membership_index: RelationMembershipIndex | None = None,
) -> torch.Tensor:
    """Count candidates safe for both endpoints of each pair.

    Deferred token-loss sampling only consumes the row counts.  Computing the
    union of both endpoint bitsets once avoids materializing two full
    pair-by-candidate masks and then intersecting them, while preserving the
    exact predicate implemented by :func:`co_membership_safe_negative_mask`.
    ``candidate_indices`` is intentionally restricted to one shared vector,
    which is the only shape used by the V2 point sampler.
    """

    if anchor_pairs.ndim != 2 or anchor_pairs.shape[1] != 2:
        raise ValueError("anchor_pairs must have shape [P,2]")
    if candidate_indices.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    if anchor_pairs.device != candidate_indices.device:
        raise ValueError("pair anchors and candidates must share one device")
    if membership_index is None:
        candidate_max = int(candidate_indices.max()) + 1 if candidate_indices.numel() else 0
        anchor_max = int(anchor_pairs.max()) + 1 if anchor_pairs.numel() else 0
        relation_max = int(proposal_point_indices.max()) + 1 if proposal_point_indices.numel() else 0
        membership_index = build_relation_membership_index(
            proposal_offsets,
            proposal_point_indices,
            point_capacity=max(candidate_max, anchor_max, relation_max),
        )
    if membership_index.point_words.device != anchor_pairs.device:
        raise ValueError("membership index and pair tensors must share one device")
    if membership_index.proposal_count != int(proposal_offsets.numel()) - 1:
        raise ValueError("membership index and proposal CSR disagree on proposal count")
    anchors = anchor_pairs.long()
    candidates = candidate_indices.long()
    if anchors.numel() and (int(anchors.min()) < 0 or int(anchors.max()) >= membership_index.point_capacity):
        raise ValueError("pair anchor is outside the relation-membership index")
    if candidates.numel() and (int(candidates.min()) < 0 or int(candidates.max()) >= membership_index.point_capacity):
        raise ValueError("candidate index is outside the relation-membership index")
    if candidates.numel() == 0:
        return torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
    endpoint_words = membership_index.point_words[anchors]
    union_words = endpoint_words[:, 0, :].clone()
    for word in range(int(union_words.shape[1])):
        union_words[:, word] |= endpoint_words[:, 1, word]
    candidate_words = membership_index.point_words[candidates]
    safe = candidates[None, :] != anchors[:, 0, None]
    safe &= candidates[None, :] != anchors[:, 1, None]
    shared = torch.zeros_like(safe)
    for word in range(int(union_words.shape[1])):
        shared |= (union_words[:, word, None] & candidate_words[:, word]).ne(0)
    safe &= ~shared
    if require_candidate_membership:
        safe &= candidate_words.ne(0).any(dim=-1)[None, :]
    return safe.sum(dim=1).long()


def sample_adaptive_unique_negative_rows(
    candidate_indices: torch.Tensor,
    requested_per_row: int,
    *,
    num_rows: int | None = None,
    valid_mask: torch.Tensor | None = None,
    excluded_indices: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> AdaptiveNegativeSelection:
    """Sample rectangular, duplicate-free negative rows with adaptive ``K``.

    The effective width is the minimum of the requested width and the number
    of unique valid candidates in every row.  ``excluded_indices`` is useful
    for making uniform, spatial, and feature-hard families disjoint.
    """

    if candidate_indices.ndim not in (1, 2):
        raise ValueError("candidate_indices must have shape [Q] or [P,Q]")
    if candidate_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("candidate_indices must be integral")
    requested_per_row = int(requested_per_row)
    if requested_per_row < 0:
        raise ValueError("requested_per_row must be non-negative")
    if candidate_indices.ndim == 1:
        if num_rows is None or int(num_rows) < 0:
            raise ValueError("num_rows is required for shared candidates")
        rows = int(num_rows)
        candidates = candidate_indices[None, :].expand(rows, -1)
    else:
        rows = int(candidate_indices.shape[0])
        if num_rows is not None and int(num_rows) != rows:
            raise ValueError("num_rows disagrees with candidate row count")
        candidates = candidate_indices
    if valid_mask is not None and valid_mask.shape != candidates.shape:
        raise ValueError("valid_mask must match the expanded candidate shape")
    if valid_mask is not None and valid_mask.device != candidates.device:
        raise ValueError("valid_mask must share the candidate device")
    if candidate_indices.numel() and int(candidate_indices.min()) < 0:
        raise ValueError("candidate_indices must be non-negative")
    if excluded_indices is not None:
        if excluded_indices.ndim != 2 or excluded_indices.shape[0] != rows:
            raise ValueError("excluded_indices must have shape [P,E]")
        if excluded_indices.device != candidates.device:
            raise ValueError("excluded indices must share the candidate device")
        if excluded_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("excluded_indices must be integral")
        if excluded_indices.numel() and int(excluded_indices.min()) < 0:
            raise ValueError("excluded_indices must be non-negative")

    sampling_generator = _generator_for_device(
        generator, device=candidates.device
    )

    row_values: list[torch.Tensor] = []
    counts: list[int] = []
    for row in range(rows):
        values = candidates[row]
        if valid_mask is not None:
            values = values[valid_mask[row].bool()]
        values = torch.unique(values.long(), sorted=True)
        if excluded_indices is not None and excluded_indices.shape[1] > 0:
            values = values[~torch.isin(values, excluded_indices[row].long())]
        row_values.append(values)
        counts.append(int(values.numel()))
    available = torch.tensor(counts, dtype=torch.long, device=candidates.device)
    selected_count = min(
        requested_per_row,
        int(available.min()) if available.numel() else 0,
    )
    selected_rows = [
        sample_unique_without_replacement(
            values,
            selected_count,
            generator=sampling_generator,
        )
        for values in row_values
    ]
    output = (
        torch.stack(selected_rows, dim=0)
        if selected_rows
        else candidates.new_empty((0, selected_count), dtype=torch.long)
    )
    return AdaptiveNegativeSelection(
        indices=output.long(),
        requested_per_row=requested_per_row,
        selected_per_row=selected_count,
        available_per_row=available,
    )


def sample_partition_triplets(
    labels: torch.Tensor,
    *,
    eligible_mask: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    num_proposals: int = 4,
    positive_pairs_per_proposal: int = 64,
    num_uniform_negatives: int = 256,
    num_spatial_hard_negatives: int = 0,
    spatial_candidate_pool: int = 1024,
    num_feature_hard_negatives: int = 0,
    feature_candidate_pool: int = 1024,
    ignored_proposal_labels: Iterable[int] = (-1,),
    exclude_ignored_from_negatives: bool = False,
    selected_proposal_labels: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ContrastiveTripletBatch:
    """Sample PartField-style training tuples from one label partition.

    Parameters
    ----------
    labels:
        Per-point proposal IDs. IDs are scene/frame local and have no semantic
        meaning across calls.
    eligible_mask:
        Points allowed to participate in this sampling call. For 2D this must
        be the visibility mask for exactly one frame; non-visible 3D points
        must never become negatives.
    ignored_proposal_labels:
        Labels that cannot be selected as positive proposals. They remain in
        the negative complement unless ``exclude_ignored_from_negatives`` is
        true. Thus 2D background label 0 may be used as a negative while 3D
        ignore label -1 can be excluded via the eligibility mask.
    """

    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}")
    if not labels.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"labels must be integral, got {labels.dtype}")
    if int(num_proposals) <= 0 or int(positive_pairs_per_proposal) <= 0:
        raise ValueError("num_proposals and positive_pairs_per_proposal must be positive")
    if any(
        int(value) < 0
        for value in (
            num_uniform_negatives,
            num_spatial_hard_negatives,
            num_feature_hard_negatives,
        )
    ):
        raise ValueError("negative counts must be non-negative")
    if (
        int(num_uniform_negatives)
        + int(num_spatial_hard_negatives)
        + int(num_feature_hard_negatives)
        <= 0
    ):
        raise ValueError("At least one negative per pair is required")
    if int(num_spatial_hard_negatives) > 0 and int(spatial_candidate_pool) <= 0:
        raise ValueError("spatial_candidate_pool must be positive")
    if int(num_feature_hard_negatives) > 0 and int(feature_candidate_pool) <= 0:
        raise ValueError("feature_candidate_pool must be positive")
    if int(feature_candidate_pool) < int(num_feature_hard_negatives):
        raise ValueError(
            "feature_candidate_pool must be at least num_feature_hard_negatives"
        )

    n = int(labels.numel())
    if eligible_mask is None:
        eligible_mask = torch.ones(n, dtype=torch.bool, device=labels.device)
    if eligible_mask.shape != labels.shape:
        raise ValueError(
            f"eligible_mask shape {tuple(eligible_mask.shape)} != labels shape {tuple(labels.shape)}"
        )
    eligible_mask = eligible_mask.bool()

    if points is not None:
        if points.shape != (n, 3):
            raise ValueError(f"points must have shape ({n}, 3), got {tuple(points.shape)}")
        if int(num_spatial_hard_negatives) > 0 and points.device != labels.device:
            raise ValueError("points and labels must be on the same device")
    elif int(num_spatial_hard_negatives) > 0:
        raise ValueError("points are required when spatial hard negatives are enabled")

    sampling_generator = _generator_for_device(generator, device=labels.device)

    ignored = _ignored_mask(labels, ignored_proposal_labels)
    proposal_labels = torch.unique(labels[eligible_mask & ~ignored], sorted=True)

    valid_labels: list[torch.Tensor] = []
    for proposal_label in proposal_labels:
        positive = eligible_mask & (labels == proposal_label)
        negative = eligible_mask & (labels != proposal_label)
        if exclude_ignored_from_negatives:
            negative &= ~ignored
        if int(positive.sum()) >= 2 and bool(negative.any()):
            valid_labels.append(proposal_label)
    if not valid_labels:
        raise ValueError("No proposal has at least two positives and one eligible negative")

    valid_label_tensor = torch.stack(valid_labels)
    if selected_proposal_labels is None:
        label_draw = torch.randint(
            int(valid_label_tensor.numel()),
            (int(num_proposals),),
            generator=sampling_generator,
            device=labels.device,
        )
        selected_labels = valid_label_tensor[label_draw]
    else:
        if (
            selected_proposal_labels.ndim != 1
            or int(selected_proposal_labels.numel()) != int(num_proposals)
        ):
            raise ValueError(
                "selected_proposal_labels must contain exactly num_proposals entries"
            )
        if selected_proposal_labels.device != labels.device:
            raise ValueError("selected_proposal_labels must share the labels device")
        selected_labels = selected_proposal_labels.to(dtype=labels.dtype)
        if not bool(torch.isin(selected_labels, valid_label_tensor).all()):
            raise ValueError(
                "selected_proposal_labels contains an invalid partition label"
            )

    pair_chunks: list[torch.Tensor] = []
    negative_chunks: list[torch.Tensor] = []
    feature_candidate_chunks: list[torch.Tensor] = []
    pair_label_chunks: list[torch.Tensor] = []
    pair_proposal_chunks: list[torch.Tensor] = []
    proposal_member_chunks: list[torch.Tensor] = []
    proposal_member_offsets = [0]

    eligible_for_negatives = eligible_mask & (
        ~ignored if exclude_ignored_from_negatives else torch.ones_like(ignored)
    )

    for proposal_occurrence, proposal_label in enumerate(selected_labels):
        positive_indices = torch.where(eligible_mask & (labels == proposal_label))[0]
        negative_mask = eligible_mask & (labels != proposal_label)
        if exclude_ignored_from_negatives:
            negative_mask &= ~ignored
        negative_candidates = torch.where(negative_mask)[0]

        pair_count = int(positive_pairs_per_proposal)
        a = _sample_with_replacement(
            positive_indices, pair_count, generator=sampling_generator
        )
        b = _sample_with_replacement(
            positive_indices, pair_count, generator=sampling_generator
        )
        # Avoid identical positive endpoints without introducing an unbounded
        # retry loop. Every proposal was checked to contain at least 2 points.
        identical = a == b
        while bool(identical.any()):
            b[identical] = _sample_with_replacement(
                positive_indices,
                int(identical.sum()),
                generator=sampling_generator,
            )
            identical = a == b

        per_pair_negatives: list[torch.Tensor] = []
        if int(num_uniform_negatives) > 0:
            uniform = _sample_with_replacement(
                negative_candidates,
                pair_count * int(num_uniform_negatives),
                generator=sampling_generator,
            ).reshape(pair_count, int(num_uniform_negatives))
            per_pair_negatives.append(uniform)

        if int(num_spatial_hard_negatives) > 0:
            assert points is not None
            pool_candidates = _sample_unique_pool(
                negative_candidates,
                int(spatial_candidate_pool),
                generator=sampling_generator,
            )

            # Use one without-replacement candidate pool per proposal. Sharing
            # it across anchors is substantially cheaper than drawing a dense
            # [pairs, all_negatives] random matrix and still makes the hard
            # mining deterministic. If a tiny synthetic proposal has fewer
            # unique negatives than requested, repeat the pool explicitly.
            hard_count = int(num_spatial_hard_negatives)
            if int(pool_candidates.numel()) < hard_count:
                repeats = (hard_count + int(pool_candidates.numel()) - 1) // int(
                    pool_candidates.numel()
                )
                pool_candidates = pool_candidates.repeat(repeats)
            pool = pool_candidates[None, :].expand(pair_count, -1)
            distances = torch.linalg.vector_norm(
                points[pool] - points[a, None, :], dim=-1
            )
            hard_local = torch.topk(
                distances,
                k=hard_count,
                dim=1,
                largest=False,
            ).indices
            hard = torch.gather(pool, 1, hard_local)
            per_pair_negatives.append(hard)

        if int(num_feature_hard_negatives) > 0:
            feature_candidates = _sample_unique_pool(
                negative_candidates,
                int(feature_candidate_pool),
                generator=sampling_generator,
            )
            target_pool_size = int(feature_candidate_pool)
            if int(feature_candidates.numel()) < target_pool_size:
                repeats = (
                    target_pool_size + int(feature_candidates.numel()) - 1
                ) // int(feature_candidates.numel())
                feature_candidates = feature_candidates.repeat(repeats)[
                    :target_pool_size
                ]
            feature_candidate_chunks.append(
                feature_candidates[None, :].expand(pair_count, -1)
            )

        pair_chunks.append(torch.stack([a, b], dim=1))
        negative_chunks.append(
            torch.cat(per_pair_negatives, dim=1)
            if per_pair_negatives
            else labels.new_empty((pair_count, 0), dtype=torch.long)
        )
        pair_label_chunks.append(proposal_label.expand(pair_count))
        pair_proposal_chunks.append(
            torch.full(
                (pair_count,),
                proposal_occurrence,
                dtype=torch.long,
                device=labels.device,
            )
        )
        proposal_member_chunks.append(positive_indices.long())
        proposal_member_offsets.append(
            proposal_member_offsets[-1] + int(positive_indices.numel())
        )

    return ContrastiveTripletBatch(
        positive_pairs=torch.cat(pair_chunks, dim=0).long(),
        negative_indices=torch.cat(negative_chunks, dim=0).long(),
        proposal_labels=torch.cat(pair_label_chunks, dim=0).long(),
        feature_candidate_indices=(
            torch.cat(feature_candidate_chunks, dim=0).long()
            if feature_candidate_chunks
            else None
        ),
        num_feature_hard_negatives=int(num_feature_hard_negatives),
        proposal_member_offsets=torch.tensor(
            proposal_member_offsets,
            dtype=torch.long,
            device=labels.device,
        ),
        proposal_member_indices=torch.cat(proposal_member_chunks, dim=0).long(),
        pair_proposal_indices=torch.cat(pair_proposal_chunks, dim=0).long(),
        eligible_indices=torch.where(eligible_for_negatives)[0].long(),
        multiscale_seed=(
            int(generator.initial_seed()) if generator is not None else 0
        ),
        num_uniform_negatives=int(num_uniform_negatives),
        num_spatial_hard_negatives=int(num_spatial_hard_negatives),
        spatial_candidate_pool=int(spatial_candidate_pool),
        feature_candidate_pool=int(feature_candidate_pool),
    )


def _sample_adaptive_spatial_negative_rows(
    *,
    candidate_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_indices: torch.Tensor,
    points: torch.Tensor,
    requested_per_row: int,
    candidate_pool: int,
    excluded_indices: torch.Tensor | None,
    generator: torch.Generator | None,
) -> AdaptiveNegativeSelection:
    """Mine duplicate-free spatial negatives from row-specific safe pools."""

    if candidate_indices.ndim != 1:
        raise ValueError("spatial candidates must be one shared vector")
    if valid_mask.shape != (anchor_indices.numel(), candidate_indices.numel()):
        raise ValueError("spatial valid_mask has the wrong shape")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if candidate_pool <= 0 and requested_per_row > 0:
        raise ValueError("candidate_pool must be positive")
    if valid_mask.device != candidate_indices.device:
        raise ValueError("spatial valid_mask must share the candidate device")
    if anchor_indices.device != candidate_indices.device:
        raise ValueError("spatial anchors and candidates must share one device")
    sampling_generator = _generator_for_device(
        generator, device=candidate_indices.device
    )

    ranked_rows: list[torch.Tensor] = []
    counts: list[int] = []
    for row in range(int(anchor_indices.numel())):
        values = candidate_indices[valid_mask[row]].long()
        values = torch.unique(values, sorted=True)
        if excluded_indices is not None and excluded_indices.shape[1] > 0:
            values = values[~torch.isin(values, excluded_indices[row].long())]
        counts.append(int(values.numel()))
        pool = sample_unique_without_replacement(
            values,
            min(int(candidate_pool), int(values.numel())),
            generator=sampling_generator,
        )
        take = min(int(requested_per_row), int(pool.numel()))
        if take == 0:
            ranked_rows.append(pool.new_empty((0,), dtype=torch.long))
            continue
        distances = torch.linalg.vector_norm(
            points[pool] - points[int(anchor_indices[row])], dim=-1
        )
        nearest = torch.topk(
            distances,
            k=take,
            largest=False,
            sorted=True,
        ).indices
        ranked_rows.append(pool[nearest])

    available = torch.tensor(counts, dtype=torch.long, device=points.device)
    selected_count = min(
        (int(row.numel()) for row in ranked_rows),
        default=0,
    )
    output = (
        torch.stack([row[:selected_count] for row in ranked_rows], dim=0)
        if ranked_rows
        else candidate_indices.new_empty((0, 0), dtype=torch.long)
    )
    return AdaptiveNegativeSelection(
        indices=output.long(),
        requested_per_row=int(requested_per_row),
        selected_per_row=selected_count,
        available_per_row=available,
    )


def sample_safe_overlapping_proposal_triplets(
    *,
    num_scene_points: int,
    visible_indices: torch.Tensor,
    proposal_offsets: torch.Tensor,
    proposal_point_indices: torch.Tensor,
    selected_proposal_indices: torch.Tensor,
    points: torch.Tensor | None = None,
    positive_pairs_per_proposal: int = 64,
    num_uniform_negatives: int = 256,
    num_spatial_hard_negatives: int = 0,
    spatial_candidate_pool: int = 1024,
    num_feature_hard_negatives: int = 0,
    feature_candidate_pool: int = 1024,
    relation_proposal_offsets: torch.Tensor | None = None,
    relation_proposal_point_indices: torch.Tensor | None = None,
    require_negative_proposal_membership: bool = False,
    defer_token_loss: bool = False,
    generator: torch.Generator | None = None,
    # Internal planner caches.  They are optional so the public/legacy call
    # contract and its validation remain unchanged.  V2 supplies these for
    # repeated groups from one physical frame to avoid rebuilding identical
    # integer membership structures on every call.
    prepared_visible_indices: torch.Tensor | None = None,
    prepared_relation_points: torch.Tensor | None = None,
    prepared_relation_index: RelationMembershipIndex | None = None,
    prepared_known_indices: torch.Tensor | None = None,
    prepared_valid_proposal_indices: torch.Tensor | None = None,
    prepared_safety_visible_indices: torch.Tensor | None = None,
    prepared_safety_relation_index: RelationMembershipIndex | None = None,
    prepared_safety_relation_offsets: torch.Tensor | None = None,
    prepared_safety_relation_points: torch.Tensor | None = None,
    deferred_safety_device: str | torch.device | None = None,
    token_positive_pair_offsets: torch.Tensor | None = None,
    token_positive_pair_indices: torch.Tensor | None = None,
    token_positive_pair_digest: str | None = None,
) -> tuple[ContrastiveTripletBatch, dict[str, Any]]:
    """Sample unique pairs and co-membership-safe negatives from raw masks.

    This is an opt-in v2 primitive; the historical with-replacement sampler is
    intentionally unchanged.  Selected proposal occurrences must be unique.
    Negative-family widths adapt downward to the smallest safe row, and no
    point ID repeats within or across the uniform, spatial, and feature-pool
    families of a row.

    ``relation_proposal_*`` may contain a larger same-frame relation graph,
    including masks from other granularities.  When omitted, the selected
    cell's complete proposal catalog defines teacher co-membership.

    ``defer_token_loss`` is the V2 token-pure fast path.  It samples and checks
    point positive pairs exactly as feasibility evidence, but records negative
    family widths analytically instead of constructing negative point IDs that
    would be discarded after voxelization.

    V2 may additionally provide an exact native-dec0 positive-pair CSR.  When
    present with deferred token loss, those rows are the authoritative pair
    plan and the expensive raw point-pair sampling/safety pass is skipped.  A
    compatibility ``positive_pairs`` tensor still carries the same pair rows;
    token-space consumers must use the explicit payload fields.
    """

    _require_integral_vector(visible_indices, name="visible_indices")
    _validate_proposal_membership_csr(proposal_offsets, proposal_point_indices)
    _require_integral_vector(
        selected_proposal_indices, name="selected_proposal_indices"
    )
    if int(num_scene_points) <= 0:
        raise ValueError("num_scene_points must be positive")
    if visible_indices.device != proposal_offsets.device:
        raise ValueError("raw proposal tensors must share one device")
    if selected_proposal_indices.device != proposal_offsets.device:
        raise ValueError("selected proposals must share the proposal device")
    if selected_proposal_indices.numel() == 0:
        raise ValueError("at least one selected proposal is required")
    if (
        torch.unique(selected_proposal_indices).numel()
        != selected_proposal_indices.numel()
    ):
        raise ValueError("selected proposal occurrences must be unique")
    if int(positive_pairs_per_proposal) <= 0:
        raise ValueError("positive_pairs_per_proposal must be positive")
    if any(
        int(value) < 0
        for value in (
            num_uniform_negatives,
            num_spatial_hard_negatives,
            num_feature_hard_negatives,
        )
    ):
        raise ValueError("negative counts must be non-negative")
    if (
        int(num_uniform_negatives)
        + int(num_spatial_hard_negatives)
        + int(num_feature_hard_negatives)
        <= 0
    ):
        raise ValueError("at least one negative family is required")
    if int(num_spatial_hard_negatives) > 0:
        if points is None:
            raise ValueError("points are required for spatial hard negatives")
        if int(spatial_candidate_pool) <= 0:
            raise ValueError("spatial_candidate_pool must be positive")
        if points.shape != (int(num_scene_points), 3):
            raise ValueError("points must have shape [num_scene_points,3]")
        if points.device != visible_indices.device:
            raise ValueError("points and proposal tensors must share one device")
    if int(num_feature_hard_negatives) > 0 and int(feature_candidate_pool) <= 0:
        raise ValueError("feature_candidate_pool must be positive")

    payload_presence = tuple(
        value is not None
        for value in (
            token_positive_pair_offsets,
            token_positive_pair_indices,
            token_positive_pair_digest,
        )
    )
    if any(payload_presence) and not all(payload_presence):
        raise ValueError(
            "native-dec0 positive-pair payload fields must be supplied together"
        )
    token_pair_payload = all(payload_presence)
    if token_pair_payload and not defer_token_loss:
        raise ValueError(
            "native-dec0 positive-pair payload requires deferred token loss"
        )

    relation_offsets = (
        proposal_offsets
        if relation_proposal_offsets is None
        else relation_proposal_offsets
    )
    # Keep the full relation CSR intact.  ``relation_unique_points`` below is
    # only a support-set cache for visibility/known accounting; it must never
    # replace the CSR member array because ``relation_offsets`` still indexes
    # every proposal occurrence (including overlap and duplicate point edges).
    relation_members = (
        proposal_point_indices
        if relation_proposal_point_indices is None
        else relation_proposal_point_indices
    )
    if (relation_proposal_offsets is None) != (
        relation_proposal_point_indices is None
    ):
        raise ValueError("relation proposal offsets and indices must be paired")
    _validate_proposal_membership_csr(relation_offsets, relation_members)
    if relation_offsets.device != proposal_offsets.device:
        raise ValueError("relation and selected proposal tensors must share one device")

    visible = (
        torch.unique(visible_indices.long(), sorted=True)
        if prepared_visible_indices is None
        else prepared_visible_indices.long()
    )
    if visible.ndim != 1 or visible.device != visible_indices.device:
        raise ValueError("prepared visible indices must share the raw vector device")
    if visible.numel() == 0 or int(visible.min()) < 0 or int(visible.max()) >= int(
        num_scene_points
    ):
        raise ValueError("visible point indices are empty or out of bounds")
    if proposal_point_indices.numel() and not bool(
        torch.isin(proposal_point_indices.long(), visible).all()
    ):
        raise ValueError("proposal membership must be a subset of visible points")
    relation_unique_points = (
        torch.unique(relation_members.long(), sorted=True)
        if prepared_relation_points is None
        else prepared_relation_points.long()
    )
    if (
        relation_unique_points.ndim != 1
        or relation_unique_points.device != visible.device
    ):
        raise ValueError("prepared relation points must share the raw vector device")
    if relation_unique_points.numel() and (
        int(relation_unique_points.min()) < 0
        or int(relation_unique_points.max()) >= int(num_scene_points)
    ):
        raise ValueError("relation points are outside the scene")
    if prepared_known_indices is None:
        relation_known_mask = torch.isin(visible, relation_unique_points)
        known_indices = visible[relation_known_mask]
        unknown_indices = visible[~relation_known_mask]
    else:
        known_indices = prepared_known_indices.long()
        if known_indices.ndim != 1 or known_indices.device != visible.device:
            raise ValueError("prepared known indices must share the raw vector device")
        unknown_indices = visible[~torch.isin(visible, known_indices)]
    supervision_eligible = (
        known_indices if require_negative_proposal_membership else visible
    )
    # An exact native-dec0 payload has already made the token-level positive
    # pair choice and relation-safe feasibility decision.  The point-space
    # membership index is only consumed by the legacy/deferred raw-pair safety
    # path below; constructing its [scene-points, proposal-words] tensor here
    # would reintroduce the expensive work the payload is meant to avoid.
    relation_index: RelationMembershipIndex | None
    if token_pair_payload:
        relation_index = prepared_relation_index
    else:
        relation_index = (
            build_relation_membership_index(
                relation_offsets,
                relation_members,
                point_capacity=int(num_scene_points),
            )
            if prepared_relation_index is None
            else prepared_relation_index
        )
    if relation_index is not None and (
        relation_index.point_words.device != visible.device
        or relation_index.point_capacity < int(num_scene_points)
        or relation_index.proposal_count != int(relation_offsets.numel()) - 1
    ):
        raise ValueError("prepared relation index does not match the raw relation CSR")
    root_seed = _sample_root_seed(generator, device=visible.device)
    # Validate on unique members: duplicated CSR edges must not make a
    # proposal look larger than the visible universe or change its pair
    # population.  This mirrors FrameProposalCatalog.valid_proposal_indices.
    selected = selected_proposal_indices.long()
    if prepared_valid_proposal_indices is None and token_pair_payload:
        # Validate only the selected raw occurrences.  The payload path does
        # not need a catalog-wide point-count scan, but it must still fail
        # closed on an out-of-range or non-representable selected occurrence.
        proposal_count = int(proposal_offsets.numel()) - 1
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= proposal_count
        ):
            raise ValueError("selected_proposal_indices contains an invalid proposal")
        for proposal_index in selected.tolist():
            start = int(proposal_offsets[int(proposal_index)])
            end = int(proposal_offsets[int(proposal_index) + 1])
            unique_size = int(
                torch.unique(proposal_point_indices[start:end].long()).numel()
            )
            if unique_size < 2 or unique_size >= int(visible.numel()):
                raise ValueError(
                    "selected_proposal_indices contains an invalid proposal"
                )
        valid_proposals = selected
    elif prepared_valid_proposal_indices is None:
        unique_sizes = _unique_proposal_member_counts(
            proposal_offsets,
            proposal_point_indices,
            point_capacity=int(num_scene_points),
        )
        valid_proposals = torch.where(
            (unique_sizes >= 2) & (unique_sizes < int(visible.numel()))
        )[0].long()
    else:
        valid_proposals = prepared_valid_proposal_indices.long()
    if valid_proposals.ndim != 1 or valid_proposals.device != visible.device:
        raise ValueError("prepared valid proposals must share the raw vector device")
    selected_validated_locally = (
        token_pair_payload and prepared_valid_proposal_indices is None
    )
    if not selected_validated_locally and not bool(
        torch.isin(selected, valid_proposals).all()
    ):
        raise ValueError("selected_proposal_indices contains an invalid proposal")

    payload_offsets = None
    payload_pairs = None
    if token_pair_payload:
        assert (
            token_positive_pair_offsets is not None
            and token_positive_pair_indices is not None
            and token_positive_pair_digest is not None
        )
        validate_token_positive_pair_payload(
            token_positive_pair_offsets,
            token_positive_pair_indices,
            token_positive_pair_digest,
            proposal_count=int(selected.numel()),
            algorithm_version=SAMPLER_ALGORITHM_VERSION,
            preselection_policy=SAMPLER_PRESELECTION_POLICY,
            proposal_labels=torch.repeat_interleave(
                selected.to(token_positive_pair_offsets.device),
                token_positive_pair_offsets.long()[1:]
                - token_positive_pair_offsets.long()[:-1],
            ),
            pair_proposal_indices=torch.repeat_interleave(
                torch.arange(
                    int(selected.numel()),
                    dtype=torch.long,
                    device=token_positive_pair_offsets.device,
                ),
                token_positive_pair_offsets.long()[1:]
                - token_positive_pair_offsets.long()[:-1],
            ),
        )
        payload_offsets = token_positive_pair_offsets.long().to(visible.device)
        payload_pairs = token_positive_pair_indices.long().to(visible.device)

    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for occurrence, selected_id in enumerate(selected.tolist()):
        proposal_index = int(selected_id)
        start = int(proposal_offsets[proposal_index])
        end = int(proposal_offsets[proposal_index + 1])
        members = torch.unique(
            proposal_point_indices[start:end].long(), sorted=True
        )
        if token_pair_payload:
            assert payload_offsets is not None and payload_pairs is not None
            pair_start = int(payload_offsets[occurrence])
            pair_end = int(payload_offsets[occurrence + 1])
            pairs = payload_pairs[pair_start:pair_end]
            if pairs.shape[0] == 0:
                dropped.append(
                    {
                        "proposal_index": proposal_index,
                        "reason": "no_native_dec0_positive_pair",
                    }
                )
                continue
            if int(pairs.shape[0]) > int(positive_pairs_per_proposal):
                raise ValueError(
                    "native-dec0 positive-pair payload exceeds requested pair budget"
                )
            # The payload's relation-safe token predicate is authoritative for
            # deferred rebuild.  Keep a conservative finite count solely for
            # compatibility diagnostics; token-space code recomputes exact
            # row-wise safe support from the same relation CSR.
            safe_counts = torch.full(
                (int(pairs.shape[0]),),
                max(1, int(supervision_eligible.numel())),
                dtype=torch.long,
                device=visible.device,
            )
            records.append(
                {
                    "proposal_index": proposal_index,
                    "members": members,
                    "pairs": pairs,
                    "uniform_width": int(num_uniform_negatives),
                    "spatial_width": min(
                        int(num_spatial_hard_negatives),
                        int(spatial_candidate_pool),
                    ),
                    "feature_width": int(feature_candidate_pool)
                    if int(num_feature_hard_negatives) > 0
                    else 0,
                    "feature_hard": min(
                        int(num_feature_hard_negatives),
                        int(feature_candidate_pool),
                    ),
                    "safe_counts": safe_counts,
                }
            )
            continue
        streams = _proposal_streams(
            root_seed=root_seed,
            proposal_index=proposal_index,
            device=visible.device,
        )
        pairs = sample_unique_positive_pairs(
            members,
            int(positive_pairs_per_proposal),
            generator=streams["positive_pairs"],
        )
        if pairs.shape[0] == 0:
            dropped.append(
                {"proposal_index": proposal_index, "reason": "no_unique_positive_pair"}
            )
            continue
        if defer_token_loss:
            # The deferred path consumes only the count of safe candidates;
            # avoid constructing two pair×candidate masks when the same
            # relation-bit union answers both endpoint queries at once.
            safety_device = (
                torch.device(deferred_safety_device)
                if deferred_safety_device is not None
                else pairs.device
            )
            if safety_device.type == "cpu" and pairs.device.type != "cpu":
                safety_visible = (
                    visible.detach().cpu()
                    if prepared_safety_visible_indices is None
                    else prepared_safety_visible_indices.long()
                )
                safety_index = (
                    relation_index
                    if prepared_safety_relation_index is None
                    else prepared_safety_relation_index
                )
                if safety_visible.device.type != "cpu" or safety_index.point_words.device.type != "cpu":
                    raise ValueError("CPU deferred safety inputs must be on CPU")
                safety_members = members.detach().cpu()
                safety_pairs = pairs.detach().cpu()
                safety_complement = safety_visible[
                    ~torch.isin(safety_visible, safety_members)
                ]
                safe_counts = co_membership_safe_negative_pair_counts(
                    safety_pairs,
                    safety_complement,
                    proposal_offsets=(
                        relation_offsets.detach().cpu()
                        if prepared_safety_relation_offsets is None
                        else prepared_safety_relation_offsets
                    ),
                    proposal_point_indices=(
                        relation_members.detach().cpu()
                        if prepared_safety_relation_points is None
                        else prepared_safety_relation_points
                    ),
                    require_candidate_membership=require_negative_proposal_membership,
                    membership_index=safety_index,
                ).to(device=pairs.device)
            else:
                complement = visible[~torch.isin(visible, members)]
                safe_counts = co_membership_safe_negative_pair_counts(
                    pairs,
                    complement,
                    proposal_offsets=relation_offsets,
                    proposal_point_indices=relation_members,
                    require_candidate_membership=require_negative_proposal_membership,
                    membership_index=relation_index,
                )
            viable_rows = safe_counts > 0
            pairs = pairs[viable_rows]
            safe_counts = safe_counts[viable_rows]
        else:
            complement = visible[~torch.isin(visible, members)]
            safe = co_membership_safe_negative_mask(
                pairs[:, 0],
                complement,
                proposal_offsets=relation_offsets,
                proposal_point_indices=relation_members,
                require_candidate_membership=require_negative_proposal_membership,
                membership_index=relation_index,
            ) & co_membership_safe_negative_mask(
                pairs[:, 1],
                complement,
                proposal_offsets=relation_offsets,
                proposal_point_indices=relation_members,
                require_candidate_membership=require_negative_proposal_membership,
                membership_index=relation_index,
            )
            viable_rows = safe.any(dim=1)
            pairs = pairs[viable_rows]
            safe = safe[viable_rows]
            safe_counts = safe.sum(dim=1).long()
        if pairs.shape[0] == 0:
            dropped.append(
                {"proposal_index": proposal_index, "reason": "no_safe_negative"}
            )
            continue

        if defer_token_loss:
            minimum_safe = int(safe_counts.min())
            uniform_width_for_record = min(
                int(num_uniform_negatives), minimum_safe
            )
            remaining_after_uniform = max(
                minimum_safe - uniform_width_for_record, 0
            )
            spatial_width_for_record = min(
                int(num_spatial_hard_negatives),
                int(spatial_candidate_pool),
                remaining_after_uniform,
            )
            remaining_after_spatial = max(
                remaining_after_uniform - spatial_width_for_record, 0
            )
            feature_width_for_record = min(
                int(feature_candidate_pool), remaining_after_spatial
            )
            if int(num_feature_hard_negatives) <= 0:
                feature_width_for_record = 0
            feature_hard_for_record = min(
                int(num_feature_hard_negatives), feature_width_for_record
            )
            if (
                uniform_width_for_record
                + spatial_width_for_record
                + feature_hard_for_record
                == 0
            ):
                dropped.append(
                    {"proposal_index": proposal_index, "reason": "adaptive_k_is_zero"}
                )
                continue
            records.append(
                {
                    "proposal_index": proposal_index,
                    "members": members,
                    "pairs": pairs,
                    "uniform_width": uniform_width_for_record,
                    "spatial_width": spatial_width_for_record,
                    "feature_width": feature_width_for_record,
                    "feature_hard": feature_hard_for_record,
                    "safe_counts": safe_counts,
                }
            )
            continue

        uniform = sample_adaptive_unique_negative_rows(
            complement,
            int(num_uniform_negatives),
            num_rows=int(pairs.shape[0]),
            valid_mask=safe,
            generator=streams["uniform"],
        )
        excluded = uniform.indices
        if int(num_spatial_hard_negatives) > 0:
            assert points is not None
            spatial = _sample_adaptive_spatial_negative_rows(
                candidate_indices=complement,
                valid_mask=safe,
                anchor_indices=pairs[:, 0],
                points=points,
                requested_per_row=int(num_spatial_hard_negatives),
                candidate_pool=int(spatial_candidate_pool),
                excluded_indices=excluded,
                generator=streams["spatial"],
            )
        else:
            spatial = AdaptiveNegativeSelection(
                indices=complement.new_empty((pairs.shape[0], 0), dtype=torch.long),
                requested_per_row=0,
                selected_per_row=0,
                available_per_row=safe_counts,
            )
        excluded = torch.cat([uniform.indices, spatial.indices], dim=1)
        if int(num_feature_hard_negatives) > 0:
            feature = sample_adaptive_unique_negative_rows(
                complement,
                int(feature_candidate_pool),
                num_rows=int(pairs.shape[0]),
                valid_mask=safe,
                excluded_indices=excluded,
                generator=streams["feature"],
            )
        else:
            feature = AdaptiveNegativeSelection(
                indices=complement.new_empty((pairs.shape[0], 0), dtype=torch.long),
                requested_per_row=0,
                selected_per_row=0,
                available_per_row=safe_counts,
            )
        if (
            uniform.selected_per_row
            + spatial.selected_per_row
            + min(int(num_feature_hard_negatives), feature.selected_per_row)
            == 0
        ):
            dropped.append(
                {"proposal_index": proposal_index, "reason": "adaptive_k_is_zero"}
            )
            continue
        records.append(
            {
                "proposal_index": proposal_index,
                "members": members,
                "pairs": pairs,
                "uniform": uniform,
                "spatial": spatial,
                "feature": feature,
                "safe_counts": safe_counts,
            }
        )

    if not records:
        raise ValueError("no selected proposal retained a safe contrastive row")
    if defer_token_loss:
        uniform_width = min(int(record["uniform_width"]) for record in records)
        spatial_width = min(int(record["spatial_width"]) for record in records)
        feature_width = min(int(record["feature_width"]) for record in records)
        feature_hard = min(
            int(num_feature_hard_negatives),
            feature_width,
            min(int(record["feature_hard"]) for record in records),
        )
    else:
        uniform_width = min(
            int(record["uniform"].selected_per_row) for record in records
        )
        spatial_width = min(
            int(record["spatial"].selected_per_row) for record in records
        )
        feature_width = min(
            int(record["feature"].selected_per_row) for record in records
        )
        feature_hard = min(int(num_feature_hard_negatives), feature_width)
    if uniform_width + spatial_width + feature_hard == 0:
        raise ValueError("adaptive negative widths are jointly zero")

    pair_chunks: list[torch.Tensor] = []
    negative_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    candidate_chunks: list[torch.Tensor] = []
    pair_proposal_chunks: list[torch.Tensor] = []
    member_chunks: list[torch.Tensor] = []
    member_offsets = [0]
    safe_counts: list[torch.Tensor] = []
    for occurrence, record in enumerate(records):
        pairs = record["pairs"]
        pair_chunks.append(pairs)
        if not defer_token_loss:
            negative_chunks.append(
                torch.cat(
                    [
                        record["uniform"].indices[:, :uniform_width],
                        record["spatial"].indices[:, :spatial_width],
                    ],
                    dim=1,
                )
            )
            if feature_hard > 0:
                candidate_chunks.append(
                    record["feature"].indices[:, :feature_width]
                )
        label_chunks.append(
            torch.full(
                (pairs.shape[0],),
                int(record["proposal_index"]),
                dtype=torch.long,
                device=pairs.device,
            )
        )
        pair_proposal_chunks.append(
            torch.full(
                (pairs.shape[0],),
                occurrence,
                dtype=torch.long,
                device=pairs.device,
            )
        )
        members = record["members"]
        member_chunks.append(members)
        member_offsets.append(member_offsets[-1] + int(members.numel()))
        safe_counts.append(record["safe_counts"])

    batch_kwargs: dict[str, Any] = dict(
        positive_pairs=torch.cat(pair_chunks, dim=0).long(),
        negative_indices=(
            torch.cat(negative_chunks, dim=0).long()
            if negative_chunks
            else torch.empty(
                (sum(int(chunk.shape[0]) for chunk in pair_chunks), 0),
                dtype=torch.long,
                device=visible.device,
            )
        ),
        proposal_labels=torch.cat(label_chunks, dim=0).long(),
        feature_candidate_indices=(
            torch.cat(candidate_chunks, dim=0).long()
            if candidate_chunks
            else None
        ),
        num_feature_hard_negatives=feature_hard,
        proposal_member_offsets=torch.tensor(
            member_offsets,
            dtype=torch.long,
            device=visible.device,
        ),
        proposal_member_indices=torch.cat(member_chunks, dim=0).long(),
        pair_proposal_indices=torch.cat(pair_proposal_chunks, dim=0).long(),
        eligible_indices=supervision_eligible,
        relation_proposal_offsets=relation_offsets,
        relation_proposal_member_indices=relation_members,
        multiscale_seed=root_seed,
        num_uniform_negatives=uniform_width,
        num_spatial_hard_negatives=spatial_width,
        spatial_candidate_pool=int(spatial_candidate_pool),
        feature_candidate_pool=feature_width,
        deferred_token_loss=(
            DeferredTokenLossPlan(
                requested_positive_pairs_per_proposal=int(
                    positive_pairs_per_proposal
                ),
                requested_uniform_negatives_per_pair=int(num_uniform_negatives),
                requested_spatial_hard_negatives_per_pair=int(
                    num_spatial_hard_negatives
                ),
                requested_feature_hard_negatives_per_pair=int(
                    num_feature_hard_negatives
                ),
                effective_uniform_negatives_per_pair=uniform_width,
                effective_spatial_hard_negatives_per_pair=spatial_width,
                effective_feature_candidate_pool_per_pair=feature_width,
                effective_feature_hard_negatives_per_pair=feature_hard,
                require_negative_proposal_membership=(
                    require_negative_proposal_membership
                ),
            )
            if defer_token_loss
            else None
        ),
        algorithm_version=SAMPLER_ALGORITHM_VERSION,
        require_negative_proposal_membership=(
            require_negative_proposal_membership
        ),
    )
    if token_pair_payload:
        assert (
            payload_offsets is not None
            and payload_pairs is not None
            and token_positive_pair_digest is not None
        )
        payload_fields = set(
            getattr(ContrastiveTripletBatch, "__dataclass_fields__", {})
        )
        payload_values = {
            "token_positive_pair_offsets": payload_offsets,
            "token_positive_pair_indices": payload_pairs,
            "token_positive_pair_digest": token_positive_pair_digest,
        }
        if set(payload_values).issubset(payload_fields):
            batch_kwargs.update(payload_values)
    batch = ContrastiveTripletBatch(**batch_kwargs)
    if token_pair_payload:
        # Keep a compatibility transport for the short interval in which an
        # older local batch dataclass is paired with this sampler.  The shared
        # V2 dataclass owns these fields once available; object.__setattr__ is
        # intentionally limited to this frozen, per-call compatibility case.
        for name, value in payload_values.items():
            if not hasattr(batch, name):
                object.__setattr__(batch, name, value)
    all_safe_counts = torch.cat(safe_counts, dim=0)
    metadata: dict[str, Any] = {
        "requested_proposal_indices": [int(value) for value in selected.tolist()],
        "retained_proposal_indices": [
            int(record["proposal_index"]) for record in records
        ],
        "dropped_proposals": dropped,
        "requested_positive_pairs_per_proposal": int(
            positive_pairs_per_proposal
        ),
        "realized_positive_pairs": int(batch.num_pairs),
        "safe_negatives_per_pair_min": int(all_safe_counts.min()),
        "safe_negatives_per_pair_max": int(all_safe_counts.max()),
        "safe_negatives_per_pair_mean": float(all_safe_counts.float().mean()),
        "uniform": {
            "requested_per_pair": int(num_uniform_negatives),
            "selected_per_pair": uniform_width,
        },
        "spatial_hard": {
            "requested_per_pair": int(num_spatial_hard_negatives),
            "selected_per_pair": spatial_width,
        },
        "feature_hard": {
            "requested_per_pair": int(num_feature_hard_negatives),
            "candidate_pool_per_pair": feature_width,
            "selected_per_pair": feature_hard,
        },
        "negative_ids_unique_within_and_across_families": True,
        "point_negative_ids_materialized": not bool(defer_token_loss),
        "point_negative_sampling_skipped": bool(defer_token_loss),
        "teacher_co_membership_filtered": True,
        "visible_count": int(visible.numel()),
        "known_count": int(known_indices.numel()),
        "unknown_count": int(unknown_indices.numel()),
        "eligible_count": int(supervision_eligible.numel()),
        "relation_proposal_count": int(relation_offsets.numel()) - 1,
        "require_negative_proposal_membership": bool(
            require_negative_proposal_membership
        ),
        "deferred_token_loss": (
            batch.deferred_token_loss.metadata()
            if batch.deferred_token_loss is not None
            else None
        ),
        "algorithm_version": SAMPLER_ALGORITHM_VERSION,
        "seed_scheme": SAMPLER_SEED_SCHEME,
        "root_seed": root_seed,
        "token_positive_pair_payload": bool(token_pair_payload),
        "token_positive_pair_digest": (
            token_positive_pair_digest if token_pair_payload else None
        ),
        "token_positive_pair_rows": int(batch.num_pairs)
        if token_pair_payload
        else 0,
    }
    return batch, metadata
