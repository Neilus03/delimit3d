"""Strict transfer of an RGBN6 V2 checkpoint into a LitePT-only artifact."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import torch


V2_CHECKPOINT_SCHEMA = "litept_rgbn6_multigranular_pretrain_checkpoint/v2"
BN_RECALIBRATION_SCHEMA = "litept_rgbn6_bn_recalibration/v2"
BACKBONE_EXPORT_SCHEMA = "litept_rgbn6_v2_backbone_export/v1"
CANONICAL_LITEPT_S_STAR_TENSOR_COUNT = 302
BACKBONE_EXPORT_FIELDS = frozenset(
    {
        "schema_version",
        "backbone_state_dict",
        "input_features",
        "input_channels",
        "litept_variant",
        "tensor_count",
        "bn_buffer_count",
        "tensor_state_sha256",
        "bn_buffer_policy",
    }
)
BN_BUFFER_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)
DISPOSABLE_MODEL_PREFIXES = (
    "projector.",
    "criterion.",
    "multiscale_projectors.",
    "multiscale_criteria.",
    "hierarchy_mask_supervisor.",
    "regularizer.",
)
FORBIDDEN_BACKBONE_TERMS = (
    "projector",
    "query",
    "temperature",
    "criterion",
    "regularizer",
    "hierarchy_mask_supervisor",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Use the exact hash recipe consumed by Mask3D's strict initializer."""

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalized_tensor_mapping(
    value: Any,
    *,
    label: str,
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a tensor mapping")
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for raw_key, raw_tensor in value.items():
        key = str(raw_key).removeprefix("module.")
        if not torch.is_tensor(raw_tensor):
            raise TypeError(f"{label}[{raw_key!r}] is not a tensor")
        if key in output:
            raise ValueError(f"{label} has a duplicate normalized key {key!r}")
        output[key] = raw_tensor.detach().cpu().clone()
    if not output:
        raise ValueError(f"{label} is empty")
    return output


def _is_bn_buffer(key: str) -> bool:
    return str(key).endswith(BN_BUFFER_SUFFIXES)


def _state_differences(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    structural: list[str] = []
    byte_differences: list[str] = []
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            structural.append(key)
            continue
        if (
            tuple(left[key].shape) != tuple(right[key].shape)
            or left[key].dtype != right[key].dtype
        ):
            structural.append(key)
        elif not torch.equal(left[key], right[key]):
            byte_differences.append(key)
    return structural, byte_differences


def _validate_v2_resolved_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("V2 checkpoint lacks resolved_config")
    expected = {
        "input_features": "rgbn6",
        "input_channels": 6,
        "voxel_reduce": "representative",
        "representative_sampling": "first",
        "sampling_mode": "coverage_multigranular_v2",
        "token_pure_dec0": True,
        "coordinate_normalization_policy": (
            "structured3d_local_y_up_to_scannet_z_up_v1"
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": resolved.get(key)}
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint is not canonical RGBN6 V2: {mismatches}")
    normalization = resolved.get("normalization")
    if (
        not isinstance(normalization, Mapping)
        or normalization.get("policy") not in {"batchnorm", "sync_batchnorm"}
    ):
        raise ValueError("checkpoint lacks a supported V2 normalization policy")
    return {
        **expected,
        "pretraining_recipe": resolved.get("pretraining_recipe"),
        "normalization": dict(normalization),
    }


def extract_v2_backbone(
    checkpoint_path: Path,
    *,
    expected_tensor_count: int = CANONICAL_LITEPT_S_STAR_TENSOR_COUNT,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Extract and audit canonical wrapper keys from one raw/recalibrated V2 file.

    ``model_state_dict`` is authoritative because the BN recalibration sidecar
    updates that mapping. The checkpoint's duplicated backbone/PointGroup maps
    are audited; only declared BN-buffer byte drift is accepted after
    recalibration.
    """

    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: checkpoint payload is not a mapping")
    if payload.get("schema_version") != V2_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{path}: expected {V2_CHECKPOINT_SCHEMA!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    input_contract = _validate_v2_resolved_config(payload)
    full = _normalized_tensor_mapping(
        payload.get("model_state_dict"),
        label="model_state_dict",
    )
    backbone: OrderedDict[str, torch.Tensor] = OrderedDict()
    excluded: list[str] = []
    unknown_non_backbone: list[str] = []
    for key, tensor in full.items():
        if key.startswith("backbone."):
            canonical = key.removeprefix("backbone.")
            lowered = canonical.lower()
            forbidden = [term for term in FORBIDDEN_BACKBONE_TERMS if term in lowered]
            if forbidden:
                raise ValueError(
                    f"forbidden auxiliary term entered backbone key {key!r}: {forbidden}"
                )
            backbone[canonical] = tensor
            continue
        excluded.append(key)
        if not key.startswith(DISPOSABLE_MODEL_PREFIXES):
            unknown_non_backbone.append(key)
    if unknown_non_backbone:
        raise ValueError(
            "unclassified non-backbone model tensors would be silently dropped: "
            f"{unknown_non_backbone[:20]}"
        )
    if not backbone or not all(key.startswith("model.") for key in backbone):
        raise ValueError("canonical LitePT wrapper keys must all begin with 'model.'")
    if len(backbone) != int(expected_tensor_count):
        raise ValueError(
            f"canonical LitePT tensor count drift: {len(backbone)} != "
            f"{int(expected_tensor_count)}"
        )
    bn_keys = sorted(key for key in backbone if _is_bn_buffer(key))
    if not bn_keys:
        raise ValueError("canonical LitePT backbone contains no BN buffers")

    recalibration = payload.get("bn_recalibration")
    recalibrated = recalibration is not None
    if recalibrated and (
        not isinstance(recalibration, Mapping)
        or recalibration.get("schema_version") != BN_RECALIBRATION_SCHEMA
        or recalibration.get("non_bn_state_unchanged") is not True
        or recalibration.get("passed") is not True
        or not isinstance(recalibration.get("changed_source_state_keys"), list)
    ):
        raise ValueError("malformed or untrusted embedded BN recalibration record")

    duplicate = _normalized_tensor_mapping(
        payload.get("backbone_state_dict"),
        label="backbone_state_dict",
    )
    duplicate_structural, duplicate_bytes = _state_differences(backbone, duplicate)
    if duplicate_structural:
        raise ValueError(
            f"backbone_state_dict structural drift: {duplicate_structural[:20]}"
        )
    non_bn_duplicate_drift = [key for key in duplicate_bytes if not _is_bn_buffer(key)]
    if non_bn_duplicate_drift or (duplicate_bytes and not recalibrated):
        raise ValueError(
            "duplicate backbone mapping differs outside declared recalibration: "
            f"{duplicate_bytes[:20]}"
        )
    if recalibrated:
        declared_changed = {
            str(key).removeprefix("module.").removeprefix("backbone.")
            for key in recalibration["changed_source_state_keys"]
        }
        invalid_declared = sorted(
            key for key in declared_changed if not _is_bn_buffer(key)
        )
        undeclared_duplicate_drift = sorted(set(duplicate_bytes) - declared_changed)
        if invalid_declared or undeclared_duplicate_drift:
            raise ValueError(
                "embedded recalibration does not explain BN duplicate drift: "
                f"non_bn_declared={invalid_declared} "
                f"undeclared={undeclared_duplicate_drift}"
            )

    pointgroup = _normalized_tensor_mapping(
        payload.get("pointgroup_model_state_dict"),
        label="pointgroup_model_state_dict",
    )
    expected_pointgroup = OrderedDict(
        (f"backbone.{key.removeprefix('model.')}", value)
        for key, value in backbone.items()
    )
    pointgroup_structural, pointgroup_bytes = _state_differences(
        expected_pointgroup,
        pointgroup,
    )
    if pointgroup_structural:
        raise ValueError(
            f"pointgroup_model_state_dict structural drift: {pointgroup_structural[:20]}"
        )
    non_bn_pointgroup_drift = [
        key for key in pointgroup_bytes if not _is_bn_buffer(key)
    ]
    if non_bn_pointgroup_drift or (pointgroup_bytes and not recalibrated):
        raise ValueError(
            "PointGroup duplicate differs outside declared recalibration: "
            f"{pointgroup_bytes[:20]}"
        )
    if recalibrated:
        declared_pointgroup = {
            f"backbone.{key.removeprefix('model.')}" for key in declared_changed
        }
        undeclared_pointgroup_drift = sorted(
            set(pointgroup_bytes) - declared_pointgroup
        )
        if undeclared_pointgroup_drift:
            raise ValueError(
                "embedded recalibration does not explain PointGroup BN drift: "
                f"{undeclared_pointgroup_drift[:20]}"
            )

    report = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_schema": V2_CHECKPOINT_SCHEMA,
        "experiment_id": payload.get("experiment_id"),
        "seed": int(payload.get("seed", -1)),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "epoch": int(payload.get("epoch", -1)),
        "update": int(payload.get("update", -1)),
        "input_contract": input_contract,
        "authoritative_source": "model_state_dict.backbone",
        "tensor_count": len(backbone),
        "tensor_state_sha256": tensor_state_sha256(backbone),
        "bn_buffer_count": len(bn_keys),
        "bn_buffer_keys": bn_keys,
        "bn_state_sha256": tensor_state_sha256(
            {key: backbone[key] for key in bn_keys}
        ),
        "non_bn_state_sha256": tensor_state_sha256(
            {key: value for key, value in backbone.items() if key not in bn_keys}
        ),
        "recalibrated": recalibrated,
        "recalibration_schema": (
            recalibration.get("schema_version")
            if isinstance(recalibration, Mapping)
            else None
        ),
        "stale_duplicate_bn_keys": sorted(duplicate_bytes),
        "stale_pointgroup_bn_keys": sorted(pointgroup_bytes),
        "excluded_source_tensor_count": len(excluded),
        "excluded_source_tensor_keys": sorted(excluded),
        "unknown_non_backbone_tensor_keys": [],
        "passed": True,
    }
    return backbone, report


def _validate_reference(
    state: Mapping[str, torch.Tensor],
    reference_path: Path,
    *,
    expected_tensor_count: int,
) -> dict[str, Any]:
    reference, report = extract_v2_backbone(
        reference_path,
        expected_tensor_count=expected_tensor_count,
    )
    structural, _ = _state_differences(state, reference)
    if structural:
        raise ValueError(f"canonical reference key/shape/dtype drift: {structural[:20]}")
    return {
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "key_shape_dtype_match": True,
        "tensor_count": len(reference),
    }


def _atomic_torch_save(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_v2_backbone(
    *,
    weights_checkpoint: Path,
    output_checkpoint: Path,
    audit_path: Path,
    bn_checkpoint: Path | None = None,
    canonical_reference: Path | None = None,
    expected_tensor_count: int = CANONICAL_LITEPT_S_STAR_TENSOR_COUNT,
) -> dict[str, Any]:
    """Export LitePT parameters with explicitly selected BN buffers only."""

    destination = Path(output_checkpoint).expanduser().resolve()
    audit_destination = Path(audit_path).expanduser().resolve()
    if destination == audit_destination:
        raise ValueError("checkpoint and audit outputs must differ")
    for path in (destination, audit_destination):
        if path.exists():
            raise FileExistsError(path)
    weights, weights_report = extract_v2_backbone(
        weights_checkpoint,
        expected_tensor_count=expected_tensor_count,
    )
    bn_path = Path(bn_checkpoint or weights_checkpoint)
    bn_state, bn_report = extract_v2_backbone(
        bn_path,
        expected_tensor_count=expected_tensor_count,
    )
    structural, _ = _state_differences(weights, bn_state)
    if structural:
        raise ValueError(f"weights and BN checkpoints are structurally different: {structural}")
    weights_non_bn = {
        key: value for key, value in weights.items() if not _is_bn_buffer(key)
    }
    bn_non_bn = {
        key: value for key, value in bn_state.items() if not _is_bn_buffer(key)
    }
    if tensor_state_sha256(weights_non_bn) != tensor_state_sha256(bn_non_bn):
        raise ValueError(
            "BN source checkpoint carries different non-BN representation weights"
        )
    source_identity_fields = ("experiment_id", "seed", "epoch", "update")
    source_identity_drift = {
        key: {
            "weights": weights_report.get(key),
            "bn": bn_report.get(key),
        }
        for key in source_identity_fields
        if weights_report.get(key) != bn_report.get(key)
    }
    if (
        source_identity_drift
        or weights_report["input_contract"] != bn_report["input_contract"]
    ):
        raise ValueError(
            "BN source is not the same V2 checkpoint identity as the weights: "
            f"{source_identity_drift}"
        )
    chosen: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in weights.items():
        chosen[key] = (bn_state[key] if _is_bn_buffer(key) else tensor).clone()
    forbidden_output = [
        key
        for key in chosen
        if any(term in key.lower() for term in FORBIDDEN_BACKBONE_TERMS)
    ]
    if forbidden_output:
        raise ValueError(f"forbidden tensors entered transfer artifact: {forbidden_output}")
    reference_report = (
        _validate_reference(
            chosen,
            canonical_reference,
            expected_tensor_count=expected_tensor_count,
        )
        if canonical_reference is not None
        else None
    )
    state_hash = tensor_state_sha256(chosen)
    artifact = {
        "schema_version": BACKBONE_EXPORT_SCHEMA,
        "backbone_state_dict": chosen,
        "input_features": "rgbn6",
        "input_channels": 6,
        "litept_variant": "litept_s_star",
        "tensor_count": len(chosen),
        "bn_buffer_count": sum(_is_bn_buffer(key) for key in chosen),
        "tensor_state_sha256": state_hash,
        "bn_buffer_policy": (
            "selected_from_weights_checkpoint"
            if Path(bn_path).resolve() == Path(weights_checkpoint).resolve()
            else "selected_from_separate_matching_checkpoint"
        ),
    }
    _atomic_torch_save(destination, artifact)
    reloaded = _torch_load(destination)
    if not isinstance(reloaded, Mapping) or reloaded.get("schema_version") != (
        BACKBONE_EXPORT_SCHEMA
    ):
        raise RuntimeError("exported backbone schema failed round-trip")
    reloaded_state = _normalized_tensor_mapping(
        reloaded.get("backbone_state_dict"),
        label="exported backbone_state_dict",
    )
    if tensor_state_sha256(reloaded_state) != state_hash:
        raise RuntimeError("exported backbone bytes failed round-trip")
    report = {
        "schema_version": BACKBONE_EXPORT_SCHEMA,
        "output_checkpoint": str(destination),
        "output_checkpoint_sha256": sha256_file(destination),
        "output_tensor_state_sha256": state_hash,
        "tensor_count": len(chosen),
        "weights_source": weights_report,
        "bn_source": bn_report,
        "bn_buffer_policy": artifact["bn_buffer_policy"],
        "bn_buffer_count": sum(_is_bn_buffer(key) for key in chosen),
        "non_bn_weights_identical_between_sources": True,
        "weights_and_bn_checkpoint_identity_match": True,
        "canonical_reference": reference_report,
        "strict_downstream_loader_contract": {
            "top_level_field": "backbone_state_dict",
            "canonical_key_prefix": "model.",
            "source_duplicate_key_shape_dtype_match": True,
            "canonical_reference_key_shape_dtype_match": (
                True if reference_report is not None else None
            ),
            "live_litept_model_strict_load": "deferred_to_downstream_preflight",
            "round_trip_tensor_hash_match": True,
        },
        "rejected_output_families": list(FORBIDDEN_BACKBONE_TERMS),
        "forbidden_output_tensor_keys": [],
        "decoder_tensor_count": 0,
        "passed": True,
    }
    _atomic_json(audit_destination, report)
    return report


def validate_backbone_export(
    path: Path,
    *,
    expected_tensor_count: int = CANONICAL_LITEPT_S_STAR_TENSOR_COUNT,
) -> dict[str, Any]:
    """Validate a transfer artifact before using it in a probe manifest."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = _torch_load(source)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != (
        BACKBONE_EXPORT_SCHEMA
    ):
        raise ValueError(f"not a V2 RGBN6 backbone export: {source}")
    extra_fields = sorted(set(payload) - BACKBONE_EXPORT_FIELDS)
    missing_fields = sorted(BACKBONE_EXPORT_FIELDS - set(payload))
    if extra_fields or missing_fields:
        raise ValueError(
            "strict backbone export top-level schema drift: "
            f"extra={extra_fields} missing={missing_fields}"
        )
    state = _normalized_tensor_mapping(
        payload.get("backbone_state_dict"),
        label="backbone_state_dict",
    )
    forbidden = [
        key
        for key in state
        if any(term in key.lower() for term in FORBIDDEN_BACKBONE_TERMS)
    ]
    expected_hash = str(payload.get("tensor_state_sha256", ""))
    observed_hash = tensor_state_sha256(state)
    bn_count = sum(_is_bn_buffer(key) for key in state)
    contract_ok = (
        payload.get("input_features") == "rgbn6"
        and payload.get("input_channels") == 6
        and payload.get("litept_variant") == "litept_s_star"
        and all(key.startswith("model.") for key in state)
        and len(state) == int(expected_tensor_count)
        and int(payload.get("tensor_count", -1)) == len(state)
        and int(payload.get("bn_buffer_count", -1)) == bn_count
        and bn_count > 0
    )
    if forbidden or observed_hash != expected_hash or not contract_ok:
        raise ValueError(
            f"invalid V2 backbone export: forbidden={forbidden} hash_match="
            f"{observed_hash == expected_hash} contract_ok={contract_ok}"
        )
    return {
        "path": str(source),
        "file_sha256": sha256_file(source),
        "tensor_state_sha256": observed_hash,
        "tensor_count": len(state),
        "bn_buffer_count": bn_count,
        "schema_version": BACKBONE_EXPORT_SCHEMA,
        "passed": True,
    }


__all__ = [
    "BACKBONE_EXPORT_SCHEMA",
    "BACKBONE_EXPORT_FIELDS",
    "BN_BUFFER_SUFFIXES",
    "BN_RECALIBRATION_SCHEMA",
    "CANONICAL_LITEPT_S_STAR_TENSOR_COUNT",
    "DISPOSABLE_MODEL_PREFIXES",
    "FORBIDDEN_BACKBONE_TERMS",
    "V2_CHECKPOINT_SCHEMA",
    "export_v2_backbone",
    "extract_v2_backbone",
    "sha256_file",
    "tensor_state_sha256",
    "validate_backbone_export",
]
