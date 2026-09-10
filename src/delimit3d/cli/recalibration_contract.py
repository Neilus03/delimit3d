#!/usr/bin/env python3
"""Shared scene-selection and normal-audit gates for RGBN6 BN recalibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SELECTION_SCHEMA_VERSION = (
    "structured3d_rgbn6_bn_recalibration_scene_selection/v1"
)
SELECTION_NAMESPACE = "bn-recalibration-v2"
SELECTION_KEY_TEMPLATE = "bn-recalibration-v2/{seed}/{scene_id}"
SELECTION_ALGORITHM = "ascending_sha256_hex_then_scene_id"
DEFAULT_SCENE_COUNT = 48
NORMAL_AUDIT_SCHEMA_VERSION = "structured3d_rgbn6_normal_contract_audit/v2"
REQUIRED_NORMAL_AUDIT_SUBGATES = (
    "scannet_reference_config",
    "transform_parity",
    "structured3d_exporter_coordinate_contract",
    "pretraining_scene_loader_coordinate_path",
    "cross_dataset_coordinate_contract",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scene_ids_sha256(scene_ids: Sequence[str]) -> str:
    """Hash the ordered scene list using its canonical newline representation."""

    encoded = "".join(f"{scene_id}\n" for scene_id in scene_ids).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_scene_ids(split_path: Path) -> list[str]:
    path = split_path.resolve(strict=True)
    scene_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scene_ids:
        raise ValueError(f"BN recalibration split is empty: {path}")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("BN recalibration split contains duplicate scenes")
    invalid = sorted(
        scene_id
        for scene_id in scene_ids
        if Path(scene_id).name != scene_id or scene_id in {".", ".."}
    )
    if invalid:
        raise ValueError(f"BN recalibration split contains invalid scene IDs: {invalid[:10]}")
    return scene_ids


def select_bn_recalibration_scene_ids(
    scene_ids: Sequence[str], *, scene_count: int, seed: int
) -> list[str]:
    """Return exactly ``scene_count`` IDs under the frozen V2 SHA256 ordering."""

    values = list(scene_ids)
    if not values:
        raise ValueError("BN recalibration candidate scene list is empty")
    if len(set(values)) != len(values):
        raise ValueError("BN recalibration candidate scene list contains duplicates")
    if scene_count <= 0 or scene_count > len(values):
        raise ValueError(
            f"Requested {scene_count} recalibration scenes from split of {len(values)}"
        )

    def ordering_key(scene_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{SELECTION_NAMESPACE}/{int(seed)}/{scene_id}".encode("utf-8")
        ).hexdigest()
        return digest, scene_id

    return sorted(values, key=ordering_key)[:scene_count]


def build_bn_recalibration_scene_selection(
    split_path: Path, *, scene_count: int, seed: int
) -> dict[str, Any]:
    """Build a self-verifying record for an exact deterministic selection."""

    path = split_path.resolve(strict=True)
    candidates = read_scene_ids(path)
    selected = select_bn_recalibration_scene_ids(
        candidates, scene_count=int(scene_count), seed=int(seed)
    )
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "namespace": SELECTION_NAMESPACE,
        "key_template": SELECTION_KEY_TEMPLATE,
        "algorithm": SELECTION_ALGORITHM,
        "seed": int(seed),
        "scene_count": int(scene_count),
        "source_scene_count": len(candidates),
        "source_split": {
            "path": str(path),
            "sha256": sha256_file(path),
        },
        "selected_scene_ids": selected,
        "selected_scene_ids_sha256": scene_ids_sha256(selected),
    }


def _require_exact_selection_record(
    record: Mapping[str, Any],
    *,
    split_path: Path,
    scene_count: int,
    seed: int,
) -> dict[str, Any]:
    expected = build_bn_recalibration_scene_selection(
        split_path, scene_count=scene_count, seed=seed
    )
    for key in (
        "schema_version",
        "namespace",
        "key_template",
        "algorithm",
        "seed",
        "scene_count",
        "source_scene_count",
        "selected_scene_ids",
        "selected_scene_ids_sha256",
    ):
        if record.get(key) != expected[key]:
            raise ValueError(f"BN recalibration normal-audit selection drift: {key}")
    source_split = record.get("source_split")
    if not isinstance(source_split, Mapping):
        raise ValueError("BN recalibration normal audit lacks its source split record")
    try:
        observed_path = Path(str(source_split.get("path"))).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("BN recalibration normal-audit split is unavailable") from exc
    if source_split.get("sha256") != expected["source_split"]["sha256"]:
        raise ValueError("BN recalibration normal-audit split hash drift")
    # The finalizer intentionally stages the frozen split on the allocated
    # node.  Path equality would reject that byte-identical copy even though
    # the hash, deterministic ordering, and selected IDs above already prove
    # its identity.  Keep both paths as provenance, but use content identity
    # for the runtime contract.
    expected_split = expected["source_split"]
    expected_split["audit_path"] = str(observed_path)
    expected_split["runtime_path"] = str(Path(expected_split["path"]))
    expected_split["relocated"] = (
        observed_path != Path(expected_split["path"])
    )
    return expected


def validate_bn_recalibration_normal_audit(
    payload: Mapping[str, Any],
    *,
    source_root: Path,
    train_split: Path,
    scene_count: int,
    seed: int,
    expected_train_split_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless an audit exactly covers the recalibration selection."""

    if (
        payload.get("schema_version") != NORMAL_AUDIT_SCHEMA_VERSION
        or payload.get("passed") is not True
    ):
        raise ValueError("RGBN6 normal audit failed or drifted")
    for subgate in REQUIRED_NORMAL_AUDIT_SUBGATES:
        value = payload.get(subgate)
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise ValueError(f"RGBN6 normal audit lacks passed subgate: {subgate}")

    resolved_root = source_root.resolve(strict=True)
    try:
        audited_root = Path(str(payload.get("source_root"))).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("BN recalibration normal audit lacks an available source root") from exc
    selection = payload.get("scene_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Normal audit lacks a BN recalibration scene-selection record")
    expected = _require_exact_selection_record(
        selection,
        split_path=train_split,
        scene_count=int(scene_count),
        seed=int(seed),
    )
    split_sha256 = str(expected["source_split"]["sha256"])
    if (
        expected_train_split_sha256 is not None
        and split_sha256 != expected_train_split_sha256
    ):
        raise ValueError("BN recalibration split differs from the frozen data manifest")

    split_records = payload.get("split_records")
    if not isinstance(split_records, list) or len(split_records) != 1:
        raise ValueError("BN recalibration normal audit must pin exactly one source split")
    split_record = split_records[0]
    if not isinstance(split_record, Mapping):
        raise ValueError("BN recalibration normal-audit split record is invalid")
    try:
        split_record_path = Path(str(split_record.get("path"))).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("BN recalibration normal-audit split record is unavailable") from exc
    source_split = selection.get("source_split")
    if not isinstance(source_split, Mapping):
        raise ValueError("BN recalibration normal audit lacks its source split record")
    try:
        audited_split_path = Path(str(source_split.get("path"))).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("BN recalibration normal-audit source split is unavailable") from exc
    if (
        split_record_path != audited_split_path
        or split_record.get("sha256") != split_sha256
    ):
        raise ValueError("BN recalibration normal-audit split record drift")

    rows = payload.get("scenes")
    if not isinstance(rows, list):
        raise ValueError("BN recalibration normal audit lacks scene rows")
    selected = list(expected["selected_scene_ids"])
    observed_ids = [
        str(row.get("scene_id")) if isinstance(row, Mapping) else ""
        for row in rows
    ]
    if observed_ids != selected:
        raise ValueError(
            "Normal audit scenes are not the exact ordered BN recalibration selection"
        )
    if (
        payload.get("scene_count") != len(selected)
        or payload.get("passed_scene_count") != len(selected)
        or payload.get("failed_scene_ids") != []
    ):
        raise ValueError("BN recalibration normal-audit scene summary is inconsistent")

    manifest_hashes: dict[str, str] = {}
    for scene_id, row in zip(selected, rows):
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise ValueError(f"Normal audit scene did not pass: {scene_id}")
        manifest_path = (resolved_root / scene_id / "source_manifest.json").resolve(
            strict=True
        )
        expected_hash = sha256_file(manifest_path)
        if row.get("manifest_sha256") != expected_hash:
            raise ValueError(
                f"Normal audit source-manifest hash drift for scene: {scene_id}"
            )
        try:
            row_manifest_path = Path(str(row.get("manifest"))).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Normal audit manifest path is unavailable for scene: {scene_id}"
            ) from exc
        audited_manifest_path = (
            audited_root / scene_id / "source_manifest.json"
        ).resolve(strict=True)
        if row_manifest_path != audited_manifest_path:
            raise ValueError(f"Normal audit manifest path drift for scene: {scene_id}")
        manifest_hashes[scene_id] = expected_hash

    return {
        "schema_version": NORMAL_AUDIT_SCHEMA_VERSION,
        "selection": expected,
        "source_root": str(resolved_root),
        "audit_source_root": str(audited_root),
        "source_root_relocated": audited_root != resolved_root,
        "source_manifest_sha256": manifest_hashes,
        "passed": True,
    }


def load_and_validate_bn_recalibration_normal_audit(
    audit_path: Path,
    *,
    source_root: Path,
    train_split: Path,
    scene_count: int,
    seed: int,
    expected_train_split_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = audit_path.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("RGBN6 normal audit must be a JSON mapping")
    payload = dict(value)
    evidence = validate_bn_recalibration_normal_audit(
        payload,
        source_root=source_root,
        train_split=train_split,
        scene_count=scene_count,
        seed=seed,
        expected_train_split_sha256=expected_train_split_sha256,
    )
    evidence.update({"path": str(path), "sha256": sha256_file(path)})
    return payload, evidence
