#!/usr/bin/env python3
"""Strictly export only the selected LitePT backbone for PointGroup."""

from __future__ import annotations

from delimit3d.identity import project_name, project_slug

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    sha256_file,
)
from delimit3d.models.litept_wrapper import LitePTBackbone

POINTGROUP_CONFIG_RELATIVE = Path(
    "configs/scannet/insseg-litept-small-v1m2.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-failed-representation-gate",
        action="store_true",
        help=(
            "Permit export of the matched g08 control after the main arm has "
            "passed. Execution checks still have to pass."
        ),
    )
    return parser.parse_args()


def _convert_wrapper_backbone_state(
    source_state: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(source_state):
        if not key.startswith("model."):
            raise RuntimeError(f"Unexpected wrapper backbone key: {key}")
        converted[f"backbone.{key.removeprefix('model.')}"] = (
            source_state[key].detach().cpu()
        )
    return converted


def _validate_actual_pointgroup_target(
    *,
    litept_root: Path,
    converted: OrderedDict[str, torch.Tensor],
) -> dict[str, object]:
    """Prove the export exactly targets the official PG-v1m2 model."""
    config_path = litept_root.resolve(strict=True) / POINTGROUP_CONFIG_RELATIVE
    if not config_path.is_file():
        raise FileNotFoundError(f"PointGroup target config missing: {config_path}")

    # LitePTBackbone has already placed this exact LitePT checkout first on
    # sys.path. Importing here deliberately builds the real downstream model,
    # rather than another wrapper approximation of its backbone. The compiled
    # clustering extension is irrelevant to model construction/state and can
    # block for minutes on the login node, so exercise PG-v1m2's supported
    # import fallback during this CPU-only state audit.
    sys.modules.setdefault("pointgroup_ops", None)
    from models import build_model
    from utils.config import Config

    config = Config.fromfile(str(config_path))
    pointgroup = build_model(config.model).cpu()
    target_state = pointgroup.state_dict()
    target_backbone = OrderedDict(
        (key, value)
        for key, value in target_state.items()
        if key.startswith("backbone.")
    )
    if set(converted) != set(target_backbone):
        missing = sorted(set(target_backbone) - set(converted))
        unexpected = sorted(set(converted) - set(target_backbone))
        raise RuntimeError(
            "Actual PointGroup backbone key drift: "
            f"missing={missing}, unexpected={unexpected}"
        )
    shape_mismatches = {
        key: [list(converted[key].shape), list(target_backbone[key].shape)]
        for key in converted
        if converted[key].shape != target_backbone[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(
            f"Actual PointGroup backbone shape drift: {shape_mismatches}"
        )

    load_result = pointgroup.load_state_dict(converted, strict=False)
    expected_fresh_keys = sorted(set(target_state) - set(target_backbone))
    # BatchNorm's loader deliberately does not report a missing
    # ``num_batches_tracked`` buffer, even though it is present in state_dict
    # and remains freshly initialized. Compare against that documented loader
    # boundary while retaining the counter in the complete fresh-head set.
    expected_reported_missing_keys = [
        key
        for key in expected_fresh_keys
        if not key.endswith(".num_batches_tracked")
    ]
    if load_result.unexpected_keys:
        raise RuntimeError(
            "Actual PointGroup rejected exported keys: "
            f"{sorted(load_result.unexpected_keys)}"
        )
    if sorted(load_result.missing_keys) != expected_reported_missing_keys:
        raise RuntimeError(
            "Actual PointGroup fresh-state boundary drift: "
            f"expected_reported={expected_reported_missing_keys}, "
            f"observed={sorted(load_result.missing_keys)}"
        )
    invalid_fresh_keys = [
        key
        for key in expected_fresh_keys
        if not key.startswith(("bias_head.", "seg_head."))
    ]
    if invalid_fresh_keys:
        raise RuntimeError(
            "Export omits state outside the PointGroup prediction heads: "
            f"{invalid_fresh_keys}"
        )
    if not any(key.startswith("bias_head.") for key in expected_fresh_keys):
        raise RuntimeError("Actual PointGroup bias head has no fresh state")
    if not any(key.startswith("seg_head.") for key in expected_fresh_keys):
        raise RuntimeError("Actual PointGroup segmentation head has no fresh state")

    return {
        "pointgroup_target_config": str(config_path.resolve(strict=True)),
        "pointgroup_target_config_sha256": sha256_file(config_path),
        "pointgroup_target_type": str(config.model.type),
        "actual_pointgroup_backbone_match": True,
        "actual_pointgroup_backbone_keys": len(target_backbone),
        "pointgroup_load_unexpected_keys": [],
        "pointgroup_load_missing_keys": expected_reported_missing_keys,
        "fresh_pointgroup_head_keys": expected_fresh_keys,
        "fresh_pointgroup_head_key_count": len(expected_fresh_keys),
    }


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Checkpoint selection experiment drift")
    gate_passed = bool(selection.get("upstream_gate_passed"))
    if not gate_passed and not args.allow_failed_representation_gate:
        raise RuntimeError("Refusing transfer export because the upstream gate failed")
    checkpoint_path = Path(selection["selected_checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source_state = checkpoint["backbone_state_dict"]

    target = LitePTBackbone(
        litept_root=str(args.litept_root),
        in_channels=6,
        grid_size=0.02,
        litept_variant="litept_s_star",
        multi_scale=False,
    ).cpu()
    load_result = target.load_state_dict(source_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Strict backbone load failed: {load_result}")
    target_state = target.state_dict()
    if set(target_state) != set(source_state):
        raise RuntimeError("Fresh target backbone key set differs from source")
    shape_mismatches = {
        key: [list(source_state[key].shape), list(target_state[key].shape)]
        for key in source_state
        if source_state[key].shape != target_state[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"Fresh target backbone shape drift: {shape_mismatches}")

    converted = _convert_wrapper_backbone_state(source_state)
    pointgroup_validation = _validate_actual_pointgroup_target(
        litept_root=args.litept_root,
        converted=converted,
    )
    payload = {
        "state_dict": converted,
        "epoch": 0,
        "best_metric_value": -1.0,
        "export": {
            "schema_version": "litept_gt3_unsam3_pointgroup_backbone/v1",
            "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
            "three_d_mode": selection["three_d_mode"],
            "selected_epoch": int(selection["selected_epoch"]),
            "selected_update": int(selection["selected_update"]),
            "selection_metric": selection.get("selection_metric"),
            "selection_view": selection.get("selection_view"),
            "sampling_profile": selection.get("sampling_profile"),
            "fixed_eval_pack": selection.get("fixed_eval_pack"),
            "code_manifest": selection.get("code_manifest"),
            "selection_report": str(args.selection.resolve(strict=True)),
            "selection_report_sha256": sha256_file(args.selection),
            "source_dataset_audit": selection["source_dataset_audit"],
            "source_checkpoint": str(checkpoint_path.resolve(strict=True)),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "strict_fresh_backbone_reload": True,
            "source_keys": len(source_state),
            "written_keys": len(converted),
            "public_checkpoint_or_template_used": False,
            "pointgroup_decoder_included": False,
            **pointgroup_validation,
            "representation_gate_passed": gate_passed,
            "failed_representation_gate_override": bool(
                args.allow_failed_representation_gate and not gate_passed
            ),
        },
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite export: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    if set(reloaded["state_dict"]) != set(converted):
        raise RuntimeError("Serialized export key drift")
    serialized_mismatches = [
        key
        for key in converted
        if not torch.equal(reloaded["state_dict"][key], converted[key])
    ]
    if serialized_mismatches:
        raise RuntimeError(
            f"Serialized export tensor drift: {serialized_mismatches}"
        )
    report = {
        **payload["export"],
        "output": str(args.output.resolve(strict=True)),
        "output_sha256": sha256_file(args.output),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
