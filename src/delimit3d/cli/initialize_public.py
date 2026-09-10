#!/usr/bin/env python3
"""Create a public-LitePT initialization with a scratch-identical small head."""

from __future__ import annotations

from delimit3d.identity import project_name, project_slug

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    sha256_file,
)
from delimit3d.training.adaptation import (
    LitePTContrastiveModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--scratch-initial", type=Path, required=True)
    parser.add_argument("--public-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-public-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(payload, dict) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    raise TypeError("Could not find a tensor state dictionary in public checkpoint")


def _public_backbone_state(
    payload: Any,
) -> dict[str, torch.Tensor]:
    source = _checkpoint_state(payload)
    mapped: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        stripped = key.removeprefix("module.")
        if stripped.startswith("backbone."):
            mapped[f"model.{stripped.removeprefix('backbone.')}"] = value
    if not mapped:
        raise ValueError("Public checkpoint contains no module.backbone.* weights")
    return mapped


def _atomic_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temp_path.replace(output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite initialization: {args.output}")
    scratch_path = args.scratch_initial.resolve(strict=True)
    public_path = args.public_checkpoint.resolve(strict=True)
    public_sha256 = sha256_file(public_path)
    if public_sha256 != args.expected_public_sha256:
        raise RuntimeError(
            f"Public checkpoint hash drift: {public_sha256} != "
            f"{args.expected_public_sha256}"
        )

    scratch = torch.load(scratch_path, map_location="cpu", weights_only=False)
    if scratch.get("schema_version") != "litept_gt3_unsam3_initial/v1":
        raise ValueError("Scratch initialization schema drift")
    if scratch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Scratch initialization experiment drift")
    if scratch.get("public_checkpoint_used") is not False:
        raise ValueError("Source initialization is not the frozen scratch state")

    model = LitePTContrastiveModel(litept_root=args.litept_root).cpu()
    model.load_state_dict(scratch["model_state_dict"], strict=True)
    scratch_state = {
        key: tensor.detach().clone()
        for key, tensor in model.state_dict().items()
    }
    public_payload = torch.load(public_path, map_location="cpu", weights_only=False)
    mapped = _public_backbone_state(public_payload)
    target_backbone = model.backbone.state_dict()
    if set(mapped) != set(target_backbone):
        missing = sorted(set(target_backbone) - set(mapped))
        unexpected = sorted(set(mapped) - set(target_backbone))
        raise RuntimeError(
            f"Public backbone key drift: missing={missing}, unexpected={unexpected}"
        )
    shape_mismatches = {
        key: [list(mapped[key].shape), list(target_backbone[key].shape)]
        for key in mapped
        if mapped[key].shape != target_backbone[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"Public backbone shape drift: {shape_mismatches}")
    model.backbone.load_state_dict(mapped, strict=True)
    combined_state = model.state_dict()
    changed_outside_backbone = [
        key
        for key in combined_state
        if not key.startswith("backbone.")
        and not torch.equal(combined_state[key], scratch_state[key])
    ]
    if changed_outside_backbone:
        raise RuntimeError(
            "Public initialization changed scratch-identical small-head state: "
            f"{changed_outside_backbone}"
        )
    public_reload_mismatches = [
        key
        for key, tensor in model.backbone.state_dict().items()
        if not torch.equal(tensor, mapped[key])
    ]
    if public_reload_mismatches:
        raise RuntimeError(
            f"Public backbone tensor drift after load: {public_reload_mismatches}"
        )

    public_record = {
        "path": str(public_path),
        "sha256": public_sha256,
        "loaded_backbone_keys": len(mapped),
        "strict_key_match": True,
        "strict_shape_match": True,
    }
    scratch_record = {
        "path": str(scratch_path),
        "sha256": sha256_file(scratch_path),
        "small_head_and_temperature_preserved_exactly": True,
    }
    payload = {
        "schema_version": "litept_gt3_unsam3_initial/v1",
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "seed": int(scratch["seed"]),
        "architecture": scratch["architecture"],
        "model_state_dict": combined_state,
        "initialization_kind": "public_litept_backbone",
        "public_checkpoint_used": True,
        "public_checkpoint": public_record,
        "source_random_initialization": scratch_record,
    }
    _atomic_save(payload, args.output)

    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    verifier = LitePTContrastiveModel(litept_root=args.litept_root).cpu()
    verifier.load_state_dict(reloaded["model_state_dict"], strict=True)
    verifier_state = verifier.state_dict()
    mismatches = [
        key
        for key in combined_state
        if not torch.equal(verifier_state[key], combined_state[key])
    ]
    if mismatches:
        raise RuntimeError(f"Serialized initialization tensor drift: {mismatches}")
    report = {
        "schema_version": "litept_gt3_unsam3_public_initial_report/v1",
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "initial_checkpoint": str(args.output.resolve(strict=True)),
        "initial_checkpoint_sha256": sha256_file(args.output),
        "seed": int(scratch["seed"]),
        "strict_reload_passed": True,
        "state_tensors": len(combined_state),
        "state_parameters": int(
            sum(tensor.numel() for tensor in combined_state.values())
        ),
        "public_checkpoint_used": True,
        "public_checkpoint": public_record,
        "source_random_initialization": scratch_record,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
