#!/usr/bin/env python3
"""Create the single random initialization shared by all upstream arms."""

from __future__ import annotations

from delimit3d.identity import project_name, project_slug

import argparse
import json
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    sha256_file,
)
from delimit3d.training.adaptation import (
    INPUT_FEATURE_CHOICES,
    LitePTContrastiveModel,
    MULTISCALE_AUXILIARY_WEIGHT,
    MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
    MULTISCALE_LEVEL_LOSS_WEIGHTS,
    MULTISCALE_LEVEL_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-features",
        choices=INPUT_FEATURE_CHOICES,
        default="rgbn6",
    )
    parser.add_argument(
        "--multiscale-supervision",
        action="store_true",
        help="Create the multi-scale deep-supervision initialization contract.",
    )
    parser.add_argument(
        "--multiscale-loss-weight",
        type=float,
        default=MULTISCALE_AUXILIARY_WEIGHT,
    )
    parser.add_argument(
        "--multiscale-warmup-epochs",
        type=int,
        default=MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite initialization: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = LitePTContrastiveModel(
        litept_root=args.litept_root,
        input_features=args.input_features,
        multiscale_supervision=args.multiscale_supervision,
        multiscale_loss_weight=args.multiscale_loss_weight,
        multiscale_warmup_epochs=args.multiscale_warmup_epochs,
    ).cpu()
    state = model.state_dict()
    payload = {
        "schema_version": "litept_gt3_unsam3_initial/v1",
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "seed": int(args.seed),
        "architecture": {
            "backbone": "LitePT-S*",
            "input_features": args.input_features,
            "in_channels": int(model.input_channels),
            "grid_size": 0.02,
            "voxel_reduce": "representative",
            "representative_sampling": model.backbone.representative_sampling,
            "native_dim": 72,
            "projection_head": "Linear(72,128)-GELU-Linear(128,128)",
            "temperature": 0.07,
            "learnable_temperature": True,
            "pretraining_recipe": model.recipe_name,
            "multiscale_supervision": bool(args.multiscale_supervision),
            "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
            "multiscale_level_loss_weights": dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            ),
            "multiscale_loss_weight": float(args.multiscale_loss_weight),
            "multiscale_warmup_epochs": int(args.multiscale_warmup_epochs),
            "multiscale_token_policy": "strict_pure_token_resampling_v1",
            "multiscale_temperature_policy": (
                "separate_learnable_temperature_per_auxiliary_level"
            ),
            "multiscale_feature_hard_policy": (
                "separate_candidate_selection_per_positive_endpoint"
            ),
        },
        "model_state_dict": state,
        "public_checkpoint_used": False,
    }
    with NamedTemporaryFile(
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temp_path.replace(args.output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    verifier = LitePTContrastiveModel(
        litept_root=args.litept_root,
        input_features=args.input_features,
        multiscale_supervision=args.multiscale_supervision,
        multiscale_loss_weight=args.multiscale_loss_weight,
        multiscale_warmup_epochs=args.multiscale_warmup_epochs,
    ).cpu()
    verifier.load_state_dict(reloaded["model_state_dict"], strict=True)
    verified_state = verifier.state_dict()
    if set(verified_state) != set(state):
        raise RuntimeError("Initialization strict-load key drift")
    if any(verified_state[key].shape != state[key].shape for key in state):
        raise RuntimeError("Initialization strict-load shape drift")
    report = {
        "schema_version": "litept_gt3_unsam3_initial_report/v1",
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "initial_checkpoint": str(args.output.resolve(strict=True)),
        "sha256": sha256_file(args.output),
        "seed": int(args.seed),
        "input_features": args.input_features,
        "in_channels": int(model.input_channels),
        "strict_reload_passed": True,
        "state_tensors": len(state),
        "state_parameters": int(
            sum(tensor.numel() for tensor in state.values())
        ),
        "public_checkpoint_used": False,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
