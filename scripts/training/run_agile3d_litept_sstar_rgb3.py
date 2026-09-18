#!/usr/bin/env python3
"""Train the minimal released-AGILE3D backbone substitution.

The reference AGILE3D protocol is kept intact: official ScanNet40 data,
RGB-only 5 cm quantization, click simulation, BCE/Dice losses, optimizer and
schedule, native validation CSV, and ``EvaluatorMO``.  The only model change
is replacing the Minkowski Res16UNet34C backbone with a trainable LitePT-S*
backbone whose finest ``dec0`` tokens are projected from 72 to 128 channels.

This is deliberately separate from ``run_agile3d_multio.py``, which is a
frozen-backbone ScanNet++ study.  Use ``--mode preflight`` before allocation,
then ``--mode smoke`` for one real GPU batch, and ``--mode train`` only after
the smoke report is clean.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from delimit3d.evaluation.agile3d_decoder import (  # noqa: E402
    Agile3DClickDecoder,
    compute_agile3d_losses,
)
from delimit3d.evaluation.agile3d_minkowski_compat import (  # noqa: E402
    install_minkowski_engine_compat,
)
from delimit3d.models.litept_wrapper import (  # noqa: E402
    LitePTBackbone,
    LitePTBackboneOutput,
)


SCHEMA = "delimit3d_agile3d_litept_sstar_rgb3_scratch/v1"
SOURCE_FILES = (
    Path(__file__).resolve(),
    REPO_ROOT / "src/delimit3d/models/litept_wrapper.py",
    REPO_ROOT / "src/delimit3d/evaluation/agile3d_decoder.py",
    REPO_ROOT / "src/delimit3d/evaluation/agile3d_minkowski_compat.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "train", "evaluate"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: YAML root must be a mapping")
    result = dict(_expand(dict(value)))
    result["_config_path"] = str(path)
    result["_config_sha256"] = sha256_file(path)
    # Relocate identical inputs without changing the experiment config hash.
    if local_root := os.environ.get("AGILE3D_STAGED_ROOT"):
        root = Path(local_root)
        for key, relative in {
            "agile3d_root": "AGILE3D", "litept_root": "LitePT",
            "scan_folder": "ScanNet/scans", "train_list": "ScanNet/train_list.json",
            "val_list": "ScanNet/val_list.json",
        }.items():
            result["paths"][key] = str(root / relative)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def git_state(path: Path) -> dict[str, Any]:
    """Record tracked revision plus any working-tree drift used by the run."""
    state: dict[str, Any] = {"head": git_head(path)}
    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain=v1"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "-C", str(path), "diff", "--binary"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        state["status"] = None
        state["tracked_diff_sha256"] = None
        return state
    state["status"] = status.splitlines()
    state["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    return state


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def path_value(config: Mapping[str, Any], key: str) -> Path:
    value = config["paths"][key]
    return Path(str(value)).expanduser().resolve()


def validate_data_contract(paths: Mapping[str, Path]) -> dict[str, int]:
    """Check the official flat labeled-Ply/list bundle without loading scenes."""
    expected_counts = {"train_list": 1200, "val_list": 312}
    scene_names: set[str] = set()
    counts: dict[str, int] = {}
    for key, expected_count in expected_counts.items():
        with paths[key].open() as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError(f"{paths[key]} must contain a JSON object")
        count = len(payload)
        counts[key] = count
        if count != expected_count:
            raise ValueError(
                f"{paths[key]} has {count} entries; expected {expected_count} "
                "from the released ScanNet40 protocol"
            )
        for sample_name in payload:
            scene_name, separator, _object_count = str(sample_name).partition("_obj_")
            if not separator:
                raise ValueError(f"malformed AGILE3D sample key: {sample_name!r}")
            scene_names.add(scene_name)

    missing = [
        paths["scan_folder"] / f"{scene_name}.ply"
        for scene_name in sorted(scene_names)
        if not (paths["scan_folder"] / f"{scene_name}.ply").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"official AGILE3D data is missing {len(missing)} scene PLY files; "
            f"first missing file is {missing[0]}"
        )

    invalid_headers: list[Path] = []
    for scene_name in sorted(scene_names):
        ply_path = paths["scan_folder"] / f"{scene_name}.ply"
        header = bytearray()
        with ply_path.open("rb") as handle:
            for _ in range(128):
                line = handle.readline()
                if not line:
                    break
                header.extend(line)
                if line.strip() == b"end_header":
                    break
        header_text = bytes(header).decode("ascii", errors="ignore")
        if not all(token in header_text for token in ("property uint8 R", "property uint8 G", "property uint8 B", "label")):
            invalid_headers.append(ply_path)
    if invalid_headers:
        raise ValueError(
            "AGILE3D requires flat PLYs with RGB and label fields; invalid "
            f"header: {invalid_headers[0]}"
        )
    counts["scenes"] = len(scene_names)
    return counts


def validate_config(config: Mapping[str, Any], *, require_data: bool) -> dict[str, Path]:
    required_paths = (
        "agile3d_root",
        "litept_root",
        "scan_folder",
        "train_list",
        "val_list",
        "output_root",
    )
    missing = [key for key in required_paths if key not in config.get("paths", {})]
    if missing:
        raise ValueError(f"config is missing paths: {missing}")

    protocol = config.get("protocol", {})
    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    if str(config.get("experiment_id")) != "agile3d_litept_sstar_rgb3_scratch_v1":
        raise ValueError("unexpected experiment_id")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("the baseline seed must be 42")
    if float(protocol.get("voxel_size_m", -1.0)) != 0.05:
        raise ValueError("the baseline voxel size must be exactly 0.05 m")
    if protocol.get("input_features") != "rgb3":
        raise ValueError("the baseline input contract must be rgb3")
    if protocol.get("multi_scale") is not False:
        raise ValueError("the baseline must expose only LitePT dec0")
    if model.get("litept_variant") != "litept_s_star":
        raise ValueError("the baseline LitePT variant must be litept_s_star")
    if int(model.get("in_channels", -1)) != 3:
        raise ValueError("the baseline LitePT input width must be 3")
    if int(model.get("feature_dim", -1)) != 72:
        raise ValueError("the LitePT dec0 width must be 72")
    if int(model.get("hidden_dim", -1)) != 128:
        raise ValueError("the AGILE3D hidden width must be 128")
    expected = {
        "batch_size": 5,
        "val_batch_size": 1,
        "epochs": 1100,
        "val_epochs": 50,
        "max_num_clicks": 20,
    }
    for key, expected_value in expected.items():
        if int(protocol.get(key, -1)) != expected_value:
            raise ValueError(f"protocol.{key} must be {expected_value}")
    if [int(value) for value in protocol.get("lr_drop", [])] != [1000]:
        raise ValueError("protocol.lr_drop must be [1000]")
    if float(optimizer.get("learning_rate", -1.0)) != 1e-4:
        raise ValueError("optimizer.learning_rate must be 1e-4")
    if float(optimizer.get("weight_decay", -1.0)) != 1e-4:
        raise ValueError("optimizer.weight_decay must be 1e-4")
    if float(optimizer.get("clip_norm", -1.0)) != 0.1:
        raise ValueError("optimizer.clip_norm must be 0.1")

    paths = {key: path_value(config, key) for key in required_paths}
    for key in ("agile3d_root", "litept_root", "scan_folder", "train_list", "val_list"):
        if not paths[key].exists():
            if require_data or key in {"agile3d_root", "litept_root"}:
                raise FileNotFoundError(f"missing {key}: {paths[key]}")
    for relative in (
        "datasets/InterMultiObj3DSegDataset.py",
        "utils/seg.py",
        "evaluation/evaluator_MO.py",
    ):
        candidate = paths["agile3d_root"] / relative
        if not candidate.is_file():
            raise FileNotFoundError(f"AGILE3D source is missing {candidate}")
    validate_data_contract(paths)
    return paths


def import_official_agile3d(agile3d_root: Path) -> dict[str, Any]:
    """Import the pinned official data, click, and evaluator implementations."""
    minkowski_backend = install_minkowski_engine_compat()
    root = str(agile3d_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from datasets.InterMultiObj3DSegDataset import build as build_dataset  # type: ignore
    from evaluation.evaluator_MO import EvaluatorMO  # type: ignore
    from utils.seg import (  # type: ignore
        extend_clicks,
        get_simulated_clicks,
        mean_iou_scene,
    )

    return {
        "build_dataset": build_dataset,
        "EvaluatorMO": EvaluatorMO,
        "extend_clicks": extend_clicks,
        "get_simulated_clicks": get_simulated_clicks,
        "mean_iou_scene": mean_iou_scene,
        "minkowski_backend": minkowski_backend,
    }


class LitePTAgile3D(nn.Module):
    """Trainable LitePT-S* plus the existing single-level AGILE3D decoder."""

    def __init__(self, config: Mapping[str, Any], *, litept_root: Path) -> None:
        super().__init__()
        model_config = config["model"]
        self.backbone = LitePTBackbone(
            litept_root=str(litept_root),
            in_channels=int(model_config["in_channels"]),
            grid_size=float(config["protocol"]["voxel_size_m"]),
            litept_variant=str(model_config["litept_variant"]),
            multi_scale=False,
            cache_training_voxelization=False,
            feature_pyramid_mode=None,
            voxel_reduce="representative",
            representative_sampling="first",
        )
        decoder_kwargs = {
            "feature_dim": int(model_config["feature_dim"]),
            "hidden_dim": int(model_config["hidden_dim"]),
            "num_heads": int(model_config["num_heads"]),
            "dim_feedforward": int(model_config["dim_feedforward"]),
            "num_decoders": int(model_config["num_decoders"]),
            "num_bg_queries": int(model_config["num_bg_queries"]),
            "dropout": float(model_config["dropout"]),
            "pre_norm": bool(model_config["pre_norm"]),
            "max_click_events": int(model_config["max_click_events"]),
            "normalize_pos_enc": bool(model_config["normalize_pos_enc"]),
            "gauss_scale": float(model_config["gauss_scale"]),
            "aux": bool(model_config["aux"]),
        }
        self.decoder = Agile3DClickDecoder(**decoder_kwargs)

    def encode_batch(
        self,
        grid_coord: Tensor,
        raw_coord: Tensor,
        features: Tensor,
        point_offsets: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        output = self.backbone(
            raw_coord,
            features,
            point_offsets=point_offsets,
            grid_coord=grid_coord,
        )
        if not isinstance(output, LitePTBackboneOutput):
            raise TypeError("single-scale LitePT returned an unexpected output type")
        offsets = output.scene_token_offsets.detach().to("cpu").tolist()
        starts = [0] + [int(value) for value in offsets[:-1]]
        ends = [int(value) for value in offsets]
        token_scenes = [output.scene_tokens[start:end] for start, end in zip(starts, ends)]
        xyz_scenes = [output.scene_xyz[start:end] for start, end in zip(starts, ends)]
        if len(token_scenes) != int(point_offsets.numel()):
            raise RuntimeError("LitePT returned the wrong number of scene token groups")
        for tokens, xyz in zip(token_scenes, xyz_scenes):
            if tokens.shape[0] != xyz.shape[0] or tokens.shape[1] != 72:
                raise RuntimeError(
                    f"unexpected LitePT dec0 shape {tuple(tokens.shape)} / {tuple(xyz.shape)}"
                )
        return token_scenes, xyz_scenes

    def decode(
        self,
        scene_tokens: Tensor,
        scene_xyz: Tensor,
        clicks: Mapping[int | str, Sequence[int]],
        click_times: Mapping[int | str, Sequence[int]],
    ) -> dict[str, Any]:
        return self.decoder(
            scene_tokens,
            scene_xyz,
            clicks=clicks,
            click_times=click_times,
        )


def make_dataset(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    split: str,
) -> tuple[Any, Any]:
    args = SimpleNamespace(
        scan_folder=str(paths["scan_folder"]),
        train_list=str(paths["train_list"]),
        val_list=str(paths["val_list"]),
        voxel_size=float(config["protocol"]["voxel_size_m"]),
        use_normals=False,
        normal_root=None,
        external_feature_root=None,
        external_feature_dim=None,
    )
    return official["build_dataset"](split=split, args=args)


def unpack_batch(batch: Any, device: torch.device) -> dict[str, Any]:
    (
        coords_batch,
        raw_coords,
        features,
        labels,
        labels_full,
        inverse_maps,
        click_idx,
        scene_names,
        num_objects,
    ) = batch
    coords_batch = coords_batch.to(device=device, non_blocking=True)
    raw_coords = raw_coords.to(device=device, dtype=torch.float32, non_blocking=True)
    features = features.to(device=device, dtype=torch.float32, non_blocking=True)
    labels_device = [value.to(device=device, dtype=torch.long, non_blocking=True) for value in labels]
    lengths = torch.tensor(
        [int(value.shape[0]) for value in labels_device], device=device, dtype=torch.long
    )
    point_offsets = torch.cumsum(lengths, dim=0)
    if int(point_offsets[-1].item()) != int(raw_coords.shape[0]):
        raise RuntimeError("official AGILE3D collate lengths do not match raw coordinates")
    return {
        "grid_coord": coords_batch[:, 1:].to(dtype=torch.int32),
        "raw_coords": raw_coords,
        "features": features,
        "point_offsets": point_offsets,
        "labels": labels_device,
        "labels_full": [value.to(device=device, dtype=torch.long) for value in labels_full],
        "inverse_maps": inverse_maps,
        "click_idx": copy.deepcopy(list(click_idx)),
        "scene_names": list(scene_names),
        "num_objects": [int(value) for value in num_objects],
    }


def split_by_offsets(value: Tensor, offsets: Tensor) -> list[Tensor]:
    ends = [int(item) for item in offsets.detach().to("cpu").tolist()]
    starts = [0] + ends[:-1]
    return [value[start:end] for start, end in zip(starts, ends)]


def initialize_training_episode(
    labels: Sequence[Tensor],
    click_idx: list[dict[Any, Any]],
) -> tuple[list[Tensor], list[dict[Any, Any]], list[dict[Any, Any]]]:
    """Copy the released AGILE3D random-object training setup."""
    labels_new: list[Tensor] = []
    clicks = copy.deepcopy(click_idx)
    for index, sample_labels in enumerate(labels):
        valid_object_ids = torch.unique(sample_labels)
        valid_object_ids = valid_object_ids[valid_object_ids != -1]
        max_num_objects = int(valid_object_ids.numel())
        if max_num_objects <= 0:
            raise RuntimeError("a training scene has no valid object labels")
        object_count = int(np.random.randint(1, min(10, max_num_objects) + 1))
        selected = valid_object_ids[torch.randperm(max_num_objects, device=sample_labels.device)[:object_count]]
        remapped = torch.zeros(sample_labels.shape, device=sample_labels.device, dtype=torch.long)
        for new_id, object_id in enumerate(selected.tolist(), start=1):
            remapped[sample_labels == int(object_id)] = int(new_id)
            clicks[index][str(new_id)] = []
        clicks[index]["0"] = []
        labels_new.append(remapped)
    return labels_new, clicks, copy.deepcopy(clicks)


def run_training_click_prefix(
    model: LitePTAgile3D,
    scene_tokens: Sequence[Tensor],
    scene_xyz: Sequence[Tensor],
    labels: Sequence[Tensor],
    clicks: list[dict[Any, Any]],
    click_times: list[dict[Any, Any]],
    official: Mapping[str, Any],
) -> tuple[list[dict[Any, Any]], list[dict[Any, Any]]]:
    model.eval()
    rounds = random.randint(0, 19)
    current_round = 0
    with torch.no_grad():
        while current_round <= rounds:
            if current_round == 0:
                predictions = [torch.zeros_like(target) for target in labels]
            else:
                predictions = [
                    model.decode(tokens, xyz, clicks[index], click_times[index])["pred_masks"].argmax(dim=1)
                    for index, (tokens, xyz) in enumerate(zip(scene_tokens, scene_xyz))
                ]
            for index, (prediction, target, xyz) in enumerate(zip(predictions, labels, scene_xyz)):
                prediction = prediction.clone()
                for object_id, click_ids in clicks[index].items():
                    if click_ids:
                        click_tensor = torch.as_tensor(click_ids, device=prediction.device, dtype=torch.long)
                        prediction[click_tensor] = int(object_id)
                new_clicks, _count, _positions, new_times = official["get_simulated_clicks"](
                    prediction,
                    target,
                    xyz,
                    current_round,
                    training=True,
                )
                if new_clicks is not None:
                    clicks[index], click_times[index] = official["extend_clicks"](
                        clicks[index], click_times[index], new_clicks, new_times
                    )
            current_round += 1
    model.train()
    return clicks, click_times


def train_batch(
    model: LitePTAgile3D,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    config: Mapping[str, Any],
    official: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    unpacked = unpack_batch(batch, device)
    scene_tokens, scene_xyz = model.encode_batch(
        unpacked["grid_coord"],
        unpacked["raw_coords"],
        unpacked["features"],
        unpacked["point_offsets"],
    )
    labels, clicks, click_times = initialize_training_episode(
        unpacked["labels"], unpacked["click_idx"]
    )
    clicks, click_times = run_training_click_prefix(
        model, scene_tokens, scene_xyz, labels, clicks, click_times, official
    )

    losses: list[Tensor] = []
    loss_bce: list[Tensor] = []
    loss_dice: list[Tensor] = []
    for tokens, xyz, target, scene_clicks, scene_times in zip(
        scene_tokens, scene_xyz, labels, clicks, click_times
    ):
        output = model.decode(tokens, xyz, scene_clicks, scene_times)
        loss, details = compute_agile3d_losses(
            output,
            target,
            scene_xyz=xyz,
            clicks=scene_clicks,
            bce_weight=float(config["loss"]["bce_weight"]),
            dice_weight=float(config["loss"]["dice_weight"]),
            alpha=float(config["loss"]["alpha"]),
            beta=float(config["loss"]["beta"]),
            radius=float(config["loss"]["radius"]),
        )
        losses.append(loss)
        loss_bce.append(details["loss_bce"])
        loss_dice.append(details["loss_dice"])
    batch_loss = torch.stack(losses).mean()
    if not torch.isfinite(batch_loss):
        raise FloatingPointError(f"non-finite training loss: {batch_loss}")
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["optimizer"]["clip_norm"])
    )
    optimizer.step()
    return {
        "loss": float(batch_loss.detach().cpu()),
        "loss_bce": float(torch.stack(loss_bce).mean().detach().cpu()),
        "loss_dice": float(torch.stack(loss_dice).mean().detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
        "scenes": len(scene_tokens),
        "tokens": int(sum(int(value.shape[0]) for value in scene_tokens)),
    }


@torch.no_grad()
def evaluate(
    model: LitePTAgile3D,
    loader: Iterable[Any],
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    output_root = paths["output_root"]
    validation_root = output_root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    result_file = validation_root / f"val_results_epoch_{int(epoch):04d}.csv"
    instance_counter = 0
    with result_file.open("w") as handle:
        for batch in loader:
            unpacked = unpack_batch(batch, device)
            scene_tokens, scene_xyz = model.encode_batch(
                unpacked["grid_coord"],
                unpacked["raw_coords"],
                unpacked["features"],
                unpacked["point_offsets"],
            )
            if len(scene_tokens) != 1:
                raise RuntimeError("official AGILE3D validation must use val_batch_size=1")
            tokens, xyz = scene_tokens[0], scene_xyz[0]
            labels = unpacked["labels"][0]
            labels_full = unpacked["labels_full"][0]
            inverse_map = torch.as_tensor(
                unpacked["inverse_maps"][0], device=device, dtype=torch.long
            )
            clicks = copy.deepcopy(unpacked["click_idx"][0])
            for object_id in list(clicks):
                clicks[object_id] = []
            click_times = copy.deepcopy(clicks)
            object_count = int(unpacked["num_objects"][0])
            current_clicks = 0
            max_clicks = object_count * int(config["protocol"]["max_num_clicks"])
            while current_clicks <= max_clicks:
                if current_clicks == 0:
                    prediction = torch.zeros_like(labels)
                else:
                    output = model.decode(tokens, xyz, clicks, click_times)
                    prediction = output["pred_masks"].argmax(dim=1)
                    prediction = prediction.clone()
                    for object_id, click_ids in clicks.items():
                        if click_ids:
                            click_tensor = torch.as_tensor(click_ids, device=device, dtype=torch.long)
                            prediction[click_tensor] = int(object_id)

                prediction_full = prediction[inverse_map]
                iou, _ = official["mean_iou_scene"](prediction_full, labels_full)
                handle.write(
                    f"{instance_counter} "
                    f"{str(unpacked['scene_names'][0]).replace('scene', '')} "
                    f"{object_count} {current_clicks / object_count} {float(iou):.10f}\n"
                )
                new_clicks, _new_count, _positions, new_times = official["get_simulated_clicks"](
                    prediction,
                    labels,
                    xyz,
                    current_clicks,
                    training=False,
                )
                if new_clicks is None:
                    break
                clicks, click_times = official["extend_clicks"](
                    clicks, click_times, new_clicks, new_times
                )
                # Match the released engine: the first correction round adds
                # one click per selected object; later rounds add one click.
                current_clicks += int(object_count if current_clicks == 0 else 1)
            instance_counter += 1
    evaluator = official["EvaluatorMO"](
        str(paths["val_list"]),
        str(result_file),
        [0.5, 0.65, 0.8, 0.85, 0.9],
    )
    metrics = {key: float(value) for key, value in evaluator.eval_results().items()}
    metrics.update({"epoch": int(epoch), "result_file": str(result_file)})
    (validation_root / f"metrics_epoch_{int(epoch):04d}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: LitePTAgile3D,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    epoch: int,
    step: int,
    config: Mapping[str, Any],
) -> None:
    payload = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "epoch": int(epoch),
        "step": int(step),
        "model": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "config_sha256": config["_config_sha256"],
        "encoder_trainable": True,
        "decoder_single_scale": True,
        "rng_state": capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed or interrupted write must not destroy the last complete epoch.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_manifest(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any] | None = None,
) -> None:
    root = paths["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_FILES
        if path.is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "repo_root": str(REPO_ROOT),
        "repo_state": git_state(REPO_ROOT),
        "agile3d_root": str(paths["agile3d_root"]),
        "agile3d_state": git_state(paths["agile3d_root"]),
        "litept_root": str(paths["litept_root"]),
        "litept_state": git_state(paths["litept_root"]),
        "data_paths": {
            "scan_folder": str(paths["scan_folder"]),
            "train_list": str(paths["train_list"]),
            "val_list": str(paths["val_list"]),
        },
        "source_hashes": source_hashes,
        "minkowski_backend": None if official is None else official.get("minkowski_backend"),
        "python": sys.executable,
        "platform": platform.platform(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}, sort_keys=False)
    )


def make_loaders(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset, collate_fn = make_dataset(config, paths, official, "train")
    val_dataset, val_collate_fn = make_dataset(config, paths, official, "val")
    protocol = config["protocol"]
    train_batch_size = 1 if smoke else int(protocol["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=not smoke,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0 if smoke else int(config["runtime"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(protocol["val_batch_size"]),
        shuffle=False,
        drop_last=False,
        collate_fn=val_collate_fn,
        num_workers=0 if smoke else int(config["runtime"]["num_workers"]),
        pin_memory=True,
    )
    return train_loader, val_loader


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("this LitePT-S* run requires a CUDA allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "the baseline is single-GPU only; visible CUDA devices="
            f"{torch.cuda.device_count()}"
        )
    device_name = torch.cuda.get_device_name(0)
    if not any(token in device_name for token in ("RTX 3090", "RTX 4090")):
        raise RuntimeError(f"expected an RTX 3090 or RTX 4090, got {device_name!r}")
    return torch.device("cuda")


def preflight(config: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    official = import_official_agile3d(paths["agile3d_root"])
    model = LitePTAgile3D(config, litept_root=paths["litept_root"])
    train_dataset, _ = make_dataset(config, paths, official, "train")
    val_dataset, _ = make_dataset(config, paths, official, "val")
    report = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "data_contract": validate_data_contract(paths),
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "backbone_trainable": any(value.requires_grad for value in model.backbone.parameters()),
        "decoder_feature_dim": model.decoder.feature_dim,
        "minkowski_backend": official["minkowski_backend"],
        "gpu_available": bool(torch.cuda.is_available()),
        "agile3d_head": git_head(paths["agile3d_root"]),
        "litept_head": git_head(paths["litept_root"]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def smoke(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
) -> dict[str, Any]:
    device = require_cuda()
    set_seed(int(config["seed"]))
    model = LitePTAgile3D(config, litept_root=paths["litept_root"]).to(device)
    train_loader, _ = make_loaders(config, paths, official, smoke=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    batch = next(iter(train_loader))
    before = {key: value.detach().clone() for key, value in model.backbone.state_dict().items()}
    stats = train_batch(model, optimizer, batch, config, official, device)
    changed = any(
        not torch.equal(before[key].cpu(), value.detach().cpu())
        for key, value in model.backbone.state_dict().items()
    )
    if not changed:
        raise RuntimeError("smoke step did not update any LitePT parameter")
    report = {
        "schema": SCHEMA,
        "mode": "smoke",
        "gpu": torch.cuda.get_device_name(device),
        "stats": stats,
        "encoder_updated": changed,
        "decoder_updated": True,
    }
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    (paths["output_root"] / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def train(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    resume: Path | None,
) -> None:
    device = require_cuda()
    set_seed(int(config["seed"]))
    model = LitePTAgile3D(config, litept_root=paths["litept_root"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    protocol = config["protocol"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in protocol["lr_drop"]],
    )
    start_epoch = 0
    step = 0
    if resume is not None:
        payload = torch.load(resume.expanduser().resolve(strict=True), map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise ValueError("checkpoint schema does not match this LitePT baseline")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["lr_scheduler"])
        if "rng_state" not in payload:
            raise ValueError("checkpoint is missing exact-resume RNG state")
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
        step = int(payload["step"])
        print(json.dumps({"event": "resumed", "completed_epochs": start_epoch,
                          "step": step, "checkpoint": str(resume)}), flush=True)

    train_loader, val_loader = make_loaders(config, paths, official)
    metrics_path = paths["output_root"] / "metrics.jsonl"
    for epoch in range(start_epoch, int(protocol["epochs"])):
        model.train()
        epoch_rows: list[dict[str, Any]] = []
        started = time.time()
        print(json.dumps({"event": "epoch_start", "epoch": epoch + 1, "step": step}), flush=True)
        batch_finished = time.monotonic()
        for batch_index, batch in enumerate(train_loader, start=1):
            batch_loaded = time.monotonic()
            row = train_batch(model, optimizer, batch, config, official, device)
            faulthandler.dump_traceback_later(600, repeat=True)
            finished = time.monotonic()
            step += 1
            row.update({"epoch": epoch + 1, "step": step})
            epoch_rows.append(row)
            if batch_index == 1 or batch_index % 10 == 0:
                print(json.dumps({"event": "batch", **row, "batch": batch_index,
                                  "batches": len(train_loader),
                                  "load_seconds": batch_loaded - batch_finished,
                                  "train_seconds": finished - batch_loaded,
                                  "epoch_seconds": time.time() - started}, sort_keys=True), flush=True)
            batch_finished = finished
        scheduler.step()
        train_row = {
            "split": "train",
            "epoch": epoch + 1,
            "step": step,
            "loss": float(np.mean([row["loss"] for row in epoch_rows])),
            "loss_bce": float(np.mean([row["loss_bce"] for row in epoch_rows])),
            "loss_dice": float(np.mean([row["loss_dice"] for row in epoch_rows])),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(train_row, sort_keys=True) + "\n")
        print(json.dumps(train_row, sort_keys=True), flush=True)

        save_checkpoint(
            paths["output_root"] / "checkpoints/checkpoint.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            config,
        )
        if (epoch + 1) in [int(value) for value in protocol["lr_drop"]] or (epoch + 1) % 20 == 0:
            save_checkpoint(
                paths["output_root"] / f"checkpoints/checkpoint{epoch + 1:04d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                step,
                config,
            )
        if (epoch + 1) % int(protocol["val_epochs"]) == 0:
            validation = evaluate(model, val_loader, config, paths, official, device, epoch + 1)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"split": "val", **validation}, sort_keys=True) + "\n")
            print(json.dumps({"split": "val", **validation}, sort_keys=True), flush=True)


def main() -> None:
    faulthandler.enable()
    # Repeated stack dumps make a future filesystem/CUDA wait diagnosable.
    faulthandler.dump_traceback_later(600, repeat=True)
    args = parse_args()
    config = load_config(args.config)
    paths = validate_config(config, require_data=args.mode != "preflight")
    if args.mode == "preflight":
        preflight(config, paths)
        return

    official = import_official_agile3d(paths["agile3d_root"])
    if args.mode == "smoke":
        write_manifest(config, paths, official)
        smoke(config, paths, official)
        return
    if args.mode == "train":
        write_manifest(config, paths, official)
        train(config, paths, official, args.resume)
        return
    if args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --mode evaluate")
        write_manifest(config, paths, official)
        device = require_cuda()
        set_seed(int(config["seed"]))
        model = LitePTAgile3D(config, litept_root=paths["litept_root"]).to(device)
        payload = torch.load(args.checkpoint.expanduser().resolve(strict=True), map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise ValueError("checkpoint schema does not match this LitePT baseline")
        model.load_state_dict(payload["model"], strict=True)
        _, val_loader = make_loaders(config, paths, official)
        metrics = evaluate(model, val_loader, config, paths, official, device, int(payload.get("epoch", -1)) + 1)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
