#!/usr/bin/env python3
"""Evaluate the matched ScanNet40 AGILE3D decoders on Articulate3D parts.

The evaluator consumes the frozen Articulate3D LitePT caches and the two
corresponding 5,000-update ScanNet40 decoder checkpoints:

  public features     -> public decoder
  Delimit3D features  -> Delimit3D decoder

Each annotated part is evaluated as a one-foreground-query AGILE3D episode.
The initial click is the deterministic token centroid and subsequent clicks
use the shared AGILE3D deepest-error policy, capped at 20 clicks per part.
Primary metrics are raw-point IoU after expanding token predictions through
the fixed inverse map, while raw points whose token is ambiguous between parts
are excluded.  Parts with at least five surviving unambiguous tokens are run;
the report gives sensitivity at 5/10/20 tokens, with 10 as the primary
threshold.  Results are written incrementally so a long single-GPU run can be
resumed without discarding completed parts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from delimit3d.evaluation.agile3d_decoder import (  # noqa: E402
    Agile3DClickDecoder,
    state_sha256,
)
from delimit3d.evaluation.agile3d_protocol import (  # noqa: E402
    append_clicks,
    center_click,
    enforce_click_labels,
    paired_scene_bootstrap,
    simulated_corrections,
)


CHECKPOINT_SCHEMA = "delimit3d_agile3d_decoder_checkpoint/v1"
CACHE_SCHEMAS = {
    "delimit3d_articulate3d_feature_cache/v1",
    "delimit3d_agile3d_feature_cache/v2",
}
RUN_SCHEMA = "delimit3d_articulate3d_agile3d_parts_evaluation/v1"
PRIMARY_METRIC = "raw_point_iou"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "evaluate", "aggregate"),
        required=True,
    )
    parser.add_argument(
        "--arm",
        choices=("public", "delimit3d", "all"),
        default="all",
        help="arm to evaluate; default evaluates both arms",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: YAML root must be a mapping")
    result = dict(value)
    result["_config_path"] = str(path)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_array(value: np.ndarray | torch.Tensor) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        str(array.dtype).encode("ascii")
        + str(tuple(array.shape)).encode("ascii")
        + array.tobytes(order="C")
    ).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def update_status(output_root: Path, **values: Any) -> None:
    path = output_root / "status.json"
    current = read_json(path) if path.exists() else {"schema": RUN_SCHEMA}
    current.update(values)
    current["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(path, current)


def protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(config["protocol"])
    required = {
        "click_budget",
        "click_center_method",
        "run_min_surviving_unambiguous_tokens",
        "primary_min_surviving_unambiguous_tokens",
        "sensitivity_min_surviving_unambiguous_tokens",
        "click_thresholds",
        "iou_thresholds",
    }
    missing = sorted(required.difference(protocol))
    if missing:
        raise ValueError(f"missing protocol fields: {missing}")
    if int(protocol["click_budget"]) != 20:
        raise ValueError("the AGILE3D-compatible Articulate3D evaluator is locked to 20 clicks")
    if str(protocol["click_center_method"]) not in {"kdtree", "dense"}:
        raise ValueError("click_center_method must be kdtree or dense")
    sensitivity = sorted(
        int(value) for value in protocol["sensitivity_min_surviving_unambiguous_tokens"]
    )
    if sensitivity != [5, 10, 20]:
        raise ValueError(f"expected sensitivity token thresholds [5, 10, 20], got {sensitivity}")
    if int(protocol["run_min_surviving_unambiguous_tokens"]) != sensitivity[0]:
        raise ValueError("run threshold must be the smallest sensitivity threshold")
    if int(protocol["primary_min_surviving_unambiguous_tokens"]) not in sensitivity:
        raise ValueError("primary threshold must be one of the sensitivity thresholds")
    return protocol


def decoder_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    decoder_config_path = Path(str(config["paths"]["decoder_config"])).expanduser().resolve(strict=True)
    decoder_config = yaml.safe_load(decoder_config_path.read_text())
    if not isinstance(decoder_config, Mapping) or not isinstance(decoder_config.get("decoder"), Mapping):
        raise ValueError(f"{decoder_config_path}: missing decoder mapping")
    # The frozen training config also stores init_seed, which is checkpoint
    # metadata and not an Agile3DClickDecoder constructor argument.
    allowed = {
        "feature_dim",
        "hidden_dim",
        "num_heads",
        "dim_feedforward",
        "num_decoders",
        "num_bg_queries",
        "dropout",
        "pre_norm",
        "max_click_events",
        "normalize_pos_enc",
        "gauss_scale",
        "aux",
    }
    kwargs = {
        key: decoder_config["decoder"][key]
        for key in allowed
        if key in decoder_config["decoder"]
    }
    # This also fails early if a changed frozen config contains an unsupported
    # decoder parameter, instead of producing a misleading checkpoint error.
    Agile3DClickDecoder(**kwargs)
    return kwargs


def output_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["paths"]["output_root"])).expanduser().resolve()


def artifact_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["paths"]["articulate3d_root"])).expanduser().resolve(strict=True)


def load_manifest(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = artifact_root(config)
    path = root / "manifest.json"
    manifest = read_json(path)
    if manifest.get("schema") != "delimit3d_articulate3d_feature_manifest/v1":
        raise ValueError(f"{path}: unexpected manifest schema")
    expected_hash = manifest.get("manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_sha256", None)
    if expected_hash != sha256_json(without_hash):
        raise ValueError(f"{path}: manifest hash mismatch")
    if manifest.get("status") != "complete":
        raise ValueError(f"{path}: manifest is not complete")
    split_path = Path(str(manifest["official_split"])).expanduser().resolve(strict=True)
    split = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(split) != 42 or len(set(split)) != 42:
        raise ValueError(f"{split_path}: expected 42 unique validation scenes")
    if sha256_file(split_path) != manifest["official_split_sha256"]:
        raise ValueError(f"{split_path}: split hash mismatch")
    records = {str(record["scene"]): dict(record) for record in manifest.get("scenes", [])}
    if len(records) != 42 or set(records) != set(split):
        raise ValueError("manifest scenes do not exactly match the official 42-scene split")
    for scene in split:
        record = records[scene]
        annotation = Path(str(record["annotation_path"])).expanduser().resolve(strict=True)
        if sha256_file(annotation) != record["annotation_sha256"]:
            raise ValueError(f"{scene}: annotation hash mismatch")
        token_record = record.get("token_to_part", {})
        mapping = Path(str(token_record["path"])).expanduser().resolve(strict=True)
        if sha256_file(mapping) != token_record["sha256"]:
            raise ValueError(f"{scene}: token-to-part mapping hash mismatch")
        for arm in ("public", "delimit3d"):
            cache_record = record.get("cache", {}).get(arm)
            if not isinstance(cache_record, Mapping):
                raise ValueError(f"{scene}: missing {arm} cache record")
            cache_path = Path(str(cache_record["path"])).expanduser().resolve(strict=True)
            if sha256_file(cache_path) != cache_record["sha256"]:
                raise ValueError(f"{scene}: {arm} cache hash mismatch")
    return manifest, records


def decoder_paths_and_hashes(config: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    root = Path(str(config["paths"]["decoder_root"])).expanduser().resolve(strict=True)
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for arm in ("public", "delimit3d"):
        configured = config["decoders"][arm]
        path = Path(str(configured["checkpoint"])).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != str(configured["checkpoint_sha256"]):
            raise ValueError(f"{arm}: decoder checkpoint hash mismatch")
        paths[arm] = path
        hashes[arm] = actual
    if not (root / "freeze" / "resolved_config.yaml").is_file():
        raise FileNotFoundError(f"{root}: frozen decoder config is missing")
    return paths, hashes


def load_decoder(path: Path, arm: str, kwargs: Mapping[str, Any]) -> Agile3DClickDecoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: decoder checkpoint schema mismatch")
    if str(payload.get("arm")) != arm:
        raise ValueError(f"{path}: checkpoint arm is {payload.get('arm')!r}, expected {arm!r}")
    if dict(payload.get("decoder_kwargs", {})) != dict(kwargs):
        raise ValueError(f"{path}: decoder kwargs do not match the frozen training config")
    decoder = Agile3DClickDecoder(**dict(kwargs))
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    actual_state_hash = state_sha256(decoder.state_dict())
    if actual_state_hash != payload.get("decoder_tensor_sha256"):
        raise ValueError(f"{path}: decoder tensor hash mismatch")
    decoder.eval()
    return decoder


def load_annotation(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping) or not isinstance(value.get("data"), Mapping):
        raise ValueError(f"{path}: unexpected Articulate3D annotation schema")
    annotations = value["data"].get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"{path}: missing data.annotations")
    return dict(value)


def load_mapping(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping) or value.get("schema") != "delimit3d_articulate3d_token_to_part_scene/v1":
        raise ValueError(f"{path}: unexpected token-to-part schema")
    expected = value.get("mapping_sha256")
    without_hash = dict(value)
    without_hash.pop("mapping_sha256", None)
    if expected != sha256_json(without_hash):
        raise ValueError(f"{path}: mapping hash mismatch")
    return dict(value)


def load_cache(
    path: Path,
    expected_scene: str,
    expected_checkpoint: str,
    cache_record: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") not in CACHE_SCHEMAS:
        raise ValueError(f"{path}: unsupported cache schema {payload.get('schema') if isinstance(payload, Mapping) else None!r}")
    result = dict(payload)
    if str(result.get("scene")) != expected_scene:
        raise ValueError(f"{path}: cache scene mismatch")
    payload_checkpoint = result.get("checkpoint_sha256", cache_record.get("checkpoint_sha256"))
    if str(payload_checkpoint) != expected_checkpoint:
        raise ValueError(f"{path}: cache checkpoint mismatch")
    if "encoder_frozen" in result and result.get("encoder_frozen") is not True:
        raise ValueError(f"{path}: cache does not certify a frozen encoder")
    for key in ("features", "scene_xyz", "inverse_map", "representative_indices"):
        if key not in result or not torch.is_tensor(result[key]):
            raise ValueError(f"{path}: missing tensor {key}")
    features = result["features"]
    scene_xyz = result["scene_xyz"]
    inverse_map = result["inverse_map"]
    representatives = result["representative_indices"]
    if features.ndim != 2 or int(features.shape[1]) != 72:
        raise ValueError(f"{path}: expected [T,72] features")
    if tuple(scene_xyz.shape) != (int(features.shape[0]), 3):
        raise ValueError(f"{path}: scene_xyz does not align with features")
    if inverse_map.ndim != 1 or representatives.ndim != 1:
        raise ValueError(f"{path}: invalid inverse/representative shape")
    source_points = result.get("source_points", result.get("points", -1))
    if int(inverse_map.shape[0]) != int(source_points):
        raise ValueError(f"{path}: inverse-map length does not match source_points")
    if int(representatives.shape[0]) != int(features.shape[0]):
        raise ValueError(f"{path}: representative count does not match tokens")
    inverse = inverse_map.detach().cpu().numpy().astype(np.int64, copy=False)
    reps = representatives.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(inverse) and (int(inverse.min()) < 0 or int(inverse.max()) >= len(features)):
        raise ValueError(f"{path}: inverse-map index out of range")
    if len(reps) and (int(reps.min()) < 0 or int(reps.max()) >= len(inverse)):
        raise ValueError(f"{path}: representative index out of range")
    if len(np.unique(reps)) != len(reps):
        raise ValueError(f"{path}: representative indices are not unique")
    return result


def build_part_specs(
    scene: str,
    annotation: Mapping[str, Any],
    mapping: Mapping[str, Any],
    inverse_map: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    raw_annotations = annotation["data"]["annotations"]
    mapping_parts = mapping.get("parts")
    if not isinstance(mapping_parts, list):
        raise ValueError(f"{scene}: token-to-part parts are missing")
    annotation_parts = {
        int(part["partId"]): dict(part)
        for part in raw_annotations
    }
    if len(annotation_parts) != len(raw_annotations):
        raise ValueError(f"{scene}: duplicate annotation part IDs")
    mapped_parts = {int(part["part_id"]): dict(part) for part in mapping_parts}
    if set(annotation_parts) != set(mapped_parts):
        raise ValueError(f"{scene}: annotation and mapping part IDs differ")

    token_owners: dict[int, set[int]] = defaultdict(set)
    part_all_tokens: dict[int, list[int]] = {}
    part_vertices: dict[int, np.ndarray] = {}
    for part_id, part in sorted(annotation_parts.items()):
        vertices = np.asarray(part.get("vertIndices", []), dtype=np.int64)
        if vertices.ndim != 1:
            raise ValueError(f"{scene}: part {part_id} vertices are not one-dimensional")
        if len(vertices) and (int(vertices.min()) < 0 or int(vertices.max()) >= len(inverse_map)):
            raise ValueError(f"{scene}: part {part_id} vertex index out of range")
        vertices = np.unique(vertices)
        tokens = sorted(set(int(value) for value in inverse_map[vertices])) if len(vertices) else []
        part_vertices[part_id] = vertices
        part_all_tokens[part_id] = tokens
        for token in tokens:
            token_owners[token].add(part_id)
    ambiguous = {token for token, owners in token_owners.items() if len(owners) > 1}
    ambiguous_array = np.asarray(sorted(ambiguous), dtype=np.int64)
    valid_token_mask = np.ones(len(np.asarray(inverse_map)), dtype=bool)
    # The mask is initially indexed by raw points below; convert it after we
    # know the token count from the maximum inverse-map value.
    token_count = int(inverse_map.max()) + 1 if len(inverse_map) else 0
    valid_token_mask = np.ones(token_count, dtype=bool)
    if len(ambiguous_array):
        valid_token_mask[ambiguous_array] = False
    specs: list[dict[str, Any]] = []
    for part_id, mapped in sorted(mapped_parts.items()):
        calculated = sorted(token for token in part_all_tokens[part_id] if token not in ambiguous)
        recorded = sorted(int(value) for value in mapped.get("surviving_unambiguous_token_ids", []))
        if calculated != recorded:
            raise ValueError(f"{scene}: part {part_id} token mapping drift")
        vertices = part_vertices[part_id]
        if len(ambiguous):
            raw_valid = ~np.isin(inverse_map[vertices], ambiguous_array)
            target_vertices = vertices[raw_valid]
        else:
            target_vertices = vertices
        specs.append(
            {
                "scene": scene,
                "part_id": int(part_id),
                "object_id": int(mapped.get("object_id", -1)),
                "label": str(mapped.get("label", "")),
                "raw_vertex_count": int(len(vertices)),
                "surviving_unambiguous_token_count": int(len(calculated)),
                "token_ids": np.asarray(calculated, dtype=np.int64),
                "raw_target_indices": np.asarray(target_vertices, dtype=np.int64),
                "ambiguous_token_count_scene": int(len(ambiguous)),
            }
        )
    raw_valid_mask = np.ones(len(inverse_map), dtype=bool)
    if len(ambiguous_array):
        raw_valid_mask &= ~np.isin(inverse_map, ambiguous_array)
    return specs, valid_token_mask, raw_valid_mask


def safe_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = int(np.count_nonzero(prediction | target))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(prediction & target) / union)


def metric_state(
    token_prediction: np.ndarray,
    token_target: np.ndarray,
    inverse_map: np.ndarray,
    valid_token_mask: np.ndarray,
    raw_valid_mask: np.ndarray,
    raw_target_indices: np.ndarray,
) -> tuple[float, float, int]:
    token_prediction = np.asarray(token_prediction, dtype=np.int64)
    token_target = np.asarray(token_target, dtype=np.int64)
    token_iou = safe_iou(
        (token_prediction == 1) & valid_token_mask,
        (token_target == 1) & valid_token_mask,
    )
    raw_prediction = token_prediction[np.asarray(inverse_map, dtype=np.int64)]
    raw_target = np.zeros(len(inverse_map), dtype=bool)
    raw_target[np.asarray(raw_target_indices, dtype=np.int64)] = True
    raw_iou = safe_iou(
        (raw_prediction == 1) & raw_valid_mask,
        raw_target & raw_valid_mask,
    )
    return raw_iou, token_iou, int(np.count_nonzero(raw_target & raw_valid_mask))


def state_at_threshold(states: Sequence[Mapping[str, Any]], threshold: int) -> Mapping[str, Any]:
    eligible = [state for state in states if int(state["total_clicks"]) <= int(threshold)]
    if not eligible:
        return states[0]
    return eligible[-1]


def evaluate_part(
    decoder: Agile3DClickDecoder,
    cache: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    device: torch.device,
    protocol: Mapping[str, Any],
    valid_token_mask: np.ndarray,
    raw_valid_mask: np.ndarray,
) -> dict[str, Any]:
    features_cpu = cache["features"]
    xyz_cpu = cache["scene_xyz"]
    inverse_map = cache["inverse_map"].detach().cpu().numpy().astype(np.int64, copy=False)
    features = features_cpu.to(device=device, dtype=torch.float32)
    xyz = xyz_cpu.to(device=device, dtype=torch.float32)
    xyz_np = xyz_cpu.detach().cpu().numpy().astype(np.float64, copy=False)
    token_ids = np.asarray(spec["token_ids"], dtype=np.int64)
    target_token = np.zeros(int(features.shape[0]), dtype=np.int64)
    target_token[token_ids] = 1
    click = center_click(token_ids, xyz_np)
    clicks: dict[str, list[int]] = {"0": [], "1": [int(click)]}
    times: dict[str, list[int]] = {"0": [], "1": [0]}
    states: list[dict[str, Any]] = []
    budget = int(protocol["click_budget"])
    decoder.eval()
    with torch.inference_mode():
        while True:
            output = decoder(features, xyz, clicks=clicks, click_times=times)
            prediction = output["pred_masks"].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            prediction = enforce_click_labels(prediction, clicks)
            raw_iou, token_iou, target_points = metric_state(
                prediction,
                target_token,
                inverse_map,
                valid_token_mask,
                raw_valid_mask,
                np.asarray(spec["raw_target_indices"], dtype=np.int64),
            )
            total_clicks = sum(len(values) for values in clicks.values())
            states.append(
                {
                    "state": int(len(states)),
                    "total_clicks": int(total_clicks),
                    "clicks_per_part": float(total_clicks),
                    "clicks": {key: list(values) for key, values in clicks.items()},
                    "click_times": {key: list(values) for key, values in times.items()},
                    "raw_point_iou": float(raw_iou),
                    "token_iou": float(token_iou),
                    "target_raw_points": int(target_points),
                }
            )
            del output
            if total_clicks >= budget:
                break
            new_clicks, new_times, events = simulated_corrections(
                prediction,
                target_token,
                xyz_np,
                clicks,
                times,
                training=False,
                click_center_method=str(protocol["click_center_method"]),
                max_clicks_per_label=budget,
            )
            if not new_clicks:
                break
            states[-1]["new_click_events"] = events
            clicks, times = append_clicks(clicks, times, new_clicks, new_times)

    thresholds: dict[str, Any] = {}
    for threshold in [int(value) for value in protocol["click_thresholds"]]:
        state = state_at_threshold(states, threshold)
        thresholds[str(threshold)] = {
            "total_clicks": int(state["total_clicks"]),
            "clicks_per_part": float(state["clicks_per_part"]),
            "raw_point_iou": float(state["raw_point_iou"]),
            "token_iou": float(state["token_iou"]),
        }
    noc: dict[str, float] = {}
    for iou_threshold in [float(value) for value in protocol["iou_thresholds"]]:
        reached = [
            float(state["clicks_per_part"])
            for state in states
            if float(state["raw_point_iou"]) >= iou_threshold
        ]
        noc[f"{iou_threshold:.2f}"] = float(min(reached) if reached else budget)
    return {
        "schema": "delimit3d_articulate3d_part_result/v1",
        "scene": str(spec["scene"]),
        "part_id": int(spec["part_id"]),
        "object_id": int(spec["object_id"]),
        "label": str(spec["label"]),
        "raw_vertex_count": int(spec["raw_vertex_count"]),
        "surviving_unambiguous_token_count": int(spec["surviving_unambiguous_token_count"]),
        "ambiguous_token_count_scene": int(spec["ambiguous_token_count_scene"]),
        "protocol_metric": PRIMARY_METRIC,
        "states": states,
        "thresholds": thresholds,
        "noc": noc,
    }


def metric_value(row: Mapping[str, Any], min_tokens: int, metric: str) -> float | None:
    if int(row["surviving_unambiguous_token_count"]) < int(min_tokens):
        return None
    if metric.startswith("raw_point_iou@"):
        threshold = metric.split("@", 1)[1]
        return float(row["thresholds"][threshold]["raw_point_iou"])
    if metric.startswith("token_iou@"):
        threshold = metric.split("@", 1)[1]
        return float(row["thresholds"][threshold]["token_iou"])
    if metric.startswith("noc@"):
        threshold = metric.split("@", 1)[1]
        return float(row["noc"][threshold])
    raise KeyError(metric)


def summarize_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    encoder_checkpoint_sha256: str,
    protocol: Mapping[str, Any],
    output_root_path: Path,
) -> dict[str, Any]:
    sensitivity = [int(value) for value in protocol["sensitivity_min_surviving_unambiguous_tokens"]]
    click_thresholds = [int(value) for value in protocol["click_thresholds"]]
    iou_thresholds = [float(value) for value in protocol["iou_thresholds"]]
    metrics: dict[str, Any] = {}
    metric_names = [
        *(f"raw_point_iou@{value}" for value in click_thresholds),
        *(f"token_iou@{value}" for value in click_thresholds),
        *(f"noc@{value:.2f}" for value in iou_thresholds),
    ]
    for minimum in sensitivity:
        selected = [row for row in rows if int(row["surviving_unambiguous_token_count"]) >= minimum]
        summary: dict[str, Any] = {"part_count": len(selected)}
        for metric in metric_names:
            values = [metric_value(row, minimum, metric) for row in selected]
            values = [float(value) for value in values if value is not None]
            if values:
                summary[metric] = float(np.mean(values))
        metrics[str(minimum)] = summary

    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene"])].append(row)
    scene_rows: list[dict[str, Any]] = []
    for scene, scene_parts in sorted(by_scene.items()):
        scene_metrics: dict[str, Any] = {}
        for minimum in sensitivity:
            selected = [
                row
                for row in scene_parts
                if int(row["surviving_unambiguous_token_count"]) >= minimum
            ]
            item: dict[str, Any] = {"part_count": len(selected)}
            for metric in metric_names:
                values = [metric_value(row, minimum, metric) for row in selected]
                values = [float(value) for value in values if value is not None]
                if values:
                    item[metric] = float(np.mean(values))
            scene_metrics[str(minimum)] = item
        scene_rows.append({"scene": scene, "metrics": scene_metrics})

    arm_root = output_root_path / arm
    parts_path = arm_root / "parts.jsonl"
    scenes_path = arm_root / "scenes.jsonl"
    with scenes_path.open("w") as handle:
        for row in scene_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "delimit3d_articulate3d_arm_part_evaluation/v1",
        "arm": arm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "part_rows": str(parts_path),
        "scene_rows": str(scenes_path),
        "part_count_evaluated": len(rows),
        "scene_count_evaluated": len(scene_rows),
        "metrics_by_minimum_surviving_tokens": metrics,
        "primary_minimum_surviving_tokens": int(protocol["primary_min_surviving_unambiguous_tokens"]),
    }
    write_json(arm_root / "arm_report.json", report)
    return report


def prepare_run_record(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    decoder_paths: Mapping[str, Path],
    decoder_hashes: Mapping[str, str],
    kwargs: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_root_path: Path,
) -> dict[str, Any]:
    evaluator_path = Path(__file__).resolve()
    config_path = Path(str(config["_config_path"])).resolve(strict=True)
    record = {
        "schema": RUN_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "repo_commit": git_commit(),
        "evaluator": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "artifact_root": str(artifact_root(config)),
        "artifact_manifest": str(artifact_root(config) / "manifest.json"),
        "artifact_manifest_sha256": str(manifest["manifest_sha256"]),
        "decoder_root": str(Path(str(config["paths"]["decoder_root"])).expanduser().resolve()),
        "decoder_config": str(Path(str(config["paths"]["decoder_config"])).expanduser().resolve()),
        "decoder_kwargs": dict(kwargs),
        "decoders": {
            arm: {"path": str(decoder_paths[arm]), "sha256": decoder_hashes[arm]}
            for arm in ("public", "delimit3d")
        },
        "protocol": dict(protocol),
        "scene_count": 42,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    path = output_root_path / "run_manifest.json"
    if path.exists():
        old = read_json(path)
        for key in (
            "experiment_id",
            "artifact_manifest_sha256",
            "config_sha256",
            "evaluator_sha256",
            "decoders",
            "decoder_kwargs",
            "protocol",
        ):
            if old.get(key) != record.get(key):
                raise ValueError(f"{path}: existing run record differs in {key}")
        record["started"] = old.get("started", record["started"])
    else:
        write_json(path, record)
    return record


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_config(config)
    manifest, records = load_manifest(config)
    decoder_paths, decoder_hashes = decoder_paths_and_hashes(config)
    kwargs = decoder_kwargs(config)
    # Loading the two checkpoints on CPU checks the exact architecture and
    # tensor hashes before any long scene evaluation is started.
    for arm in ("public", "delimit3d"):
        decoder = load_decoder(decoder_paths[arm], arm, kwargs)
        del decoder

    encoder_hashes = {
        arm: str(manifest["encoders"][arm]["checkpoint_sha256"])
        for arm in ("public", "delimit3d")
    }
    total_parts = 0
    eligible_counts = {"5": 0, "10": 0, "20": 0}
    scene_summaries: list[dict[str, Any]] = []
    for index, scene in enumerate(sorted(records), start=1):
        record = records[scene]
        annotation = load_annotation(Path(str(record["annotation_path"])).expanduser().resolve(strict=True))
        mapping = load_mapping(
            Path(str(record["token_to_part"]["path"])).expanduser().resolve(strict=True)
        )
        caches: dict[str, dict[str, Any]] = {}
        for arm in ("public", "delimit3d"):
            cache_path = Path(str(record["cache"][arm]["path"])).expanduser().resolve(strict=True)
            caches[arm] = load_cache(
                cache_path,
                scene,
                encoder_hashes[arm],
                record["cache"][arm],
            )
        public_inverse = caches["public"]["inverse_map"].detach().cpu()
        delimit_inverse = caches["delimit3d"]["inverse_map"].detach().cpu()
        if not torch.equal(public_inverse, delimit_inverse):
            raise ValueError(f"{scene}: public and Delimit3D inverse maps differ")
        for key in ("scene_xyz", "representative_indices"):
            if not torch.equal(caches["public"][key].detach().cpu(), caches["delimit3d"][key].detach().cpu()):
                raise ValueError(f"{scene}: public and Delimit3D {key} differ")
        public_geometry = str(caches["public"].get("geometry_sha256", ""))
        delimit_geometry = str(caches["delimit3d"].get("geometry_sha256", ""))
        if public_geometry != delimit_geometry:
            raise ValueError(f"{scene}: public and Delimit3D geometry hashes differ")
        specs, _valid_token_mask, _raw_valid_mask = build_part_specs(
            scene,
            annotation,
            mapping,
            public_inverse.numpy().astype(np.int64, copy=False),
        )
        scene_counts = {
            str(threshold): sum(
                int(spec["surviving_unambiguous_token_count"]) >= threshold
                for spec in specs
            )
            for threshold in (5, 10, 20)
        }
        total_parts += len(specs)
        for threshold in (5, 10, 20):
            eligible_counts[str(threshold)] += scene_counts[str(threshold)]
        scene_summaries.append(
            {
                "scene": scene,
                "part_count": len(specs),
                "eligible_part_counts": scene_counts,
                "token_count": int(caches["public"]["features"].shape[0]),
                "point_count": int(caches["public"]["inverse_map"].shape[0]),
                "geometry_sha256": public_geometry,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "preflight",
                    "scene": scene,
                    "completed": index,
                    "total": len(records),
                    "parts": len(specs),
                    "eligible": scene_counts,
                }
            ),
            flush=True,
        )
        del caches
    result = {
        "schema": "delimit3d_articulate3d_parts_preflight/v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "scene_count": len(records),
        "part_count": total_parts,
        "eligible_part_counts": eligible_counts,
        "encoder_checkpoint_sha256": encoder_hashes,
        "decoder_checkpoint_sha256": decoder_hashes,
        "decoder_kwargs": kwargs,
        "protocol": protocol,
        "scenes": scene_summaries,
    }
    write_json(output_root(config) / "preflight.json", result)
    return {
        "config": config,
        "protocol": protocol,
        "manifest": manifest,
        "records": records,
        "decoder_paths": decoder_paths,
        "decoder_hashes": decoder_hashes,
        "encoder_hashes": encoder_hashes,
        "decoder_kwargs": kwargs,
        "preflight": result,
    }


def evaluate_arm(ctx: Mapping[str, Any], arm: str) -> dict[str, Any]:
    config = ctx["config"]
    protocol = ctx["protocol"]
    records = ctx["records"]
    output_root_path = output_root(config)
    arm_root = output_root_path / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    parts_path = arm_root / "parts.jsonl"
    existing_rows = read_jsonl(parts_path)
    existing: dict[tuple[str, int], dict[str, Any]] = {
        (str(row["scene"]), int(row["part_id"])): row for row in existing_rows
    }
    decoder = load_decoder(ctx["decoder_paths"][arm], arm, ctx["decoder_kwargs"]).to("cuda")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    set_seed(int(config["seed"]) + (0 if arm == "public" else 1))
    run_minimum = int(protocol["run_min_surviving_unambiguous_tokens"])
    total_to_run = int(ctx["preflight"]["eligible_part_counts"][str(run_minimum)])
    completed = sum(
        1
        for row in existing.values()
        if int(row.get("surviving_unambiguous_token_count", 0)) >= run_minimum
    )
    started = time.time()
    update_status(
        output_root_path,
        phase="evaluate",
        arm=arm,
        completed=completed,
        total=total_to_run,
        scene=None,
        part=None,
    )
    for scene_index, scene in enumerate(sorted(records), start=1):
        record = records[scene]
        cache_path = Path(str(record["cache"][arm]["path"])).expanduser().resolve(strict=True)
        cache = load_cache(
            cache_path,
            scene,
            ctx["encoder_hashes"][arm],
            record["cache"][arm],
        )
        inverse = cache["inverse_map"].detach().cpu().numpy().astype(np.int64, copy=False)
        annotation = load_annotation(Path(str(record["annotation_path"])).expanduser().resolve(strict=True))
        mapping = load_mapping(Path(str(record["token_to_part"]["path"])).expanduser().resolve(strict=True))
        specs, valid_token_mask, raw_valid_mask = build_part_specs(scene, annotation, mapping, inverse)
        for spec in specs:
            if int(spec["surviving_unambiguous_token_count"]) < run_minimum:
                continue
            key = (scene, int(spec["part_id"]))
            if key in existing:
                continue
            row = evaluate_part(
                decoder,
                cache,
                spec,
                device=device,
                protocol=protocol,
                valid_token_mask=valid_token_mask,
                raw_valid_mask=raw_valid_mask,
            )
            row["arm"] = arm
            row["decoder_checkpoint_sha256"] = ctx["decoder_hashes"][arm]
            row["encoder_checkpoint_sha256"] = ctx["encoder_hashes"][arm]
            row["scene_index"] = scene_index
            append_jsonl(parts_path, row)
            existing[key] = row
            completed += 1
            update_status(
                output_root_path,
                phase="evaluate",
                arm=arm,
                completed=completed,
                total=total_to_run,
                scene=scene,
                part=int(spec["part_id"]),
                elapsed_seconds=float(time.time() - started),
            )
            print(
                json.dumps(
                    {
                        "phase": "evaluate",
                        "arm": arm,
                        "scene": scene,
                        "part": int(spec["part_id"]),
                        "completed": completed,
                        "total": total_to_run,
                        "elapsed_seconds": float(time.time() - started),
                        "raw_point_iou_at_1": row["thresholds"]["1"]["raw_point_iou"],
                    }
                ),
                flush=True,
            )
        del cache
        torch.cuda.empty_cache()
    rows = list(existing.values())
    rows.sort(key=lambda row: (str(row["scene"]), int(row["part_id"])))
    report = summarize_arm(
        rows,
        arm=arm,
        checkpoint=ctx["decoder_paths"][arm],
        checkpoint_sha256=ctx["decoder_hashes"][arm],
        encoder_checkpoint_sha256=ctx["encoder_hashes"][arm],
        protocol=protocol,
        output_root_path=output_root_path,
    )
    update_status(
        output_root_path,
        phase="evaluate_arm_complete",
        arm=arm,
        completed=len(rows),
        total=total_to_run,
    )
    del decoder
    torch.cuda.empty_cache()
    return report


def aggregate_reports(ctx: Mapping[str, Any], arm_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    config = ctx["config"]
    protocol = ctx["protocol"]
    sensitivity = [int(value) for value in protocol["sensitivity_min_surviving_unambiguous_tokens"]]
    click_thresholds = [int(value) for value in protocol["click_thresholds"]]
    iou_thresholds = [float(value) for value in protocol["iou_thresholds"]]
    output_root_path = output_root(config)
    scene_tables: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("public", "delimit3d"):
        rows = read_jsonl(output_root_path / arm / "parts.jsonl")
        for scene in sorted({str(row["scene"]) for row in rows}):
            scene_tables.setdefault(scene, {})[arm] = next(
                row for row in read_jsonl(output_root_path / arm / "scenes.jsonl")
                if str(row["scene"]) == scene
            )
    bootstrap: dict[str, Any] = {}
    for minimum in sensitivity:
        for click_threshold in click_thresholds:
            metric = f"raw_point_iou@{click_threshold}"
            public_values = {
                scene: float(table["public"]["metrics"][str(minimum)][metric])
                for scene, table in scene_tables.items()
                if "public" in table and metric in table["public"]["metrics"].get(str(minimum), {})
            }
            delimit_values = {
                scene: float(table["delimit3d"]["metrics"][str(minimum)][metric])
                for scene, table in scene_tables.items()
                if "delimit3d" in table and metric in table["delimit3d"]["metrics"].get(str(minimum), {})
            }
            paired = set(public_values).intersection(delimit_values)
            if paired:
                bootstrap[f"{minimum}:{metric}"] = paired_scene_bootstrap(
                    {scene: public_values[scene] for scene in paired},
                    {scene: delimit_values[scene] for scene in paired},
                    metric,
                    samples=int(config["bootstrap"]["samples"]),
                    seed=int(config["bootstrap"]["seed"]) + minimum + click_threshold,
                )
        for iou_threshold in iou_thresholds:
            metric = f"noc@{iou_threshold:.2f}"
            public_values = {
                scene: float(table["public"]["metrics"][str(minimum)][metric])
                for scene, table in scene_tables.items()
                if "public" in table and metric in table["public"]["metrics"].get(str(minimum), {})
            }
            delimit_values = {
                scene: float(table["delimit3d"]["metrics"][str(minimum)][metric])
                for scene, table in scene_tables.items()
                if "delimit3d" in table and metric in table["delimit3d"]["metrics"].get(str(minimum), {})
            }
            paired = set(public_values).intersection(delimit_values)
            if paired:
                bootstrap[f"{minimum}:{metric}"] = paired_scene_bootstrap(
                    {scene: public_values[scene] for scene in paired},
                    {scene: delimit_values[scene] for scene in paired},
                    metric,
                    samples=int(config["bootstrap"]["samples"]),
                    seed=int(config["bootstrap"]["seed"]) + minimum + int(round(iou_threshold * 100)),
                )
    report = {
        "schema": "delimit3d_articulate3d_paired_part_aggregate/v1",
        "experiment_id": str(config["experiment_id"]),
        "manifest_sha256": ctx["manifest"]["manifest_sha256"],
        "scene_count": len(scene_tables),
        "primary_minimum_surviving_tokens": int(protocol["primary_min_surviving_unambiguous_tokens"]),
        "public": dict(arm_reports["public"]),
        "delimit3d": dict(arm_reports["delimit3d"]),
        "paired_scene_bootstrap_delimit3d_minus_public": bootstrap,
        "metric_definition": {
            "primary": "raw point IoU after inverse-map expansion",
            "ambiguous_tokens": "excluded from primary token and raw-point scores",
            "part_selection": "one annotated part per AGILE3D episode",
            "click_policy": "center initial click plus one deepest-error correction click",
            "click_budget": int(protocol["click_budget"]),
        },
    }
    write_json(output_root_path / "aggregate.json", report)
    update_status(
        output_root_path,
        phase="complete",
        arm="all",
        completed=int(ctx["preflight"]["eligible_part_counts"][str(protocol["run_min_surviving_unambiguous_tokens"])]),
        total=int(ctx["preflight"]["eligible_part_counts"][str(protocol["run_min_surviving_unambiguous_tokens"])]),
    )
    return report


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_root_path = output_root(config)
    output_root_path.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate":
        # Aggregate-only mode still validates the immutable run inputs before
        # trusting partial reports.
        ctx = preflight(config)
        reports = {
            arm: read_json(output_root_path / arm / "arm_report.json")
            for arm in ("public", "delimit3d")
        }
        aggregate_reports(ctx, reports)
        return 0

    ctx = preflight(config)
    prepare_run_record(
        config,
        ctx["manifest"],
        ctx["decoder_paths"],
        ctx["decoder_hashes"],
        ctx["decoder_kwargs"],
        ctx["protocol"],
        output_root_path,
    )
    if args.mode == "preflight":
        update_status(output_root_path, phase="preflight_complete", scene_count=42)
        return 0

    arms = ["public", "delimit3d"] if args.arm == "all" else [args.arm]
    arm_reports: dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        arm_reports[arm] = evaluate_arm(ctx, arm)
    if args.arm == "all":
        aggregate_reports(ctx, arm_reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
