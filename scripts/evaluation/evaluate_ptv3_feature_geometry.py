
#!/usr/bin/env python3
"""Matched PTv3 frozen-feature diagnostic for the post-adaptation go/no-go.

This is deliberately upstream of AGILE3D decoder training.  It compares the
public PTv3 encoder with the completed 256-update Delimit3D PTv3 encoder on
the same frozen ScanNet++ validation scenes, points, object queries, and
candidate sets.

Modes:
  prepare      freeze the validation manifest and resolved provenance
  evaluate     evaluate one arm (public or delimit3d)
  aggregate    paired scene bootstrap and the predeclared decision rule
  wait-and-run wait for the final adaptation checkpoint, evaluate both arms,
               then aggregate automatically
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.training.ptv3_adaptation import PTv3Encoder

SCHEMA = "delimit3d_ptv3_feature_geometry_gate/v1"
FEATURE_DIM = 64
QUERY_TOP_FRACTIONS = (0.001, 0.01, 0.05, 0.10)
KNN_K = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("prepare", "evaluate", "aggregate", "wait-and-run"),
        required=True,
    )
    parser.add_argument("--arm", choices=("public", "delimit3d"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    return parser.parse_args()


def resolve_path(value: str | Path, *, strict: bool = True) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.exists():
        return path.resolve(strict=strict)
    if str(path).startswith("/cluster/"):
        relative = str(path).removeprefix("/cluster/")
        mounts = []
        configured = os.environ.get("DELIMIT3D_EULER_MOUNT")
        if configured:
            mounts.append(Path(configured))
        mounts.extend((Path("/tmp/euler_cluster_nedela_rw"), Path("/tmp/euler_cluster_nedela")))
        for mount in mounts:
            candidate = mount / relative
            if candidate.exists() or not strict:
                return candidate.resolve(strict=strict)
    return path.resolve(strict=strict)


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve(strict=True).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: YAML root must be a mapping")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def output_root(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["paths"]["output_root"], strict=False)


def manifest_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    bundle = resolve_path(config["paths"]["val_bundle"], strict=True)
    return bundle, bundle / "scene_manifest.json"


def load_scene_manifest(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    bundle, path = manifest_paths(config)
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or not isinstance(payload.get("scenes"), list):
        raise ValueError(f"{path}: expected a scene manifest with a scenes list")
    return bundle, dict(payload)


def scene_data_path(bundle: Path, record: Mapping[str, Any]) -> Path:
    configured = record.get("data_path")
    if configured:
        return resolve_path(str(configured), strict=True)
    scene = str(record["scene"])
    candidates = sorted((bundle / "data").glob("*.npz"))
    matches = [path for path in candidates if path.stem == scene]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{scene}: scene manifest has no usable data_path")


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    bundle, scene_manifest = load_scene_manifest(config)
    scene_manifest_path = bundle / "scene_manifest.json"
    scenes = list(scene_manifest["scenes"])
    expected = int(config.get("protocol", {}).get("scene_count", len(scenes)))
    if len(scenes) != expected:
        raise RuntimeError(f"expected {expected} validation scenes, found {len(scenes)}")
    records = []
    for record in scenes:
        path = scene_data_path(bundle, record)
        records.append({
            "scene": str(record["scene"]),
            "data_path": str(path),
            "data_sha256": str(record.get("data_sha256") or sha256_file(path)),
            "points": int(record.get("points", -1)),
            "object_count": int(len(record.get("objects", []))),
            "objects": record.get("objects", []),
        })
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    freeze = root / "freeze"
    freeze.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved.setdefault("provenance", {})
    resolved["provenance"] = dict(resolved["provenance"])
    resolved["provenance"].update({
        "schema": SCHEMA,
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=False
        ).stdout.strip() or "unknown",
        "hostname": platform.node(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "ptv3_source_commit": config.get("model", {}).get("ptv3_source_commit"),
        "scene_manifest_path": str(scene_manifest_path),
        "scene_manifest_sha256": sha256_file(scene_manifest_path),
        "scene_count": len(records),
    })
    (freeze / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )
    manifest = {
        "schema": SCHEMA,
        "scene_manifest_path": str(scene_manifest_path),
        "scene_manifest_sha256": sha256_file(scene_manifest_path),
        "bundle": str(bundle),
        "scenes": records,
        "public_checkpoint": str(
            resolve_path(config["arms"]["public"]["checkpoint"], strict=False)
        ),
        "adapted_checkpoint": str(
            resolve_path(config["arms"]["delimit3d"]["checkpoint"], strict=False)
        ),
    }
    write_json(freeze / "feature_gate_manifest.json", manifest)
    write_json(root / "prepare_report.json", {
        "schema": SCHEMA,
        "scene_count": len(records),
        "scene_manifest_sha256": manifest["scene_manifest_sha256"],
        "prepared": True,
        "adapted_checkpoint_exists": Path(manifest["adapted_checkpoint"]).exists(),
    })
    return manifest


def payload_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            state = payload.get(key)
            if isinstance(state, Mapping):
                return state
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise TypeError("checkpoint does not contain a tensor state mapping")


def adapted_backbone_state(payload: Any) -> dict[str, torch.Tensor]:
    state = payload_state(payload)
    result: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = str(raw_name).removeprefix("module.")
        if name.startswith("encoder.backbone."):
            name = name.removeprefix("encoder.")
        elif name.startswith("encoder."):
            name = name.removeprefix("encoder.")
        elif not name.startswith("backbone."):
            continue
        result[name] = value.detach().cpu()
    if not result:
        raise RuntimeError("adapted checkpoint has no encoder.backbone tensors")
    return result


def encoder_state_sha256(model: torch.nn.Module) -> str:
    # The adaptation checkpoint records the runner's parameter-only hash.
    # Keep this definition identical so buffers cannot create a false drift.
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_encoder(config: Mapping[str, Any], arm: str, device: torch.device):
    model_cfg = config["model"]
    public_path = resolve_path(config["arms"]["public"]["checkpoint"], strict=True)
    checkpoint_path = resolve_path(config["arms"][arm]["checkpoint"], strict=True)
    expected = config["arms"][arm].get("checkpoint_sha256")
    observed = sha256_file(checkpoint_path)
    if expected and str(expected) not in {"auto", "unknown"} and observed != str(expected):
        raise RuntimeError(
            f"{arm} checkpoint SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    encoder = PTv3Encoder(
        ptv3_root=resolve_path(model_cfg["ptv3_root"], strict=True),
        checkpoint=public_path,
        grid_size=float(model_cfg.get("grid_size", 0.02)),
        enable_flash=bool(model_cfg.get("enable_flash", True)),
        shuffle_orders=bool(model_cfg.get("shuffle_orders", True)),
    )
    payload = None
    if arm == "delimit3d":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = adapted_backbone_state(payload)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"adapted PTv3 state mismatch missing={list(missing)[:8]} "
                f"unexpected={list(unexpected)[:8]}"
            )
    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    loaded_hash = encoder_state_sha256(encoder)
    checkpoint_meta = {}
    if arm == "delimit3d":
        checkpoint_meta = {
            "checkpoint_update": payload.get("update") if payload is not None else None,
            "checkpoint_encoder_state_sha256": payload.get("encoder_state_sha256") if payload is not None else None,
        }
        expected_encoder = checkpoint_meta.get("checkpoint_encoder_state_sha256")
        if expected_encoder and str(expected_encoder) != loaded_hash:
            raise RuntimeError(
                f"loaded adapted encoder hash mismatch: checkpoint says "
                f"{expected_encoder}, loaded state is {loaded_hash}"
            )
    return encoder, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": observed,
        "encoder_state_sha256": loaded_hash,
        "checkpoint_meta": checkpoint_meta,
    }


def center_shift(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    shift = np.array(
        [(minimum[0] + maximum[0]) / 2.0, (minimum[1] + maximum[1]) / 2.0, minimum[2]],
        dtype=np.float32,
    )
    return (points - shift).astype(np.float32, copy=False)


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path}: points must have shape [N,3]")
    if colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError(f"{path}: points/colors/normals are not aligned")
    normals = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-8
    )
    return points, colors, normals


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    sorted_target = np.asarray(target, dtype=np.float64)[order]
    positives = float(sorted_target.sum())
    if positives <= 0:
        return float("nan")
    hits = np.cumsum(sorted_target)
    ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    return float(np.sum((sorted_target * hits) / (positives * ranks)))


def topk_size(n: int, fraction: float) -> int:
    return max(1, min(n, int(math.ceil(float(n) * float(fraction)))))


def query_metrics(
    *,
    unit: np.ndarray,
    points: np.ndarray,
    point_instance: np.ndarray,
    point_semantic: np.ndarray,
    target_mask: np.ndarray,
    same_class_mask: np.ndarray,
    different_class_mask: np.ndarray,
    target_instance: int,
    target_semantic: int,
    query_index: int,
    seed: int,
) -> dict[str, float | int]:
    n = int(len(unit))
    if not 0 <= int(query_index) < n:
        raise ValueError("query index outside the scene")
    candidate = np.ones(n, dtype=bool)
    candidate[int(query_index)] = False
    score = unit @ unit[int(query_index)]
    # Use each manifest object's raw mesh mask as the retrieval target.  The
    # point_instance array is a deterministic first-claim label map used for
    # kNN purity; on overlapping masks it can erase a later object's points,
    # which must not make that valid query appear to have zero positives.
    target = np.asarray(target_mask, dtype=bool) & candidate
    same_class_distractor = (
        np.asarray(same_class_mask, dtype=bool) & ~target & candidate
    )
    different_class = (
        np.asarray(different_class_mask, dtype=bool) & ~target & candidate
    )
    candidate_indices = np.flatnonzero(candidate)
    candidate_scores = score[candidate]
    target_candidate = target[candidate]
    same_class_candidate = same_class_distractor[candidate]
    order = np.argsort(-candidate_scores, kind="stable")
    ranked_scores = candidate_scores[order]
    ranked_target = target_candidate[order]
    ranked_same_class = same_class_candidate[order]
    target_count = int(ranked_target.sum())
    if target_count <= 0:
        raise ValueError("query has no target points")
    result: dict[str, float | int] = {
        "query": int(query_index),
        "instance": int(target_instance),
        "semantic_class": int(target_semantic),
        "target_points": target_count,
        "candidate_points": int(len(candidate_indices)),
        "ap": average_precision(target_candidate, candidate_scores),
    }
    for fraction in QUERY_TOP_FRACTIONS:
        key = f"{int(fraction * 1000):04d}bp"
        k = topk_size(len(ranked_target), fraction)
        top_target = ranked_target[:k]
        tp = int(top_target.sum())
        result[f"recall_at_{key}"] = float(tp / target_count)
        result[f"iou_at_{key}"] = float(tp / max(target_count + k - tp, 1))
        result[f"same_class_fp_at_{key}"] = float(
            ranked_same_class[:k].sum() / max(k, 1)
        )
    rng = np.random.default_rng(int(seed))
    positive = np.flatnonzero(target)
    same_class = np.flatnonzero(same_class_distractor)
    diff_class = np.flatnonzero(different_class)
    def sample(values: np.ndarray, limit: int = 256) -> np.ndarray:
        if len(values) <= limit:
            return values
        return rng.choice(values, size=limit, replace=False)
    pos_scores = score[sample(positive)]
    same_scores = score[sample(same_class)]
    diff_scores = score[sample(diff_class)]
    result["positive_similarity"] = float(pos_scores.mean())
    result["same_class_similarity"] = (
        float(same_scores.mean()) if len(same_scores) else float("nan")
    )
    result["different_class_similarity"] = (
        float(diff_scores.mean()) if len(diff_scores) else float("nan")
    )
    result["same_class_margin"] = (
        float(pos_scores.mean() - same_scores.mean()) if len(same_scores) else float("nan")
    )
    result["different_class_margin"] = (
        float(pos_scores.mean() - diff_scores.mean()) if len(diff_scores) else float("nan")
    )
    if len(same_scores):
        result["same_class_hard_margin"] = float(
            pos_scores.mean() - np.quantile(same_scores, 0.90)
        )
    else:
        result["same_class_hard_margin"] = float("nan")
    return result


def build_point_labels(
    record: Mapping[str, Any], arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_arrays: Mapping[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    points, _, _ = arrays
    n = len(points)
    point_instance = np.full(n, -1, dtype=np.int32)
    point_semantic = np.full(n, -1, dtype=np.int32)
    masks: dict[int, np.ndarray] = {}
    semantics: dict[int, int] = {}
    data_path = resolve_path(str(record["_data_path"]), strict=True)
    for obj in record.get("objects", []):
        instance = int(obj["instance"])
        key = f"target_{instance}"
        if target_arrays is not None:
            if instance not in target_arrays:
                raise KeyError(f"{data_path}: missing {key}")
            indices = np.asarray(target_arrays[instance], dtype=np.int64)
        else:
            with np.load(data_path, allow_pickle=False) as data:
                if key not in data.files:
                    raise KeyError(f"{data_path}: missing {key}")
                indices = np.asarray(data[key], dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < n)]
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        masks[instance] = mask
        semantic = int(obj.get("semantic_class", -1))
        semantics[instance] = semantic
        # The frozen bundle is expected to contain disjoint instances.  If
        # it does not, retain the first manifest object deterministically.
        free = mask & (point_instance < 0)
        point_instance[free] = instance
        point_semantic[free] = semantic
    return point_instance, point_semantic, masks, semantics


def geometry_metrics(unit: np.ndarray, seed: int, max_points: int) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    m = min(int(max_points), len(unit))
    indices = rng.choice(len(unit), size=m, replace=False) if m < len(unit) else np.arange(len(unit))
    x = unit[indices].astype(np.float64, copy=False)
    norms = np.linalg.norm(x, axis=1)
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = (centered.T @ centered) / max(len(x) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    effective_rank = float(total * total / max(float((eigenvalues * eigenvalues).sum()), 1e-12))
    pc1_fraction = float(eigenvalues[-1] / max(total, 1e-12))
    pair_mean = float(
        (np.sum(x, axis=0) @ np.sum(x, axis=0) - len(x))
        / max(len(x) * max(len(x) - 1, 1), 1)
    )
    return {
        "geometry_sample_points": int(len(x)),
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_std": float(norms.std()),
        "mean_pairwise_cosine": pair_mean,
        "effective_rank": effective_rank,
        "pc1_variance_fraction": pc1_fraction,
    }


def knn_metrics(
    unit: np.ndarray,
    point_instance: np.ndarray,
    point_semantic: np.ndarray,
    query_rows: Sequence[Mapping[str, Any]],
    seed: int,
    max_candidates: int,
) -> dict[str, float]:
    query_indices = np.asarray([int(row["query"]) for row in query_rows], dtype=np.int64)
    query_indices = np.unique(query_indices)
    rng = np.random.default_rng(int(seed))
    candidate_indices = (
        rng.choice(len(unit), size=max_candidates, replace=False)
        if len(unit) > max_candidates else np.arange(len(unit))
    )
    candidate_features = unit[candidate_indices]
    values = []
    semantic_values = []
    distractor_values = []
    for query in query_indices:
        query_scores = candidate_features @ unit[int(query)]
        local_query = np.flatnonzero(candidate_indices == query)
        if len(local_query):
            query_scores[local_query[0]] = -np.inf
        k = min(KNN_K, len(query_scores))
        order = np.argpartition(-query_scores, kth=max(k - 1, 0))[:k]
        order = order[np.argsort(-query_scores[order], kind="stable")]
        neighbors = candidate_indices[order]
        same_instance = point_instance[neighbors] == point_instance[int(query)]
        same_class = (
            point_semantic[neighbors] >= 0
            ) & (point_semantic[neighbors] == point_semantic[int(query)])
        valid_query = point_instance[int(query)] >= 0
        if not valid_query:
            continue
        values.append(float(same_instance.mean()))
        semantic_values.append(float(same_class.mean()))
        distractor_values.append(float((same_class & ~same_instance).mean()))
    return {
        "knn_query_count": int(len(values)),
        "knn_instance_purity_16": float(np.mean(values)) if values else float("nan"),
        "knn_semantic_purity_16": float(np.mean(semantic_values)) if values else float("nan"),
        "knn_same_class_distractor_16": float(np.mean(distractor_values)) if values else float("nan"),
    }


def evaluate_scene(
    encoder: PTv3Encoder,
    arm: str,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    feature_seed: int,
    geometry_sample_points: int,
    knn_candidates: int,
    save_sample_dir: Path | None,
) -> dict[str, Any]:
    data_path = resolve_path(str(record["_data_path"]), strict=True)
    with np.load(data_path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        target_arrays = {
            int(obj["instance"]): np.asarray(data[f"target_{int(obj['instance'])}"], dtype=np.int64)
            for obj in record.get("objects", [])
        }
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError(f"{data_path}: points/colors/normals are not aligned")
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    shifted = center_shift(points)
    inputs = build_input_features(
        points, colors, use_colors=True, use_normals=True, normals=normals
    ).astype(np.float32, copy=False)
    with torch.inference_mode():
        output = encoder(
            points=torch.from_numpy(shifted).to(device),
            features=torch.from_numpy(inputs).to(device),
            seed=int(feature_seed),
        )
    native = output.point_features.detach().float().cpu().numpy()
    if native.shape != (len(points), FEATURE_DIM) or not np.isfinite(native).all():
        raise RuntimeError(f"{arm}/{record['scene']}: invalid feature tensor {native.shape}")
    norms = np.linalg.norm(native, axis=1, keepdims=True)
    unit = native / np.maximum(norms, 1e-12)
    point_instance, point_semantic, masks, semantics = build_point_labels(
        record, (points, colors, normals), target_arrays=target_arrays
    )
    query_rows: list[dict[str, Any]] = []
    for obj in record.get("objects", []):
        instance = int(obj["instance"])
        if instance not in masks:
            continue
        semantic = int(semantics[instance])
        target_mask = masks[instance]
        same_class_mask = np.zeros_like(target_mask, dtype=bool)
        different_class_mask = np.zeros_like(target_mask, dtype=bool)
        for other_instance, other_mask in masks.items():
            if other_instance == instance:
                continue
            if int(semantics[other_instance]) == semantic:
                same_class_mask |= other_mask
            else:
                different_class_mask |= other_mask
        for query in obj.get("queries", []):
            query = int(query)
            if not masks[instance][query]:
                continue
            query_rows.append(query_metrics(
                unit=unit, points=shifted, point_instance=point_instance,
                point_semantic=point_semantic, target_mask=target_mask,
                same_class_mask=same_class_mask,
                different_class_mask=different_class_mask,
                target_instance=instance, target_semantic=semantic,
                query_index=query,
                seed=stable_seed(feature_seed, record["scene"], instance, query),
            ))
    scene_query = {}
    for key in (
        "ap", "recall_at_0001bp", "recall_at_0010bp", "recall_at_0050bp",
        "recall_at_0100bp", "iou_at_0001bp", "iou_at_0010bp", "iou_at_0050bp",
        "iou_at_0100bp", "same_class_fp_at_0001bp", "same_class_fp_at_0010bp",
        "same_class_fp_at_0050bp", "same_class_fp_at_0100bp",
        "positive_similarity", "same_class_similarity", "different_class_similarity",
        "same_class_margin", "same_class_hard_margin", "different_class_margin",
    ):
        values = np.asarray([float(row[key]) for row in query_rows], dtype=np.float64)
        scene_query[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    geometry = geometry_metrics(unit, stable_seed(feature_seed, record["scene"], "geometry"), geometry_sample_points)
    knn = knn_metrics(
        unit, point_instance, point_semantic, query_rows,
        stable_seed(feature_seed, record["scene"], "knn"), knn_candidates,
    )
    if save_sample_dir is not None:
        save_sample_dir.mkdir(parents=True, exist_ok=True)
        sample_size = min(20000, len(points))
        rng = np.random.default_rng(stable_seed(feature_seed, record["scene"], "saved"))
        indices = rng.choice(len(points), size=sample_size, replace=False)
        np.savez_compressed(
            save_sample_dir / f"{record['scene']}.npz",
            points=shifted[indices], features=native[indices],
            instance=point_instance[indices], semantic=point_semantic[indices],
            indices=indices,
        )
    return {
        "scene": str(record["scene"]),
        "arm": arm,
        "points": int(len(points)),
        "feature_dim": int(native.shape[1]),
        "query_count": int(len(query_rows)),
        "query_metrics": query_rows,
        "scene_query_metrics": scene_query,
        "geometry_metrics": geometry,
        "knn_metrics": knn,
        "data_path": str(data_path),
        "data_sha256": str(record.get("data_sha256") or sha256_file(data_path)),
    }


def records_with_paths(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    bundle, manifest = load_scene_manifest(config)
    records = []
    for source in manifest["scenes"]:
        record = dict(source)
        record["_data_path"] = str(scene_data_path(bundle, source))
        records.append(record)
    return records


def evaluate(config: Mapping[str, Any], arm: str, device_name: str) -> dict[str, Any]:
    if arm not in {"public", "delimit3d"}:
        raise ValueError("evaluate requires --arm public or delimit3d")
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    records = records_with_paths(config)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    encoder, checkpoint_meta = load_encoder(config, arm, device)
    initial_hash = encoder_state_sha256(encoder)
    output_path = root / f"{arm}_scenes.jsonl"
    sample_dir = root / "feature_samples" / arm if bool(
        config.get("metrics", {}).get("save_feature_samples", False)
    ) else None
    feature_seed = int(config.get("seed", 20260911))
    geometry_sample_points = int(config.get("metrics", {}).get("geometry_sample_points", 20000))
    knn_candidates = int(config.get("metrics", {}).get("knn_candidates", 8192))
    completed = 0
    with output_path.open("w") as stream:
        for index, source in enumerate(records):
            result = evaluate_scene(
                encoder, arm, source, device=device,
                feature_seed=feature_seed, geometry_sample_points=geometry_sample_points,
                knn_candidates=knn_candidates, save_sample_dir=sample_dir,
            )
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            completed += 1
            print(json.dumps({
                "mode": "evaluate", "arm": arm, "scene": source["scene"],
                "completed": completed, "total": len(records),
                "queries": result["query_count"],
            }), flush=True)
            del result
            if device.type == "cuda":
                torch.cuda.empty_cache()
    final_hash = encoder_state_sha256(encoder)
    if final_hash != initial_hash:
        raise RuntimeError(f"{arm}: encoder state changed during feature evaluation")
    report = {
        "schema": SCHEMA,
        "arm": arm,
        "checkpoint": checkpoint_meta,
        "scene_count_expected": len(records),
        "scene_count_completed": completed,
        "encoder_state_sha256_before": initial_hash,
        "encoder_state_sha256_after": final_hash,
        "encoder_unchanged": final_hash == initial_hash,
        "scene_records": str(output_path),
        "passed": bool(completed == len(records) and final_hash == initial_hash),
    }
    write_json(root / f"{arm}_feature_report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"{arm}: feature evaluation failed")
    return report


def read_scene_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scene_metric(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = []
    for record in records:
        source = record["scene_query_metrics"]
        if key in source:
            values.append(float(source[key]))
        elif key in record.get("geometry_metrics", {}):
            values.append(float(record["geometry_metrics"][key]))
        else:
            values.append(float(record.get("knn_metrics", {}).get(key, "nan")))
    arr = np.asarray(values, dtype=np.float64)
    return {str(record["scene"]): float(value) for record, value in zip(records, arr)}


def bootstrap_delta(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    public_path = root / "public_scenes.jsonl"
    adapted_path = root / "delimit3d_scenes.jsonl"
    if not public_path.exists() or not adapted_path.exists():
        raise FileNotFoundError("both public_scenes.jsonl and delimit3d_scenes.jsonl are required")
    public = read_scene_records(public_path)
    adapted = read_scene_records(adapted_path)
    public_by_scene = {str(row["scene"]): row for row in public}
    adapted_by_scene = {str(row["scene"]): row for row in adapted}
    scenes = sorted(set(public_by_scene) & set(adapted_by_scene))
    if not scenes:
        raise RuntimeError("no paired scenes")
    metric_keys = [
        "ap", "recall_at_0010bp", "recall_at_0050bp", "iou_at_0010bp",
        "same_class_fp_at_0010bp", "same_class_margin", "same_class_hard_margin",
        "different_class_margin", "knn_instance_purity_16",
        "knn_semantic_purity_16", "knn_same_class_distractor_16",
        "effective_rank", "pc1_variance_fraction", "mean_pairwise_cosine",
    ]
    bootstrap_samples = int(config.get("bootstrap", {}).get("samples", 10000))
    base_seed = int(config.get("bootstrap", {}).get("seed", 20260911))
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        public_values = scene_metric([public_by_scene[s] for s in scenes], key)
        adapted_values = scene_metric([adapted_by_scene[s] for s in scenes], key)
        deltas = np.asarray([
            adapted_values[s] - public_values[s]
            for s in scenes
            if np.isfinite(adapted_values[s]) and np.isfinite(public_values[s])
        ], dtype=np.float64)
        mean, low, high = bootstrap_delta(deltas, bootstrap_samples, stable_seed(base_seed, key))
        metrics[key] = {
            "scene_count": int(len(deltas)),
            "public_mean": float(np.nanmean(list(public_values.values()))),
            "delimit3d_mean": float(np.nanmean(list(adapted_values.values()))),
            "delta_delimit3d_minus_public": float(np.nanmean(deltas)) if len(deltas) else float("nan"),
            "paired_bootstrap_mean": mean,
            "paired_bootstrap_ci95_low": low,
            "paired_bootstrap_ci95_high": high,
        }
    gates = config.get("gate", {})
    adapted_update = None
    adapted_report_path = root / "delimit3d_feature_report.json"
    if adapted_report_path.exists():
        try:
            adapted_report = json.loads(adapted_report_path.read_text())
            adapted_update = adapted_report.get("checkpoint", {}).get("checkpoint_meta", {}).get("checkpoint_update")
        except Exception:
            adapted_update = None
    comparison_label = (
        f"public PTv3 versus {int(adapted_update)}-update Delimit3D PTv3"
        if adapted_update is not None else
        "public PTv3 versus Delimit3D PTv3"
    )
    ap = metrics["ap"]
    margin = metrics["same_class_margin"]
    fp = metrics["same_class_fp_at_0010bp"]
    semantic = metrics["knn_semantic_purity_16"]
    gate_values = {
        "ap_delta_threshold": float(gates.get("min_ap_delta", 0.02)),
        "same_class_margin_delta_threshold": float(gates.get("min_same_class_margin_delta", 0.02)),
        "same_class_fp_max_increase": float(gates.get("max_same_class_fp_increase", 0.02)),
        "semantic_purity_max_drop": float(gates.get("max_semantic_purity_drop", 0.05)),
    }
    ap_pass = (
        ap["delta_delimit3d_minus_public"] >= gate_values["ap_delta_threshold"]
        and ap["paired_bootstrap_ci95_low"] > 0
    )
    margin_pass = (
        margin["delta_delimit3d_minus_public"] >= gate_values["same_class_margin_delta_threshold"]
        and margin["paired_bootstrap_ci95_low"] > 0
    )
    fp_pass = fp["delta_delimit3d_minus_public"] <= gate_values["same_class_fp_max_increase"]
    semantic_pass = semantic["delta_delimit3d_minus_public"] >= -gate_values["semantic_purity_max_drop"]
    green = bool(ap_pass and margin_pass and fp_pass and semantic_pass)
    supportive = int(ap_pass) + int(margin_pass) + int(fp_pass) + int(semantic_pass)
    decision = "green" if green else ("amber" if supportive >= 2 else "red")
    report = {
        "schema": SCHEMA,
        "comparison": comparison_label,
        "adapted_checkpoint_update": int(adapted_update) if adapted_update is not None else None,
        "feature_level_only": True,
        "scenes": scenes,
        "metrics": metrics,
        "gate_values": gate_values,
        "gate_checks": {
            "retrieval_ap": bool(ap_pass),
            "same_class_hard_negative_margin": bool(margin_pass),
            "same_class_false_positive_guard": bool(fp_pass),
            "semantic_neighbor_retention_guard": bool(semantic_pass),
        },
        "decision": decision,
        "commit_to_full_agile3d_decoder": bool(green),
        "interpretation": (
            "Feature geometry and frozen retrieval support proceeding to the "
            "full decoder experiment."
            if green else
            "Do not treat feature spread alone as sufficient; inspect the "
            "paired metrics before committing to full decoder training."
        ),
    }
    write_json(root / "aggregate_feature_gate.json", report)
    return report


def checkpoint_ready(config: Mapping[str, Any]) -> tuple[bool, str]:
    path = resolve_path(config["arms"]["delimit3d"]["checkpoint"], strict=False)
    if not path.exists():
        return False, f"waiting for {path}"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"checkpoint not readable yet: {exc}"
    expected_updates = int(config.get("training", {}).get("updates", 256))
    if int(payload.get("update", -1)) < expected_updates:
        return False, f"checkpoint update={payload.get('update')} < {expected_updates}"
    report_path = path.parent / "train_report.json"
    if not report_path.exists():
        return False, "final checkpoint exists; waiting for train_report.json"
    report = json.loads(report_path.read_text())
    if int(report.get("updates", -1)) < expected_updates:
        return False, "train_report does not confirm the final update"
    return True, f"final checkpoint update={payload.get('update')} ready"


def wait_and_run(config: Mapping[str, Any], poll_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    start = time.time()
    while True:
        ready, message = checkpoint_ready(config)
        print(json.dumps({"mode": "wait-and-run", "ready": ready, "message": message}), flush=True)
        if ready:
            break
        if timeout_seconds and time.time() - start >= timeout_seconds:
            raise TimeoutError(message)
        time.sleep(max(5, int(poll_seconds)))
    prepare(config)
    script = str(Path(__file__).resolve())
    processes = []
    logs = []
    for arm, visible_gpu in (("public", "0"), ("delimit3d", "1")):
        log = output_root(config) / f"{arm}_feature_eval.log"
        handle = log.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        processes.append(subprocess.Popen(
            [sys.executable, script, "--config", str(Path(sys.argv[sys.argv.index("--config") + 1]).resolve()),
             "--mode", "evaluate", "--arm", arm, "--device", "cuda"],
            cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
        ))
        logs.append(str(log))
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"feature evaluation failed with exit codes {codes}; logs={logs}")
    return aggregate(config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "prepare":
        print(json.dumps(prepare(config), indent=2, sort_keys=True))
    elif args.mode == "evaluate":
        report = evaluate(config, str(args.arm), args.device)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.mode == "aggregate":
        print(json.dumps(aggregate(config), indent=2, sort_keys=True))
    else:
        poll = args.poll_seconds
        if poll is None:
            poll = int(config.get("wait", {}).get("poll_seconds", 60))
        result = wait_and_run(config, poll, int(args.timeout_seconds))
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
