#!/usr/bin/env python3
"""Measure spatial selectivity of the matched ScanNet40 LitePT features.

This diagnostic deliberately consumes the immutable native dec0 feature caches
produced by ``run_agile3d_multio.py``.  It never runs an encoder or a decoder:
the public and Delimit3D arms share the cached tokenization, inverse map, and
representative indices.  Metrics are computed per scene/object and aggregated
with room-entry means plus a paired bootstrap clustered by base scene ID.

The intended claim is narrow: after the 256-update adaptation, a click token
should stay close to the selected instance while becoming easier to separate
from a different instance with the same semantic name.  Same-name instances
are retained as official annotation records; for ``wall`` we report the pair
counts and do not interpret every annotation split as a physically distinct
wall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


CACHE_SCHEMA = "delimit3d_agile3d_feature_cache/v2"
SCHEMA = "delimit3d_scannetval_feature_selectivity/v1"
DEFAULT_ROOT = Path("/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1")
DEFAULT_OUTPUT = Path(
    "/cluster/work/igp_psr/nedela/"
    "delimit3d_feature_selectivity_scannetval_20260915"
)
ARMS = ("public", "delimit3d")
ARM_LABEL = {
    "public": "LitePT (public)",
    "delimit3d": "LitePT + Delimit3D (256 updates)",
}
EPS = 1e-8
MIN_TARGET_TOKENS = 32


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(block)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", action="append", help="one scene; repeatable")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--query-samples", type=int, default=8)
    parser.add_argument("--region-samples", type=int, default=48)
    parser.add_argument("--coherence-pairs", type=int, default=256)
    parser.add_argument("--retrieval-queries", type=int, default=2)
    parser.add_argument(
        "--visuals",
        action="store_true",
        help="write pair visual(s); requires matplotlib in the cluster environment",
    )
    parser.add_argument(
        "--visual-pairs", type=int, default=3, help="number of pair panels"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=10000, help="paired scene bootstrap resamples"
    )
    return parser.parse_args()


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def load_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "selection_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "delimit3d_scannet40_matched/v1":
        raise ValueError(f"unexpected selection manifest schema: {manifest.get('schema')}")
    return manifest, sha256_file(path)


def load_cache(root: Path, arm: str, scene: str) -> dict[str, Any]:
    path = root / "feature_cache" / arm / f"{scene}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"{path}: expected {CACHE_SCHEMA}, got {payload.get('schema')}")
    return payload


def scene_seed(seed: int, scene: str, *parts: Any) -> int:
    text = ":".join([str(seed), scene, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def base_scene_id(scene: str) -> str:
    """Return the ScanNet base scene ID, grouping scan suffixes together."""
    return str(scene).split("_", 1)[0]


def sample_indices(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= count:
        return values.copy()
    return np.asarray(rng.choice(values, size=count, replace=False), dtype=np.int64)


def unit(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), EPS)


def mean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else float("nan")


def percentile_or_nan(values: Sequence[float] | np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else float("nan")


def average_precision(binary: np.ndarray, scores: np.ndarray) -> float:
    """AP for a binary ranked list, retaining all candidates in the cache."""
    binary = np.asarray(binary, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    if binary.size == 0 or not np.any(binary):
        return float("nan")
    # Stable tie-breaking by candidate index makes this deterministic.
    order = np.argsort(-scores, kind="mergesort")
    hits = binary[order].astype(np.float64)
    cumulative = np.cumsum(hits)
    positions = np.arange(1, len(hits) + 1, dtype=np.float64)
    return float(np.sum((cumulative / positions) * hits) / np.sum(hits))


def rank_metrics(scores: np.ndarray, target: np.ndarray, *, fixed_k: Sequence[int]) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target, dtype=bool)
    order = np.argsort(-scores, kind="mergesort")
    ranked = target[order]
    n_pos = int(np.sum(target))
    output: dict[str, Any] = {
        "ap": average_precision(target.astype(np.int8), scores),
        "positive_count": n_pos,
        "candidate_count": int(len(target)),
    }
    for k in fixed_k:
        k = int(min(max(1, k), len(ranked)))
        output[f"precision_at_{k}"] = float(np.mean(ranked[:k]))
        output[f"leakage_at_{k}"] = float(1.0 - np.mean(ranked[:k]))
        output[f"recall_at_{k}"] = float(np.sum(ranked[:k]) / max(1, n_pos))
    size_k = int(min(max(1, n_pos), len(ranked)))
    output["target_size_k"] = size_k
    output["precision_at_target_size"] = float(np.mean(ranked[:size_k]))
    output["leakage_at_target_size"] = float(1.0 - np.mean(ranked[:size_k]))
    output["recall_at_target_size"] = float(np.sum(ranked[:size_k]) / max(1, n_pos))
    return output


def bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap adapted-public deltas clustered by ScanNet base scene.

    The validation manifest has multiple ``sceneXXXX_YY`` scan entries for
    some ``sceneXXXX`` base scenes. Room-level deltas are averaged within a
    base scene before the paired bootstrap so repeated scans do not count as
    independent clusters.
    """
    per_base_scene: dict[str, list[float]] = defaultdict(list)
    valid_room_count = 0
    for row in rows:
        public = row.get("public", {}).get(value_key)
        adapted = row.get("delimit3d", {}).get(value_key)
        if public is None or adapted is None:
            continue
        if not np.isfinite(public) or not np.isfinite(adapted):
            continue
        valid_room_count += 1
        scene = str(row["scene"])
        cluster = str(row.get("base_scene") or base_scene_id(scene))
        per_base_scene[cluster].append(float(adapted) - float(public))
    cluster_values = np.asarray(
        [mean_or_nan(v) for _cluster, v in sorted(per_base_scene.items())], dtype=np.float64
    )
    cluster_values = cluster_values[np.isfinite(cluster_values)]
    if not len(cluster_values):
        return {
            "bootstrap_unit": "base_scene",
            "base_scene_count": 0,
            "room_count": int(valid_room_count),
            "mean_delta": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "base_scene_deltas": {},
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster_values), size=(int(n_boot), len(cluster_values)))
    samples = np.mean(cluster_values[indices], axis=1)
    return {
        "bootstrap_unit": "base_scene",
        "base_scene_count": int(len(cluster_values)),
        "room_count": int(valid_room_count),
        "mean_delta": float(np.mean(cluster_values)),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "base_scene_deltas": {
            cluster: float(np.mean(values))
            for cluster, values in sorted(per_base_scene.items())
            if np.all(np.isfinite(values))
        },
    }


def _normal_array(record: Mapping[str, Any], cache: Mapping[str, Any]) -> np.ndarray | None:
    path = Path(str(record.get("normal_path", "")))
    if not path.exists():
        return None
    try:
        normals = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except Exception:
        return None
    reps = _as_numpy(cache["representative_indices"], np.int64)
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) <= int(reps.max(initial=-1)):
        return None
    normals = normals[reps]
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), EPS)


def _official_instance_labels(record: Mapping[str, Any], cache: Mapping[str, Any]) -> np.ndarray:
    """Read the official PLY instance labels at the cached representatives.

    The matched cache was created from this same official PLY ordering.  We
    still re-read the labels here and compare the resulting token memberships
    with ``object_token_indices`` so a stale/misinterpreted cache cannot silently
    turn into an instance-selectivity result.
    """
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("official PLY validation requires the plyfile package") from exc
    path = Path(str(record["data_path"]))
    ply = PlyData.read(path)["vertex"].data
    labels = np.asarray(ply["label"], dtype=np.int64).reshape(-1)
    reps = _as_numpy(cache["representative_indices"], np.int64)
    inverse = _as_numpy(cache["inverse_map"], np.int64)
    if len(labels) != len(inverse):
        raise ValueError(
            f"{record['scene']}: official PLY labels {len(labels)} != inverse map {len(inverse)}"
        )
    if np.any(reps < 0) or np.any(reps >= len(labels)):
        raise ValueError(f"{record['scene']}: representative index outside official PLY labels")
    return labels[reps]


def _geometry_pairs(
    query_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    query_normals: np.ndarray | None,
    candidate_normals: np.ndarray | None,
    values: np.ndarray,
    *,
    relation: str,
    bins: dict[str, list[float]],
    stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    if not len(query_xyz) or not len(candidate_xyz):
        return
    delta = query_xyz[:, None, :] - candidate_xyz[None, :, :]
    distances = np.linalg.norm(delta, axis=2).reshape(-1)
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if query_normals is not None and candidate_normals is not None:
        normal = np.abs(
            np.sum(query_normals[:, None, :] * candidate_normals[None, :, :], axis=2)
        ).reshape(-1)
    else:
        normal = np.full_like(distances, np.nan)
    for d_name, d_lo, d_hi in zip(
        bins["distance_names"], bins["distance"][:-1], bins["distance"][1:]
    ):
        dm = (distances >= d_lo) & (distances < d_hi)
        if not np.any(dm):
            continue
        for n_name, n_lo, n_hi in zip(
            bins["normal_names"], bins["normal"][:-1], bins["normal"][1:]
        ):
            nm = np.ones_like(dm, dtype=bool) if not np.isfinite(normal).any() else (
                (normal >= n_lo) & (normal < n_hi)
            )
            mask = dm & nm & np.isfinite(flat_values)
            if not np.any(mask):
                continue
            key = f"distance_{d_name}__normal_{n_name}"
            cell = stats.setdefault(relation, {}).setdefault(
                key, {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}
            )
            selected = flat_values[mask].astype(np.float64)
            cell["count"] += float(len(selected))
            cell["sum"] += float(np.sum(selected))
            cell["sum_sq"] += float(np.sum(selected * selected))


def _matched_geometry_margin(
    query_xyz: np.ndarray,
    positive_xyz: np.ndarray,
    negative_xyz: np.ndarray,
    query_normals: np.ndarray | None,
    positive_normals: np.ndarray | None,
    negative_normals: np.ndarray | None,
    positive_values: np.ndarray,
    negative_values: np.ndarray,
    *,
    bins: dict[str, list[float]],
    stats: dict[str, dict[str, float]],
) -> None:
    """Accumulate positive-minus-negative similarities in matched geometry cells.

    For each query and distance/normal cell, the positive and negative
    candidate pools are averaged separately, then one paired margin is added
    only when both pools are present.  This compares same-query candidates at
    the same coarse geometry rather than merely reporting two unbalanced pair
    distributions.
    """
    if not len(query_xyz) or not len(positive_xyz) or not len(negative_xyz):
        return
    pos_dist = np.linalg.norm(query_xyz[:, None, :] - positive_xyz[None, :, :], axis=2)
    neg_dist = np.linalg.norm(query_xyz[:, None, :] - negative_xyz[None, :, :], axis=2)
    if query_normals is not None and positive_normals is not None and negative_normals is not None:
        pos_norm = np.abs(np.sum(query_normals[:, None, :] * positive_normals[None, :, :], axis=2))
        neg_norm = np.abs(np.sum(query_normals[:, None, :] * negative_normals[None, :, :], axis=2))
    else:
        pos_norm = np.full_like(pos_dist, np.nan)
        neg_norm = np.full_like(neg_dist, np.nan)
    pos_values = np.asarray(positive_values, dtype=np.float32)
    neg_values = np.asarray(negative_values, dtype=np.float32)
    for q in range(len(query_xyz)):
        for d_name, d_lo, d_hi in zip(
            bins["distance_names"], bins["distance"][:-1], bins["distance"][1:]
        ):
            pos_d = (pos_dist[q] >= d_lo) & (pos_dist[q] < d_hi)
            neg_d = (neg_dist[q] >= d_lo) & (neg_dist[q] < d_hi)
            if not np.any(pos_d) or not np.any(neg_d):
                continue
            for n_name, n_lo, n_hi in zip(
                bins["normal_names"], bins["normal"][:-1], bins["normal"][1:]
            ):
                if np.isfinite(pos_norm).any() and np.isfinite(neg_norm).any():
                    pos_m = pos_d & (pos_norm[q] >= n_lo) & (pos_norm[q] < n_hi)
                    neg_m = neg_d & (neg_norm[q] >= n_lo) & (neg_norm[q] < n_hi)
                else:
                    pos_m, neg_m = pos_d, neg_d
                if not np.any(pos_m) or not np.any(neg_m):
                    continue
                pos_mean = float(np.mean(pos_values[q, pos_m]))
                neg_mean = float(np.mean(neg_values[q, neg_m]))
                margin = pos_mean - neg_mean
                key = f"distance_{d_name}__normal_{n_name}"
                cell = stats.setdefault(
                    key,
                    {"count": 0.0, "sum_margin": 0.0, "sum_positive": 0.0, "sum_negative": 0.0},
                )
                cell["count"] += 1.0
                cell["sum_margin"] += margin
                cell["sum_positive"] += pos_mean
                cell["sum_negative"] += neg_mean


def _pair_metric(
    scene: str,
    record: Mapping[str, Any],
    caches: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    query_samples: int,
    region_samples: int,
    coherence_pairs_count: int,
    retrieval_queries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    # The manifest is authoritative for object semantic names and ordering.  The
    # cache's token memberships are authoritative for the representative-token
    # mapping and are checked below for exact shared membership.
    objects = list(record["objects"])
    object_by_id = {int(item["instance"]): item for item in objects}
    memberships: dict[int, np.ndarray] = {}
    for object_id, item in object_by_id.items():
        values = _as_numpy(caches["public"]["object_token_indices"].get(str(object_id), []), np.int64)
        adapted_values = _as_numpy(
            caches["delimit3d"]["object_token_indices"].get(str(object_id), []), np.int64
        )
        if not np.array_equal(values, adapted_values):
            raise ValueError(f"{scene}: token membership drift for object {object_id}")
        memberships[object_id] = values

    xyz = _as_numpy(caches["public"]["scene_xyz"], np.float32)
    adapted_xyz = _as_numpy(caches["delimit3d"]["scene_xyz"], np.float32)
    inverse = _as_numpy(caches["public"]["inverse_map"], np.int64)
    adapted_inverse = _as_numpy(caches["delimit3d"]["inverse_map"], np.int64)
    reps = _as_numpy(caches["public"]["representative_indices"], np.int64)
    adapted_reps = _as_numpy(caches["delimit3d"]["representative_indices"], np.int64)
    if not np.array_equal(inverse, adapted_inverse) or not np.array_equal(reps, adapted_reps):
        raise ValueError(f"{scene}: shared inverse/representative map drift")
    if not np.allclose(xyz, adapted_xyz, rtol=0, atol=0):
        raise ValueError(f"{scene}: shared token coordinates drift")

    official_token_labels = _official_instance_labels(record, caches["public"])
    official_ids = {int(value) for value in np.unique(official_token_labels) if int(value) > 0}
    manifest_ids = set(object_by_id)
    if official_ids != manifest_ids:
        raise ValueError(
            f"{scene}: official PLY object IDs differ from selection manifest "
            f"(official={sorted(official_ids)}, manifest={sorted(manifest_ids)})"
        )
    mapping_mismatches: list[int] = []
    for object_id in sorted(manifest_ids):
        expected = np.flatnonzero(official_token_labels == int(object_id)).astype(np.int64)
        if not np.array_equal(expected, memberships[object_id]):
            mapping_mismatches.append(int(object_id))
    if mapping_mismatches:
        raise ValueError(f"{scene}: object_token_indices disagree with official PLY for {mapping_mismatches}")
    official_background_tokens = int(np.count_nonzero(official_token_labels <= 0))
    eligible_ids = {
        int(object_id)
        for object_id, values in memberships.items()
        if len(values) >= MIN_TARGET_TOKENS
    }

    feature_by_arm = {
        arm: unit(_as_numpy(caches[arm]["features"], np.float32)) for arm in ARMS
    }
    token_count = int(len(xyz))
    normals = _normal_array(record, caches["public"])
    geometry_bins = {
        "distance": [0.0, 0.25, 0.5, 1.0, 2.0, float("inf")],
        "distance_names": ["0_025", "025_050", "050_100", "100_200", "200_inf"],
        "normal": [0.0, 0.5, 0.8, 0.95, 1.00001],
        "normal_names": ["0_050", "050_080", "080_095", "095_1"],
    }

    # Keep the count separate from the per-object sampled pair array.  The
    # previous implementation allowed those names to alias, so after the
    # first object NumPy received an ndarray where it expected an integer
    # ``size`` and raised its heterogeneous-shape ValueError.
    coherence_pairs_count = int(coherence_pairs_count)
    if coherence_pairs_count < 0:
        raise ValueError(f"{scene}: coherence_pairs_count must be non-negative")

    object_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    geometry_stats: dict[str, dict[str, dict[str, float]]] = {}
    geometry_matched_stats: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    for object_id, item in object_by_id.items():
        target_tokens = memberships[object_id]
        if len(target_tokens) < MIN_TARGET_TOKENS:
            continue
        target_name = str(item.get("semantic_name", "unavailable"))
        rng = np.random.default_rng(scene_seed(seed, scene, object_id, "sampling"))
        query_tokens = sample_indices(target_tokens, query_samples, rng)
        region_tokens = sample_indices(target_tokens, region_samples, rng)
        region_tokens = np.unique(region_tokens)
        # Keep self-similarity out of the coherence estimate when possible.
        coherence_values: dict[str, list[float]] = {arm: [] for arm in ARMS}
        coherence_pair_indices = np.asarray(
            rng.integers(0, len(target_tokens), size=(coherence_pairs_count, 2)), dtype=np.int64
        )
        coherence_pair_indices = coherence_pair_indices[
            target_tokens[coherence_pair_indices[:, 0]]
            != target_tokens[coherence_pair_indices[:, 1]]
        ]
        for arm in ARMS:
            feats = feature_by_arm[arm]
            if len(coherence_pair_indices):
                coherence_values[arm] = np.sum(
                    feats[target_tokens[coherence_pair_indices[:, 0]]]
                    * feats[target_tokens[coherence_pair_indices[:, 1]]],
                    axis=1,
                ).astype(np.float64).tolist()

        same_class_ids = [
            other_id
            for other_id, other in object_by_id.items()
            if other_id != object_id
            and str(other.get("semantic_name", "unavailable")) == target_name
            and len(memberships[other_id]) >= MIN_TARGET_TOKENS
        ]
        pair_metrics_by_arm: dict[str, dict[str, Any]] = {
            arm: {
                "target_sim": [],
                "other_sim": [],
                "hard_other_sim": [],
                "margins": [],
                "other_by_id": {},
            }
            for arm in ARMS
        }
        distractor_sampling: dict[str, list[int]] = {}
        if same_class_ids:
            for other_id in same_class_ids:
                other_tokens = sample_indices(
                    memberships[other_id], region_samples, rng
                )
                if not len(other_tokens):
                    continue
                distractor_sampling[str(int(other_id))] = other_tokens.tolist()
                for arm in ARMS:
                    feats = feature_by_arm[arm]
                    q = feats[query_tokens]
                    pos = feats[region_tokens]
                    neg = feats[other_tokens]
                    pos_sim = (q @ pos.T).astype(np.float32)
                    neg_sim = (q @ neg.T).astype(np.float32)
                    # Mean similarity is the principal margin.  The 95th
                    # percentile of a distractor captures hard leakage.
                    pos_mean = np.mean(pos_sim, axis=1)
                    neg_mean = np.mean(neg_sim, axis=1)
                    hard = np.percentile(neg_sim, 95, axis=1)
                    pair_metrics_by_arm[arm].setdefault("target_by_query", pos_mean)
                    pair_metrics_by_arm[arm]["other_by_id"][int(other_id)] = neg_mean
                    pair_metrics_by_arm[arm].setdefault("hard_other_by_id", {})[
                        int(other_id)
                    ] = hard
                    pair_metrics_by_arm[arm]["target_sim"].extend(pos_mean.tolist())
                    pair_metrics_by_arm[arm]["other_sim"].extend(neg_mean.tolist())
                    pair_metrics_by_arm[arm]["hard_other_sim"].extend(hard.tolist())
                    pair_metrics_by_arm[arm]["margins"].extend((pos_mean - neg_mean).tolist())
                    # Keep per-target/distractor rows local to this pair.  The
                    # arm-level accumulators above are intentionally pooled for
                    # the object summary, while the pair row must not include
                    # margins from earlier same-class distractors.
                    pair_metrics_by_arm[arm].setdefault("_current_margin", {})[
                        int(other_id)
                    ] = mean_or_nan(pos_mean - neg_mean)
                    if normals is not None:
                        _geometry_pairs(
                            xyz[query_tokens], xyz[region_tokens],
                            normals[query_tokens], normals[region_tokens], pos_sim,
                            relation=f"{arm}:same_instance", bins=geometry_bins, stats=geometry_stats,
                        )
                        _geometry_pairs(
                            xyz[query_tokens], xyz[other_tokens],
                            normals[query_tokens], normals[other_tokens], neg_sim,
                            relation=f"{arm}:same_class_other", bins=geometry_bins, stats=geometry_stats,
                        )
                        _matched_geometry_margin(
                            xyz[query_tokens], xyz[region_tokens], xyz[other_tokens],
                            normals[query_tokens], normals[region_tokens], normals[other_tokens],
                            pos_sim, neg_sim, bins=geometry_bins,
                            stats=geometry_matched_stats[arm],
                        )
                pair_rows.append(
                    {
                        "scene": scene,
                        "base_scene": base_scene_id(scene),
                        "target_instance": object_id,
                        "target_name": target_name,
                        "distractor_instance": int(other_id),
                        "target_tokens": int(len(target_tokens)),
                        "distractor_tokens": int(len(memberships[other_id])),
                        "public_margin": float(
                            pair_metrics_by_arm["public"]["_current_margin"][int(other_id)]
                        ),
                        "delimit3d_margin": float(
                            pair_metrics_by_arm["delimit3d"]["_current_margin"][int(other_id)]
                        ),
                    }
                )

        retrieval_q = sample_indices(
            target_tokens, min(retrieval_queries, len(target_tokens)), rng
        )
        sampling_ids = {
            "query_tokens": query_tokens.tolist(),
            "region_tokens": region_tokens.tolist(),
            "coherence_pairs": coherence_pair_indices.tolist(),
            "retrieval_queries": retrieval_q.tolist(),
            "distractor_tokens": distractor_sampling,
        }
        sampling_id_sha256 = sha256_json(sampling_ids)
        arm_rows: dict[str, Any] = {}
        for arm in ARMS:
            feats = feature_by_arm[arm]
            target_mask = np.zeros(token_count, dtype=bool)
            target_mask[target_tokens] = True
            # Use the same deterministic query tokens for both arms.  Retrieval
            # is sampled at two queries/object to keep 312 scenes tractable.
            retrieval_rows: list[dict[str, Any]] = []
            for query in retrieval_q:
                scores = feats @ feats[int(query)]
                # The clicked/query token is not a candidate.  This avoids a
                # free true positive in AP and fixed-budget leakage. Remove it
                # from the arrays rather than leaving a false -inf candidate,
                # so candidate counts and fixed budgets are exact.
                candidate_mask = np.ones(token_count, dtype=bool)
                candidate_mask[int(query)] = False
                candidate_target = target_mask[candidate_mask]
                scores = scores[candidate_mask]
                retrieval_rows.append(
                    rank_metrics(scores, candidate_target, fixed_k=(128, 512, 2048))
                )
            pair_summary = pair_metrics_by_arm[arm]
            if pair_summary["other_by_id"]:
                other_matrix = np.column_stack(
                    [pair_summary["other_by_id"][key] for key in sorted(pair_summary["other_by_id"])]
                )
                hard_other_by_query = np.max(other_matrix, axis=1)
                hard_percentile_by_query = np.max(
                    np.column_stack(
                        [pair_summary["hard_other_by_id"][key] for key in sorted(pair_summary["hard_other_by_id"])]
                    ),
                    axis=1,
                )
                target_by_query = np.asarray(pair_summary["target_by_query"], dtype=np.float32)
                hard_margin_by_query = target_by_query - hard_other_by_query
            else:
                hard_other_by_query = np.asarray([], dtype=np.float32)
                hard_percentile_by_query = np.asarray([], dtype=np.float32)
                hard_margin_by_query = np.asarray([], dtype=np.float32)
            arm_rows[arm] = {
                "sampling_id_sha256": sampling_id_sha256,
                "coherence": mean_or_nan(coherence_values[arm]),
                "coherence_pair_count": int(len(coherence_pair_indices)),
                "target_token_count": int(len(target_tokens)),
                "retrieval_ap": mean_or_nan([r["ap"] for r in retrieval_rows]),
                "retrieval_precision_at_128": mean_or_nan(
                    [r.get("precision_at_128", float("nan")) for r in retrieval_rows]
                ),
                "retrieval_precision_at_512": mean_or_nan(
                    [r.get("precision_at_512", float("nan")) for r in retrieval_rows]
                ),
                "retrieval_precision_at_2048": mean_or_nan(
                    [r.get("precision_at_2048", float("nan")) for r in retrieval_rows]
                ),
                "retrieval_leakage_at_target_size": mean_or_nan(
                    [r.get("leakage_at_target_size", float("nan")) for r in retrieval_rows]
                ),
                "retrieval_recall_at_target_size": mean_or_nan(
                    [r.get("recall_at_target_size", float("nan")) for r in retrieval_rows]
                ),
                "same_class_pair_count": int(len(pair_summary["other_by_id"])),
                "same_class_query_count": int(len(hard_margin_by_query)),
                "same_class_target_sim": mean_or_nan(pair_summary.get("target_by_query", [])),
                "same_class_other_sim": mean_or_nan(hard_other_by_query),
                "same_class_hard_other_sim": mean_or_nan(hard_percentile_by_query),
                "same_class_margin": mean_or_nan(hard_margin_by_query),
                "same_class_margin_p50": percentile_or_nan(hard_margin_by_query, 50),
                "same_class_target_sim_p50": percentile_or_nan(pair_summary.get("target_by_query", []), 50),
                "same_class_other_sim_p50": percentile_or_nan(hard_other_by_query, 50),
            }
        object_rows.append(
            {
                "scene": scene,
                "base_scene": base_scene_id(scene),
                "instance": int(object_id),
                "semantic_name": target_name,
                "instance_points": int(item.get("instance_points", 0)),
                "token_count": int(len(target_tokens)),
                "same_class_other_instances": [int(v) for v in same_class_ids],
                "sampling_id_sha256": sampling_id_sha256,
                "arms": arm_rows,
            }
        )

    # Finalize geometry cells after all object pairs have been accumulated.
    geometry_summary: dict[str, dict[str, Any]] = {}
    for relation, cells in geometry_stats.items():
        geometry_summary[relation] = {}
        for key, value in cells.items():
            count = value["count"]
            mean = value["sum"] / count if count else float("nan")
            variance = max(0.0, value["sum_sq"] / count - mean * mean) if count else float("nan")
            geometry_summary[relation][key] = {
                "count": int(count),
                "mean_cosine": float(mean),
                "std_cosine": float(math.sqrt(variance)) if np.isfinite(variance) else float("nan"),
            }
    geometry_matched_summary: dict[str, dict[str, Any]] = {}
    for arm, cells in geometry_matched_stats.items():
        geometry_matched_summary[arm] = {}
        for key, value in cells.items():
            count = value["count"]
            geometry_matched_summary[arm][key] = {
                "count": int(count),
                "mean_margin": float(value["sum_margin"] / count) if count else float("nan"),
                "mean_positive_cosine": float(value["sum_positive"] / count) if count else float("nan"),
                "mean_negative_cosine": float(value["sum_negative"] / count) if count else float("nan"),
            }

    scene_arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        eligible = [
            row["arms"][arm]
            for row in object_rows
            if row["arms"][arm]["target_token_count"] >= MIN_TARGET_TOKENS
        ]
        scene_arm_summary[arm] = {
            "object_count": int(len(eligible)),
            "same_class_object_count": int(
                sum(int(row["arms"][arm]["same_class_pair_count"]) > 0 for row in object_rows)
            ),
            "token_count": token_count,
            "coherence": mean_or_nan([row["coherence"] for row in eligible]),
            "retrieval_ap": mean_or_nan([row["retrieval_ap"] for row in eligible]),
            "retrieval_leakage_at_target_size": mean_or_nan(
                [row["retrieval_leakage_at_target_size"] for row in eligible]
            ),
            "same_class_margin": mean_or_nan(
                [row["same_class_margin"] for row in eligible if np.isfinite(row["same_class_margin"])]
            ),
            "same_class_other_sim": mean_or_nan(
                [row["same_class_other_sim"] for row in eligible if np.isfinite(row["same_class_other_sim"])]
            ),
        }
    return (
        {
            "scene": scene,
            "base_scene": base_scene_id(scene),
            "token_count": token_count,
            "point_count": int(len(inverse)),
            "normal_tokens_available": bool(normals is not None),
            "official_background_tokens": official_background_tokens,
            "official_object_count": int(len(object_by_id)),
            "eligible_object_count": int(len(eligible_ids)),
            "excluded_below_min_tokens": int(len(object_by_id) - len(eligible_ids)),
            "object_count": int(len(object_rows)),
            "same_class_pair_count": int(len(pair_rows)),
            "arms": scene_arm_summary,
        },
        object_rows,
        {
            "pair_rows": pair_rows,
            "geometry": geometry_summary,
            "geometry_matched": geometry_matched_summary,
        },
    )


def _numeric_aggregate(
    scene_summaries: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    # Room means give every room entry equal descriptive weight. Object means
    # are retained as an additional view; raw token/point weighted means are
    # calculated from sums and counts. Inference below clusters room deltas by
    # base scene because repeated scan suffixes are not independent.
    base_scenes = {
        str(row.get("base_scene") or base_scene_id(str(row["scene"])))
        for row in scene_summaries
    }
    out: dict[str, Any] = {
        "scene_count": int(len(scene_summaries)),
        "base_scene_count": int(len(base_scenes)),
        "scene_balanced": {},
        "object_balanced": {},
        "pair_balanced": {},
    }
    metrics = (
        "coherence",
        "retrieval_ap",
        "retrieval_leakage_at_target_size",
        "same_class_margin",
        "same_class_other_sim",
    )
    for metric in metrics:
        out["scene_balanced"][metric] = {}
        for arm in ARMS:
            values = [
                row["arms"][arm].get(metric, float("nan"))
                for row in scene_summaries
                if np.isfinite(row["arms"][arm].get(metric, float("nan")))
            ]
            out["scene_balanced"][metric][arm] = mean_or_nan(values)
        out["scene_balanced"][metric]["delimit3d_minus_public"] = (
            out["scene_balanced"][metric]["delimit3d"]
            - out["scene_balanced"][metric]["public"]
        )
        out.setdefault("scene_bootstrap", {})[metric] = bootstrap_delta(
            [
                {
                    "scene": row["scene"],
                    "base_scene": row.get("base_scene") or base_scene_id(str(row["scene"])),
                    "public": {metric: row["arms"]["public"].get(metric, float("nan"))},
                    "delimit3d": {metric: row["arms"]["delimit3d"].get(metric, float("nan"))},
                }
                for row in scene_summaries
            ],
            metric,
            n_boot=bootstrap,
            seed=scene_seed(seed, metric, "base-scene-bootstrap"),
        )
        out["object_balanced"][metric] = {}
        for arm in ARMS:
            values = [
                row["arms"][arm].get(metric, float("nan"))
                for row in object_rows
                if np.isfinite(row["arms"][arm].get(metric, float("nan")))
            ]
            out["object_balanced"][metric][arm] = mean_or_nan(values)
        out["object_balanced"][metric]["delimit3d_minus_public"] = (
            out["object_balanced"][metric]["delimit3d"]
            - out["object_balanced"][metric]["public"]
        )

    # Keep the wall stratum visible without letting it determine the headline
    # result.  A ``wall`` row means two separately annotated official instance
    # IDs; it does not assert that every annotation split is a separate physical
    # wall.  The complementary non-wall stratum covers all other semantic names.
    strata_rows = {
        "wall": [row for row in object_rows if str(row.get("semantic_name", "")).lower() == "wall"],
        "non_wall": [row for row in object_rows if str(row.get("semantic_name", "")).lower() != "wall"],
    }
    out["semantic_strata"] = {}
    for name, rows in strata_rows.items():
        pair_subset = [row for row in pair_rows if (str(row.get("target_name", "")).lower() == "wall") == (name == "wall")]
        out["semantic_strata"][name] = {
            "object_count": int(len(rows)),
            "scene_count": int(len({str(row["scene"]) for row in rows})),
            "same_class_pair_count": int(len(pair_subset)),
            "object_balanced": {},
        }
        for metric in metrics:
            out["semantic_strata"][name]["object_balanced"][metric] = {}
            for arm in ARMS:
                values = [
                    row["arms"][arm].get(metric, float("nan"))
                    for row in rows
                    if np.isfinite(row["arms"][arm].get(metric, float("nan")))
                ]
                out["semantic_strata"][name]["object_balanced"][metric][arm] = mean_or_nan(values)
            out["semantic_strata"][name]["object_balanced"][metric]["delimit3d_minus_public"] = (
                out["semantic_strata"][name]["object_balanced"][metric]["delimit3d"]
                - out["semantic_strata"][name]["object_balanced"][metric]["public"]
            )
        if pair_subset:
            public = [float(row["public_margin"]) for row in pair_subset if np.isfinite(row["public_margin"])]
            adapted = [float(row["delimit3d_margin"]) for row in pair_subset if np.isfinite(row["delimit3d_margin"])]
            out["semantic_strata"][name]["pair_balanced_margin"] = {
                "public": mean_or_nan(public),
                "delimit3d": mean_or_nan(adapted),
                "delimit3d_minus_public": mean_or_nan(adapted) - mean_or_nan(public),
            }
        else:
            out["semantic_strata"][name]["pair_balanced_margin"] = {
                "public": float("nan"),
                "delimit3d": float("nan"),
                "delimit3d_minus_public": float("nan"),
            }

    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in object_rows]
        token_count = sum(
            int(row.get("target_token_count", 0))
            for row in arm_rows
        )
        point_count = sum(int(row.get("instance_points", 0)) for row in object_rows)
        out.setdefault("token_weighted", {})[arm] = {
            metric: (
                float(
                    np.sum(
                        [
                            float(row.get(metric, np.nan)) * int(row.get("target_token_count", 0))
                            for row in arm_rows
                            if np.isfinite(row.get(metric, np.nan))
                        ]
                    )
                    / max(
                        1,
                        sum(
                            int(row.get("target_token_count", 0))
                            for row in arm_rows
                            if np.isfinite(row.get(metric, np.nan))
                        ),
                    )
                )
                if token_count
                else float("nan")
            )
            for metric in metrics
        }
        out.setdefault("raw_point_weighted", {})[arm] = {
            metric: (
                float(
                    np.sum(
                        [
                            float(row["arms"][arm].get(metric, np.nan)) * int(row.get("instance_points", 0))
                            for row in object_rows
                            if np.isfinite(row["arms"][arm].get(metric, np.nan))
                        ]
                    )
                    / max(
                        1,
                        sum(
                            int(row.get("instance_points", 0))
                            for row in object_rows
                            if np.isfinite(row["arms"][arm].get(metric, np.nan))
                        ),
                    )
                )
                if point_count
                else float("nan")
            )
            for metric in metrics
        }
        out["token_weighted"][arm]["token_count"] = int(token_count)
        out["raw_point_weighted"][arm]["point_count"] = int(point_count)
    for group in ("token_weighted", "raw_point_weighted"):
        for metric in metrics:
            out[group].setdefault("delimit3d_minus_public", {})[metric] = (
                out[group]["delimit3d"][metric] - out[group]["public"][metric]
            )

    for metric in (
        "public_margin",
        "delimit3d_margin",
    ):
        values = [float(row[metric]) for row in pair_rows if np.isfinite(row[metric])]
        out["pair_balanced"][metric] = mean_or_nan(values)
    out["pair_balanced"]["delimit3d_minus_public_margin"] = (
        out["pair_balanced"].get("delimit3d_margin", float("nan"))
        - out["pair_balanced"].get("public_margin", float("nan"))
    )
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_name[str(row["target_name"])].append(row)
    out["pair_balanced_by_semantic_name"] = {}
    for name, rows in sorted(by_name.items()):
        public = [float(row["public_margin"]) for row in rows if np.isfinite(row["public_margin"])]
        adapted = [float(row["delimit3d_margin"]) for row in rows if np.isfinite(row["delimit3d_margin"])]
        out["pair_balanced_by_semantic_name"][name] = {
            "pair_count": int(len(rows)),
            "public_margin": mean_or_nan(public),
            "delimit3d_margin": mean_or_nan(adapted),
            "delimit3d_minus_public_margin": mean_or_nan(adapted) - mean_or_nan(public),
        }
    return out


def _aggregate_geometry_cells(geometry_by_scene: Mapping[str, Any]) -> dict[str, Any]:
    """Pool the already matched per-query geometry cells across scenes."""
    accum: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    for scene_payload in geometry_by_scene.values():
        for arm in ARMS:
            for key, cell in scene_payload.get("matched_margin_cells", {}).get(arm, {}).items():
                out = accum[arm].setdefault(
                    key,
                    {"count": 0.0, "sum_margin": 0.0, "sum_positive": 0.0, "sum_negative": 0.0},
                )
                count = float(cell.get("count", 0))
                out["count"] += count
                out["sum_margin"] += count * float(cell.get("mean_margin", np.nan))
                out["sum_positive"] += count * float(cell.get("mean_positive_cosine", np.nan))
                out["sum_negative"] += count * float(cell.get("mean_negative_cosine", np.nan))
    summary: dict[str, Any] = {}
    for arm, cells in accum.items():
        summary[arm] = {}
        for key, cell in cells.items():
            count = cell["count"]
            summary[arm][key] = {
                "count": int(count),
                "mean_margin": float(cell["sum_margin"] / count) if count else float("nan"),
                "mean_positive_cosine": float(cell["sum_positive"] / count) if count else float("nan"),
                "mean_negative_cosine": float(cell["sum_negative"] / count) if count else float("nan"),
            }
    return summary


def _visualize_pairs(
    output: Path,
    pair_rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    *,
    seed: int,
    limit: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = sorted(
        [row for row in pair_rows if np.isfinite(row["public_margin"]) and np.isfinite(row["delimit3d_margin"])],
        key=lambda row: float(row["delimit3d_margin"] - row["public_margin"]),
        reverse=True,
    )[: int(limit)]
    visual_dir = output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for pair in chosen:
        scene = str(pair["scene"])
        target_id = int(pair["target_instance"])
        distractor_id = int(pair["distractor_instance"])
        public = load_cache(cache_root, "public", scene)
        adapted = load_cache(cache_root, "delimit3d", scene)
        xyz = _as_numpy(public["scene_xyz"], np.float32)
        feats = {
            "public": unit(_as_numpy(public["features"], np.float32)),
            "delimit3d": unit(_as_numpy(adapted["features"], np.float32)),
        }
        target = _as_numpy(public["object_token_indices"][str(target_id)], np.int64)
        distractor = _as_numpy(public["object_token_indices"][str(distractor_id)], np.int64)
        target = sample_indices(target, 2500, np.random.default_rng(scene_seed(seed, scene, target_id, "visual-target")))
        distractor = sample_indices(distractor, 2500, np.random.default_rng(scene_seed(seed, scene, distractor_id, "visual-distractor")))
        candidate = np.unique(np.concatenate([target, distractor]))
        # PCA over token coordinates makes the panel view independent of the
        # ScanNet axis convention while preserving scene-local geometry.
        centered = xyz[candidate] - np.mean(xyz[candidate], axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt[:2].T
        index = {int(value): i for i, value in enumerate(candidate)}
        tmask = np.array([int(v) in set(target.tolist()) for v in candidate])
        dmask = np.array([int(v) in set(distractor.tolist()) for v in candidate])
        rng = np.random.default_rng(scene_seed(seed, scene, target_id, distractor_id, "visual-query"))
        query = int(rng.choice(target))
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
        panel_data = [
            ("Token geometry", np.where(tmask, "#e45756", np.where(dmask, "#2e86ab", "#b9c0c8")), None),
            ("Public cosine", None, feats["public"] @ feats["public"][query]),
            ("Delimit3D cosine", None, feats["delimit3d"] @ feats["delimit3d"][query]),
            ("Target / distractor", np.where(tmask, "#e45756", np.where(dmask, "#2e86ab", "#e8e8e8")), None),
            ("Public: high similarity", None, feats["public"] @ feats["public"][query]),
            ("Delimit3D: high similarity", None, feats["delimit3d"] @ feats["delimit3d"][query]),
        ]
        for ax, (title, colors, scores) in zip(axes.flat, panel_data):
            if scores is None:
                ax.scatter(projected[:, 0], projected[:, 1], s=4, c=colors, alpha=0.65, linewidths=0)
            else:
                sc = ax.scatter(projected[:, 0], projected[:, 1], s=4, c=scores[candidate], cmap="viridis", vmin=-0.1, vmax=1.0, alpha=0.8, linewidths=0)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            if int(query) in index:
                ax.scatter([projected[index[query], 0]], [projected[index[query], 1]], s=65, c="black", marker="x", linewidths=1.5)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
        target_name = str(pair["target_name"])
        fig.suptitle(
            f"{scene}: {target_name} instance {target_id} vs same-name instance {distractor_id} — "
            f"margin public {pair['public_margin']:.3f} → Delimit3D {pair['delimit3d_margin']:.3f}",
            fontsize=13,
        )
        out_path = visual_dir / f"feature_selectivity_{scene}_target{target_id}_distractor{distractor_id}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(str(out_path))
    if chosen:
        # A small montage is convenient for the paper review and remains a
        # direct derivative of the per-pair panels above.
        fig, axes = plt.subplots(len(paths), 1, figsize=(10, 3.0 * len(paths)))
        axes = np.atleast_1d(axes)
        for ax, path in zip(axes, paths):
            image = plt.imread(path)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(Path(path).name, fontsize=9)
        montage = visual_dir / "feature_selectivity_top_pairs_montage.png"
        fig.savefig(montage, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(montage))
    return paths


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    root = args.experiment_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest, manifest_sha = load_manifest(root)
    records = {str(item["scene"]): item for item in manifest["validation_scenes"]}
    scenes = sorted(records)
    if args.scene:
        missing = sorted(set(args.scene) - set(records))
        if missing:
            raise KeyError(f"scenes absent from validation manifest: {missing}")
        scenes = [str(scene) for scene in args.scene]
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise ValueError("no scenes selected")

    config = {
        "schema": SCHEMA,
        "experiment_root": str(root),
        "selection_manifest": str(root / "selection_manifest.json"),
        "selection_manifest_sha256": manifest_sha,
        "cache_reports": {
            arm: str(root / f"cache_report_{arm}.json") for arm in ARMS
        },
        "arms": {
            "public": {
                "label": ARM_LABEL["public"],
                "checkpoint_sha256": "86408c6371555aeee2a8eda1184b55a411f9748784c275349149b509b661d518",
                "feature_space": "native dec0 representative-token features",
            },
            "delimit3d": {
                "label": ARM_LABEL["delimit3d"],
                "checkpoint_sha256": "4717369861cff81fe784efda8cd8a7653de46d35635730f10c893da563050282",
                "feature_space": "native dec0 representative-token features",
            },
        },
        "scene_selection": {
            "split": "official ScanNet40 validation list in selection_manifest.json",
            "scene_count": int(len(scenes)),
            "base_scene_count": int(len({base_scene_id(scene) for scene in scenes})),
            "scenes": scenes,
        },
        "sampling": {
            "seed": int(args.seed),
            "minimum_representative_tokens_per_object": MIN_TARGET_TOKENS,
            "query_samples_per_object": int(args.query_samples),
            "region_samples_per_object": int(args.region_samples),
            "coherence_pairs_per_object": int(args.coherence_pairs),
            "retrieval_queries_per_object": int(args.retrieval_queries),
            "fixed_candidate_budgets_tokens": [128, 512, 2048],
            "target_size_control": "K = number of target representative tokens",
        },
        "geometry_bins": {
            "distance_m": [0.0, 0.25, 0.5, 1.0, 2.0, "inf"],
            "abs_normal_cosine": [0.0, 0.5, 0.8, 0.95, 1.00001],
            "normal_source": "representative_indices applied to manifest normal_path",
        },
        "aggregation": {
            "primary": "scene-balanced mean; paired bootstrap clustered by ScanNet base scene",
            "descriptive_scene_unit": "sceneXXXX_YY room entry",
            "bootstrap_cluster_unit": "sceneXXXX base scene",
            "additional": ["object-balanced", "token-weighted", "raw-point-weighted"],
            "bootstrap": "paired base-scene-cluster bootstrap over selected validation rooms",
            "bootstrap_resamples": int(args.bootstrap),
        },
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    scene_summaries: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    geometry_by_scene: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        try:
            caches = {arm: load_cache(root, arm, scene) for arm in ARMS}
            summary, objects, extras = _pair_metric(
                scene,
                records[scene],
                caches,
                seed=int(args.seed),
                query_samples=int(args.query_samples),
                region_samples=int(args.region_samples),
                coherence_pairs_count=int(args.coherence_pairs),
                retrieval_queries=int(args.retrieval_queries),
            )
            scene_summaries.append(summary)
            object_rows.extend(objects)
            pair_rows.extend(extras["pair_rows"])
            geometry_by_scene[scene] = {
                "unmatched_pair_cells": extras["geometry"],
                "matched_margin_cells": extras["geometry_matched"],
            }
            print(
                f"SCENE {index}/{len(scenes)} {scene} tokens={summary['token_count']} "
                f"objects={summary['object_count']} pairs={summary['same_class_pair_count']} "
                f"coh={summary['arms']['public']['coherence']:.4f}->"
                f"{summary['arms']['delimit3d']['coherence']:.4f} "
                f"ap={summary['arms']['public']['retrieval_ap']:.4f}->"
                f"{summary['arms']['delimit3d']['retrieval_ap']:.4f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"scene": scene, "type": type(exc).__name__, "error": str(exc)})
            print(f"ERROR {scene}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    completed_scene_names = {str(row["scene"]) for row in scene_summaries}
    missing_scenes = sorted(set(scenes) - completed_scene_names)
    if errors or missing_scenes or len(completed_scene_names) != len(scenes):
        raise RuntimeError(
            f"incomplete feature-selectivity run: completed={len(scene_summaries)} "
            f"selected={len(scenes)} errors={len(errors)} missing={missing_scenes[:8]}"
        )

    aggregate = _numeric_aggregate(
        scene_summaries,
        object_rows,
        pair_rows,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    aggregate["schema"] = SCHEMA
    aggregate["errors"] = errors
    aggregate["scene_count_completed"] = int(len(scene_summaries))
    aggregate["object_count_completed"] = int(len(object_rows))
    aggregate["pair_count_completed"] = int(len(pair_rows))
    aggregate["runtime_seconds"] = float(time.time() - started)
    aggregate["denominators"] = {
        "selected_scan_entries": int(len(scenes)),
        "completed_scan_entries": int(len(scene_summaries)),
        "completed_base_scenes": int(len({base_scene_id(row["scene"]) for row in scene_summaries})),
        "eligible_objects_min32": int(len(object_rows)),
        "ordered_same_class_pairs": int(len(pair_rows)),
        "official_background_tokens_total": int(
            sum(int(row.get("official_background_tokens", 0)) for row in scene_summaries)
        ),
        "official_objects_total": int(
            sum(int(row.get("official_object_count", 0)) for row in scene_summaries)
        ),
        "eligible_objects_total": int(
            sum(int(row.get("eligible_object_count", 0)) for row in scene_summaries)
        ),
        "excluded_objects_below32_total": int(
            sum(int(row.get("excluded_below_min_tokens", 0)) for row in scene_summaries)
        ),
    }
    aggregate["geometry_matched"] = _aggregate_geometry_cells(geometry_by_scene)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=True))
    (output / "scene_summaries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in scene_summaries)
    )
    (output / "object_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in object_rows)
    )
    (output / "same_class_pair_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in pair_rows)
    )
    (output / "geometry_by_scene.json").write_text(
        json.dumps(geometry_by_scene, indent=2, sort_keys=True, allow_nan=True)
    )
    (output / "geometry_matched_summary.json").write_text(
        json.dumps(aggregate["geometry_matched"], indent=2, sort_keys=True, allow_nan=True)
    )

    if args.visuals and pair_rows:
        paths = _visualize_pairs(
            output,
            pair_rows,
            root,
            seed=int(args.seed),
            limit=int(args.visual_pairs),
        )
        (output / "visuals_manifest.json").write_text(
            json.dumps({"paths": paths, "pair_selection": "largest adapted-public margin"}, indent=2)
        )
    receipt = {
        "schema": f"{SCHEMA}/receipt",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "run_config_sha256": sha256_file(output / "run_config.json"),
        "aggregate_sha256": sha256_file(output / "aggregate.json"),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(
        f"COMPLETE scenes={len(scene_summaries)}/{len(scenes)} objects={len(object_rows)} "
        f"pairs={len(pair_rows)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
