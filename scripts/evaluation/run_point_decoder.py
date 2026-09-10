#!/usr/bin/env python3
"""Train/evaluate the matched frozen-backbone point-decoder experiment.

The script has two explicit modes:

``--prepare``
    Freeze a deterministic ScanNet++ train-scene selection, object/query
    selection, source hashes, and one shared decoder initialization.

``--arm public|delimit3d``
    Load one frozen LitePT checkpoint, train only a decoder cloned from the
    shared initialization on the prepared training scenes, and evaluate the
    resulting decoder on the fixed validation bundle.

No encoder parameter is ever passed to an optimizer.  The validation bundle
is used only for the final held-out evaluation; GT masks are used to create
the downstream decoder's training targets on the disjoint ScanNet++ train
scenes and to score the validation predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Allow running the file directly from a source checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from delimit3d.data.single_scene_dataset import build_input_features  # noqa: E402
from delimit3d.evaluation.features import (  # noqa: E402
    center_shift,
    infer_features,
    load_backbone,
    sha256_file,
)
from delimit3d.evaluation.point_decoder import (  # noqa: E402
    PointConditionedObjectDecoder,
    balanced_feature_rows,
    load_initialization,
    save_initialization,
    state_sha256,
)
from delimit3d.evaluation.prompt_metrics import retrieval_metrics  # noqa: E402


SCHEMA = "delimit3d_scannetpp_point_decoder_experiment/v1"
INIT_SCHEMA = "delimit3d_point_decoder_initialization/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="prepare the immutable selection manifest and shared decoder init",
    )
    parser.add_argument(
        "--arm",
        choices=("public", "delimit3d"),
        help="run one frozen encoder arm",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve(strict=True).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise TypeError(f"{path}: config must be a mapping")
    return dict(config)


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _scene_seed(base_seed: int, scene: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{int(base_seed)}:{scene}".encode()).digest()[:8], "little"
    )


def _read_scene_ids(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.expanduser().resolve(strict=True).read_text().splitlines()]
    scenes = [scene for scene in scenes if scene and not scene.startswith("#")]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError(f"{path}: split is empty or contains duplicate scenes")
    return scenes


def _load_class_mapping(scannetpp_root: Path) -> tuple[list[str], dict[str, str]]:
    metadata = scannetpp_root / "metadata" / "semantic_benchmark"
    classes = (metadata / "top100_instance.txt").read_text().splitlines()
    with (metadata / "map_benchmark.csv").open(newline="") as handle:
        mapping = {
            row["class"]: (row["instance_map_to"] or row["class"])
            for row in csv.DictReader(handle)
        }
    if not classes or not mapping:
        raise ValueError("ScanNet++ semantic benchmark metadata is empty")
    return classes, mapping


def _pack_dir(pack_root: Path, scene: str) -> Path:
    path = pack_root / scene / "training_pack"
    if not path.is_dir():
        raise FileNotFoundError(f"missing ScanNet++ training pack: {path}")
    return path


def _raw_scan_dir(scannetpp_root: Path, scene: str) -> Path:
    path = scannetpp_root / "data" / scene / "scans"
    if not path.is_dir():
        raise FileNotFoundError(f"missing ScanNet++ scan directory: {path}")
    return path


def _segment_group_indices(
    seg_indices: np.ndarray,
    group_segments: Sequence[int],
    *,
    sorted_seg_indices: np.ndarray | None = None,
    sort_order: np.ndarray | None = None,
) -> np.ndarray:
    """Return point indices for one segment group without rescanning the scene.

    ScanNet++ annotations contain many segment groups.  Calling ``np.isin``
    over the full point array once per group is needlessly expensive, so the
    caller may provide one sorted segment index and reuse it for all groups.
    """

    if sorted_seg_indices is None or sort_order is None:
        sort_order = np.argsort(seg_indices, kind="stable")
        sorted_seg_indices = seg_indices[sort_order]
    values = np.unique(np.asarray(group_segments, dtype=np.int64))
    starts = np.searchsorted(sorted_seg_indices, values, side="left")
    ends = np.searchsorted(sorted_seg_indices, values, side="right")
    nonempty = [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]
    if not nonempty:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate([sort_order[start:end] for start, end in nonempty]).astype(np.int64, copy=False)


def _load_object_masks(
    scene: str,
    *,
    scannetpp_root: Path,
    pack_root: Path,
    classes: Sequence[str],
    class_mapping: Mapping[str, str],
    minimum_instance_points: int,
    maximum_objects_per_scene: int,
    prompts_per_object: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, str]]:
    """Load RGBN arrays and deterministic object/query selection for a scene."""

    pack = _pack_dir(pack_root, scene)
    scan = _raw_scan_dir(scannetpp_root, scene)
    array_paths = [pack / f"{key}.npy" for key in ("points", "colors", "normals")]
    source_paths = array_paths + [scan / "segments.json", scan / "segments_anno.json"]
    if not all(path.exists() for path in source_paths):
        missing = [str(path) for path in source_paths if not path.exists()]
        raise FileNotFoundError(f"{scene}: missing source files {missing}")

    points, colors, normals = [np.load(path).astype(np.float32, copy=False) for path in array_paths]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{scene}: points must have shape [N,3]")
    if colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError(f"{scene}: RGBN arrays are not aligned")
    seg_payload = json.loads((scan / "segments.json").read_text())
    seg_indices = np.asarray(seg_payload["segIndices"], dtype=np.int64)
    if seg_indices.shape != (len(points),):
        raise ValueError(f"{scene}: segments.json does not align with points.npy")
    groups = json.loads((scan / "segments_anno.json").read_text())["segGroups"]
    sort_order = np.argsort(seg_indices, kind="stable")
    sorted_seg_indices = seg_indices[sort_order]

    masks: dict[int, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for group in groups:
        label = class_mapping.get(str(group["label"]), str(group["label"]))
        if label not in classes:
            continue
        instance = int(group["objectId"])
        if instance in masks:
            raise ValueError(f"{scene}: duplicate object id {instance}")
        # The point-to-segment contract is the same mesh-vertex contract used
        # by the existing ScanNet++ frozen validation bundle.
        indices = _segment_group_indices(
            seg_indices,
            group["segments"],
            sorted_seg_indices=sorted_seg_indices,
            sort_order=sort_order,
        )
        if len(indices) < int(minimum_instance_points):
            continue
        masks[instance] = indices.astype(np.int64, copy=False)
        metadata.append(
            {
                "instance": instance,
                "semantic_class": int(classes.index(label)),
                "semantic_name": label,
                "instance_points": int(len(indices)),
            }
        )

    rng = np.random.default_rng(_scene_seed(seed, scene))
    metadata.sort(key=lambda item: int(item["instance"]))
    before_cap = len(metadata)
    if before_cap > int(maximum_objects_per_scene):
        chosen = sorted(
            int(index) for index in rng.choice(before_cap, int(maximum_objects_per_scene), replace=False)
        )
        metadata = [metadata[index] for index in chosen]
    for item in metadata:
        indices = masks[int(item["instance"])]
        if len(indices) < int(prompts_per_object):
            raise ValueError(f"{scene}: object has fewer points than prompts")
        item["queries"] = sorted(
            int(value) for value in rng.choice(indices, int(prompts_per_object), replace=False)
        )
    source_hashes = {str(path): file_sha256(path) for path in source_paths}
    # Return selected metadata and masks separately.  The latter remains local
    # to a job and is never written to the repository.
    selected_masks = [masks[int(item["instance"])] for item in metadata]
    return points, colors, normals, metadata, {**source_hashes, "selected_count": str(before_cap)} | {
        f"target:{int(item['instance'])}": hashlib.sha256(np.asarray(mask, dtype=np.int64).tobytes()).hexdigest()
        for item, mask in zip(metadata, selected_masks)
    }


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    scannetpp_root = Path(paths["scannetpp_root"]).expanduser().resolve(strict=True)
    pack_root = Path(paths["train_pack_root"]).expanduser().resolve(strict=True)
    train_split = Path(paths["train_split"]).expanduser().resolve(strict=True)
    val_bundle = Path(paths["val_bundle"]).expanduser().resolve(strict=True)
    output_root = Path(paths["output_root"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = config["protocol"]
    seed = int(config["seed"])
    classes, class_mapping = _load_class_mapping(scannetpp_root)
    train_scenes_all = _read_scene_ids(train_split)
    val_manifest_path = val_bundle / "scene_manifest.json"
    val_manifest = json.loads(val_manifest_path.read_text())
    val_scenes = [str(record["scene"]) for record in val_manifest["scenes"]]
    train_count = int(protocol["train_scene_count"])
    if train_count <= 0 or train_count > len(train_scenes_all):
        raise ValueError("train_scene_count exceeds the official ScanNet++ train split")
    train_scenes = train_scenes_all[:train_count]
    overlap = sorted(set(train_scenes) & set(val_scenes))
    if overlap:
        raise ValueError(f"train/validation scene leakage: {overlap}")

    train_records: list[dict[str, Any]] = []
    for index, scene in enumerate(train_scenes):
        points, _colors, _normals, objects, hashes = _load_object_masks(
            scene,
            scannetpp_root=scannetpp_root,
            pack_root=pack_root,
            classes=classes,
            class_mapping=class_mapping,
            minimum_instance_points=int(protocol["minimum_instance_points"]),
            maximum_objects_per_scene=int(protocol["maximum_objects_per_scene"]),
            prompts_per_object=int(protocol["prompts_per_object"]),
            seed=seed,
        )
        train_records.append(
            {
                "scene": scene,
                "points": int(len(points)),
                "objects": objects,
                "source_hashes": hashes,
            }
        )
        print(
            json.dumps(
                {
                    "mode": "prepare",
                    "scene": scene,
                    "completed": index + 1,
                    "total": len(train_scenes),
                    "points": int(len(points)),
                    "objects": len(objects),
                }
            ),
            flush=True,
        )

    init_path = output_root / "decoder_init.pt"
    if not init_path.exists():
        init_report = save_initialization(
            init_path,
            seed=int(config["decoder"]["init_seed"]),
            feature_dim=int(config["decoder"]["feature_dim"]),
            hidden_dim=int(config["decoder"]["hidden_dim"]),
            bottleneck_dim=int(config["decoder"]["bottleneck_dim"]),
            metadata={
                "experiment": str(config["experiment_id"]),
                "purpose": "same fresh decoder cloned by public and Delimit3D arms",
            },
        )
    else:
        _decoder, init_report = load_initialization(
            init_path,
            expected_feature_dim=int(config["decoder"]["feature_dim"]),
            expected_hidden_dim=int(config["decoder"]["hidden_dim"]),
            expected_bottleneck_dim=int(config["decoder"]["bottleneck_dim"]),
        )

    manifest = {
        "schema": SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "repo_commit": str(config.get("repo_commit", "unknown")),
        "seed": seed,
        "protocol": dict(protocol),
        "train_split": str(train_split),
        "train_split_sha256": file_sha256(train_split),
        "validation_bundle": str(val_bundle),
        "validation_scene_manifest_sha256": file_sha256(val_manifest_path),
        "train_scenes": train_records,
        "validation_scenes": val_scenes,
        "decoder_initialization": init_report,
        "decoder_initialization_path": str(init_path),
        "classes_sha256": _json_sha256(classes),
        "class_mapping_sha256": _json_sha256(class_mapping),
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    manifest_path = output_root / "selection_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"refusing to overwrite changed selection manifest: {manifest_path}"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"mode": "prepare_complete", "manifest": str(manifest_path), **init_report}, indent=2))
    return manifest


def _verify_prepared(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve(strict=True)
    manifest_path = output_root / "selection_manifest.json"
    init_path = output_root / "decoder_init.pt"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("selection manifest schema mismatch")
    if manifest.get("experiment_id") != config["experiment_id"]:
        raise ValueError("selection manifest belongs to another experiment")
    expected_hash = manifest.get("manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_sha256", None)
    if expected_hash != _json_sha256(without_hash):
        raise ValueError("selection manifest hash mismatch")
    if manifest.get("decoder_initialization_path") != str(init_path):
        raise ValueError("decoder initialization path drift")
    _decoder, report = load_initialization(
        init_path,
        expected_feature_dim=int(config["decoder"]["feature_dim"]),
        expected_hidden_dim=int(config["decoder"]["hidden_dim"]),
        expected_bottleneck_dim=int(config["decoder"]["bottleneck_dim"]),
    )
    if report["tensor_state_sha256"] != manifest["decoder_initialization"]["tensor_state_sha256"]:
        raise ValueError("decoder initialization hash drift")
    return manifest, init_path


def _rebuild_scene_selection(
    scene: str,
    *,
    config: Mapping[str, Any],
    expected_record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[int, np.ndarray]]:
    paths = config["paths"]
    scannetpp_root = Path(paths["scannetpp_root"]).expanduser().resolve(strict=True)
    pack_root = Path(paths["train_pack_root"]).expanduser().resolve(strict=True)
    protocol = config["protocol"]
    classes, class_mapping = _load_class_mapping(scannetpp_root)
    points, colors, normals, objects, hashes = _load_object_masks(
        scene,
        scannetpp_root=scannetpp_root,
        pack_root=pack_root,
        classes=classes,
        class_mapping=class_mapping,
        minimum_instance_points=int(protocol["minimum_instance_points"]),
        maximum_objects_per_scene=int(protocol["maximum_objects_per_scene"]),
        prompts_per_object=int(protocol["prompts_per_object"]),
        seed=int(config["seed"]),
    )
    if objects != expected_record["objects"]:
        raise ValueError(f"{scene}: deterministic object/query selection drift")
    # Reconstruct target indices without a second copy of all data.  The raw
    # segment arrays are small relative to the retained feature tensor and the
    # mask generation is identical to _load_object_masks.
    scan = _raw_scan_dir(scannetpp_root, scene)
    seg_indices = np.asarray(json.loads((scan / "segments.json").read_text())["segIndices"], dtype=np.int64)
    groups = json.loads((scan / "segments_anno.json").read_text())["segGroups"]
    sort_order = np.argsort(seg_indices, kind="stable")
    sorted_seg_indices = seg_indices[sort_order]
    targets: dict[int, np.ndarray] = {}
    expected_instances = {int(item["instance"]) for item in objects}
    for group in groups:
        instance = int(group["objectId"])
        if instance not in expected_instances:
            continue
        indices = _segment_group_indices(
            seg_indices,
            group["segments"],
            sorted_seg_indices=sorted_seg_indices,
            sort_order=sort_order,
        )
        targets[instance] = indices.astype(np.int64, copy=False)
    if set(targets) != expected_instances:
        raise ValueError(f"{scene}: failed to rebuild all selected target masks")
    # Verify source hashes stored during preparation.  Target hashes are also
    # checked to detect changes to annotation order/content.
    for key, value in expected_record["source_hashes"].items():
        if key.startswith("target:") or key == "selected_count":
            continue
        observed = file_sha256(Path(key))
        if observed != value:
            raise ValueError(f"{scene}: source hash drift for {key}")
    for item in objects:
        target_hash = hashlib.sha256(np.asarray(targets[int(item["instance"])], dtype=np.int64).tobytes()).hexdigest()
        expected = expected_record["source_hashes"].get(f"target:{int(item['instance'])}")
        if expected is not None and target_hash != expected:
            raise ValueError(f"{scene}: target mask hash drift for object {item['instance']}")
    return points, colors, normals, objects, targets


def _unit_features(native: np.ndarray) -> np.ndarray:
    native = np.asarray(native, dtype=np.float32)
    norms = np.linalg.norm(native, axis=1, keepdims=True)
    return native / np.maximum(norms, 1e-8)


def _train_scene(
    decoder: PointConditionedObjectDecoder,
    optimizer: torch.optim.Optimizer,
    *,
    unit_features: np.ndarray,
    objects: Sequence[Mapping[str, Any]],
    targets: Mapping[int, np.ndarray],
    samples_per_class: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    row_features: list[np.ndarray] = []
    row_labels: list[np.ndarray] = []
    for object_index, item in enumerate(objects):
        target = targets[int(item["instance"])]
        for query_offset, query in enumerate(item["queries"]):
            rng = np.random.default_rng(seed + object_index * 1009 + query_offset * 9176)
            indices, labels = balanced_feature_rows(
                unit_features,
                target,
                int(query),
                samples_per_class=samples_per_class,
                rng=rng,
            )
            q = unit_features[int(query)]
            p = unit_features[indices]
            q_rows = np.broadcast_to(q[None, :], p.shape).copy()
            relation = np.concatenate([p, q_rows, p - q_rows, p * q_rows], axis=1)
            row_features.append(relation.astype(np.float32, copy=False))
            row_labels.append(labels.astype(np.float32, copy=False))
    if not row_features:
        return {"rows": 0, "updates": 0, "mean_loss": float("nan")}
    features = torch.from_numpy(np.concatenate(row_features, axis=0)).to(device=device)
    labels = torch.from_numpy(np.concatenate(row_labels, axis=0)).to(device=device)
    # The generator is CPU-backed so the same mini-batch order is used by both
    # arms even if Slurm assigns different GPU models.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(labels), generator=generator, device="cpu")
    losses: list[float] = []
    updates = 0
    decoder.train()
    for start in range(0, len(order), int(batch_size)):
        indices = order[start : start + int(batch_size)].to(device=device)
        optimizer.zero_grad(set_to_none=True)
        # The decoder's public forward normalizes its inputs and builds the
        # same relation; feeding relation columns directly would duplicate it.
        # Split back into point/query features so the contract remains tested.
        p = features[indices, : decoder.feature_dim]
        q = features[indices, decoder.feature_dim : 2 * decoder.feature_dim]
        logits = decoder(p, q)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        updates += 1
    del features, labels
    return {
        "rows": int(len(order)),
        "updates": int(updates),
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
    }


def _binary_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int | bool]:
    metrics = dict(retrieval_metrics(target, score, cosine_threshold=float(threshold)))
    precision = float(metrics["precision_fixed"])
    recall = float(metrics["recall_fixed"])
    metrics["f1_fixed"] = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return metrics


@torch.no_grad()
def _predict_scores(
    decoder: PointConditionedObjectDecoder,
    unit_features: np.ndarray | torch.Tensor,
    query: int,
    *,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    if isinstance(unit_features, torch.Tensor):
        point = unit_features.to(device=device)
    else:
        point = torch.from_numpy(np.asarray(unit_features, dtype=np.float32)).to(device=device)
    query_tensor = point[int(query)]
    decoder.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(point), int(chunk_size)):
        logits = decoder(point[start : start + int(chunk_size)], query_tensor)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def _cosine_reference(
    unit_features: np.ndarray,
    target: np.ndarray,
    query: int,
    *,
    threshold: float,
) -> dict[str, float | int | bool]:
    candidate = np.ones(len(unit_features), dtype=bool)
    candidate[int(query)] = False
    scores = np.einsum("ij,j->i", unit_features, unit_features[int(query)])[candidate]
    return _binary_metrics(np.asarray(target, dtype=bool)[candidate], scores, threshold)


def run_arm(config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    manifest, init_path = _verify_prepared(config)
    paths = config["paths"]
    val_bundle = Path(paths["val_bundle"]).expanduser().resolve(strict=True)
    scannetpp_root = Path(paths["scannetpp_root"]).expanduser().resolve(strict=True)
    pack_root = Path(paths["train_pack_root"]).expanduser().resolve(strict=True)
    output_root = Path(paths["output_root"]).expanduser()
    arm_cfg = config["arms"][arm]
    checkpoint = Path(arm_cfg["checkpoint"]).expanduser().resolve(strict=True)
    observed_checkpoint_hash = file_sha256(checkpoint)
    if observed_checkpoint_hash != arm_cfg["checkpoint_sha256"]:
        raise ValueError(f"{arm}: checkpoint hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("the point-decoder experiment requires a CUDA GPU")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    _set_seed(int(config["seed"]))

    model = load_backbone(
        checkpoint,
        litept_root=Path(paths["litept_root"]),
        in_channels=6,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    encoder_state_before = state_sha256(model.state_dict())
    decoder, init_report = load_initialization(
        init_path,
        expected_feature_dim=int(config["decoder"]["feature_dim"]),
        expected_hidden_dim=int(config["decoder"]["hidden_dim"]),
        expected_bottleneck_dim=int(config["decoder"]["bottleneck_dim"]),
    )
    decoder.to(device)
    decoder_init_hash = state_sha256(decoder.state_dict())
    if decoder_init_hash != manifest["decoder_initialization"]["tensor_state_sha256"]:
        raise ValueError(f"{arm}: decoder was not cloned from the shared initialization")
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(config["decoder"]["learning_rate"]),
        weight_decay=float(config["decoder"]["weight_decay"]),
    )
    protocol = config["protocol"]
    train_stats: list[dict[str, Any]] = []
    train_start = time.time()
    for index, record in enumerate(manifest["train_scenes"]):
        scene = str(record["scene"])
        points, colors, normals, objects, targets = _rebuild_scene_selection(
            scene, config=config, expected_record=record
        )
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        input_features = build_input_features(
            points,
            colors,
            use_colors=True,
            use_normals=True,
            normals=normals,
        )
        shifted, _shift = center_shift(points)
        native = infer_features(model, points=shifted, features=input_features, device=device)
        unit = _unit_features(native)
        scene_stat = _train_scene(
            decoder,
            optimizer,
            unit_features=unit,
            objects=objects,
            targets=targets,
            samples_per_class=int(protocol["train_samples_per_class"]),
            batch_size=int(protocol["train_batch_size"]),
            seed=int(config["seed"]) + index * 100003,
            device=device,
        )
        scene_stat.update({"scene": scene, "completed": index + 1, "total": len(manifest["train_scenes"])})
        train_stats.append(scene_stat)
        print(json.dumps({"arm": arm, "phase": "train", **scene_stat}), flush=True)
        del points, colors, normals, input_features, shifted, native, unit, targets
        torch.cuda.empty_cache()
    train_elapsed = time.time() - train_start
    decoder.eval()
    decoder_state_hash = state_sha256(decoder.state_dict())
    arm_root = output_root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    decoder_path = arm_root / "decoder_final.pt"
    torch.save(
        {
            "schema": "delimit3d_point_decoder_trained/v1",
            "experiment_id": config["experiment_id"],
            "arm": arm,
            "checkpoint_sha256": observed_checkpoint_hash,
            "decoder_initialization_sha256": decoder_init_hash,
            "decoder_state_sha256": decoder_state_hash,
            "state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
            "train_stats": train_stats,
        },
        decoder_path,
    )

    val_manifest = json.loads((val_bundle / "scene_manifest.json").read_text())
    metrics_path = arm_root / "queries.jsonl"
    scenes_path = arm_root / "scenes.jsonl"
    objects_path = arm_root / "objects.jsonl"
    query_count = 0
    object_count = 0
    scene_count = 0
    with metrics_path.open("w") as query_stream, scenes_path.open("w") as scene_stream, objects_path.open("w") as object_stream:
        eval_start = time.time()
        for index, record in enumerate(val_manifest["scenes"]):
            scene = str(record["scene"])
            data_path = Path(record["data_path"]).expanduser().resolve(strict=True)
            if file_sha256(data_path) != record["data_sha256"]:
                raise ValueError(f"{scene}: validation data hash mismatch")
            data = np.load(data_path)
            points = np.asarray(data["points"], dtype=np.float32)
            colors = np.asarray(data["colors"], dtype=np.float32)
            normals = np.asarray(data["normals"], dtype=np.float32)
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
            input_features = build_input_features(
                points,
                colors,
                use_colors=True,
                use_normals=True,
                normals=normals,
            )
            shifted, _shift = center_shift(points)
            native = infer_features(model, points=shifted, features=input_features, device=device)
            unit = _unit_features(native)
            unit_tensor = torch.from_numpy(unit).to(device=device)
            scene_query_metrics: list[dict[str, Any]] = []
            for item in record["objects"]:
                instance = int(item["instance"])
                target_indices = np.asarray(data[f"target_{instance}"], dtype=np.int64)
                target_full = np.zeros(len(points), dtype=bool)
                target_full[target_indices] = True
                per_query: list[dict[str, Any]] = []
                eval_queries = list(item["queries"][: int(protocol["eval_prompts_per_object"])])
                if not eval_queries:
                    raise ValueError(f"{scene}: object {instance} has no evaluation query")
                for query in eval_queries:
                    query = int(query)
                    candidate = np.ones(len(points), dtype=bool)
                    candidate[query] = False
                    scores = _predict_scores(
                        decoder,
                        unit_tensor,
                        query,
                        device=device,
                        chunk_size=int(protocol["prediction_chunk_size"]),
                    )[candidate]
                    target = target_full[candidate]
                    values = _binary_metrics(target, scores, float(protocol["decoder_threshold"]))
                    cosine = _cosine_reference(
                        unit,
                        target_full,
                        query,
                        threshold=float(protocol["cosine_threshold"]),
                    )
                    row = {
                        "scene": scene,
                        "instance": instance,
                        "semantic_class": int(item["semantic_class"]),
                        "semantic_name": str(item["semantic_name"]),
                        "instance_points": int(item["instance_points"]),
                        "query": query,
                        "clicks": 1,
                        "candidate_points": int(candidate.sum()),
                        "target_points": int(target.sum()),
                        "decoder": values,
                        "cosine_reference": cosine,
                    }
                    query_stream.write(json.dumps(row) + "\n")
                    per_query.append(row)
                    scene_query_metrics.append(row)
                    query_count += 1
                object_metric = {
                    "scene": scene,
                    "instance": instance,
                    "semantic_class": int(item["semantic_class"]),
                    "semantic_name": str(item["semantic_name"]),
                    "clicks": 1,
                    "decoder": {
                        key: float(np.mean([float(row["decoder"][key]) for row in per_query]))
                        for key in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
                    },
                    "cosine_reference": {
                        key: float(np.mean([float(row["cosine_reference"][key]) for row in per_query]))
                        for key in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
                    },
                }
                object_stream.write(json.dumps(object_metric) + "\n")
                object_count += 1
            scene_metric = {
                "scene": scene,
                "objects": len(record["objects"]),
                "queries": len(scene_query_metrics),
                "decoder": {
                    key: float(np.mean([float(row["decoder"][key]) for row in scene_query_metrics]))
                    for key in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
                },
                "cosine_reference": {
                    key: float(np.mean([float(row["cosine_reference"][key]) for row in scene_query_metrics]))
                    for key in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
                },
            }
            scene_stream.write(json.dumps(scene_metric) + "\n")
            scene_count += 1
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "phase": "eval",
                        "scene": scene,
                        "completed": index + 1,
                        "total": len(val_manifest["scenes"]),
                        "queries": len(scene_query_metrics),
                    }
                ),
                flush=True,
            )
            data.close()
            del points, colors, normals, input_features, shifted, native, unit, unit_tensor
            torch.cuda.empty_cache()

    encoder_state_after = state_sha256(model.state_dict())
    report = {
        "schema": "delimit3d_point_decoder_arm_report/v1",
        "experiment_id": config["experiment_id"],
        "arm": arm,
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "repo_commit": config.get("repo_commit", "unknown"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": observed_checkpoint_hash,
        "decoder_initialization": init_report,
        "decoder_initialization_tensor_sha256": decoder_init_hash,
        "decoder_final": str(decoder_path),
        "decoder_final_sha256": file_sha256(decoder_path),
        "decoder_final_tensor_sha256": decoder_state_hash,
        "encoder_tensor_sha256_before": encoder_state_before,
        "encoder_tensor_sha256_after": encoder_state_after,
        "encoder_state_unchanged": encoder_state_before == encoder_state_after,
        "train_scenes": len(manifest["train_scenes"]),
        "validation_scenes": scene_count,
        "objects": object_count,
        "queries": query_count,
        "evaluation_prompts_per_object": int(protocol["eval_prompts_per_object"]),
        "click_budget": 1,
        "optimizer_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
        "train_seconds": train_elapsed,
        "eval_seconds": time.time() - eval_start,
        "train_stats": train_stats,
        "protocol": dict(protocol),
        "passed": (
            encoder_state_before == encoder_state_after
            and scene_count == len(val_manifest["scenes"])
            and query_count == sum(
                min(int(protocol["eval_prompts_per_object"]), len(item["queries"]))
                for rec in val_manifest["scenes"]
                for item in rec["objects"]
            )
        ),
    }
    if not report["passed"]:
        raise RuntimeError(f"{arm}: arm invariant failed")
    (arm_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"arm": arm, "phase": "complete", **{key: report[key] for key in ("job_id", "train_scenes", "validation_scenes", "objects", "queries", "encoder_state_unchanged", "decoder_final_tensor_sha256")}}, indent=2), flush=True)
    return report


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.prepare:
        prepare(config)
        if args.arm:
            raise ValueError("--prepare and --arm are mutually exclusive")
        return
    if not args.arm:
        raise ValueError("one of --prepare or --arm is required")
    run_arm(config, args.arm)


if __name__ == "__main__":
    main()
