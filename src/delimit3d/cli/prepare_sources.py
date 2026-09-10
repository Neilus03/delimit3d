#!/usr/bin/env python3
"""Build compact GT3 + UnSAM3 contrastive sources for ScanNet scenes.

The UnSAMv2 model is loaded once per invocation and reused across a contiguous
scene chunk. Pixel masks exist only long enough to project them onto the
canonical scene points; completed scene packs retain compact point-index
proposal caches, not HxW masks.

The per-scene manifest is written last. An existing valid manifest is reused;
an existing incomplete directory fails closed rather than being silently
overwritten.
"""

from __future__ import annotations

from delimit3d.identity import project_name, project_slug

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from delimit3d.source.core.lifting.project import project_points_to_image
from delimit3d.source.core.lifting.visibility import compute_visible_points
from delimit3d.source.datasets.scannet.adapter import ScanNetSceneAdapter
from delimit3d.source.datasets.scannet.prepare import _load_sensor_data_class, _save_mat
try:
    from delimit3d.training.contracts import (
        EXPERIMENT_ID,
        GRANULARITY_KEYS,
        RAW_CACHE_SCHEMA,
        SCENE_SOURCE_SCHEMA,
        SOURCE_CELLS,
        audit_scene_source_manifest,
        frame_split,
        load_scene_ids,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - kept for direct execution
    from delimit3d.training.contracts import (
        EXPERIMENT_ID,
        GRANULARITY_KEYS,
        RAW_CACHE_SCHEMA,
        SCENE_SOURCE_SCHEMA,
        SOURCE_CELLS,
        audit_scene_source_manifest,
        frame_split,
        load_scene_ids,
        sha256_file,
    )
from delimit3d.data.region_sampling import sphere_crop_indices
from delimit3d.data.unsam_proposals import (
    UnSAMProposalSet,
    load_proposal_set,
    proposal_overlap_statistics,
    proposals_from_teacher_output,
)


log = logging.getLogger("prepare_partfield_contrastive_gt3_unsam3_sources")

GRANULARITIES = {"g02": 0.2, "g05": 0.5, "g08": 0.8}
LABEL_FILES = {
    "g02": "labels_g0.2.npy",
    "g05": "labels_g0.5.npy",
    "g08": "labels_g0.8.npy",
}
CROP_SEEDS = tuple(range(42, 50))
POINT_MAX = 50_000
POSE_CANDIDATES = 64
MIN_POSITIVE_POINTS = 2


@dataclass(frozen=True)
class ProjectedProposalFrame:
    frame_id: str
    split: str
    eligible_global_indices: np.ndarray
    proposal_global_indices: tuple[np.ndarray, ...]
    proposal_order_indices: np.ndarray
    proposal_source_indices: np.ndarray
    raw_stats: dict[str, Any]
    used_relaxed_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-ids-file", type=Path, required=True)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-stop", type=int, default=None)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--hierarchy-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument(
        "--proposal-input-root",
        type=Path,
        default=None,
        help=(
            "Audit/smoke-only source of existing raw proposal files laid out as "
            "<root>/<scene_id>/raw_unsam_g0.2/<frame>.npz (and g0.5/g0.8). "
            "Production source generation omits this option."
        ),
    )
    return parser.parse_args()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    display_path: str
    if relative_to is None:
        display_path = str(resolved)
    else:
        display_path = str(resolved.relative_to(relative_to.resolve(strict=True)))
    return {
        "path": display_path,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _array_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        **_file_record(path, relative_to=relative_to),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    if len(items) < int(count):
        raise ValueError(f"Only {len(items)} items available; need {count}")
    positions = np.linspace(0, len(items) - 1, num=int(count), dtype=np.int64)
    if np.unique(positions).size != int(count):
        raise RuntimeError("Evenly spaced selection produced duplicate indices")
    return [items[int(index)] for index in positions]


def _select_frames(
    adapter: ScanNetSceneAdapter,
    *,
    frame_ids_override: Sequence[str] | None = None,
    candidate_frame_ids: Sequence[str] | None = None,
) -> tuple[list[Any], list[dict[str, str]], list[str], list[str]]:
    all_frames = adapter.list_frames()
    if frame_ids_override is not None:
        by_id = {frame.frame_id: frame for frame in all_frames}
        missing = [frame_id for frame_id in frame_ids_override if frame_id not in by_id]
        if missing:
            raise FileNotFoundError(
                f"{adapter.scene_id}: smoke proposal frames missing in scene: {missing}"
            )
        selected = [by_id[frame_id] for frame_id in frame_ids_override]
        for frame in selected:
            adapter.load_pose_c2w(frame)
        frame_ids = [frame.frame_id for frame in selected]
        train_ids, heldout_ids = frame_split(frame_ids)
        return selected, [], train_ids, heldout_ids
    if candidate_frame_ids is not None:
        by_id = {frame.frame_id: frame for frame in all_frames}
        missing = [
            frame_id for frame_id in candidate_frame_ids if frame_id not in by_id
        ]
        if missing:
            raise FileNotFoundError(
                f"{adapter.scene_id}: selectively extracted candidates missing: {missing}"
            )
        candidates = [by_id[frame_id] for frame_id in candidate_frame_ids]
    elif len(all_frames) < POSE_CANDIDATES:
        raise ValueError(
            f"{adapter.scene_id} has {len(all_frames)} frames; "
            f"the frozen contract requires {POSE_CANDIDATES} pose candidates"
        )
    else:
        candidates = _evenly_spaced(all_frames, POSE_CANDIDATES)
    valid: list[Any] = []
    invalid: list[dict[str, str]] = []
    for frame in candidates:
        try:
            adapter.load_pose_c2w(frame)
        except Exception as exc:
            invalid.append({"frame_id": frame.frame_id, "reason": repr(exc)})
        else:
            valid.append(frame)
    selected = _evenly_spaced(valid, 16)
    frame_ids = [frame.frame_id for frame in selected]
    train_ids, heldout_ids = frame_split(frame_ids)
    return selected, invalid, train_ids, heldout_ids


def _sensor_candidate_indices(total_frames: int) -> np.ndarray:
    if int(total_frames) < POSE_CANDIDATES:
        raise ValueError(
            f"Sensor has {total_frames} frames; need {POSE_CANDIDATES}"
        )
    indices = np.linspace(
        0,
        int(total_frames) - 1,
        num=POSE_CANDIDATES,
        dtype=np.int64,
    )
    if np.unique(indices).size != POSE_CANDIDATES:
        raise RuntimeError("Sensor candidate selection produced duplicate indices")
    return indices


def _ensure_selective_rgbd(scene_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Extract or verify the exact 64 RGB-D candidates used by this run."""

    manifest_path = scene_dir / "selective_rgbd_gt3_unsam3_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version")
            != "scannet_selective_rgbd_gt3_unsam3/v1"
            or not manifest.get("passed")
        ):
            raise RuntimeError(f"Invalid selective RGB-D manifest: {manifest_path}")
        return (
            [str(value) for value in manifest["candidate_indices"]],
            _file_record(manifest_path),
        )

    scene_id = scene_dir.name
    sens_path = scene_dir / f"{scene_id}.sens"
    if not sens_path.is_file():
        raise FileNotFoundError(sens_path)
    SensorData = _load_sensor_data_class()
    sensor = SensorData(str(sens_path))
    total_sensor_frames = int(len(sensor.frames))
    indices = _sensor_candidate_indices(total_sensor_frames)

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)
    matrices = {
        "intrinsic_color.txt": sensor.intrinsic_color,
        "extrinsic_color.txt": sensor.extrinsic_color,
        "intrinsic_depth.txt": sensor.intrinsic_depth,
        "extrinsic_depth.txt": sensor.extrinsic_depth,
    }
    for filename, matrix in matrices.items():
        path = intrinsic_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            _save_mat(matrix, path)

    valid_pose_ids: list[str] = []
    for ordinal, frame_index_value in enumerate(indices):
        frame_index = int(frame_index_value)
        frame = sensor.frames[frame_index]
        color_path = color_dir / f"{frame_index}.jpg"
        depth_path = depth_dir / f"{frame_index}.png"
        pose_path = pose_dir / f"{frame_index}.txt"
        if not depth_path.is_file():
            depth_data = frame.decompress_depth(sensor.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                sensor.depth_height, sensor.depth_width
            )
            imageio.imwrite(depth_path, depth)
        if not color_path.is_file():
            color = frame.decompress_color(sensor.color_compression_type)
            imageio.imwrite(color_path, color, quality=95)
        pose = np.asarray(frame.camera_to_world)
        if not pose_path.is_file():
            _save_mat(pose, pose_path)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            valid_pose_ids.append(str(frame_index))
        if ordinal == 0 or (ordinal + 1) % 16 == 0:
            log.info(
                "%s extracted/verified %d/%d RGB-D candidates",
                scene_id,
                ordinal + 1,
                len(indices),
            )

    del sensor
    candidate_ids = [str(int(index)) for index in indices]
    checks = {
        "exact_candidate_count": len(candidate_ids) == POSE_CANDIDATES,
        "enough_valid_pose_candidates": len(valid_pose_ids) >= 16,
        "candidate_color_complete": all(
            (color_dir / f"{frame_id}.jpg").is_file()
            or (color_dir / f"{frame_id}.png").is_file()
            for frame_id in candidate_ids
        ),
        "candidate_depth_complete": all(
            (depth_dir / f"{frame_id}.png").is_file()
            for frame_id in candidate_ids
        ),
        "candidate_pose_complete": all(
            (pose_dir / f"{frame_id}.txt").is_file()
            for frame_id in candidate_ids
        ),
        "intrinsics_complete": all(
            (intrinsic_dir / filename).is_file() for filename in matrices
        ),
    }
    manifest = {
        "schema_version": "scannet_selective_rgbd_gt3_unsam3/v1",
        "scene_id": scene_id,
        "sens_path": str(sens_path.resolve(strict=True)),
        "total_sensor_frames": total_sensor_frames,
        "candidate_indices": [int(value) for value in indices],
        "valid_pose_candidate_indices": [int(value) for value in valid_pose_ids],
        "checks": checks,
        "passed": all(checks.values()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not manifest["passed"]:
        raise RuntimeError(f"{scene_id}: selective RGB-D checks failed: {checks}")
    return candidate_ids, _file_record(manifest_path)


def _resize_depth(depth_m: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if depth_m.shape == image_shape:
        return depth_m
    height, width = image_shape
    return cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)


def _project_geometry(
    *,
    adapter: ScanNetSceneAdapter,
    frame: Any,
    points: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = adapter.load_pose_c2w(frame)
    intrinsics = adapter.load_intrinsics(frame)
    depth_m = _resize_depth(adapter.load_depth_m(frame), image_shape)
    u, v, z, valid = project_points_to_image(points, pose, intrinsics)
    visible, visible_u, visible_v = compute_visible_points(
        u, v, z, valid, depth_m, adapter.get_visibility_config()
    )
    return (
        visible.astype(np.int64, copy=False),
        visible_u.astype(np.int64, copy=False),
        visible_v.astype(np.int64, copy=False),
    )


class ProposalProvider:
    def proposals(
        self,
        *,
        scene_id: str,
        frame: Any,
        image: np.ndarray,
        granularity_key: str,
    ) -> tuple[UnSAMProposalSet, bool]:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        raise NotImplementedError

    def frame_ids_override(self, scene_id: str) -> list[str] | None:
        del scene_id
        return None

    def validate_scene(self, *, scene_id: str, frame_ids: Sequence[str]) -> None:
        """Validate provider-side scene inputs before projection.

        Generated proposals need no extra validation.  Existing-proposal
        providers can override this hook to require an accepted source
        manifest and a complete per-scene file set.
        """

        del scene_id, frame_ids

    def scene_provenance(self, scene_id: str) -> dict[str, Any]:
        del scene_id
        return {}


class TeacherProposalProvider(ProposalProvider):
    def __init__(
        self, *, device: str, model_cfg: str | None
    ) -> None:
        import torch

        from delimit3d.source.core.teacher.unsamv2 import UnSAMv2Teacher

        self.torch = torch
        self.teacher = UnSAMv2Teacher(
            device=device,
            model_cfg=model_cfg,
            overwrite=False,
            debug_first_n_frames=0,
        )
        self.teacher._ensure_loaded()
        self._provenance = {
            "mode": "generated_in_memory",
            "teacher": "UnSAMv2Teacher",
            "model_cfg": self.teacher.model_cfg,
            "checkpoint": _file_record(self.teacher.checkpoint_path),
        }

    def proposals(
        self,
        *,
        scene_id: str,
        frame: Any,
        image: np.ndarray,
        granularity_key: str,
    ) -> tuple[UnSAMProposalSet, bool]:
        del scene_id, frame
        granularity = GRANULARITIES[granularity_key]
        used_fallback = False
        with self.torch.inference_mode(), self.teacher._autocast_context():
            masks_data = self.teacher._mask_generator.generate(
                image, gra=granularity
            )
            if len(masks_data) == 0:
                masks_data = self.teacher._relaxed_mask_generator.generate(
                    image, gra=granularity
                )
                used_fallback = True
        return (
            proposals_from_teacher_output(
                masks_data,
                image_shape=(int(image.shape[0]), int(image.shape[1])),
            ),
            used_fallback,
        )

    def provenance(self) -> dict[str, Any]:
        return self._provenance


class DirectoryProposalProvider(ProposalProvider):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)

    def proposals(
        self,
        *,
        scene_id: str,
        frame: Any,
        image: np.ndarray,
        granularity_key: str,
    ) -> tuple[UnSAMProposalSet, bool]:
        del image
        tag = format(GRANULARITIES[granularity_key], "g")
        scene_root = self.root / scene_id
        if not scene_root.is_dir():
            scene_root = self.root
        path = scene_root / f"raw_unsam_g{tag}" / f"{frame.frame_id}.npz"
        return load_proposal_set(path), False

    def provenance(self) -> dict[str, Any]:
        return {
            "mode": "existing_raw_proposals_for_audit_or_smoke",
            "root": str(self.root),
        }

    def frame_ids_override(self, scene_id: str) -> list[str] | None:
        scene_root = self.root / scene_id
        if not scene_root.is_dir():
            scene_root = self.root
        manifest_path = scene_root / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_ids = manifest.get("frame_ids")
        if not isinstance(frame_ids, list):
            frame_ids = manifest.get("selection", {}).get("frame_ids")
        if not isinstance(frame_ids, list):
            raise ValueError(f"{manifest_path}: no ordered frame_ids for smoke reuse")
        return [str(frame_id) for frame_id in frame_ids]


def _project_proposals(
    *,
    adapter: ScanNetSceneAdapter,
    points: np.ndarray,
    selected_frames: Sequence[Any],
    train_frame_ids: Sequence[str],
    provider: ProposalProvider,
) -> dict[str, list[ProjectedProposalFrame]]:
    train_set = set(train_frame_ids)
    output: dict[str, list[ProjectedProposalFrame]] = {
        key: [] for key in GRANULARITY_KEYS
    }
    for frame in selected_frames:
        image = adapter.load_rgb(frame)
        actual_image_shape = (int(image.shape[0]), int(image.shape[1]))
        geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for key in GRANULARITY_KEYS:
            proposals, used_fallback = provider.proposals(
                scene_id=adapter.scene_id,
                frame=frame,
                image=image,
                granularity_key=key,
            )
            if proposals.num_proposals == 0:
                raise RuntimeError(f"{adapter.scene_id}/{frame.frame_id}/{key}: no proposals")
            if proposals.image_shape != actual_image_shape:
                raise RuntimeError(
                    f"{adapter.scene_id}/{frame.frame_id}/{key}: "
                    f"proposal image shape {proposals.image_shape} != "
                    f"RGB image shape {actual_image_shape}"
                )
            if geometry is None:
                geometry = _project_geometry(
                    adapter=adapter,
                    frame=frame,
                    points=points,
                    image_shape=actual_image_shape,
                )
            visible, visible_u, visible_v = geometry
            membership = proposals.masks[:, visible_v, visible_u]
            proposal_points = tuple(
                visible[row].astype(np.int64, copy=False)
                for row in membership
            )
            output[key].append(
                ProjectedProposalFrame(
                    frame_id=frame.frame_id,
                    split="train" if frame.frame_id in train_set else "heldout",
                    eligible_global_indices=visible,
                    proposal_global_indices=proposal_points,
                    proposal_order_indices=np.arange(
                        proposals.num_proposals, dtype=np.int32
                    ),
                    proposal_source_indices=proposals.source_indices.astype(
                        np.int32, copy=False
                    ),
                    raw_stats=proposal_overlap_statistics(proposals),
                    used_relaxed_fallback=used_fallback,
                )
            )
            log.info(
                "%s frame=%s %s raw_proposals=%d visible=%d",
                adapter.scene_id,
                frame.frame_id,
                key,
                proposals.num_proposals,
                visible.size,
            )
    return output


def _global_to_local(crop_indices: np.ndarray, num_points: int) -> np.ndarray:
    lookup = np.full(int(num_points), -1, dtype=np.int64)
    lookup[crop_indices] = np.arange(crop_indices.size, dtype=np.int64)
    return lookup


def _local_frame(
    frame: ProjectedProposalFrame,
    global_to_local: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    eligible = global_to_local[frame.eligible_global_indices]
    eligible = eligible[eligible >= 0]
    proposal_members: list[np.ndarray] = []
    retained_indices: list[int] = []
    for proposal_index, global_members in enumerate(frame.proposal_global_indices):
        local = global_to_local[global_members]
        local = np.unique(local[local >= 0])
        if local.size >= MIN_POSITIVE_POINTS and local.size < eligible.size:
            proposal_members.append(local.astype(np.int64, copy=False))
            retained_indices.append(proposal_index)
    return (
        np.unique(eligible).astype(np.int64, copy=False),
        proposal_members,
        np.asarray(retained_indices, dtype=np.int64),
    )


def _partition_trainable_count(labels: np.ndarray) -> int:
    supervised = labels[labels >= 0]
    if supervised.size < 3 or np.unique(supervised).size < 2:
        return 0
    _, counts = np.unique(supervised, return_counts=True)
    return int(np.count_nonzero(counts >= MIN_POSITIVE_POINTS))


def _crop_audit(
    *,
    crop_indices: np.ndarray,
    projected: dict[str, list[ProjectedProposalFrame]],
    labels_by_key: dict[str, np.ndarray],
    num_points: int,
    min_train_frames_per_level: int = 8,
    min_heldout_frames_per_level: int = 3,
) -> dict[str, Any]:
    lookup = _global_to_local(crop_indices, num_points)
    rows: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    heldout_counts: dict[str, int] = {}
    train_visible_union = np.zeros(crop_indices.size, dtype=bool)
    heldout_visible_union = np.zeros(crop_indices.size, dtype=bool)
    for key in GRANULARITY_KEYS:
        level_rows: list[dict[str, Any]] = []
        for frame in projected[key]:
            eligible, proposal_members, _ = _local_frame(frame, lookup)
            if frame.split == "train":
                train_visible_union[eligible] = True
            else:
                heldout_visible_union[eligible] = True
            level_rows.append(
                {
                    "frame_id": frame.frame_id,
                    "split": frame.split,
                    "visible_crop_points": int(eligible.size),
                    "trainable_proposals": len(proposal_members),
                }
            )
        rows[key] = level_rows
        counts[f"2d/{key}"] = int(
            sum(
                row["trainable_proposals"]
                for row in level_rows
                if row["split"] == "train"
            )
        )
        heldout_counts[f"2d/{key}"] = int(
            sum(
                row["trainable_proposals"]
                for row in level_rows
                if row["split"] == "heldout"
            )
        )

    for key in GRANULARITY_KEYS:
        count = _partition_trainable_count(labels_by_key[key][crop_indices])
        counts[f"3d/{key}"] = count
        heldout_counts[f"3d/{key}"] = count

    train_frames_per_level = {
        key: sum(
            row["trainable_proposals"] > 0
            for row in rows[key]
            if row["split"] == "train"
        )
        for key in GRANULARITY_KEYS
    }
    heldout_frames_per_level = {
        key: sum(
            row["trainable_proposals"] > 0
            for row in rows[key]
            if row["split"] == "heldout"
        )
        for key in GRANULARITY_KEYS
    }
    valid = bool(
        all(counts[cell] > 0 for cell in SOURCE_CELLS)
        and all(heldout_counts[cell] > 0 for cell in SOURCE_CELLS)
        and min(train_frames_per_level.values()) >= int(min_train_frames_per_level)
        and min(heldout_frames_per_level.values()) >= int(min_heldout_frames_per_level)
        and float(train_visible_union.mean()) >= 0.10
        and float(heldout_visible_union.mean()) >= 0.05
    )
    score = (
        min(heldout_frames_per_level.values()),
        sum(heldout_counts[f"2d/{key}"] for key in GRANULARITY_KEYS),
        min(train_frames_per_level.values()),
        sum(counts[f"2d/{key}"] for key in GRANULARITY_KEYS),
        int(heldout_visible_union.sum()),
        int(train_visible_union.sum()),
    )
    return {
        "valid": valid,
        "score": list(score),
        "trainable_proposals": counts,
        "heldout_trainable_proposals": heldout_counts,
        "train_frames_per_2d_level": train_frames_per_level,
        "heldout_frames_per_2d_level": heldout_frames_per_level,
        "train_visible_fraction": float(train_visible_union.mean()),
        "heldout_visible_fraction": float(heldout_visible_union.mean()),
        "frames": rows,
    }


def _select_crop(
    *,
    points: np.ndarray,
    projected: dict[str, list[ProjectedProposalFrame]],
    labels_by_key: dict[str, np.ndarray],
    min_train_frames_per_level: int = 8,
    min_heldout_frames_per_level: int = 3,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
    for seed in CROP_SEEDS:
        indices = sphere_crop_indices(
            points,
            rng=np.random.default_rng(seed),
            point_max=POINT_MAX,
        )
        audit = _crop_audit(
            crop_indices=indices,
            projected=projected,
            labels_by_key=labels_by_key,
            num_points=points.shape[0],
            min_train_frames_per_level=min_train_frames_per_level,
            min_heldout_frames_per_level=min_heldout_frames_per_level,
        )
        audit["seed"] = seed
        candidates.append((indices, audit))
    valid = [candidate for candidate in candidates if candidate[1]["valid"]]
    selection_pool = valid if valid else candidates
    selected_indices, selected_audit = max(
        selection_pool,
        key=lambda item: (tuple(item[1]["score"]), -int(item[1]["seed"])),
    )
    selected_audit["valid_pool_available"] = bool(valid)
    selected_audit["selection_mode"] = (
        "best_valid_crop" if valid else "best_audited_fallback"
    )
    reports = [
        {key: value for key, value in audit.items() if key != "frames"}
        for _, audit in candidates
    ]
    return selected_indices, selected_audit, reports


def _nesting_violations(fine: np.ndarray, coarse: np.ndarray) -> int:
    return int(
        sum(
            np.unique(coarse[fine == label]).size != 1
            for label in np.unique(fine[fine >= 0])
        )
    )


def _g08_matches_declared_gt(
    *,
    labels_g08: np.ndarray,
    gt_instances: np.ndarray,
    hierarchy_meta: dict[str, Any],
) -> bool:
    if labels_g08.shape != gt_instances.shape:
        return False
    expected = np.full(labels_g08.shape, -1, dtype=np.int32)
    objects = hierarchy_meta.get("objects", [])
    seen_object_labels: set[int] = set()
    seen_raw_ids: set[int] = set()
    for row in objects:
        object_label = int(row["object_label_g08"])
        raw_id = int(row["raw_gt_instance_id"])
        if object_label in seen_object_labels or raw_id in seen_raw_ids:
            return False
        seen_object_labels.add(object_label)
        seen_raw_ids.add(raw_id)
        # ScanNet aggregation JSON IDs are one-based. The existing processed
        # `instance.npy` convention used by this repository stores them
        # zero-based, while the hierarchy metadata deliberately retains the
        # original aggregation ID.
        expected[gt_instances == raw_id - 1] = object_label
    return bool(np.array_equal(labels_g08.astype(np.int32, copy=False), expected))


def _write_raw_cache(
    *,
    path: Path,
    crop_indices: np.ndarray,
    projected: dict[str, list[ProjectedProposalFrame]],
    num_points: int,
) -> dict[str, Any]:
    lookup = _global_to_local(crop_indices, int(num_points))
    frame_keys: list[str] = []
    physical_frame_ids: list[str] = []
    level_keys: list[str] = []
    split_codes: list[int] = []
    eligible_chunks: list[np.ndarray] = []
    frame_offsets = [0]
    proposal_frame_offsets = [0]
    proposal_point_chunks: list[np.ndarray] = []
    proposal_point_offsets = [0]
    proposal_order_indices: list[int] = []
    proposal_source_indices: list[int] = []
    multiple_membership_points = 0
    excess_memberships = 0

    for key in GRANULARITY_KEYS:
        for frame in projected[key]:
            eligible, members, retained = _local_frame(frame, lookup)
            eligible_chunks.append(eligible)
            frame_offsets.append(frame_offsets[-1] + int(eligible.size))
            proposal_frame_offsets.append(
                proposal_frame_offsets[-1] + len(members)
            )
            membership_count = np.zeros(crop_indices.size, dtype=np.int32)
            for member_indices, original_index in zip(members, retained, strict=True):
                proposal_point_chunks.append(member_indices)
                proposal_point_offsets.append(
                    proposal_point_offsets[-1] + int(member_indices.size)
                )
                proposal_order_indices.append(
                    int(frame.proposal_order_indices[original_index])
                )
                proposal_source_indices.append(
                    int(frame.proposal_source_indices[original_index])
                )
                membership_count[member_indices] += 1
            multiple_membership_points += int(np.count_nonzero(membership_count > 1))
            excess_memberships += int(
                np.maximum(membership_count - 1, 0).sum(dtype=np.int64)
            )
            frame_keys.append(f"{key}/{frame.frame_id}")
            physical_frame_ids.append(frame.frame_id)
            level_keys.append(key)
            split_codes.append(0 if frame.split == "train" else 1)

    if not proposal_point_chunks:
        raise RuntimeError("Compact raw cache would contain no proposals")
    np.savez_compressed(
        path,
        schema_version=np.asarray(RAW_CACHE_SCHEMA),
        crop_indices_sha256=np.asarray(sha256_file(path.parent / "crop_indices.npy")),
        frame_keys=np.asarray(frame_keys),
        physical_frame_ids=np.asarray(physical_frame_ids),
        granularity_keys=np.asarray(level_keys),
        split_codes=np.asarray(split_codes, dtype=np.int8),
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        eligible_point_indices=np.concatenate(eligible_chunks).astype(np.int64),
        proposal_frame_offsets=np.asarray(proposal_frame_offsets, dtype=np.int64),
        proposal_point_offsets=np.asarray(proposal_point_offsets, dtype=np.int64),
        proposal_point_indices=np.concatenate(proposal_point_chunks).astype(np.int64),
        proposal_order_indices=np.asarray(proposal_order_indices, dtype=np.int32),
        proposal_source_indices=np.asarray(proposal_source_indices, dtype=np.int32),
    )
    return {
        "virtual_frames": len(frame_keys),
        "retained_proposals": len(proposal_order_indices),
        "projected_points_in_multiple_proposals": multiple_membership_points,
        "projected_excess_memberships": excess_memberships,
    }


def _copy_or_save_scene_arrays(
    *,
    work_dir: Path,
    crop_indices: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    labels_by_key: dict[str, np.ndarray],
) -> None:
    np.save(work_dir / "crop_indices.npy", crop_indices.astype(np.int64))
    np.save(work_dir / "points.npy", points[crop_indices].astype(np.float32))
    np.save(work_dir / "colors.npy", colors[crop_indices].astype(np.float32))
    np.save(work_dir / "normals.npy", normals[crop_indices].astype(np.float32))
    for key in GRANULARITY_KEYS:
        np.save(
            work_dir / LABEL_FILES[key],
            labels_by_key[key][crop_indices].astype(np.int32),
        )


def _build_scene(
    *,
    scene_id: str,
    scans_root: Path,
    hierarchy_root: Path,
    processed_root: Path,
    output_root: Path,
    provider: ProposalProvider,
) -> Path:
    final_dir = output_root / scene_id
    final_manifest = final_dir / "source_manifest.json"
    if final_manifest.exists():
        audit_scene_source_manifest(final_manifest, expected_scene_id=scene_id)
        log.info("Reused valid source pack: %s", final_manifest)
        return final_manifest
    if final_dir.exists():
        raise RuntimeError(
            f"Incomplete source directory exists without a valid manifest: {final_dir}"
        )

    scene_dir = scans_root / scene_id
    hierarchy_dir = hierarchy_root / scene_id
    pack_dir = hierarchy_dir / "training_pack"
    processed_dir = processed_root / scene_id
    required = [
        scene_dir,
        pack_dir,
        processed_dir / "coord.npy",
        processed_dir / "instance.npy",
        pack_dir / "points.npy",
        pack_dir / "colors.npy",
        pack_dir / "normals.npy",
        pack_dir / "partfield_hierarchy_meta.json",
        hierarchy_dir / "partfield_gt_hierarchy_summary.json",
    ] + [pack_dir / LABEL_FILES[key] for key in GRANULARITY_KEYS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{scene_id}: missing source inputs: {missing}")

    points = np.asarray(np.load(pack_dir / "points.npy", allow_pickle=False))
    colors = np.asarray(np.load(pack_dir / "colors.npy", allow_pickle=False))
    normals = np.asarray(np.load(pack_dir / "normals.npy", allow_pickle=False))
    labels_by_key = {
        key: np.asarray(np.load(pack_dir / LABEL_FILES[key], allow_pickle=False))
        for key in GRANULARITY_KEYS
    }
    processed_points = np.asarray(
        np.load(processed_dir / "coord.npy", allow_pickle=False)
    )
    gt_instances = np.asarray(
        np.load(processed_dir / "instance.npy", allow_pickle=False)
    )
    hierarchy_meta = json.loads(
        (pack_dir / "partfield_hierarchy_meta.json").read_text(encoding="utf-8")
    )
    n_points = int(points.shape[0])
    if points.shape != (n_points, 3):
        raise ValueError(f"{scene_id}: invalid points shape {points.shape}")
    if colors.shape != (n_points, 3) or normals.shape != (n_points, 3):
        raise ValueError(f"{scene_id}: RGB/normal shape drift")
    if any(labels.shape != (n_points,) for labels in labels_by_key.values()):
        raise ValueError(f"{scene_id}: hierarchy label shape drift")
    points_aligned = bool(
        processed_points.shape == points.shape and np.array_equal(processed_points, points)
    )
    g08_exact_gt = _g08_matches_declared_gt(
        labels_g08=labels_by_key["g08"],
        gt_instances=gt_instances,
        hierarchy_meta=hierarchy_meta,
    )
    g02_nested = _nesting_violations(
        labels_by_key["g02"], labels_by_key["g05"]
    ) == 0
    g05_nested = _nesting_violations(
        labels_by_key["g05"], labels_by_key["g08"]
    ) == 0
    if not all((points_aligned, g08_exact_gt, g02_nested, g05_nested)):
        raise RuntimeError(
            f"{scene_id}: 3D source contract failed: points={points_aligned}, "
            f"g08_gt={g08_exact_gt}, g02_g05={g02_nested}, g05_g08={g05_nested}"
        )

    frame_override = provider.frame_ids_override(scene_id)
    candidate_frame_ids: list[str] | None = None
    selective_rgbd_record: dict[str, Any] | None = None
    if frame_override is None:
        candidate_frame_ids, selective_rgbd_record = _ensure_selective_rgbd(
            scene_dir
        )
    adapter = ScanNetSceneAdapter(scene_dir)
    selected_frames, invalid_poses, train_frames, heldout_frames = _select_frames(
        adapter,
        frame_ids_override=frame_override,
        candidate_frame_ids=candidate_frame_ids,
    )
    projected = _project_proposals(
        adapter=adapter,
        points=points,
        selected_frames=selected_frames,
        train_frame_ids=train_frames,
        provider=provider,
    )
    crop_indices, selected_crop, crop_candidates = _select_crop(
        points=points,
        projected=projected,
        labels_by_key=labels_by_key,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{scene_id}.", dir=output_root) as temp_name:
        work_dir = Path(temp_name) / "pack"
        work_dir.mkdir()
        _copy_or_save_scene_arrays(
            work_dir=work_dir,
            crop_indices=crop_indices,
            points=points,
            colors=colors,
            normals=normals,
            labels_by_key=labels_by_key,
        )
        raw_cache_audit = _write_raw_cache(
            path=work_dir / "raw_projection_cache.npz",
            crop_indices=crop_indices,
            projected=projected,
            num_points=n_points,
        )
        raw_overlap_preserved = bool(
            raw_cache_audit["projected_points_in_multiple_proposals"] > 0
        )
        checks = {
            "points_aligned": points_aligned,
            "crop_indices_valid": bool(
                crop_indices.ndim == 1
                and crop_indices.size == min(n_points, POINT_MAX)
                and np.unique(crop_indices).size == crop_indices.size
                and int(crop_indices.min()) >= 0
                and int(crop_indices.max()) < n_points
            ),
            "g08_exact_gt": g08_exact_gt,
            "g02_nested_in_g05": g02_nested,
            "g05_nested_in_g08": g05_nested,
            "raw_overlap_preserved": raw_overlap_preserved,
            "all_six_train_cells_nonempty": all(
                int(selected_crop["trainable_proposals"][cell]) > 0
                for cell in SOURCE_CELLS
            ),
            "all_six_heldout_cells_nonempty": all(
                int(selected_crop["heldout_trainable_proposals"][cell]) > 0
                for cell in SOURCE_CELLS
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{scene_id}: compact source checks failed: {checks}")

        artifacts = {
            "points": _array_record(work_dir / "points.npy", relative_to=work_dir),
            "colors": _array_record(work_dir / "colors.npy", relative_to=work_dir),
            "normals": _array_record(work_dir / "normals.npy", relative_to=work_dir),
            "crop_indices": _array_record(
                work_dir / "crop_indices.npy", relative_to=work_dir
            ),
            "labels_g02": _array_record(
                work_dir / LABEL_FILES["g02"], relative_to=work_dir
            ),
            "labels_g05": _array_record(
                work_dir / LABEL_FILES["g05"], relative_to=work_dir
            ),
            "labels_g08": _array_record(
                work_dir / LABEL_FILES["g08"], relative_to=work_dir
            ),
            "raw_projection_cache": _file_record(
                work_dir / "raw_projection_cache.npz", relative_to=work_dir
            ),
        }
        manifest = {
            "schema_version": SCENE_SOURCE_SCHEMA,
            "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
            "scene_id": scene_id,
            "source_cells": list(SOURCE_CELLS),
            "crop": {
                "selection": "valid-crop-first over sphere-crop seeds 42..49",
                "seed": int(selected_crop["seed"]),
                "indices": "crop_indices.npy",
                "num_points": int(crop_indices.size),
                "candidates": crop_candidates,
                "selected_audit": {
                    key: value
                    for key, value in selected_crop.items()
                    if key != "frames"
                },
            },
            "frames": {
                "selection": (
                    (
                        "64 integer-linspace candidates; reject invalid poses; "
                        "16 integer-linspace valid frames"
                    )
                    if frame_override is None
                    else (
                        "explicit existing-proposal frame order for audit/smoke only; "
                        "not eligible as a production training source"
                    )
                ),
                "physical": [frame.frame_id for frame in selected_frames],
                "train": train_frames,
                "heldout": heldout_frames,
                "invalid_pose_candidates": invalid_poses,
            },
            "counts": {
                "trainable_proposals": selected_crop["trainable_proposals"],
                "heldout_trainable_proposals": selected_crop[
                    "heldout_trainable_proposals"
                ],
                "raw_cache": raw_cache_audit,
            },
            "artifacts": artifacts,
            "provenance": {
                "scene_dir": str(scene_dir.resolve(strict=True)),
                "hierarchy_pack": str(pack_dir.resolve(strict=True)),
                "processed_scene": str(processed_dir.resolve(strict=True)),
                "hierarchy_summary": _file_record(
                    hierarchy_dir / "partfield_gt_hierarchy_summary.json"
                ),
                "hierarchy_meta": _file_record(
                    pack_dir / "partfield_hierarchy_meta.json"
                ),
                "processed_points": _file_record(processed_dir / "coord.npy"),
                "processed_instances": _file_record(processed_dir / "instance.npy"),
                "proposal_provider": provider.provenance(),
                "selective_rgbd": selective_rgbd_record,
            },
            "checks": checks,
        }
        manifest_path = work_dir / "source_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_scene_source_manifest(manifest_path, expected_scene_id=scene_id)
        work_dir.rename(final_dir)
    audit_scene_source_manifest(final_manifest, expected_scene_id=scene_id)
    log.info("Committed compact source pack: %s", final_manifest)
    return final_manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scene_ids = load_scene_ids(args.scene_ids_file)
    stop = len(scene_ids) if args.scene_stop is None else int(args.scene_stop)
    if not 0 <= int(args.scene_start) < stop <= len(scene_ids):
        raise ValueError(
            f"Invalid half-open scene range [{args.scene_start}, {stop}) "
            f"for {len(scene_ids)} scenes"
        )
    selected_scene_ids = scene_ids[int(args.scene_start) : stop]
    provider: ProposalProvider
    if args.proposal_input_root is None:
        provider = TeacherProposalProvider(device=args.device, model_cfg=args.model_cfg)
    else:
        provider = DirectoryProposalProvider(args.proposal_input_root)

    manifests: list[dict[str, Any]] = []
    for scene_id in selected_scene_ids:
        manifest_path = _build_scene(
            scene_id=scene_id,
            scans_root=args.scans_root,
            hierarchy_root=args.hierarchy_root,
            processed_root=args.processed_root,
            output_root=args.output_root,
            provider=provider,
        )
        manifests.append(
            {
                "scene_id": scene_id,
                "manifest": str(manifest_path.resolve(strict=True)),
                "sha256": sha256_file(manifest_path),
            }
        )
    chunk_manifest = args.output_root / (
        f"chunk_{int(args.scene_start):04d}_{stop:04d}.json"
    )
    chunk_manifest.write_text(
        json.dumps(
            {
                "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
                "scene_ids_file": _file_record(args.scene_ids_file),
                "scene_start": int(args.scene_start),
                "scene_stop": stop,
                "count": len(manifests),
                "proposal_provider": provider.provenance(),
                "scenes": manifests,
                "complete": len(manifests) == len(selected_scene_ids),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(chunk_manifest)


if __name__ == "__main__":
    main()
