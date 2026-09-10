#!/usr/bin/env python3
"""Write compact RGBN6 + GT3 + UnSAMv2 contrastive sources for Structured3D.

The exporter reads RGB, depth, calibration, and poses directly from the
official Structured3D ZIP archives.  It writes no extracted RGB-D frames:
each committed scene consists only of the crop tensors, three 3D hierarchy
labels, a compressed projected-UnSAM cache, and a checked manifest.
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

import numpy as np

from delimit3d.source.common.types import FrameRecord, VisibilityConfig
from delimit3d.source.datasets.structured3d.prepare import CAMERA_AXIS_CONVERSION
from delimit3d.source.datasets.structured3d.reader import Structured3DReader

from delimit3d.training.contracts import (
    EXPERIMENT_ID,
    GRANULARITY_KEYS,
    SCENE_SOURCE_SCHEMA,
    SOURCE_CELLS,
    audit_scene_source_manifest,
    frame_split_all_available,
    load_scene_ids,
    sha256_file,
)
from delimit3d.cli.prepare_sources import (
    DirectoryProposalProvider,
    POINT_MAX,
    _array_record,
    _copy_or_save_scene_arrays,
    _crop_audit,
    _file_record,
    _nesting_violations,
    _project_proposals,
    _select_crop,
    _write_raw_cache,
)


log = logging.getLogger("prepare_partfield_contrastive_structured3d_unsam3_sources")
LABEL_FILES = {"g02": "labels_g0.2.npy", "g05": "labels_g0.5.npy", "g08": "labels_g0.8.npy"}
RAW_SOURCE_SCHEMA = "structured3d_partfield_unsam3_raw/v1"
RAW_LEVEL_TAGS = {"g02": "0.2", "g05": "0.5", "g08": "0.8"}


def _scene_ids(path: Path) -> list[str]:
    return load_scene_ids(path)


class Structured3DRawProposalProvider(DirectoryProposalProvider):
    """Read only accepted, per-scene Structured3D raw UnSAMv2 packs."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._validated_manifests: dict[str, Path] = {}

    def validate_scene(self, *, scene_id: str, frame_ids: Sequence[str]) -> None:
        scene_root = self.root / scene_id
        manifest_path = scene_root / "source_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{scene_id}: accepted raw UnSAMv2 source manifest is missing: "
                f"{manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RAW_SOURCE_SCHEMA:
            raise RuntimeError(
                f"{manifest_path}: expected {RAW_SOURCE_SCHEMA!r}, got "
                f"{manifest.get('schema_version')!r}"
            )
        if manifest.get("dataset") != "structured3d":
            raise RuntimeError(f"{manifest_path}: not a Structured3D raw manifest")
        if manifest.get("source_passed") is not True:
            raise RuntimeError(f"{manifest_path}: raw source gate did not pass")

        selection = manifest.get("selection", {})
        manifest_frames = [str(value) for value in selection.get("physical", [])]
        expected_frames = [str(value) for value in frame_ids]
        if set(manifest_frames) != set(expected_frames):
            raise RuntimeError(
                f"{manifest_path}: physical frame set differs from archive selection; "
                f"manifest={len(manifest_frames)} selected={len(expected_frames)}"
            )

        levels = manifest.get("levels", {})
        for key, tag in RAW_LEVEL_TAGS.items():
            level = levels.get(key)
            if not isinstance(level, dict):
                raise RuntimeError(f"{manifest_path}: missing raw level {key}")
            rows = {
                str(row.get("frame_id")): row
                for row in level.get("frames", [])
                if isinstance(row, dict)
            }
            raw_dir = scene_root / f"raw_unsam_g{tag}"
            flat_dir = scene_root / f"flat_unsam_g{tag}"
            for frame_id in expected_frames:
                row = rows.get(frame_id)
                if row is None:
                    raise RuntimeError(
                        f"{manifest_path}: {key} lacks frame {frame_id}"
                    )
                raw_path = raw_dir / f"{frame_id}.npz"
                flat_path = flat_dir / f"{frame_id}.npy"
                if not raw_path.is_file() or not flat_path.is_file():
                    raise FileNotFoundError(
                        f"{scene_id}/{key}/{frame_id}: raw/flat proposal pair is "
                        f"incomplete ({raw_path}, {flat_path})"
                    )
        self._validated_manifests[scene_id] = manifest_path

    def proposals(self, **kwargs: Any):
        scene_id = str(kwargs["scene_id"])
        if scene_id not in self._validated_manifests:
            raise RuntimeError(
                f"{scene_id}: validate_scene must run before loading raw proposals"
            )
        return super().proposals(**kwargs)

    def provenance(self) -> dict[str, Any]:
        return {
            "mode": "existing_validated_raw_unsamv2_proposals",
            "root": str(self.root),
            "schema_version": RAW_SOURCE_SCHEMA,
            "frame_split": "compact source recomputes deterministic 90% train/10% heldout-validation",
        }

    def scene_provenance(self, scene_id: str) -> dict[str, Any]:
        manifest_path = self._validated_manifests[scene_id]
        return {"source_manifest": _file_record(manifest_path)}


@dataclass(frozen=True)
class _ArchiveFrame:
    room: str
    frame: str
    record: FrameRecord


class Structured3DArchiveAdapter:
    """Minimal SceneAdapter-compatible reader without persistent extraction."""

    def __init__(self, *, scene_id: str, reader: Structured3DReader) -> None:
        self.scene_id = scene_id
        self.reader = reader
        self._frames: list[_ArchiveFrame] | None = None
        self._by_id: dict[str, _ArchiveFrame] = {}

    def _base(self, value: _ArchiveFrame) -> str:
        return (
            f"Structured3D/{self.scene_id}/2D_rendering/{value.room}/"
            f"perspective/full/{value.frame}"
        )

    def list_frames(self) -> list[FrameRecord]:
        if self._frames is None:
            root = f"Structured3D/{self.scene_id}/2D_rendering"
            rows: list[_ArchiveFrame] = []
            for room in sorted(self.reader.listdir(root)):
                base = f"{root}/{room}/perspective/full"
                try:
                    frames = sorted(self.reader.listdir(base), key=lambda value: int(value))
                except ValueError:
                    frames = sorted(self.reader.listdir(base))
                for frame in frames:
                    prefix = f"{base}/{frame}"
                    required = (
                        f"{prefix}/rgb_rawlight.png",
                        f"{prefix}/depth.png",
                        f"{prefix}/camera_pose.txt",
                    )
                    if not all(path in self.reader.names_mapper for path in required):
                        continue
                    frame_id = f"{room}:{frame}"
                    record = FrameRecord(
                        frame_id=frame_id,
                        rgb_path=Path(f"archive://{prefix}/rgb_rawlight.png"),
                        depth_path=Path(f"archive://{prefix}/depth.png"),
                        pose_path=Path(f"archive://{prefix}/camera_pose.txt"),
                        intrinsics_path=Path(f"archive://{prefix}/camera_pose.txt"),
                    )
                    row = _ArchiveFrame(room=room, frame=frame, record=record)
                    rows.append(row)
                    self._by_id[frame_id] = row
            if not rows:
                raise RuntimeError(f"{self.scene_id}: no complete archive frames")
            self._frames = rows
        return [row.record for row in self._frames]

    def _frame(self, record: FrameRecord) -> _ArchiveFrame:
        if not self._by_id:
            self.list_frames()
        return self._by_id[record.frame_id]

    def _camera(self, record: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = self._frame(record)
        cam_r, cam_t, cam_f = self.reader.read_camera(f"{self._base(value)}/camera_pose.txt")
        if cam_f is None or np.asarray(cam_f).shape != (2,):
            raise ValueError(f"{self.scene_id}/{record.frame_id}: missing camera FOV")
        depth = self.reader.read_depth(f"{self._base(value)}/depth.png")
        height, width = depth.shape
        fx, fy = np.asarray(cam_f, dtype=np.float32)
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 2], intrinsic[1, 2] = width / 2.0, height / 2.0
        intrinsic[0, 0] = intrinsic[0, 2] / np.tan(fx)
        intrinsic[1, 1] = intrinsic[1, 2] / np.tan(fy)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = cam_r @ CAMERA_AXIS_CONVERSION.T
        pose[:3, 3] = cam_t
        return pose, intrinsic, depth

    def load_rgb(self, record: FrameRecord) -> np.ndarray:
        # Structured3DReader reverses BGR with ``[..., ::-1]``, which is a
        # negative-stride NumPy view.  Torchvision's UnSAMv2 transform cannot
        # ingest that view, so materialize a contiguous RGB array here.
        return np.ascontiguousarray(
            self.reader.read_color(f"{self._base(self._frame(record))}/rgb_rawlight.png")
        )

    def load_depth_m(self, record: FrameRecord) -> np.ndarray:
        return self._camera(record)[2].astype(np.float32) / 1000.0

    def load_pose_c2w(self, record: FrameRecord) -> np.ndarray:
        return self._camera(record)[0]

    def load_intrinsics(self, record: FrameRecord) -> np.ndarray:
        return self._camera(record)[1]

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=0.01,
            z_tolerance_m=1.0,
            depth_scale_to_m=1.0,
            depth_aligned_to_rgb=True,
        )


def _g08_matches_gt(
    labels_g08: np.ndarray, gt_ids: np.ndarray, hierarchy_meta: dict[str, Any]
) -> bool:
    if labels_g08.shape != gt_ids.shape:
        return False
    expected = np.full(labels_g08.shape, -1, dtype=np.int32)
    for row in hierarchy_meta.get("objects", []):
        expected[gt_ids == int(row["raw_gt_instance_id"])] = int(row["object_label_g08"])
    return bool(np.array_equal(labels_g08.astype(np.int32, copy=False), expected))


def _select_all_complete_valid_pose_frames(
    adapter: Structured3DArchiveAdapter,
) -> tuple[list[FrameRecord], list[dict[str, str]], list[str], list[str]]:
    """Keep every complete archive frame whose pose and calibration can be used."""

    candidates = adapter.list_frames()
    selected: list[FrameRecord] = []
    invalid: list[dict[str, str]] = []
    for frame in candidates:
        try:
            adapter.load_pose_c2w(frame)
            adapter.load_intrinsics(frame)
        except Exception as exc:
            invalid.append({"frame_id": frame.frame_id, "reason": repr(exc)})
        else:
            selected.append(frame)
    if not selected:
        raise RuntimeError(f"{adapter.scene_id}: no archive frame has a valid pose and calibration")
    train_ids, heldout_ids = frame_split_all_available([frame.frame_id for frame in selected])
    return selected, invalid, train_ids, heldout_ids


def _build_scene(
    *,
    scene_id: str,
    scans_root: Path,
    hierarchy_root: Path,
    output_root: Path,
    reader: Structured3DReader,
    provider: Structured3DRawProposalProvider,
    full_scene: bool = False,
) -> Path:
    final_dir = output_root / scene_id
    final_manifest = final_dir / "source_manifest.json"
    if final_manifest.exists():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        existing_provider = existing.get("provenance", {}).get("proposal_provider", {})
        if existing_provider.get("mode") != "existing_validated_raw_unsamv2_proposals":
            raise RuntimeError(
                f"{scene_id}: existing compact pack was not built from the accepted "
                f"raw UnSAMv2 provider; use a fresh output root: {final_dir}"
            )
        provider.validate_scene(
            scene_id=scene_id,
            frame_ids=[str(value) for value in existing.get("frames", {}).get("physical", [])],
        )
        audit_scene_source_manifest(final_manifest, expected_scene_id=scene_id)
        return final_manifest
    if final_dir.exists():
        raise RuntimeError(f"{scene_id}: incomplete output directory: {final_dir}")

    scene_dir = scans_root / scene_id
    pack_dir = hierarchy_root / scene_id / "training_pack"
    required = [
        scene_dir / "training_pack" / "points.npy",
        scene_dir / "gt_instance_ids.npy",
        pack_dir / "points.npy",
        pack_dir / "colors.npy",
        pack_dir / "normals.npy",
        pack_dir / "partfield_hierarchy_meta.json",
    ] + [pack_dir / LABEL_FILES[key] for key in GRANULARITY_KEYS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{scene_id}: missing inputs: {missing}")

    points = np.asarray(np.load(pack_dir / "points.npy", allow_pickle=False))
    colors = np.asarray(np.load(pack_dir / "colors.npy", allow_pickle=False))
    normals = np.asarray(np.load(pack_dir / "normals.npy", allow_pickle=False))
    labels = {key: np.asarray(np.load(pack_dir / LABEL_FILES[key], allow_pickle=False)) for key in GRANULARITY_KEYS}
    base_points = np.asarray(np.load(scene_dir / "training_pack" / "points.npy", allow_pickle=False))
    gt_ids = np.asarray(np.load(scene_dir / "gt_instance_ids.npy", allow_pickle=False))
    hierarchy_meta = json.loads((pack_dir / "partfield_hierarchy_meta.json").read_text())
    n_points = int(points.shape[0])
    points_aligned = bool(base_points.shape == points.shape and np.array_equal(base_points, points))
    g08_exact_gt = _g08_matches_gt(labels["g08"], gt_ids, hierarchy_meta)
    g02_nested = _nesting_violations(labels["g02"], labels["g05"]) == 0
    g05_nested = _nesting_violations(labels["g05"], labels["g08"]) == 0
    if not (
        points.shape == (n_points, 3)
        and colors.shape == (n_points, 3)
        and normals.shape == (n_points, 3)
        and all(value.shape == (n_points,) for value in labels.values())
        and points_aligned and g08_exact_gt and g02_nested and g05_nested
    ):
        raise RuntimeError(f"{scene_id}: 3D pack contract failed")

    adapter = Structured3DArchiveAdapter(scene_id=scene_id, reader=reader)
    selected_frames, invalid_poses, train_frames, heldout_frames = (
        _select_all_complete_valid_pose_frames(adapter)
    )
    provider.validate_scene(
        scene_id=scene_id,
        frame_ids=[frame.frame_id for frame in selected_frames],
    )
    projected = _project_proposals(
        adapter=adapter,
        points=points,
        selected_frames=selected_frames,
        train_frame_ids=train_frames,
        provider=provider,
    )
    if full_scene:
        crop_indices = np.arange(n_points, dtype=np.int64)
        selected_crop = _crop_audit(
            crop_indices=crop_indices,
            projected=projected,
            labels_by_key=labels,
            num_points=n_points,
            min_train_frames_per_level=1,
            min_heldout_frames_per_level=1,
        )
        selected_crop["seed"] = None
        selected_crop["valid_pool_available"] = bool(selected_crop["valid"])
        selected_crop["selection_mode"] = "full_scene"
        crop_candidates = [
            {key: value for key, value in selected_crop.items() if key != "frames"}
        ]
        expected_crop_size = n_points
        crop_mode = "full_scene"
        crop_seed: int | None = None
    else:
        crop_indices, selected_crop, crop_candidates = _select_crop(
            points=points,
            projected=projected,
            labels_by_key=labels,
            # Structured3D has variable room/frame counts; unlike ScanNet's fixed
            # 16-frame contract, a valid scene may have only one train and one
            # validation view per granularity.
            min_train_frames_per_level=1,
            min_heldout_frames_per_level=1,
        )
        expected_crop_size = min(n_points, POINT_MAX)
        crop_mode = "sphere_50k"
        crop_seed = int(selected_crop["seed"])

    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{scene_id}.", dir=output_root) as temp_name:
        work_dir = Path(temp_name) / "pack"
        work_dir.mkdir()
        _copy_or_save_scene_arrays(
            work_dir=work_dir, crop_indices=crop_indices, points=points,
            colors=colors, normals=normals, labels_by_key=labels,
        )
        raw_cache_audit = _write_raw_cache(
            path=work_dir / "raw_projection_cache.npz", crop_indices=crop_indices,
            projected=projected, num_points=n_points,
        )
        checks = {
            "points_aligned": points_aligned,
            "crop_indices_valid": bool(
                crop_indices.ndim == 1
                and crop_indices.size == expected_crop_size
                and np.unique(crop_indices).size == crop_indices.size
                and int(crop_indices.min()) >= 0
                and int(crop_indices.max()) < n_points
            ),
            "g08_exact_gt": g08_exact_gt,
            "g02_nested_in_g05": g02_nested,
            "g05_nested_in_g08": g05_nested,
            "raw_overlap_preserved": bool(raw_cache_audit["projected_points_in_multiple_proposals"] > 0),
            "all_six_train_cells_nonempty": all(int(selected_crop["trainable_proposals"][cell]) > 0 for cell in SOURCE_CELLS),
            "all_six_heldout_cells_nonempty": (
                not heldout_frames
                or all(int(selected_crop["heldout_trainable_proposals"][cell]) > 0 for cell in SOURCE_CELLS)
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"{scene_id}: source checks failed: {checks}; "
                f"trainable={selected_crop['trainable_proposals']}; "
                f"heldout={selected_crop['heldout_trainable_proposals']}; "
                f"crop_candidates={crop_candidates}"
            )
        artifacts = {
            "points": _array_record(work_dir / "points.npy", relative_to=work_dir),
            "colors": _array_record(work_dir / "colors.npy", relative_to=work_dir),
            "normals": _array_record(work_dir / "normals.npy", relative_to=work_dir),
            "crop_indices": _array_record(work_dir / "crop_indices.npy", relative_to=work_dir),
            "labels_g02": _array_record(work_dir / LABEL_FILES["g02"], relative_to=work_dir),
            "labels_g05": _array_record(work_dir / LABEL_FILES["g05"], relative_to=work_dir),
            "labels_g08": _array_record(work_dir / LABEL_FILES["g08"], relative_to=work_dir),
            "raw_projection_cache": _file_record(work_dir / "raw_projection_cache.npz", relative_to=work_dir),
        }
        manifest = {
            "schema_version": SCENE_SOURCE_SCHEMA,
            "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
            "scene_id": scene_id,
            "source_cells": list(SOURCE_CELLS),
            "dataset": "structured3d",
            "crop": {
                "mode": crop_mode,
                "indices": "crop_indices.npy",
                "num_points": int(crop_indices.size),
                "seed": crop_seed,
                "candidates": crop_candidates,
                "selected_audit": {
                    key: value for key, value in selected_crop.items() if key != "frames"
                },
            },
            "frames": {
                "selection_mode": "all_complete_valid_pose",
                "selection": "all complete archive RGB/depth/pose frames with usable pose and calibration; deterministic 90% train/10% heldout-validation split when possible",
                "archive_complete_candidates": len(selected_frames) + len(invalid_poses),
                "physical": [frame.frame_id for frame in selected_frames],
                "train": train_frames,
                "heldout": heldout_frames,
                "invalid_pose_candidates": invalid_poses,
            },
            "counts": {"trainable_proposals": selected_crop["trainable_proposals"], "heldout_trainable_proposals": selected_crop["heldout_trainable_proposals"], "raw_cache": raw_cache_audit},
            "artifacts": artifacts,
            "provenance": {
                "scans_scene": str(scene_dir.resolve()),
                "hierarchy_pack": str(pack_dir.resolve()),
                "raw_rgbd": "Structured3D perspective_full ZIP archives read in place",
                "proposal_provider": provider.provenance(),
                "proposal_scene": provider.scene_provenance(scene_id),
            },
            "checks": checks,
        }
        manifest_path = work_dir / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        audit_scene_source_manifest(manifest_path, expected_scene_id=scene_id)
        work_dir.rename(final_dir)
    return final_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--scene-ids-file", type=Path)
    source_group.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-stop", type=int, default=None)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--hierarchy-root", type=Path, required=True)
    parser.add_argument("--raw-zips-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--proposal-input-root",
        type=Path,
        required=True,
        help=(
            "Accepted Structured3D raw UnSAMv2 root containing "
            "<scene_id>/source_manifest.json and raw/flat proposal files."
        ),
    )
    parser.add_argument(
        "--exclude-scene-ids-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/splits/structured3d_unsamv2_excluded_scenes.list",
        help="Comment-tolerant scene list excluded from the raw/source dataset.",
    )
    parser.add_argument(
        "--full-scene",
        action="store_true",
        help="Keep every hierarchy point instead of a 50k sphere crop.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Commit successful scenes in this chunk even if a later scene fails occupancy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scene_ids = (
        _scene_ids(args.scene_ids_file)
        if args.scene_ids_file is not None
        else sorted(
            path.name
            for path in args.scans_root.iterdir()
            if path.is_dir() and path.name.startswith("scene_")
        )
    )
    excluded_scene_ids = set(load_scene_ids(args.exclude_scene_ids_file))
    scene_ids = [scene_id for scene_id in scene_ids if scene_id not in excluded_scene_ids]
    if not scene_ids:
        raise ValueError("Scene ID list is empty after exclusions")
    stop = len(scene_ids) if args.scene_stop is None else int(args.scene_stop)
    if not 0 <= args.scene_start < stop <= len(scene_ids):
        raise ValueError("Invalid half-open scene range")
    zip_paths = sorted(args.raw_zips_dir.glob("Structured3D_perspective_full_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No perspective archives under {args.raw_zips_dir}")
    reader = Structured3DReader([str(path) for path in zip_paths])
    provider = Structured3DRawProposalProvider(args.proposal_input_root)
    results = []
    failures: list[dict[str, str]] = []
    for scene_id in scene_ids[args.scene_start:stop]:
        try:
            manifest = _build_scene(
                scene_id=scene_id, scans_root=args.scans_root,
                hierarchy_root=args.hierarchy_root, output_root=args.output_root,
                reader=reader, provider=provider, full_scene=bool(args.full_scene),
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            log.exception("Failed %s", scene_id)
            failures.append({"scene_id": scene_id, "error": repr(exc)})
            continue
        results.append({"scene_id": scene_id, "manifest": str(manifest), "sha256": sha256_file(manifest)})
        log.info("Committed %s", manifest)
    report = args.output_root / f"chunk_{args.scene_start:05d}_{stop:05d}.json"
    payload = {
        "schema_version": "structured3d_gt3_unsam3_chunk/v1",
        "scene_start": args.scene_start,
        "scene_stop": stop,
        "crop_mode": "full_scene" if args.full_scene else "sphere_50k",
        "complete": len(results) == stop - args.scene_start and not failures,
        "excluded_scene_ids": sorted(excluded_scene_ids),
        "scenes": results,
        "failures": failures,
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(report)
    if failures and not args.continue_on_error:
        raise RuntimeError(f"{len(failures)} scene(s) failed in this chunk")


if __name__ == "__main__":
    main()
