#!/usr/bin/env python3
"""Export the ScanNet/PartField-style UnSAMv2 proposals for Structured3D.

This is deliberately a 2D-only exporter.  It runs the same in-memory
``UnSAMv2Teacher._mask_generator.generate(image, gra=...)`` path used by the
PartField contrastive source preparation, at g02/g05/g08.  It does not run the
projection, SVD, HDBSCAN, or any 3D hierarchy checks, so it can cover
Structured3D scenes whose optional 3D training packs are absent.

For every complete valid-pose frame and each granularity, the exporter keeps
both representations needed for auditing and downstream use:

* ``raw_unsam_g*.npz``: bit-packed overlapping proposals with teacher metadata;
* ``flat_unsam_g*.npy``: the exact area-ordered first-mask-wins 2D derivative.

Files are written atomically and reused on restart.  A scene manifest is
written only after all three granularities have the exact selected frame set.
"""

from __future__ import annotations

from delimit3d.identity import project_name, project_slug

import argparse
import json
import logging
import os
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np

from delimit3d.source.common.types import FrameRecord, VisibilityConfig
from delimit3d.source.core.teacher.unsamv2 import UnSAMv2Teacher
from delimit3d.training.contracts import (
    GRANULARITY_KEYS,
    frame_split_all_available,
    load_scene_ids,
    sha256_file,
)
from delimit3d.data.unsam_proposals import (
    flatten_proposals,
    load_proposal_set,
    proposal_overlap_statistics,
    proposals_from_teacher_output,
    save_proposal_set,
)


log = logging.getLogger("prepare_partfield_structured3d_unsam3_raw")
SCHEMA_VERSION = "structured3d_partfield_unsam3_raw/v1"
GRANULARITIES = {"g02": 0.2, "g05": 0.5, "g08": 0.8}
# The official perspective_full archives are contiguous scene ranges.  Keeping
# this small index in code avoids making every GPU task scan all 17 multi-GB
# ZIP central directories just to discover which archive owns its four scenes.
ARCHIVE_SCENE_STARTS = (
    (0, "00"),
    (200, "01"),
    (400, "02"),
    (600, "03"),
    (800, "04"),
    (1000, "05"),
    (1200, "06"),
    (1400, "07"),
    (1800, "08"),
    (2000, "10"),
    (2200, "11"),
    (2400, "12"),
    (2600, "13"),
    (2800, "14"),
    (3000, "15"),
    (3200, "16"),
    (3400, "17"),
)


class Structured3DReader:
    """Minimal ZIP-backed reader, avoiding the heavy Structured3D package init."""

    def __init__(self, files: str | list[str]):
        if isinstance(files, str):
            files = [files]
        self.readers = [zipfile.ZipFile(path, "r") for path in files]
        self.names_mapper: dict[str, int] = {}
        for index, reader in enumerate(self.readers):
            for name in reader.namelist():
                self.names_mapper[name] = index

    def listdir(self, dir_name: str) -> list[str]:
        dir_name = dir_name.lstrip(os.path.sep).rstrip(os.path.sep)
        prefix = dir_name + os.path.sep
        values = {
            name.replace(prefix, "", 1).split(os.path.sep)[0]
            for name in self.names_mapper
            if name.startswith(prefix)
        }
        values.discard("")
        return sorted(values)

    def read(self, file_name: str) -> bytes:
        return self.readers[self.names_mapper[file_name]].read(file_name)

    def read_camera(self, camera_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        z2y_top_m = np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        )
        cam_extr = np.fromstring(self.read(camera_path), dtype=np.float32, sep=" ")
        cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
        if cam_extr.shape[0] > 3:
            cam_front, cam_up = cam_extr[3:6], cam_extr[6:9]
            cam_n = np.cross(cam_front, cam_up)
            cam_r = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
            cam_r = np.matmul(z2y_top_m, cam_r)
            cam_f = cam_extr[9:11]
        else:
            cam_r = np.eye(3, dtype=np.float32)
            cam_f = None
        return cam_r, cam_t, cam_f

    def read_depth(self, depth_path: str) -> np.ndarray:
        data = np.frombuffer(self.read(depth_path), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    def read_color(self, color_path: str) -> np.ndarray:
        data = np.frombuffer(self.read(color_path), np.uint8)
        color = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]
        return color


@dataclass(frozen=True)
class _ArchiveFrame:
    room: str
    frame: str
    record: FrameRecord


class Structured3DArchiveAdapter:
    """SceneAdapter subset needed by the PartField 2D proposal path."""

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
        cam_r, cam_t, cam_f = self.reader.read_camera(
            f"{self._base(value)}/camera_pose.txt"
        )
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
        camera_axis_conversion = np.array(
            [[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float32
        )
        pose[:3, :3] = cam_r @ camera_axis_conversion.T
        pose[:3, 3] = cam_t
        return pose, intrinsic, depth

    def load_rgb(self, record: FrameRecord) -> np.ndarray:
        return np.ascontiguousarray(
            self.reader.read_color(f"{self._base(self._frame(record))}/rgb_rawlight.png")
        )

    def load_depth_m(self, record: FrameRecord) -> np.ndarray:
        return self._camera(record)[2].astype(np.float32) / 1000.0

    def load_pose_c2w(self, record: FrameRecord) -> np.ndarray:
        return self._camera(record)[0]

    def load_intrinsics(self, record: FrameRecord) -> np.ndarray:
        return self._camera(record)[1]


def _select_all_complete_valid_pose_frames(
    adapter: Structured3DArchiveAdapter,
) -> tuple[list[FrameRecord], list[dict[str, str]], list[str], list[str]]:
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
        raise RuntimeError(f"{adapter.scene_id}: no archive frame has valid calibration")
    train, heldout = frame_split_all_available(
        [frame.frame_id for frame in selected]
    )
    return selected, invalid, train, heldout


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(path)


class TeacherProposalProvider:
    def __init__(self, *, device: str, model_cfg: str | None) -> None:
        import torch

        self.torch = torch
        print("Loading UnSAMv2 teacher", flush=True)
        self.teacher = UnSAMv2Teacher(
            device=device,
            model_cfg=model_cfg,
            overwrite=False,
            debug_first_n_frames=0,
        )
        self.teacher._ensure_loaded()
        print("UnSAMv2 teacher ready", flush=True)

    def proposals(
        self,
        *,
        image: np.ndarray,
        granularity: float,
    ) -> tuple[Any, bool]:
        used_fallback = False
        with self.torch.inference_mode(), self.teacher._autocast_context():
            masks_data = self.teacher._mask_generator.generate(image, gra=granularity)
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
        return {
            "mode": "generated_in_memory",
            "teacher": "UnSAMv2Teacher",
            "model_cfg": self.teacher.model_cfg,
            "checkpoint": str(self.teacher.checkpoint_path),
        }


def _scene_ids(scans_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in scans_root.iterdir()
        if path.is_dir() and path.name.startswith("scene_")
    )


def _archive_path_for_scene(
    *, scene_id: str, archives_by_code: dict[str, Path]
) -> Path:
    try:
        scene_number = int(scene_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid Structured3D scene ID: {scene_id}") from exc
    selected_code: str | None = None
    for start, code in ARCHIVE_SCENE_STARTS:
        if scene_number >= start:
            selected_code = code
        else:
            break
    if selected_code is None or selected_code not in archives_by_code:
        raise FileNotFoundError(
            f"No perspective archive mapping for {scene_id}; "
            f"available codes={sorted(archives_by_code)}"
        )
    return archives_by_code[selected_code]


def _validate_pair(raw_path: Path, flat_path: Path) -> tuple[Any, np.ndarray]:
    proposals = load_proposal_set(raw_path)
    expected_flat = flatten_proposals(proposals)
    saved_flat = np.load(flat_path, allow_pickle=False)
    if saved_flat.dtype != expected_flat.dtype:
        raise ValueError(
            f"Flat dtype mismatch for {flat_path}: "
            f"{saved_flat.dtype} != {expected_flat.dtype}"
        )
    if not np.array_equal(saved_flat, expected_flat):
        raise ValueError(
            f"Flat mask is not the exact derivative of {raw_path}: {flat_path}"
        )
    return proposals, expected_flat


def _relative_pair_records(
    *, output_dir: Path, frame_id: str, granularity: float
) -> tuple[Path, Path]:
    tag = format(float(granularity), "g")
    raw_path = output_dir / f"raw_unsam_g{tag}" / f"{frame_id}.npz"
    flat_path = output_dir / f"flat_unsam_g{tag}" / f"{frame_id}.npy"
    return raw_path, flat_path


def _scene_complete(
    *, manifest_path: Path, expected_frame_ids: list[str]
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    if not bool(manifest.get("source_passed")):
        return False
    expected_raw = sorted(f"{frame_id}.npz" for frame_id in expected_frame_ids)
    expected_flat = sorted(f"{frame_id}.npy" for frame_id in expected_frame_ids)
    for key in GRANULARITY_KEYS:
        value = GRANULARITIES[key]
        raw_dir = manifest_path.parent / f"raw_unsam_g{format(value, 'g')}"
        flat_dir = manifest_path.parent / f"flat_unsam_g{format(value, 'g')}"
        raw_names = sorted(path.name for path in raw_dir.glob("*.npz"))
        flat_names = sorted(path.name for path in flat_dir.glob("*.npy"))
        if raw_names != expected_raw or flat_names != expected_flat:
            return False
    return True


def _export_scene(
    *,
    scene_id: str,
    reader: Structured3DReader,
    output_root: Path,
    provider: TeacherProposalProvider,
) -> Path:
    scene_dir = output_root / scene_id
    manifest_path = scene_dir / "source_manifest.json"
    adapter = Structured3DArchiveAdapter(scene_id=scene_id, reader=reader)
    selected_frames, invalid_poses, train_frames, heldout_frames = (
        _select_all_complete_valid_pose_frames(adapter)
    )
    frame_ids = [frame.frame_id for frame in selected_frames]
    if _scene_complete(manifest_path=manifest_path, expected_frame_ids=frame_ids):
        log.info("Reusing complete %s", manifest_path)
        return manifest_path

    scene_dir.mkdir(parents=True, exist_ok=True)
    for key in GRANULARITY_KEYS:
        value = GRANULARITIES[key]
        (scene_dir / f"raw_unsam_g{format(value, 'g')}").mkdir(exist_ok=True)
        (scene_dir / f"flat_unsam_g{format(value, 'g')}").mkdir(exist_ok=True)

    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in GRANULARITY_KEYS}
    for frame_index, frame in enumerate(selected_frames):
        image: np.ndarray | None = None
        split = "train" if frame.frame_id in set(train_frames) else "heldout"
        for key in GRANULARITY_KEYS:
            granularity = GRANULARITIES[key]
            raw_path, flat_path = _relative_pair_records(
                output_dir=scene_dir,
                frame_id=frame.frame_id,
                granularity=granularity,
            )
            reused = False
            used_fallback = False
            if raw_path.is_file() and flat_path.is_file():
                proposals, _ = _validate_pair(raw_path, flat_path)
                reused = True
            elif raw_path.is_file():
                proposals = load_proposal_set(raw_path)
                _atomic_save_npy(flat_path, flatten_proposals(proposals))
                proposals, _ = _validate_pair(raw_path, flat_path)
                reused = True
            else:
                if image is None:
                    image = adapter.load_rgb(frame)
                proposals, used_fallback = provider.proposals(
                    image=image,
                    granularity=granularity,
                )
                if proposals.num_proposals <= 0:
                    raise RuntimeError(
                        f"{scene_id}/{frame.frame_id}/{key}: no proposals"
                    )
                save_proposal_set(raw_path, proposals)
                _atomic_save_npy(flat_path, flatten_proposals(proposals))
                proposals, _ = _validate_pair(raw_path, flat_path)

            stats = proposal_overlap_statistics(proposals)
            rows[key].append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame.frame_id,
                    "split": split,
                    "reused": reused,
                    "used_relaxed_fallback": used_fallback,
                    "image_shape": list(proposals.image_shape),
                    **stats,
                    "raw_file": str(raw_path.relative_to(scene_dir)),
                    "flat_file": str(flat_path.relative_to(scene_dir)),
                }
            )
            log.info(
                "%s frame=%s %s raw=%d flat=%d overlap_pixels=%d%s",
                scene_id,
                frame.frame_id,
                key,
                stats["num_raw_proposals"],
                stats["num_flat_proposals"],
                stats["overlap_pixels"],
                " [reused]" if reused else "",
            )

    checks: dict[str, bool] = {}
    expected_raw_names = sorted(f"{frame_id}.npz" for frame_id in frame_ids)
    expected_flat_names = sorted(f"{frame_id}.npy" for frame_id in frame_ids)
    for key in GRANULARITY_KEYS:
        value = GRANULARITIES[key]
        raw_dir = scene_dir / f"raw_unsam_g{format(value, 'g')}"
        flat_dir = scene_dir / f"flat_unsam_g{format(value, 'g')}"
        checks[f"{key}_exact_raw_frame_set"] = sorted(
            path.name for path in raw_dir.glob("*.npz")
        ) == expected_raw_names
        checks[f"{key}_exact_flat_frame_set"] = sorted(
            path.name for path in flat_dir.glob("*.npy")
        ) == expected_flat_names
        checks[f"{key}_all_frames_nonempty"] = all(
            row["num_raw_proposals"] > 0 for row in rows[key]
        )
        checks[f"{key}_raw_overlap_preserved"] = any(
            row["overlap_pixels"] > 0 for row in rows[key]
        )
    expected_splits = {"train", "heldout"} if heldout_frames else {"train"}
    checks["train_and_heldout_present"] = set(
        row["split"] for key in GRANULARITY_KEYS for row in rows[key]
    ) == expected_splits
    checks["three_granularities_present"] = all(
        len(rows[key]) == len(frame_ids) for key in GRANULARITY_KEYS
    )
    source_passed = all(checks.values())
    if not source_passed:
        raise RuntimeError(f"{scene_id}: raw UnSAM checks failed: {checks}")

    levels: dict[str, Any] = {}
    for key in GRANULARITY_KEYS:
        value = GRANULARITIES[key]
        levels[key] = {
            "granularity": value,
            "raw_dir": f"raw_unsam_g{format(value, 'g')}",
            "flat_dir": f"flat_unsam_g{format(value, 'g')}",
            "totals": {
                "raw_proposals": int(sum(row["num_raw_proposals"] for row in rows[key])),
                "flat_proposals": int(sum(row["num_flat_proposals"] for row in rows[key])),
                "raw_mask_pixels": int(sum(row["raw_mask_pixels"] for row in rows[key])),
                "union_pixels": int(sum(row["union_pixels"] for row in rows[key])),
                "overlap_pixels": int(sum(row["overlap_pixels"] for row in rows[key])),
                "relaxed_fallback_frames": int(
                    sum(row["used_relaxed_fallback"] for row in rows[key])
                ),
            },
            "frames": rows[key],
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_passed": source_passed,
        "scene_id": scene_id,
        "dataset": "structured3d",
        "selection": {
            "mode": "all_complete_valid_pose",
            "rule": (
                "all complete archive RGB/depth/pose frames with usable pose and "
                "calibration; deterministic 90% train/10% heldout-validation split "
                "when possible"
            ),
            "archive_complete_candidates": len(selected_frames) + len(invalid_poses),
            "physical": frame_ids,
            "train": train_frames,
            "heldout": heldout_frames,
            "invalid_pose_candidates": invalid_poses,
        },
        "teacher": provider.provenance(),
        "serialization": {
            "raw": "bit-packed overlapping bool proposals with teacher metadata",
            "flat": "area-ordered first-mask-wins int32 derivative",
        },
        "levels": levels,
        "checks": checks,
        "inputs": {
            "raw_rgbd": "Structured3D perspective_full ZIP archives read in place",
            "archive_count": len(reader.readers),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log.info(
        "Committed %s (%d frames, %d bytes)",
        manifest_path,
        len(frame_ids),
        manifest_path.stat().st_size,
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--scene-ids-file", type=Path)
    source_group.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-stop", type=int, default=None)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--raw-zips-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-scene-ids-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/splits/structured3d_unsamv2_excluded_scenes.list",
        help="Comment-tolerant scene list excluded from the raw/source dataset.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-cfg", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.scene_ids_file is not None:
        scene_ids = load_scene_ids(args.scene_ids_file)
    else:
        scene_ids = _scene_ids(args.scans_root)
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Scene ID list is empty or contains duplicates")
    excluded_scene_ids = set(load_scene_ids(args.exclude_scene_ids_file))
    scene_ids = [scene_id for scene_id in scene_ids if scene_id not in excluded_scene_ids]
    if not scene_ids:
        raise ValueError("Scene ID list is empty after exclusions")
    stop = len(scene_ids) if args.scene_stop is None else int(args.scene_stop)
    if not 0 <= args.scene_start < stop <= len(scene_ids):
        raise ValueError("Invalid half-open scene range")

    raw_zip_paths = sorted(
        args.raw_zips_dir.glob("Structured3D_perspective_full_*.zip")
    )
    if not raw_zip_paths:
        raise FileNotFoundError(f"No perspective archives under {args.raw_zips_dir}")
    archives_by_code = {
        path.stem.rsplit("_", 1)[-1]: path for path in raw_zip_paths
    }
    selected_scene_ids = scene_ids[args.scene_start:stop]
    groups: OrderedDict[Path, list[str]] = OrderedDict()
    for scene_id in selected_scene_ids:
        archive_path = _archive_path_for_scene(
            scene_id=scene_id, archives_by_code=archives_by_code
        )
        groups.setdefault(archive_path, []).append(scene_id)

    args.output_root.mkdir(parents=True, exist_ok=True)
    provider = TeacherProposalProvider(device=args.device, model_cfg=args.model_cfg)
    result_by_scene: dict[str, dict[str, str]] = {}
    for archive_path, grouped_scene_ids in groups.items():
        log.info(
            "Opening archive %s for %d scene(s)",
            archive_path.name,
            len(grouped_scene_ids),
        )
        reader = Structured3DReader([str(archive_path)])
        for scene_id in grouped_scene_ids:
            manifest = _export_scene(
                scene_id=scene_id,
                reader=reader,
                output_root=args.output_root,
                provider=provider,
            )
            result_by_scene[scene_id] = {
                "scene_id": scene_id,
                "manifest": str(manifest),
                "sha256": sha256_file(manifest),
            }
    results = [result_by_scene[scene_id] for scene_id in selected_scene_ids]
    report = args.output_root / f"chunk_{args.scene_start:05d}_{stop:05d}.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "structured3d_partfield_unsam3_chunk/v1",
                "scene_start": args.scene_start,
                "scene_stop": stop,
                "complete": len(results) == stop - args.scene_start,
                "excluded_scene_ids": sorted(excluded_scene_ids),
                "scenes": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
