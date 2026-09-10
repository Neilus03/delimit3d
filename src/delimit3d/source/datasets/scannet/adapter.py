from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from delimit3d.source.common.types import FrameRecord, GeometryRecord, VisibilityConfig
from delimit3d.source.datasets.base import SceneAdapter
from delimit3d.source.datasets.scannet.benchmark import (
    parse_scannet_eval_benchmarks,
    primary_scannet_eval_benchmark,
)
from delimit3d.source.datasets.scannet.evaluation import ScanNetEvaluationHooks
from delimit3d.source.datasets.scannet.gt import load_scannet_gt_instance_ids
from delimit3d.source.datasets.scannet.metadata import (
    DEFAULT_DEPTH_SCALE_TO_M,
    DEFAULT_VISIBILITY_MIN_DEPTH_M,
    DEFAULT_VISIBILITY_Z_TOLERANCE_M,
    GEOMETRY_SUFFIX,
)
from delimit3d.source.datasets.scannet.prepare import extract_rgbd, is_rgbd_prepared


class ScanNetSceneAdapter(SceneAdapter):
    def __init__(
        self,
        scene_root: Path,
        eval_benchmarks: list[str] | tuple[str, ...] | str | None = None,
    ):
        super().__init__(scene_root=scene_root)
        self.eval_benchmarks = parse_scannet_eval_benchmarks(eval_benchmarks)
        self.eval_benchmark = primary_scannet_eval_benchmark(self.eval_benchmarks)
        self._frames_cache: list[FrameRecord] | None = None

    @property
    def dataset_name(self) -> str:
        return "scannet"

    def prepare(self) -> None:
        if is_rgbd_prepared(self.scene_root):
            print(f"Scene {self.scene_id} is already prepared.")
            return
        extract_rgbd(self.scene_root)

    def list_frames(self) -> list[FrameRecord]:
        if self._frames_cache is not None:
            return list(self._frames_cache)

        color_dir = self.scene_root / "color"
        depth_dir = self.scene_root / "depth"
        pose_dir = self.scene_root / "pose"
        intrinsics_path = self.scene_root / "intrinsic" / "intrinsic_color.txt"

        if not color_dir.is_dir():
            raise FileNotFoundError(
                f"Missing color directory at {color_dir}. "
                "Call adapter.prepare() first."
            )
        if not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Missing depth directory at {depth_dir}. "
                "Call adapter.prepare() first."
            )
        if not pose_dir.is_dir():
            raise FileNotFoundError(
                f"Missing pose directory at {pose_dir}. "
                "Call adapter.prepare() first."
            )
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"Missing intrinsics file at {intrinsics_path}. "
                "Call adapter.prepare() first."
            )

        color_files = {
            p.stem: p
            for p in color_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png"}
        }
        depth_frame_ids = {p.stem for p in depth_dir.iterdir() if p.suffix == ".png"}
        pose_frame_ids = {p.stem for p in pose_dir.iterdir() if p.suffix == ".txt"}

        frame_ids = sorted(color_files.keys(), key=lambda x: int(x))

        frames = [
            FrameRecord(
                frame_id=frame_id,
                rgb_path=color_files[frame_id],
                depth_path=depth_dir / f"{frame_id}.png",
                pose_path=pose_dir / f"{frame_id}.txt",
                intrinsics_path=intrinsics_path,
            )
            for frame_id in frame_ids
            if frame_id in depth_frame_ids and frame_id in pose_frame_ids
        ]

        self._frames_cache = frames
        return list(frames)

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        return np.array(Image.open(frame.rgb_path).convert("RGB"))

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        depth = np.array(Image.open(frame.depth_path), dtype=np.float32)
        return depth * DEFAULT_DEPTH_SCALE_TO_M

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        pose = np.loadtxt(frame.pose_path)
        if pose.shape != (4, 4):
            raise ValueError(f"Invalid pose matrix shape in {frame.pose_path}: {pose.shape}")
        if np.isnan(pose).any() or np.isinf(pose).any():
            raise ValueError(f"Invalid pose matrix in {frame.pose_path}")
        return pose

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.intrinsics_path)[:3, :3]

    def _load_geometry_vertex_data(self):
        mesh_path = self.scene_root / f"{self.scene_id}{GEOMETRY_SUFFIX}"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing ScanNet geometry file: {mesh_path}")

        plydata = PlyData.read(str(mesh_path))
        if "vertex" not in plydata:
            raise RuntimeError(f"PLY file has no vertex element: {mesh_path}")

        vertex_data = plydata["vertex"].data
        if len(vertex_data) == 0:
            raise RuntimeError(f"Loaded empty geometry from {mesh_path}")

        return vertex_data

    def load_geometry_points(self) -> np.ndarray:
        vertex_data = self._load_geometry_vertex_data()
        points = np.stack(
            [
                np.asarray(vertex_data["x"], dtype=np.float32),
                np.asarray(vertex_data["y"], dtype=np.float32),
                np.asarray(vertex_data["z"], dtype=np.float32),
            ],
            axis=1,
        )
        return points

    def load_geometry_colors(self) -> np.ndarray | None:
        vertex_data = self._load_geometry_vertex_data()
        names = set(vertex_data.dtype.names or [])

        if not {"red", "green", "blue"}.issubset(names):
            return None

        colors = np.stack(
            [
                np.asarray(vertex_data["red"], dtype=np.float32),
                np.asarray(vertex_data["green"], dtype=np.float32),
                np.asarray(vertex_data["blue"], dtype=np.float32),
            ],
            axis=1,
        )

        if colors.max() > 1.0:
            colors = colors / 255.0

        return colors

    def get_geometry_record(self) -> GeometryRecord:
        return GeometryRecord(
            geometry_path=self.scene_root / f"{self.scene_id}{GEOMETRY_SUFFIX}",
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=DEFAULT_VISIBILITY_MIN_DEPTH_M,
            z_tolerance_m=DEFAULT_VISIBILITY_Z_TOLERANCE_M,
            depth_scale_to_m=DEFAULT_DEPTH_SCALE_TO_M,
            depth_aligned_to_rgb=True,
        )

    def load_gt_instance_ids(self) -> np.ndarray | None:
        return load_scannet_gt_instance_ids(
            self.scene_root,
            self.scene_id,
            eval_benchmark=self.eval_benchmark,
        )

    def get_evaluation_hooks(self) -> ScanNetEvaluationHooks:
        return ScanNetEvaluationHooks(self.eval_benchmarks)
