from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    rgb_path: Path
    depth_path: Path
    pose_path: Path
    intrinsics_path: Path


@dataclass(frozen=True)
class GeometryRecord:
    geometry_path: Path
    geometry_type: str


@dataclass(frozen=True)
class VisibilityConfig:
    min_depth_m: float
    z_tolerance_m: float
    depth_scale_to_m: float
    depth_aligned_to_rgb: bool = True


@dataclass
class TeacherOutput:
    granularity: float
    frame_mask_paths: list[Path]
    total_masks: int


@dataclass
class ClusterOutput:
    granularity: float
    labels: np.ndarray
    features: np.ndarray
    seen_mask: np.ndarray
    ply_path: Optional[Path]
    labels_path: Optional[Path]
    stats: dict