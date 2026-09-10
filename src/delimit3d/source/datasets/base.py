from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from delimit3d.source.common.types import FrameRecord, GeometryRecord, VisibilityConfig
from delimit3d.source.eval.base import DatasetEvaluationHooks


class SceneAdapter(ABC):
    def __init__(self, scene_root: Path):
        self.scene_root = Path(scene_root)
        self.scene_id = self.scene_root.name

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_frames(self) -> list[FrameRecord]:
        raise NotImplementedError

    @abstractmethod
    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_geometry_points(self) -> np.ndarray:
        raise NotImplementedError

    def load_geometry_colors(self) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def get_geometry_record(self) -> GeometryRecord:
        raise NotImplementedError

    @abstractmethod
    def get_visibility_config(self) -> VisibilityConfig:
        raise NotImplementedError

    def load_gt_instance_ids(self) -> Optional[np.ndarray]:
        return None

    def get_evaluation_hooks(self) -> DatasetEvaluationHooks:
        return DatasetEvaluationHooks()