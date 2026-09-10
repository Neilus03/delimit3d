from __future__ import annotations

from abc import ABC, abstractmethod

from delimit3d.source.common.types import TeacherOutput
from delimit3d.source.datasets.base import SceneAdapter


class TeacherModel(ABC):
    @abstractmethod
    def run(
        self,
        adapter: SceneAdapter,
        granularity: float,
        frame_skip: int,
    ) -> TeacherOutput:
        raise NotImplementedError