from __future__ import annotations

from pathlib import Path
from typing import Any

from delimit3d.source.common.types import ClusterOutput


class DatasetEvaluationHooks:
    """Dataset-owned evaluation/reporting/verification extension points."""

    def scene_metric_fieldnames(self) -> list[str]:
        return []

    def evaluate_scene(
        self,
        adapter: Any,
        cluster_outputs: list[ClusterOutput],
    ) -> dict[str, Any] | None:
        return None

    def flatten_scene_summary(self, scene_summary: dict[str, Any]) -> dict[str, Any]:
        return {}

    def expected_output_paths(
        self,
        scene_dir: Path,
        granularities: list[float],
        require_oracle: bool = True,
        require_training_pack: bool = True,
    ) -> list[Path]:
        return []

    def verify_summary(
        self,
        scene_dir: Path,
        summary: dict[str, Any],
        granularities: list[float],
        require_oracle: bool = True,
        require_training_pack: bool = True,
    ) -> list[str]:
        return []

    def running_summary_payload(
        self,
        done_rows: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        return {}

    def render_run_summary(
        self,
        run_summary: dict[str, Any],
        granularities: list[float],
    ) -> list[str]:
        return []

