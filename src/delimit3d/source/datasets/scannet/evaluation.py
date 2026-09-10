from __future__ import annotations

from pathlib import Path
from typing import Any

from delimit3d.identity import project_slug
from delimit3d.source.common.types import ClusterOutput
from delimit3d.source.datasets.scannet.benchmark import (
    SCANNET_EVAL_BENCHMARK_20,
    parse_scannet_eval_benchmarks,
    primary_scannet_eval_benchmark,
)
from delimit3d.source.eval.base import DatasetEvaluationHooks


def _safe_mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _bucket_key(name: str) -> str | None:
    lower = str(name).strip().lower()
    if lower.startswith("small"):
        return "small"
    if lower.startswith("medium"):
        return "medium"
    if lower.startswith("large"):
        return "large"
    return None


def _oracle_metric_suffix(benchmark: str | None) -> str:
    if benchmark in {None, "", SCANNET_EVAL_BENCHMARK_20}:
        return ""
    return f"_{benchmark}"


def _benchmark_title(benchmark: str) -> str:
    if benchmark == "scannet20":
        return "ScanNet20"
    if benchmark == "scannet200":
        return "ScanNet200"
    return benchmark


class ScanNetEvaluationHooks(DatasetEvaluationHooks):
    def __init__(self, eval_benchmarks: list[str] | tuple[str, ...] | str | None = None):
        self.eval_benchmarks = parse_scannet_eval_benchmarks(eval_benchmarks)
        self.primary_eval_benchmark = primary_scannet_eval_benchmark(self.eval_benchmarks)

    def scene_metric_fieldnames(self) -> list[str]:
        fields = [
            "oracle_nmi",
            "oracle_ari",
            "oracle_ap25_small",
            "oracle_ap50_small",
            "oracle_ap25_medium",
            "oracle_ap50_medium",
            "oracle_ap25_large",
            "oracle_ap50_large",
            "oracle_map_25_95_small",
            "oracle_map_25_95_medium",
            "oracle_map_25_95_large",
            "oracle_topk_iou025_r1",
            "oracle_topk_iou025_r3",
            "oracle_topk_iou025_r5",
            "oracle_topk_iou050_r1",
            "oracle_topk_iou050_r3",
            "oracle_topk_iou050_r5",
            "oracle_winner_share_g0_2",
            "oracle_winner_share_g0_5",
            "oracle_winner_share_g0_8",
            "oracle_winner_share_no_match",
        ]
        if "scannet200" in self.eval_benchmarks:
            fields.extend(
                [
                    "oracle_nmi_scannet200",
                    "oracle_ari_scannet200",
                    "oracle_ap25_small_scannet200",
                    "oracle_ap50_small_scannet200",
                    "oracle_ap25_medium_scannet200",
                    "oracle_ap50_medium_scannet200",
                    "oracle_ap25_large_scannet200",
                    "oracle_ap50_large_scannet200",
                ]
            )
        return fields

    def evaluate_scene(
        self,
        adapter: Any,
        cluster_outputs: list[ClusterOutput],
    ) -> dict[str, Any] | None:
        from delimit3d.source.eval.scannet_oracle import evaluate_and_save_scannet_oracle

        oracle_summaries: dict[str, dict[str, Any]] = {}
        for eval_benchmark in self.eval_benchmarks:
            oracle_summaries[eval_benchmark] = evaluate_and_save_scannet_oracle(
                adapter=adapter,
                cluster_outputs=cluster_outputs,
                eval_benchmark=eval_benchmark,
            )

        primary_oracle_summary = oracle_summaries.get(self.primary_eval_benchmark)
        if primary_oracle_summary is None and oracle_summaries:
            primary_oracle_summary = next(iter(oracle_summaries.values()))
        if primary_oracle_summary is None:
            return None

        return {
            "eval_benchmark": self.primary_eval_benchmark,
            "eval_benchmarks": list(self.eval_benchmarks),
            "oracle_summary": {
                "eval_benchmark": primary_oracle_summary["eval_benchmark"],
                "metrics_path": str(primary_oracle_summary["metrics_path"]),
                "labels_path": str(primary_oracle_summary["labels_path"]),
                "ply_path": str(primary_oracle_summary["ply_path"]),
                "oracle_results": primary_oracle_summary["oracle_results"],
                "additional_metrics": primary_oracle_summary["additional_metrics"],
                "clustering_metrics": primary_oracle_summary["clustering_metrics"],
            },
            "oracle_summaries": {
                benchmark: {
                    "eval_benchmark": summary["eval_benchmark"],
                    "metrics_path": str(summary["metrics_path"]),
                    "labels_path": str(summary["labels_path"]),
                    "ply_path": str(summary["ply_path"]),
                    "oracle_results": summary["oracle_results"],
                    "additional_metrics": summary["additional_metrics"],
                    "clustering_metrics": summary["clustering_metrics"],
                }
                for benchmark, summary in oracle_summaries.items()
            },
        }

    def _flatten_single_oracle_summary(
        self,
        flat: dict[str, Any],
        oracle_summary: dict[str, Any],
        benchmark: str | None,
    ) -> None:
        suffix = _oracle_metric_suffix(benchmark)

        clustering_metrics = oracle_summary.get("clustering_metrics", {}) or {}
        flat[f"oracle_nmi{suffix}"] = clustering_metrics.get("NMI")
        flat[f"oracle_ari{suffix}"] = clustering_metrics.get("ARI")

        oracle_results = oracle_summary.get("oracle_results", {}) or {}
        for bucket_name, bucket_metrics in oracle_results.items():
            bucket = _bucket_key(bucket_name)
            if bucket is None or not isinstance(bucket_metrics, dict):
                continue
            flat[f"oracle_ap25_{bucket}{suffix}"] = bucket_metrics.get("AP25")
            flat[f"oracle_ap50_{bucket}{suffix}"] = bucket_metrics.get("AP50")
            flat[f"oracle_count_{bucket}{suffix}"] = bucket_metrics.get("Count")

        additional_metrics = oracle_summary.get("additional_metrics", {}) or {}

        map_by_bucket = additional_metrics.get("oracle_mAP_25_95_by_bucket", {}) or {}
        for bucket_name, value in map_by_bucket.items():
            bucket = _bucket_key(bucket_name)
            if bucket is None:
                continue
            flat[f"oracle_map_25_95_{bucket}{suffix}"] = value

        topk = additional_metrics.get("topk_proposal_coverage", {}) or {}

        topk_025 = topk.get("iou_0.25", {}) or {}
        flat[f"oracle_topk_iou025_r1{suffix}"] = topk_025.get("R_at_least_1")
        flat[f"oracle_topk_iou025_r3{suffix}"] = topk_025.get("R_at_least_3")
        flat[f"oracle_topk_iou025_r5{suffix}"] = topk_025.get("R_at_least_5")

        topk_050 = topk.get("iou_0.50", {}) or {}
        flat[f"oracle_topk_iou050_r1{suffix}"] = topk_050.get("R_at_least_1")
        flat[f"oracle_topk_iou050_r3{suffix}"] = topk_050.get("R_at_least_3")
        flat[f"oracle_topk_iou050_r5{suffix}"] = topk_050.get("R_at_least_5")

        winner_share = additional_metrics.get("winner_granularity_share", {}) or {}
        for key, value in winner_share.items():
            safe_key = str(key).replace(".", "_")
            flat[f"oracle_winner_share_{safe_key}{suffix}"] = value

    def flatten_scene_summary(self, scene_summary: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        oracle_summaries = scene_summary.get("oracle_summaries", {}) or {}
        if oracle_summaries:
            for benchmark, oracle_summary in oracle_summaries.items():
                self._flatten_single_oracle_summary(flat, oracle_summary, benchmark)
        else:
            oracle_summary = scene_summary.get("oracle_summary", {}) or {}
            if oracle_summary:
                self._flatten_single_oracle_summary(flat, oracle_summary, SCANNET_EVAL_BENCHMARK_20)
        return flat

    def expected_output_paths(
        self,
        scene_dir: Path,
        granularities: list[float],
        require_oracle: bool = True,
        require_training_pack: bool = True,
    ) -> list[Path]:
        if not require_oracle:
            return []

        expected: list[Path] = []
        label_prefix = project_slug()
        for benchmark in self.eval_benchmarks:
            suffix = "" if benchmark == SCANNET_EVAL_BENCHMARK_20 else f"_{benchmark}"
            expected.extend(
                [
                    scene_dir / f"oracle_metrics{suffix}.json",
                    scene_dir / f"{label_prefix}_oracle_best_combined_labels{suffix}.npy",
                    scene_dir / f"{label_prefix}_oracle_best_combined{suffix}.ply",
                ]
            )
        return expected

    def verify_summary(
        self,
        scene_dir: Path,
        summary: dict[str, Any],
        granularities: list[float],
        require_oracle: bool = True,
        require_training_pack: bool = True,
    ) -> list[str]:
        missing_reasons: list[str] = []
        if not require_oracle:
            return missing_reasons

        if summary.get("oracle_summary") is None:
            missing_reasons.append("summary missing oracle_summary")

        actual_eval_benchmarks = summary.get("eval_benchmarks")
        if actual_eval_benchmarks is None:
            actual_eval_benchmarks = [summary.get("eval_benchmark")]
        try:
            actual_eval_benchmarks = parse_scannet_eval_benchmarks(actual_eval_benchmarks)
        except Exception:
            actual_eval_benchmarks = None

        if actual_eval_benchmarks != self.eval_benchmarks:
            missing_reasons.append(
                "summary eval_benchmarks mismatch: "
                f"expected {self.eval_benchmarks}, found {actual_eval_benchmarks}"
            )

        oracle_summaries = summary.get("oracle_summaries") or {}
        missing_benchmarks = [
            benchmark for benchmark in self.eval_benchmarks if benchmark not in oracle_summaries
        ]
        if missing_benchmarks:
            missing_reasons.append(
                f"summary missing oracle_summaries for benchmarks: {missing_benchmarks}"
            )

        return missing_reasons

    def running_summary_payload(
        self,
        done_rows: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        payload = {
            "summary/running_avg_oracle_nmi": _safe_mean([r.get("oracle_nmi") for r in done_rows]),
            "summary/running_avg_oracle_ari": _safe_mean([r.get("oracle_ari") for r in done_rows]),
        }
        if "scannet200" in self.eval_benchmarks:
            payload["summary/running_avg_oracle_nmi_scannet200"] = _safe_mean(
                [r.get("oracle_nmi_scannet200") for r in done_rows]
            )
            payload["summary/running_avg_oracle_ari_scannet200"] = _safe_mean(
                [r.get("oracle_ari_scannet200") for r in done_rows]
            )
        return payload

    def _render_single_benchmark_summary(
        self,
        run_summary: dict[str, Any],
        benchmark: str,
        granularities: list[float],
    ) -> list[str]:
        suffix = _oracle_metric_suffix(benchmark)
        scene_results = run_summary.get("scene_results", []) or []
        done_rows = [
            row for row in scene_results if row.get("status") in {"done", "skipped_done"}
        ]
        if not done_rows:
            return []

        lines: list[str] = []
        bucket_names = ["small", "medium", "large"]
        bucket_labels = {
            "small": "Small (scene-wise tertile)",
            "medium": "Medium (scene-wise tertile)",
            "large": "Large (scene-wise tertile)",
        }

        n_scenes = len(done_rows)
        title = _benchmark_title(benchmark)

        lines.append("\n" + "=" * 70)
        lines.append(f"FINAL RESULTS ACROSS {n_scenes} SCENES ({title})")
        lines.append("=" * 70)
        lines.append(f"{'Size Bucket':<28} | {'Avg AP@25':<12} | {'Avg AP@50':<12}")
        lines.append("-" * 70)

        ap25_values: list[float | None] = []
        ap50_values: list[float | None] = []
        for bucket in bucket_names:
            ap25 = _safe_mean([row.get(f"oracle_ap25_{bucket}{suffix}") for row in done_rows])
            ap50 = _safe_mean([row.get(f"oracle_ap50_{bucket}{suffix}") for row in done_rows])
            ap25_values.append(ap25)
            ap50_values.append(ap50)
            lines.append(
                f"{bucket_labels[bucket]:<28} | "
                f"{_format_metric(ap25):<12} | "
                f"{_format_metric(ap50):<12}"
            )

        lines.append("-" * 70)
        lines.append(
            f"{f'GLOBAL AVG   (n={n_scenes} scenes)':<28} | "
            f"{_format_metric(_safe_mean(ap25_values)):<12} | "
            f"{_format_metric(_safe_mean(ap50_values)):<12}"
        )
        lines.append("=" * 70)

        lines.append("=" * 70)
        lines.append(f"ADDITIONAL ORACLE METRICS ({title}, AVERAGED ACROSS SCENES)")
        lines.append("=" * 70)
        for bucket in bucket_names:
            map_value = _safe_mean(
                [row.get(f"oracle_map_25_95_{bucket}{suffix}") for row in done_rows]
            )
            lines.append(
                f"{bucket_labels[bucket]:<28} | "
                f"mAP@[.25:.95]={_format_metric(map_value)} (n={n_scenes})"
            )

        topk025_r1 = _safe_mean([row.get(f"oracle_topk_iou025_r1{suffix}") for row in done_rows])
        topk025_r3 = _safe_mean([row.get(f"oracle_topk_iou025_r3{suffix}") for row in done_rows])
        topk025_r5 = _safe_mean([row.get(f"oracle_topk_iou025_r5{suffix}") for row in done_rows])
        topk050_r1 = _safe_mean([row.get(f"oracle_topk_iou050_r1{suffix}") for row in done_rows])
        topk050_r3 = _safe_mean([row.get(f"oracle_topk_iou050_r3{suffix}") for row in done_rows])
        topk050_r5 = _safe_mean([row.get(f"oracle_topk_iou050_r5{suffix}") for row in done_rows])
        lines.append("")
        lines.append(
            "Top-k coverage @0.25: "
            f"R>=1={_format_metric(topk025_r1)}, "
            f"R>=3={_format_metric(topk025_r3)}, "
            f"R>=5={_format_metric(topk025_r5)}"
        )
        lines.append(
            "Top-k coverage @0.50: "
            f"R>=1={_format_metric(topk050_r1)}, "
            f"R>=3={_format_metric(topk050_r3)}, "
            f"R>=5={_format_metric(topk050_r5)}"
        )

        winner_parts = []
        for granularity in granularities:
            safe_g = str(granularity).replace(".", "_")
            winner_value = _safe_mean(
                [row.get(f"oracle_winner_share_g{safe_g}{suffix}") for row in done_rows]
            )
            winner_parts.append(f"g{granularity}={_format_metric(winner_value)}")
        no_match = _safe_mean(
            [row.get(f"oracle_winner_share_no_match{suffix}") for row in done_rows]
        )
        winner_parts.append(f"no_match={_format_metric(no_match)}")
        lines.append("Winner granularity share: " + ", ".join(winner_parts))
        lines.append("=" * 70)

        lines.append("\n" + "=" * 70)
        lines.append(f"CLUSTERING CONSISTENCY METRICS ({title}, AVERAGED ACROSS SCENES)")
        lines.append("=" * 70)
        nmi = _safe_mean([row.get(f"oracle_nmi{suffix}") for row in done_rows])
        ari = _safe_mean([row.get(f"oracle_ari{suffix}") for row in done_rows])
        lines.append(f"NMI: {_format_metric(nmi)} (n={n_scenes})")
        lines.append(f"ARI: {_format_metric(ari)} (n={n_scenes})")
        lines.append("=" * 70)
        lines.append(f"Total scenes evaluated: {n_scenes}")
        lines.append("Granularities tested: " + ", ".join(str(g) for g in granularities))
        lines.append("=" * 70)
        return lines

    def render_run_summary(
        self,
        run_summary: dict[str, Any],
        granularities: list[float],
    ) -> list[str]:
        lines: list[str] = []
        for benchmark in self.eval_benchmarks:
            lines.extend(self._render_single_benchmark_summary(run_summary, benchmark, granularities))
        return lines
