#!/usr/bin/env python3
"""Create reproducible S3DIS result tables and static plots.

The script consumes the completed cross-dataset experiment root produced by
the matched S3DIS evaluator.  It reads the aggregate and both arm reports,
checks paired identity coverage and aggregate consistency, and writes a small
report bundle without modifying the experiment inputs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IOU_CLICKS = (1, 3, 5, 10, 15, 20)
NOC_TARGETS = (50, 65, 80, 85, 90)
IOU_METRICS = tuple(f"IoU@{click}" for click in IOU_CLICKS)
NOC_METRICS = tuple(f"NoC@{target}" for target in NOC_TARGETS)
ALL_METRICS = IOU_METRICS + NOC_METRICS
ARMS = ("public", "delimit3d")
ARM_LABELS = {"public": "Public LitePT", "delimit3d": "Delimit3D"}
ARM_COLORS = {"public": "#3b6fb6", "delimit3d": "#d97732"}
ACCENT = "#4f5965"
GRID = "#d7dce2"
TEXT = "#20252b"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object rows in {path}")
                rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_enough(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#89939e")
    ax.spines["bottom"].set_color("#89939e")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def validate_inputs(root: Path, aggregate: dict[str, Any]) -> dict[str, Any]:
    if aggregate.get("dataset") != "S3DIS":
        raise ValueError(f"aggregate dataset is not S3DIS: {aggregate.get('dataset')!r}")
    endpoints = aggregate.get("endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != {"MO", "SO"}:
        raise ValueError("aggregate must contain completed MO and SO endpoints")

    validation: dict[str, Any] = {
        "endpoint_identity_sets_equal": {},
        "aggregate_means_match_reports": {},
        "report_counts": {},
    }
    for endpoint in ("MO", "SO"):
        reports: dict[str, dict[str, Any]] = {}
        identity_sets: dict[str, set[str]] = {}
        for arm in ARMS:
            report_path = root / arm / "evaluation" / endpoint / "evaluation_report.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"missing evaluation report: {report_path}")
            report = read_json(report_path)
            units = report.get("units")
            if not isinstance(units, list) or not units:
                raise ValueError(f"missing units in {report_path}")
            identities = [str(row["identity"]) for row in units]
            if len(set(identities)) != len(identities):
                raise ValueError(f"duplicate identities in {report_path}")
            reports[arm] = report
            identity_sets[arm] = set(identities)
            validation["report_counts"][f"{endpoint}/{arm}"] = len(units)

        equal = identity_sets["public"] == identity_sets["delimit3d"]
        validation["endpoint_identity_sets_equal"][endpoint] = equal
        if not equal:
            only_public = sorted(identity_sets["public"] - identity_sets["delimit3d"])[:5]
            only_delimit3d = sorted(identity_sets["delimit3d"] - identity_sets["public"])[:5]
            raise ValueError(
                f"unpaired {endpoint} identities; public-only={only_public}, "
                f"delimit3d-only={only_delimit3d}"
            )

        aggregate_endpoint = endpoints[endpoint]
        aggregate_metrics = aggregate_endpoint["metrics"]
        for arm in ARMS:
            for metric in ALL_METRICS:
                values = [float(row["metrics"][metric]) for row in reports[arm]["units"]]
                observed = float(np.mean(values))
                recorded = float(aggregate_metrics[arm][metric])
                if not close_enough(observed, recorded):
                    raise ValueError(
                        f"aggregate mismatch for {endpoint}/{arm}/{metric}: "
                        f"observed={observed} recorded={recorded}"
                    )
                validation["aggregate_means_match_reports"][f"{endpoint}/{arm}/{metric}"] = True

    geometry_path = root / "geometry_comparison.json"
    if geometry_path.is_file():
        geometry = read_json(geometry_path)
        validation["geometry_byte_identity"] = bool(geometry.get("geometry_byte_identity"))
        validation["geometry_scene_count"] = int(geometry.get("scene_count", 0))
        if not validation["geometry_byte_identity"]:
            raise ValueError("geometry comparison is not byte-identical")
    else:
        validation["geometry_byte_identity"] = None

    final_provenance_path = root / "lead_audit" / "s3dis_v2_final_provenance.json"
    if final_provenance_path.is_file():
        final_provenance = read_json(final_provenance_path)
        validation["final_provenance_status"] = final_provenance.get("status")
        if final_provenance.get("status") != "complete":
            raise ValueError(f"final provenance is not complete: {final_provenance.get('status')!r}")
        validation["final_source_commit"] = final_provenance.get("source", {}).get("commit")
    else:
        validation["final_provenance_status"] = None

    return validation


def summary_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint in ("MO", "SO"):
        payload = aggregate["endpoints"][endpoint]
        for metric in ALL_METRICS:
            stats = payload["paired_bootstrap"][metric]
            public = float(payload["metrics"]["public"][metric])
            delimit3d = float(payload["metrics"]["delimit3d"][metric])
            observed_delta = delimit3d - public
            recorded_delta = float(stats["delta"])
            if not close_enough(observed_delta, recorded_delta):
                raise ValueError(
                    f"paired delta mismatch for {endpoint}/{metric}: "
                    f"observed={observed_delta} recorded={recorded_delta}"
                )
            rows.append(
                {
                    "endpoint": endpoint,
                    "bootstrap_unit": payload["bootstrap_unit"],
                    "paired_units": int(stats["paired_units"]),
                    "metric": metric,
                    "public": public,
                    "delimit3d": delimit3d,
                    "delta": recorded_delta,
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "samples": int(stats["samples"]),
                    "seed": int(stats["seed"]),
                }
            )
    return rows


def strata_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return IoU@5 object/scene strata with the correct interpretation.

    MO uses the aggregate's descriptive object strata for semantic and size
    behavior; its ordinary semantic/size scene strata are based on the first
    requested object and are therefore not used here. SO is already one object
    per unit, so its ordinary strata are the appropriate object breakdown.
    """

    rows: list[dict[str, Any]] = []
    for endpoint in ("MO", "SO"):
        payload = aggregate["endpoints"][endpoint]
        groups: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []
        if endpoint == "MO":
            groups.append(
                (
                    "MO object (descriptive)",
                    payload.get("descriptive_object_strata", {}),
                    ("semantic/", "size/"),
                )
            )
            groups.append(
                (
                    "MO scene (requested count)",
                    payload.get("strata", {}),
                    ("requested_count/",),
                )
            )
        else:
            groups.append(("SO object", payload.get("strata", {}), ("semantic/", "size/")))

        for scope, strata, prefixes in groups:
            for name, metrics in sorted(strata.items()):
                if not name.startswith(prefixes) or "IoU@5" not in metrics:
                    continue
                stats = metrics["IoU@5"]
                rows.append(
                    {
                        "endpoint": endpoint,
                        "stratum_scope": scope,
                        "group": name.split("/", 1)[0],
                        "stratum": name.split("/", 1)[1],
                        "paired_units": int(stats["paired_units"]),
                        "delta_iou5_pp": 100.0 * float(stats["delta"]),
                        "ci95_lower_pp": 100.0 * float(stats["ci95_lower"]),
                        "ci95_upper_pp": 100.0 * float(stats["ci95_upper"]),
                        "samples": int(stats["samples"]),
                        "seed": int(stats["seed"]),
                    }
                )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(metric: str, value: float) -> str:
    if metric.startswith("IoU"):
        return f"{100.0 * value:.2f}%"
    return f"{value:.2f}"


def format_delta(metric: str, value: float) -> str:
    if metric.startswith("IoU"):
        return f"{100.0 * value:+.2f} pp"
    return f"{value:+.2f} clicks"


def format_ci(metric: str, lower: float, upper: float) -> str:
    if metric.startswith("IoU"):
        return f"[{100.0 * lower:+.2f}, {100.0 * upper:+.2f}] pp"
    return f"[{lower:+.2f}, {upper:+.2f}] clicks"


def write_markdown_report(
    path: Path,
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    strata: list[dict[str, Any]],
    root: Path,
    validation: dict[str, Any],
) -> None:
    lines = [
        "# Current S3DIS results",
        "",
        f"Experiment: `{aggregate['experiment_id']}`  ",
        f"Metric scope: {aggregate['metric_scope']}  ",
        "IoU is higher-is-better; NoC (number of clicks) is lower-is-better. "
        "Intervals are paired bootstrap 95% CIs over the unit shown in the table.",
        "",
        "## Summary",
        "",
    ]
    for endpoint in ("MO", "SO"):
        endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
        n = endpoint_rows[0]["paired_units"]
        unit = endpoint_rows[0]["bootstrap_unit"]
        lines.extend(
            [
                f"### {endpoint} ({n} paired {unit}s)",
                "",
                "| Metric | Public LitePT | Delimit3D | Δ (Delimit3D − Public) | Paired 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in endpoint_rows:
            lines.append(
                f"| {row['metric']} | {format_metric(row['metric'], row['public'])} | "
                f"{format_metric(row['metric'], row['delimit3d'])} | "
                f"{format_delta(row['metric'], row['delta'])} | "
                f"{format_ci(row['metric'], row['ci95_lower'], row['ci95_upper'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## IoU@5 strata",
            "",
            "MO semantic and size rows use the descriptive object-level strata. "
            "SO rows are object-level because SO has one requested object per unit. "
            "MO requested-count rows remain scene-level.",
            "",
            "| Endpoint | Scope | Stratum | n | Δ IoU@5 (pp) | Paired 95% CI (pp) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in strata:
        lines.append(
            f"| {row['endpoint']} | {row['stratum_scope']} | {row['group']}/{row['stratum']} | "
            f"{row['paired_units']} | {row['delta_iou5_pp']:+.2f} | "
            f"[{row['ci95_lower_pp']:+.2f}, {row['ci95_upper_pp']:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Validation and provenance",
            "",
            f"- Aggregate: `{root / 'aggregate.json'}`",
            f"- Final provenance status: `{validation.get('final_provenance_status')}`",
            f"- Source commit recorded by final provenance: `{validation.get('final_source_commit')}`",
            f"- Paired identities equal for MO and SO: `{all(validation['endpoint_identity_sets_equal'].values())}`",
            f"- Geometry byte identity: `{validation.get('geometry_byte_identity')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_curves(aggregate: dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for row_index, endpoint in enumerate(("MO", "SO")):
        payload = aggregate["endpoints"][endpoint]
        ax_iou = axes[row_index, 0]
        ax_noc = axes[row_index, 1]
        for ax in (ax_iou, ax_noc):
            style_axes(ax)

        x_iou = np.asarray(IOU_CLICKS, dtype=float)
        for arm in ARMS:
            values = [100.0 * float(payload["metrics"][arm][f"IoU@{click}"]) for click in IOU_CLICKS]
            ax_iou.plot(
                x_iou,
                values,
                marker="o",
                linewidth=2.1,
                markersize=5,
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm],
            )
        ax_iou.set_title(f"{endpoint} mean IoU ({payload['bootstrap_unit']} n={payload['paired_bootstrap']['IoU@1']['paired_units']})")
        ax_iou.set_xlabel("Clicks per requested object")
        ax_iou.set_ylabel("Mean IoU (%)")
        ax_iou.set_xticks(IOU_CLICKS)
        ax_iou.set_ylim(0, 100)
        ax_iou.legend(frameon=False, loc="lower right", fontsize=8)

        x_noc = np.asarray(NOC_TARGETS, dtype=float)
        for arm in ARMS:
            values = [float(payload["metrics"][arm][f"NoC@{target}"]) for target in NOC_TARGETS]
            ax_noc.plot(
                x_noc,
                values,
                marker="o",
                linewidth=2.1,
                markersize=5,
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm],
            )
        ax_noc.set_title(f"{endpoint} click efficiency")
        ax_noc.set_xlabel("Target IoU (%)")
        ax_noc.set_ylabel("NoC (clicks/object)")
        ax_noc.set_xticks(NOC_TARGETS)
        ax_noc.set_ylim(0, 21)
        ax_noc.legend(frameon=False, loc="upper left", fontsize=8)

    fig.suptitle("S3DIS matched transfer: interactive performance", fontsize=15, fontweight="bold", color=TEXT)
    fig.savefig(out_dir / "s3dis_performance_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "s3dis_performance_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_deltas(aggregate: dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)
    for row_index, endpoint in enumerate(("MO", "SO")):
        payload = aggregate["endpoints"][endpoint]
        ax_iou = axes[row_index, 0]
        ax_noc = axes[row_index, 1]
        for ax in (ax_iou, ax_noc):
            style_axes(ax)

        iou_stats = [payload["paired_bootstrap"][metric] for metric in IOU_METRICS]
        iou_delta = np.asarray([100.0 * float(stat["delta"]) for stat in iou_stats])
        iou_lower = np.asarray([100.0 * float(stat["ci95_lower"]) for stat in iou_stats])
        iou_upper = np.asarray([100.0 * float(stat["ci95_upper"]) for stat in iou_stats])
        x_iou = np.arange(len(IOU_METRICS))
        ax_iou.errorbar(
            x_iou,
            iou_delta,
            yerr=np.vstack((iou_delta - iou_lower, iou_upper - iou_delta)),
            fmt="o",
            color=ARM_COLORS["delimit3d"],
            ecolor=ARM_COLORS["delimit3d"],
            elinewidth=1.5,
            capsize=3,
            markersize=5,
        )
        ax_iou.axhline(0, color=ACCENT, linewidth=1.0)
        ax_iou.set_title(f"{endpoint} paired IoU change")
        ax_iou.set_ylabel("Delimit3D − Public (pp)")
        ax_iou.set_xticks(x_iou, [str(click) for click in IOU_CLICKS])
        ax_iou.set_xlabel("Clicks per requested object")

        noc_stats = [payload["paired_bootstrap"][metric] for metric in NOC_METRICS]
        noc_delta = np.asarray([float(stat["delta"]) for stat in noc_stats])
        noc_lower = np.asarray([float(stat["ci95_lower"]) for stat in noc_stats])
        noc_upper = np.asarray([float(stat["ci95_upper"]) for stat in noc_stats])
        x_noc = np.arange(len(NOC_METRICS))
        ax_noc.errorbar(
            x_noc,
            noc_delta,
            yerr=np.vstack((noc_delta - noc_lower, noc_upper - noc_delta)),
            fmt="o",
            color=ARM_COLORS["delimit3d"],
            ecolor=ARM_COLORS["delimit3d"],
            elinewidth=1.5,
            capsize=3,
            markersize=5,
        )
        ax_noc.axhline(0, color=ACCENT, linewidth=1.0)
        ax_noc.set_title(f"{endpoint} paired click change")
        ax_noc.set_xlabel("Target IoU (%)")
        ax_noc.set_ylabel("Delimit3D − Public (clicks/object)")
        ax_noc.set_xticks(x_noc, [str(target) for target in NOC_TARGETS])

    fig.suptitle("S3DIS paired improvements with 95% bootstrap intervals", fontsize=15, fontweight="bold", color=TEXT)
    fig.savefig(out_dir / "s3dis_paired_deltas.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "s3dis_paired_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def load_unit_metrics(root: Path, endpoint: str) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        report = read_json(root / arm / "evaluation" / endpoint / "evaluation_report.json")
        result[arm] = {
            str(row["identity"]): {metric: float(row["metrics"][metric]) for metric in ALL_METRICS}
            for row in report["units"]
        }
    return result


def plot_distributions(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    for endpoint, ax in zip(("MO", "SO"), axes):
        style_axes(ax)
        units = load_unit_metrics(root, endpoint)
        identities = sorted(set(units["public"]) & set(units["delimit3d"]))
        values: list[np.ndarray] = []
        for click in (1, 5, 10):
            metric = f"IoU@{click}"
            values.append(
                np.asarray(
                    [100.0 * (units["delimit3d"][identity][metric] - units["public"][identity][metric]) for identity in identities],
                    dtype=float,
                )
            )
        box = ax.boxplot(
            values,
            positions=(1, 2, 3),
            widths=0.48,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": TEXT, "markeredgecolor": TEXT, "markersize": 4},
            medianprops={"color": TEXT, "linewidth": 1.2},
            whiskerprops={"color": ACCENT},
            capprops={"color": ACCENT},
            flierprops={"marker": ".", "markerfacecolor": ACCENT, "markeredgecolor": ACCENT, "alpha": 0.35},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(ARM_COLORS["delimit3d"])
            patch.set_alpha(0.28)
            patch.set_edgecolor(ARM_COLORS["delimit3d"])
        ax.axhline(0, color=ACCENT, linewidth=1.0)
        ax.set_title(f"{endpoint} unit-level change (n={len(identities)})")
        ax.set_xlabel("Clicks per requested object")
        ax.set_ylabel("Delimit3D − Public IoU (pp)")
        ax.set_xticks((1, 2, 3), ("1", "5", "10"))
    fig.suptitle("Distribution of paired IoU changes", fontsize=15, fontweight="bold", color=TEXT)
    fig.savefig(out_dir / "s3dis_unit_delta_distributions.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "s3dis_unit_delta_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_strata(strata: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    for row_index, group in enumerate(("semantic", "size")):
        for col_index, endpoint in enumerate(("MO", "SO")):
            ax = axes[row_index, col_index]
            style_axes(ax)
            selected = [
                row
                for row in strata
                if row["endpoint"] == endpoint and row["group"] == group
            ]
            selected.sort(key=lambda row: row["delta_iou5_pp"])
            if not selected:
                ax.text(0.5, 0.5, "No strata", ha="center", va="center", color=TEXT)
                ax.set_axis_off()
                continue
            y = np.arange(len(selected))
            delta = np.asarray([row["delta_iou5_pp"] for row in selected])
            lower = np.asarray([row["ci95_lower_pp"] for row in selected])
            upper = np.asarray([row["ci95_upper_pp"] for row in selected])
            ax.errorbar(
                delta,
                y,
                xerr=np.vstack((delta - lower, upper - delta)),
                fmt="o",
                color=ARM_COLORS["delimit3d"],
                ecolor=ARM_COLORS["delimit3d"],
                elinewidth=1.25,
                capsize=2.5,
                markersize=4.5,
            )
            ax.axvline(0, color=ACCENT, linewidth=1.0)
            ax.set_yticks(y, [f"{row['stratum']} (n={row['paired_units']})" for row in selected])
            ax.set_xlabel("Delimit3D − Public IoU@5 (pp)")
            ax.set_title(f"{endpoint} {group}")
            ax.invert_yaxis()
    fig.suptitle("S3DIS IoU@5 changes by object stratum", fontsize=15, fontweight="bold", color=TEXT)
    fig.savefig(out_dir / "s3dis_iou5_strata.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "s3dis_iou5_strata.pdf", bbox_inches="tight")
    plt.close(fig)


def write_provenance(
    path: Path,
    root: Path,
    aggregate: dict[str, Any],
    validation: dict[str, Any],
    outputs: list[Path],
) -> None:
    input_paths = [
        root / "aggregate.json",
        root / "geometry_comparison.json",
        root / "lead_audit" / "s3dis_v2_final_provenance.json",
    ]
    for endpoint in ("MO", "SO"):
        for arm in ARMS:
            input_paths.append(root / arm / "evaluation" / endpoint / "evaluation_report.json")
    input_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in input_paths
        if path.is_file()
    }
    path.write_text(
        json.dumps(
            {
                "schema": "delimit3d_s3dis_results_report/v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": aggregate["experiment_id"],
                "dataset": "S3DIS",
                "metric_scope": aggregate["metric_scope"],
                "source_root": str(root),
                "aggregate_sha256": sha256_file(root / "aggregate.json"),
                "input_hashes": input_hashes,
                "validation": validation,
                "outputs": [str(output) for output in outputs],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="completed S3DIS experiment root")
    parser.add_argument("--output", type=Path, required=True, help="directory for generated report files")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    aggregate_path = root / "aggregate.json"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"missing aggregate: {aggregate_path}")
    aggregate = read_json(aggregate_path)
    validation = validate_inputs(root, aggregate)
    rows = summary_rows(aggregate)
    strata = strata_rows(aggregate)
    output.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "endpoint",
        "bootstrap_unit",
        "paired_units",
        "metric",
        "public",
        "delimit3d",
        "delta",
        "ci95_lower",
        "ci95_upper",
        "samples",
        "seed",
    ]
    write_csv(output / "s3dis_summary_long.csv", rows, summary_fields)
    strata_fields = [
        "endpoint",
        "stratum_scope",
        "group",
        "stratum",
        "paired_units",
        "delta_iou5_pp",
        "ci95_lower_pp",
        "ci95_upper_pp",
        "samples",
        "seed",
    ]
    write_csv(output / "s3dis_iou5_strata.csv", strata, strata_fields)
    write_markdown_report(output / "s3dis_results.md", aggregate, rows, strata, root, validation)

    plot_outputs = [
        output / "s3dis_performance_curves.png",
        output / "s3dis_performance_curves.pdf",
        output / "s3dis_paired_deltas.png",
        output / "s3dis_paired_deltas.pdf",
        output / "s3dis_unit_delta_distributions.png",
        output / "s3dis_unit_delta_distributions.pdf",
        output / "s3dis_iou5_strata.png",
        output / "s3dis_iou5_strata.pdf",
    ]
    plot_curves(aggregate, output)
    plot_deltas(aggregate, output)
    plot_distributions(root, output)
    plot_strata(strata, output)
    write_provenance(output / "report_provenance.json", root, aggregate, validation, plot_outputs)

    print(
        json.dumps(
            {
                "output": str(output),
                "summary": str(output / "s3dis_results.md"),
                "plots": [str(path) for path in plot_outputs],
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
