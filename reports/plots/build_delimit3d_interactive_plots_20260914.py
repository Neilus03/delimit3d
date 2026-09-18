#!/usr/bin/env python3
"""Render matched LitePT/Delimit3D interactive-result figures.

The source of truth is the completed ``aggregate.json`` from each matched
evaluation.  The script deliberately keeps only the two causal arms:

* public LitePT: the original public checkpoint, frozen for evaluation;
* LitePT + Delimit3D: the public checkpoint after 256 Delimit3D updates,
  frozen before downstream decoder training.

Each output is written as both a publication-friendly PDF and a directly
viewable PNG.  The paired panels use the stored paired bootstrap intervals;
they are not recomputed from rounded values.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "ScanNet40": Path(
        "/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1"
    ),
    "S3DIS": Path(
        "/cluster/work/igp_psr/nedela/delimit3d_s3dis_agile3d_matched_v2"
    ),
}

ARMS = ("public", "delimit3d")
ARM_LABELS = {
    "public": "LitePT (public)",
    "delimit3d": "LitePT + Delimit3D (256 updates)",
}
COLORS = {
    "public": "#3f70b7",
    "delimit3d": "#df792d",
}
TARGET_IOUS = (50, 65, 80, 85, 90)
OUTPUT_DPI = 220

TEXT = "#202832"
SPINE = "#8d9aa8"
GRID = "#d4dce5"
ZERO = "#4f5b67"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def report_count(root: Path, endpoint: str) -> int:
    report = read_json(root / "public" / "evaluation" / endpoint / "evaluation_report.json")
    for key in ("count", "n_units", "num_units", "n"):
        if key in report:
            return int(report[key])
    raise KeyError(f"No evaluation count found in {root / 'public' / 'evaluation' / endpoint}")


def metric_numbers(metrics: dict[str, Any], prefix: str) -> list[int]:
    values = []
    for key in metrics:
        if key.startswith(prefix):
            values.append(int(key.split("@", 1)[1]))
    return sorted(values)


def load_dataset(dataset: str, root: Path) -> dict[str, Any]:
    aggregate_path = root / "aggregate.json"
    aggregate = read_json(aggregate_path)
    endpoints: dict[str, Any] = {}

    for endpoint in ("MO", "SO"):
        endpoint_data = aggregate["endpoints"][endpoint]
        metrics = endpoint_data["metrics"]
        paired = endpoint_data["paired_bootstrap"]

        clicks = [
            click
            for click in metric_numbers(metrics["public"], "IoU@")
            if f"IoU@{click}" in metrics["delimit3d"]
            and f"IoU@{click}" in paired
        ]
        targets = [
            target
            for target in TARGET_IOUS
            if f"NoC@{target}" in metrics["public"]
            and f"NoC@{target}" in metrics["delimit3d"]
            and f"NoC@{target}" in paired
        ]
        if not clicks or targets != list(TARGET_IOUS):
            raise ValueError(f"Unexpected metric keys for {dataset}/{endpoint}")

        iou = {
            arm: [100.0 * float(metrics[arm][f"IoU@{click}"]) for click in clicks]
            for arm in ARMS
        }
        noc = {
            arm: [float(metrics[arm][f"NoC@{target}"]) for target in targets]
            for arm in ARMS
        }

        paired_iou = {
            "delta": [100.0 * float(paired[f"IoU@{click}"]["delta"]) for click in clicks],
            "low": [100.0 * float(paired[f"IoU@{click}"]["ci95_lower"]) for click in clicks],
            "high": [100.0 * float(paired[f"IoU@{click}"]["ci95_upper"]) for click in clicks],
        }
        paired_noc = {
            "delta": [float(paired[f"NoC@{target}"]["delta"]) for target in targets],
            "low": [float(paired[f"NoC@{target}"]["ci95_lower"]) for target in targets],
            "high": [float(paired[f"NoC@{target}"]["ci95_upper"]) for target in targets],
        }

        first_pair = paired[f"IoU@{clicks[0]}"]
        endpoints[endpoint] = {
            "count": report_count(root, endpoint),
            "bootstrap_unit": endpoint_data.get("bootstrap_unit"),
            "clicks": clicks,
            "targets": targets,
            "iou": iou,
            "noc": noc,
            "paired_iou": paired_iou,
            "paired_noc": paired_noc,
            "paired_units": int(first_pair["paired_units"]),
            "bootstrap_samples": int(first_pair["samples"]),
            "bootstrap_seed": first_pair.get("seed"),
        }

    return {
        "dataset": dataset,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256(aggregate_path),
        "experiment_id": aggregate.get("experiment_id"),
        "metric_scope": aggregate.get("metric_scope"),
        "endpoints": endpoints,
    }


def configure_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=1.0, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=TEXT, labelsize=12)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def style_figure(fig: plt.Figure, title: str) -> None:
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=24, fontweight="bold", color=TEXT, y=0.975)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.875,
        wspace=0.26,
        hspace=0.31,
    )


def plot_performance(dataset: str, data: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.7))
    style_figure(fig, f"{dataset} matched transfer: interactive performance")

    for row, endpoint in enumerate(("MO", "SO")):
        endpoint_data = data["endpoints"][endpoint]
        clicks = endpoint_data["clicks"]
        targets = endpoint_data["targets"]
        count = endpoint_data["count"]
        unit = endpoint_data["bootstrap_unit"] or ("scene" if endpoint == "MO" else "object")

        iou_ax = axes[row, 0]
        configure_axes(iou_ax)
        for arm in ARMS:
            iou_ax.plot(
                clicks,
                endpoint_data["iou"][arm],
                color=COLORS[arm],
                marker="o",
                markersize=8,
                linewidth=3.0,
                label=ARM_LABELS[arm],
            )
        iou_ax.set_title(f"{endpoint} mean IoU ({unit} n={count})", fontsize=17, pad=10)
        iou_ax.set_xlabel("Clicks per requested object", fontsize=14, labelpad=8)
        iou_ax.set_ylabel("Mean IoU (%)", fontsize=14, labelpad=8)
        iou_ax.set_ylim(0, 100)
        iou_ax.set_xticks(clicks)
        iou_ax.set_xlim(min(clicks) - 0.8, max(clicks) + 0.8)
        iou_ax.legend(frameon=False, loc="lower right", fontsize=12)

        noc_ax = axes[row, 1]
        configure_axes(noc_ax)
        for arm in ARMS:
            noc_ax.plot(
                targets,
                endpoint_data["noc"][arm],
                color=COLORS[arm],
                marker="o",
                markersize=8,
                linewidth=3.0,
                label=ARM_LABELS[arm],
            )
        noc_ax.set_title(f"{endpoint} click efficiency", fontsize=17, pad=10)
        noc_ax.set_xlabel("Target IoU (%)", fontsize=14, labelpad=8)
        noc_ax.set_ylabel("NOC (clicks/object)", fontsize=14, labelpad=8)
        noc_ax.set_ylim(0, 21)
        noc_ax.set_xticks(targets)
        noc_ax.set_xlim(min(targets) - 2, max(targets) + 2)
        noc_ax.legend(frameon=False, loc="upper left", fontsize=12)

    return fig


def delta_ylim(low: list[float], high: list[float], kind: str) -> tuple[float, float]:
    low_value = min(min(low), 0.0)
    high_value = max(max(high), 0.0)
    span = max(high_value - low_value, 1.0)
    pad = max(0.07 * span, 0.35 if kind == "iou" else 0.12)
    lower = low_value - pad if low_value < 0 else 0.0
    upper = high_value + pad if high_value > 0 else 0.0
    if high_value <= 0:
        upper = max(0.2 if kind == "noc" else 0.25, 0.06 * span)
    if low_value >= 0:
        lower = -max(0.2 if kind == "noc" else 0.4, 0.06 * span)
    return lower, upper


def plot_delta_panel(
    ax: plt.Axes,
    x: list[int],
    values: dict[str, list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    kind: str,
) -> None:
    configure_axes(ax)
    delta = np.asarray(values["delta"], dtype=float)
    lower = np.asarray(values["low"], dtype=float)
    upper = np.asarray(values["high"], dtype=float)
    error = np.vstack((delta - lower, upper - delta))
    ax.errorbar(
        x,
        delta,
        yerr=error,
        fmt="o",
        color=COLORS["delimit3d"],
        ecolor=COLORS["delimit3d"],
        elinewidth=2.0,
        capsize=5,
        capthick=1.8,
        markersize=8,
    )
    ax.axhline(0, color=ZERO, linewidth=1.8, zorder=0)
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.set_xticks(x)
    ax.set_xlim(min(x) - 0.8, max(x) + 0.8)
    ax.set_ylim(*delta_ylim(values["low"], values["high"], kind))


def plot_paired(dataset: str, data: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.7))
    style_figure(fig, f"{dataset} paired improvements with 95% bootstrap intervals")
    adapted = "LitePT + Delimit3D"
    public = "LitePT (public)"

    for row, endpoint in enumerate(("MO", "SO")):
        endpoint_data = data["endpoints"][endpoint]
        plot_delta_panel(
            axes[row, 0],
            endpoint_data["clicks"],
            endpoint_data["paired_iou"],
            f"{endpoint} paired IoU change",
            "Clicks per requested object",
            f"{adapted} − {public} (pp)",
            "iou",
        )
        plot_delta_panel(
            axes[row, 1],
            endpoint_data["targets"],
            endpoint_data["paired_noc"],
            f"{endpoint} paired click change",
            "Target IoU (%)",
            f"{adapted} − {public} (clicks/object)",
            "noc",
        )

    return fig


def save_figure(fig: plt.Figure, stem: str, pdf: PdfPages) -> None:
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    title = fig._suptitle.get_text() if fig._suptitle is not None else stem
    fig.savefig(png_path, dpi=OUTPUT_DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white", metadata={"Title": title, "Creator": "Delimit3D plotting script"})
    pdf.savefig(fig, facecolor="white", metadata={"Title": title})
    plt.close(fig)


def write_provenance(dataset_data: dict[str, dict[str, Any]]) -> None:
    data_path = OUT_DIR / "delimit3d-interactive-plot-data.json"
    payload = {
        "schema": "delimit3d_matched_interactive_plots/v1",
        "generated": date.today().isoformat(),
        "arms": ARM_LABELS,
        "datasets": dataset_data,
    }
    with data_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    lines = [
        "# Matched interactive plots",
        "",
        "Generated from the completed matched-evaluation `aggregate.json` files.",
        "The two plotted arms are intentionally distinct:",
        "",
        "- `LitePT (public)`: the original public LitePT checkpoint, frozen for evaluation.",
        "- `LitePT + Delimit3D (256 updates)`: that public checkpoint after 256 Delimit3D adaptation updates, frozen before downstream decoder training.",
        "",
        "The performance panels plot aggregate mean IoU and NoC. The paired panels plot adapted-minus-public deltas with the stored 95% paired bootstrap intervals. IoU values are displayed in percent; NoC remains clicks per object.",
        "",
        "Click budgets are read directly from the aggregate files: ScanNet40 has 1, 3, 5, 10, and 15 clicks; S3DIS additionally has 20 clicks.",
        "",
        "## Source aggregates",
        "",
    ]
    for dataset, data in dataset_data.items():
        lines.extend(
            [
                f"- **{dataset}**: `{data['aggregate_path']}`",
                f"  - SHA-256: `{data['aggregate_sha256']}`",
                f"  - MO count: `{data['endpoints']['MO']['count']}`; SO count: `{data['endpoints']['SO']['count']}`",
                f"  - Paired bootstrap samples: `{data['endpoints']['MO']['bootstrap_samples']}`; seed: `{data['endpoints']['MO']['bootstrap_seed']}`",
            ]
        )
    with (OUT_DIR / "delimit3d-interactive-plots-provenance.md").open("w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 1.0,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "savefig.facecolor": "white",
        }
    )

    dataset_data = {
        dataset: load_dataset(dataset, root) for dataset, root in DATASETS.items()
    }
    write_provenance(dataset_data)

    combined_path = OUT_DIR / "delimit3d-interactive-plots.pdf"
    with PdfPages(combined_path) as combined_pdf:
        for dataset, data in dataset_data.items():
            slug = dataset.lower()
            save_figure(
                plot_performance(dataset, data),
                f"{slug}-interactive-performance",
                combined_pdf,
            )
            save_figure(
                plot_paired(dataset, data),
                f"{slug}-paired-improvements",
                combined_pdf,
            )


if __name__ == "__main__":
    main()
