#!/usr/bin/env python3
"""Create reproducible quantitative and qualitative AGILE3D-style plots.

The script consumes an evaluation output root containing public/ and delimit3d/
evaluation/<panel> traces plus aggregate.json. It can be run on an interim
output root or on the final experiment root without changing the plots.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

CLICK_VALUES = (1, 3, 5, 10, 15, 20)
NOC_VALUES = (0.50, 0.65, 0.80, 0.85, 0.90)
ARM_LABELS = {"public": "Public LitePT", "delimit3d": "Delimit3D"}
ARM_COLORS = {"public": "#3366cc", "delimit3d": "#d95f02"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(path: str | Path) -> Path:
    """Resolve paths recorded through the pf-pc69 Euler SSHFS mount."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    text = str(candidate)
    mount = "/tmp/euler_cluster_nedela_rw"
    if mount in text:
        alternative = Path(text.replace(mount, "/cluster", 1))
        if alternative.exists():
            return alternative
    return candidate


def find_experiment_root(run_root: Path) -> Path:
    # Interim runs carry symlinks to the manifest and caches, but their
    # training logs live in the parent experiment root. Prefer a root that
    # actually contains the arm logs, then fall back to the manifest parent.
    if (run_root / "public" / "train_log.jsonl").exists() or (run_root / "delimit3d" / "train_log.jsonl").exists():
        return run_root
    if (run_root.parent / "selection_manifest.json").exists():
        return run_root.parent
    if (run_root / "selection_manifest.json").exists():
        return run_root
    return run_root


def panel_root(run_root: Path, arm: str, panel: str) -> Path:
    return run_root / arm / "evaluation" / panel


def load_panel(run_root: Path, arm: str, panel: str) -> list[dict[str, Any]]:
    path = panel_root(run_root, arm, panel) / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing evaluation trace: {path}")
    return read_jsonl(path)


def load_manifest(run_root: Path) -> dict[str, Any] | None:
    experiment_root = find_experiment_root(run_root)
    path = experiment_root / "selection_manifest.json"
    return read_json(path) if path.exists() else None


def metric_report(aggregate: dict[str, Any], panel: str) -> dict[str, Any]:
    try:
        return aggregate["panels"][panel]
    except KeyError as exc:
        raise KeyError(f"aggregate does not contain panel {panel!r}") from exc


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8a929d")
    ax.spines["bottom"].set_color("#8a929d")
    ax.tick_params(colors="#39414d", labelsize=9)
    ax.xaxis.label.set_color("#20252b")
    ax.yaxis.label.set_color("#20252b")
    ax.title.set_color("#20252b")


def plot_performance(aggregate: dict[str, Any], panel: str, out_dir: Path, tag: str) -> None:
    report = metric_report(aggregate, panel)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_iou, ax_delta, ax_noc, ax_dist = axes.flat

    for ax in axes.flat:
        style_axes(ax)

    x = np.asarray(CLICK_VALUES, dtype=float)
    for arm in ("public", "delimit3d"):
        values = [100.0 * float(report[arm][f"iou@{c}"]) for c in CLICK_VALUES]
        ax_iou.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    ax_iou.set_title(f"ScanNet++ {panel}: interactive quality")
    ax_iou.set_xlabel("Clicks per requested object")
    ax_iou.set_ylabel("Mean IoU (%)")
    ax_iou.set_xticks(CLICK_VALUES)
    ax_iou.legend(frameon=False, loc="upper left")
    ax_iou.annotate(
        f"+{100 * (report['delimit3d']['iou@1'] - report['public']['iou@1']):.2f} pp at 1 click",
        xy=(1, 100 * report["delimit3d"]["iou@1"]),
        xytext=(2.2, 100 * report["delimit3d"]["iou@1"] + 3.5),
        arrowprops={"arrowstyle": "->", "color": ARM_COLORS["delimit3d"], "lw": 1.0},
        fontsize=9,
        color="#30343b",
    )

    deltas = np.asarray([100.0 * float(report["bootstrap"][f"iou@{c}"]["observed_difference"]) for c in CLICK_VALUES])
    lower = np.asarray([100.0 * float(report["bootstrap"][f"iou@{c}"]["ci95_lower"]) for c in CLICK_VALUES])
    upper = np.asarray([100.0 * float(report["bootstrap"][f"iou@{c}"]["ci95_upper"]) for c in CLICK_VALUES])
    ax_delta.fill_between(x, lower, upper, color=ARM_COLORS["delimit3d"], alpha=0.18, label="Paired 95% scene bootstrap")
    ax_delta.plot(x, deltas, marker="o", color=ARM_COLORS["delimit3d"], linewidth=2.2, label="Delimit3D − Public")
    ax_delta.axhline(0.0, color="#59636e", linewidth=1.0)
    ax_delta.axhline(2.0, color="#7b8794", linewidth=1.0, linestyle="--", alpha=0.9)
    ax_delta.axhline(1.5, color="#7b8794", linewidth=1.0, linestyle=":", alpha=0.9)
    ax_delta.text(20.0, 2.15, "primary IoU@1 gate", fontsize=8, ha="right", color="#59636e")
    ax_delta.text(20.0, 1.65, "primary IoU@5 gate", fontsize=8, ha="right", color="#59636e")
    ax_delta.set_title("Paired improvement with interaction")
    ax_delta.set_xlabel("Clicks per requested object")
    ax_delta.set_ylabel("IoU difference (percentage points)")
    ax_delta.set_xticks(CLICK_VALUES)
    ax_delta.legend(frameon=False, loc="upper left", fontsize=8)

    target = np.asarray([100.0 * v for v in NOC_VALUES])
    for arm in ("public", "delimit3d"):
        values = [float(report[arm][f"noc@{v:.2f}"]) for v in NOC_VALUES]
        ax_noc.plot(target, values, marker="o", linewidth=2.2, markersize=5, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax_noc.set_title("Click efficiency")
    ax_noc.set_xlabel("Target IoU (%)")
    ax_noc.set_ylabel("NoC (clicks per object)")
    ax_noc.set_xticks(target)
    ax_noc.legend(frameon=False, loc="lower right")

    scene_diffs_1 = 100.0 * np.asarray(list(report["bootstrap"]["iou@1"]["scene_differences"].values()), dtype=float)
    scene_diffs_5 = 100.0 * np.asarray(list(report["bootstrap"]["iou@5"]["scene_differences"].values()), dtype=float)
    scene_diffs_10 = 100.0 * np.asarray(list(report["bootstrap"]["iou@10"]["scene_differences"].values()), dtype=float)
    box = ax_dist.boxplot(
        [scene_diffs_1, scene_diffs_5, scene_diffs_10],
        positions=[1, 2, 3],
        widths=0.48,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#20252b", "markeredgecolor": "#20252b", "markersize": 4},
        medianprops={"color": "#20252b", "linewidth": 1.2},
        whiskerprops={"color": "#59636e"},
        capprops={"color": "#59636e"},
        flierprops={"marker": ".", "markerfacecolor": "#7b8794", "markeredgecolor": "#7b8794", "alpha": 0.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(ARM_COLORS["delimit3d"])
        patch.set_alpha(0.28)
        patch.set_edgecolor(ARM_COLORS["delimit3d"])
    rng = np.random.default_rng(20260911)
    for i, values in enumerate((scene_diffs_1, scene_diffs_5, scene_diffs_10), start=1):
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax_dist.scatter(np.full(len(values), i) + jitter, values, s=12, alpha=0.42, color=ARM_COLORS["delimit3d"], edgecolors="none")
    ax_dist.axhline(0.0, color="#59636e", linewidth=1.0)
    ax_dist.set_title("Scene-level paired improvements")
    ax_dist.set_xlabel("Interaction checkpoint")
    ax_dist.set_ylabel("Delimit3D − Public IoU (pp)")
    ax_dist.set_xticks([1, 2, 3], ["1 click", "5 clicks", "10 clicks"])

    fig.suptitle(
        f"Delimit3D interactive segmentation comparison — {panel} — {tag}",
        fontsize=16,
        fontweight="bold",
        color="#20252b",
    )
    fig.text(0.01, 0.005, "Frozen encoders; identical fresh multi-object decoder; paired ScanNet++ scenes.", fontsize=8.5, color="#59636e")
    fig.savefig(out_dir / "interactive_performance.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "interactive_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def _scene_rows(run_root: Path, arm: str, panel: str) -> dict[str, dict[str, Any]]:
    path = panel_root(run_root, arm, panel) / "scenes.jsonl"
    if not path.exists():
        return {}
    return {str(row["scene"]): row for row in read_jsonl(path)}


def plot_scene_heatmap(run_root: Path, aggregate: dict[str, Any], panel: str, out_dir: Path, tag: str) -> None:
    report = metric_report(aggregate, panel)
    scene_names = sorted(report["bootstrap"]["iou@5"]["scene_differences"])
    rows_public = _scene_rows(run_root, "public", panel)
    rows_delimit = _scene_rows(run_root, "delimit3d", panel)
    matrix = []
    for scene in scene_names:
        if scene not in rows_public or scene not in rows_delimit:
            continue
        values = []
        for c in CLICK_VALUES:
            p = rows_public[scene]["thresholds"].get(str(c), {}).get("mean_iou")
            d = rows_delimit[scene]["thresholds"].get(str(c), {}).get("mean_iou")
            values.append(np.nan if p is None or d is None else 100.0 * (float(d) - float(p)))
        matrix.append((scene, values))
    matrix.sort(key=lambda item: np.nan_to_num(item[1][2], nan=-999.0), reverse=True)
    names = [item[0] for item in matrix]
    data = np.asarray([item[1] for item in matrix], dtype=float)
    fig_height = max(7.0, 0.23 * len(names) + 1.9)
    fig, ax = plt.subplots(figsize=(10.6, fig_height), constrained_layout=True)
    style_axes(ax)
    vmax = max(1.0, float(np.nanmax(np.abs(data)))) if data.size else 1.0
    image = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"Scene-level IoU change — {panel} — {tag}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clicks per requested object")
    ax.set_ylabel("Validation scene (sorted by IoU@5 gain)")
    ax.set_xticks(np.arange(len(CLICK_VALUES)), [str(c) for c in CLICK_VALUES])
    ax.set_yticks(np.arange(len(names)), names)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Delimit3D − Public IoU (percentage points)")
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_dir / "scene_delta_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "scene_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(experiment_root: Path, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    style_axes(ax)
    for arm in ("public", "delimit3d"):
        path = experiment_root / arm / "train_log.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        updates = np.asarray([int(row["update"]) for row in rows], dtype=int)
        losses = np.asarray([float(row["loss"]) for row in rows], dtype=float)
        ax.plot(updates, losses, color=ARM_COLORS[arm], alpha=0.14, linewidth=0.7)
        if len(losses) >= 25:
            kernel = np.ones(25, dtype=float) / 25.0
            smooth = np.convolve(losses, kernel, mode="valid")
            ax.plot(updates[24:], smooth, color=ARM_COLORS[arm], linewidth=2.0, label=ARM_LABELS[arm])
    ax.set_title(f"Decoder training loss — completed logs — {tag}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Total decoder loss")
    ax.legend(frameon=False)
    fig.text(0.01, 0.005, "Thin lines show per-episode loss; thick lines show a 25-update moving average.", fontsize=8.5, color="#59636e")
    fig.savefig(out_dir / "training_loss.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "training_loss.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_scene_record(manifest: dict[str, Any], scene: str) -> dict[str, Any]:
    for record in list(manifest.get("validation_scenes", [])) + list(manifest.get("train_scenes", [])):
        if str(record.get("scene")) == scene:
            return record
    raise KeyError(f"scene {scene} is absent from selection_manifest.json")


def _state_for_clicks(row: dict[str, Any], clicks_per_object: int) -> tuple[dict[str, Any], int]:
    desired = int(clicks_per_object) * int(row["object_count"])
    states = list(row["states"])
    state = min(states, key=lambda item: abs(int(item["total_clicks"]) - desired))
    return state, int(state["state"])


def _load_cache(experiment_root: Path, arm: str, scene: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    path = experiment_root / "feature_cache" / arm / f"{scene}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing feature cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    inverse = payload["inverse_map"]
    xyz = payload["scene_xyz"]
    if hasattr(inverse, "numpy"):
        inverse = inverse.numpy()
    if hasattr(xyz, "numpy"):
        xyz = xyz.numpy()
    return np.asarray(inverse, dtype=np.int64), np.asarray(xyz, dtype=np.float32)


def _display_indices(target: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(target)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    picks: list[np.ndarray] = []
    labels = np.unique(target)
    per_object = max(800, min(2500, max_points // max(1, len(labels) * 2)))
    used = 0
    for label in labels:
        ids = np.flatnonzero(target == label)
        if len(ids) == 0:
            continue
        take = min(len(ids), per_object if label != 0 else max(1000, per_object // 2))
        chosen = rng.choice(ids, size=take, replace=False) if len(ids) > take else ids
        picks.append(np.asarray(chosen, dtype=np.int64))
        used += len(chosen)
    remaining = max_points - used
    if remaining > 0:
        mask = np.ones(n, dtype=bool)
        if picks:
            mask[np.concatenate(picks)] = False
        ids = np.flatnonzero(mask)
        if len(ids) > remaining:
            ids = rng.choice(ids, size=remaining, replace=False)
        picks.append(np.asarray(ids, dtype=np.int64))
    return np.concatenate(picks)[:max_points]


def _label_colors(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), 0.78, dtype=np.float32)
    foreground = labels > 0
    if np.any(foreground):
        colors[foreground] = palette[(labels[foreground] - 1) % len(palette)]
    return colors


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.52 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()


def plot_scene_panel(run_root: Path, aggregate: dict[str, Any], panel: str, out_dir: Path, tag: str, scene: str | None = None) -> str:
    manifest = load_manifest(run_root)
    if manifest is None:
        raise FileNotFoundError("selection_manifest.json is required for qualitative scene panels")
    report = metric_report(aggregate, panel)
    if scene is None:
        scene = max(report["bootstrap"]["iou@5"]["scene_differences"], key=report["bootstrap"]["iou@5"]["scene_differences"].get)
    rows = {arm: next(row for row in load_panel(run_root, arm, panel) if str(row["scene"]) == scene) for arm in ("public", "delimit3d")}
    record = _load_scene_record(manifest, scene)
    data_path = resolve_path(str(record["data_path"]))
    data = np.load(data_path, allow_pickle=True)
    points = np.asarray(data["points"], dtype=np.float32)
    rgb = np.asarray(data["colors"], dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    target = np.zeros(len(points), dtype=np.int16)
    for local, instance in enumerate(rows["public"]["panel_objects"], start=1):
        indices = np.asarray(data[f"target_{int(instance)}"], dtype=np.int64)
        target[indices] = local
    keep = _display_indices(target, max_points=26000, seed=20260911)
    experiment_root = find_experiment_root(run_root)
    inverse_by_arm = {arm: _load_cache(experiment_root, arm, scene)[0] for arm in ("public", "delimit3d")}
    xyz_by_arm = {arm: _load_cache(experiment_root, arm, scene)[1] for arm in ("public", "delimit3d")}
    colors = np.asarray(plt.get_cmap("tab20")(np.linspace(0.02, 0.96, 20))[:, :3], dtype=np.float32)
    columns = (1, 5, 10, 20)
    fig = plt.figure(figsize=(16.2, 11.0), constrained_layout=False)
    grid = fig.add_gridspec(4, 4, left=0.01, right=0.99, bottom=0.045, top=0.91, wspace=0.01, hspace=0.04)
    row_titles = ["RGB input", "Ground truth", "Public LitePT", "Delimit3D"]
    for col, clicks in enumerate(columns):
        for row_idx, mode in enumerate(("rgb", "gt", "public", "delimit3d")):
            ax = fig.add_subplot(grid[row_idx, col], projection="3d")
            if mode == "rgb":
                display = rgb[keep]
            elif mode == "gt":
                display = _label_colors(target[keep], colors)
            else:
                arm = mode
                state, state_index = _state_for_clicks(rows[arm], clicks)
                mask_path = Path(rows[arm]["mask_path"])
                with np.load(mask_path, allow_pickle=False) as masks:
                    token_prediction = np.asarray(masks[f"state_{state_index:03d}"], dtype=np.int16)
                raw_prediction = token_prediction[inverse_by_arm[arm]]
                display = _label_colors(raw_prediction[keep], colors)
                click_xyz: list[np.ndarray] = []
                for values in state.get("clicks", {}).values():
                    for token in values:
                        token = int(token)
                        if 0 <= token < len(xyz_by_arm[arm]):
                            click_xyz.append(xyz_by_arm[arm][token])
                if click_xyz:
                    click_xyz_array = np.asarray(click_xyz)
                    ax.scatter(click_xyz_array[:, 0], click_xyz_array[:, 1], click_xyz_array[:, 2], s=11, c="#111111", marker="x", linewidths=0.8, depthshade=False)
            ax.scatter(points[keep, 0], points[keep, 1], points[keep, 2], s=0.35 if mode != "rgb" else 0.22, c=display, linewidths=0, depthshade=False)
            _set_equal_3d(ax, points)
            if row_idx == 0:
                ax.set_title(f"{clicks} click{'s' if clicks != 1 else ''}/object", fontsize=11, pad=2, color="#20252b")
            if col == 0:
                ax.text2D(-0.02, 0.5, row_titles[row_idx], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10, color="#20252b")
    fig.suptitle(
        f"ScanNet++ interactive multi-object segmentation — scene {scene} — {panel} — {tag}",
        fontsize=15,
        fontweight="bold",
        color="#20252b",
    )
    fig.text(0.01, 0.012, "Colored foreground labels are local object IDs 1–5; grey is background. Black crosses mark simulated clicks.", fontsize=8.5, color="#59636e")
    output_name = f"scene_panel_{scene}.png"
    fig.savefig(out_dir / output_name, dpi=220, bbox_inches="tight")
    # The dense 3D qualitative panel is intentionally delivered as a raster
    # figure; vector PDF rendering is extremely slow and produces no useful
    # additional information for this diagnostic.
    plt.close(fig)
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="interim or final evaluation output root")
    parser.add_argument("--panel", default="MO-5", choices=("MO-1", "MO-5", "MO-10"))
    parser.add_argument("--output", type=Path, help="output directory; defaults to <run-root>/visuals")
    parser.add_argument("--tag", default="evaluation", help="short tag included in plot titles")
    parser.add_argument("--scene", help="scene ID for qualitative panel; defaults to largest IoU@5 gain")
    parser.add_argument("--skip-qualitative", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = (args.output or run_root / "visuals").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate_path = run_root / "aggregate.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"missing aggregate report: {aggregate_path}")
    aggregate = read_json(aggregate_path)
    plot_performance(aggregate, args.panel, output, args.tag)
    plot_scene_heatmap(run_root, aggregate, args.panel, output, args.tag)
    plot_training_loss(find_experiment_root(run_root), output, args.tag)
    selected_scene = None
    if not args.skip_qualitative:
        selected_scene = plot_scene_panel(run_root, aggregate, args.panel, output, args.tag, args.scene)
    summary = {
        "run_root": str(run_root),
        "panel": args.panel,
        "tag": args.tag,
        "selected_scene": selected_scene,
        "files": sorted(str(path) for path in output.iterdir() if path.is_file()),
    }
    (output / "visuals_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
