#!/usr/bin/env python3
"""Build a filtered LaTeX-like/PDF/PNG table for Articulate3D features."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


SUMMARY = Path(
    "/cluster/work/igp_psr/nedela/articulate3d_delimit3d_eval_v1/"
    "metrics_feature_v1/summary.json"
)
OUT = SUMMARY.parent
TEX = OUT / "articulate3d_feature_level_table.tex"
PDF = OUT / "articulate3d_feature_level_table.pdf"
PNG = OUT / "articulate3d_feature_level_table.png"
FILTERED_JSON = OUT / "articulate3d_feature_level_table.json"

THRESHOLDS = (5, 10, 20)
METRICS = ("mAP", "cosine_margin")
METRIC_LABELS = {
    "mAP": "mAP",
    "cosine_margin": "Cosine margin",
}

CAPTION_PREFIX = (
    "Quantitative decoder-free results on the Articulate3D feature-level "
    "part-token selectivity benchmark measured at three token-eligibility "
    "thresholds."
)
CAPTION_SUFFIX = (
    "mAP is the average precision for retrieving tokens from the same "
    "annotated part against tokens from other annotated parts in the scene; cosine "
    "margin is the mean positive-minus-negative cosine similarity. Higher is "
    "better. Public LitePT is the frozen public LitePT checkpoint. LitePT + "
    "Delimit3D (256 updates) is the corresponding frozen checkpoint after 256 "
    "Delimit3D adaptation updates. For each eligible part, feature tokens are "
    "independently L2-normalized; one token is queried against same-part positives "
    "and other-part negatives. Results are means over 42 validation scenes. "
    "The ≥5, ≥10, and ≥20 thresholds specify the minimum number of unambiguous "
    "feature tokens required per part; ≥10 is the primary setting, while ≥5 and "
    "≥20 test sensitivity to the inclusion of small or sparsely represented parts. "
    "Cos. denotes cosine margin."
)
CAPTION_BODY = CAPTION_PREFIX + " " + CAPTION_SUFFIX
CAPTION = "Table 1: " + CAPTION_BODY


def fmt_mean(value: float) -> str:
    return f"{float(value):.4f}"


def fmt_delta(value: float) -> str:
    return f"{float(value):+.4f}"


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("≥", r"$\geq$")
    )


def build_filtered(summary: dict) -> dict:
    rows = []
    for threshold in THRESHOLDS:
        source = summary["thresholds"][str(threshold)]
        for metric in METRICS:
            rows.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "eligible_parts": int(source["eligible_parts_sum"]["public"]),
                    "public": float(source["public"][metric]["mean"]),
                    "delimit3d": float(source["delimit3d"][metric]["mean"]),
                    "delta": float(source["delta"][metric]["mean"]),
                }
            )
    return {
        "schema": "delimit3d_articulate3d_feature_level_table/v1",
        "source_summary": str(SUMMARY),
        "scene_count": int(summary["scene_count"]),
        "recall_metrics_included": False,
        "confidence_intervals_included": False,
        "metrics": list(METRICS),
        "thresholds": list(THRESHOLDS),
        "caption": CAPTION,
        "rows": rows,
    }


def build_tex(data: dict) -> str:
    lookup = {(row["threshold"], row["metric"]): row for row in data["rows"]}
    lines = [
        r"% Generated from the completed Articulate3D feature-only summary.",
        r"% Recall metrics and confidence intervals are deliberately omitted.",
        r"% Requires \usepackage{booktabs,multirow,graphicx} and \usepackage[table]{xcolor}.",
        r"\begin{table}[t]",
        r"\centering",
        "\\caption{\\textbf{"
        + latex_escape(CAPTION_PREFIX)
        + "} "
        + latex_escape(CAPTION_SUFFIX)
        + "}",
        r"\label{tab:articulate3d-feature-level}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.5}",
        r"\begin{tabular}{cl|rrrrrr}",
        r"\toprule",
        r" & \textbf{Method} & \textbf{mAP}$_{\geq5}$ $\uparrow$ & \textbf{mAP}$_{\geq10}$ $\uparrow$ & \textbf{mAP}$_{\geq20}$ $\uparrow$ & \textbf{Cos.}$_{\geq5}$ $\uparrow$ & \textbf{Cos.}$_{\geq10}$ $\uparrow$ & \textbf{Cos.}$_{\geq20}$ $\uparrow$ \\",
        r"\midrule",
    ]
    public_values = [
        fmt_mean(lookup[(threshold, "mAP")]["public"]) for threshold in THRESHOLDS
    ] + [
        fmt_mean(lookup[(threshold, "cosine_margin")]["public"])
        for threshold in THRESHOLDS
    ]
    delimit_values = [
        fmt_mean(lookup[(threshold, "mAP")]["delimit3d"]) for threshold in THRESHOLDS
    ] + [
        fmt_mean(lookup[(threshold, "cosine_margin")]["delimit3d"])
        for threshold in THRESHOLDS
    ]
    lines.append(
        r"\multirow{2}{*}{\rotatebox[origin=c]{90}{\textbf{Articulate3D}}}"
        + " & LitePT (public) & "
        + " & ".join(public_values)
        + r" \\"
    )
    lines.append(
        r"\rowcolor{black!4} \cellcolor{white} & \textbf{LitePT + Delimit3D (256 updates)} & \textbf{"
        + "} & \\textbf{".join(delimit_values)
        + r"} \\"
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_visual(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )

    lookup = {(row["threshold"], row["metric"]): row for row in data["rows"]}
    fig = plt.figure(figsize=(8.0, 5.0), facecolor="white")
    ax = fig.add_axes([0.018, 0.40, 0.964, 0.30])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # The geometry mirrors the reference: caption above, sparse rules, and a
    # vertical benchmark label beside the two method rows.
    widths = [0.055, 0.315] + [0.105] * 6
    x_edges = [0.0]
    for width in widths:
        x_edges.append(x_edges[-1] + width)
    y_top = 1.0
    header_height = 0.22
    body_height = 0.39
    y_body_top = y_top - header_height
    y_bottom = y_body_top - 2 * body_height
    ink = "#222222"
    light_ink = "#333333"
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x_edges[1], y_bottom), 1 - x_edges[1], body_height,
                           facecolor="#f5f5f5", edgecolor="none", zorder=-1))

    ax.plot([x_edges[0], x_edges[-1]], [y_top, y_top], color=ink, linewidth=1.2)
    ax.plot(
        [x_edges[0], x_edges[-1]],
        [y_body_top, y_body_top],
        color=ink,
        linewidth=0.85,
    )
    ax.plot([x_edges[0], x_edges[-1]], [y_bottom, y_bottom], color=ink, linewidth=1.2)
    ax.plot(
        [x_edges[2], x_edges[2]],
        [y_bottom, y_top],
        color=ink,
        linewidth=0.85,
    )

    header_y = y_top - header_height / 2
    ax.text(
        (x_edges[1] + x_edges[2]) / 2,
        header_y,
        "Method",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=ink,
    )
    ax.text(
        (x_edges[2] + x_edges[3]) / 2,
        y_top - header_height / 2,
        r"mAP$_{\geq5}$ ↑",
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=ink,
    )
    ax.text(
        (x_edges[3] + x_edges[4]) / 2,
        y_top - header_height / 2,
        r"mAP$_{\geq10}$ ↑",
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=ink,
    )
    for column, header in zip(
        range(4, 8),
        (r"mAP$_{\geq20}$ ↑", r"Cos.$_{\geq5}$ ↑", r"Cos.$_{\geq10}$ ↑", r"Cos.$_{\geq20}$ ↑"),
        strict=True,
    ):
        ax.text(
            (x_edges[column] + x_edges[column + 1]) / 2,
            y_top - header_height / 2,
            header,
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color=ink,
        )

    row_centers = [
        y_body_top - body_height / 2,
        y_body_top - 3 * body_height / 2,
    ]
    ax.text(
        (x_edges[0] + x_edges[1]) / 2,
        sum(row_centers) / 2,
        "Articulate3D",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.2,
        fontweight="bold",
        color=ink,
    )
    methods = ["LitePT (public)", "LitePT + Delimit3D (256 updates)"]
    for row_index, (method, row_y) in enumerate(zip(methods, row_centers, strict=True)):
        bold = row_index == 1
        ax.text(
            x_edges[1] + 0.012,
            row_y,
            method,
            ha="left",
            va="center",
            fontsize=7.7,
            fontweight="bold" if bold else "normal",
            color=ink,
        )
        values = [
            lookup[(threshold, "mAP")]["delimit3d" if bold else "public"]
            for threshold in THRESHOLDS
        ] + [
            lookup[(threshold, "cosine_margin")]["delimit3d" if bold else "public"]
            for threshold in THRESHOLDS
        ]
        for offset, value in enumerate(values, start=2):
            ax.text(
                (x_edges[offset] + x_edges[offset + 1]) / 2,
                row_y,
                fmt_mean(value),
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold" if bold else "normal",
                color=light_ink,
            )

    # Wrap using rendered text widths, then position the table from the actual
    # caption bounds so caption edits cannot create overlaps or large gaps.
    from matplotlib.font_manager import FontProperties
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def wrap_caption(value: str, weight: str) -> str:
        font = FontProperties(family="DejaVu Serif", size=7.8, weight=weight)
        limit = fig.bbox.width * 0.96
        lines, words = [], []
        for word in value.split():
            candidate = " ".join(words + [word])
            if words and renderer.get_text_width_height_descent(candidate, font, False)[0] > limit:
                lines.append(" ".join(words))
                words = []
            words.append(word)
        lines.append(" ".join(words))
        return "\n".join(lines)
    prefix = fig.text(
        0.02,
        0.965,
        wrap_caption("Table 1: " + CAPTION_PREFIX, "bold"),
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=light_ink,
        linespacing=1.0,
    )
    fig.canvas.draw()
    prefix_bottom = prefix.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
    suffix = fig.text(
        0.02,
        prefix_bottom - 0.012,
        wrap_caption(CAPTION_SUFFIX, "normal"),
        ha="left",
        va="top",
        fontsize=7.8,
        color=light_ink,
        linespacing=1.0,
    )
    fig.canvas.draw()
    caption_bottom = suffix.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
    table_height = 0.24
    ax.set_position([0.018, caption_bottom - 0.035 - table_height, 0.964, table_height])
    fig.savefig(PDF, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PNG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    if summary.get("status") != "complete" or int(summary.get("scene_count", 0)) != 42:
        raise RuntimeError("feature summary is not the completed 42-scene result")
    data = build_filtered(summary)
    FILTERED_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    TEX.write_text(build_tex(data))
    build_visual(data)
    print(json.dumps({"pdf": str(PDF), "png": str(PNG), "tex": str(TEX)}, indent=2))


if __name__ == "__main__":
    main()
