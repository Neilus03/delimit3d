#!/usr/bin/env python3
"""Render a compact, mesh-topology feature-selectivity figure.

The feature caches are indexed by the vertices in the AGILE3D ScanNet PLY,
while the AGILE3D PLY itself contains only vertices and labels.  This script
therefore resolves the corresponding official ScanNet ``vh_clean_2`` mesh,
checks that its vertex array is byte-identical to the cached source vertices,
and uses only its original triangle faces.  Scores are expanded through the
cache inverse map onto those same mesh vertices for both arms.

The output is deliberately a single readable four-panel figure:

* RGB context with the selected pair marked;
* official instance colors for the pair;
* public LitePT cosine to one native token query;
* Delimit3D cosine to that identical query.

The displayed wall pair is an effect-selected illustrative example from the
precomputed pair table.  It is not a geometry-selected or population-
representative example.  All pair and query metadata are written beside the
figure.  Query self-exclusion is used for the optional retrieval metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


DEFAULT_CACHE_ROOT = Path(
    "/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1"
)
DEFAULT_OUTPUT = Path(
    "/cluster/work/igp_psr/nedela/delimit3d_feature_selectivity_scannetval_20260915"
    "/wall_gain_scene0704_00_target17_distractor24/mesh_visuals"
)
DEFAULT_MESH = Path(
    "/cluster/work/igp_psr/nedela/scannet_raw/scans/scene0704_00"
    "/scene0704_00_vh_clean_2.ply"
)
DEFAULT_SOURCE = Path(
    "/cluster/work/igp_psr/nedela/agile3d_data/ScanNet/scans/scene0704_00.ply"
)
SCENE = "scene0704_00"
TARGET_ID = 17
DISTRACTOR_ID = 24
QUERY_TOKEN = 9185
ARMS = ("public", "delimit3d")
ARM_LABELS = {
    "public": "LitePT (public)",
    "delimit3d": "LitePT + Delimit3D (256 updates)",
}
# The full cosine range is retained.  The earlier point-cloud figure used a
# clipped lower limit, which hid strongly negative adapted similarities.
COSINE_LIMITS = (-1.0, 1.0)
SEED = 20260915


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--source-ply", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--scene", default=SCENE)
    p.add_argument("--target", type=int, default=TARGET_ID)
    p.add_argument("--distractor", type=int, default=DISTRACTOR_ID)
    p.add_argument("--query-token", type=int, default=QUERY_TOKEN)
    p.add_argument("--max-context-faces", type=int, default=400000)
    p.add_argument("--max-zoom-faces", type=int, default=400000)
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--export-only", action="store_true", help="skip the expensive 3-D redraw and refresh PLY/PDF receipts from an existing PNG")
    return p.parse_args()


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def npy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    out = np.asarray(value)
    return out.astype(dtype, copy=False) if dtype is not None else out


def load_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path}: missing vertex element")
    v = ply["vertex"].data
    names = {str(n).lower(): str(n) for n in (v.dtype.names or ())}
    try:
        xyz = np.stack(
            [np.asarray(v[names[k]], dtype=np.float32) for k in ("x", "y", "z")],
            axis=1,
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing XYZ vertex field") from exc
    rgb_names = []
    for choices in (("red", "r"), ("green", "g"), ("blue", "b")):
        found = next((names[k] for k in choices if k in names), None)
        rgb_names.append(found)
    rgb = None
    if all(x is not None for x in rgb_names):
        rgb = np.stack([np.asarray(v[x], dtype=np.float32) for x in rgb_names], axis=1)
        if float(np.nanmax(rgb)) > 1.5:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    label_name = next((names[k] for k in ("label", "instance", "instance_id") if k in names), None)
    labels = np.asarray(v[label_name], dtype=np.int64) if label_name else None
    return xyz, rgb, labels


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from plyfile import PlyData

    ply = PlyData.read(str(path), known_list_len={"face": {"vertex_indices": 3}})
    xyz, rgb, _ = load_vertices(path)
    if "face" not in ply:
        raise RuntimeError(
            f"{path}: no face element. A point-cloud triangulation would fabricate topology; "
            "the mesh renderer requires the original ScanNet triangle mesh."
        )
    raw = ply["face"].data["vertex_indices"]
    try:
        faces = np.asarray(raw, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            faces = np.stack([np.asarray(x, dtype=np.int64) for x in raw], axis=0)
    except (TypeError, ValueError) as exc:
        faces = np.stack([np.asarray(x, dtype=np.int64) for x in raw], axis=0)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: expected triangular faces, got shape {faces.shape}")
    if len(faces) == 0:
        raise ValueError(f"{path}: empty face element")
    if int(faces.min()) < 0 or int(faces.max()) >= len(xyz):
        raise ValueError(f"{path}: face index outside vertex range")
    return xyz, faces, rgb


def load_cache(cache_root: Path, arm: str, scene: str) -> dict[str, Any]:
    path = cache_root / "feature_cache" / arm / f"{scene}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "delimit3d_agile3d_feature_cache/v2":
        raise ValueError(f"{path}: unexpected cache schema {payload.get('schema')!r}")
    return payload


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def average_precision(binary: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    hits = np.asarray(binary, dtype=np.int8)[order].astype(np.float64)
    if not np.any(hits):
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float64)
    return float(np.sum(precision * hits) / np.sum(hits))


def metrics_for_query(
    public: Mapping[str, Any], adapted: Mapping[str, Any], *, target: int, distractor: int, query: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target_tokens = np.asarray(public["object_token_indices"][str(target)], dtype=np.int64)
    distractor_tokens = np.asarray(public["object_token_indices"][str(distractor)], dtype=np.int64)
    if query not in set(target_tokens.tolist()):
        raise ValueError(f"query token {query} is not in target instance {target}")
    fp = normalize(npy(public["features"], np.float32))
    fa = normalize(npy(adapted["features"], np.float32))
    if fp.shape != fa.shape:
        raise ValueError(f"feature shape drift: {fp.shape} vs {fa.shape}")
    scores = {"public": fp @ fp[query], "delimit3d": fa @ fa[query]}
    candidates = np.ones(len(fp), dtype=bool)
    candidates[query] = False
    tmask = np.zeros(len(fp), dtype=bool)
    tmask[target_tokens] = True
    tmask &= candidates
    dmask = np.zeros(len(fp), dtype=bool)
    dmask[distractor_tokens] = True
    out: dict[str, Any] = {
        "query_token": int(query),
        "query_self_excluded": True,
        "target_token_count": int(tmask.sum()),
        "distractor_token_count": int(dmask.sum()),
    }
    for arm in ARMS:
        score = scores[arm]
        order = np.argsort(-score[candidates], kind="mergesort")
        candidate_ids = np.flatnonzero(candidates)
        ordered_ids = candidate_ids[order]
        k = min(int(tmask.sum()), len(ordered_ids))
        top = ordered_ids[:k]
        retrieval_target = float(np.mean(tmask[top])) if k else float("nan")
        retrieval_distractor = float(np.mean(dmask[top])) if k else float("nan")
        out[arm] = {
            "target_similarity": float(np.mean(score[tmask])),
            "same_class_distractor_similarity": float(np.mean(score[dmask])),
            "target_minus_distractor_margin": float(np.mean(score[tmask]) - np.mean(score[dmask])),
            "top_target_size_precision": retrieval_target,
            "top_target_size_same_class_leakage": retrieval_distractor,
            "target_retrieval_ap": average_precision(tmask[candidates], score[candidates]),
        }
    out["delimit3d_minus_public"] = {
        k: float(out["delimit3d"][k] - out["public"][k]) for k in out["public"]
    }
    arrays = {
        "public": scores["public"],
        "delimit3d": scores["delimit3d"],
        "target": target_tokens,
        "distractor": distractor_tokens,
    }
    return out, arrays


def choose_faces(
    faces: np.ndarray,
    xyz: np.ndarray,
    *,
    roi: tuple[np.ndarray, np.ndarray] | None,
    target_vertices: np.ndarray,
    distractor_vertices: np.ndarray,
    max_faces: int,
) -> np.ndarray:
    """Select source faces deterministically, preserving source topology."""
    if roi is None:
        in_roi = np.ones(len(xyz), dtype=bool)
    else:
        lo, hi = roi
        in_roi = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    mask = in_roi[faces].all(axis=1)
    ids = np.flatnonzero(mask)
    if len(ids) <= max_faces:
        return faces[ids]
    # Keep every source face touching either displayed official instance and
    # use a stable stride for context. No new edge or triangle is introduced.
    special_vertices = np.zeros(len(xyz), dtype=bool)
    special_vertices[target_vertices] = True
    special_vertices[distractor_vertices] = True
    special = mask & special_vertices[faces].any(axis=1)
    special_ids = np.flatnonzero(special)
    budget = max(0, int(max_faces) - len(special_ids))
    remaining_ids = np.flatnonzero(mask & ~special)
    if len(remaining_ids) > budget:
        stride = max(1, int(math.ceil(len(remaining_ids) / max(1, budget))))
        remaining_ids = remaining_ids[::stride][:budget]
    chosen = np.unique(np.concatenate([special_ids, remaining_ids]))
    return faces[chosen]


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-8)


def shade(colors: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = face_normals(vertices, faces)
    light = np.asarray([0.35, -0.55, 0.76], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(normals @ light, 0.0, 1.0)
    factor = (0.76 + 0.24 * lambert)[:, None]
    return np.clip(np.asarray(colors, dtype=np.float32) * factor, 0.0, 1.0)


def instance_colors(vertex_labels: np.ndarray, target: int, distractor: int) -> np.ndarray:
    out = np.tile(np.asarray([0.72, 0.75, 0.79], dtype=np.float32), (len(vertex_labels), 1))
    out[vertex_labels == target] = [0.075, 0.36, 0.80]
    out[vertex_labels == distractor] = [0.94, 0.44, 0.055]
    return out


def write_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    vertex_labels: np.ndarray | None = None,
    score: np.ndarray | None = None,
) -> None:
    """Write a binary PLY retaining selected original faces and vertex IDs."""
    from plyfile import PlyData, PlyElement

    color_u8 = np.clip(np.rint(np.asarray(colors) * 255.0), 0, 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if vertex_labels is not None:
        dtype.append(("instance_id", "i4"))
    if score is not None:
        dtype.append(("cosine", "f4"))
    v = np.empty(len(vertices), dtype=dtype)
    v["x"], v["y"], v["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    v["red"], v["green"], v["blue"] = color_u8[:, 0], color_u8[:, 1], color_u8[:, 2]
    if vertex_labels is not None:
        v["instance_id"] = np.asarray(vertex_labels, dtype=np.int32)
    if score is not None:
        v["cosine"] = np.asarray(score, dtype=np.float32)
    f = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = faces.astype(np.int32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")], text=False).write(str(path))


def render(
    *,
    output: Path,
    mesh_xyz: np.ndarray,
    mesh_faces: np.ndarray,
    mesh_rgb: np.ndarray | None,
    labels: np.ndarray,
    dense_scores: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    click: np.ndarray,
    target: int,
    distractor: int,
    query: int,
    context_faces: np.ndarray,
    zoom_faces: np.ndarray,
    target_vertices: np.ndarray,
    distractor_vertices: np.ndarray,
    alignment_error: float,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib import patches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.backends.backend_pdf import PdfPages

    output.mkdir(parents=True, exist_ok=True)
    face_verts = mesh_xyz[zoom_faces]
    zoom_lo = face_verts.reshape(-1, 3).min(axis=0)
    zoom_hi = face_verts.reshape(-1, 3).max(axis=0)
    zoom_center = 0.5 * (zoom_lo + zoom_hi)
    zoom_radius = max(float(np.max(zoom_hi - zoom_lo)) * 0.54, 1e-3)
    ctx_verts = mesh_xyz[context_faces]
    ctx_lo = ctx_verts.reshape(-1, 3).min(axis=0)
    ctx_hi = ctx_verts.reshape(-1, 3).max(axis=0)
    ctx_center = 0.5 * (ctx_lo + ctx_hi)
    ctx_radius = max(float(np.max(ctx_hi - ctx_lo)) * 0.53, 1e-3)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=COSINE_LIMITS[0], vmax=COSINE_LIMITS[1], clip=True)

    # Shared source topology per panel, with modest face lighting. Score color
    # is computed before shading and remains tied to the one shared scale.
    rgb_v = mesh_rgb if mesh_rgb is not None else np.tile([0.66, 0.69, 0.73], (len(mesh_xyz), 1))
    gt_v = instance_colors(labels, target, distractor)

    fig = plt.figure(figsize=(17.4, 11.6), facecolor="#f7f8fa")
    grid = fig.add_gridspec(2, 2, left=0.035, right=0.935, bottom=0.105, top=0.875, wspace=0.025, hspace=0.08)
    axes = [fig.add_subplot(grid[0, 0], projection="3d"), fig.add_subplot(grid[0, 1], projection="3d"),
            fig.add_subplot(grid[1, 0], projection="3d"), fig.add_subplot(grid[1, 1], projection="3d")]

    def panel(
        ax: Any,
        faces: np.ndarray,
        vertex_color: np.ndarray,
        *,
        title: str,
        context: bool = False,
        scalar: np.ndarray | None = None,
    ) -> None:
        if len(faces) == 0:
            raise RuntimeError(f"empty face selection for panel {title!r}")
        verts = mesh_xyz[faces]
        if scalar is None:
            face_color = shade(vertex_color[faces].mean(axis=1), mesh_xyz, faces)
        else:
            # Average the actual scalar first, then apply the shared
            # normalization. Averaging colormap RGB values would make a
            # triangle's displayed color depend on the colormap curve rather
            # than its mean cosine.
            face_color = cmap(norm(np.mean(scalar[faces], axis=1)))
        print(f"panel {title}: faces={len(faces):,} verts={verts.shape} colors={face_color.shape}", flush=True)
        # Supplying an explicit RGBA edge array avoids a Matplotlib 3-D
        # projection bug where ``edgecolors='none'`` leaves an empty 2-D
        # edge buffer for large collections. Zero linewidth keeps the output
        # visually edge-free while retaining a valid collection buffer.
        edge_color = np.asarray(face_color)
        if edge_color.shape[1] == 3:
            edge_color = np.c_[edge_color, np.ones(len(edge_color), dtype=np.float32)]
        poly = Poly3DCollection(verts, facecolors=face_color, edgecolors=edge_color, linewidths=0.0, alpha=0.98)
        poly.set_rasterized(True)
        ax.add_collection3d(poly)
        lo, hi, center, radius = (ctx_lo, ctx_hi, ctx_center, ctx_radius) if context else (zoom_lo, zoom_hi, zoom_center, zoom_radius)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_proj_type("ortho")
        # A modest optical zoom makes the scene fill the panel while keeping
        # the shared camera and limits identical across all four panels.
        ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.32 if context else 1.48)
        ax.view_init(elev=22.0, azim=-60.0)
        ax.set_axis_off()
        ax.set_title(title, fontsize=13.5, fontweight="bold", color="#18212b", pad=4)

    # Context panel: full-scene source faces, RGB with pair surfaces identified
    # by the legend and the GT panel.  Matplotlib's 3-D Poly3DCollection does
    # not reliably support an edge-only ``facecolors='none'`` overlay across
    # versions, so we keep the source faces as the sole rendered geometry.
    panel(axes[0], context_faces, rgb_v, title="RGB context · fixed scene geometry", context=True)
    axes[0].scatter([click[0]], [click[1]], [click[2]], marker="x", s=105, c="#111827", linewidths=2.0, depthshade=False)
    axes[0].text2D(0.03, 0.04, "blue = target wall 17   orange = same-class wall 24   × = query", transform=axes[0].transAxes, fontsize=10, color="#263442")

    panel(axes[1], zoom_faces, gt_v, title="Official instances · wall 17 vs wall 24")
    axes[1].scatter([click[0]], [click[1]], [click[2]], marker="x", s=115, c="#111827", linewidths=2.1, depthshade=False)
    axes[1].text2D(0.03, 0.04, "target wall 17        same-class distractor wall 24", transform=axes[1].transAxes, fontsize=10, color="#263442")

    for ax, arm in zip(axes[2:], ARMS, strict=True):
        scores = dense_scores[arm]
        panel(ax, zoom_faces, scores, title=f"{ARM_LABELS[arm]} · cosine to token {query}", scalar=scores)
        ax.scatter([click[0]], [click[1]], [click[2]], marker="x", s=115, c="#111827", linewidths=2.1, depthshade=False)
        ax.text2D(0.03, 0.04, f"target→distractor margin {metrics[arm]['target_minus_distractor_margin']:+.3f}   × query", transform=ax.transAxes, fontsize=10, color="#263442")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[2:], fraction=0.018, pad=0.012, shrink=0.82)
    cb.set_label("native dec0 cosine", fontsize=10, labelpad=8)
    cb.ax.tick_params(labelsize=9)
    legend = [patches.Patch(facecolor="#1565c0", label="target wall 17"), patches.Patch(facecolor="#ef6c00", label="same-class wall 24"), patches.Patch(facecolor="#8d98a5", label="other official instances")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.50, 0.925), ncol=3, frameon=False, fontsize=10.5)
    fig.suptitle("ScanNet40 · one native query on two separately annotated wall instances", fontsize=19, fontweight="bold", color="#14202c", y=0.966)
    fig.text(0.5, 0.930, "Illustrative effect-selected pair; identical original mesh, vertex map, query, camera, and cosine scale for both arms", ha="center", fontsize=10.5, color="#536170")
    pub, ad = metrics["public"], metrics["delimit3d"]
    footer = (
        f"Query token {query} is excluded from target-size retrieval/AP.  "
        f"Query-excluded target→distractor margin: {pub['target_minus_distractor_margin']:+.3f} → {ad['target_minus_distractor_margin']:+.3f}.  "
        f"Alignment max error: {alignment_error:.2e} m.  Original ScanNet topology: {len(mesh_faces):,} triangles."
    )
    fig.text(0.035, 0.050, footer, fontsize=9.4, color="#536170")
    fig.text(0.035, 0.031, "Scores are native 72-D dec0 features, L2-normalized per token. The pair was chosen for a strong measured margin gain and should not be read as a population estimate.", fontsize=9.1, color="#536170")

    png = output / f"mesh_feature_selectivity_{SCENE}_wall{target}_vs_wall{distractor}.png"
    pdf = output / f"mesh_feature_selectivity_{SCENE}_wall{target}_vs_wall{distractor}.pdf"
    fig.savefig(png, dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    with PdfPages(pdf) as pages:
        pages.savefig(fig, dpi=230, bbox_inches="tight")
    plt.close(fig)

    # Export the exact selected original faces with per-vertex score values.
    rgb_ply = output / f"mesh_{SCENE}_rgb_context.ply"
    gt_ply = output / f"mesh_{SCENE}_official_instances.ply"
    pub_ply = output / f"mesh_{SCENE}_public_cosine.ply"
    ad_ply = output / f"mesh_{SCENE}_delimit3d_cosine.ply"
    # Use the zoom face selection for all exported maps so the PLY opens as a
    # manageable close-up. Context remains in the figure.
    write_colored_ply(rgb_ply, mesh_xyz, zoom_faces, rgb_v, vertex_labels=labels)
    write_colored_ply(gt_ply, mesh_xyz, zoom_faces, gt_v, vertex_labels=labels)
    # PLY stores the unshaded vertex color so an interactive viewer can still
    # read the score through the explicit ``cosine`` field.  The figure adds
    # a small fixed light modulation only at face render time.
    write_colored_ply(pub_ply, mesh_xyz, zoom_faces, cmap(norm(dense_scores["public"])), vertex_labels=labels, score=dense_scores["public"])
    write_colored_ply(ad_ply, mesh_xyz, zoom_faces, cmap(norm(dense_scores["delimit3d"])), vertex_labels=labels, score=dense_scores["delimit3d"])
    return {"png": str(png), "pdf": str(pdf), "rgb_ply": str(rgb_ply), "gt_ply": str(gt_ply), "public_ply": str(pub_ply), "delimit3d_ply": str(ad_ply)}


def maybe_html(
    output: Path,
    mesh_xyz: np.ndarray,
    zoom_faces: np.ndarray,
    dense_scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    target: int,
    distractor: int,
    query: int,
    click: np.ndarray,
) -> str | None:
    """Write a self-contained Plotly HTML if Plotly is available."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except Exception:
        return None
    # Plotly Mesh3d accepts source triangle indices directly.  A single
    # interactive panel overlays the two score maps as tabs/trace toggles.
    tri = zoom_faces.astype(np.int64)
    traces = []
    for arm in ARMS:
        s = np.asarray(dense_scores[arm], dtype=np.float32)
        traces.append(
            go.Mesh3d(
                x=mesh_xyz[:, 0], y=mesh_xyz[:, 1], z=mesh_xyz[:, 2],
                i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                intensity=s, intensitymode="vertex", colorscale="Viridis", cmin=COSINE_LIMITS[0], cmax=COSINE_LIMITS[1],
                flatshading=False, name=ARM_LABELS[arm], visible=(arm == "public"), showscale=True,
                colorbar=dict(title="cosine"), hovertemplate="cosine=%{intensity:.3f}<extra></extra>",
            )
        )
    traces.append(
        go.Scatter3d(
            x=[float(click[0])], y=[float(click[1])], z=[float(click[2])],
            mode="markers", name=f"query token {query}",
            marker=dict(size=7, color="#111827", symbol="diamond", line=dict(width=1, color="white")),
            hovertemplate=f"query token {query}<extra></extra>",
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{SCENE}: wall {target} vs wall {distractor}, query token {query}",
        template="plotly_white", width=1100, height=800,
        updatemenus=[dict(type="buttons", direction="right", x=0.01, y=1.12, buttons=[
            dict(label=ARM_LABELS["public"], method="update", args=[{"visible": [True, False, True]}]),
            dict(label=ARM_LABELS["delimit3d"], method="update", args=[{"visible": [False, True, True]}]),
        ])],
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        annotations=[dict(text="Original ScanNet triangles; shared vertex correspondence", x=0.01, y=0.01, xref="paper", yref="paper", showarrow=False, font=dict(size=11, color="#536170"))],
    )
    path = output / f"mesh_feature_selectivity_{SCENE}_wall{target}_vs_wall{distractor}.html"
    plot(fig, filename=str(path), auto_open=False, include_plotlyjs=True)
    return str(path)


def main() -> None:
    a = args()
    if a.scene != SCENE or a.target != TARGET_ID or a.distractor != DISTRACTOR_ID:
        raise ValueError("This checked visual currently supports scene0704_00 wall 17 vs wall 24 only")
    cache_root = a.cache_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    mesh_path = a.mesh.expanduser().resolve(strict=True)
    source_path = a.source_ply.expanduser().resolve(strict=True)
    print(f"loading original mesh: {mesh_path}", flush=True)
    mesh_xyz, mesh_faces, mesh_rgb = load_mesh(mesh_path)
    print(f"original mesh loaded: {len(mesh_xyz):,} vertices / {len(mesh_faces):,} faces", flush=True)
    source_xyz, source_rgb, source_labels = load_vertices(source_path)
    if not np.array_equal(mesh_xyz, source_xyz):
        raise ValueError("original mesh XYZ is not byte-identical to AGILE source vertex XYZ")
    if mesh_rgb is not None and source_rgb is not None and not np.array_equal(mesh_rgb, source_rgb):
        raise ValueError("original mesh RGB is not identical to AGILE source RGB")
    if source_labels is None:
        raise ValueError("AGILE source PLY has no official vertex label field")
    labels = source_labels
    print("verified source mesh vertex identity and RGB", flush=True)
    print("loading matched public/adapted feature caches", flush=True)
    public = load_cache(cache_root, "public", a.scene)
    adapted = load_cache(cache_root, "delimit3d", a.scene)
    inv = npy(public["inverse_map"], np.int64)
    inv_a = npy(adapted["inverse_map"], np.int64)
    if not np.array_equal(inv, inv_a):
        raise ValueError("public/adapted inverse-map drift")
    reps = npy(public["representative_indices"], np.int64)
    shift = npy(public["shift"], np.float32)
    shifted_xyz = mesh_xyz - shift
    scene_xyz = npy(public["scene_xyz"], np.float32)
    if len(inv) != len(shifted_xyz) or len(labels) != len(shifted_xyz):
        raise ValueError(f"raw correspondence mismatch: inverse={len(inv)}, mesh={len(shifted_xyz)}, labels={len(labels)}")
    if len(reps) and not np.array_equal(shifted_xyz[reps], scene_xyz):
        err = float(np.max(np.abs(shifted_xyz[reps] - scene_xyz)))
        raise ValueError(f"representative/token coordinate mismatch: {err:.3e} m")
    alignment_error = 0.0
    for arm in ARMS:
        c = public if arm == "public" else adapted
        if not np.array_equal(npy(c["scene_xyz"], np.float32), scene_xyz):
            raise ValueError(f"{arm}: scene_xyz drift")
    target_vertices = np.flatnonzero(labels == a.target)
    distractor_vertices = np.flatnonzero(labels == a.distractor)
    if len(target_vertices) == 0 or len(distractor_vertices) == 0:
        raise ValueError("selected official instances have no source vertices")
    metrics, arrays = metrics_for_query(public, adapted, target=a.target, distractor=a.distractor, query=a.query_token)
    dense_scores = {arm: arrays[arm][inv] for arm in ARMS}
    # Pair-centered zoom includes both complete official instance extents and
    # a 15% margin. Faces are retained only when all vertices fall in the box,
    # avoiding clipped triangles and any fabricated geometry.
    pair_xyz = shifted_xyz[np.concatenate([target_vertices, distractor_vertices])]
    pair_lo = pair_xyz.min(axis=0)
    pair_hi = pair_xyz.max(axis=0)
    margin = max(float(np.max(pair_hi - pair_lo)) * 0.14, 0.10)
    roi = (pair_lo - margin, pair_hi + margin)
    context_faces = choose_faces(mesh_faces, shifted_xyz, roi=None, target_vertices=target_vertices, distractor_vertices=distractor_vertices, max_faces=a.max_context_faces)
    zoom_faces = choose_faces(mesh_faces, shifted_xyz, roi=roi, target_vertices=target_vertices, distractor_vertices=distractor_vertices, max_faces=a.max_zoom_faces)
    if len(zoom_faces) == 0:
        raise RuntimeError("pair ROI contains no original triangles")
    click = scene_xyz[a.query_token]
    metadata = {
        "schema": "delimit3d_mesh_feature_selectivity/v1",
        "scene": a.scene,
        "target_instance": a.target,
        "target_name": "wall",
        "distractor_instance": a.distractor,
        "distractor_name": "wall",
        "query_token": a.query_token,
        "selection_source": "effect-selected illustrative wall pair from precomputed same_class_pair_metrics.jsonl",
        "selection_warning": "This is a strong-effect example, not a geometry-selected or population-representative example.",
        "feature_stage": "native dec0 representative-token features",
        "normalization": "per-token L2 normalization before cosine",
        "fixed_cosine_limits": list(COSINE_LIMITS),
        "query_self_excluded_for_retrieval_metrics": True,
        "original_mesh_path": str(mesh_path),
        "original_mesh_sha256": sha256_file(mesh_path),
        "agile_source_path": str(source_path),
        "agile_source_sha256": sha256_file(source_path),
        "mesh_vertex_count": int(len(mesh_xyz)),
        "mesh_face_count": int(len(mesh_faces)),
        "context_face_count": int(len(context_faces)),
        "zoom_face_count": int(len(zoom_faces)),
        "mesh_vertex_identity": "np.array_equal(original_mesh_xyz, agile_source_xyz) == true",
        "representative_token_alignment_max_abs_m": float(alignment_error),
        "shared_raw_vertex_count": int(len(inv)),
        "arms": ARM_LABELS,
        "cache_root": str(cache_root),
        "metrics": metrics,
    }
    if a.export_only:
        # The full original mesh has already been inspected in the existing
        # PNG. Refresh the machine-readable mesh exports without repeating the
        # multi-minute Matplotlib 3-D projection.
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import colors as mcolors
        import matplotlib.pyplot as plt
        from PIL import Image
        output.mkdir(parents=True, exist_ok=True)
        rgb_v = mesh_rgb if mesh_rgb is not None else np.tile([0.66, 0.69, 0.73], (len(shifted_xyz), 1))
        gt_v = instance_colors(labels, a.target, a.distractor)
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=COSINE_LIMITS[0], vmax=COSINE_LIMITS[1], clip=True)
        paths = {
            "png": str(output / f"mesh_feature_selectivity_{SCENE}_wall{a.target}_vs_wall{a.distractor}.png"),
            "pdf": str(output / f"mesh_feature_selectivity_{SCENE}_wall{a.target}_vs_wall{a.distractor}.pdf"),
            "rgb_ply": str(output / f"mesh_{SCENE}_rgb_context.ply"),
            "gt_ply": str(output / f"mesh_{SCENE}_official_instances.ply"),
            "public_ply": str(output / f"mesh_{SCENE}_public_cosine.ply"),
            "delimit3d_ply": str(output / f"mesh_{SCENE}_delimit3d_cosine.ply"),
        }
        write_colored_ply(Path(paths["rgb_ply"]), shifted_xyz, zoom_faces, rgb_v, vertex_labels=labels)
        write_colored_ply(Path(paths["gt_ply"]), shifted_xyz, zoom_faces, gt_v, vertex_labels=labels)
        write_colored_ply(Path(paths["public_ply"]), shifted_xyz, zoom_faces, cmap(norm(dense_scores["public"])), vertex_labels=labels, score=dense_scores["public"])
        write_colored_ply(Path(paths["delimit3d_ply"]), shifted_xyz, zoom_faces, cmap(norm(dense_scores["delimit3d"])), vertex_labels=labels, score=dense_scores["delimit3d"])
        png_path = Path(paths["png"])
        if not png_path.exists():
            raise FileNotFoundError(f"--export-only requires an existing PNG: {png_path}")
        Image.open(png_path).convert("RGB").save(paths["pdf"], "PDF", resolution=230.0)
        plt.close("all")
    else:
        paths = render(output=output, mesh_xyz=shifted_xyz, mesh_faces=mesh_faces, mesh_rgb=mesh_rgb, labels=labels, dense_scores=dense_scores, metrics=metrics, click=click, target=a.target, distractor=a.distractor, query=a.query_token, context_faces=context_faces, zoom_faces=zoom_faces, target_vertices=target_vertices, distractor_vertices=distractor_vertices, alignment_error=alignment_error, metadata=metadata)
    html = None if a.no_html else maybe_html(output, shifted_xyz, zoom_faces, dense_scores, labels, a.target, a.distractor, a.query_token, click)
    if html:
        paths["html"] = html
    metadata["outputs"] = {k: {"path": v, "sha256": sha256_file(Path(v))} for k, v in paths.items()}
    metadata["runtime"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "matplotlib": __import__("matplotlib").__version__, "hostname": platform.node()}
    metadata["source_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    meta_path = output / "mesh_visual_receipt.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({"complete": True, "outputs": paths, "receipt": str(meta_path), "metrics": metrics}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
