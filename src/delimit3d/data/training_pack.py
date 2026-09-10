"""Load and validate a single Delimit3D training-pack directory.

This is the most important data file.  It reads the public training-pack
contract (``scene_meta.json`` + ``.npy`` arrays), validates it, and returns a
clean :class:`TrainingPackScene` dataclass ready for downstream consumption.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SUPPORTED_PACK_VERSIONS = {"1.0"}

REQUIRED_META_FIELDS = (
    "scene_id",
    "num_points",
    "granularities",
    "label_files",
)

PACK_DIR_NAMES = ("training_pack", "litept_pack")


# ── helpers ──────────────────────────────────────────────────────────────


def _resolve_pack_dir(path: Path) -> Path:
    """Accept a pack dir directly, or a parent scene dir containing one.

    Priority: ``training_pack/`` > ``litept_pack/``.
    If *path* itself already contains ``scene_meta.json`` it is returned as-is.
    """
    if (path / "scene_meta.json").is_file():
        return path
    for name in PACK_DIR_NAMES:
        candidate = path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No training pack found at or under {path} "
        f"(looked for {PACK_DIR_NAMES})"
    )


def _load_scene_meta(pack_dir: Path) -> dict[str, Any]:
    meta_path = pack_dir / "scene_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"scene_meta.json not found in {pack_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)

    for field in REQUIRED_META_FIELDS:
        if field not in meta:
            raise KeyError(
                f"scene_meta.json in {pack_dir} missing required field: {field}"
            )

    version = meta.get("pack_version")
    if version is not None and version not in SUPPORTED_PACK_VERSIONS:
        raise ValueError(
            f"Unsupported pack_version {version!r} "
            f"(supported: {SUPPORTED_PACK_VERSIONS})"
        )

    return meta


def _resolve_label_file(
    scene_meta: dict[str, Any],
    granularity: float,
    pack_dir: Path,
) -> Path:
    label_file_map: dict[str, str] = scene_meta["label_files"]
    g_key = f"g{granularity}"
    if g_key not in label_file_map:
        raise KeyError(
            f"Requested granularity {granularity} (key={g_key!r}) not found "
            f"in label_files; available: {sorted(label_file_map)}"
        )
    label_path = pack_dir / label_file_map[g_key]
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file declared in scene_meta but missing: {label_path}"
        )
    return label_path


def _validate_scene_arrays(
    *,
    num_points_declared: int,
    points: np.ndarray,
    colors: np.ndarray | None,
    normals: np.ndarray | None,
    labels: np.ndarray,
    valid_points: np.ndarray,
    seen_points: np.ndarray,
    supervision_mask: np.ndarray,
) -> None:
    N = num_points_declared

    if points.shape != (N, 3):
        raise ValueError(
            f"points.npy shape {points.shape}, expected ({N}, 3)"
        )
    if colors is not None and colors.shape[0] != N:
        raise ValueError(
            f"colors.npy has {colors.shape[0]} rows, expected {N}"
        )
    if normals is not None:
        if normals.shape != (N, 3):
            raise ValueError(
                f"normals.npy shape {normals.shape}, expected ({N}, 3)"
            )

    for name, arr in [
        ("labels", labels),
        ("valid_points", valid_points),
        ("seen_points", seen_points),
        ("supervision_mask", supervision_mask),
    ]:
        if arr.shape != (N,):
            raise ValueError(f"{name} shape {arr.shape}, expected ({N},)")

    if supervision_mask.sum() == 0:
        raise ValueError("supervision_mask is all-zero — no supervised points")

    supervised_labels = labels[supervision_mask.astype(bool)]
    if (supervised_labels >= 0).sum() == 0:
        raise ValueError(
            "No non-negative labels under supervision_mask — "
            "pseudo-labels may be broken"
        )


def _load_seen_points_with_legacy_fallback(
    pack_dir: Path,
    valid_points: np.ndarray,
) -> np.ndarray:
    """Load seen-points mask with legacy litept fallback.

    Some older ``litept_pack`` exports do not include ``seen_points.npy``.
    For backward compatibility we fall back to ``valid_points`` so existing
    training code can proceed.
    """
    seen_points_path = pack_dir / "seen_points.npy"
    if seen_points_path.exists():
        return np.load(seen_points_path).astype(bool)

    log.warning(
        "[%s] missing seen_points.npy; falling back to valid_points.npy "
        "(legacy litept_pack compatibility)",
        pack_dir,
    )
    return valid_points.copy()


# ── main dataclass ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingPackScene:
    """All arrays and metadata from one exported training pack,
    resolved to a single granularity."""

    scene_id: str
    scene_dir: Path
    training_pack_dir: Path

    points: np.ndarray               # (N, 3) float
    colors: np.ndarray | None        # (N, 3) float/uint8 or None
    normals: np.ndarray | None     # (N, 3) float — optional vertex normals
    labels: np.ndarray               # (N,)   int  — single granularity
    valid_points: np.ndarray         # (N,)   bool
    seen_points: np.ndarray          # (N,)   bool
    supervision_mask: np.ndarray     # (N,)   bool

    scene_meta: dict[str, Any]
    granularity: float

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class MultiGranTrainingPackScene:
    """Training-pack with labels loaded for multiple granularities at once.

    Geometry, masks, and metadata are shared; only labels differ per
    granularity.
    """

    scene_id: str
    scene_dir: Path
    training_pack_dir: Path

    points: np.ndarray               # (N, 3) float
    colors: np.ndarray | None        # (N, 3) float/uint8 or None
    normals: np.ndarray | None       # (N, 3) float — optional vertex normals
    labels_by_granularity: dict[str, np.ndarray]  # {"g02": (N,), ...}
    valid_points: np.ndarray         # (N,)   bool
    seen_points: np.ndarray          # (N,)   bool
    supervision_mask: np.ndarray     # (N,)   bool

    scene_meta: dict[str, Any]
    granularities: tuple[str, ...]   # ("g02", "g05", "g08")

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


# ── main loader ──────────────────────────────────────────────────────────


def load_training_pack_scene(
    scene_dir: str | Path,
    granularity: float,
) -> TrainingPackScene:
    """Load a training-pack for one scene at one granularity.

    Parameters
    ----------
    scene_dir:
        Path to the scene directory, or directly to the ``training_pack/``
        / ``litept_pack/`` subdirectory.
    granularity:
        Which granularity to load (must exist in ``scene_meta["label_files"]``).
    """
    scene_dir = Path(scene_dir)
    pack_dir = _resolve_pack_dir(scene_dir)

    meta = _load_scene_meta(pack_dir)
    num_points_declared: int = meta["num_points"]

    # ── geometry ──
    points = np.load(pack_dir / "points.npy")

    opt = meta.get("optional_files_present", {})
    has_colors = opt.get("colors.npy", (pack_dir / "colors.npy").exists())
    colors: np.ndarray | None = None
    if has_colors:
        colors_path = pack_dir / "colors.npy"
        if not colors_path.exists():
            raise FileNotFoundError(
                "scene_meta declares colors.npy present but file is missing"
            )
        colors = np.load(colors_path)

    normals_path = pack_dir / "normals.npy"
    has_normals = opt.get("normals.npy", normals_path.exists())
    normals: np.ndarray | None = None
    if has_normals:
        if not normals_path.exists():
            raise FileNotFoundError(
                "scene_meta declares normals.npy present but file is missing"
            )
        normals = np.load(normals_path).astype(np.float32, copy=False)

    # ── labels ──
    label_path = _resolve_label_file(meta, granularity, pack_dir)
    labels = np.load(label_path)

    # ── masks ──
    valid_points = np.load(pack_dir / "valid_points.npy").astype(bool)
    seen_points = _load_seen_points_with_legacy_fallback(pack_dir, valid_points)
    supervision_mask = np.load(pack_dir / "supervision_mask.npy").astype(bool)

    _validate_scene_arrays(
        num_points_declared=num_points_declared,
        points=points,
        colors=colors,
        normals=normals,
        labels=labels,
        valid_points=valid_points,
        seen_points=seen_points,
        supervision_mask=supervision_mask,
    )

    # scene_dir for TrainingPackScene: if the user passed the pack dir
    # directly, go one level up to get the scene directory.
    if scene_dir == pack_dir:
        resolved_scene_dir = pack_dir.parent
    else:
        resolved_scene_dir = scene_dir

    return TrainingPackScene(
        scene_id=meta["scene_id"],
        scene_dir=resolved_scene_dir,
        training_pack_dir=pack_dir,
        points=points,
        colors=colors,
        normals=normals,
        labels=labels,
        valid_points=valid_points,
        seen_points=seen_points,
        supervision_mask=supervision_mask,
        scene_meta=meta,
        granularity=granularity,
    )


def _gran_key_to_float(g_key: str) -> float:
    """Convert a dot-free granularity key to a float for label file lookup.

    ``"g02"`` -> ``0.2``,  ``"g05"`` -> ``0.5``,  ``"g08"`` -> ``0.8``

    Also accepts the dotted on-disk format (e.g. ``"g0.2"`` -> ``0.2``).
    """
    stripped = g_key.lstrip("g")
    if "." in stripped:
        return float(stripped)
    return int(stripped) / 10.0


def load_training_pack_scene_multi(
    scene_dir: str | Path,
    granularities: tuple[str, ...] = ("g02", "g05", "g08"),
) -> MultiGranTrainingPackScene:
    """Load a training-pack for one scene with multiple granularities.

    Parameters
    ----------
    scene_dir:
        Path to the scene directory, or directly to the pack subdirectory.
    granularities:
        Dot-free granularity keys, e.g. ``("g02", "g05", "g08")``.
        Each is converted to the dotted on-disk format for label file lookup.
    """
    scene_dir = Path(scene_dir)
    pack_dir = _resolve_pack_dir(scene_dir)

    meta = _load_scene_meta(pack_dir)
    num_points_declared: int = meta["num_points"]

    points = np.load(pack_dir / "points.npy")

    opt = meta.get("optional_files_present", {})
    has_colors = opt.get("colors.npy", (pack_dir / "colors.npy").exists())
    colors: np.ndarray | None = None
    if has_colors:
        colors_path = pack_dir / "colors.npy"
        if not colors_path.exists():
            raise FileNotFoundError(
                "scene_meta declares colors.npy present but file is missing"
            )
        colors = np.load(colors_path)

    normals_path = pack_dir / "normals.npy"
    has_normals = opt.get("normals.npy", normals_path.exists())
    normals_m: np.ndarray | None = None
    if has_normals:
        if not normals_path.exists():
            raise FileNotFoundError(
                "scene_meta declares normals.npy present but file is missing"
            )
        normals_m = np.load(normals_path).astype(np.float32, copy=False)

    valid_points = np.load(pack_dir / "valid_points.npy").astype(bool)
    seen_points = _load_seen_points_with_legacy_fallback(pack_dir, valid_points)
    supervision_mask = np.load(pack_dir / "supervision_mask.npy").astype(bool)

    labels_by_gran: dict[str, np.ndarray] = {}
    for g_key in granularities:
        g_float = _gran_key_to_float(g_key)
        label_path = _resolve_label_file(meta, g_float, pack_dir)
        labels = np.load(label_path)
        _validate_scene_arrays(
            num_points_declared=num_points_declared,
            points=points,
            colors=colors,
            normals=normals_m,
            labels=labels,
            valid_points=valid_points,
            seen_points=seen_points,
            supervision_mask=supervision_mask,
        )
        labels_by_gran[g_key] = labels

    if scene_dir == pack_dir:
        resolved_scene_dir = pack_dir.parent
    else:
        resolved_scene_dir = scene_dir

    return MultiGranTrainingPackScene(
        scene_id=meta["scene_id"],
        scene_dir=resolved_scene_dir,
        training_pack_dir=pack_dir,
        points=points,
        colors=colors,
        normals=normals_m,
        labels_by_granularity=labels_by_gran,
        valid_points=valid_points,
        seen_points=seen_points,
        supervision_mask=supervision_mask,
        scene_meta=meta,
        granularities=granularities,
    )


# ── summary / diagnostics ───────────────────────────────────────────────


def summarize_training_pack_scene(scene: TrainingPackScene) -> dict[str, Any]:
    """Compute and return summary statistics.  Printed once at startup."""
    labels = scene.labels
    mask = scene.supervision_mask

    supervised_labels = labels[mask]
    instance_ids = supervised_labels[supervised_labels >= 0]
    unique_ids, counts = np.unique(instance_ids, return_counts=True)

    stats: dict[str, Any] = {
        "scene_id": scene.scene_id,
        "granularity": scene.granularity,
        "num_points": scene.num_points,
        "num_seen": int(scene.seen_points.sum()),
        "num_supervised": int(mask.sum()),
        "num_pseudo_instances": len(unique_ids),
    }

    if len(counts) > 0:
        stats.update(
            {
                "min_instance_size": int(counts.min()),
                "mean_instance_size": float(counts.mean()),
                "median_instance_size": float(np.median(counts)),
                "max_instance_size": int(counts.max()),
                "std_instance_size": float(counts.std()),
                "total_labeled_points": int(counts.sum()),
                "labeled_fraction": float(counts.sum() / scene.num_points),
            }
        )

    return stats


def print_training_pack_summary(scene: TrainingPackScene) -> dict[str, Any]:
    """Summarize, log, and return stats dict."""
    stats = summarize_training_pack_scene(scene)
    log.info(
        "[%s/g%s] %d pts | %d seen | %d supervised | %d instances | "
        "sizes: min=%s  mean=%s  median=%s  max=%s | labeled %.1f%%",
        stats["scene_id"],
        stats["granularity"],
        stats["num_points"],
        stats["num_seen"],
        stats["num_supervised"],
        stats["num_pseudo_instances"],
        stats.get("min_instance_size", "n/a"),
        f'{stats["mean_instance_size"]:.1f}' if "mean_instance_size" in stats else "n/a",
        f'{stats["median_instance_size"]:.0f}' if "median_instance_size" in stats else "n/a",
        stats.get("max_instance_size", "n/a"),
        stats.get("labeled_fraction", 0.0) * 100,
    )
    return stats


# ── smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m delimit3d.data.training_pack <scene_dir> [granularity]")
        raise SystemExit(1)

    scene_dir = sys.argv[1]
    granularity = float(sys.argv[2]) if len(sys.argv) == 3 else 0.5

    scene = load_training_pack_scene(scene_dir, granularity)
    stats = print_training_pack_summary(scene)
    print()
    for k, v in stats.items():
        print(f"  {k:24s}: {v}")
