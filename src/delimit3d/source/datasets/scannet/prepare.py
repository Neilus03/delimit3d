from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _resolve_scannet_pair_root() -> Path:
    default_root = "/home/nedela/projects/LitePT/datasets/preprocessing/scannet/scannet_pair"
    return Path(
        os.path.expandvars(
            os.path.expanduser(
                os.environ.get(
                    "DELIMIT3D_SCANNET_PAIR_ROOT",
                    os.environ.get("CHORUS_SCANNET_PAIR_ROOT", default_root),
                )
            )
        )
    ).resolve()


def _load_sensor_data_class():
    scannet_pair_root = _resolve_scannet_pair_root()
    sensor_py = scannet_pair_root / "SensorData.py"

    if not sensor_py.exists():
        raise FileNotFoundError(
            f"Missing SensorData.py at {sensor_py}. "
            "Set DELIMIT3D_SCANNET_PAIR_ROOT to the folder containing SensorData.py."
        )

    if str(scannet_pair_root) not in sys.path:
        sys.path.insert(0, str(scannet_pair_root))

    from SensorData import SensorData  # type: ignore

    return SensorData


def _save_mat(matrix: np.ndarray, filename: Path) -> None:
    with filename.open("w", encoding="utf-8") as f:
        for line in matrix:
            np.savetxt(f, line[np.newaxis], fmt="%f")


def is_rgbd_prepared(scene_dir: Path) -> bool:
    scene_dir = Path(scene_dir)

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    required_files = [
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic_depth.txt",
    ]

    if not color_dir.is_dir():
        return False
    if not depth_dir.is_dir():
        return False
    if not pose_dir.is_dir():
        return False
    if not intrinsic_dir.is_dir():
        return False

    for path in required_files:
        if not path.exists() or path.stat().st_size == 0:
            return False

    num_color = len([p for p in color_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    num_depth = len([p for p in depth_dir.iterdir() if p.suffix == ".png"])
    num_pose = len([p for p in pose_dir.iterdir() if p.suffix == ".txt"])

    return num_color > 0 and num_depth > 0 and num_pose > 0


def extract_rgbd(scene_dir: Path) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "ScanNet extraction requires 'imageio'. Install it in your active environment."
        ) from exc

    scene_dir = Path(scene_dir)

    if is_rgbd_prepared(scene_dir):
        print(f"RGB-D already prepared for scene {scene_dir.name}, skipping extraction.")
        return

    scene_id = scene_dir.name
    sens_path = scene_dir / f"{scene_id}.sens"

    if not sens_path.exists():
        raise FileNotFoundError(
            f"Missing ScanNet .sens file: {sens_path}. "
            "Scene is not prepared and raw ScanNet source is missing."
        )

    SensorData = _load_sensor_data_class()
    sd = SensorData(str(sens_path))

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    for d in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        d.mkdir(parents=True, exist_ok=True)

    intrinsic_color_path = intrinsic_dir / "intrinsic_color.txt"
    extrinsic_color_path = intrinsic_dir / "extrinsic_color.txt"
    intrinsic_depth_path = intrinsic_dir / "intrinsic_depth.txt"
    extrinsic_depth_path = intrinsic_dir / "extrinsic_depth.txt"

    if not intrinsic_color_path.exists():
        _save_mat(sd.intrinsic_color, intrinsic_color_path)
    if not extrinsic_color_path.exists():
        _save_mat(sd.extrinsic_color, extrinsic_color_path)
    if not intrinsic_depth_path.exists():
        _save_mat(sd.intrinsic_depth, intrinsic_depth_path)
    if not extrinsic_depth_path.exists():
        _save_mat(sd.extrinsic_depth, extrinsic_depth_path)

    for i, frame in enumerate(sd.frames):
        color_path = color_dir / f"{i}.jpg"
        depth_path = depth_dir / f"{i}.png"
        pose_path = pose_dir / f"{i}.txt"

        if not depth_path.exists():
            depth_data = frame.decompress_depth(sd.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                sd.depth_height,
                sd.depth_width,
            )
            imageio.imwrite(depth_path, depth)

        if not color_path.exists():
            color = frame.decompress_color(sd.color_compression_type)
            imageio.imwrite(color_path, color, quality=95)

        if not pose_path.exists():
            _save_mat(frame.camera_to_world, pose_path)
