from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
try:  # optional: only needed when voxelizing raw Structured3D geometry
    import open3d as o3d
except ModuleNotFoundError:  # pragma: no cover - depends on the host environment
    o3d = None

from delimit3d.source.datasets.structured3d.reader import Structured3DReader


PREPARED_VERSION = "structured3d_prepare_v2_axis_pose"
CAMERA_AXIS_CONVERSION = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float32)


def is_prepared(scene_dir: Path) -> bool:
    scene_dir = Path(scene_dir)
    marker = scene_dir / ".prepared"
    if not marker.exists():
        return False

    try:
        version = marker.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        version = ""

    required_paths = [
        scene_dir / "color",
        scene_dir / "depth",
        scene_dir / "pose",
        scene_dir / "intrinsic" / "intrinsic_color.txt",
        scene_dir / "intrinsic" / "intrinsic_depth.txt",
        scene_dir / f"{scene_dir.name}_vh_clean_2.ply",
    ]
    if not all(path.exists() for path in required_paths):
        return False

    return version == PREPARED_VERSION


def _count_frame_files(scene_dir: Path) -> tuple[int, int, int]:
    color_count = len([p for p in (scene_dir / "color").glob("*") if p.suffix.lower() in {".jpg", ".png"}])
    depth_count = len([p for p in (scene_dir / "depth").glob("*.png")])
    pose_count = len([p for p in (scene_dir / "pose").glob("*.txt")])
    return color_count, depth_count, pose_count


def prepare_structured3d_scene(scene_id: str, raw_zips_dir: str, output_scans_root: str) -> None:
    """
    Prepare a Structured3D scene directory in the ScanNet-like scene format
    consumed by the Delimit3D source adapter.

    Outputs (under <output_scans_root>/<scene_id>/):
    - color/{i}.jpg
    - depth/{i}.png  (uint16 depth in mm)
    - pose/{i}.txt   (4x4 C2W)
    - intrinsic/intrinsic_{color,depth}.txt (3x3)
    - <scene_id>_vh_clean_2.ply (voxelized point cloud)
    - gt_instance_ids.npy (instance id per voxelized point)
    - .prepared marker
    """

    if o3d is None:
        raise RuntimeError(
            "Preparing raw Structured3D geometry requires Open3D. "
            "Install the optional 'sources' dependencies with "
            "python -m pip install -e '.[sources]'."
        )

    scene_dir = Path(output_scans_root) / scene_id
    if is_prepared(scene_dir):
        return

    print(f"Preparing Structured3D scene: {scene_id}")

    zip_files = [
        os.path.join(raw_zips_dir, f)
        for f in os.listdir(raw_zips_dir)
        if "perspective_full" in f and f.endswith(".zip")
    ]
    if len(zip_files) == 0:
        raise FileNotFoundError(
            f"No perspective_full ZIPs found in {raw_zips_dir}. "
            "Expected files matching '*perspective_full*.zip'."
        )

    # Official release puts per-pixel instance.png under Structured3D_bbox.zip, not in
    # perspective_full_*.zip (RGB/depth/semantic only). Merge bbox so prepare can fuse GT.
    bbox_zip = os.path.join(raw_zips_dir, "Structured3D_bbox.zip")
    if os.path.isfile(bbox_zip):
        zip_files.append(bbox_zip)
    else:
        print(
            "Warning: Structured3D_bbox.zip not found in raw_zips_dir; "
            "instance masks will be unavailable and gt_instance_ids.npy will not be written. "
            "Download Structured3D_bbox.zip from the official Structured3D distribution."
        )

    reader = Structured3DReader(zip_files)

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    for d in [color_dir, depth_dir, pose_dir, intrinsic_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        rooms = reader.listdir(f"Structured3D/{scene_id}/2D_rendering")
    except Exception as e:
        raise FileNotFoundError(f"Could not find scene {scene_id} in ZIP files: {e}") from e

    global_coords: list[np.ndarray] = []
    global_colors: list[np.ndarray] = []
    global_instances: list[np.ndarray | None] = []
    global_instance_lengths: list[int] = []
    any_instances = False

    frame_idx = 0
    intrinsic_saved = False

    for room in rooms:
        prsp_path = f"Structured3D/{scene_id}/2D_rendering/{room}/perspective/full"
        try:
            frames = reader.listdir(prsp_path)
        except Exception:
            continue

        for frame in frames:
            try:
                cam_r, cam_t, cam_f = reader.read_camera(f"{prsp_path}/{frame}/camera_pose.txt")
                depth = reader.read_depth(f"{prsp_path}/{frame}/depth.png")
                color = reader.read_color(f"{prsp_path}/{frame}/rgb_rawlight.png")
            except Exception as e:
                print(f"Skipping {scene_id} room {room} frame {frame} due to error: {e}")
                continue

            # Instance is optional (some releases/zips do not provide instance masks).
            instance = None
            try:
                instance = reader.read_instance(f"{prsp_path}/{frame}/instance.png")
                any_instances = True
            except Exception:
                instance = None

            if cam_f is None:
                print(f"Skipping {scene_id} room {room} frame {frame} due to missing cam_f")
                continue

            if depth is None or depth.ndim != 2:
                print(f"Skipping {scene_id} room {room} frame {frame} due to invalid depth shape")
                continue

            height, width = depth.shape
            fx, fy = cam_f

            # Intrinsics as in pointcept preprocessing (fx,fy are angle terms).
            K = np.eye(3, dtype=np.float32)
            K[0, 2], K[1, 2] = width / 2.0, height / 2.0
            K[0, 0] = K[0, 2] / np.tan(fx)
            K[1, 1] = K[1, 2] / np.tan(fy)

            # Pose: camera-to-world (C2W) in the legacy pinhole-camera coordinates.
            # The fused geometry below first converts pinhole camera coordinates with
            # CAMERA_AXIS_CONVERSION, then applies Structured3D cam_r/cam_t:
            #   Xw(row) = (Xcam(row) @ CAMERA_AXIS_CONVERSION) @ cam_r.T + cam_t
            # To invert back to the original pinhole Xcam, the
            # saved column-vector C2W rotation must include the same axis conversion.
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam_r @ CAMERA_AXIS_CONVERSION.T
            c2w[:3, 3] = cam_t

            # 1) Export 2D frames
            cv2.imwrite(str(color_dir / f"{frame_idx}.jpg"), color[..., ::-1])  # RGB->BGR for cv2
            cv2.imwrite(str(depth_dir / f"{frame_idx}.png"), depth)
            np.savetxt(str(pose_dir / f"{frame_idx}.txt"), c2w)

            if not intrinsic_saved:
                np.savetxt(str(intrinsic_dir / "intrinsic_color.txt"), K)
                np.savetxt(str(intrinsic_dir / "intrinsic_depth.txt"), K)
                intrinsic_saved = True

            # Also save per-frame intrinsics. Structured3D camera FOV can vary per frame/room.
            np.savetxt(str(intrinsic_dir / f"{frame_idx}.txt"), K)

            # 2) Back-project depth to world points
            valid_mask = (depth > 0) & (depth < 65535)
            v, u = np.where(valid_mask)
            if v.size == 0:
                frame_idx += 1
                continue

            z = depth[valid_mask].astype(np.float32) / 1000.0  # mm->m
            x = (u.astype(np.float32) - K[0, 2]) * z / K[0, 0]
            y = (v.astype(np.float32) - K[1, 2]) * z / K[1, 1]

            pts_camera = np.stack([x, y, z], axis=-1)
            pts_camera = pts_camera @ CAMERA_AXIS_CONVERSION

            # Pointcept: world = (coord/1000) @ cam_r.T + cam_t
            pts_world = (pts_camera @ cam_r.T) + cam_t

            global_coords.append(pts_world)
            global_colors.append(color[valid_mask])
            if instance is not None:
                global_instances.append(instance[valid_mask])
                global_instance_lengths.append(int(v.size))
            else:
                global_instances.append(None)
                global_instance_lengths.append(int(v.size))

            frame_idx += 1

    if len(global_coords) == 0:
        raise RuntimeError(f"Scene {scene_id} resulted in empty point cloud / no valid frames.")

    all_coords = np.concatenate(global_coords, axis=0).astype(np.float32)
    all_colors = np.concatenate(global_colors, axis=0).astype(np.uint8)
    all_instances = None
    if any_instances and len(global_instances) > 0:
        filled_instances = [
            inst if inst is not None else np.zeros(length, dtype=np.int32)
            for inst, length in zip(global_instances, global_instance_lengths)
        ]
        all_instances = np.concatenate(filled_instances, axis=0).astype(np.int32)

    # NOTE: Do not apply an additional global axis flip here unless you also apply the
    # corresponding transform to every saved pose. Delimit3D projects 2D masks using the poses
    # into the geometry coordinate frame; geometry and poses must stay consistent.

    # 3) Voxel downsample (take first point per voxel, deterministic via np.unique index policy)
    grid_size = 0.02
    grid_coord = np.floor(all_coords / grid_size).astype(np.int64)
    _, unique_idx = np.unique(grid_coord, axis=0, return_index=True)

    down_coords = all_coords[unique_idx]
    down_colors = all_colors[unique_idx]
    down_instances = None
    if all_instances is not None:
        down_instances = all_instances[unique_idx]
        down_instances[down_instances == 65535] = 0

    # Save geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(down_coords)
    pcd.colors = o3d.utility.Vector3dVector(down_colors.astype(np.float32) / 255.0)

    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)

    if down_instances is not None:
        np.save(str(scene_dir / "gt_instance_ids.npy"), down_instances)

    (scene_dir / ".prepared").write_text(PREPARED_VERSION + "\n", encoding="utf-8")
    color_count, depth_count, pose_count = _count_frame_files(scene_dir)
    print(
        f"Finished {scene_id}. Saved {len(down_coords)} points and {frame_idx} frames "
        f"(color={color_count}, depth={depth_count}, pose={pose_count})."
    )
