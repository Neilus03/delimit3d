
from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from delimit3d.source.common.types import FrameRecord, TeacherOutput
from delimit3d.source.core.teacher.base import TeacherModel
from delimit3d.source.datasets.base import SceneAdapter


def _map_scratch2_to_euler_work(p: Path) -> Path:
    user = os.environ.get("USER", "nedela")
    s = str(p)
    scratch_prefix = f"/scratch2/{user}"
    euler_prefix = f"/cluster/work/igp_psr/{user}"
    if s.startswith(scratch_prefix + "/"):
        return Path(euler_prefix + s[len(scratch_prefix) :])
    return p


def _resolve_path_for_env(p: Path) -> Path:
    mapped = _map_scratch2_to_euler_work(p)
    if mapped != p and mapped.exists():
        return mapped
    return p


def _resolve_unsam_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    default_root = repo_root / "UnSAMv2"
    root = Path(
        os.path.expandvars(
            os.path.expanduser(
                os.environ.get("UNSAM_ROOT", str(default_root))
            )
        )
    ).resolve()
    # If a user sets UNSAM_ROOT to /scratch2/... on Euler, prefer the mapped work path.
    root = _resolve_path_for_env(root)
    if root.exists():
        return root

    # Convenience fallback for Euler: allow keeping UnSAMv2 on /cluster/work/...
    user = os.environ.get("USER", "nedela")
    euler_candidate = Path(f"/cluster/work/igp_psr/{user}/UnSAMv2").resolve()
    if euler_candidate.exists():
        return euler_candidate

    return root


def _resolve_checkpoint_path(unsam_root: Path) -> Path:
    user = os.environ.get("USER", "nedela")

    default_scratch_ckpt = _resolve_path_for_env(
        Path(f"/scratch2/{user}/UnSAMv2/sam2/checkpoints/unsamv2/unsamv2_plus_ckpt.pt")
    )

    local_ckpt = unsam_root / "sam2" / "checkpoints" / "unsamv2_plus_ckpt.pt"
    legacy_repo_ckpt = unsam_root / "checkpoints" / "unsamv2_plus_ckpt.pt"


    candidates = [
        default_scratch_ckpt,
        local_ckpt,
        legacy_repo_ckpt,
    ]

    default_choice = next((p for p in candidates if p.exists()), local_ckpt)

    return Path(
        os.path.expandvars(
            os.path.expanduser(
                os.environ.get("UNSAMV2_CKPT", str(default_choice))
            )
        )
    ).resolve()


def _build_mask_generator(model, sam2_cls):
    return sam2_cls(
        model=model,
        points_per_side=32,
        points_per_batch=256,
        mask_threshold=-1,
        pred_iou_thresh=0.77,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
        output_mode="binary_mask",
    )


def _build_relaxed_mask_generator(model, sam2_cls):
    return sam2_cls(
        model=model,
        points_per_side=32,
        points_per_batch=256,
        mask_threshold=-1,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
        output_mode="binary_mask",
    )


class UnSAMv2Teacher(TeacherModel):
    def __init__(
        self,
        device: str | None = None,
        model_cfg: str | None = None,
        debug_first_n_frames: int = 10,
        overwrite: bool = False,
    ):
        self.device = device or os.environ.get(
            "DELIMIT3D_DEVICE", os.environ.get("CHORUS_DEVICE", "cuda:0")
        )
        self.model_cfg = model_cfg or os.environ.get("UNSAMV2_CFG", "configs/unsamv2_small.yaml")
        self.debug_first_n_frames = int(debug_first_n_frames)
        self.overwrite = bool(overwrite)

        self.unsam_root = _resolve_unsam_root()
        self.checkpoint_path = _resolve_checkpoint_path(self.unsam_root)
        self.sam2_python_root = self.unsam_root / "sam2"

        self._loaded = False
        self._mask_generator = None
        self._relaxed_mask_generator = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.sam2_python_root.exists():
            raise FileNotFoundError(
                f"Expected UnSAMv2 checkout at {self.sam2_python_root}. "
                "Set UNSAM_ROOT or place UnSAMv2 in the repository root."
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"UnSAMv2 checkpoint not found: {self.checkpoint_path}\n"
                "Set UNSAMV2_CKPT or place the checkpoint in the expected location."
            )

        if str(self.sam2_python_root) not in sys.path:
            sys.path.insert(0, str(self.sam2_python_root))

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore

        print(f"Loading UnSAMv2 on device={self.device}")
        model = build_sam2(
            self.model_cfg,
            str(self.checkpoint_path),
            device=self.device,
            apply_postprocessing=True,
        )

        self._mask_generator = _build_mask_generator(model, SAM2AutomaticMaskGenerator)
        self._relaxed_mask_generator = _build_relaxed_mask_generator(model, SAM2AutomaticMaskGenerator)
        self._loaded = True

    def _autocast_context(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def run(
        self,
        adapter: SceneAdapter,
        granularity: float,
        frame_skip: int,
    ) -> TeacherOutput:
        self._ensure_loaded()

        output_dir = adapter.scene_root / f"unsam_masks_g{granularity}"
        frames = adapter.list_frames()[::frame_skip]
        return self.run_on_frames(
            adapter=adapter,
            granularity=granularity,
            frames=frames,
            output_dir=output_dir,
            overwrite=self.overwrite,
        )

    def run_on_frames(
        self,
        adapter: SceneAdapter,
        granularity: float,
        frames: list[FrameRecord],
        output_dir: Path,
        overwrite: bool | None = None,
        mask_dtype: np.dtype | str = np.int32,
    ) -> TeacherOutput:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        should_overwrite = self.overwrite if overwrite is None else bool(overwrite)
        mask_dtype = np.dtype(mask_dtype)
        print(
            f"Teacher UnSAMv2: dataset={adapter.dataset_name}, scene={adapter.scene_id}, "
            f"granularity={granularity}, frames={len(frames)}"
        )

        frame_mask_paths: list[Path] = []
        total_masks = 0
        expected_mask_paths = [output_dir / f"{frame.frame_id}.npy" for frame in frames]

        if frames and not should_overwrite and all(path.exists() for path in expected_mask_paths):
            for frame, mask_path in zip(frames, expected_mask_paths, strict=True):
                frame_mask = np.load(mask_path)
                saved_count = int(np.max(frame_mask)) if frame_mask.size > 0 else 0
                total_masks += saved_count
                frame_mask_paths.append(mask_path)
                print(f"Reusing existing masks for frame {frame.frame_id}: {saved_count}")
            print(
                f"Teacher complete: scene={adapter.scene_id}, granularity={granularity}, "
                f"total_saved_masks={total_masks}"
            )
            return TeacherOutput(
                granularity=granularity,
                frame_mask_paths=frame_mask_paths,
                total_masks=total_masks,
            )

        self._ensure_loaded()

        for frame_idx, frame in enumerate(frames):
            mask_path = output_dir / f"{frame.frame_id}.npy"

            if mask_path.exists() and not should_overwrite:
                frame_mask = np.load(mask_path)
                saved_count = int(np.max(frame_mask)) if frame_mask.size > 0 else 0
                total_masks += saved_count
                frame_mask_paths.append(mask_path)
                print(f"Reusing existing masks for frame {frame.frame_id}: {saved_count}")
                continue

            image = adapter.load_rgb(frame)

            with torch.inference_mode(), self._autocast_context():
                masks_data = self._mask_generator.generate(image, gra=granularity)
                used_fallback = False
                if len(masks_data) == 0:
                    masks_data = self._relaxed_mask_generator.generate(image, gra=granularity)
                    used_fallback = True

            frame_mask = np.zeros((image.shape[0], image.shape[1]), dtype=mask_dtype)
            local_mask_id = 1

            masks_sorted = sorted(masks_data, key=lambda m: m.get("area", 0), reverse=True)
            for mask_dict in masks_sorted:
                bool_mask = mask_dict.get("segmentation")
                if bool_mask is None:
                    continue

                if local_mask_id > np.iinfo(mask_dtype).max:
                    raise RuntimeError(
                        f"Too many masks for dtype={mask_dtype}: local_mask_id={local_mask_id}"
                    )
                fill_region = bool_mask & (frame_mask == 0)
                if np.any(fill_region):
                    frame_mask[fill_region] = local_mask_id
                    local_mask_id += 1

            saved_count = local_mask_id - 1
            total_masks += saved_count
            np.save(mask_path, frame_mask)
            frame_mask_paths.append(mask_path)

            suffix = " [fallback]" if used_fallback else ""
            print(f"Saved {saved_count} masks for frame {frame.frame_id}{suffix}")

            if frame_idx < self.debug_first_n_frames:
                ious = [m.get("predicted_iou", 0.0) for m in masks_data]
                if ious:
                    print(
                        f"  debug: raw_masks={len(masks_data)}, "
                        f"iou[min/mean/max]={min(ious):.3f}/{np.mean(ious):.3f}/{max(ious):.3f}"
                    )
                else:
                    print("  debug: raw_masks=0 (even after fallback)")

        print(
            f"Teacher complete: scene={adapter.scene_id}, granularity={granularity}, total_saved_masks={total_masks}"
        )

        return TeacherOutput(
            granularity=granularity,
            frame_mask_paths=frame_mask_paths,
            total_masks=total_masks,
        )
