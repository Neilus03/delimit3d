#!/usr/bin/env python3
"""Compare native and fallback LitePT forward backends on fixed prompts.

This is a small numerical sanity check for the pf-pc69 fast-path run.  It
does not train a decoder or modify a checkpoint; it records the cosine score
vectors for a fixed prefix of the ScanNet++ validation bundle so two backend
processes can be compared exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.evaluation.features import center_shift, infer_features, load_backbone
from delimit3d.evaluation.prompt_metrics import retrieval_metrics


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene-count", type=int, default=3)
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a mapping")
    return value


def main() -> None:
    args = _args()
    config = _load_yaml(args.config.resolve(strict=True))
    val_bundle = Path(config["paths"]["val_bundle"]).expanduser().resolve(strict=True)
    val_manifest = json.loads((val_bundle / "scene_manifest.json").read_text())
    scenes = list(val_manifest["scenes"][: int(args.scene_count)])
    if not scenes:
        raise ValueError("validation bundle has no scenes")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("spotcheck requires CUDA")
    torch.cuda.set_device(0)
    model = load_backbone(
        args.checkpoint.resolve(strict=True),
        litept_root=Path(config["paths"]["litept_root"]),
        in_channels=6,
        device=device,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    score_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    offsets = [0]
    for scene_record in scenes:
        scene = str(scene_record["scene"])
        data_path = Path(scene_record["data_path"]).expanduser().resolve(strict=True)
        data = np.load(data_path)
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        inputs = build_input_features(points, colors, use_colors=True, use_normals=True, normals=normals)
        shifted, _ = center_shift(points)
        native = infer_features(model, points=shifted, features=inputs, device=device)
        unit = native / np.maximum(np.linalg.norm(native, axis=1, keepdims=True), 1e-8)
        for item in scene_record["objects"]:
            instance = int(item["instance"])
            query = int(item["queries"][0])
            target_full = np.zeros(len(points), dtype=bool)
            target_full[np.asarray(data[f"target_{instance}"], dtype=np.int64)] = True
            candidate = np.ones(len(points), dtype=bool)
            candidate[query] = False
            scores = np.einsum("ij,j->i", unit, unit[query])[candidate].astype(np.float32)
            target = target_full[candidate]
            metrics = retrieval_metrics(target, scores, cosine_threshold=0.70)
            rows.append({"scene": scene, "instance": instance, "query": query, "metrics": metrics})
            score_chunks.append(scores)
            target_chunks.append(target)
            offsets.append(offsets[-1] + len(scores))
        data.close()
        del points, colors, normals, inputs, shifted, native, unit
        torch.cuda.empty_cache()
    scores = np.concatenate(score_chunks).astype(np.float32, copy=False)
    targets = np.concatenate(target_chunks).astype(bool, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        scores=scores,
        targets=targets,
        offsets=np.asarray(offsets, dtype=np.int64),
        keys=np.asarray([f"{r['scene']}:{r['instance']}:{r['query']}" for r in rows]),
    )
    summary = {
        "scene_count": len(scenes),
        "query_count": len(rows),
        "mean_metrics": {
            key: float(np.mean([float(row["metrics"][key]) for row in rows]))
            for key in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou")
        },
        "score_file": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
