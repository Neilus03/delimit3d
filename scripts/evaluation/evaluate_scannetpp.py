"""Evaluate frozen native LitePT features on point-prompted ScanNet++ objects.

The bundle supplies deterministic scene/object/query records and checkpoint
hashes.  Ground truth is used only to choose prompts and compute metrics; it is
never passed to LitePT.  The evaluator writes compact JSONL records so the
large point arrays and features remain outside Git.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from delimit3d.evaluation.features import (
    build_input_features,
    center_shift,
    infer_features,
    load_backbone,
    sha256_file,
)
from delimit3d.identity import data_root, project_name, project_slug
from delimit3d.evaluation.prompt_metrics import retrieval_metrics


BUNDLE = Path(os.environ.get("DELIMIT3D_PROMPT_BUNDLE", os.environ.get("FULLVAL_BUNDLE", ""))).expanduser().resolve(strict=True)
OUTPUT = Path(os.environ.get("DELIMIT3D_PROMPT_OUTPUT", os.environ.get("FULLVAL_OUTPUT", ""))).expanduser().resolve()
MODEL_NAME = os.environ.get("DELIMIT3D_PROMPT_MODEL", os.environ.get("FULLVAL_MODEL", ""))
SHARD_ID = int(os.environ.get("DELIMIT3D_PROMPT_SHARD", os.environ.get("FULLVAL_SHARD", "0")))
LITEPT_ROOT = Path(os.environ.get("DELIMIT3D_LITEPT_ROOT", str(data_root("litept")))).expanduser().resolve(strict=True)
if not MODEL_NAME:
    raise RuntimeError("Set DELIMIT3D_PROMPT_MODEL (or FULLVAL_MODEL) to public or posttrained")
if MODEL_NAME not in {"public", "posttrained"}:
    raise ValueError(f"Unsupported model name: {MODEL_NAME!r}")
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
PROTOCOL = MANIFEST["protocol"]


def array_sha(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.dtype, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()



def _scene_seed(scene: str) -> int:
    payload = f"{PROTOCOL['seed']}:{scene}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _load_scene_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or colors.shape != xyz.shape or normals.shape != xyz.shape:
        raise ValueError(f"{path}: points/colors/normals shape mismatch")
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return xyz, colors, normals


def main() -> None:
    device = torch.device(os.environ.get("DELIMIT3D_DEVICE", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ScanNet++ point-prompt evaluation requires a CUDA allocation")
    torch.cuda.set_device(device)
    checkpoint = Path(MANIFEST["assets"][MODEL_NAME]["path"]).expanduser().resolve(strict=True)
    expected_checkpoint_sha = MANIFEST["assets"][MODEL_NAME]["sha256"]
    observed_checkpoint_sha = sha256_file(checkpoint)
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError(f"Checkpoint hash drift: {checkpoint} != {expected_checkpoint_sha}")
    model = load_backbone(checkpoint, litept_root=LITEPT_ROOT, in_channels=6, device=device)
    model_hash_before = tensor_sha(model)
    scene_manifest = json.loads((BUNDLE / "scene_manifest.json").read_text(encoding="utf-8"))
    records = scene_manifest["scenes"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "project_name": project_name(),
        "project_slug": project_slug(),
        "model": MODEL_NAME,
        "shard": SHARD_ID,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint_sha256": observed_checkpoint_sha,
        "model_tensor_sha256": model_hash_before,
        "protocol": PROTOCOL,
        "scenes_expected": len(records),
        "scenes_completed": 0,
        "objects": 0,
        "queries": 0,
        "optimizer_updates_during_export": 0,
    }
    with (OUTPUT / "scenes.jsonl").open("w", encoding="utf-8") as stream:
        for index, rec in enumerate(records):
            scene = str(rec["scene"])
            path = Path(rec["data_path"]).expanduser().resolve(strict=True)
            if sha256_file(path) != rec["data_sha256"]:
                raise RuntimeError(f"Input data hash drift: {path}")
            xyz, colors, normals = _load_scene_arrays(path)
            points, _shift = center_shift(xyz)
            features = build_input_features(
                xyz, colors, use_colors=True, use_normals=True, normals=normals
            ).astype(np.float32, copy=False)
            with torch.inference_mode():
                native = infer_features(model, points=points, features=features, device=device)
            unit = native.astype(np.float64)
            unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
            rows = []
            gt_digest = hashlib.sha256()
            for obj in rec["objects"]:
                instance = int(obj["instance"])
                with np.load(path, allow_pickle=False) as data:
                    inds = np.asarray(data[f"target_{instance}"], dtype=np.int64)
                gt_digest.update(inds.tobytes())
                target_full = np.zeros(len(xyz), dtype=bool)
                target_full[inds] = True
                for query in obj["queries"]:
                    query = int(query)
                    candidates = np.arange(len(xyz)) != query
                    target = target_full[candidates]
                    score = np.einsum("ij,j->i", unit, unit[query])[candidates]
                    values = retrieval_metrics(target, score)
                    values.update({
                        "instance": instance,
                        "semantic_class": int(obj["semantic_class"]),
                        "semantic_name": obj.get("semantic_name"),
                        "instance_points": int(obj["instance_points"]),
                        "query": query,
                        "query_coord": points[query].tolist(),
                        "candidate_points": int(candidates.sum()),
                        "target_points_excluding_query": int(target.sum()),
                    })
                    rows.append(values)
            result = {
                "scene": scene,
                "points": len(xyz),
                "input_sha256": array_sha(np.concatenate([points, features], axis=1)),
                "gt_instance_sha256": gt_digest.hexdigest(),
                "eligible_objects_before_cap": rec["eligible_objects_before_cap"],
                "objects": rec["objects"],
                "queries": rows,
            }
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if index < 2:
                np.savez_compressed(OUTPUT / f"{scene}_features.npz", points=points, colors=colors, features=native)
            report["scenes_completed"] += 1
            report["objects"] += len(rec["objects"])
            report["queries"] += len(rows)
            print(json.dumps({"project": project_name(), "model": MODEL_NAME, "scene": scene, "completed": index + 1, "total": len(records)}), flush=True)
            del native, unit
            torch.cuda.empty_cache()
    report["records_sha256"] = sha256_file(OUTPUT / "scenes.jsonl")
    report["model_state_unchanged"] = tensor_sha(model) == model_hash_before
    report["passed"] = bool(
        report["model_state_unchanged"]
        and report["scenes_completed"] == len(records)
        and report["queries"] == 4 * report["objects"]
    )
    if not report["passed"]:
        raise RuntimeError(f"ScanNet++ evaluation failed: {report}")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SCANNETPP EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
