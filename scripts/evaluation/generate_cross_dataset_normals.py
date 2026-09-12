#!/usr/bin/env python3
"""Deterministic cross-domain normal synthesis for the LitePT RGBN6 contract.

Official AGILE3D S3DIS/KITTI-360 PLYs contain XYZ/RGB/labels but no normals.
This utility creates encoder-side normal sidecars without touching the official
files. Normals are estimated with Open3D KNN PCA and oriented by a
centroid-facing convention. The method, source hashes, and sidecar hashes are
recorded so the input distribution is explicit and reproducible.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, threading
from pathlib import Path
import numpy as np

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)

def read_points(path: Path) -> np.ndarray:
    from plyfile import PlyData
    data = PlyData.read(path)["vertex"].data
    points = np.stack([data[key] for key in ("x", "y", "z")], axis=1)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all(): raise ValueError(f"{path}: invalid XYZ")
    return points

def estimate_normals(points: np.ndarray, knn: int) -> np.ndarray:
    import open3d as o3d
    if len(points) < max(4, knn): raise ValueError(f"need at least {max(4, knn)} points, got {len(points)}")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(knn)))
    normals = np.asarray(cloud.normals, dtype=np.float32)
    if normals.shape != points.shape or not np.isfinite(normals).all(): raise RuntimeError("Open3D returned malformed normals")
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms <= 1.0e-8): raise RuntimeError("Open3D returned zero normals")
    normals /= norms[:, None]
    radial = points - points.mean(axis=0, dtype=np.float64).astype(np.float32)
    score = np.einsum("ij,ij->i", normals, radial)
    flip = (score < 0) | ((np.abs(score) <= 1.0e-7) & (normals[:, 2] < 0))
    normals[flip] *= -1.0
    return normals.astype(np.float32, copy=False)

def scenes_from_val(root: Path) -> list[str]:
    value = json.loads((root / "val_list.json").read_text())
    keys = value.keys() if isinstance(value, dict) else value
    return sorted({str(key).rsplit("_obj_", 1)[0] for key in keys})

def crop_entries(root: Path) -> list[tuple[str, int, Path]]:
    object_ids = np.load(root / "single/object_ids.npy", allow_pickle=True)
    return [(str(scene), int(index), root / "single" / "crops" / str(scene) / f"{scene}_crop_{int(index)}.ply") for scene, index in object_ids]

def worker_context(root: Path, name: str):
    directory = root / "workers"; directory.mkdir(parents=True, exist_ok=True)
    dump(directory / f"{name}.pid.json", {"pid": os.getpid(), "host": platform.node(), "name": name, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "argv": sys.argv})
    heartbeat = directory / f"{name}.heartbeat.jsonl"
    stopped = threading.Event()
    def beat():
        while not stopped.is_set():
            with heartbeat.open("a") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "host": platform.node(), "time": time.time(), "name": name}) + "\n")
            stopped.wait(30)
    thread = threading.Thread(target=beat, daemon=True); thread.start()
    return stopped, thread

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("S3DIS", "KITTI360"), required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--scope", choices=("scenes", "crops", "all"), default="all")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    root = args.official_root.expanduser().resolve(strict=True); output = args.output_root.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    normal_root = output / "normals" / args.dataset; normal_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    jobs: list[tuple[str, Path, str]] = []
    if args.scope in ("scenes", "all"): jobs.extend((scene, path, "scene") for scene, path in ((s, root / "scans" / f"{s}.ply") for s in scenes_from_val(root)))
    if args.scope in ("crops", "all"): jobs.extend((f"{scene}_crop_{index}", path, "crop") for scene, index, path in crop_entries(root))
    if args.max_items is not None: jobs = jobs[: int(args.max_items)]
    stopped, heartbeat_thread = worker_context(output, f"generate-normals-{args.dataset}-{args.scope}")
    for index, (name, source, kind) in enumerate(jobs, start=1):
        if not source.exists(): raise FileNotFoundError(source)
        target = normal_root / f"{name}.npy"; source_hash = file_sha256(source)
        if target.exists():
            normals = np.load(target, mmap_mode="r")
            if normals.ndim != 2 or normals.shape[1] != 3 or not np.isfinite(normals).all(): raise ValueError(f"invalid existing normal sidecar: {target}")
            entries.append({"name": name, "kind": kind, "source": str(source), "source_sha256": source_hash, "normal": str(target), "normal_sha256": file_sha256(target), "points": int(normals.shape[0]), "reused": True})
            continue
        points = read_points(source); normals = estimate_normals(points, args.knn)
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}"); np.save(temporary, normals)
        generated = Path(str(temporary) + ".npy") if not temporary.exists() else temporary; os.replace(generated, target)
        entries.append({"name": name, "kind": kind, "source": str(source), "source_sha256": source_hash, "normal": str(target), "normal_sha256": file_sha256(target), "points": int(len(points)), "knn": int(args.knn), "reused": False})
        if index == 1 or index % 5 == 0: print(json.dumps({"dataset": args.dataset, "scope": args.scope, "completed": index, "total": len(jobs), "name": name, "points": len(points)}), flush=True)
        del points, normals
    stopped.set(); heartbeat_thread.join(timeout=1)
    report = {"schema": "delimit3d_cross_dataset_normals/v1", "dataset": args.dataset, "official_root": str(root), "output_root": str(output), "scope": args.scope, "method": "open3d_estimate_normals_knn_pca_centroid_orient", "knn": int(args.knn), "open3d_version": __import__("open3d").__version__, "input_contract": "RGB followed by synthesized unit normals (RGBN6)", "entries": entries, "source_list_hashes": {name: file_sha256(root / name) for name in ("val_list.json", "single/object_ids.npy", "single/object_classes.txt")}, "created_epoch": time.time(), "host": platform.node()}
    report_path = output / "lead_audit" / f"normal_generation_{args.dataset}_{args.scope}.json"; dump(report_path, report)
    print(json.dumps({"completed": len(entries), "report": str(report_path), "report_sha256": file_sha256(report_path)}), flush=True)

if __name__ == "__main__": main()
