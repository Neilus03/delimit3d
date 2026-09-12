#!/usr/bin/env python3
"""Matched frozen LitePT transfer under the official AGILE3D ScanNet40 metrics.

This is not a reproduction of the original end-to-end MinkowskiEngine model.
All writes use a new external root. Source is frozen from a clean Euler commit.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tarfile
import shutil
import threading
import time
import numpy as np
import torch
import yaml
import run_agile3d_multio as core
from delimit3d.evaluation import scannet40_protocol as protocol
from delimit3d.evaluation.agile3d_protocol import file_sha256, json_sha256, build_token_targets, representative_indices_for_points, scene_seed, simulated_corrections, append_clicks, enforce_click_labels, raw_object_ious
from delimit3d.evaluation.agile3d_decoder import load_initialization, save_initialization, state_sha256

SCHEMA = "delimit3d_scannet40_matched/v1"
REPO = Path(__file__).resolve().parents[2]


def dump(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp); os.replace(tmp, path)


def resolve(value):
    path = Path(value).expanduser()
    if str(path).startswith("/cluster/") and not path.exists():
        mount = os.environ.get("DELIMIT3D_EULER_MOUNT")
        if not mount:
            raise RuntimeError("explicit DELIMIT3D_EULER_MOUNT required off Euler")
        path = Path(mount) / str(path).removeprefix("/cluster/")
    return path


def root_of(config):
    root = resolve(config["paths"]["output_root"])
    if protocol.EXPERIMENT not in str(root):
        raise ValueError("refusing non-Stage-B artifact root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def verify_config(config):
    if config["experiment_id"] != protocol.EXPERIMENT:
        raise ValueError("unexpected experiment ID")
    p = config["protocol"]
    expected = {"active_litept_level": "dec0", "voxel_size_m": .02, "voxel_reduce": "representative", "representative_sampling": "first", "click_budget": 20}
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"protocol contract drift: {key}")
    if config["decoder"]["hidden_dim"] != 128 or config["decoder"]["feature_dim"] != 72:
        raise ValueError("native 72-to-128 decoder required")
    if p["train_updates"] != 5000 and not config.get("smoke_only"):
        raise ValueError("full matched budget must be 5000")
    if config.get("encoder_adaptation_uses_scannet_labels") is not False:
        raise ValueError("encoder adaptation must remain ScanNet-label-free")
    if config["optimizer"]["cache_memory_scenes"] < 1:
        raise ValueError("bounded feature-cache capacity required")


def install_adapter():
    # Reuse the already tested native decoder/cache extraction implementation.
    # Dataset, initialization clicks, manifest checks and IO are supplied here.
    core.SCHEMA = SCHEMA
    core.verify_config = verify_config
    core.output_root = root_of
    core.resolve_external = lambda value, strict=True: resolve(value).resolve(strict=strict)
    core._load_record_arrays = lambda record: protocol.load_scene(record, resolve)
    core.load_prepared = load_prepared
    core._target_from_episode = target_from_episode
    core.json_dump = dump


def official_manifests(config):
    source = resolve(config["paths"]["official_root"])
    return {name: source / relative for name, relative in {
        "train": "train_list.json", "val": "val_list.json", "so_ids": "single/object_ids.npy", "so_classes": "single/object_classes.txt"}.items()}


def freeze(config):
    """Must run on Euler in the clean source worktree before any GPU job."""
    verify_config(config)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("commit the Stage-B worktree before freezing")
    config = dict(config); config["paths"] = dict(config["paths"]); config["repo_commit"] = core.git_commit()
    root = root_of(config); out = root / "freeze"; out.mkdir(exist_ok=True)
    dependency = resolve(config["paths"].get("litept_original_root", config["paths"]["litept_root"]))
    dependency_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dependency, text=True).strip()
    dependency_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=dependency, text=True)
    dependency_files = sorted([p for folder in ("litept", "libs/pointrope") for p in (dependency/folder).rglob("*") if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h", ".so") and "__pycache__" not in p.parts and "build" not in p.parts and "dist" not in p.parts])
    dependency_entries = {str(p.relative_to(dependency)): file_sha256(p) for p in dependency_files}
    frozen_dependency = out/"litept_source"
    for file in dependency_files:
        target=frozen_dependency/file.relative_to(dependency);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha256(target)!=dependency_entries[str(file.relative_to(dependency))]:raise ValueError("frozen LitePT dependency drift")
        if not target.exists():shutil.copy2(file,target)
    dependency_archive=out/"litept_source.tar"
    if not dependency_archive.exists():
        with tarfile.open(dependency_archive,"w") as tar:
            for name in dependency_entries:tar.add(frozen_dependency/name,arcname=name,recursive=False)
    dependency_report={"original_root":str(dependency),"commit":dependency_commit,"tracked_dirty_status":dependency_status,"source_entries":dependency_entries,"archive":str(dependency_archive),"archive_sha256":file_sha256(dependency_archive)}
    dump(out/"litept_dependency.json",dependency_report)
    config["paths"]["litept_original_root"]=str(dependency)
    config["paths"]["litept_root"]=str(frozen_dependency)
    resolved = out / "resolved_config.yaml"
    content = yaml.safe_dump(config, sort_keys=False)
    if resolved.exists() and resolved.read_text() != content:
        raise RuntimeError("frozen YAML differs; use a new explicit artifact version")
    resolved.write_text(content)
    archive = out / f"source_{config['repo_commit']}.tar"
    if not archive.exists():
        subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", config["repo_commit"]], cwd=REPO, check=True)
    init = root / "decoder_init.pt"
    if init.exists():
        _, init_report = load_initialization(init, expected_kwargs=core.decoder_kwargs(config))
    else:
        init_report = save_initialization(init, seed=int(config["decoder"]["init_seed"]), decoder_kwargs=core.decoder_kwargs(config))
    lists = {}
    for name, path in official_manifests(config).items():
        h = file_sha256(path)
        expected = config["official_manifest_sha256"][name]
        if h != expected:
            raise ValueError(f"official list changed: {name}")
        target = out / "manifests" / path.relative_to(resolve(config["paths"]["official_root"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != h:
            raise ValueError("frozen list drift")
        target.write_bytes(path.read_bytes())
        lists[name] = {"path": str(path), "sha256": h, "frozen_path": str(target)}
    checkpoints = {}
    for arm, a in config["arms"].items():
        h = file_sha256(resolve(a["checkpoint"]))
        if h != a["checkpoint_sha256"]:
            raise ValueError(f"encoder checkpoint drift: {arm}")
        checkpoints[arm] = {"path": a["checkpoint"], "sha256": h}
    environment = {"host": platform.node(), "python": sys.version, "torch": torch.__version__, "cuda_version": torch.version.cuda,
                   "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
    dump(out / "environment_cpu.json", environment)
    report = {"repo_commit": config["repo_commit"], "resolved_config": str(resolved), "resolved_config_sha256": file_sha256(resolved),
              "source_archive": str(archive), "source_archive_sha256": file_sha256(archive),
              "source_entries": {name:file_sha256(REPO/name) for name in subprocess.check_output(["git","ls-files"],cwd=REPO,text=True).splitlines() if (REPO/name).is_file()}, "decoder_initialization": init_report,
              "decoder_initialization_file_sha256": file_sha256(init), "environment": str(out / "environment_cpu.json"),
              "environment_sha256": file_sha256(out / "environment_cpu.json"), "lists": lists, "checkpoints": checkpoints, "litept_dependency": dependency_report, "litept_dependency_report_sha256": file_sha256(out/"litept_dependency.json")}
    dump(out / "provenance.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def verify_freeze(config):
    root = root_of(config)
    report = json.loads((root / "freeze/provenance.json").read_text())
    for name,h in report["source_entries"].items():
        if file_sha256(REPO/name)!=h:raise ValueError(f"executing source differs from frozen archive: {name}")
    if file_sha256(root/"decoder_init.pt")!=report["decoder_initialization_file_sha256"]:raise ValueError("decoder init file hash drift")
    if config["repo_commit"] != report["repo_commit"]:
        raise ValueError("commit drift")
    for path_key, hash_key in [("source_archive", "source_archive_sha256"), ("resolved_config", "resolved_config_sha256"), ("environment", "environment_sha256")]:
        if file_sha256(resolve(report[path_key])) != report[hash_key]:
            raise ValueError(f"frozen provenance drift: {path_key}")
    for name, path in official_manifests(config).items():
        if file_sha256(path) != report["lists"][name]["sha256"]:
            raise ValueError(f"official manifest drift: {name}")
    for arm, a in config["arms"].items():
        if file_sha256(resolve(a["checkpoint"])) != report["checkpoints"][arm]["sha256"]:
            raise ValueError(f"encoder checkpoint drift: {arm}")
    dependency=report["litept_dependency"]
    if file_sha256(resolve(dependency["archive"]))!=dependency["archive_sha256"]:raise ValueError("LitePT dependency archive drift")
    for name,h in dependency["source_entries"].items():
        if file_sha256(resolve(config["paths"]["litept_root"])/name)!=h:raise ValueError(f"LitePT dependency drift: {name}")
    frozen_config=yaml.safe_load(resolve(report["resolved_config"]).read_text())
    if json_sha256(config)!=json_sha256(frozen_config):raise ValueError("runtime config differs from frozen resolved YAML")
    return report


def prepare(config):
    verify_config(config); provenance = verify_freeze(config); root = root_of(config)
    if not config.get("smoke_only"):
        audit_complete=resolve(config["paths"]["geometry_audit"]).parent/"data_audit.json"
        if not audit_complete.exists():raise RuntimeError("full data audit has not completed")
    manifest_path = root / "selection_manifest.json"
    if manifest_path.exists():
        return load_prepared(config)[0]
    lists = official_manifests(config)
    train, val = (protocol.read_official_list(lists[x]) for x in ("train", "val"))
    ids = np.load(lists["so_ids"]); classes = np.loadtxt(lists["so_classes"], dtype=str)
    semantics = {(str(scene), int(obj)): str(cls) for (scene, obj), cls in zip(ids, classes, strict=True)}
    audit_path = resolve(config["paths"]["geometry_audit"])
    audit = {x["scene"]: x for x in core.read_jsonl(audit_path)}
    selected = {"train": [key.rsplit("_obj_", 1)[0] for key in train], "val": [key.rsplit("_obj_", 1)[0] for key in val]}
    if config.get("smoke_only"):
        selected = {"train": [config["smoke_scenes"]["train"]], "val": [config["smoke_scenes"]["val"]]}
    records = {"train": [], "val": []}
    for split, scenes in selected.items():
        for scene in scenes:
            a = audit.get(scene)
            required = ("xyz_exact", "rgb_exact", "normal_finite", "normal_shape_correct", "official_archive_crc_match")
            if not a or not all(a[x] for x in required):
                raise ValueError(f"{scene}: complete successful official data audit required")
            pack = Path(a["processed_dir"])
            from plyfile import PlyData
            path=resolve(a["official_ply"])
            if file_sha256(path)!=a["official_ply_sha256"]:raise ValueError(f"{scene}: official PLY changed since audit")
            labels=np.asarray(PlyData.read(path)["vertex"].data["label"],dtype=np.int64)
            objects=protocol.objects_from_official_labels(labels,scene,semantics)
            record = {"scene": scene, "kind": split, "points": a["n"], "objects": objects, "data_path": a["official_ply"],
                      "data_sha256": a["official_ply_sha256"], "normal_path": str(pack / "normal.npy"),
                      "source_hashes": {a["official_ply"]: a["official_ply_sha256"], **{str(pack/(k+".npy")): v for k,v in a["processed_sha256"].items()}},
                      "normal_alignment_verified": True,
                      "ground_truth_authority":"official_AGILE3D_PLY_label",
                      "processed_instance_plus_one_exact_diagnostic":a["instance_plus_one_exact"]}
            # Use the same shifted coordinates and representative-first voxelization as extraction.
            if file_sha256(resolve(pack/"coord.npy"))!=a["processed_sha256"]["coord"]:raise ValueError("normal alignment coordinate source changed")
            coord=np.asarray(np.load(resolve(pack/"coord.npy")),dtype=np.float32)
            shifted,_=core._center_shift_numpy(coord)
            reps=representative_indices_for_points(shifted,voxel_size=config["protocol"]["voxel_size_m"])
            surviving=set(int(x) for x in np.unique(labels[reps]) if x>0)
            missing=[o["instance"] for o in objects if o["instance"] not in surviving]
            if missing:raise ValueError(f"{scene}: official objects without representative tokens: {missing}; launch blocked, no exclusions")
            record["all_official_objects_have_representative_tokens"]=True
            record["representative_indices_sha256"]=core.array_sha256(reps)
            record["representative_token_count"]=len(reps)
            records[split].append(record)
    if set(selected["train"]) & set(selected["val"]):
        raise ValueError("train/validation overlap")
    rng = np.random.default_rng(config["seed"])
    episodes = []; updates = config["protocol"]["train_updates"]
    # Shuffled epochs guarantee every official train scene is visited before repeats.
    while len(episodes) < updates:
        for index in rng.permutation(len(records["train"])):
            if len(episodes) == updates: break
            record = records["train"][int(index)]; object_ids = [o["instance"] for o in record["objects"]]
            count = min(3,len(object_ids)) if config.get("smoke_only") else int(rng.integers(1, min(10, len(object_ids))+1))
            episodes.append({"update": len(episodes)+1, "scene": record["scene"], "objects": [int(x) for x in rng.choice(object_ids, count, replace=False)], "click_prefix": 0 if config.get("smoke_only") else int(rng.integers(0, 20))})
    episode_path = root / "train_episodes.jsonl"; core.write_jsonl(episode_path, episodes)
    mo = [{"scene": key.rsplit("_obj_",1)[0], "objects": [int(value["obj"][str(k)]) for k in range(1, len(value["obj"])+1)], "official_key": key} for key,value in val.items() if key.rsplit("_obj_",1)[0] in selected["val"]]
    so = [{"scene": str(scene), "objects": [int(obj)], "semantic_name": str(cls)} for (scene,obj),cls in zip(ids,classes,strict=True) if str(scene) in selected["val"]]
    manifest = {"schema": SCHEMA, "repo_commit": config["repo_commit"], "experiment_id": config["experiment_id"],
                "train_scenes": records["train"], "validation_scenes": records["val"], "panels": {"MO": mo,"SO":so},
                "train_episodes_path": str(episode_path), "train_episodes_sha256": file_sha256(episode_path),
                "decoder_initialization_path": str(root / "decoder_init.pt"), "decoder_initialization": provenance["decoder_initialization"],
                "geometry_audit_sha256": file_sha256(audit_path), "official_lists": provenance["lists"], "provenance_sha256": file_sha256(root / "freeze/provenance.json")}
    manifest["manifest_sha256"] = json_sha256(manifest); dump(manifest_path, manifest)
    dump(root / "prepare_report.json", {"train_scenes": len(records["train"]), "val_scenes": len(records["val"]), "MO":len(mo), "SO":len(so), "updates":len(episodes),"manifest_sha256":manifest["manifest_sha256"]})
    return manifest


def load_prepared(config):
    root = root_of(config); manifest = json.loads((root / "selection_manifest.json").read_text())
    provenance=json.loads((root/"freeze/provenance.json").read_text())
    if manifest.get("schema")!=SCHEMA or manifest.get("experiment_id")!=config["experiment_id"]:raise ValueError("manifest schema/experiment mismatch")
    if manifest.get("provenance_sha256")!=file_sha256(root/"freeze/provenance.json"):raise ValueError("manifest provenance linkage mismatch")
    if resolve(manifest["decoder_initialization_path"]).resolve()!=(root/"decoder_init.pt").resolve() or manifest["decoder_initialization"]!=provenance["decoder_initialization"]:raise ValueError("manifest decoder initialization not frozen initialization")
    check = dict(manifest); expected = check.pop("manifest_sha256")
    if json_sha256(check) != expected or manifest["repo_commit"] != config["repo_commit"]:
        raise ValueError("selection manifest integrity/commit mismatch")
    init = resolve(manifest["decoder_initialization_path"])
    _, report = load_initialization(init, expected_kwargs=core.decoder_kwargs(config))
    if report["tensor_state_sha256"] != manifest["decoder_initialization"]["tensor_state_sha256"]:
        raise ValueError("decoder initialization drift")
    if file_sha256(resolve(manifest["train_episodes_path"])) != manifest["train_episodes_sha256"]:
        raise ValueError("episode schedule drift")
    return manifest, init


def target_from_episode(cache, object_ids):
    target = torch.zeros(cache["features"].shape[0], dtype=torch.long)
    for local_id, obj in enumerate(object_ids, 1):
        indices = cache["object_token_indices"][str(obj)]
        if len(indices) == 0:
            raise ValueError(f"{cache['scene']}: requested object {obj} has no representative token")
        if torch.any(target[indices] != 0): raise ValueError("objects overlap in token space")
        target[indices] = local_id
    seed = scene_seed(20260912, cache["scene"], "initial:" + ",".join(map(str,object_ids)))
    clicks,times = protocol.initial_clicks(target.numpy(), cache["scene_xyz"].numpy(), seed)
    return target,clicks,times,list(object_ids)


class CacheLRU:
    def __init__(self, root, arm, capacity):
        self.root,self.arm,self.capacity,self.values=root,arm,capacity,OrderedDict()
    def get(self, scene):
        if scene in self.values: self.values.move_to_end(scene)
        else:
            self.values[scene]=core.load_cache(self.root,self.arm,scene)
            while len(self.values)>self.capacity: self.values.popitem(last=False)
        return self.values[scene]


def cache_features(config,arm):
    root=root_of(config);existing=root/f"cache_report_{arm}.json"
    if existing.exists():
        previous=json.loads(existing.read_text())
        for row in previous["cache_rows"]:
            if not row.get("file_sha256") or file_sha256(core._cache_path(root,arm,row["scene"]))!=row["file_sha256"]:raise ValueError("previously published cache binary changed; refusing to refresh hashes")
    manifest,_=load_prepared(config)
    records={r["scene"]:r for r in manifest["train_scenes"]+manifest["validation_scenes"]}
    original_dump=core.json_dump
    def publish(path,report):
        if Path(path).name==f"cache_report_{arm}.json":
            for row in report["cache_rows"]:
                file=core._cache_path(root,arm,row["scene"])
                payload=torch.load(file,map_location="cpu",weights_only=False)
                record=records[row["scene"]]
                if core.array_sha256(payload["representative_indices"])!=record["representative_indices_sha256"] or len(payload["representative_indices"])!=record["representative_token_count"]:raise ValueError("GPU representative geometry differs from CPU frozen preflight")
                if payload.get("arm") is None:
                    payload["arm"]=arm;save_torch(file,payload)
                elif payload["arm"]!=arm:raise ValueError("cache arm drift")
                row["file_sha256"]=file_sha256(file)
        original_dump(path,report)
    # Publish only complete, hashed reports; a killed augmentation leaves caches
    # resumable and never blesses a partially populated report.
    core.json_dump=publish
    try:return core.cache_features(config,arm)
    finally:core.json_dump=original_dump


def verify_caches(config):
    root=root_of(config);manifest,_=load_prepared(config)
    expected={r["scene"] for r in manifest["train_scenes"]+manifest["validation_scenes"]}
    for arm in ("public","delimit3d"):
        report=json.loads((root/f"cache_report_{arm}.json").read_text())
        if report.get("arm")!=arm:raise ValueError("cache report wrong arm")
        if report["selection_manifest_sha256"]!=manifest["manifest_sha256"] or report["checkpoint_sha256"]!=config["arms"][arm]["checkpoint_sha256"] or report["repo_commit"]!=config["repo_commit"]:raise ValueError("cache report provenance mismatch")
        if not report["encoder_state_unchanged"] or report["encoder_tensor_sha256_before"]!=report["encoder_tensor_sha256_after"]:raise ValueError("cache encoder changed")
        if {r["scene"] for r in report["cache_rows"]}!=expected or len(report["cache_rows"])!=len(expected):raise ValueError("cache report does not cover exact manifest scene set")
        for row in report["cache_rows"]:
            if file_sha256(core._cache_path(root,arm,row["scene"]))!=row.get("file_sha256"):raise ValueError("cache binary hash mismatch")
            c=core.load_cache(root,arm,row["scene"])
            if c.get("arm")!=arm or c.get("scene")!=row["scene"]:raise ValueError("cache payload wrong arm/scene")
            if c["checkpoint_sha256"]!=report["checkpoint_sha256"] or c["selection_manifest_sha256"]!=manifest["manifest_sha256"] or c["repo_commit"]!=config["repo_commit"] or c["encoder_tensor_sha256"]!=report["encoder_tensor_sha256_after"]:raise ValueError("cache payload provenance mismatch")
            if c["geometry_sha256"]!=row["geometry_sha256"]:raise ValueError("cache geometry drift")
            if c["empty_objects"]:raise ValueError("unrepresentable official objects in cache")
    return core._verify_cross_arm_geometry(root,manifest)


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"]);np.random.set_state(value["numpy"]);torch.set_rng_state(value["torch"]);torch.cuda.set_rng_state_all(value["cuda"])


def train(config, arm, resume=None):
    root=root_of(config); manifest,init=load_prepared(config)
    geometry=verify_caches(config)
    if not geometry or not geometry["geometry_byte_identity"]: raise ValueError("both matched caches required")
    decoder,initial=load_initialization(init,expected_kwargs=core.decoder_kwargs(config));decoder.to("cuda")
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=config["optimizer"]["learning_rate"],weight_decay=config["optimizer"]["weight_decay"])
    if {id(p) for g in optimizer.param_groups for p in g["params"]}!={id(p) for p in decoder.parameters()}:
        raise ValueError("optimizer contains parameters outside decoder")
    core.set_seed(config["seed"]);begin=0
    if resume:
        ckpt=torch.load(resolve(resume),map_location="cpu",weights_only=False)
        for key,value in {"arm":arm,"repo_commit":config["repo_commit"],"episode_schedule_sha256":manifest["train_episodes_sha256"],"decoder_initialization_sha256":initial["tensor_state_sha256"]}.items():
            if ckpt[key]!=value: raise ValueError(f"resume mismatch {key}")
        cache_report=json.loads((root/f"cache_report_{arm}.json").read_text())
        if ckpt.get("schema")!=core.CHECKPOINT_SCHEMA or ckpt.get("experiment_id")!=config["experiment_id"] or ckpt.get("decoder_kwargs")!=core.decoder_kwargs(config):raise ValueError("resume schema/experiment/decoder mismatch")
        if ckpt.get("encoder_tensor_sha256")!=cache_report["encoder_tensor_sha256_after"] or ckpt.get("encoder_frozen") is not True:raise ValueError("resume encoder invariant mismatch")
        if state_sha256(ckpt["decoder_state_dict"])!=ckpt.get("decoder_tensor_sha256"):raise ValueError("resume tensor hash mismatch")
        if not 0<ckpt["update"]<=config["protocol"]["train_updates"]:raise ValueError("resume update out of bounds")
        if not all(k in ckpt.get("rng_state",{}) for k in ("python","numpy","torch","cuda")):raise ValueError("resume RNG state missing")
        saved_rng=ckpt["rng_state"]
        if not isinstance(saved_rng["cuda"],(list,tuple)) or len(saved_rng["cuda"])!=1 or any(not torch.is_tensor(v) or v.dtype!=torch.uint8 or v.numel()==0 for v in saved_rng["cuda"]):raise ValueError("resume requires one nonempty CUDA RNG state")
        if not torch.is_tensor(saved_rng["torch"]) or saved_rng["torch"].dtype!=torch.uint8 or saved_rng["torch"].numel()==0:raise ValueError("invalid CPU Torch RNG state")
        saved_optimizer=ckpt["optimizer_state_dict"]
        saved_ids=[i for g in saved_optimizer["param_groups"] for i in g["params"]]
        if len(saved_ids)!=len(set(saved_ids)) or len(saved_ids)!=len(list(decoder.parameters())) or not set(saved_optimizer["state"]).issubset(saved_ids):raise ValueError("resume optimizer contains unexpected parameter identities")
        if any(g["lr"]!=config["optimizer"]["learning_rate"] or g["weight_decay"]!=config["optimizer"]["weight_decay"] for g in saved_optimizer["param_groups"]):raise ValueError("resume optimizer settings drift")
        decoder.load_state_dict(ckpt["decoder_state_dict"]);optimizer.load_state_dict(saved_optimizer)
        restore_rng(ckpt["rng_state"]);begin=ckpt["update"]
    out=root/arm;out.mkdir(exist_ok=True);log=out/"train_log.jsonl"
    if log.exists() and not resume:raise ValueError("existing training log requires explicit resume")
    episodes=core.read_jsonl(resolve(manifest["train_episodes_path"]));cache=CacheLRU(root,arm,config["optimizer"]["cache_memory_scenes"])
    before=file_sha256(resolve(config["arms"][arm]["checkpoint"]));start=time.time()
    cache_report=json.loads((root/f"cache_report_{arm}.json").read_text())
    with log.open("a") as handle:
        for episode in episodes[begin:]:
            values=core._train_one_update(decoder,optimizer,cache.get(episode["scene"]),episode,device=torch.device("cuda"),config=config)
            row={"update":episode["update"],"scene":episode["scene"],"seconds":time.time()-start,**values};handle.write(json.dumps(row)+"\n");handle.flush()
            update=episode["update"]
            if update%100==0 or update==begin+1:print(json.dumps(row),flush=True)
            if update%250==0 or update==len(episodes):
                payload={"schema":core.CHECKPOINT_SCHEMA,"experiment_id":config["experiment_id"],"arm":arm,"update":update,"repo_commit":config["repo_commit"],"decoder_kwargs":core.decoder_kwargs(config),"decoder_state_dict":{k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()},"optimizer_state_dict":optimizer.state_dict(),"rng_state":rng_state(),"decoder_tensor_sha256":state_sha256(decoder.state_dict()),"decoder_initialization_sha256":initial["tensor_state_sha256"],"encoder_tensor_sha256":cache_report["encoder_tensor_sha256_after"],"encoder_frozen":True,"episode_schedule_sha256":manifest["train_episodes_sha256"]}
                save_torch(out/f"decoder_u{update:05d}.pt",payload)
    after=file_sha256(resolve(config["arms"][arm]["checkpoint"]))
    if before!=after or not cache_report["encoder_state_unchanged"]:raise ValueError("frozen encoder drift")
    final=out/f"decoder_u{len(episodes):05d}.pt"
    report={"updates":len(episodes),"repo_commit":config["repo_commit"],"arm":arm,"checkpoint":str(final),"checkpoint_sha256":file_sha256(final),"decoder_initialization_sha256":initial["tensor_state_sha256"],"encoder_checkpoint_sha256_before":before,"encoder_checkpoint_sha256_after":after,"encoder_state_unchanged":True,"encoder_optimizer_parameter_count":0,"optimizer_parameter_names":[name for name,_ in decoder.named_parameters()],"episode_schedule_sha256":manifest["train_episodes_sha256"],"resume_rng_bit_exact_state_restored":bool(resume),"seconds":time.time()-start}
    dump(out/"train_report.json",report);return report


def evaluate_episode(decoder, cache, record, episode, config, prediction_dir=None):
    object_ids=episode["objects"];count=len(object_ids)
    target,clicks,times,_=target_from_episode(cache,object_ids)
    points,_,_,objects,masks=protocol.load_scene(record,resolve)
    raw=np.zeros(len(points),dtype=np.int64)
    for i,obj in enumerate(object_ids,1):raw[masks[obj]]=i
    features=cache["features"].cuda();xyz=cache["scene_xyz"].cuda();inverse=cache["inverse_map"].numpy()
    states=[{"total_clicks":0,"mean_iou":0.,"object_ious":{str(i):0. for i in range(1,count+1)},"clicks":{},"click_times":{}}]
    saved={};decoder.eval();perfect=False
    with torch.inference_mode():
        for total in range(count,20*count+1):
            if not perfect:
                prediction=decoder(features,xyz,clicks=clicks,click_times=times)["pred_masks"].argmax(1).cpu().numpy()
                prediction=enforce_click_labels(prediction,clicks)
            ious=raw_object_ious(prediction,raw,inverse,count)
            states.append({"total_clicks":total,"mean_iou":float(np.mean(list(ious.values()))),"object_ious":ious,"clicks":{k:list(v) for k,v in clicks.items()},"click_times":{k:list(v) for k,v in times.items()},"perfect_prediction_padded":perfect})
            if prediction_dir and total in {count,3*count,5*count,10*count,15*count,20*count}:
                saved[f"click_{total//count}"]=prediction.astype(np.int16)
            if total==20*count:break
            new,new_times,events=simulated_corrections(prediction,target.numpy(),cache["scene_xyz"].numpy(),clicks,times,training=False,click_center_method="kdtree")
            # Official code accounts virtual clicks after no error remains.
            if not new:perfect=True
            else:clicks,times=append_clicks(clicks,times,new,new_times)
    trace={"scene":record["scene"],"object_ids":object_ids,"object_count":count,"states":states,
           "objects":[next(o for o in objects if o["instance"]==obj) for obj in object_ids]}
    if prediction_dir:
        prediction_dir.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(prediction_dir,**saved)
        trace["prediction_path"]=str(prediction_dir)
    trace["metrics"]=protocol.metrics_from_trace(trace)
    return trace


def evaluate(config,arm,mode,checkpoint=None):
    root=root_of(config);manifest,_=load_prepared(config)
    for candidate in ("public","delimit3d"):
        if not (root/candidate/"train_report.json").exists() and not config.get("smoke_only"):
            raise RuntimeError("final evaluation waits for both training reports")
    path=resolve(checkpoint) if checkpoint else root/arm/f"decoder_u{config['protocol']['train_updates']:05d}.pt"
    payload=torch.load(path,map_location="cpu",weights_only=False)
    cache_report=json.loads((root/f"cache_report_{arm}.json").read_text())
    required={"schema":core.CHECKPOINT_SCHEMA,"experiment_id":config["experiment_id"],"arm":arm,"update":config["protocol"]["train_updates"],"repo_commit":config["repo_commit"],"decoder_kwargs":core.decoder_kwargs(config),"episode_schedule_sha256":manifest["train_episodes_sha256"],"decoder_initialization_sha256":manifest["decoder_initialization"]["tensor_state_sha256"],"encoder_tensor_sha256":cache_report["encoder_tensor_sha256_after"],"encoder_frozen":True}
    if any(payload.get(k)!=v for k,v in required.items()):raise ValueError("evaluation checkpoint is not the exact matched final arm checkpoint")
    decoder=core._checkpoint_decoder(path,config).cuda()
    out=root/arm/"evaluation"/mode;out.mkdir(parents=True,exist_ok=True)
    records={r["scene"]:r for r in manifest["validation_scenes"]};cache=CacheLRU(root,arm,2);episodes=manifest["panels"][mode]
    if config.get("smoke_only") and mode=="SO":episodes=episodes[:2]
    completed=[];result_tmp=out/"official_results.csv.tmp"
    with result_tmp.open("w") as results:
        for index,episode in enumerate(episodes):
            key=episode["scene"]+("" if mode=="MO" else f"_obj_{episode['objects'][0]}")
            trace_path=out/"traces"/f"{key}.json"
            if trace_path.exists():
                trace=json.loads(trace_path.read_text())
                if trace["checkpoint_sha256"]!=file_sha256(path):raise ValueError("evaluation resume checkpoint drift")
            else:
                predictions=out/"predictions"/f"{key}.npz" if index<config["protocol"]["visual_scene_count"] else None
                trace=evaluate_episode(decoder,cache.get(episode["scene"]),records[episode["scene"]],episode,config,predictions)
                trace["checkpoint_sha256"]=file_sha256(path);dump(trace_path,trace)
            results.writelines(protocol.official_lines(index,trace,mode))
            completed.append({"identity":key,"scene":episode["scene"],"object_count":trace["object_count"],"objects":trace["objects"],"metrics":trace["metrics"],"trace":str(trace_path)})
            if index%10==0:print(json.dumps({"mode":mode,"arm":arm,"completed":index+1,"total":len(episodes)}),flush=True)
    os.replace(result_tmp,out/"official_results.csv")
    report={"arm":arm,"mode":mode,"units":completed,"count":len(completed),"metrics":{metric:float(np.mean([r["metrics"][metric] for r in completed])) for metric in completed[0]["metrics"]},"official_csv":str(out/"official_results.csv"),"official_csv_sha256":file_sha256(out/"official_results.csv"),"checkpoint_sha256":file_sha256(path),"metric_scope":"matched LitePT comparison under AGILE3D metric protocol"}
    dump(out/"evaluation_report.json",report);return report


def aggregate(config):
    root=root_of(config);result={"experiment_id":config["experiment_id"],"endpoints":{}}
    for mode in ("MO","SO"):
        reports={a:json.loads((root/a/"evaluation"/mode/"evaluation_report.json").read_text()) for a in ("public","delimit3d")}
        maps={a:{r["identity"]:r["metrics"] for r in reports[a]["units"]} for a in reports}
        paired=protocol.paired_bootstrap(maps["public"],maps["delimit3d"],**config["bootstrap"])
        groups=defaultdict(list)
        for row in reports["public"]["units"]:
            groups[f"requested_count/{row['object_count']}"] .append(row["identity"])
            if mode=="SO":
                obj=row["objects"][0];n=obj["instance_points"]
                groups["size/"+("small_lt1000" if n<1000 else "medium_1000to9999" if n<10000 else "large_ge10000")].append(row["identity"])
                groups[f"semantic/{obj['semantic_name']}"] .append(row["identity"])
        strata={name:protocol.paired_bootstrap({k:maps['public'][k] for k in keys},{k:maps['delimit3d'][k] for k in keys},**config["bootstrap"]) for name,keys in groups.items()}
        result["endpoints"][mode]={"metrics":{a:reports[a]["metrics"] for a in reports},"paired_bootstrap":paired,"bootstrap_unit":"scene" if mode=="MO" else "object","strata":strata}
        if mode=="MO":
            # Target-level MO strata are descriptive: the official primary unit is a scene.
            object_maps={a:{} for a in reports};object_groups=defaultdict(list)
            for a,report in reports.items():
                for row in report["units"]:
                    trace=json.loads(resolve(row["trace"]).read_text())
                    for local_id,obj in enumerate(trace["objects"],1):
                        key=f"{row['scene']}/{obj['instance']}"
                        object_trace={"object_count":trace["object_count"],"states":[dict(s,mean_iou=s["object_ious"][str(local_id)]) for s in trace["states"]]}
                        object_maps[a][key]=protocol.metrics_from_trace(object_trace)
                        if a=="public":
                            n=obj["instance_points"];object_groups["size/"+("small_lt1000" if n<1000 else "medium_1000to9999" if n<10000 else "large_ge10000")].append(key)
                            object_groups[f"semantic/{obj['semantic_name']}"] .append(key)
            result["endpoints"][mode]["descriptive_object_strata"]={name:protocol.paired_bootstrap({k:object_maps['public'][k] for k in keys},{k:object_maps['delimit3d'][k] for k in keys},**config["bootstrap"]) for name,keys in object_groups.items()}
    dump(root/"aggregate.json",result)
    plot_results(config,result)
    return result


def plot_results(config,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=root_of(config);out=root/"visuals";out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,mode in zip(axes,("MO","SO")):
        for arm,color in (("public","#6d7787"),("delimit3d","#187d83")):
            m=result["endpoints"][mode]["metrics"][arm]
            ax.plot(protocol.CLICKS,[100*m[f"IoU@{x}"] for x in protocol.CLICKS],"o-",label=arm,color=color)
        ax.set(title=f"ScanNet40 {mode}",xlabel="Clicks per requested object",ylabel="IoU (%)",xticks=protocol.CLICKS);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/"interactive_iou.png",dpi=180);fig.savefig(out/"interactive_iou.pdf");plt.close(fig)
    manifest,_=load_prepared(config);records={r["scene"]:r for r in manifest["validation_scenes"]}
    for trace_path in sorted((root/"public/evaluation/MO/traces").glob("*.json"))[:config["protocol"]["visual_scene_count"]]:
        public=json.loads(trace_path.read_text());scene=public["scene"]
        if not public.get("prediction_path"):continue
        points,_,_,_,_=protocol.load_scene(records[scene],resolve);indices=np.linspace(0,len(points)-1,min(30000,len(points)),dtype=int)
        fig,axes=plt.subplots(2,3,figsize=(12,7),subplot_kw={"projection":"3d"})
        for i,arm in enumerate(("public","delimit3d")):
            trace=json.loads((root/arm/"evaluation/MO/traces"/trace_path.name).read_text());cache=core.load_cache(root,arm,scene)
            with np.load(resolve(trace["prediction_path"])) as pred:
                for j,click in enumerate((1,5,15)):
                    labels=pred[f"click_{click}"][cache["inverse_map"].numpy()][indices]
                    ax=axes[i,j];ax.scatter(*points[indices].T,c=labels,cmap="tab20",s=.3,vmin=0,vmax=20);ax.set_title(f"{arm} · {click} clicks/object");ax.set_axis_off();ax.view_init(35,-60)
        fig.tight_layout();fig.savefig(out/f"{scene}_interactive.png",dpi=180);plt.close(fig)


def gpu_preflight(config):
    inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total","--format=csv,noheader"],text=True)
    print(inventory,flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError("exactly one bound CUDA GPU required")
    name=torch.cuda.get_device_name(0)
    if config.get("smoke_only") and "RTX 4090" not in name:raise ValueError("real smoke requires RTX4090")
    if not any(x in name for x in ("RTX 3090","RTX 4090","A100")) or "2080" in name:raise ValueError(f"unapproved GPU {name}")
    report={"host":platform.node(),"pid":os.getpid(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu_name":name,"nvidia_smi":inventory,"torch":torch.__version__,"python":sys.version,"cuda":torch.version.cuda}
    return report


@contextlib.contextmanager
def worker_files(config,name):
    directory=root_of(config)/"workers";directory.mkdir(exist_ok=True)
    dump(directory/f"{name}.pid.json",{"pid":os.getpid(),"host":platform.node(),"name":name,"tmux":os.environ.get("TMUX"),"slurm_job_id":os.environ.get("SLURM_JOB_ID")})
    stop=threading.Event()
    def beat():
        while not stop.is_set():
            with (directory/f"{name}.heartbeat.jsonl").open("a") as h:h.write(json.dumps({"pid":os.getpid(),"time":time.time(),"host":platform.node()})+"\n")
            stop.wait(30)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join(timeout=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--mode",required=True,choices=("freeze","prepare","cache-features","smoke","train","evaluate","aggregate","preflight"))
    parser.add_argument("--arm",choices=("public","delimit3d"));parser.add_argument("--panel",choices=("MO","SO"),default="MO");parser.add_argument("--checkpoint");parser.add_argument("--resume")
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text());verify_config(config);install_adapter()
    if args.mode=="freeze":freeze(config);return
    if args.mode=="prepare":prepare(config);return
    verify_freeze(config)
    if args.mode in ("train","cache-features","smoke","evaluate"):
        if not config.get("smoke_only"):
            from launch_scannet40_matched import gate
            gate(config)
        environment=gpu_preflight(config);dump(root_of(config)/"workers"/f"environment_{args.mode}_{args.arm or 'both'}_{os.getpid()}.json",environment)
    with worker_files(config,f"{args.mode}_{args.arm or 'both'}_{args.panel}"):
        if args.mode=="cache-features":cache_features(config,args.arm)
        elif args.mode=="smoke":verify_caches(config);core.smoke(config)
        elif args.mode=="train":train(config,args.arm,args.resume)
        elif args.mode=="evaluate":evaluate(config,args.arm,args.panel,args.checkpoint)
        elif args.mode=="aggregate":aggregate(config)
        elif args.mode=="preflight":print(json.dumps(verify_freeze(config),indent=2))

if __name__=="__main__":main()
