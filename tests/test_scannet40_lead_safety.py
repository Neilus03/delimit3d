"""Adversarial CPU integration checks for Stage-B experiment identity boundaries.

No GPU calls: the resume harness retains real CPU tensors/checkpoint/AdamW state,
but stops before RNG restore/training; evaluation stops before decoder CUDA load.
"""
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/evaluation"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("run_scannet40_matched")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))


@pytest.fixture
def frozen(tmp_path, monkeypatch, runner):
    r = runner
    root = tmp_path / r.protocol.EXPERIMENT
    freeze = root / "freeze"
    freeze.mkdir(parents=True)
    source = tmp_path / "executing_source"
    source.mkdir()
    (source / "runner.py").write_text("ORIGINAL = True\n")
    monkeypatch.setattr(r, "REPO", source)
    official = tmp_path / "official"
    config = {"experiment_id": r.protocol.EXPERIMENT, "repo_commit": "commit-a",
              "paths": {"output_root": str(root), "official_root": str(official),
                        "litept_root": str(freeze / "litept_source")},
              "arms": {}, "optimizer": {"learning_rate": 0.0001}}
    lists = {}
    for name, path in r.official_manifests(config).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + "-official-identity").encode())
        lists[name] = {"path": str(path), "sha256": r.file_sha256(path)}
    checkpoints = {}
    for arm in ("public", "delimit3d"):
        p = tmp_path / (arm + ".pt")
        p.write_bytes((arm + "-encoder").encode())
        config["arms"][arm] = {"checkpoint": str(p), "checkpoint_sha256": r.file_sha256(p)}
        checkpoints[arm] = {"path": str(p), "sha256": r.file_sha256(p)}
    init = root / "decoder_init.pt"
    init.write_bytes(b"frozen-init-file")
    for name in ("source.tar", "environment.json", "litept.tar"):
        (freeze / name).write_bytes(name.encode())
    dep = freeze / "litept_source/model.py"
    dep.parent.mkdir()
    dep.write_text("FROZEN_DEPENDENCY = True\n")
    cfg = freeze / "resolved_config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    init_report = {"tensor_state_sha256": "fixed-init-tensor"}
    provenance = {"repo_commit": "commit-a",
                  "source_entries": {"runner.py": r.file_sha256(source / "runner.py")},
                  "decoder_initialization_file_sha256": r.file_sha256(init),
                  "decoder_initialization": init_report,
                  "source_archive": str(freeze / "source.tar"),
                  "source_archive_sha256": r.file_sha256(freeze / "source.tar"),
                  "resolved_config": str(cfg), "resolved_config_sha256": r.file_sha256(cfg),
                  "environment": str(freeze / "environment.json"),
                  "environment_sha256": r.file_sha256(freeze / "environment.json"),
                  "lists": lists, "checkpoints": checkpoints,
                  "litept_dependency": {"archive": str(freeze / "litept.tar"),
                     "archive_sha256": r.file_sha256(freeze / "litept.tar"),
                     "source_entries": {"model.py": r.file_sha256(dep)}}}
    write_json(freeze / "provenance.json", provenance)
    return {"config": config, "root": root, "source": source, "provenance": provenance,
            "init_report": init_report, "dependency": dep}


def test_frozen_files_and_runtime_config_baseline(runner, frozen):
    assert runner.verify_freeze(frozen["config"]) == frozen["provenance"]


def test_runtime_optimizer_drift_is_rejected_even_with_intact_yaml(runner, frozen):
    config = copy.deepcopy(frozen["config"])
    config["optimizer"]["learning_rate"] *= 10
    with pytest.raises(ValueError, match="runtime config"):
        runner.verify_freeze(config)


@pytest.mark.parametrize("which", ["source", "init", "dependency", "yaml"])
def test_file_bytes_cannot_drift_behind_unchanged_hash_claims(runner, frozen, which):
    path = {"source": frozen["source"] / "runner.py",
            "init": frozen["root"] / "decoder_init.pt",
            "dependency": frozen["dependency"],
            "yaml": frozen["root"] / "freeze/resolved_config.yaml"}[which]
    path.write_bytes(path.read_bytes() + b"\nDRIFT")
    with pytest.raises(ValueError):
        runner.verify_freeze(frozen["config"])


@pytest.fixture
def prepared(runner, frozen, monkeypatch):
    root = frozen["root"]
    schedule = root / "train_episodes.jsonl"
    schedule.write_text('{"update":1,"scene":"train"}\n')
    manifest = {"schema": runner.SCHEMA, "experiment_id": runner.protocol.EXPERIMENT,
                "repo_commit": "commit-a",
                "provenance_sha256": runner.file_sha256(root / "freeze/provenance.json"),
                "decoder_initialization_path": str(root / "decoder_init.pt"),
                "decoder_initialization": frozen["init_report"],
                "train_episodes_path": str(schedule), "train_episodes_sha256": runner.file_sha256(schedule),
                "train_scenes": [{"scene": "train"}], "validation_scenes": [{"scene": "val"}]}
    monkeypatch.setattr(runner.core, "decoder_kwargs", lambda config: {})
    monkeypatch.setattr(runner, "load_initialization", lambda *args, **kwargs: (None, frozen["init_report"]))
    def save(m):
        m = copy.deepcopy(m)
        m.pop("manifest_sha256", None)
        m["manifest_sha256"] = runner.json_sha256(m)
        write_json(root / "selection_manifest.json", m)
        return m
    manifest = save(manifest)
    return dict(frozen, manifest=manifest, save=save)


def test_prepared_manifest_baseline(runner, prepared):
    assert runner.load_prepared(prepared["config"])[0] == prepared["manifest"]


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong-schema"), ("experiment_id", "different-study"),
    ("provenance_sha256", "unrelated-freeze")])
def test_self_rehashed_manifest_cannot_change_frozen_identity(runner, prepared, field, value):
    changed = copy.deepcopy(prepared["manifest"])
    changed[field] = value
    prepared["save"](changed)
    with pytest.raises(ValueError):
        runner.load_prepared(prepared["config"])


def test_manifest_cannot_alias_an_alternate_initialization(runner, prepared):
    alias = prepared["root"] / "alternate_init.pt"
    alias.write_bytes((prepared["root"] / "decoder_init.pt").read_bytes())
    changed = copy.deepcopy(prepared["manifest"])
    changed["decoder_initialization_path"] = str(alias)
    prepared["save"](changed)
    with pytest.raises(ValueError, match="frozen initialization"):
        runner.load_prepared(prepared["config"])


@pytest.fixture
def caches(runner, prepared):
    reports, payloads = {}, {}
    m, root = prepared["manifest"], prepared["root"]
    for arm in ("public", "delimit3d"):
        rows = []
        for scene in ("train", "val"):
            p = root / "feature_cache" / arm / (scene + ".pt")
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema": runner.core.CACHE_SCHEMA, "arm": arm, "scene": scene,
                       "checkpoint_sha256": prepared["config"]["arms"][arm]["checkpoint_sha256"],
                       "selection_manifest_sha256": m["manifest_sha256"],
                       "repo_commit": "commit-a", "encoder_tensor_sha256": arm + "-encoder",
                       "geometry_sha256": scene + "-geometry", "empty_objects": [],
                       "features": torch.ones(2, 72)}
            torch.save(payload, p)
            rows.append({"scene": scene, "path": str(p), "geometry_sha256": scene + "-geometry",
                         "tokens": 2, "file_sha256": runner.file_sha256(p)})
            payloads[(arm, scene)] = payload
        report = {"arm": arm, "selection_manifest_sha256": m["manifest_sha256"],
                  "checkpoint_sha256": prepared["config"]["arms"][arm]["checkpoint_sha256"],
                  "repo_commit": "commit-a", "encoder_state_unchanged": True,
                  "encoder_tensor_sha256_before": arm + "-encoder",
                  "encoder_tensor_sha256_after": arm + "-encoder", "cache_rows": rows}
        write_json(root / ("cache_report_" + arm + ".json"), report)
        reports[arm] = report
    return dict(prepared, reports=reports, payloads=payloads)


def test_complete_matched_cache_baseline(runner, caches):
    result = runner.verify_caches(caches["config"])
    assert result["geometry_byte_identity"] and result["scene_count"] == 2


def test_equal_incomplete_cache_subsets_are_rejected(runner, caches):
    for arm, report in caches["reports"].items():
        report["cache_rows"] = report["cache_rows"][:1]
        write_json(caches["root"] / ("cache_report_" + arm + ".json"), report)
    with pytest.raises(ValueError, match="exact manifest scene set"):
        runner.verify_caches(caches["config"])


@pytest.mark.parametrize("field,value", [
    ("arm", "delimit3d"), ("checkpoint_sha256", "wrong-checkpoint"),
    ("selection_manifest_sha256", "wrong-manifest"), ("repo_commit", "wrong-commit")])
def test_cache_report_identity_is_bound_to_requested_arm(runner, caches, field, value):
    report = caches["reports"]["public"]
    report[field] = value
    write_json(caches["root"] / "cache_report_public.json", report)
    with pytest.raises(ValueError):
        runner.verify_caches(caches["config"])


@pytest.mark.parametrize("field,value", [
    ("arm", "delimit3d"), ("scene", "other-scene"), ("checkpoint_sha256", "wrong-checkpoint"),
    ("selection_manifest_sha256", "wrong-manifest"), ("encoder_tensor_sha256", "wrong-encoder")])
def test_payload_identity_rejected_even_if_binary_hash_is_refreshed(runner, caches, field, value):
    payload = caches["payloads"][("public", "train")]
    payload[field] = value
    path = caches["root"] / "feature_cache/public/train.pt"
    torch.save(payload, path)
    report = caches["reports"]["public"]
    report["cache_rows"][0]["file_sha256"] = runner.file_sha256(path)
    write_json(caches["root"] / "cache_report_public.json", report)
    with pytest.raises(ValueError):
        runner.verify_caches(caches["config"])


def test_cache_feature_bytes_tamper_rejected_with_intact_metadata(runner, caches):
    path = caches["root"] / "feature_cache/public/train.pt"
    payload = caches["payloads"][("public", "train")]
    payload["features"][0, 0] += 100
    torch.save(payload, path)
    with pytest.raises(ValueError, match="binary hash"):
        runner.verify_caches(caches["config"])


class ReachedResumeRestore(Exception):
    pass


class ReachedEvaluationDecoder(Exception):
    pass


@pytest.fixture
def resume_harness(tmp_path, monkeypatch, runner):
    root = tmp_path / runner.protocol.EXPERIMENT
    root.mkdir()
    class CPUDecoder(torch.nn.Linear):
        def to(self, *args, **kwargs):
            assert args == ("cuda",)
            return self  # CPU-only adversarial harness; never calls CUDA.
    decoder = CPUDecoder(2, 1)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.0001, weight_decay=0.0001)
    initial = {"tensor_state_sha256": "same-init"}
    manifest = {"train_episodes_sha256": "same-schedule", "train_episodes_path": str(root / "episodes.jsonl"),
                "decoder_initialization": initial, "validation_scenes": [], "panels": {"MO": []}}
    config = {"experiment_id": runner.protocol.EXPERIMENT, "repo_commit": "commit-a",
              "paths": {"output_root": str(root)}, "seed": 1,
              "protocol": {"train_updates": 2},
              "optimizer": {"learning_rate": 0.0001, "weight_decay": 0.0001, "cache_memory_scenes": 1}}
    for arm in ("public", "delimit3d"):
        (root / arm).mkdir()
        write_json(root / arm / "train_report.json", {"updates": 2})
        write_json(root / ("cache_report_" + arm + ".json"),
                   {"encoder_state_unchanged": True, "encoder_tensor_sha256_after": arm + "-encoder"})
    ckpt = {"schema": runner.core.CHECKPOINT_SCHEMA, "experiment_id": config["experiment_id"],
            "arm": "public", "repo_commit": "commit-a", "update": 1, "decoder_kwargs": {},
            "decoder_state_dict": copy.deepcopy(decoder.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "decoder_tensor_sha256": runner.state_sha256(decoder.state_dict()),
            "decoder_initialization_sha256": "same-init", "episode_schedule_sha256": "same-schedule",
            "encoder_frozen": True, "encoder_tensor_sha256": "public-encoder",
            "rng_state": {"python": __import__("random").getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(), "cuda": [torch.zeros(16, dtype=torch.uint8)]}}
    checkpoint = root / "resume.pt"
    monkeypatch.setattr(runner, "load_prepared", lambda cfg: (manifest, root / "decoder_init.pt"))
    monkeypatch.setattr(runner, "verify_caches", lambda cfg: {"geometry_byte_identity": True})
    monkeypatch.setattr(runner, "load_initialization", lambda *args, **kwargs: (decoder, initial))
    monkeypatch.setattr(runner.core, "decoder_kwargs", lambda cfg: {})
    monkeypatch.setattr(runner.core, "set_seed", lambda seed: None)
    def restore(value):
        raise ReachedResumeRestore
    monkeypatch.setattr(runner, "restore_rng", restore)
    def evaluate_decoder(*args, **kwargs):
        raise ReachedEvaluationDecoder
    monkeypatch.setattr(runner.core, "_checkpoint_decoder", evaluate_decoder)
    def save(value):
        torch.save(value, checkpoint)
    save(ckpt)
    return {"config": config, "checkpoint": checkpoint, "ckpt": ckpt, "save": save}


def test_valid_resume_reaches_rng_restore_without_gpu(runner, resume_harness):
    with pytest.raises(ReachedResumeRestore):
        runner.train(resume_harness["config"], "public", resume=resume_harness["checkpoint"])


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong-schema"), ("experiment_id", "wrong-study"), ("arm", "delimit3d"),
    ("repo_commit", "wrong-commit"), ("decoder_kwargs", {"different": True}),
    ("decoder_initialization_sha256", "wrong-init"), ("episode_schedule_sha256", "wrong-schedule"),
    ("encoder_tensor_sha256", "wrong-encoder"), ("encoder_frozen", False),
    ("decoder_tensor_sha256", "wrong-tensor-hash"), ("update", 0), ("update", 3),
    ("rng_state", {"python": None, "numpy": None, "torch": None})])
def test_invalid_resume_is_rejected_before_rng_restore(runner, resume_harness, field, value):
    ckpt = copy.deepcopy(resume_harness["ckpt"])
    ckpt[field] = value
    resume_harness["save"](ckpt)
    with pytest.raises(ValueError):
        runner.train(resume_harness["config"], "public", resume=resume_harness["checkpoint"])


def test_optimizer_rejects_non_decoder_parameter(runner, resume_harness, monkeypatch):
    real = torch.optim.AdamW
    def contaminated(parameters, **kwargs):
        opt = real(parameters, **kwargs)
        opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))]})
        return opt
    monkeypatch.setattr(torch.optim, "AdamW", contaminated)
    with pytest.raises(ValueError, match="outside decoder"):
        runner.train(resume_harness["config"], "public", resume=resume_harness["checkpoint"])


def test_final_evaluation_rejects_interim_checkpoint_before_cuda(runner, resume_harness):
    with pytest.raises(ValueError, match="exact matched final"):
        runner.evaluate(resume_harness["config"], "public", "MO", checkpoint=resume_harness["checkpoint"])


@pytest.mark.parametrize("field,value", [
    ("arm", "delimit3d"), ("encoder_tensor_sha256", "wrong-encoder"),
    ("decoder_initialization_sha256", "wrong-init"), ("episode_schedule_sha256", "wrong-schedule")])
def test_final_evaluation_rejects_wrong_identity_before_cuda(runner, resume_harness, field, value):
    ckpt = copy.deepcopy(resume_harness["ckpt"])
    ckpt["update"] = 2
    ckpt[field] = value
    resume_harness["save"](ckpt)
    with pytest.raises(ValueError, match="exact matched final"):
        runner.evaluate(resume_harness["config"], "public", "MO", checkpoint=resume_harness["checkpoint"])


def test_valid_final_evaluation_reaches_decoder_without_gpu(runner, resume_harness):
    ckpt = copy.deepcopy(resume_harness["ckpt"])
    ckpt["update"] = 2
    resume_harness["save"](ckpt)
    with pytest.raises(ReachedEvaluationDecoder):
        runner.evaluate(resume_harness["config"], "public", "MO", checkpoint=resume_harness["checkpoint"])


def test_cache_reuse_cannot_republish_corrupted_existing_binary(runner, caches, monkeypatch):
    path = caches["root"] / "feature_cache/public/train.pt"
    payload = caches["payloads"][("public", "train")]
    payload["features"][0, 0] += 100
    torch.save(payload, path)
    old_report = copy.deepcopy(caches["reports"]["public"])
    # The frozen core's reuse path only verifies metadata and returns fresh rows.
    # Simulate that GPU extraction boundary without ever invoking a GPU.
    def reused_core(config, arm):
        result = copy.deepcopy(old_report)
        for row in result["cache_rows"]:
            row.pop("file_sha256", None)
            row["reused"] = True
        return result
    monkeypatch.setattr(runner.core, "cache_features", reused_core)
    with pytest.raises(ValueError):
        runner.cache_features(caches["config"], "public")


def test_resume_rejects_empty_cuda_rng_stream(runner, resume_harness):
    ckpt = copy.deepcopy(resume_harness["ckpt"])
    ckpt["rng_state"]["cuda"] = []
    resume_harness["save"](ckpt)
    with pytest.raises(ValueError):
        runner.train(resume_harness["config"], "public", resume=resume_harness["checkpoint"])


@pytest.mark.parametrize("field,value", [("lr", 0.5), ("weight_decay", 0.5)])
def test_resume_optimizer_recipe_cannot_override_frozen_yaml(runner, resume_harness, field, value):
    ckpt = copy.deepcopy(resume_harness["ckpt"])
    ckpt["optimizer_state_dict"]["param_groups"][0][field] = value
    resume_harness["save"](ckpt)
    with pytest.raises(ValueError):
        runner.train(resume_harness["config"], "public", resume=resume_harness["checkpoint"])


def test_bounded_smoke_pipeline_rejects_production_before_gpu(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production config reached GPU preflight")
    monkeypatch.setattr(runner, "gpu_preflight", forbidden)
    with pytest.raises(ValueError, match="rejects production"):
        runner.smoke_pipeline({"smoke_only": False})
