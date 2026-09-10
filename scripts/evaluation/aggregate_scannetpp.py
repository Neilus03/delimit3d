"""Merge four frozen shard outputs, validate matching inputs/prompts, and aggregate."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("DELIMIT3D_PROMPT_ROOT", os.environ.get("FULLVAL_ROOT", ""))).expanduser().resolve(strict=True)
OUT = ROOT / "aggregate"
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / "freeze/manifest.json").read_text())
models = ("public", "posttrained")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


reports = {}
scenes = {}
for model in models:
    reports[model] = []
    merged = {}
    for shard in (0,):
        base = ROOT / f"results/{model}/shard{shard}"
        report = json.loads((base / "report.json").read_text())
        assert report["passed"] and report["model"] == model and report["shard"] == shard
        assert report["checkpoint_sha256"] == manifest["assets"][model]["sha256"]
        assert report["records_sha256"] == file_sha(base / "scenes.jsonl")
        reports[model].append(report)
        for record in read_jsonl(base / "scenes.jsonl"):
            assert record["scene"] not in merged
            merged[record["scene"]] = record
    scenes[model] = merged

expected = [line.strip() for line in (ROOT / "freeze/scenes.txt").read_text().splitlines() if line.strip()]
assert sorted(scenes["public"]) == sorted(expected)
assert sorted(scenes["posttrained"]) == sorted(expected)

query_rows = []
object_rows = []
scene_rows = []
metric_keys = ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "coverage50_fixed")
for scene in expected:
    pub = scenes["public"][scene]
    post = scenes["posttrained"][scene]
    assert pub["input_sha256"] == post["input_sha256"]
    assert pub["gt_instance_sha256"] == post["gt_instance_sha256"]
    assert pub["objects"] == post["objects"]
    assert len(pub["queries"]) == len(post["queries"])
    paired = {}
    for model, record in (("public", pub), ("posttrained", post)):
        for row in record["queries"]:
            key = (int(row["instance"]), int(row["query"]))
            paired.setdefault(key, {})[model] = row
            query_rows.append({"scene": scene, "model": model, **row})
    for key in sorted(paired):
        assert set(paired[key]) == set(models)
        for field in ("semantic_class", "instance_points", "candidate_points", "target_points_excluding_query", "query_coord"):
            assert paired[key]["public"][field] == paired[key]["posttrained"][field]
    for object_record in pub["objects"]:
        instance = int(object_record["instance"])
        row = {
            "scene": scene,
            "instance": instance,
            "semantic_class": int(object_record["semantic_class"]),
            "instance_points": int(object_record["instance_points"]),
        }
        for model in models:
            queries = [v[model] for key, v in paired.items() if key[0] == instance]
            assert len(queries) == manifest["protocol"]["prompts_per_object"]
            for metric in metric_keys:
                row[f"{model}_{metric}"] = float(np.mean([float(q[metric]) for q in queries]))
        for metric in metric_keys:
            row[f"delta_{metric}"] = row[f"posttrained_{metric}"] - row[f"public_{metric}"]
        object_rows.append(row)
    scene_objects = [r for r in object_rows if r["scene"] == scene]
    scene_row = {"scene": scene, "objects": len(scene_objects), "queries": len(pub["queries"]), "points": pub["points"]}
    for model in models:
        for metric in metric_keys:
            scene_row[f"{model}_{metric}"] = float(np.mean([r[f"{model}_{metric}"] for r in scene_objects])) if scene_objects else float("nan")
    for metric in metric_keys:
        scene_row[f"delta_{metric}"] = scene_row[f"posttrained_{metric}"] - scene_row[f"public_{metric}"]
    scene_rows.append(scene_row)


def mean(rows, field):
    values = np.asarray([r[field] for r in rows], dtype=float)
    return float(np.nanmean(values))


rng = np.random.default_rng(20260909)
valid_scene_rows = [r for r in scene_rows if r["objects"] > 0]
bootstrap = {}
for metric in metric_keys:
    values = np.asarray([r[f"delta_{metric}"] for r in valid_scene_rows], dtype=float)
    samples = np.mean(values[rng.integers(0, len(values), size=(20000, len(values)))], axis=1)
    bootstrap[metric] = {
        "paired_scene_mean_delta": float(values.mean()),
        "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "scenes_positive": int(np.sum(values > 0)),
        "scenes_negative": int(np.sum(values < 0)),
        "scenes_tied": int(np.sum(values == 0)),
    }

aggregate = {}
for model in models:
    aggregate[model] = {metric: mean(object_rows, f"{model}_{metric}") for metric in metric_keys}
delta = {metric: aggregate["posttrained"][metric] - aggregate["public"][metric] for metric in metric_keys}
summary = {
    "project_name": os.environ.get("DELIMIT3D_NAME", "Delimit3D"),
    "schema": "chorus_point_prompted_fullval_aggregate/v1",
    "protocol": manifest["protocol"],
    "checkpoints": {model: manifest["assets"][model] for model in models},
    "jobs": {model: [r["job_id"] for r in reports[model]] for model in models},
    "counts": {
        "official_validation_scenes": len(expected),
        "scenes_with_eligible_objects": len(valid_scene_rows),
        "objects": len(object_rows),
        "queries": len(query_rows) // 2,
    },
    "aggregate_equal_object_weight": aggregate,
    "posttrained_minus_public": delta,
    "paired_scene_bootstrap": bootstrap,
    "predeclared_screen": {
        "rule": "posttrained-public >= +0.02 AP and >= +0.02 fixed IoU, and positive AP in at least two thirds of eligible scenes",
        "passed": bool(
            delta["ap"] >= 0.02
            and delta["iou_fixed"] >= 0.02
            and bootstrap["ap"]["scenes_positive"] >= np.ceil(2 * len(valid_scene_rows) / 3)
        ),
    },
    "full_validation_inference": {
        "rule": "paired scene-bootstrap 95% lower bound above zero for AP and fixed IoU",
        "passed": bool(bootstrap["ap"]["ci95"][0] > 0 and bootstrap["iou_fixed"]["ci95"][0] > 0),
    },
}

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
for name, rows in (("queries.csv", query_rows), ("objects.csv", object_rows), ("scenes.csv", scene_rows)):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

# Compact, reproducible plots use object-level estimates and paired scene deltas.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, title in zip(axes, ("ap", "iou_fixed", "oracle_iou"), ("Retrieval AP", "IoU at cosine 0.70", "Oracle-threshold IoU")):
    values = [100 * aggregate[m][metric] for m in models]
    ax.bar([0, 1], values, color=["#5B718A", "#D26A3A"])
    ax.set_xticks([0, 1], ["Public LitePT", "Posttrained 256 updates"])
    ax.set_ylim(0, 100); ax.set_ylabel("% · equal object weight"); ax.set_title(title); ax.grid(axis="y", alpha=.2)
    for i, value in enumerate(values): ax.text(i, value + 1, f"{value:.2f}", ha="center")
fig.suptitle(f"Point-prompted class-agnostic object recovery · ScanNet++ val · {len(object_rows)} objects · {len(query_rows)//2} prompts")
fig.tight_layout(); fig.savefig(OUT / "aggregate_metrics.png", dpi=190); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, title in zip(axes, ("ap", "iou_fixed"), ("Per-scene AP change", "Per-scene fixed-IoU change")):
    values = 100 * np.asarray([r[f"delta_{metric}"] for r in valid_scene_rows])
    ax.axvline(0, color="black", lw=1); ax.hist(values, bins=30, color="#D26A3A", alpha=.85)
    lo, hi = bootstrap[metric]["ci95"]
    ax.set_xlabel("Posttrained minus public · percentage points"); ax.set_ylabel("Scenes"); ax.set_title(title + f"\nmean {100*values.mean()/100:+.2f}, paired 95% CI [{100*lo:+.2f}, {100*hi:+.2f}]")
    ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(OUT / "paired_scene_deltas.png", dpi=190); plt.close(fig)

hashes = {p.name: file_sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
(OUT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2))
print(json.dumps(summary, indent=2))
