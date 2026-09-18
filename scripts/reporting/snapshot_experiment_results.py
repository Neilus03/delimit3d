#!/usr/bin/env python3
"""Extract compact, hashed experiment evidence without copying per-scene data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def snapshot(root: Path) -> dict:
    sources = {}

    def read(relative: str) -> dict:
        path = root / relative
        raw = path.read_bytes()
        sources[relative] = {
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()
        }
        return json.loads(raw)

    frozen = {}
    for dataset, directory, counts in (
        ("ScanNet40", "delimit3d_scannet40_agile3d_matched_v1", {"MO": 312, "SO": 10357}),
        ("S3DIS", "delimit3d_s3dis_agile3d_matched_v2", {"MO": 53, "SO": 2330}),
    ):
        aggregate = read(f"{directory}/aggregate.json")
        endpoints = {}
        for endpoint, expected in counts.items():
            identities = []
            for arm in ("public", "delimit3d"):
                report = read(f"{directory}/{arm}/evaluation/{endpoint}/evaluation_report.json")
                if report["count"] != expected or len(report["units"]) != expected:
                    raise ValueError(f"Incomplete {dataset}/{endpoint}/{arm}")
                ids = [unit["identity"] for unit in report["units"]]
                if len(set(ids)) != expected:
                    raise ValueError(f"Duplicate identities in {dataset}/{endpoint}/{arm}")
                identities.append(set(ids))
            if identities[0] != identities[1]:
                raise ValueError(f"Unmatched {dataset}/{endpoint}")
            values = aggregate["endpoints"][endpoint]
            endpoints[endpoint] = {
                "count": expected, "metrics": values["metrics"],
                "paired_bootstrap": values["paired_bootstrap"],
                "bootstrap_unit": values["bootstrap_unit"],
            }
        frozen[dataset] = endpoints

    joint = {}
    for arm, directory in (
        ("public_control_A", "delimit3d_scannet40_agile3d_joint_v1_control_A_eval1"),
        ("delimit3d", "delimit3d_scannet40_agile3d_joint_v1_eval3"),
    ):
        report = read(f"{directory}/evaluation_report.json")
        if not report["complete"]:
            raise ValueError(f"Incomplete joint evaluation: {arm}")
        panels = {}
        for endpoint, expected in (("MO", 312), ("SO", 10357)):
            panel = report["panels"][endpoint]
            if not panel["complete"] or panel["count"] != expected:
                raise ValueError(f"Incomplete joint evaluation: {arm}/{endpoint}")
            panels[endpoint] = {k: panel[k] for k in ("count", "metrics")}
        joint[arm] = {"checkpoint_sha256": report["checkpoint_sha256"], "panels": panels}

    parts = read("delimit3d_articulate3d_agile3d_parts_eval_v1/aggregate.json")
    part_arms = {}
    for arm in ("public", "delimit3d"):
        values = parts[arm]
        if values["part_count_evaluated"] != 1958 or values["scene_count_evaluated"] != 42:
            raise ValueError(f"Incomplete Articulate3D arm: {arm}")
        part_arms[arm] = {k: values[k] for k in (
            "checkpoint_sha256", "encoder_checkpoint_sha256",
            "part_count_evaluated", "scene_count_evaluated",
            "metrics_by_minimum_surviving_tokens",
        )}
    feature = read("delimit3d_feature_selectivity_scannetval_20260915/aggregate.json")
    pretraining = read("structured3d_litept_hierarchy_5cm_336ep_20260917_v1/progress.json")
    mask = read("scratch_baselines_20260916/mask3d/litept_mask3d_ca_scratch_600_seed42/eval_epoch_0250.json")
    return {
        "schema": "delimit3d_experiment_evidence/v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "frozen_encoder_interactive": frozen,
        "joint_encoder_decoder_interactive": joint,
        "articulate3d_interactive": {
            "arms": part_arms, "metric_definition": parts["metric_definition"],
            "primary_minimum_surviving_tokens": 10,
            "paired_scene_bootstrap": {
                k: {field: value for field, value in v.items() if field != "scene_differences"}
                for k, v in parts["paired_scene_bootstrap_delimit3d_minus_public"].items()
            },
        },
        "scannet_feature_selectivity": {k: feature[k] for k in (
            "scene_count_completed", "object_count_completed", "pair_count_completed",
            "scene_balanced", "pair_balanced",
        )},
        "pretraining_feature_panels_interim": pretraining,
        "mask3d_scratch_validation_interim": {
            "epoch": mask["epoch"],
            "raw_queries": {k: mask["validation"]["raw_queries"][k] for k in (
                "CA-AP", "CA-AP50", "CA-AP25", "canonical_scannet_leaderboard_metric"
            )},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
