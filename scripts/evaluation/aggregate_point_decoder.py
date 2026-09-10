#!/usr/bin/env python3
"""Aggregate paired public/Delimit3D point-decoder arm reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _aggregate(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    keys = ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
    return {key: float(np.mean([float(row[field][key]) for row in rows])) for key in keys}


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve(strict=True)
    public = _load_jsonl(root / "public" / "queries.jsonl")
    delimit = _load_jsonl(root / "delimit3d" / "queries.jsonl")
    if len(public) != len(delimit):
        raise ValueError("public and Delimit3D query counts differ")
    public_by_key = {(row["scene"], int(row["instance"]), int(row["query"])): row for row in public}
    delimit_by_key = {(row["scene"], int(row["instance"]), int(row["query"])): row for row in delimit}
    if set(public_by_key) != set(delimit_by_key):
        raise ValueError("public and Delimit3D query identities differ")
    paired = []
    for key in sorted(public_by_key):
        p = public_by_key[key]
        d = delimit_by_key[key]
        paired.append(
            {
                "scene": key[0],
                "instance": key[1],
                "query": key[2],
                "public": p["decoder"],
                "delimit3d": d["decoder"],
                "delta": {
                    metric: float(d["decoder"][metric]) - float(p["decoder"][metric])
                    for metric in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
                },
            }
        )
    scenes = sorted({row["scene"] for row in paired})
    scene_rows = []
    for scene in scenes:
        rows = [row for row in paired if row["scene"] == scene]
        scene_rows.append(
            {
                "scene": scene,
                "queries": len(rows),
                "public": _aggregate([{"decoder": row["public"]} for row in rows], "decoder"),
                "delimit3d": _aggregate([{"decoder": row["delimit3d"]} for row in rows], "decoder"),
                "delta": {
                    metric: float(np.mean([row["delta"][metric] for row in rows]))
                    for metric in ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
                },
            }
        )
    metrics = ("ap", "iou_fixed", "precision_fixed", "recall_fixed", "oracle_iou", "f1_fixed")
    summary = {
        "schema": "delimit3d_scannetpp_point_decoder_aggregate/v1",
        "query_count": len(paired),
        "scene_count": len(scene_rows),
        "public": {metric: float(np.mean([row["public"][metric] for row in paired])) for metric in metrics},
        "delimit3d": {metric: float(np.mean([row["delimit3d"][metric] for row in paired])) for metric in metrics},
        "delimit3d_minus_public": {metric: float(np.mean([row["delta"][metric] for row in paired])) for metric in metrics},
        "paired_scene": {
            metric: {
                "mean_delta": float(np.mean([row["delta"][metric] for row in scene_rows])),
                "scenes_positive": int(sum(row["delta"][metric] > 0 for row in scene_rows)),
                "scenes_negative": int(sum(row["delta"][metric] < 0 for row in scene_rows)),
                "scenes_tied": int(sum(row["delta"][metric] == 0 for row in scene_rows)),
            }
            for metric in metrics
        },
        "decision_rule": {
            "pass": "Delimit3D improves AP and fixed IoU by at least 0.02, with positive paired-scene AP in at least two thirds of scenes",
            "passed": (
                float(np.mean([row["delta"]["ap"] for row in paired])) >= 0.02
                and float(np.mean([row["delta"]["iou_fixed"] for row in paired])) >= 0.02
                and sum(row["delta"]["ap"] > 0 for row in scene_rows) >= (2 * len(scene_rows) + 2) // 3
            ),
        },
        "paired_queries": paired,
        "paired_scenes": scene_rows,
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("query_count", "scene_count", "public", "delimit3d", "delimit3d_minus_public", "decision_rule")}, indent=2))


if __name__ == "__main__":
    main()
