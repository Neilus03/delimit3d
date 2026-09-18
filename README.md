# Delimit3D

Delimit3D studies how region-supervised 3D representations transfer to object
and part segmentation. The current priority, following the 16 September 2026
supervisor meeting, is a controlled comparison of **randomly initialized LitePT
versus LitePT pretrained from scratch with PartField-style multigranular
supervision**. Evaluate both ordinary class-agnostic instance segmentation
(LitePT + mask-only Mask3D) and interactive segmentation (LitePT + AGILE3D).
Interactive results complement the automatic segmentation endpoint.

The earlier **public LitePT + 256-update Delimit3D adaptation** study remains a
separate, completed source of evidence. Its Structured3D adaptation uses
category-agnostic 2D pseudomasks; the 2D teacher is absent at inference. Those
results do not establish the benefit of the new scratch-pretraining recipe.

## Start here

- [Current status and next decisions](CURRENT.md): dated run snapshot and open comparison gaps.
- [Research direction](docs/RESEARCH.md): meeting decisions, hypotheses, and experiment priorities.
- [Verified results](docs/results/20260918_experiment_summary.md): completed comparisons, interim panels, and hashed sources.
- [Experiment map](docs/EXPERIMENTS.md): configurations, runners, branches, and external artifact roots.
- [Reproducibility protocol](docs/protocol.md): what must remain matched within each pair.
- [LitePT RGB3 AGILE3D baseline](docs/agile3d_litept_scratch.md): preflight, training, recovery, and validation.

## Evidence as of 18 September 2026

The table compares **LitePT (public)** with **LitePT + Delimit3D (256 updates)**,
using frozen encoders and each encoder's corresponding trained decoder.
IoU is in percent; MO means multi-object and SO means single-object.

| Evaluation | Units | IoU@5 public | IoU@5 adapted |
|---|---:|---:|---:|
| ScanNet40 MO | 312 scenes | 59.19 | 64.16 |
| ScanNet40 SO | 10,357 objects | 42.99 | 49.16 |
| S3DIS MO | 53 scenes | 45.82 | 52.38 |
| S3DIS SO | 2,330 objects | 37.82 | 45.93 |
| Articulate3D parts, ≥10 tokens | 1,663 parts / 42 scenes | 22.61 | 27.41 |

These are matched LitePT experiments using AGILE3D-style readouts and metric
protocols, not literal reproductions of the released AGILE3D model and full
training recipe. Articulate3D uses raw-point IoU after inverse-map expansion;
≥10 means the minimum surviving unambiguous feature tokens per part.
The [results ledger](docs/results/20260918_experiment_summary.md) also reports
the completed joint encoder/decoder control and the nearly unchanged one-click
Articulate3D result.

The new 5 cm pretraining and scratch baselines are still in progress. The
RGB3 AGILE3D baseline and RGBN6 pretraining are different input contracts;
resolve that difference before an initialization-only downstream comparison.

## Repository scope

This package contains LitePT/PTv3/Sonata adaptation, frozen-feature diagnostics,
interactive evaluation, the LitePT RGB3 scratch runner, and regression tests.
Historical experiments retain their original recipes and branch snapshots.
The 5 cm resolution contract applies to the new scratch/pretraining comparison;
it does not retroactively change the earlier 2 cm experiments.

Data, pretrained weights, checkpoints, rendered figures, and runtime logs live
outside Git. External LitePT, AGILE3D, Sonata, and CHORUS implementations are
not vendored. This is research infrastructure with explicit environment and
dataset dependencies, not a self-contained model release.

## Runtime identity and paths

The Python import package is stable as `delimit3d`. Human-facing run names and
default artifact locations are controlled by one variable:

```bash
export DELIMIT3D_NAME=Delimit3D
```

Changing `DELIMIT3D_NAME` changes the project label and, unless an explicit
artifact root is supplied, changes the default artifact subdirectory through a
normalized slug. The path variables in [`env.example`](env.example) are
examples for the Euler account and can be replaced on another host.

Useful defaults are exposed by `delimit3d.identity`:

```python
from delimit3d.identity import artifact_root, data_root, project_name
```

## Install

On Euler, use the existing LitePT environment or an equivalent environment
that provides the pinned LitePT dependencies (including its external
`litept` package). From this repository:

```bash
python -m pip install -e '.[plots,dev]'

# Only needed when voxelizing raw Structured3D geometry from the archives:
python -m pip install -e '.[sources]'
```

The LitePT source tree and its weights are intentionally not vendored. Set
`DELIMIT3D_LITEPT_ROOT` to the external LitePT checkout before loading a model.

## Main entry points

The current RGB3 scratch baseline has separate preflight, smoke, train and
evaluate modes. Start with the [runbook](docs/agile3d_litept_scratch.md):

```bash
python scripts/training/run_agile3d_litept_sstar_rgb3.py \
  --config configs/training/agile3d_litept_sstar_rgb3_scannet40.yaml \
  --mode preflight
```

The commands below illustrate the earlier adaptation interface. They are not
the resolved launch recipe for the current 336-epoch pretraining run; use its
immutable recipe and source record linked in the experiment map.

Create an exact random initialization or a public-LitePT initialization:

```bash
delimit3d-init-scratch \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" \
  --output "$DELIMIT3D_ARTIFACT_ROOT/initialization/scratch.pt" \
  --seed 42 --input-features rgbn6 --multiscale-supervision

delimit3d-init-public \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" \
  --scratch-initial "$DELIMIT3D_ARTIFACT_ROOT/initialization/scratch.pt" \
  --public-checkpoint /path/to/public_litept.pth \
  --expected-public-sha256 "<sha256>" \
  --output "$DELIMIT3D_ARTIFACT_ROOT/initialization/public.pt"
```

Run the existing distributed adaptation contract with a clean, immutable
source/config/checkpoint manifest:

```bash
python -m delimit3d.training.runner \
  --dataset structured3d \
  --source-family 2d \
  --source-root /path/to/structured3d/source_manifests \
  --train-split /path/to/structured3d/train.txt \
  --holdout-split /path/to/structured3d/holdout.txt \
  --initial-checkpoint "$DELIMIT3D_ARTIFACT_ROOT/initialization/public.pt" \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" \
  --output-dir "$DELIMIT3D_ARTIFACT_ROOT/runs/public_256" \
  --input-features rgbn6 --backbone-lr 0.0001 --head-lr 0.003
```

The runner retains the old scientific checkpoint/schema identifiers so the
existing validated source manifests and checkpoints remain loadable. New
resolved configs, checkpoints, and summaries also record `project_name` and
`project_slug` from `DELIMIT3D_NAME`.

The source preparation entry points consume external Structured3D archives and
UnSAMv2:

```bash
delimit3d-prepare-structured3d-raw --help
delimit3d-prepare-structured3d-sources --help
```

The current fixed point-prompted ScanNet++ evaluator is available as a script.
It consumes the frozen bundle manifest through environment variables and writes
compact records under the external run root:

```bash
export DELIMIT3D_PROMPT_BUNDLE=/path/to/frozen_bundle/freeze
export DELIMIT3D_PROMPT_MODEL=public
export DELIMIT3D_PROMPT_OUTPUT="$DELIMIT3D_ARTIFACT_ROOT/scannetpp/public"
python scripts/evaluation/evaluate_scannetpp.py
```

Use the same bundle, scene manifest, prompts, and evaluator for `posttrained`
to keep the comparison paired. `scripts/evaluation/aggregate_scannetpp.py`
then aggregates the two compact JSONL outputs.

## Experiment records

[`manifests/legacy_evidence.json`](manifests/legacy_evidence.json) records the
initial extraction from CHORUS and the earlier checkpoint identities.
[`docs/migration_from_chorus.md`](docs/migration_from_chorus.md) is the historical
migration record; [CURRENT.md](CURRENT.md) takes precedence for current direction.
Small reviewed metric summaries and report source code are versioned. Large
outputs stay in the external run roots recorded in the experiment map.

Run CPU regression checks in an environment providing the project dependencies:

```bash
python -m pytest -q
```

GPU preflight/smoke checks require the external model source trees and prepared
datasets. A passing CPU suite does not establish full GPU training correctness.
