# Delimit3D

Delimit3D is a small research package for adapting pretrained 3D scene
encoders with category-agnostic, multigranular 2D pseudomasks. The intended
first endpoint is point-conditioned selection of an object or part in a real
3D scan. The adaptation uses RGBN6 inputs and the existing LitePT-S* backbone;
2D teacher outputs are used while building the synthetic source packs and are
not required at downstream inference.

The repository is deliberately separate from the historical CHORUS checkout.
It contains the tested adaptation/data contracts, strict LitePT checkpoint
loading, source preparation entry points, frozen point-prompt metrics, and
focused regression tests. Data, weights, UnSAMv2, LitePT, logs, plots, and
Slurm outputs remain in external Euler roots.

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

Create an exact random initialization or a public-LitePT initialization:

```bash
delimit3d-init-scratch \\
  --litept-root "$DELIMIT3D_LITEPT_ROOT" \\
  --output "$DELIMIT3D_ARTIFACT_ROOT/initialization/scratch.pt" \\
  --seed 42 --input-features rgbn6 --multiscale-supervision

delimit3d-init-public \\
  --litept-root "$DELIMIT3D_LITEPT_ROOT" \\
  --scratch-initial "$DELIMIT3D_ARTIFACT_ROOT/initialization/scratch.pt" \\
  --public-checkpoint /path/to/public_litept.pth \\
  --expected-public-sha256 <sha256> \\
  --output "$DELIMIT3D_ARTIFACT_ROOT/initialization/public.pt"
```

Run the existing distributed adaptation contract with a clean, immutable
source/config/checkpoint manifest:

```bash
python -m delimit3d.training.runner \\
  --dataset structured3d \\
  --source-family 2d \\
  --source-root /path/to/structured3d/source_manifests \\
  --train-split /path/to/structured3d/train.txt \\
  --holdout-split /path/to/structured3d/holdout.txt \\
  --initial-checkpoint "$DELIMIT3D_ARTIFACT_ROOT/initialization/public.pt" \\
  --litept-root "$DELIMIT3D_LITEPT_ROOT" \\
  --output-dir "$DELIMIT3D_ARTIFACT_ROOT/runs/public_256" \\
  --input-features rgbn6 --backbone-lr 0.001 --head-lr 0.003
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

## Scientific boundary

The primary claim is representation-level: a short 2D-region adaptation can
make a pretrained 3D feature field more selective for a point-conditioned
object/part readout. The frozen readout metrics are kept separate from learned
Mask3D or PointGroup transfer metrics. The latter remain historical diagnostics
and are not the default success criterion for this repository.

The next learned experiment is deliberately small: attach the same fresh,
class-agnostic point-conditioned decoder to frozen public and adapted features,
then report one-click and correction-click metrics on a fixed ScanNet++ split.
Only after that reproduces the frozen readout should low-label, part, extra
backbone, or extra dataset experiments be added.

## Provenance

[`manifests/legacy_evidence.json`](manifests/legacy_evidence.json) records the
old checkout commit, the two current LitePT checkpoint hashes, the paired
ScanNet++ report, and the external artifact roots that remain authoritative.
[`docs/migration_from_chorus.md`](docs/migration_from_chorus.md) explains what
was extracted and what intentionally remains in the legacy archive.

No file in the old `chorus` checkout is deleted or reset by this repository.
