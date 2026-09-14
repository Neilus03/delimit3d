# Joint Delimit3D LitePT + AGILE3D ScanNet40 training

This run keeps the completed `experiment/scannet40-agile3d-matched-v1`
ScanNet40 protocol fixed and changes one factor: the Delimit3D-trained LitePT
encoder is trainable together with the fresh AGILE3D decoder. The public arm
is not launched.

The run uses all 1,200 official training scenes, the exact parent 5,000-update
episode file, the same 72-to-128 decoder architecture and byte-identical
decoder initialization, RGBN6/2 cm/representative-first/dec0 features, the
same AGILE3D click-prefix and correction policies, AdamW weight decay `1e-4`,
gradient clipping `0.1`, and one RTX 4090. Encoder and decoder optimizer
groups both use learning rate `1e-4`. AMP float16 is enabled to fit joint
LitePT backpropagation on 24 GB; the precision is recorded in every
checkpoint.

Preparation is separate from the parent artifact root:

```bash
cd /cluster/home/nedela/nedela/projects/delimit3d-scannet40-joint
PY=/cluster/work/igp_psr/nedela/litept-env/bin/python
CFG=configs/evaluation/scannet40_agile3d_joint_v1.yaml
$PY scripts/evaluation/run_scannet40_joint.py --config "$CFG" --mode freeze
$PY scripts/evaluation/run_scannet40_joint.py \
  --config /cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_retry3/freeze/resolved_config.yaml \
  --mode prepare
```

The freeze step refuses a dirty worktree and pins the source archive, copied
LitePT dependency, Delimit3D checkpoint, parent schedule, decoder
initialization and hashes. The first training worker performs one gradient
check, requires nonzero encoder and decoder gradients, restores the initial
states/RNG, recreates an empty optimizer, and then starts update 1. Joint
checkpoints are atomically written at updates 250, 500, ..., 5,000 and carry
both model states, optimizer/scaler state and all RNG state.

Submit only the one requested arm after the freeze/prepare preflight:

```bash
ROOT=/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_retry3
PROV="$ROOT/freeze/provenance.json"
export JOINT_CONFIG="$ROOT/freeze/resolved_config.yaml"
export JOINT_SOURCE_ARCHIVE="$ROOT/freeze/$(basename "$(jq -r .source_archive "$PROV")")"
export JOINT_SOURCE_SHA256="$(jq -r .source_archive_sha256 "$PROV")"
export JOINT_PYTHON=/cluster/work/igp_psr/nedela/litept-env/bin/python
sbatch --test-only scripts/evaluation/scannet40_joint.sbatch
sbatch scripts/evaluation/scannet40_joint.sbatch
```

The joint AMP path keeps model execution in float16 but computes the
cross-entropy/Dice loss in float32. It begins with a unit loss scale and a
long growth interval because the first encoder+decoder backward pass can
overflow at the default GradScaler scale on a 24 GB RTX 4090; this is
recorded in the frozen precision configuration.

The training report and final checkpoint are under
`$ROOT/delimit3d/`. A nonzero encoder gradient and a changed encoder
parameter hash are required evidence that this is joint training; scalar
loss alone is not a transfer result.
