# Sonata short adaptation setup

This experiment is separate from the LitePT and AGILE3D evaluation families. It
adapts the public encoder-only Sonata checkpoint for 256 updates on the same
label-free Structured3D 2D pseudomask objective used by Delimit3D.

The encoder input remains Sonata's official nine-channel contract: centered XYZ,
RGB, and surface normal. The bridge uses the official 2 cm grid and composes
Sonata's raw-point inverse with its hierarchical pooling maps, exposing a dense
feature for every source point. The three source cells are `2d/g02`, `2d/g05`,
and `2d/g08`; the projection head is disposable and is not part of the exported
encoder claim.

External immutable inputs:

- Sonata source: `/cluster/work/igp_psr/nedela/sonata`, commit
  `18c09ff8d713494f78a8213792262b910977a65d`.
- Sonata checkpoint: `/cluster/work/igp_psr/nedela/sonata_weights/sonata.pth`,
  SHA-256 `c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`.
- Structured3D source manifests:
  `/cluster/work/igp_psr/nedela/partfield_contrastive_gt3_unsam3_scratch/structured3d_sources_raw_unsamv2_90val`.
- Training split:
  `/cluster/work/igp_psr/nedela/partfield_contrastive_gt3_unsam3_scratch/structured3d_splits/train.txt`,
  SHA-256 `c71f5fd48e0e58eab185e5138ede5a53dd23dc5cc510696fcbaabaa164a1a9b9`.

The tracked setup is in `configs/adaptation/sonata_rgbn6_256.example.yaml`,
`src/delimit3d/training/sonata_adaptation.py`, and
`scripts/adaptation/run_sonata_adaptation.py`. The external artifact root is
`/cluster/work/igp_psr/nedela/delimit3d_sonata_rgbn6_256_v1`.

On Euler, prepare the immutable record with:

```bash
cd /cluster/home/nedela/nedela/projects/delimit3d
export DELIMIT3D_STRUCTURED3D_SOURCE_ROOT=/cluster/work/igp_psr/nedela/partfield_contrastive_gt3_unsam3_scratch/structured3d_sources_raw_unsamv2_90val
export DELIMIT3D_STRUCTURED3D_TRAIN_SPLIT=/cluster/work/igp_psr/nedela/partfield_contrastive_gt3_unsam3_scratch/structured3d_splits/train.txt
export PYTHONPATH=$PWD/src:/cluster/work/igp_psr/nedela/sonata
/cluster/work/igp_psr/nedela/litept-env/bin/python \
  scripts/adaptation/run_sonata_adaptation.py \
  --config configs/adaptation/sonata_rgbn6_256.example.yaml --mode prepare
```

Before training, submit the one-scene CUDA contract check. It requests one
RTX 4090, 8 CPU cores, and 64 GiB of host RAM and performs no optimizer step:

```bash
sbatch --export=ALL scripts/adaptation/sonata_256.sbatch smoke
```

The smoke must produce `smoke.json` with finite dense features, a nonempty
raw-to-final token map, and an unchanged encoder state hash. Only then should
the 256-update run be submitted:

```bash
sbatch --export=ALL scripts/adaptation/sonata_256.sbatch train
```

The training job writes checkpoints at updates 64, 128, and 256 and keeps the
encoder trainable while optimizing AdamW (`1e-4`, weight decay `1e-4`, clip
`0.1`). Its primary decision rule is deliberately narrow: compare the 256-step
Sonata adaptation to the public Sonata checkpoint on the same raw-feature
geometry and the same downstream point-conditioned decoder protocol before
making any cross-encoder claim. The adaptation loss by itself is not a
transfer result.
