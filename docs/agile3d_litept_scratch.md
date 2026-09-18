# LitePT-S* RGB3 AGILE3D baseline

This path replaces the released AGILE3D Minkowski backbone with LitePT and
uses a single-level compatible decoder and an explicit quantization utility
backend. It uses
the official ScanNet40 dataset, RGB-only 5 cm preprocessing, one finest-scale
LitePT `dec0` feature map, the existing single-level AGILE3D-compatible
decoder, the released click protocol, and `EvaluatorMO`.

The default Euler request is one RTX 4090.  To use one RTX 3090, override the
GPU resource at submission time.  The configured data is the existing
official AGILE3D conversion under `/cluster/work/igp_psr/nedela/agile3d_data/ScanNet`;
the raw nested ScanNet download is not a compatible substitute because it has
no flat labeled PLY/list contract.

```bash
sbatch --gpus=rtx_3090:1 \
  scripts/training/agile3d_litept_sstar_rgb3.sbatch preflight
```

Run modes are:

```bash
scripts/training/agile3d_litept_sstar_rgb3.sbatch preflight
scripts/training/agile3d_litept_sstar_rgb3.sbatch smoke
scripts/training/agile3d_litept_sstar_rgb3.sbatch train
```

For a continuation, set `AGILE3D_LITEPT_RESUME` to the last checkpoint before
submitting `train`.  For a standalone validation, set
`AGILE3D_LITEPT_CHECKPOINT` and use `evaluate`.

The batch launcher requests 160 GB of node-local scratch and stages the exact
Python environment, source trees, labeled PLYs, and scene lists there before
training. Only outputs remain on shared storage. This avoids the Lustre read
stall observed in job 14528852. `AGILE3D_LOCAL_ROOT` can explicitly select a
previously staged directory on the allocated node; otherwise it uses a fresh
directory below `$TMPDIR`. A `runtime.ready` marker is written only after a
successful environment copy. The input config hash and training settings stay
unchanged; the manifest records the relocated input paths.

The runner reports the first batch and every tenth batch, including data-wait
and training time. A ten-minute watchdog emits Python stacks if batch progress
stops. Checkpoints use an atomic replacement so an interrupted write preserves
the previous complete checkpoint. Preserve the previous manifest and metrics
alongside the resume checkpoint when recovering an existing output directory.

`preflight` checks the external AGILE3D/LitePT/data roots and reports model
dimensions.  `smoke` performs one real CUDA forward/backward/optimizer step
and verifies that LitePT parameters change.  Only `train` starts the 1100
epoch run.  The job writes the resolved config, source hashes, checkpoints,
official validation CSV files, and `EvaluatorMO` metrics under the configured
external output root.

Euler's LitePT environment does not include the compiled MinkowskiEngine
extension.  The runner therefore uses a narrowly scoped NumPy compatibility
backend for the released dataset's `sparse_quantize` and
`batched_coordinates` utilities; the manifest records this as
`numpy_utility_compat_v1`.  The AGILE3D sparse-convolution backbone is not
used by this LitePT experiment.

The existing frozen ScanNet++ runner is intentionally not modified.

## Comparison boundary and current run

At the September 18 snapshot, recovery job `14561205` is running from the
preserved epoch-12 checkpoint, with epoch 36 completed. See [CURRENT.md](../CURRENT.md)
for the dated status. Training runs for 1,100 epochs and validates every 50.

This RGB3 model has a different input contract from the RGBN6 pretraining run.
It cannot by itself serve as an initialization-only control for that encoder.
Keep the older custom RGBN6 scratch attempt and the exact upstream AGILE3D
reference separately named. The NumPy utility preserves aligned maps within
this runner; equivalence to every native MinkowskiEngine representative/order
choice has not been established by the CPU contract tests.
