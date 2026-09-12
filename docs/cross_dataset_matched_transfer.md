# Matched S3DIS and KITTI-360 transfer evaluation

This follow-up evaluates the frozen public LitePT and frozen 256-update Delimit3D LitePT encoders with the two 5,000-update decoder checkpoints from the matched ScanNet40 transfer. It uses the official AGILE3D S3DIS and KITTI-360 validation files and metric accounting, but it does not reproduce the original AGILE3D MinkowskiEngine model or its end-to-end training.

The evaluation is zero-shot with respect to S3DIS and KITTI-360. Their labels define downstream targets and metrics only; they are never passed to Delimit3D adaptation or decoder training. The decoder checkpoints were trained on ScanNet40 and are linked by commit, episode schedule, decoder initialization, and encoder hashes in each frozen config.

Official AGILE3D assets are under /cluster/work/igp_psr/nedela/agile3d_data/S3DIS and /cluster/work/igp_psr/nedela/agile3d_data/KITTI360. The cross-dataset roots are separate:

- /cluster/work/igp_psr/nedela/delimit3d_s3dis_agile3d_matched_v1
- /cluster/work/igp_psr/nedela/delimit3d_kitti360_agile3d_matched_v1

The released PLYs contain XYZ, RGB and labels but no normals. Because the active LitePT checkpoint is the RGBN6 model, a sidecar generation step is required and is explicit in protocol.normal_policy. It uses Open3D KNN PCA normals (knn=30) and a deterministic outward-from-centroid orientation. This is a new cross-domain input asset, recorded with source and sidecar hashes; it must not be described as a literal original AGILE3D reproduction.

The official multi-object list uses full-scene PLYs and raw instance IDs. The official single-object list uses binary target crops. The crop index in single/object_ids.npy is zero-based, while the crop PLY target label is one; the frozen config records single_object_label_offset: 1. Single-object feature caches therefore use the exact official crop geometry rather than silently substituting a full-scene record.

## Preparation

Run all commands after SSH to Euler, from the committed follow-up worktree. The two normal-generation jobs are submitted through scripts/evaluation/cross_dataset_normals.sbatch with one explicit DATASET, OFFICIAL_ROOT and OUTPUT_ROOT each. After both reports exist and their list hashes match the audit, freeze and prepare each YAML config.

GPU cache and evaluation workers request one RTX 4090 each, run in tmux or Slurm, write PID and heartbeat files, and verify the exact approved GPU before loading an encoder. Caching and evaluation are separate from the current ScanNet40 and ScanNet++ roots. Aggregate only after both arms and both panels have reports.

The primary results are official-compatible multi-object CSVs and paired scene bootstrap intervals. Single-object CSVs and paired object bootstrap intervals are secondary. The aggregate includes IoU at 1/3/5/10/15 clicks, NoC at 50/65/80/85/90, requested-count strata, object-size and semantic strata, and descriptive object-level MO strata.

Visualization is descriptive only. visuals/closeups contains point-level two-arm closeups at 1/5/15 clicks/object. visuals/meshes contains common closeup geometry colored by TP/FP/FN/background for each arm and click budget, as ASCII PLY plus PNG. Meshes are generated after prediction and never affect the metrics.
