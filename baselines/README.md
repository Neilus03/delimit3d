# LitePT baseline and pretraining experiment

## Resolution shared by every stage

**Use 5 cm (0.05 m) throughout:** both scratch baselines, subsequent pretraining, pretrained downstream fine-tuning, and downstream evaluation. This is an explicit user requirement from 16 September 2026. Pretraining follows the launch of both baselines; their completion is not a prerequisite.

Before future launches, check the resolved voxel/grid size and the resolution recorded in cached geometry/features. Do not silently reuse 2 cm caches. Hold downstream preprocessing, random decoder initialization, seed, optimizer, schedule, and evaluation fixed within each scratch-versus-pretrained comparison.

Machine-readable record: `/cluster/work/igp_psr/nedela/scratch_baselines_20260916/resolution_contract.json`.

## Baselines

Both use LitePT-S-star, RGBN6 inputs, random encoder and decoder initialization, trainable encoders, and seed 42. No public or adapted checkpoint initializes either baseline.

| Baseline | Training scenes | Epochs | Batch | Optimizer / schedule | Location |
|---|---:|---:|---:|---|---|
| Mask-only Mask3D | 1,201 | 600 | 5, final batch 1 | AdamW, WD .01, OneCycle peak 1e-4; 144,600 updates | Euler, one A100 80 GB |
| AGILE3D ScanNet40 | 1,200 | 1,100 | 5 | AdamW, WD .0001, LR .0001; decay .1 after epoch 1,000; gradient clipping .1; 264,000 updates | pf-pc69.ethz.ch, GPU 0 only, one RTX 4090 |

Mask3D has **no class/objectness prediction head, no class logits, no classification or semantic CE, and no classification term in Hungarian matching**. It trains with mask BCE and Dice, including auxiliary layers. Proposals rank by mask confidence only. Its legacy `num_classes=2` field is unused compatibility metadata. Template: `mask3d_maskonly_decoder_random_seed42.pt`. Tests check absent classification parameters/logits/CE, semantic-label invariance, mask gradients and proposal scoring; GPU checks additionally exercise training, evaluation and resume.

AGILE3D encodes true batches of five, averages per-scene losses, shuffles every epoch, augments coordinates and normals, and samples shared 0–19-round interaction prefixes. Its Dice matches the official implementation; parallel exact KD-tree click selection matches the sequential reference. Validation never silently removes official objects. AGILE3D uses running BatchNorm statistics during validation; Mask3D uses train-no-update BatchNorm. Preserve these policies in subsequent paired comparisons.

Paper schedules: https://arxiv.org/pdf/2210.03105 and https://arxiv.org/html/2306.00977v4#A1.SS2 . These are LitePT adaptations. Original Mask3D uses 2 cm; this experiment explicitly uses the user's requested 5 cm everywhere.

## Execution and recovery

Live launch record: `/cluster/work/igp_psr/nedela/scratch_baselines_20260916/launch_record.json`. It owns current job IDs and locations. Frozen Euler source: `.../source/`; pf-pc69 run root: `/home/nedela/scratch_baselines_20260916/`. AGILE3D tmux session: `agile3d-scratch-5cm-20260916`.

Checkpoint `latest.pt` saves each completed epoch with optimizer/scheduler state; AGILE3D also saves scaler/RNG. Both seed each epoch independently. Validation occurs every 50 epochs and at the end: Mask3D CA-AP and AGILE3D ScanNet40 MO. Final SO evaluation is separate. AGILE3D periodic snapshots contain evaluation weights only; resume from `latest.pt`. Euler allocations continue after clean 108-hour boundaries; pf-pc69 resumes those boundaries immediately. Errors stop execution. Smoke and benchmark weights never initialize full training.

pf-pc69 has a roughly 6 GB home quota and full local scratch. Its 3,024 PLY/normal files are SHA256-verified copies in `/dev/shm/nedela_agile3d_20260916/data`, reached through the run's `data` symlink. This dataset storage is ephemeral and must be re-staged after a reboot. Durable checkpoints remain in the home run directory. Existing unrelated files were not deleted.

## Initial timing evidence

At 5 cm on one 4090, 20 benchmark batches averaged 1.383 seconds/update; the first 51 full-training updates averaged 1.224 seconds/update. This projects roughly four training days plus validation/checkpointing, consistent with the author's 4–5-day original-model runtime reported by the user. These are early estimates, not a completed-run duration. The previous 2 cm Mask3D timing must not be treated as the measured ETA of the new 5 cm mask-only run.
