## Final requested setup

Both baselines use **5 cm voxels**, as explicitly requested on 16 September. AGILE3D runs on GPU 0 only on pf-pc69.ethz.ch (one RTX 4090), for all 1,100 epochs. Mask3D remains on Euler, one A100 80 GB, 600 epochs, strict mask-only training. The earlier 2 cm notes below are historical.

The pf-pc69 20-batch benchmark at 5 cm averaged 1.3829 seconds/update (1.2696 excluding the first batch): approximately 4.23 training days for 264,000 updates, before validation/checkpoints. This short-sample estimate is consistent with the author's 4–5-day original-model runtime reported by the user, but is not a final ETA. Full training must start from fresh random weights; benchmark/smoke weights are never used.

pf-pc69 home has a roughly 6 GB user quota, and its scratch disk is full. Dataset copies live in `/dev/shm/nedela_agile3d_20260916/data`, accessed through the run's `data` symlink. All 3,024 PLY/normal files are SHA256-checked against Euler records before training. Dataset RAM storage is ephemeral and must be re-staged after a reboot; durable checkpoints stay under `/home/nedela/scratch_baselines_20260916/agile3d`. `latest.pt` includes optimizer/scheduler/scaler/RNG and supports resume. Every-50-epoch snapshots contain evaluation weights only, limiting home-space use. The tmux launcher resumes clean 108-hour boundaries immediately; failures stop. Euler's original AGILE3D job was cancelled.

## Routing and mask-only correction

AGILE3D is moving to one RTX 4090 on pf-pc69.ethz.ch; its Euler full job was cancelled. Mask3D stays on Euler. The requested Mask3D contract is now strictly mask-only: no class/objectness head, no class logits, no classification or semantic cross-entropy, no classification term in Hungarian matching. Training uses mask BCE and Dice (including auxiliary layers); proposals rank by mask confidence alone. The retained num_classes=2 field is unused compatibility metadata, not a prediction head. The old binary object/no-object template is not used. New template: mask3d_maskonly_decoder_random_seed42.pt. Mask-only checks and successful resume gate the new 600-epoch job. The earlier text below describes the initial setup and is superseded on these points.

# Scratch baseline training, 16 September 2026

Two independent, fully trainable LitePT-S-star RGBN6 baselines, seed 42. Both encoders and decoders start randomly; no public or Delimit3D weights are loaded. Their outputs and smoke-test outputs are separate.

| Baseline | Training scenes | Epochs | Batch | Optimizer / schedule |
|---|---:|---:|---:|---|
| Class-agnostic Mask3D | 1,201 | 600 | 5, final batch 1 | AdamW, WD .01, OneCycle peak 1e-4, 144,600 updates |
| AGILE3D ScanNet40 | 1,200 | 1,100 | 5 | AdamW, WD .0001, LR .0001; multiply by .1 after epoch 1,000; gradient clipping .1; 264,000 updates |

Paper sources: https://arxiv.org/pdf/2210.03105 and https://arxiv.org/html/2306.00977v4#A1.SS2 . The 1,100-epoch AGILE3D schedule is ScanNet40, not the separate ScanNet20 schedule.

These are LitePT adaptations, not literal reproductions of the original sparse backbones. Inputs use 2 cm RGBN6 for both; original AGILE3D uses 5 cm RGB. Mask3D has a binary object/no-object head. AGILE3D's new runner uses official per-token Dice reduction; the older joint runner is unchanged. Training objects are sampled from those represented after augmentation/voxelization, as in the original quantized training. Validation requires every official object and never silently removes one. AGILE3D uses running-statistics BatchNorm at validation, Mask3D uses its existing train-no-update evaluation policy. Keep these policies fixed for subsequent pretrained comparisons.

AGILE3D shuffles all scenes each epoch, applies XY flips and Z rotations to coordinates and normals, jointly encodes all five scenes, averages their losses, and takes one optimizer step. Exact KD-tree centers are computed across eight CPU threads, with a reference-parity test. It draws a shared 0–19-round simulated interaction prefix per batch. Mask3D retains the full-scene augmentation profile and true batch-five runner.

Checkpoints contain model, optimizer and scheduler state; AGILE3D also saves AMP scaler and RNG. Both independently seed each epoch, including Mask3D's newly started loader workers, making job boundaries reproducible. Atomic `latest.pt` is saved after each epoch and validation. Checkpoints are retained every 50 epochs. Full validation runs every 50 epochs and at the end: Mask3D CA-AP and AGILE3D ScanNet40 MO. Final ScanNet40 SO remains a separate evaluation, not a training prerequisite.

The GPU preflight checks official Dice values and gradients, scratch training with batches of five, encoder/decoder changes, held-out evaluation, and restoring both jobs for a second epoch. It does not establish convergence. Full jobs require both successful Slurm dependency and a passed check marker. They run on one A100 80 GB each with 128 GB CPU RAM and 160 GB local disk. At a clean epoch boundary after 108 hours, each saves and schedules its own continuation under the unchanged schedule. Errors stop rather than resubmitting. No preflight weights are used by full training.

Frozen source is under `/cluster/work/igp_psr/nedela/scratch_baselines_20260916/source/`; launchers check file SHA256 before executing. Full run roots are `.../agile3d/` and `.../mask3d/litept_mask3d_ca_scratch_600_seed42/`. Inspect `status.json`, `updates.jsonl` (AGILE3D), `metrics.jsonl` (Mask3D), and `logs/`.

Time estimates must be updated from full-batch measurements. Previous Mask3D evidence suggests roughly 5.5–6 GPU days plus queueing; AGILE3D's old 5,000-update batch-one run is not a reliable estimate for this batch-five schedule. Both full budgets are fixed regardless of smoke-test duration.
