# Historical scratch-baseline readiness audit — 16 September 2026

**Superseded planning record.** Read [CURRENT.md](../CURRENT.md) and the
[experiment map](../docs/EXPERIMENTS.md) for the September 18 state. Baselines
have since launched. The original assessment below proposed binary objectness,
2 cm geometry and work still to implement; subsequent decisions selected
**mask-only (no class/objectness head), 5 cm**, and a separate RGB3 LitePT
AGILE3D implementation. The old “not running” and permission/readiness wording
below is a dated audit, not a current restriction or status. Do not execute
its superseded configuration proposal unchanged.

---

> **Resolution rule for the full experiment:** scratch baselines, subsequent pretraining, pretrained downstream fine-tuning, and evaluation all use **5 cm (0.05 m)**. Pretraining starts after both baselines have launched; it need not wait for their completion. Future launch checks must verify resolved voxel size and the resolution of any cached features/geometry. Authoritative record: `/cluster/work/igp_psr/nedela/scratch_baselines_20260916/resolution_contract.json`.

> **Latest correction:** Both baselines now use **5 cm** at the user's request. AGILE3D is moving to **one RTX 4090 on pf-pc69**, tmux `agile3d-scratch-5cm-20260916`; Euler AGILE3D job 14367498 was cancelled. The 20-batch 5 cm benchmark averaged 1.383 s/update, projecting 4.23 training days plus validation/checkpoints. Mask3D is strictly **mask-only**: no class/objectness prediction head, no classification or semantic CE, mask-only Hungarian matching, BCE+Dice including auxiliary outputs, proposal ranking from mask confidence. Its new Euler job is **14371975**, gated on **14371895**. Prior job IDs and binary-head/2 cm descriptions below are historical. Live launch record: `/cluster/work/igp_psr/nedela/scratch_baselines_20260916/launch_record.json`.

> **Setup update, 16 September 2026:** Both scratch baselines have now been implemented and submitted. AGILE3D job **14367498** (1,100 epochs) and Mask3D job **14367499** (600 epochs) depend on GPU check job **14367190**, currently queued. These are submissions, not a claim that full training has begun. Fresh random encoders/decoders, batch five, full schedules, per-epoch resume checkpoints, periodic validation, and walltime continuation are configured. The original slower AGILE3D smoke completed two epochs and evaluation; the new check additionally verifies exact parallel click simulation, clipping, and Mask3D resume.
>
> Implementation and run instructions: `/cluster/home/nedela/nedela/projects/delimit3d-baselines/baselines/README.md`. Frozen source and launch record: `/cluster/work/igp_psr/nedela/scratch_baselines_20260916/`. The remaining text below records the earlier audit before implementation.

# Scratch baseline readiness — 16 September 2026

This is a read-only implementation and artifact audit, plus a proposed execution plan. No baseline or pretraining job was submitted and no training code was changed.

## Confirmed baseline contract after checking the papers

The user clarified that the full original training budgets must be matched. This supersedes the earlier proposal to select a shorter AGILE3D budget from a pilot learning curve.

- **Both baselines start with randomly initialized LitePT and randomly initialized decoders.** No public checkpoint, adapted checkpoint, frozen pretrained features or pretrained BatchNorm statistics are used. A later resume may restore only that same scratch run and its optimizer/scheduler/RNG state.
- **AGILE3D / ScanNet40: 1,100 complete epochs**, batch five, AdamW with LR 1e-4 and weight decay 1e-4; multiply LR by 0.1 after 1,000 epochs. With the current 1,200-scene manifest, this is 240 optimizer updates per epoch and **264,000 total optimizer updates**, covering 1,320,000 scene presentations.
- **Mask3D: 600 complete epochs**, batch five, AdamW and OneCycleLR with peak LR 1e-4. With the current 1,201-scene list and a retained final partial batch, this is 241 updates per epoch and **144,600 total updates**. The official trainer600 YAML actually sets `max_epochs: 601`; this paper/code discrepancy must be explicit. The new plan follows the user's requested paper budget of exactly 600 completed epochs, rather than silently inheriting 601.
- Small smoke checks establish implementation correctness and throughput only. They do not change either full baseline budget or determine an early stopping point.
- Preserve epoch shuffling, augmentations, interaction sampling, batch loss reduction and learning-rate progression. Gradient accumulation can match an effective batch size but must not be presented as automatically identical to a true batch, particularly for BatchNorm and loss normalization.

Primary sources checked on 16 September 2026:

1. [AGILE3D camera-ready paper, Appendix A.2](https://arxiv.org/html/2306.00977v4#A1.SS2): 1,100 ScanNet40 epochs, AdamW, weight decay 1e-4, LR 1e-4, decay after 1,000. The paper's separate ScanNet20 variant uses 850 epochs and is not the selected ScanNet40 baseline.
2. [Official AGILE3D main.py](https://github.com/ywyue/AGILE3D/blob/main/main.py) and [training script](https://github.com/ywyue/AGILE3D/blob/main/scripts/train_multi_scannet40.sh): batch five; fresh construction with an empty default resume path; epoch-based AdamW/MultiStepLR training.
3. [Mask3D paper, Section III-A](https://arxiv.org/pdf/2210.03105): 600 epochs, AdamW, peak LR 1e-4 and one-cycle schedule; approximately 78 hours for its original backbone on an A40, which is not a measured LitePT runtime.
4. [Official Mask3D indoor data configuration](https://github.com/JonasSchult/Mask3D/blob/main/conf/data/indoor.yaml): batch five; [trainer600 configuration](https://github.com/JonasSchult/Mask3D/blob/main/conf/trainer/trainer600.yaml): literal 601 setting; [base configuration](https://github.com/JonasSchult/Mask3D/blob/main/conf/config_base_instance_segmentation.yaml): checkpoint and backbone checkpoint null, backbone unfrozen; [scheduler](https://github.com/JonasSchult/Mask3D/blob/main/conf/scheduler/onecyclelr.yaml): per-step OneCycleLR.

The requested models replace the original sparse-convolution backbone with LitePT, and Mask3D becomes class-agnostic. These are explicit model/task adaptations. AGILE3D's paper uses 5 cm voxels while the existing LitePT path uses 2 cm and RGBN6; that input-resolution difference must be recorded and resolved explicitly when claiming broader recipe parity. This audit fixes the epoch/initialization contract, not every remaining implementation difference.

## Assessment

Neither requested baseline should be launched with the existing production configurations unchanged. Both have substantial reusable infrastructure.

| Baseline | Available | Still required |
|---|---|---|
| LitePT + class-agnostic Mask3D | Random backbone initialization; binary object/no-object decoder; set losses; class-agnostic AP; full-scene RGBN6 training infrastructure; data and environment archives | A new full-scene class-agnostic configuration and matching random decoder template; an adapted launcher; fresh small-scene/GPU verification; a resume plan beyond the five-day limit |
| LitePT + AGILE3D | Joint encoder/decoder training demonstrated; full ScanNet40 data; deterministic training episodes and decoder initialization; complete MO/SO evaluation demonstrated | A named random-initialization arm; seed-before-construction initialization or a verified random initialization checkpoint; 1,100-epoch shuffled training with batch five and LR decay at epoch 1,000; intermediate evaluation; scratch GPU verification |

Existing public-initialized and adapted-initialized AGILE3D runs are not scratch baselines. Existing full-scene class-aware Mask3D runs are not class-agnostic baselines.

## 1. Mask3D

Use the implementation under `/cluster/home/nedela/nedela/projects/chorus/student`.

Relevant existing files:

- `scripts/run_litept_mask3d.py`: supports `litept_mask3d_class_agnostic`, two output classes including no-object, scratch initialization, full-scene training, OneCycleLR, checkpoints and resume, and CA-AP evaluation.
- `student/models/litept_mask3d_initialization.py`: explicitly supports `backbone_source: scratch`, forbids importing official decoder weights in the dedicated runner, and checks decoder template identity.
- `student/metrics/litept_mask3d_ca_ap.py`: implements object probability times mask quality and class-agnostic AP. Object/no-object supervision must remain; removing all classification/objectness supervision would break the intended query-scoring contract.
- `configs/litept_mask3d_scratch_trainable.yaml`: already class-agnostic, but uses 60,000-point sphere crops, 400 epochs and batch two. It is an older recipe, not the recent full-scene recipe.
- `configs/litept_mask3d_scannet_rgbn6_scratch_fullfpn_bntrain_601_seed42.yaml`: recent full-scene RGBN6 scratch recipe, but class-aware with 19 outputs including no-object.
- `scripts/create_litept_mask3d_decoder_template.py`: can generate a fresh two-class decoder. Check and record query-count metadata for the new configuration.
- `scripts/run_litept_mask3d_scannet_mask3d_parity_fast_local.sbatch`: reusable local environment/data staging and resume infrastructure; its default decoder and configuration are class-aware.

Proposed new configuration: `configs/litept_mask3d_scannet_rgbn6_classagnostic_scratch_seed42.yaml`.

Build it from the recent full-scene scratch recipe, explicitly setting:

- RGBN6 LitePT-S*, 2 cm, representative-first voxelization, native full feature pyramid.
- Random encoder, fresh binary object/no-object decoder, both trainable from update one; no public weights and no backbone warmup freeze.
- Full scenes, existing normal-aware augmentation and fixed RGB normalization.
- 100 training and 100 evaluation queries, top 100 raw proposals, no DBSCAN as the primary metric. This is the query policy used by the documented successful class-agnostic memorization gate; it needs to be rechecked in the new recipe.
- Effective batch five scenes; AdamW peak LR 1e-4 for encoder and decoder, weight decay 1e-2, auxiliary losses, 600-epoch OneCycleLR schedule.
- Full validation every 50 epochs and at the final epoch; more frequent checkpoints for recovery. Fix and record the BatchNorm evaluation policy for both future initialization arms.
- Dedicated output root, configuration/code record, decoder hash and seed.

The historical report `student/reports/litept_mask3d_one_scene_overfit.md` documents a passed class-agnostic scratch gate on 13 August: CA-AP50 1.0 on one training scene at step 575. This is useful prior engineering evidence, not a fresh verification of the proposed configuration or a generalization result.

Data inspected: all 1,513 files named by the ScanNet20 GT cache manifest exist; the approximately 6.3 GB data archive, 27.3 GB environment archive and old binary decoder template exist. This was an existence/inventory check, not a full rehash of those archives.

## 2. AGILE3D

Use the joint-training checkout `/cluster/home/nedela/nedela/projects/delimit3d-scannet40-joint`, not the frozen-feature runner in the matched checkout or the ScanNet++ runner on main.

Ready and demonstrated:

- `scripts/evaluation/run_scannet40_joint.py` recomputes LitePT features with gradients, optimizes encoder and decoder, checks nonzero gradients, and saves both model states plus optimizer/scaler/RNG state.
- Existing joint training completed 5,000 updates for both adapted and public initialization on one RTX 4090.
- All 3,024 PLY/normal paths for 1,200 training and 312 validation scenes currently exist.
- `scripts/evaluation/run_scannet40_joint_eval.py` has completed the full 312-scene multi-object and 10,357-object single-object panels for an existing trained arm.

Required changes:

1. Add an explicit `scratch` initialization arm. The runner only recognizes `delimit3d` and `public_control_A`; loading a random checkpoint under the adapted arm's name would create a misleading experiment record.
2. Create and hash the random backbone before training. `src/delimit3d/cli/initialize_scratch.py` is reusable initialization machinery, but its output must be checked against this runner's exact LitePT contract. If constructing directly in the joint runner, seed **before** construction: currently `set_seed` is called after loading the backbone.
3. Keep a verified untrained decoder initialization shared with the future pretrained arm. Existing random decoder snapshots can be reused if their architecture and seed are the chosen contract; do not load a trained decoder.
4. Generalize the experiment/output/manifest checks, episode preparation, training loop and evaluation checks together. Training currently requires exactly 5,000 updates, copies the parent's fixed 5,000-row schedule, and evaluation requires the completed 5,000-update report.
5. Add validation of intermediate checkpoints on a fixed small panel. Full MO/SO evaluation is too expensive at every checkpoint. Generate deterministic shuffled full epochs through epoch 1,100; add the official LR drop at epoch 1,000 and save/restore scheduler state.
6. Implement true batch-five joint training (or explicitly qualify and validate any memory-saving alternative), training augmentation, and the upstream batch/interaction loss reduction. Audit model-dependent click sampling against the official engine.
7. Run a scratch gradient/finite-loss check, small-scene learning check and peak-memory/throughput measurement. Existing AMP success with pretrained features does not prove random initialization is stable.

The current runner uses batch one, fixed LR 1e-4, weight decay 1e-4, clipping 0.1, FP16 AMP and a fixed unaugmented scene geometry path. This is a custom LitePT adaptation of AGILE3D, not a reproduction of its full original training recipe.

The local upstream training script `/cluster/work/igp_psr/nedela/AGILE3D/scripts/train_multi_scannet40.sh` specifies 1,100 epochs; `main.py` defaults to batch five and LR decay at epoch 1,000. Its dataset has 1,200 training entries and includes geometric augmentation. Therefore the current 5,000-update single-scene recipe (about 4.17 scene passes) must not be described as a sufficient standard scratch-training budget. Upstream uses a different backbone, so its exact duration also cannot be copied onto LitePT.

## 3. Object universe and comparison contract

The immediately reusable automatic benchmark is class-agnostic ScanNet20 things, with 1,201 training and 312 validation scenes. AGILE3D uses its ScanNet40 instance labels, 1,200 training scenes and the same count of validation scenes. `scene0380_01` is the extra Mask3D training scene.

Recommended initial scope: retain each framework's existing benchmark definition and match scratch versus pretrained **within each framework**. Report the label-universe difference. If the specific claim is about interactive versus automatic performance on exactly the same objects, add a shared-object evaluation or extend Mask3D to the AGILE3D labels first; simply changing the decoder to binary does not align the datasets.

For each future pair, fix the downstream seed, random decoder snapshot, training scenes/order, augmentation policy, optimizer/schedule, update budget, input contract, checkpoint-selection rule and evaluation settings. Vary encoder initialization only. The two different frameworks do not need the same optimizer or decoder.

## 4. Full budgets and runtime limits

The full budgets are now fixed at 600 Mask3D epochs and 1,100 AGILE3D epochs. Times exclude queues and preparation.

| Work | Fixed budget | Time estimate |
|---|---:|---|
| Mask3D baseline | 600 epochs / 144,600 updates with the current list and batch five | About 129.6 training hours; allow roughly 5.5–6 days including validation/staging on one A100 80 GB |
| AGILE3D baseline | 1,100 epochs / 264,000 updates with the current list and batch five | Not yet measured for the required batch-five LitePT runner; benchmark representative full batches before quoting a wall time |
| AGILE3D full final MO/SO evaluation | 312 scenes / 10,357 single-object episodes | Existing adapted model evaluation took 25h24m35s on one RTX 4090; new model duration may differ |

Mask3D timing is extrapolated from the previous full-scene class-aware LitePT scratch run: mean 777.47 seconds per epoch across 553 distinct completed epochs. The new class-agnostic run itself has not been timed.

The previous AGILE3D estimate of six hours for 5,000 batch-one updates describes the old short-run implementation. Multiplying it by 264,000/5,000 would undercount scene work by a factor of five. Multiplying by 1,320,000/5,000 instead estimates the old serial execution cost, not the runtime of the required batched/augmented runner. Neither the earlier 60-hour estimate nor the serial two-month extrapolation is a measured duration for the new baseline.

Mask3D requires planned resume across the five-day scheduler limit. Previous job 12500337 timed out after five days, with logs through epoch 553 and a saved epoch-550 checkpoint. Keep the complete 600-epoch OneCycle schedule from the first update. Checkpoints and inspections must not change the scheduler horizon. AGILE3D likewise requires restartable epoch/sampler/optimizer/scheduler/RNG state through the full 1,100 epochs.

## 5. Execution sequence

1. Freeze random initialization, 600/1,100 epoch budgets, batch-five semantics, data lists, optimizer schedules and task definitions.
2. Prepare the full-scene class-agnostic Mask3D configuration/launcher and implement the missing epoch-based, batch-five AGILE3D scratch path.
3. Run small separate GPU checks: finite gradients in both networks, small-scene learning, correct evaluation, loss reduction and exact resume continuity. Measure throughput on representative scenes with the intended batch/voxel settings.
4. Start the full scratch baselines after those checks: Mask3D for 600 epochs and AGILE3D for 1,100 epochs. Validate at fixed milestones; use the complete schedules regardless of whether early improvements are fast or slow.
5. Evaluate final and declared best-validation checkpoints. Later pretrained comparisons use the same complete downstream schedules and fresh matching decoder initialization.

## 6. Verification limits and source artifacts

- Live repository/configuration review, scheduler/accounting checks, data-file existence checks and existing completion reports were inspected.
- Six relevant Python files passed AST syntax parsing; the two reusable launchers passed shell syntax checking.
- A focused CPU test attempt timed out during shared-environment startup/imports; it did not reach a test result. No fresh GPU verification or full baseline run was performed.
- Current queued/running jobs are other evaluations; neither new scratch baseline is running.

Runtime artifacts:

- `/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_control_A/public_control_A/train_report.json`
- `/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_retry3/delimit3d/train_report.json`
- `/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_eval3/evaluation_report.json`
- `/cluster/work/igp_psr/nedela/student_runs/litept_mask3d_rgbn6_scratch_fullfpn_bntrain_601_seed42/litept_mask3d_scannet_rgbn6_scratch_fullfpn_bntrain_601_seed42/metrics.jsonl`
- `/cluster/work/igp_psr/nedela/student_runs/litept_mask3d_rgbn6_scratch_fullfpn_bntrain_601_seed42/litept_mask3d_scannet_rgbn6_scratch_fullfpn_bntrain_601_seed42/initialization_report.json`
