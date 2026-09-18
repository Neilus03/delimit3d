# Current project state

**Snapshot: 18 September 2026, approximately 22:45 CEST.** Scheduler state and
training counters are time-specific. Recheck artifacts before quoting progress.

## Priority

Finish a controlled scratch versus PartField-pretrained LitePT comparison on
class-agnostic ordinary and interactive instance segmentation. The
[research direction](docs/RESEARCH.md) records the meeting decisions; the
[experiment map](docs/EXPERIMENTS.md) routes to the actual implementations.
The new experiment family uses 5 cm throughout.

## Active work

| Run | Verified snapshot | Evidence and limit |
|---|---|---|
| Structured3D RGBN6 LitePT pretraining | Job `14455351` RUNNING on `eu-g6-070`; update 15,540, epoch index 63 | Budget 82,040 updates / 336 epochs. Epoch-50 feature panel complete; downstream transfer still untested. |
| LitePT + mask-only Mask3D scratch | Job `14371975` RUNNING on `eu-a65-07`; latest metric epoch 282 | Budget 600 epochs. Epoch-250 raw CA-AP / AP50 / AP25 = 25.58 / 32.11 / 36.02%; interim validation. |
| LitePT RGB3 + compatible AGILE3D decoder | Job `14561205` RUNNING on `eu-g6-059`; completed epoch 36, 8,640 updates; epoch 37 progressing | Budget 1,100 epochs. Recovery uses node-local runtime/data and atomic checkpoints. No held-out milestone yet in this snapshot. |
| Exact released AGILE3D reference | Live `main.py` process on `pf-pc69`, GPU 1 active | Separate upstream backbone and runtime; not the LitePT baseline. No final evaluation claim. |

Scheduler/accounting and current metric/log files were inspected. The remote
reference also had a live training process and GPU allocation. These checks
establish execution, not scientific completion or comprehensive resource health.

The earlier custom RGBN6 AGILE3D scratch attempt is a distinct experiment. Its
18 September recovery directory contains diagnostic FP32 probes; a finite
eight-batch probe does not constitute a completed or resumed full baseline.
No `run_scannet40_scratch` process was found on pf at this snapshot.

## Completed earlier evidence

- Frozen public versus 256-update adapted LitePT: ScanNet40 (312 MO / 10,357
  SO), S3DIS (53 MO / 2,330 SO), and Articulate3D (1,958 parts / 42 scenes;
  primary ≥10-token subset: 1,663 parts).
- Joint encoder/decoder ScanNet40: both `public_control_A` and adapted
  initialization completed 5,000 training updates and full MO/SO evaluation.
- ScanNet feature selectivity: 312 scans, 10,323 eligible instances and
  71,694 ordered same-category pairs.

See [verified values and limitations](docs/results/20260918_experiment_summary.md)
and [the compact hashed evidence](docs/results/evidence_20260918.json). These
supersede earlier partial evaluation counts, without changing old run records.

## Next decisions and checks

1. Complete the active budgets and declared evaluation milestones. Pretraining
   needs checkpointed continuation if it exceeds the existing allocation.
2. Resolve RGB3 versus RGBN6 before pairing AGILE3D scratch and pretrained arms.
   Pin the input/model contract, initial decoder, schedule, seed and evaluator.
3. Prepare the pretrained Mask3D arm against the same mask-only scratch recipe;
   compare held-out CA-AP, not training loss or feature selectivity alone.
4. Report fixed-checkpoint comparisons with uncertainty and separate ordinary,
   interactive and part endpoints. Keep exact upstream and adapted LitePT
   system comparisons clearly labeled.

This repository update launches no new experiment and changes no running job.
Historical plans and invalid initialization controls are retained with labels.
