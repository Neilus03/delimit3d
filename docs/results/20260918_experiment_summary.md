# Experiment results — 18 September 2026

Read from external artifacts at `2026-09-18T20:47:07.374286+00:00`. The
[evidence snapshot](evidence_20260918.json) stores selected full-precision
metrics, bootstrap summaries, input paths and SHA-256 hashes. This is a
documentation refresh, not a rerun of training or evaluation.

## Completed frozen-encoder comparisons

Arms: **LitePT (public)** and **LitePT + Delimit3D (256 updates)**.
Each frozen encoder uses its corresponding separately trained decoder, starting
from the matched fresh decoder initialization. MO = multi-object; SO =
single-object. IoU values below are percentages; NoC is clicks (lower is better).

| Dataset / endpoint | Count | IoU@1 public → adapted | IoU@5 public → adapted | IoU@15 public → adapted | NoC@80 public → adapted |
|---|---:|---:|---:|---:|---:|
| S3DIS MO | 53 | 20.33 → 23.00 | 45.82 → 52.38 | 61.63 → 68.12 | 19.13 → 17.50 |
| S3DIS SO | 2,330 | 15.31 → 20.64 | 37.82 → 45.93 | 58.02 → 64.93 | 16.79 → 15.16 |
| ScanNet40 MO | 312 | 34.62 → 35.73 | 59.19 → 64.16 | 66.97 → 72.99 | 15.68 → 14.11 |
| ScanNet40 SO | 10,357 | 24.14 → 26.08 | 42.99 → 49.16 | 55.92 → 63.85 | 15.17 → 13.86 |

Coverage was rechecked in both arm reports: 312/10,357 ScanNet40 MO/SO
units and 53/2,330 S3DIS MO/SO units, with identical identity sets within each
pair. These are matched LitePT experiments under the recorded AGILE3D metric
protocol, not literal reproductions of the original model/full training recipe.
Earlier plots and LaTeX sources retain their September 14 date.

## Completed joint encoder/decoder control

Both arms train encoder and decoder for 5,000 updates. `public_control_A` is
an internal label for the public-initialized arm, not an official model name.
Compare it with adapted initialization under the same joint-training recipe.
Both reports mark MO and SO complete. These are short downstream training
runs, not the new 1,100-epoch scratch baselines.

| Endpoint | Count per arm | IoU@1 public → adapted | IoU@5 public → adapted | IoU@15 public → adapted | NoC@80 public → adapted |
|---|---:|---:|---:|---:|---:|
| MO | 312 | 50.40 → 53.23 | 69.04 → 70.75 | 74.09 → 74.89 | 12.67 → 12.11 |
| SO | 10,357 | 38.14 → 45.79 | 57.10 → 63.17 | 65.53 → 70.57 | 12.41 → 11.13 |

The table reports descriptive aggregate means; it does not establish
statistical significance for the joint-control difference. This update does
not recompute a paired confidence interval for that comparison.

## Completed Articulate3D interactive parts

Both arms completed 1,958 parts across 42 scenes. The primary **≥10 tokens**
subset contains 1,663 parts; sensitivity subsets contain 1,958 (≥5) and 1,343
(≥20). Thresholds are minimum surviving unambiguous feature-token counts per
part, not retrieval depths. Decoders come from the matched frozen ScanNet40
experiment and are paired with their corresponding encoders.

| Primary raw-point metric | LitePT (public) | LitePT + Delimit3D (256 updates) |
|---|---:|---:|
| IoU@1 (%) | 12.47 | 12.50 |
| IoU@5 (%) | 22.61 | 27.41 |
| IoU@10 (%) | 31.62 | 39.27 |
| IoU@20 (%) | 43.58 | 54.05 |
| NoC@50 | 14.02 | 12.50 |
| NoC@80 | 18.17 | 17.33 |

Primary IoU is measured on raw points after inverse-map expansion with
ambiguous tokens excluded. One-click performance is nearly unchanged; gains
appear with correction clicks. The table uses part-weighted means. Stored
paired bootstrap intervals use scenes, so their point estimates need not equal
the difference of these part-weighted means. Feature-only part retrieval
results are a separate endpoint.

## Supporting feature evidence

The completed ScanNet diagnostic covers 312 scans, 10,323 eligible instances
and 71,694 ordered same-category instance pairs. Scene-balanced retrieval AP
rises from 49.17% to 56.77%; target-size non-target leakage falls from 53.00%
to 46.25%; the hardest same-class margin rises from 0.08679 to 0.12612.
Within-instance coherence slightly decreases (0.79463 → 0.78596). The
pair-balanced same-category margin rises from 0.23256 to 0.30331. These
diagnostics support selectivity; they do not prove a semantic-grouping mechanism.

## New scratch/pretraining family — interim only

These results concern randomly initialized RGBN6 LitePT with the newer 5 cm
recipe. They are not rows to append to the earlier 2 cm public-adaptation table.

- Mask-only Mask3D epoch 250: raw CA-AP **25.58%**, CA-AP50 **32.11%**,
  CA-AP25 **36.02%**. This is class-agnostic validation during a 600-epoch
  run, not the canonical class-aware ScanNet leaderboard metric.
- Pretraining epoch 0 → 50 feature panel: ScanNet retrieval AP
  **10.86% → 64.71%**, leakage **87.46% → 39.54%**; Articulate3D ≥10-token
  feature mAP **42.31% → 86.81%**. The epoch-50 checkpoint hash is recorded
  in the snapshot. These are decoder-free panels, not pretrained Mask3D or
  AGILE3D transfer results.
- The RGB3 AGILE3D scratch arm has a different channel contract from this
  RGBN6 pretraining arm. A matched downstream comparison remains outstanding.

Run progress and unresolved decisions are in [CURRENT.md](../../CURRENT.md).
The exact experiment roots and source branches are in the
[experiment map](../EXPERIMENTS.md).

## Rebuilding this evidence

The standard-library extractor `scripts/reporting/snapshot_experiment_results.py`
reads the source files and hashes the same bytes it parses. It rejects incomplete
frozen identity coverage or missing joint/part counts. Large per-scene/object
reports remain external. The snapshot captures already computed metrics; it
does not independently re-evaluate checkpoints or prove every historical
training-match condition.
