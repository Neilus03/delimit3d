# Experiment map

Updated 18 September 2026. Euler paths below use
`/cluster/work/igp_psr/nedela/` as the external artifact base. Data and weights
are private/external prerequisites; links to source branches are not downloadable
dataset or checkpoint releases. [CURRENT.md](../CURRENT.md) carries run status.

## Current scratch/pretraining family

| Experiment | Source | Artifact root / contract |
|---|---|---|
| LitePT RGB3 AGILE3D scratch | [runner](../scripts/training/run_agile3d_litept_sstar_rgb3.py), [config](../configs/training/agile3d_litept_sstar_rgb3_scannet40.yaml), [runbook](agile3d_litept_scratch.md) | `delimit3d_agile3d_litept_sstar_rgb3_scratch_v1`; RGB3, 5 cm, dec0, batch five, 1,100 epochs |
| Mask-only Mask3D scratch | [baseline branch at 9ed37df](https://github.com/Neilus03/delimit3d/tree/9ed37df/baselines); external CHORUS `student` source snapshot | `scratch_baselines_20260916/mask3d/litept_mask3d_ca_scratch_600_seed42`; RGBN6, 5 cm, 600 epochs |
| Structured3D LitePT pretraining | External immutable `code.tar`, `code_manifest.json`, `recipe.json`, `recipe_parity.json`, `production_command_14455351_1789652341.json` | `structured3d_litept_hierarchy_5cm_336ep_20260917_v1`; random initialization, RGBN6, sphere-50k, 2 GPUs, 2,930 scenes, 336 epochs / 82,040 updates |
| Earlier custom RGBN6 AGILE3D scratch | [scratch runner at 9ed37df](https://github.com/Neilus03/delimit3d/blob/9ed37df/scripts/evaluation/run_scannet40_scratch.py) | pf `/home/nedela/scratch_baselines_20260916`; distinct from main's RGB3 substitution; recovery probes under `recovery_20260918` |
| Exact upstream AGILE3D reference | External AGILE3D commit `52f7b2eeaa8d7e8ed8a57ebd9707f459977e3586` | pf `/scratch2/nedela/agile3d_exact_20260917`; `source`, `runtime`, `output_py310`, `eval_fullval_output` |

The shared resolution decision is stored at
`scratch_baselines_20260916/resolution_contract.json`. The original
`launch_record.json` is historical; its queued/running descriptions are not
current status. Pretraining feature milestones live in `progress.json`;
optimization progress lives in `training/metrics.jsonl`.

The RGB3 runner uses an explicit prequantized LitePT path, preserving external
click/label token indices. Its compatible decoder and NumPy quantization
utility must be recorded as implementation differences from the upstream
system. A native-backend equivalence claim requires a separate audit.

## Completed short-adaptation comparisons

| Family | Source branch snapshot | External root |
|---|---|---|
| Frozen LitePT + learned decoder, ScanNet40 | [4a18882](https://github.com/Neilus03/delimit3d/tree/4a18882), `scripts/evaluation/run_scannet40_matched.py` | `delimit3d_scannet40_agile3d_matched_v1` |
| Frozen LitePT, S3DIS transfer | [2863277](https://github.com/Neilus03/delimit3d/tree/2863277), matched cross-dataset evaluator | `delimit3d_s3dis_agile3d_matched_v2` |
| Joint encoder/decoder, ScanNet40 | [ecc286f](https://github.com/Neilus03/delimit3d/tree/ecc286f), `scripts/evaluation/run_scannet40_joint.py` and `run_scannet40_joint_eval.py` | Train: `delimit3d_scannet40_agile3d_joint_v1_retry3`, `delimit3d_scannet40_agile3d_joint_v1_control_A`; evaluate: `delimit3d_scannet40_agile3d_joint_v1_eval3`, `delimit3d_scannet40_agile3d_joint_v1_control_A_eval1` |
| Articulate3D interactive parts | [runner](../scripts/evaluation/run_articulate3d_parts.py), [config](../configs/evaluation/articulate3d_agile3d_parts_v1.yaml) | `delimit3d_articulate3d_agile3d_parts_eval_v1`; features from `articulate3d_delimit3d_eval_v1`; decoder paired by arm from frozen ScanNet40 study |
| ScanNet feature selectivity | [diagnostic](../scripts/diagnostics/feature_selectivity_scannetval.py) | `delimit3d_feature_selectivity_scannetval_20260915` |

Branch snapshots are navigation pointers. The executed source/config and hashes
in each run's immutable record remain authoritative, including changes made
before their later Git commit. No branch is silently merged into another
experimental recipe during documentation cleanup.

## Secondary and historical paths

- Original ScanNet++ frozen readout: `scripts/evaluation/evaluate_scannetpp.py`.
  The custom ScanNet++ multi-object runner is not the official ScanNet40 path.
- PTv3 matched adaptation: `scripts/adaptation/run_ptv3_matched_adaptation.py`,
  versioned `configs/adaptation/ptv3_*` recipes and
  [feature gate](ptv3_feature_geometry_gate.md). Local-disk v1/v2 retry configs
  preserve separate historical attempts; they are not newly validated results.
- Sonata: [setup](sonata_adaptation.md),
  `scripts/adaptation/run_sonata_matched_adaptation.py` and
  `configs/adaptation/sonata_rgbn6_256_matched.example.yaml`. Despite the config
  filename, the model input includes XYZ + RGB + normals (nine channels).
- The [September 11 feature brief](delimit3d_feature_geometry_evaluation.md)
  and [September 16 readiness audit](../reports/scratch_baseline_readiness_20260916.md)
  are historical design records, not the current launch checklist.
- [Report sources](../reports/README.md) rebuild figures/tables from external
  artifacts; generated media and LaTeX build products are ignored by Git.

## Refresh the evidence summary

```bash
python3 scripts/reporting/snapshot_experiment_results.py \
  --artifact-root /cluster/work/igp_psr/nedela \
  --output /tmp/delimit3d-evidence.json
```

The extractor checks frozen-evaluation identity coverage and joint/part counts,
then saves selected metrics and SHA-256 hashes of the exact input bytes. Review
a new dated summary before committing it; live job status is checked separately.
