# Matched ScanNet40 frozen transfer

Scientific question: does 256-update label-free Delimit3D adaptation improve a pretrained LitePT encoder's point-conditioned interactive object segmentation under a matched fresh decoder?

This is a matched LitePT comparison under the AGILE3D metric protocol. It is not a reproduction of AGILE3D's original backbone or its 1,100-epoch joint supervised training. Ground-truth ScanNet40 instances train only the fresh 128-dimensional decoder. No encoder is instantiated in the decoder training worker; the optimizer owns exactly the decoder parameter IDs.

Source lives on Euler in branch `experiment/scannet40-agile3d-matched-v1`, separate worktree `/cluster/home/nedela/nedela/projects/delimit3d-scannet40-matched`. External products live under `/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1`. Existing Stage-A artifacts and its immutable source archives are untouched. The Stage-A wrapper and independent gate verifier are included here for provenance; their separate deployed locations remain documented in the lead audit.

## Data and metrics

Use all 1,200 official training and 312 official validation scenes. ScanNet40 denotes the 40-class source benchmark, not 40 scenes. Official lists are copied byte-identically into `configs/evaluation/manifests/scannet40` and frozen externally with all four MO/SO list hashes. The official PLY establishes coordinates, RGB and instance IDs; processed normals may be used only after every point's coordinates and RGB match exactly. The official PLY label field alone defines object IDs, masks and sizes. Processed-instance identity is recorded diagnostically and never used to redefine official targets. Label `-1` and nonrequested objects become background, including in the IoU union.

Native LitePT extraction retains RGB then unit normals, centered XY and grounded Z, 2cm voxelization, representative-first mapping and 72-dimensional dec0 tokens. CPU preparation rejects any positive object without a surviving representative token; it never drops official targets silently. Source PLY and normals are hashed on read. Every cache carries the code commit, source geometry, checkpoint and encoder tensor hashes; both arms must cover the exact full manifest scene set. Feature-cache memory during decoder training is bounded to two scenes.

Official MO output is a space-delimited, headerless `.csv`: `sample_index scene_without_prefix requested_count average_click_count mean_requested_object_IoU`. At zero clicks the score is zero; initial clicks jump to one per object, then one global corrective click is added at a time, including background. No per-label cap is imposed. Perfect predictions are repeated through 20 clicks per requested object. NoC thresholds apply to scene-mean IoU; they are not averages of individual-object NoCs. Fractional clicks per object are retained. Exact float roundtrips avoid threshold-crossing errors from rounded output.

SO uses each of the 10,357 official full-scene object identities, original PLY instance ID and integer click counts 0 through20. It includes all40 classes. Primary MO averages scenes; secondary SO averages objects. We report IoU@1/3/5/10/15, NoC@50/65/80/85/90, paired scene/object bootstrap intervals, full traces, size/class/requested-count strata, and prediction/interactive curve plots. The MO object strata are descriptive additional analyses, not the official scene endpoint.

## Matching and recovery

The shared decoder initialization tensor is generated once per frozen root at seed20260912. Both arms use the same seed, deterministic episodes, AdamW LR1e-4, weight decay1e-4, clip norm0.1 and5000 updates. Shuffled epochs guarantee every training scene is visited before repeats. The200-event time table suffices for official maximumK10. A checkpoint is saved atomically every250 updates and at the terminal update, including decoder, optimizer and Python/NumPy/Torch/CUDA RNG state. Resume rejects mismatched schema, experiment, arm, commit, architecture, initialization, episode schedule, encoder, tensor state, update bounds or missing RNG state. Final evaluation rejects interim or wrong-arm checkpoints and waits for both final reports.

The prior Stage-A continuation restored update1000 decoder and AdamW states without RNG state; it remains explicitly non-bit-exact. That limitation is not inherited by this new checkpoint format.

## Preparation and smoke

All commands below run after explicit SSH to Euler. The worktree must be clean and committed before freezing. `freeze` writes resolved YAML, complete source archive, per-file hashes, immutable decoder initialization, checkpoint/list hashes, CPU environment and a snapshot/archive of LitePT runtime source, including dirty upstream files as they existed. GPU execution reads the frozen dependency snapshot.

```
cd /cluster/home/nedela/nedela/projects/delimit3d-scannet40-matched
PY=/cluster/work/igp_psr/nedela/litept-env/bin/python
$PY scripts/evaluation/run_scannet40_matched.py --config configs/evaluation/scannet40_agile3d_smoke_v1.yaml --mode freeze
$PY scripts/evaluation/run_scannet40_matched.py --config /cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1/smoke_v1/freeze/resolved_config.yaml --mode prepare
```

The smoke root is separate. It uses scene0000_00 for training and scene0568_00 for validation, both encoders, K3 forward/backward and2 matched decoder updates, native official MO/SO output and plots. `scannet40_smoke.sbatch` requests `--gpus=rtx_4090:1`,8 CPUs,64GiB and30 minutes. Runtime asserts exactly one visible RTX4090. It extracts the hashed source archive and runs every GPU command in a dedicated tmux session/socket with PID and heartbeat files. The submitter supplies the exact frozen archive/config/Python/root environment variables. No production cache or training is launched by preparation or smoke.

CPU tests include decoder/protocol fixtures, official CSV accounting and exact threshold-boundary checks. Independent protocol fixtures run the generated CSVs through pristine upstream evaluators. The source used by execution must match every frozen source-file hash. The smoke completion marker requires both encoders, both small trainings, MO/SO evaluation and aggregate plots to finish.

## Full launch decision

The explicit `launch_scannet40_matched.py --config <full frozen resolved YAML>` entrypoint creates `delimit3d-scannet40-matched-v1` on the selected host only after lead review. It refuses occupied GPUs or existing sessions. It re-runs the independent final Stage-A verifier through explicit Euler SSH and requires status`passed`, completed Stage-A aggregation/visuals, complete CPU tests, matching real RTX4090 smoke provenance, full data preparation, source/checkpoint/list/init/environment invariants. The full run then caches both arms, checks complete byte-identical geometry, trains both and automatically evaluates only after both exact5000-update reports exist.

Gate thresholds: MO-5 IoU@1 delta>=0.02; IoU@5 delta>=0.015; both paired95% scene-bootstrap lower bounds>0; NoC@80 delta<=0. A pending, failed or mixed final Stage-A gate withholds the full run. Training-loss differences never substitute for this downstream decision. The full launch must wait for a final throughput/cost estimate from the real smoke and the lead's documented decision.

The full audit exposed a label-namespace discrepancy in scene0217_00: its raw aggregation repeats31 segment groups as a second set of IDs. Official PLY retains the first IDs, while processed arrays carry the later IDs. Both represent the same partitions and exact point geometry. The resolution is preserved in `protocol_audit/scene0217_00_resolution.json` (SHA256 b716bb611a3ab6ead432834620cac408b6b9c99f4eacf78ba69e517bcf834eae). This is why official PLY IDs remain authoritative and processed-label agreement is a diagnostic. No data are changed or official targets excluded.
