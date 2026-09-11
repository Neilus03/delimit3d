# Independent-seed 256-update adaptation — preregistered plan

> **Protocol correction (2026-09-11).** The completed r2 run below is **not a
> valid seed-only repeat of public-LitePT posttraining**. It loaded the scratch
> initialization `initial_representative_rgbn6_seed42.pt` (SHA
> `bd064e058c3770c652cc5367f7547a97169c2613946a14fae0f126859e3f15a7`,
> `public_checkpoint_used=false`) and used backbone LR `0.001`. The earlier
> positive public-posttraining run loaded `public_initial.pt` (SHA
> `67785e9d66881225aea61c090850d14f11e754978004f5e2e55200103a096db8`, with
> public LitePT backbone SHA
> `86408c6371555aeee2a8eda1184b55a411f9748784c275349149b509b661d518`) and
> used backbone LR `0.0001`. The downstream decoder was matched, but the
> encoder starting point and LR were not. The negative aggregate is retained
> as a protocol/infrastructure diagnostic and must not be used as evidence
> that public-LitePT posttraining fails.

This was intended as the next experiment after the completed 512-update
duration test. The intended design kept the Delimit3D recipe, RGBN6 contract,
source cells, fixed evaluation pack, **public** LitePT initialization, world
size, optimizer settings, and downstream protocol unchanged. The only planned
scientific change was the stochastic training seed: the previous 256-update
result used seed 42; this run used seed 20260911. Because the actual r2 launch
did not use the public initialization, the planned reproducibility hypothesis
was not tested.

**Causal hypothesis.** If the 256-update transfer gain reflects a stable effect
of multigranular 2D pseudomask adaptation rather than one favorable stochastic
trajectory, changing only the seed should preserve a positive frozen-feature
point-selection gain over public LitePT.

**Predeclared success criterion.** The run passes the reproducibility screen if
the adapted arm improves public LitePT by at least +0.02 AP and +0.02 fixed IoU,
has positive paired-scene AP in at least two thirds of the 50 validation scenes,
and clears the existing broad feature-health checks without a new concentration
warning. The same learned-decoder gate is used for the prior 256 result and for
this seed; no threshold will be changed after seeing the result.

**Decision rule.** If a correctly public-initialized seed repeat passes, the
256-update effect is sufficiently reproducible to justify a second seed or a
narrowly regularized continuation, followed by a standard multi-click curve.
If it fails while the public arm and protocol remain valid, the original 256
gain is treated as seed-sensitive. The completed r2 run cannot trigger either
branch because its initialization and LR do not match the intended experiment.

The choice follows the experimental logic in related work. PARTFIELD learns a
feedforward part feature field from mixed 2D and 3D proposals using contrastive
learning and evaluates feature grouping and interactive selection, so a
point-conditioned frozen-feature endpoint is a relevant diagnostic. AGILE3D
reports low-click interactive segmentation and emphasizes that the one-click
regime is informative, while S2AM3D identifies cross-view inconsistency as a
central risk when transferring 2D segmentation priors into 3D. A seed-only
repeat is the lowest-cost test of whether our apparent 256-update gain is a
stable representation effect before adding another mechanism or dataset.

- [PARTFIELD (ICCV 2025)](https://arxiv.org/abs/2504.11451)
- [AGILE3D (ICLR 2024)](https://arxiv.org/abs/2306.00977)
- [S2AM3D (arXiv)](https://arxiv.org/abs/2512.00995)

**Expected cost.** On `pf-pc69` with two RTX 4090 GPUs, native adaptation is
expected to take about 35–45 minutes of compute for 256 updates plus up to 50
minutes of source preload through the current Euler mount. The matched decoder
should take about 10 minutes, followed by a short copy to Euler. The end-to-end
estimate is therefore about 1.5–2 hours, with no 2080 Ti allocation.

**Launch record.** The first corrected launch started on `pf-pc69.ethz.ch` at
2026-09-11 00:17:10 CEST with launcher PID `4086876` and two DDP workers on
two RTX 4090 GPUs. The adaptation log is
`/home/nedela/delimit3d_runs/logs/adaptation_256_seed20260911_native_20260911.log`.
The immutable execution-provenance SHA-256 is
`9a533050a13d9cea592a6f91d9640cd6da28923c2cbe571f1566c85eac486030`; the
corrected launcher script SHA-256 is
`dbef400cfe645d12aa6251022bf6113a7f0d3b6fd6db3b6b72259613df0fc21b`.

Two earlier attempts exited before model construction: one used an invalid
epoch CLI value and the other encountered the runner's refusal to overwrite an
empty output directory. Their logs are retained as infrastructure provenance;
neither produced a checkpoint or scientific metric. The detached post-run chain
(`postprocess PID 4087567`) correctly refused to evaluate after the adaptation
failed.

At 01:10:01 CEST the adaptation reached the fixed-pack validation and exited
before model construction with `Fixed-eval seed drift`: the runner coupled the
training seed to the immutable pack seed. No checkpoint or scientific metric
was produced. The `pf-pc69` SSH service then became unreachable, so the planned
relaunch is pending host recovery.

The repository now contains a focused seed-contract fix: `--seed` remains the
training/sampling seed, while `--initialization-seed` and `--fixed-eval-seed`
allow a controlled repeat to retain the exact seed-42 initialization and fixed
evaluation pack. The fix also records both artifact seeds in the resolved
configuration and adds a unit test. The updated runner was copied to the
execution host, compiled, imported successfully, and its new CLI flags were
checked before relaunch. The full experiment itself is the real two-rank GPU
integration check because it exercises the same DDP, source preload, fixed-pack,
initialization, and update path used for the scientific result.

**Relaunch record.** The corrected repeat was launched on `pf-pc69.ethz.ch` at
2026-09-11 06:58:56 CEST with launcher PID `4187463` and DDP workers
`4187491/4187492` on two RTX 4090 GPUs. Its log is
`/home/nedela/delimit3d_runs/logs/adaptation_256_seed20260911_native_20260911_r2.log`.
The immutable code-manifest SHA-256 is
`20d73f11b6661653819e012da69d4539a83fb2b6ac59db20c9821cbecdd5945c`, the
execution-provenance SHA-256 is
`a362bb1fd04708fbce23c6f6238e9c3aa7627932ff1d700a6714582a0a053a08`, and the
launcher SHA-256 is
`efbd6b7e8ea20e44d3b6aed2c3c16fd0bbfd90558157f97d23310217167627ab`.
The run uses training/sampling seed `20260911`, initialization artifact seed
`42`, and fixed-evaluation seed `42`; no 2080 Ti is allocated. The detached
matched decoder postprocess was launched as PID `4187827` after the checkpoint
and run-summary existence checks passed, and its completed report is recorded
below. The initialization artifact is explicitly scratch, as recorded in the
run summary; the seed value `42` does not make it the public checkpoint.

**Result.** The corrected runner completed the r2 protocol on the two RTX
4090s, but the scientific repeat was invalidated by the initialization mismatch
described above.
The adaptation took `2036.0058015063405` seconds after model construction
(about 33.9 minutes; the end-to-end wall time was longer because the Euler
source mount was preloaded). The sampled training loss decreased from `6.7275`
at update 1 to `4.8457` at update 256, so the upstream objective did optimize.
The final raw endpoint reported a fixed-comparable native dec0 cosine gap of
`0.35023377887784574`, native-selection macro gap of `0.3546075248256481`,
effective rank `6.682948679663241`, mean feature standard deviation
`1.0274802053172607`, and minimum feature standard deviation
`0.030919020995497704`. All recorded mechanics, finite-value, source-cell,
seed-contract, world-size, and provenance checks passed. The formal
`health_selection_status` remains
`pending_external_five_stage_bn_recalibrated_gate` because that external
sidecar was not run; this raw checkpoint is therefore not production
qualified.

The checkpoint and run summary are immutable and have been copied to Euler
and the shared work filesystem:

| artifact | path | SHA-256 |
|---|---|---|
| run summary | `/cluster/work/igp_psr/nedela/delimit3d/adaptation_256_seed20260911_native_20260911_r2/run_summary.json` | `b04e141010ecb36cc2309908cacb3d6e05f9bfccfa2a58719dd324895f4c426a` |
| adapted checkpoint | `/cluster/work/igp_psr/nedela/delimit3d/adaptation_256_seed20260911_native_20260911_r2/checkpoints/audit_e0001_u00000256.pt` | `8d8189693c8edd25a25da9a22ec51a3b35a84b938e94ba25f5da12b14bca6400` |
| metrics trace | `/cluster/work/igp_psr/nedela/delimit3d/adaptation_256_seed20260911_native_20260911_r2/metrics.jsonl` | `7505ce4526f3477335840e1914129c7f4d49ea99215968d416bbaf4cf4e8156d` |
| resolved config | `/cluster/work/igp_psr/nedela/delimit3d/adaptation_256_seed20260911_native_20260911_r2/resolved_config.json` | `5111fcd8b6109369de23087abf38a9054ae0203b7029f19b3fa6aaaadfecf3f0` |
| execution provenance | `/cluster/work/igp_psr/nedela/delimit3d/prep/adaptation_256_seed20260911_native_20260911_r2/execution_provenance.json` | `a362bb1fd04708fbce23c6f6238e9c3aa7627932ff1d700a6714582a0a053a08` |

The same fresh point-conditioned decoder was then trained on the frozen
public and scratch-initialized adapted features using the existing 50/50 ScanNet++ split, 777
objects/queries, one click per query, and 45,697 trainable decoder parameters.
The decoder report passed its internal integrity checks and confirmed
`encoder_state_unchanged=true`.

| frozen encoder | AP | fixed IoU | precision | recall | oracle IoU | F1 |
|---|---:|---:|---:|---:|---:|---:|
| public LitePT | 0.408487 | 0.155685 | 0.188703 | 0.840771 | 0.330785 | 0.231693 |
| Delimit3D, 256 updates from scratch initialization, seed `20260911` | 0.322039 | 0.070943 | 0.081946 | 0.846415 | 0.254696 | 0.117214 |
| adapted minus public | -0.086449 | -0.084743 | -0.106757 | +0.005644 | -0.076088 | -0.114479 |

The numerical gate would be **false for this mismatched comparison**, but it is
not a valid application of the preregistered public-posttraining gate.
Paired-scene AP was
positive on only `6/50` scenes (negative on `44/50`); fixed IoU was positive on
`1/50` and negative on `49/50`; oracle IoU was positive on `5/50` and
negative on `45/50`. The aggregate is not a thresholding accident: recall is
roughly unchanged, while precision, fixed IoU, oracle IoU, and F1 all fall.

This is a **protocol-audit result, not a negative reproducibility result**. It
shows that a scratch-initialized 256-update model with the current high LR
performs below public LitePT under the matched decoder. It does not tell us
whether the public-LitePT posttraining effect is seed-stable. The earlier
512-update run also recorded `public_checkpoint_used=false`, so its negative
duration comparison cannot be interpreted as a public-initialized 256-versus-
512 study either. The next scientifically valid action is one correctly
public-initialized repeat using public-initialization SHA
`67785e9d...` and the original public-posttraining LR `0.0001`, with every
other artifact and the matched decoder held fixed. No such correction has
been launched yet.

The matched aggregate is stored at
`/cluster/work/igp_psr/nedela/delimit3d/point_decoder_seed20260911_v1/aggregate.json`
with SHA-256
`bab5169e01ad870a3fd68dc6224e04d3ef4a0acda3fcfa5405af24457224ecf2`.
