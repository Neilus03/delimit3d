# Independent-seed 256-update adaptation — preregistered plan

This is the next experiment after the completed 512-update duration test. It
keeps the Delimit3D recipe, RGBN6 contract, source cells, fixed evaluation pack,
initial public LitePT state, world size, optimizer settings, and downstream
protocol unchanged. The only scientific change is the stochastic seed: the
previous 256-update result used seed 42; this run uses seed 20260911. The
checkpoint will be evaluated with the same matched fresh point-conditioned
decoder on the same 50/50 ScanNet++ train/validation scene split.

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

**Decision rule.** If this seed also passes, the 256-update effect is sufficiently
reproducible to justify a second seed or a narrowly regularized continuation,
followed by a standard multi-click curve. If it fails while the public arm and
protocol remain valid, the original 256 gain is treated as seed-sensitive and we
stop scaling this recipe. A failed run is still retained as evidence.

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
configuration and adds a unit test. It will be copied to the execution host,
compiled, and smoke-tested before relaunch; the scientific success gate is
unchanged.

**Status.** Relaunch pending host recovery and the seed-contract preflight. The
result and exact hashes will be appended to this file when the run finishes.
