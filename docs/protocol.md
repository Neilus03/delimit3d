# Reproducibility protocol

## Experiment records

Every run has a unique external artifact directory and an immutable resolved
configuration, code snapshot/commit, source and checkpoint hashes, runtime,
host/GPU identity and launch record. Record dirty source-file hashes when a
run predates a commit; a later commit must not be substituted for the code
actually executed. Keep datasets, checkpoints, caches, logs and rendered
outputs outside Git. Small reviewed summaries with source hashes and the
scripts that produce them may be versioned.

## Matched comparisons

- For the current scratch versus pretrained question, vary only encoder
  initialization. Match architecture, input channels, decoder initialization,
  scene lists/order, augmentation, optimizer/schedule, seed, training budget,
  checkpoint selection and evaluator. Record full fine-tuning or frozen-encoder
  training explicitly. RGB3 and RGBN6 are distinct contracts.
- The new scratch/pretraining family uses 5 cm throughout, including caches
  and downstream evaluation. Earlier RGBN6 short-adaptation studies keep their
  original 2 cm representative-first voxelization. Never reuse a 2 cm cache
  in the 5 cm family or rewrite an old result as a new-resolution result.
- Public versus adapted frozen-feature comparisons share inputs, prompts,
  candidate sets, geometry and inverse maps, with a fresh byte-identical
  decoder initialization trained separately for each encoder. Evaluate each
  encoder with its corresponding trained decoder.
- Exact upstream AGILE3D, LitePT RGB3 substitution, custom RGBN6 AGILE3D, and
  the 5,000-update frozen/joint studies are separately named experiments.
  Match within a family; cross-family numbers are system comparisons.
- ScanNet20 things (Mask3D) and ScanNet40 objects (AGILE3D) have different
  label universes and 1,201 versus 1,200 training scenes. Equal validation
  scene counts do not make the tasks identical.

## Evaluation and reporting

Ground truth selects prompts and computes frozen-feature metrics; it never
enters the frozen encoder forward. Decoder training uses its declared labels.
Keep feature retrieval AP, fixed/oracle IoU, automatic CA-AP and interactive
IoU/NoC separate. Always report the click budget, object universe, denominator,
averaging unit, checkpoint pair and confidence-interval unit.

Use official MO/SO lists where claimed. Do not drop hard objects silently.
Articulate3D's ≥5 / ≥10 / ≥20 thresholds are minimum surviving unambiguous-token
eligibility thresholds; ≥10 is primary. Its primary interactive metric is
raw-point IoU after inverse-map expansion. Part-weighted means and scene-paired
bootstrap differences need not have identical deltas.

Partial evaluations use the exact completed identity intersection and are
labeled interim with a capture time and hashes. Final status requires complete
identity coverage and aggregate reports. Loss, feature panels, a passing smoke
or a running job are not final transfer evidence. Validation at epoch 250 of
600 is an interim result, even if that checkpoint's evaluation has finished.

## Recovery and release hygiene

Save resumable optimizer/scheduler/scaler/RNG state without changing the full
training horizon. Preserve prior attempts and logs; use atomic checkpoint
replacement. Stage runtime and inputs on node-local storage when required,
and record relocated paths and original configuration identity.

Do not let parallel evaluators share an atomic-write temporary filename or
overwrite a common status document; use separate status paths or serialize
them. The historical Articulate3D runner's shared `status.json` requires this
care when reusing its split-arm launch mode.

Before publishing, review source/config changes, validate syntax and local
document links, run relevant regression tests, and exclude generated/private
artifacts. Cleaning the source worktree must preserve external experiment
directories and other active worktrees.
