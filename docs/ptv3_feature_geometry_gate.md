
# PTv3 feature-level gate before full AGILE3D training

This diagnostic is the checkpoint-gated upstream test for the matched PTv3
adaptation. It compares the public PTv3 encoder with the completed
256-update Delimit3D PTv3 encoder on the same frozen 50-scene ScanNet++
validation bundle.

It does not train a decoder and it does not claim interactive segmentation
performance. Ground-truth instance masks are used only to define the fixed
query targets and to score frozen feature retrieval.

## Metrics

For each fixed object query, both arms use the same normalized dense PTv3
point features and the same candidate points.

The primary metrics are:

- mean average precision for retrieving the clicked instance;
- same-instance minus same-class-different-instance cosine margin;
- the 90th-percentile hard same-class margin.

Supporting metrics are retrieval recall/IoU at fixed candidate fractions,
same-class distractor rate, feature-space effective rank and first-PC
fraction, 16-nearest-neighbor instance purity, semantic-neighbor retention,
and raw feature norm statistics.

All means are paired at the scene level. The aggregate report contains a
10,000-resample paired scene bootstrap interval for every metric.

## Decision rule

The report is GREEN only if all of these hold:

- retrieval AP improves by at least 0.02 and its paired 95% interval is above
  zero;
- same-class cosine margin improves by at least 0.02 and its paired interval
  is above zero;
- same-class distractor rate at the 1% candidate fraction does not increase
  by more than 0.02;
- semantic 16-NN purity does not decrease by more than 0.05.

GREEN authorizes the full frozen-backbone AGILE3D decoder experiment.
AMBER means that at least two supporting checks are positive, but the
predeclared primary evidence is incomplete; run a limited decoder smoke or
inspect strata before committing to the full decoder budget. RED means that
the adaptation did not produce a useful frozen-feature signal; do not spend
the full AGILE3D decoder budget on this checkpoint.

A higher effective rank or more colorful PCA is supporting evidence only. It
cannot make the decision by itself.

## Provenance

The evaluator records the public and adapted checkpoint SHA-256, the loaded
encoder tensor hash, PTv3 source revision, frozen scene-manifest hash, source
array hashes, coordinate/input contracts, and the code commit. It asserts
that the encoder hash is unchanged during evaluation.

The wait-and-run mode can be launched before the adaptation finishes. It
polls for the final checkpoint_u000256.pt and train_report.json, then
evaluates the two arms concurrently on GPU 0 and GPU 1 and writes
aggregate_feature_gate.json under the external artifact root.
