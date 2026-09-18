# Delimit3D feature-geometry evaluation brief

**Historical brief (11 September):** the proposal below is preserved as written.
Subsequent feature and interactive experiments have completed; see the
[September 18 results ledger](results/20260918_experiment_summary.md) and
[current research direction](RESEARCH.md). This is not the current status or
launch checklist. The original pinned AGILE3D protocol is in Section 12.
**Date captured:** 2026-09-11

**Purpose:** Preserve a quantitative and qualitative evaluation plan for testing whether short Delimit3D adaptation changes LitePT features in a useful, instance-sensitive, multi-granular way, and whether that change transfers to AGILE3D-style interactive segmentation.

## 1. Core research question

Does a short, label-free, multi-granular Delimit3D adaptation reshape frozen LitePT features so that they are more useful for distinguishing object instances and object parts, especially between different instances of the same semantic class, and does this representation-level change improve interactive segmentation when only a small decoder is trained afterward?

The intended comparison is not only:

> Delimit3D features look more dispersed than original LitePT features.

It is:

> Delimit3D features form a more useful relational geometry: points that should belong together remain coherent, while points that should not be confused—particularly same-class points from different instances—become more separable.

## 2. Mechanistic hypothesis

### Main hypothesis

LitePT's original training may encourage useful semantic invariance: points from different instances of the same class can remain relatively similar because semantic consistency is beneficial. The Delimit3D contrastive objective adds a more direct pressure for instance- and part-sensitive separation at multiple granularities.

Therefore, after adaptation, normalized features may occupy more useful angular directions and exhibit a more hierarchical similarity structure, without necessarily increasing the literal dimensionality or norm of the feature space.

This is a hypothesis, not an assumption to present as an established property of LitePT. We should not claim that LitePT makes all same-class instances identical. Its encoder may retain instance, spatial, geometric, offset, and contextual information. The test is whether Delimit3D measurably changes the balance between semantic invariance and instance separability.

### Predicted relational structure

When the supervision supports the corresponding hierarchy, the predicted similarity ordering is approximately:

```text
same fine region
    > same object, different part or fine region
    > different object, same semantic class
    > unrelated different object/class
```

The exact absolute ordering is less important than the margins and the rate of violations. The goal is not to make all points different from one another. It is to make the feature space compact within the correct object and discriminative across the correct boundaries.

### What “wider feature space” should mean operationally

“Wider” is potentially useful shorthand, but it is not a sufficient scientific endpoint. In particular, if features are L2-normalized, they lie on a common unit hypersphere. The relevant changes are then angular spread, anisotropy, effective rank, and—most importantly—task-relevant pairwise separation.

The preferred wording is therefore:

> Delimit3D increases the effective and relational capacity of the normalized feature geometry.

This is stronger and safer than saying that it physically expands the feature space.

## 3. Evaluation arms

The primary representation comparison should use:

1. **Public LitePT:** frozen original public checkpoint.
2. **Delimit3D:** frozen LitePT after the short adaptation, initially the 256-update checkpoint of interest.
3. **Optional controls:** project initialization, additional adaptation budgets, and replicated adaptation seeds if available.
4. **AGILE3D:** external task-specific system comparison at the downstream level, under a clearly documented protocol.

For the decoder-transfer experiment, public LitePT and Delimit3D must use the same:

- scenes and train/validation split;
- input tensors and feature dimensionality;
- voxelization and inverse mapping;
- feature stage and normalization;
- click sampling policy and candidate points;
- decoder architecture and initialization;
- optimizer, learning rate, number of updates, and stopping rule;
- random seeds or replicated-seed protocol;
- evaluation code and metric implementation.

Raw feature-space statistics should primarily compare public LitePT with Delimit3D. A direct comparison of feature spread with AGILE3D is not meaningful unless the representations, input modality, tokenization, stage, and normalization are made comparable. AGILE3D is mainly an external system-level downstream anchor.

## 4. Quantitative feature analysis

### 4.1 Matched pair categories

For each query point or token, construct object-balanced and scene-balanced candidate pairs. The exact label definitions should be frozen before running the final analysis.

| Pair category | Measurement question |
|---|---|
| Same fine mask / same local region | Are local features coherent? |
| Same object, different fine region or part | Does object identity remain coherent across subparts? |
| Different instance, same semantic class | Can the representation reject semantic distractors? |
| Different instance, different class | Is there general object-level separation? |
| Spatially adjacent, different instance | Does the representation respect object boundaries? |
| Spatially distant, same instance | Does it preserve nonlocal object coherence? |

If explicit part labels are unavailable, do not call the result a part-level measurement. Use the available coarse/fine mask hierarchy, or report same-instance similarity in spatial-distance bins instead.

### 4.2 Pairwise similarity and margins

For each feature arm, compute cosine-similarity distributions and per-query margins such as:

```text
same-instance similarity
    minus same-class-different-instance similarity
```

Recommended outputs:

- median and mean cosine similarity for every pair type;
- object-balanced and scene-balanced distributions;
- same-instance versus same-class-distractor margin;
- same-fine versus same-object-different-part margin;
- ROC-AUC or pairwise ranking AUC for instance discrimination;
- violation rate of the predicted similarity ordering;
- bootstrap confidence intervals over scenes and objects.

The same-class-different-instance comparison is especially important for testing the “two opposing walls” intuition. Spatially close different-instance pairs should be included, because a gain that only separates very distant points may be a geometry or scene-layout effect rather than improved instance selectivity.

### 4.3 Retrieval and neighborhood metrics

Continue the existing frozen-feature retrieval style of analysis, but make the hard-negative structure explicit:

- positive-point-conditioned same-instance retrieval AP;
- Recall@K for the clicked instance;
- k-nearest-neighbor instance purity;
- false-positive rate for same-class distractors;
- retrieval performance in difficult same-class scenes;
- retrieval performance by object size, click location, and boundary distance.

Use the exact same candidate set, query points, normalization, and scoring for both arms. Report the result as frozen-feature retrieval, not as official instance-segmentation AP.

### 4.4 Feature-space occupancy, anisotropy, and effective capacity

Measure the global geometry, but treat it as supporting evidence rather than the primary endpoint:

- feature norms before and after normalization;
- covariance eigenvalue spectrum;
- effective rank / participation ratio;
- fraction of variance explained by the first principal components;
- mean pairwise cosine similarity;
- angular-distance distribution;
- optional hyperspherical uniformity or hubness statistics.

These analyses distinguish several possibilities:

- genuine use of more directions in feature space;
- reduced anisotropy;
- a change caused only by feature norms;
- or indiscriminate noise-like dispersion.

If PCA is visualized, use identical points and a shared PCA basis for LitePT and Delimit3D. Separate PCA fits can make two feature distributions look different simply because their coordinate systems differ.

### 4.5 Semantic-versus-instance probe

Add a small held-out diagnostic to check whether the adaptation trades away too much semantic information:

- a frozen linear probe or nearest-neighbor readout for semantic class;
- the same-instance retrieval and same-class-distractor metrics above;
- optionally, a pair classifier using only feature similarity and spatial controls.

The desired outcome is not necessarily higher semantic-class accuracy. A plausible positive result is improved instance discrimination while retaining enough semantic information for the decoder. A large loss of semantic information should be reported explicitly rather than hidden behind improved spread.

## 5. Qualitative analysis

The qualitative analysis should make the hypothesized failure modes visible in both feature space and scene space.

### Recommended visual panels

1. **Shared PCA or UMAP:** identical sampled points for LitePT and Delimit3D, colored once by semantic class and once by instance ID.
2. **Same-class distractor scenes:** for a click on one wall, chair, table, or other repeated class, show the top-k feature-similarity points and highlight false positives on another instance.
3. **Multi-part objects:** show a chair or table with fine-region colors, then show similarity from a query on one part to the rest of the object and to other objects.
4. **Boundary cases:** visualize points near two adjacent instances and compare whether the adapted feature neighborhoods stop at the true boundary.
5. **Nonlocal coherence:** show a large object or disconnected visible parts and whether a query retrieves the correct object across spatial distance.
6. **Decoder readout:** show clicked points, predicted masks, ground truth, and typical LitePT-versus-Delimit3D corrections under identical clicks.

The qualitative panels should be selected from predefined categories, not only from visually impressive successes. Include at least one success and one failure for each important category.

PCA, t-SNE, and UMAP should illustrate the quantitative result; they should not be used as proof that the representation is better by themselves.

## 6. Downstream interactive evaluation

### Single-object interactive segmentation

Single-object evaluation is important and should not be treated as disposable warm-up work. It is the cleanest functional test of query selectivity:

> Given one click on an object, does the frozen adapted representation make that object easier for a small decoder to isolate from nearby and same-class distractors?

This setting is particularly useful for:

- same-class neighboring instances;
- clicks near boundaries;
- large objects with multiple parts;
- small objects and sparse point support;
- and different click budgets.

It connects most directly to the feature retrieval analysis because there is less confounding from multiple-query interaction.

### Multi-object interactive segmentation

Multi-object evaluation should be the main AGILE3D-style downstream assay. It tests whether the representation remains useful when several clicked objects must be segmented simultaneously and assigned separate masks.

It probes:

- inter-object query competition;
- same-class instance separation;
- background rejection;
- cross-object leakage;
- and whether improved feature geometry helps global scene consistency.

If following the official AGILE3D-style protocol, keep the standard IoU-at-click-budget and number-of-clicks reporting separate for single-object and multi-object evaluation. Do not collapse the two tasks into a single average score.

### Recommended downstream hierarchy

- **Primary downstream result:** multi-object interactive segmentation.
- **Controlled diagnostic:** single-object interactive segmentation.
- **Representation mechanism:** frozen feature geometry and retrieval.
- **External anchor:** AGILE3D system-level comparison under matched and transparently disclosed conditions.

## 7. Controls and failure checks

Before interpreting the result as improved representation quality, check:

- Are LitePT and Delimit3D evaluated on exactly the same tokens, scenes, and candidate points?
- Is the apparent spread caused by feature norms rather than directions?
- Are same-class comparisons matched for spatial distance and object size?
- Are large walls or large objects dominating point-weighted averages?
- Are object and scene bootstrap units used instead of treating all points as independent?
- Is the feature stage fixed? A gain at one stage should not be generalized to the whole encoder.
- Are PCA/UMAP visualizations based on a shared sample and shared projection basis?
- Are click locations, click order, and decoder random seeds controlled?
- Is there any source/evaluation scene leakage?
- Is the decoder actually frozen during the feature diagnostic and only trained in the downstream experiment?
- Are the AGILE3D comparison conditions different in RGB/RGBN6 input, voxel size, resolution, or training data exposure?

Do not interpret a higher effective rank or visually wider cloud as a positive result if same-instance retrieval, hard-negative rejection, or downstream segmentation does not improve.

## 8. Predeclared interpretation matrix

| Feature geometry | Single-object | Multi-object | Interpretation |
|---|---|---|---|
| Improves | Improves | Improves | Strong evidence for the proposed mechanism and transfer benefit |
| Improves | Improves | No improvement | Better local/query selectivity, but multi-query interaction or decoder capacity is limiting |
| Improves | No improvement | Improves | Representation may help global competition or shared-scene reasoning more than isolated retrieval |
| Improves | No improvement | No improvement | Geometry changed, but not in a decoder-usable way; inspect interface, tokenization, and normalization |
| No clear geometry gain | Improves | Improves | The “wider feature space” explanation is incomplete; investigate boundary, spatial, or query-alignment changes |
| Spread increases only | No improvement | No improvement | Treat as dispersion/noise, not improved representation quality |
| Improves over LitePT | Below AGILE3D | Below AGILE3D | Still a meaningful LitePT-to-Delimit3D transfer result; do not overstate an AGILE3D win |
| Improves over LitePT | Above AGILE3D | Above AGILE3D | Strongest result, subject to strict protocol and input/training-data comparability |

## 9. Minimum viable evaluation package

When this is implemented, the smallest useful package should contain:

1. A frozen feature cache manifest for public LitePT and Delimit3D, including checkpoint hash, scene list, feature stage, normalization, tokenization, and inverse-map metadata.
2. A CSV or Parquet table of matched pair statistics by scene, object, semantic class, distance bin, and pair category.
3. A quantitative report with cosine margins, retrieval, kNN purity, effective rank, anisotropy, and bootstrap confidence intervals.
4. A qualitative gallery with shared projections, hard same-class scenes, boundary cases, multi-part objects, and decoder masks.
5. A downstream table for frozen-backbone single-object and multi-object decoder transfer.
6. A provenance record that clearly separates upstream feature diagnostics, decoder-transfer performance, and the external AGILE3D comparison.

## 10. Suggested paper figure sequence

If the results support the hypothesis, a compact paper story could use:

- **Figure 1:** conceptual diagram: semantic invariance versus multi-granular instance-sensitive geometry;
- **Figure 2:** pairwise feature geometry, especially same-instance versus same-class-different-instance separation;
- **Figure 3:** qualitative same-class distractor and multi-part examples;
- **Figure 4:** frozen-decoder single-object and multi-object curves;
- **Table 1:** public LitePT, Delimit3D, and AGILE3D comparison with training/input/protocol disclosures.

One strong relational feature figure is more valuable than many generic PCA plots.

## 11. Final claim boundary

The intended claim should eventually be calibrated to the evidence:

> A short Delimit3D adaptation reshapes normalized LitePT features toward a more hierarchical, instance-sensitive geometry. On matched scenes, same-object points remain coherent across compatible regions and scales, while different instances—including same-class distractors—become less confusable. A small decoder trained on the frozen adapted features then improves interactive single-object and/or multi-object segmentation relative to the original LitePT representation.

Only if the official or matched downstream comparison supports it should the final sentence be strengthened to a claim relative to AGILE3D.

## 12. Pinned AGILE3D execution protocol

This section supersedes the earlier ScanNet++-first two-stage draft. The primary AGILE3D comparison must use the official ScanNet40 split, because AGILE3D trains on ScanNet40 and evaluates one multi-object-trained model in both multi-object and single-object settings. ScanNet++ is never used for training in this experiment. It is, at most, a later zero-shot cross-dataset evaluation, not the gate for the AGILE3D result.

### 12.1 Three-track experiment

| Track | Frozen representation | Trainable model | Purpose |
|---|---|---|---|
| `L-Frozen-AG` | Public LitePT checkpoint | Fresh AGILE3D-style decoder | Primary baseline |
| `D256-Frozen-AG` | LitePT after the fixed 256-update Delimit3D adaptation | Fresh decoder with byte-identical initialization | Causal representation comparison |
| `AGILE3D-Official` | Released AGILE3D system/checkpoint | None in our run | External benchmark anchor |

The first two tracks are the scientific A/B comparison. The third track is an external system comparison and must be reported with its different input and training conditions. Do not compare raw feature-space spread between the first two tracks and AGILE3D's internal features.

Do not literally take the released AGILE3D decoder weights and attach them to LitePT: the official decoder is coupled to the official backbone interface and feature distribution. For the first two tracks, port the decoder mechanism and train one fresh decoder copy per frozen LitePT representation on ScanNet40. The two copies must start from identical weights, but each is optimized on its corresponding frozen feature cache. The released AGILE3D checkpoint remains the separate full-system reference.

### 12.2 Dataset and splits

Use the released AGILE3D ScanNet40 processed data and pin hashes for:

- the official ScanNet40 training scene list;
- the official multi-object validation episode list;
- the official single-object `object_ids` list;
- the official single-object class list;
- the scene point clouds and labels;
- and the evaluator revision.

Train the two LitePT-based decoder arms only on the official multi-object training episodes. Evaluate the same trained decoder on:

1. the official multi-object validation episodes; and
2. the official single-object object list.

The resulting decoder checkpoints are ScanNet40-trained and then frozen for all later cross-domain evaluation. No ScanNet++ points, labels, episodes, or validation scores are used to train, select, or early-stop the decoder.

Do not train a separate single-object decoder for the headline result. This mirrors AGILE3D's design: one model is trained in the multi-object setting and evaluated in both settings. A single-object-trained decoder can be added later as an ablation, but it must not replace the common primary model.

Do not impose a new minimum-object-size filter, custom non-overlap resampling rule, or custom object list on the official benchmark. If a target object cannot be represented after LitePT voxelization, record it as a deterministic tokenization failure and report the count. Do not silently remove difficult objects from the denominator.

The official AGILE3D benchmark uses RGB input and 5 cm quantization in its released implementation. The public/Delimit3D arms should retain the current LitePT contract—RGBN6 and 2 cm dec0 tokens—for the causal comparison. This makes the first result a matched LitePT-versus-Delimit3D comparison under the AGILE3D interaction protocol, not an apples-to-apples reimplementation of the official AGILE3D backbone.

### 12.3 Frozen representation contract

For each ScanNet40 scene, cache once per arm:

```text
scene RGBN6 + centered XYZ
    -> frozen LitePT forward pass
    -> dec0 token features F[N,72]
    -> representative token XYZ[N,3]
    -> raw-mesh-to-token inverse map
    -> token ambiguity and coverage audit
```

The public and Delimit3D caches must have identical token coordinates, inverse maps, representative indices, scene ordering, and source labels. Only feature values may differ. Store checkpoint hashes, loaded-tensor hashes, preprocessing settings, scene manifest hash, and code revision in the cache manifest.

The encoder is completely frozen during decoder training:

- run feature extraction under `no_grad`;
- set every encoder parameter to `requires_grad=False`;
- exclude every encoder parameter from the optimizer;
- assert that the encoder state hash is unchanged before and after training;
- train only the 72-to-128 adapter and AGILE3D decoder parameters.

Disable geometric augmentation in the primary cached experiment. Applying a random coordinate transform without recomputing the corresponding LitePT features would create an invalid feature-coordinate pair. An online or precomputed augmented-cache experiment can be a later robustness ablation.

### 12.4 Decoder architecture

Implement the AGILE3D mask/query decoder in native PyTorch against the LitePT cache. The primary configuration is:

```yaml
input_dim: 72
hidden_dim: 128
num_heads: 8
dim_feedforward: 1024
num_decoders: 3
num_bg_queries: 10
dropout: 0.0
pre_norm: false
shared_decoder: false
auxiliary_outputs: true
positional_encoding: fourier
normalize_positional_encoding: true
gauss_scale: 1.0
click_time_capacity: 200
feature_levels: [dec0]
```

Use one LitePT dec0 level in the primary experiment. Do not add LitePT multilevel fusion until after the causal comparison; otherwise an architectural change is mixed with the representation change.

Preserve the relevant AGILE3D operations in this order:

1. project token features from 72D to 128D;
2. construct foreground queries from all clicked positive tokens, retaining object membership;
3. construct background queries from ten learned background queries plus clicked background tokens;
4. add normalized Fourier coordinate encodings and 1D temporal click encodings;
5. apply click-to-scene cross-attention;
6. apply click-to-click self-attention;
7. apply the feed-forward block;
8. apply scene-to-click cross-attention;
9. form mask logits from scene features and query mask embeddings;
10. take the per-object maximum over that object's click queries;
11. return `K+1` token labels, with background as label 0.

The decoder receives local episode object IDs only. Semantic class labels are never passed as inputs.

### 12.5 Training episodes and optimization

For every training scene with `M` eligible objects:

1. sample `K` uniformly from `1,...,min(10,M)`;
2. sample `K` distinct objects without replacement;
3. remap them to local labels `1,...,K`;
4. map every nonselected point to background label 0;
5. sample a prefix length uniformly from `0,...,19`;
6. reproduce the upstream AGILE3D training interaction loop, including its initial positive-click generation and subsequent error-correction clicks;
7. compute the loss on the resulting click state.

Use a batch size of one scene episode for the LitePT cache. Keep the scene/episode schedule, prefix lengths, and local random seeds identical between `L-Frozen-AG` and `D256-Frozen-AG`. A shared decoder initialization is required for the paired comparison: save one initial decoder state and clone it byte-for-byte for both arms.

The screening run is 5,000 optimizer updates per arm, with checkpoints at 1,000, 2,500, and 5,000. The final comparison should not rely on a possibly under-trained 5,000-update decoder alone. Extend the same runs to a predeclared 20,000-update endpoint, saving checkpoints at 10,000 and 20,000. If compute permits, repeat the 20,000-update endpoint with three paired decoder seeds; the single paired seed is a screening result.

Use:

```yaml
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 1.0e-4
grad_clip_norm: 0.1
losses: [cross_entropy, multiclass_dice]
cross_entropy_weight: 1.0
dice_weight: 2.0
auxiliary_loss_weights: same_as_main
click_weight_alpha: 0.8
click_weight_beta: 2.0
click_weight_radius_m: 0.3
final_updates: 20000
lr_drop_update: 18000
lr_drop_factor: 0.1
```

The name `loss_bce` in the AGILE3D source refers to multiclass cross-entropy. Preserve the upstream loss structure and auxiliary-output losses, but record the actual implementation in the provenance manifest.

### 12.6 Exact interaction policy

Use the official AGILE3D simulator semantics, not a new click heuristic:

- initialize with an all-background prediction;
- for multi-object episodes, generate one initial positive center click per requested object;
- predict the `K+1`-label scene mask;
- force already clicked token labels into the displayed prediction;
- form error groups from target-label/prediction-label pairs;
- rank groups by the largest distance from an error point to the group boundary;
- place the next click at the error point farthest from that boundary;
- add one correction click per global round after initialization;
- stop at 20 clicks per object on average.

The official implementation uses dense pairwise distances. For 2 cm LitePT scenes, use an exact Euclidean nearest-neighbor/KD-tree implementation only as a computational substitution. Validate it against dense distances on small fixtures, preserve the upstream error-group semantics, use a local deterministic RNG for the upstream tie shuffle, and use stable lowest-index selection for exact distance ties.

For single-object evaluation, use the official object list and binary target mask. For multi-object evaluation, use the official episode object groups and local labels. Do not call `K=1` multi-object episodes the official single-object result; report them separately from the official single-object evaluator.

### 12.7 Metrics and aggregation

Run the official-compatible evaluators and save the raw per-episode traces. Report:

- single-object: IoU@1, @2, @3, @5, @10, @15 and NoC@50, @65, @80, @85, @90;
- multi-object: IoU@1, @3, @5, @10, @15 and NoC@50, @65, @80, @85, @90;
- optional diagnostic: IoU@20 and curves stratified by the number of requested objects `K`.

For every episode, save scene ID, object IDs, `K`, random seed, click coordinates, click signs, click times, token indices, full-mesh predictions, per-object IoU, and the evaluator inputs. Expand token predictions through the fixed inverse map before computing mesh-level IoU.

For the public-versus-Delimit3D causal comparison:

- compute Delimit3D minus LitePT at every click budget;
- average objects within each scene before the primary aggregate;
- bootstrap scenes, not individual points, with 10,000 deterministic resamples;
- report the paired mean delta and 95% confidence interval;
- report both official evaluator aggregates and scene-balanced aggregates.

The primary causal endpoint is multi-object IoU@1 and IoU@5. A screening pass is:

1. Delimit3D improves MO IoU@1 by at least 2.0 percentage points;
2. Delimit3D improves MO IoU@5 by at least 1.5 percentage points;
3. the paired 95% scene-bootstrap lower bound is above zero for both;
4. Delimit3D does not increase MO NoC@80.

These thresholds are a resource-extension gate, not a definition of scientific truth. A mixed or smaller positive result remains reportable; it simply does not authorize a stronger AGILE3D headline without confirmation.

### 12.8 External AGILE3D comparison

Evaluate the released AGILE3D checkpoint using its official ScanNet40 scripts/evaluator, or reproduce it from the official training entry point and verify against the released result files. The released implementation uses RGB features at 5 cm and its original backbone, while the LitePT arms use RGBN6 at 2 cm and a frozen pretrained backbone. The official training script uses a much longer end-to-end training schedule than the decoder-only experiment.

Therefore use two precise claim levels:

**Level 1 — causal claim:**

> Under identical decoder training and identical ScanNet40 interactive episodes, the frozen 256-update Delimit3D representation is better than the frozen public LitePT representation.

**Level 2 — qualified external benchmark claim:**

> On the ScanNet40 AGILE3D interactive benchmark, the LitePT+Delimit3D system with RGBN6/2 cm input scores higher/lower than the released AGILE3D RGB/5 cm system under the reported evaluator.

Use “better than AGILE3D” without qualification only after adding a condition-matched AGILE3D control with the same raw input modality, voxelization, train/validation episodes, and disclosed training budget. The first experiment does not need that expensive control to establish the LitePT-to-Delimit3D causal result.

### 12.9 Execution order

1. Pin the official ScanNet40 data/evaluator and the AGILE3D source revision.
2. Implement and CPU-test the decoder, loss, token expansion, episode labels, and click simulator.
3. Run a one-scene GPU smoke test with `K=3`, one backward pass, one optimizer step, and ten click events.
4. Export public and Delimit3D feature caches and verify identical geometry/inverse-map hashes.
5. Run the paired 5,000-update screening training for both arms.
6. Evaluate both arms on official SO and MO episodes, including raw traces and bootstrap intervals.
7. Run the frozen-feature geometry diagnostics on the same held-out scenes without changing the checkpoints.
8. Extend to the predeclared 20,000-update endpoint and, if possible, three paired decoder seeds.
9. Evaluate the released AGILE3D system under its official evaluator and place it in a separate external-comparison table.
10. Only afterward evaluate the ScanNet40-trained frozen encoder+decoder on other datasets using their official AGILE3D preprocessed data and evaluators; never use those target datasets for adaptation, decoder training, checkpoint selection, or early stopping.

### 12.10 Zero-shot cross-domain evaluation

After the ScanNet40 decoder experiment is complete, reuse the exact frozen checkpoints without any parameter update:

```text
Structured3D-trained Delimit3D encoder
    + ScanNet40-trained AGILE3D-style decoder
    -> zero-shot target-dataset evaluation
```

Run the same target-dataset evaluation for the public LitePT encoder arm. This preserves the causal comparison outside the training domain:

```text
public LitePT + its ScanNet40-trained decoder copy
vs.
Delimit3D-256 LitePT + its identically initialized ScanNet40-trained decoder copy
```

The primary zero-shot comparison therefore evaluates the two complete frozen pairs produced on ScanNet40. As an optional secondary portability test, cross-swap the decoder copies (`public features + Delimit3D-trained decoder` and `Delimit3D features + public-trained decoder`) without retraining. Report this separately: it measures feature/decoder compatibility, not the causal benefit of allowing each representation its own matched decoder.

Use the official AGILE3D preprocessed files, scene lists, object/episode manifests, click simulator, and evaluators for each available target dataset. Prioritize S3DIS and KITTI-360 because the AGILE3D release provides benchmark scripts for them. ARKitScenes can be included where the released preprocessing and label/evaluation support are available. ScanNet++ should be evaluated only if an equivalent fixed instance-label and episode manifest exists; it remains an evaluation-only target.

For every target dataset:

1. freeze the ScanNet40-trained decoder and the encoder;
2. load the official target point clouds, colors, coordinates, labels, and episode lists;
3. construct the additional LitePT-required RGBN6 input deterministically if the AGILE3D files provide RGB but not normals;
4. run the unchanged LitePT inference wrapper and the same dec0-to-token mapping policy;
5. use the target dataset's official single-object and/or multi-object click episodes;
6. run adaptive click simulation with no target-dataset training or calibration;
7. compute the official IoU-at-click-budget and NoC metrics;
8. report public-versus-Delimit3D deltas separately for every target dataset.

The AGILE3D files are authoritative for geometry, labels, scene identity, and interaction episodes, but they may not contain every channel required by the LitePT RGBN6 checkpoint. Do not silently substitute RGB-only input. If normals or another LitePT channel must be derived, pin the derivation code, normal convention, coordinate convention, and resulting input hash.

The cross-domain result answers a different question from the ScanNet40 result:

> Does the Structured3D-trained Delimit3D adaptation improve the usefulness of a ScanNet40-trained interactive decoder even when the target scene distribution changes?

It should be reported as zero-shot transfer. It must not be used to tune the decoder, choose a checkpoint, or retroactively redefine the ScanNet40 primary endpoint.
