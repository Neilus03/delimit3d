# Research scope

Delimit3D tests one narrow hypothesis:

> Multigranular, category-agnostic 2D pseudomasks can provide useful negative
> structure when briefly adapting a pretrained 3D scene encoder, making a point
> select a more spatially specific object or part in a real scan.

The adaptation is label-free with respect to 3D annotations, but it is not
teacher-free: UnSAMv2 supplies the 2D pseudomasks while building the synthetic
source packs. The teacher is absent at downstream inference.

The main endpoint is point-conditioned selection. A fixed native-feature
readout is a representation diagnostic. The decisive next test is a fresh,
class-agnostic point decoder trained identically on frozen public and adapted
features, followed by one-click and correction-click evaluation. Object and
part results must be reported separately. Mask3D and PointGroup remain
historical transfer probes rather than the central success criterion.

The old CHORUS checkout remains the scientific archive for the rejected
scratch-to-Mask3D transfer and the complete calibration history. See
[`../manifests/legacy_evidence.json`](../manifests/legacy_evidence.json).

## Current evidence

The matched frozen-encoder point-decoder endpoint is now the primary decision
test. The 256-update Delimit3D snapshot improved AP over public LitePT by
0.0852 on 50 ScanNet++ validation scenes and 777 one-click queries, but its
fixed-IoU gain was only 0.0113 and did not pass the predeclared gate. Extending
the identical adaptation to 512 updates reversed the result: AP was 0.1886
versus 0.4085 for public LitePT and fixed IoU was 0.0510 versus 0.1557. The
512 snapshot also lowered the frozen cosine readout and feature spread. Thus,
longer adaptation is not a justified next step by itself. The independent
training-seed repeat (training seed `20260911`, exact seed-42 initialization
and fixed evaluation pack) reached a similar native fixed-panel cosine gap of
`0.3502`, but its matched fresh point decoder scored AP `0.3220` and fixed IoU
`0.0709`, versus public LitePT AP `0.4085` and fixed IoU `0.1557`. Paired-scene
AP was positive on only `6/50` scenes and fixed IoU on `1/50`; the learned
decoder integrity checks passed and the encoder remained frozen. The
predeclared reproducibility gate failed. The sampled contrastive loss still
decreased from `6.7275` at update 1 to `4.8457` at update 256, showing that
objective optimization alone did not predict transfer.

The combined evidence is now that the apparent seed-42 256-update transfer
gain is not reproducible under the same learned-decoder protocol, while the
512-update endpoint is clearly harmful. Fixed cosine separation and ordinary
upstream loss are therefore representation diagnostics rather than sufficient
transfer criteria. We should stop scaling this adaptation recipe and only
launch another run after specifying a new mechanism-level hypothesis and the
same matched point-conditioned gate. Mask3D/PointGroup numbers remain
historical probes; the current scientific scope is point-conditioned,
class-agnostic object/part selection.

The full seed-repeat provenance and artifact hashes are recorded in
[`results/adaptation_256_seed20260911_native_4090.md`](results/adaptation_256_seed20260911_native_4090.md).
