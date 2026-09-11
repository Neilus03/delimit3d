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
test. The valid 256-update public-LitePT posttraining snapshot improved AP
over public LitePT by `0.0852` on 50 ScanNet++ validation scenes and 777
one-click queries, but its fixed-IoU gain was only `0.0113` and did not pass the
predeclared gate. A later 512-update run and the 20260911 repeat both recorded
`public_checkpoint_used=false`: they started from the scratch
`initial_representative_rgbn6_seed42.pt` artifact and therefore are not valid
public-initialized duration or seed comparisons. The repeat scored AP `0.3220`
and fixed IoU `0.0709` against public AP `0.4085` and fixed IoU `0.1557`, but
that negative aggregate is retained only as a protocol audit. It used backbone
LR `0.001`, whereas the valid public posttraining run used `0.0001`. The
sampled contrastive loss still decreased from `6.7275` at update 1 to `4.8457`
at update 256, showing that objective optimization alone did not predict
transfer, but this run cannot test public-posttraining reproducibility.

The combined evidence is not yet enough to decide whether public-LitePT
posttraining is seed-stable or whether longer public-initialized adaptation is
harmful. Fixed cosine separation and ordinary upstream loss remain
representation diagnostics rather than sufficient transfer criteria. Before
changing the mechanism, we need one correctly public-initialized repeat with
the original `0.0001` backbone LR and the same matched point-conditioned gate.
Mask3D/PointGroup numbers remain historical probes; the current scientific
scope is point-conditioned, class-agnostic object/part selection.

The full seed-repeat provenance and artifact hashes are recorded in
[`results/adaptation_256_seed20260911_native_4090.md`](results/adaptation_256_seed20260911_native_4090.md).
