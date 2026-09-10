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
