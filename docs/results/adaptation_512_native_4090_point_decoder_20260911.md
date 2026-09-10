# 512-update native Delimit3D adaptation — downstream result

This run tests whether simply extending the short, label-free Delimit3D
adaptation from 256 to 512 optimizer updates improves the transfer signal. The
recipe, source cells (`2d/g02`, `2d/g05`, `2d/g08`), RGBN6 contract, sampling,
and SyncBN mechanics were held fixed. The 512-update continuation was trained
on two RTX 4090 GPUs with the native FlashAttention and PointROPE backends. A
checkpoint at update 256 and the final checkpoint at update 512 were retained
from the same run.

The downstream comparison uses the same frozen public LitePT encoder, the
256-update snapshot, or the 512-update snapshot, respectively. Each arm gets a
separately optimized copy of the same fresh 45,697-parameter point-conditioned
decoder, cloned from one initialization tensor. The decoder trains on 50
ScanNet++ scenes with four prompts per object and is evaluated on 50 disjoint
validation scenes with one click per object. There are 777 paired object/query
records. The encoder is excluded from the optimizer; the reports verify
`encoder_state_unchanged: true` for every arm.

## Result

| metric | public LitePT | Delimit3D 256 updates | Delimit3D 512 updates | 512 − public |
| --- | ---: | ---: | ---: | ---: |
| AP | 0.40849 | **0.49370** | 0.18858 | **−0.21991** |
| fixed IoU (threshold 0.50) | 0.15569 | **0.16703** | 0.05102 | **−0.10467** |
| fixed precision | 0.18870 | 0.19546 | 0.05667 | −0.13203 |
| fixed recall | 0.84077 | **0.88847** | 0.80420 | −0.03657 |
| oracle IoU (best threshold) | 0.33078 | **0.39837** | 0.16184 | **−0.16895** |
| fixed F1 | 0.23169 | **0.24251** | 0.08653 | −0.14516 |

The 512-update arm is lower than public LitePT on every headline metric. It is
also lower than the 256-update snapshot on every metric: AP decreases by
0.30512, fixed IoU by 0.11601, oracle IoU by 0.23653, and fixed F1 by 0.15598.
All 50 paired scenes are negative for the 512-update versus public AP,
fixed-IoU, and oracle-IoU comparisons. The predeclared learned-decoder gate
(at least +0.02 AP, +0.02 fixed IoU, and positive AP in at least two thirds of
scenes) is therefore **not passed**. The earlier 256-update run remains a
narrow ranking improvement whose fixed-IoU gain did not clear that gate.

For the secondary, untrained native cosine readout on the same validation
queries:

| cosine readout | public LitePT | Delimit3D 256 updates | Delimit3D 512 updates |
| --- | ---: | ---: | ---: |
| AP | 0.47466 | **0.53649** | 0.27533 |
| fixed IoU (threshold 0.70) | 0.11921 | **0.16476** | 0.03221 |
| oracle IoU | 0.36438 | **0.41773** | 0.21473 |

This agrees with the learned-decoder result: the 512 checkpoint is not merely
harder for the decoder to fit. Its frozen feature ranking is already worse on
this point-conditioned object-selection endpoint.

## Upstream trajectory and health

The fixed Structured3D panel was evaluated in the native, token-pure `dec0`
space. Higher cosine gap is better for the pretraining objective.

| checkpoint | fixed macro cosine gap | native selection macro | effective rank | minimum feature std |
| --- | ---: | ---: | ---: | ---: |
| public initialization (update 0) | 0.17885 | 0.17885 | 18.959 | 0.14432 |
| Delimit3D update 256 | **0.34844** | **0.34663** | 6.158 | 0.03247 |
| Delimit3D update 512 | 0.25552 | 0.28784 | 6.040 | 0.02354 |

The 512 run passed all mechanical and finite-metric checks, reached exactly 512
updates, and preserved the RGBN6 and sampling contracts. Its broad selector
thresholds still mark the checkpoint as numerically usable (`effective_rank`
and feature standard deviations are nonzero), but the trajectory shows clear
concentration and loss of feature spread relative to update 256. This is a
health warning and evidence of over-adaptation for this recipe; it is not by
itself a claim of rank-one collapse.

The adaptation's recorded fixed-pack cells all had full positive-pair yield.
The auxiliary conflict budget, clipping telemetry, code/data manifests,
initialization hash, and world-size checks passed. The audit checkpoint uses
the same raw-checkpoint convention as the 256 comparison; no hidden BN
recalibration sidecar was introduced.

## Decision

This experiment answers the duration question negatively. Extending the same
objective and schedule from 256 to 512 updates does not provide a stronger
initialization; it destroys the positive 256-update signal on the matched
point-conditioned endpoint. Do not scale this recipe by duration alone and do
not use the 512 checkpoint for a production or downstream claim.

The useful next scientific question is therefore checkpoint selection and
regularization around the narrow 256-update window: determine whether the
positive signal is reproducible and stable across seeds, and whether a
constrained continuation can retain the 256 representation geometry without
the 512 feature concentration. No new GPU run is launched by this result.

## Provenance

- Adaptation output: `/cluster/work/igp_psr/nedela/delimit3d/adaptation_512_native_20260910/`.
- 256 snapshot: `checkpoints/audit_e0001_u00000256.pt`, SHA-256
  `c011000724fd3d7816972b545fb775966a339517d85697810fd43a429ff63dcc`.
- 512 snapshot: `checkpoints/audit_e0002_u00000512.pt`, SHA-256
  `e6d95cc240fefcf576ecbc7c64129a2c7108d7b2735d8c9a8429286044bbe6f0`.
- Adaptation `run_summary.json` SHA-256
  `93fec25d87aae466aef02eed84b091d63d287d7972cb84a894c982ea27cc2c59`.
- Adaptation code-manifest SHA-256
  `6f839bf5777edb7820128340b3504e1c7835df48bcf0d450b99b15c0330a98f5`.
- Execution-provenance SHA-256
  `33d4b04b2e4b22c601c3ae42186a1e652b00fe90183d95867e1dd35d2f77e8f2`.
- Matched decoder output: `/cluster/work/igp_psr/nedela/delimit3d/point_decoder_native_512_v1/`.
- Decoder aggregate SHA-256
  `226a5a432691e20b6fb164b83bc48c6e3e0b1cb66b9edbaea0683589f4ac008c`.
- Delimit3D decoder report SHA-256
  `41de535fcf0df9e47019f5367f617f5456e23e604043d081bff781bd5e161589`.
- Final decoder tensor SHA-256
  `4bdf6a309e492f3cdc898f2a9cbc60640c1a28390ca8e634deff3d97010cd075`.
- Native execution host: `pf-pc69.ethz.ch`, two RTX 4090 GPUs for adaptation
  and one RTX 4090 for the decoder evaluation; PyTorch `2.4.1+cu124`.
- Source repository commit: `95b898a`.
