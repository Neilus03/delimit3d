# Frozen point-decoder comparison — 2026-09-10

This is the first fast-path run of the primary learned endpoint described in
[`RESEARCH.md`](../RESEARCH.md).  The public LitePT-S* checkpoint and the
256-update Delimit3D checkpoint were frozen.  Each arm received a separately
trained copy of the same 45,697-parameter point-conditioned MLP, initialized
from one shared decoder state.  The decoder saw four clicks per object while
training on 50 ScanNet++ train scenes and was evaluated with one click per
object on the same fixed 50-scene validation bundle.

The run produced 777 paired object/query records.  The candidate set was every
validation mesh vertex except the clicked point; the target was the selected
ScanNet++ instance mask.  The encoder was never passed to the optimizer, and
the reports verify `encoder_state_unchanged: true` for both arms.

| metric | public LitePT | Delimit3D | Delimit3D − public |
| --- | ---: | ---: | ---: |
| AP | 0.40848 | 0.49371 | **+0.08523** |
| fixed IoU (threshold 0.50) | 0.15569 | 0.16703 | +0.01134 |
| fixed precision | 0.18870 | 0.19545 | +0.00675 |
| fixed recall | 0.84076 | 0.88847 | +0.04771 |
| oracle IoU (best threshold) | 0.33081 | 0.39835 | **+0.06755** |
| fixed F1 | 0.23170 | 0.24251 | +0.01081 |

For reference, the untrained native cosine readout on the same queries was:

| cosine readout | public LitePT | Delimit3D | Delimit3D − public |
| --- | ---: | ---: | ---: |
| AP | 0.47466 | 0.53649 | **+0.06183** |
| fixed IoU (threshold 0.70) | 0.11920 | 0.16476 | **+0.04556** |
| oracle IoU | 0.36442 | 0.41774 | **+0.05332** |

This secondary readout is not the primary learned-decoder endpoint, but it
shows that the ranking improvement is already present before fitting the
decoder.  The learned decoder improves absolute fixed IoU for both arms and
compresses the public-versus-Delimit3D IoU gap at the chosen threshold.

The improvement is broad for ranking quality: Delimit3D is better on 45/50
paired scenes for AP, with a scene-bootstrap 95% interval of approximately
[0.0675, 0.1051] AP points.  Oracle IoU is better on 44/50 scenes.  Fixed IoU
is better on 30/50 scenes, with a scene-bootstrap interval of approximately
[0.0008, 0.0223] points.

The predeclared learned-decoder gate required at least +0.02 AP, +0.02 fixed
IoU, and positive AP in at least two thirds of scenes.  The gate is **not
passed** because fixed IoU improves by only +0.01134; the AP and paired-scene
conditions pass.  This is evidence for a stronger point-conditioned ranking
signal, not evidence of a large universal mask-quality gain.

## Provenance

- Code commit: `9d58ca4` (`Fix point decoder relation training call`).
- Public checkpoint SHA-256: `86408c6371555aeee2a8eda1184b55a411f9748784c275349149b509b661d518`.
- Delimit3D checkpoint SHA-256: `4717369861cff81fe784efda8cd8a7653de46d35635730f10c893da563050282`.
- Shared decoder initialization tensor SHA-256: `8b7161ce4b52a7c684ccb1294728283710fce17164035cba03144813a9f70e37`.
- Durable artifacts: `/cluster/work/igp_psr/nedela/delimit3d/point_decoder_v1/pf_4090_20260910/`.
- Execution host: `pf-pc69.ethz.ch`, one RTX 4090 per arm; PyTorch `2.4.1+cu124`.
- The local fast path used the same torch scaled-dot-product-attention shim
  for both arms and the PyTorch PointROPE fallback.  The shim is recorded in
  `runtime/flash_attn_stub.py` under the durable artifact root.  This keeps the
  comparison paired, but the run should be repeated with the native kernel
  before treating small absolute differences as final.

## Native confirmation

The same public-versus-256 comparison was subsequently rerun with the native
FlashAttention and PointROPE backends on `pf-pc69.ethz.ch`. The learned-decoder
metrics were unchanged at the decision level (public AP 0.40849, Delimit3D AP
0.49370; public fixed IoU 0.15569, Delimit3D fixed IoU 0.16703), confirming that
the earlier fast-path result was not caused by the fallback attention path.
The native output is retained alongside the 512-update duration test at
`/cluster/work/igp_psr/nedela/delimit3d/point_decoder_native_512_v1/`.
