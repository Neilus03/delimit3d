# Third-party notices

Delimit3D is a research extraction from an internal CHORUS/LitePT workspace.
The following external projects and derived components are required for the
runtime or are represented by adapters in this repository:

- **LitePT**: the external LitePT checkout is supplied through
  `DELIMIT3D_LITEPT_ROOT`; it is not vendored here. See the public
  [LitePT repository](https://github.com/Neilus03/LitePT) for its upstream
  license and notices.
- **UnSAMv2 / SAM2**: the 2D teacher checkout and checkpoint remain external
  under `DELIMIT3D_UNSAM_ROOT`. The source preparation adapter imports the
  teacher at runtime and does not copy its weights.
- **PyTorch, NumPy, SciPy, OpenCV, Pillow, imageio, plyfile**: installed as
  environment dependencies and governed by their respective licenses.
- **Mask3D-derived code**: Mask3D decoder code is intentionally excluded from
  the core Delimit3D package. The historical CHORUS repository contains the
  corresponding notice for old transfer experiments.
- **AGILE3D-compatible evaluation**: the core repository currently implements
  the fixed point-prompted readout protocol. Any future interactive decoder
  code derived from AGILE3D will carry a separate, explicit notice.

Before public release, review every copied file and add the exact upstream
license text required by LitePT, UnSAMv2/SAM2, and any future decoder/evaluator
implementation.


- **AGILE3D-compatible multi-object decoder and click protocol**: the
  Delimit3D interactive evaluator adapts the query decoder, positional/click
  encoding, auxiliary-loss structure, and simulated-correction policy from
  AGILE3D at revision
  b73638da41edbabe52a1b578d52ddeb8fa552173.
  Upstream repository: https://github.com/ywyue/AGILE3D
  Upstream license: MIT (copyright Yuanwen Yue and ETH Zurich). The
  adaptation is implemented natively in PyTorch because MinkowskiEngine is
  not part of the Delimit3D runtime; LitePT provides the frozen token
  interface.


## Sonata / Point Transformer V3

The Sonata adaptation path uses the public encoder-only Sonata implementation
from `facebookresearch/sonata`, source revision
`18c09ff8d713494f78a8213792262b910977a65d` (retrieved 2026-09-13). The source
code is licensed under the Apache License 2.0; the public pretrained weights
are distributed by Meta under CC-BY-NC 4.0. The external source checkout and
weights remain under `/cluster/work/igp_psr/nedela` and are not vendored into
this repository. The resolved checkpoint SHA-256 is recorded in the adaptation
YAML and run provenance. See:

- https://github.com/facebookresearch/sonata
- https://github.com/facebookresearch/sonata/blob/main/LICENSE
- https://huggingface.co/facebook/sonata

## Point Transformer V3

The public PTv3 adaptation path uses the detached Point Transformer V3 model
from Pointcept at source revision
`3229e9b7de1770c8ad17c316f8e349982de509f8`. The external source checkout and
public ScanNet checkpoint remain outside this repository under the configured
artifact root; the checkpoint SHA-256 and source revision are recorded in each
run's provenance. The upstream implementation is released under the MIT
license. See:

- https://github.com/Pointcept/PointTransformerV3
- https://huggingface.co/Pointcept/PointTransformerV3
- https://github.com/Pointcept/PointTransformerV3/blob/main/LICENSE
