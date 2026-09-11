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

