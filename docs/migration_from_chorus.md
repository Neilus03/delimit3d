# Migration from CHORUS

The new repository was created as a selective extraction from the legacy
checkout at `/cluster/home/nedela/nedela/projects/chorus`, branch
`cleanup/provenance-preserving-2026-09-08`, commit
`0948a906cf71dd9a857dd33eef3dd902e0bad6fd`.

The new private GitHub remote is
`https://github.com/Neilus03/delimit3d`. Its Euler checkout is
`/cluster/home/nedela/nedela/projects/delimit3d`; external artifacts live under
`/cluster/work/igp_psr/nedela/delimit3d`.

Copied into the installable `delimit3d` package:

- multigranular proposal sampling and deterministic coverage contracts;
- RGBN6 input construction and Structured3D source-manifest loading;
- the symmetric region contrastive loss and hierarchy delivery code;
- the LitePT-S* wrapper and strict backbone/checkpoint export;
- compact source preparation entry points for Structured3D and UnSAMv2;
- model-free native-feature extraction and point-prompt metrics;
- focused sampler, training-pack, checkpoint, metric, and wrapper tests.

The old import names and schema identifiers were retained only where they are
part of a validated checkpoint/source contract. Package-facing imports now use
`delimit3d`. The runtime display name is supplied by `DELIMIT3D_NAME`, so the
project can be renamed without editing the scientific code.

Intentionally left in CHORUS:

- the handoff and complete C-family calibration history;
- Mask3D/PointGroup training farms and old decoder experiments;
- presentation builders, UI dependencies, generated figures, logs, and reports;
- raw datasets, checkpoints, UnSAMv2/LitePT subtrees, and large run artifacts.

No legacy file was deleted, reset, cleaned, or moved by this migration.
