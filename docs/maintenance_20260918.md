# Repository maintenance — 18 September 2026

The README and research/protocol documents now follow the supervisor-directed
scratch versus pretrained comparison. `CURRENT.md` is the dated starting point;
`docs/results/evidence_20260918.json` records completed metrics and source hashes.
Historical experiment plans are retained with supersession notices.

## Preserved work

The pending LitePT prequantized-input path, RGB3 scratch runner, recovery/staging
fixes, NumPy utility backend, matched Sonata/PTv3 work, Articulate3D evaluator,
feature diagnostics and report sources were reviewed for inclusion. Their
existing experiment settings were retained. This cleanup starts no training,
changes no scheduler job, and does not merge active experiment worktrees.

Before cleanup, all nonignored modified/untracked files and the tracked patch
were saved under:

`/cluster/work/igp_psr/nedela/delimit3d_repo_cleanup/20260918T224609`

- `pre_cleanup_changes.tar.gz`: original pending source/report/config files.
- `tracked.patch` and `status.txt`: original Git diff and worktree inventory.
- `archived/`: dated RGBN6 recovery probe and two `.pre_*` runner copies.

Rendered outputs remain in place and ignored. Table/mesh rendering source was
moved into `reports/plots/`. LaTeX auxiliary files are now ignored. No dataset,
checkpoint, result directory or other worktree was deleted or reset.

## Source branches

The existing frozen ScanNet40, joint ScanNet40, S3DIS and scratch-baseline
branches are preserved separately on GitHub so the experiment map can link to
real source snapshots. Main integrates the previously pending work in this
checkout and the revised documentation. External CHORUS pretraining/Mask3D
source snapshots remain external dependencies, not newly vendored code.

## Verification

- Python AST parsing, JSON parsing and shell syntax checks cover versioned and
  newly included source/config/report files.
- Relative links in the revised documentation resolve locally.
- The evidence extractor checks frozen-evaluation identity sets and complete
  joint/part counts; recorded source hashes and historical plotting hashes are
  rechecked against external files.
- CPU suite: **131 passed, 2 CUDA-only skips**, in an isolated source copy on
  pf using the existing LitePT runtime, with CUDA disabled and one CPU thread.
  Two stale CHORUS monkeypatch targets were repaired to reference
  `delimit3d.training.adaptation`; the subsequent complete suite passed.
- Parsed **113 Python files, 22 YAML configs, 15 shell/Slurm scripts**, and
  all included JSON documents. Revised local documentation links resolve.
- Verified **16 evidence-source hashes** and both historical plot aggregate
  hashes. Before whitespace cleanup, all four RGB3 source-file hashes matched
  the running job's manifest. The utility backend subsequently lost one trailing
  blank line; runner, decoder and LitePT wrapper remain byte-identical. The
  original utility bytes are preserved in the cleanup archive.
- The first Euler test attempt timed out during environment startup; it did
  not produce test results. The successful CPU run used a temporary copy and
  did not modify any live training environment.
- GPU training and full dataset evaluation were not rerun for maintenance.
