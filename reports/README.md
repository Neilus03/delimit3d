# Report sources

The current verified summary is
[the September 18 results ledger](../docs/results/20260918_experiment_summary.md).
Report scripts read external experiment roots; inspect their path arguments or
defaults before running on a different machine. Generated figures, PDFs, meshes,
logs and LaTeX build products are ignored by Git.

| Source | Purpose |
|---|---|
| `plots/build_delimit3d_interactive_plots_20260914.py` | ScanNet40/S3DIS frozen-encoder curves and paired deltas from completed aggregates |
| `plots/delimit3d-interactive-plot-data.json` | Small extracted September 14 plotting input, including source hashes |
| `plots/render_scannet_feature_table.py` | Fixed September 15 ScanNet selectivity table; outputs to ignored `artifacts/scannet_feature_selectivity_table` |
| `plots/render_mesh_feature_selectivity.py` | Mesh feature-selectivity visual with recorded inputs and outputs |
| `build_articulate3d_feature_table.py` | Decoder-free Articulate3D table; ≥10-token primary, ≥5/≥20 sensitivity |
| `latex/delimit3d_interactive_tables_20260914.tex` | Historical completed ScanNet40/S3DIS interactive tables |
| `scratch_baseline_readiness_20260916.md` | Historical readiness audit with superseded proposals explicitly marked |

The mesh renderer was moved from `mesh_visuals/`; the ScanNet table renderer
was moved from `artifacts/scannet_feature_selectivity_table/render.py`. Existing
rendered artifacts were retained. One-off RGBN6 recovery probes and old
`.pre_retry` / `.pre_invalid_visit` source copies were archived outside Git
during the September 18 cleanup; current runners and reproducible report
sources are versioned.
