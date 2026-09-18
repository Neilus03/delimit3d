# Interactive-table export provenance

This export contains only the two requested method arms: `LitePT (public)`
(`public` in the evaluation artifacts) and `LitePT + Delimit3D (256 updates)`
(`delimit3d`). The values in the PDFs are percentages and are rounded to one
decimal place to match the attached table style.

The evaluation scope is the completed matched LitePT comparison under the
AGILE3D metric protocol; it is not a literal reproduction of the AGILE3D model.
The baseline is the original public LitePT checkpoint, frozen for downstream
evaluation. The second arm starts from that public checkpoint, applies 256
Delimit3D adaptation updates, and freezes the resulting adapted LitePT encoder.
Both frozen representations receive separate downstream decoders cloned from
the same byte-identical initialization. The KITTI-360 rows are intentionally
omitted because that evaluation is not yet available.

## Values used

### Table 2: single-object segmentation (SO)

| Dataset | Arm | IoU@1 | IoU@2 | IoU@3 | Units |
|---|---|---:|---:|---:|---:|
| ScanNet40 | LitePT (public) | 24.142721 | 31.131569 | 36.019545 | 10,357 objects |
| ScanNet40 | LitePT + Delimit3D (256 updates) | 26.084001 | 35.179984 | 41.297797 | 10,357 objects |
| S3DIS | LitePT (public) | 15.309147 | 23.745934 | 29.679630 | 2,330 objects |
| S3DIS | LitePT + Delimit3D (256 updates) | 20.641038 | 30.573673 | 37.084273 | 2,330 objects |

`IoU@2` is the arithmetic mean of the exact `total_clicks = 2` rows in each
arm's official SO result stream. IoU@1 and IoU@3 are the corresponding exact
click-budget means in the same streams and are also present in the aggregate
reports.

### Table 4: multi-object segmentation (MO)

| Train → Eval | Arm | IoU@5 | IoU@10 | IoU@15 | NoC@80 | NoC@85 | NoC@90 | Units |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ScanNet40 → ScanNet40 | LitePT (public) | 59.1855 | 64.2797 | 66.9670 | 15.6816 | 17.3669 | 18.6359 | 312 scenes |
| ScanNet40 → ScanNet40 | LitePT + Delimit3D (256 updates) | 64.1586 | 69.6808 | 72.9917 | 14.1106 | 16.5120 | 18.0263 | 312 scenes |
| ScanNet40 → S3DIS-A5 | LitePT (public) | 45.8155 | 55.3895 | 61.6343 | 19.1332 | 20.0000 | 20.0000 | 53 scenes |
| ScanNet40 → S3DIS-A5 | LitePT + Delimit3D (256 updates) | 52.3761 | 63.6686 | 68.1240 | 17.5001 | 18.8570 | 19.5461 | 53 scenes |

## Source artifacts

### ScanNet40

- Experiment root: `/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1`
- Final provenance: `lead_audit/scannet40_v1_final_provenance.json`
- Aggregate: `aggregate.json`, SHA-256 `495d55a93b9b28df2c24022fb77685d6788a8b379550f587dae0ceb5c0626715`
- Source commit: `4a1888213f0507eae86b51d16df76b7d8809ba7b`
- SO result streams: `public/evaluation/SO/official_results.csv` and `delimit3d/evaluation/SO/official_results.csv`
- MO result streams: `public/evaluation/MO/official_results.csv` and `delimit3d/evaluation/MO/official_results.csv`

### S3DIS

- Experiment root: `/cluster/work/igp_psr/nedela/delimit3d_s3dis_agile3d_matched_v2`
- Final provenance: `lead_audit/s3dis_v2_final_provenance.json`
- Aggregate: `aggregate.json`, SHA-256 `232df73f9de8b057f1c8886c0c683008784b817d9f5b32021a4a6897013e4c6f`
- Source commit: `286327745118bdc1bb6b6d2f199097f9cf19bec9`
- SO result streams: `public/evaluation/SO/official_results.csv` and `delimit3d/evaluation/SO/official_results.csv`
- MO result streams: `public/evaluation/MO/official_results.csv` and `delimit3d/evaluation/MO/official_results.csv`

The table source is `delimit3d_interactive_tables_20260914.tex`. Compile it
with TeX Live 2024; defining `OnlyTableTwo` or `OnlyTableFour` on the command
line emits the corresponding single-table PDF.
