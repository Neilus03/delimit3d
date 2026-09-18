# Matched interactive plots

Generated from the completed matched-evaluation `aggregate.json` files.
The two plotted arms are intentionally distinct:

- `LitePT (public)`: the original public LitePT checkpoint, frozen for evaluation.
- `LitePT + Delimit3D (256 updates)`: that public checkpoint after 256 Delimit3D adaptation updates, frozen before downstream decoder training.

The performance panels plot aggregate mean IoU and NoC. The paired panels plot adapted-minus-public deltas with the stored 95% paired bootstrap intervals. IoU values are displayed in percent; NoC remains clicks per object.

Click budgets are read directly from the aggregate files: ScanNet40 has 1, 3, 5, 10, and 15 clicks; S3DIS additionally has 20 clicks.

## Source aggregates

- **ScanNet40**: `/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1/aggregate.json`
  - SHA-256: `495d55a93b9b28df2c24022fb77685d6788a8b379550f587dae0ceb5c0626715`
  - MO count: `312`; SO count: `10357`
  - Paired bootstrap samples: `10000`; seed: `None`
- **S3DIS**: `/cluster/work/igp_psr/nedela/delimit3d_s3dis_agile3d_matched_v2/aggregate.json`
  - SHA-256: `232df73f9de8b057f1c8886c0c683008784b817d9f5b32021a4a6897013e4c6f`
  - MO count: `53`; SO count: `2330`
  - Paired bootstrap samples: `10000`; seed: `20260912`
