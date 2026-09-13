#!/bin/sh
set -eu
repo=/tmp/euler_cluster_nedela_rw/home/nedela/nedela/projects/delimit3d
env=/tmp/delimit3d_sonata_pf_v1/env
export PYTHONPATH="$repo/src:$repo/scripts/adaptation"
export CUDA_VISIBLE_DEVICES=0,1
exec "$env/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$repo/scripts/adaptation/run_ptv3_matched_adaptation.py" \
  --config "$repo/configs/adaptation/ptv3_rgbn6_256_matched.example.yaml" \
  --mode train

