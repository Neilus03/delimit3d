#!/bin/bash
set -euo pipefail
RUN=/home/nedela/scratch_baselines_20260916
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN/source/src:$RUN/runtime:$RUN/litept"
export LITEPT_ROOT="$RUN/litept"
unset POINTROPE_FORCE_TORCH
exec /scratch/nedela/delimit3d_scannet40_runtime_v1/bin/python "$RUN/source/scripts/evaluation/run_scannet40_scratch.py" --manifest "$RUN/runtime/pf_selection_manifest.json" --grid-size .05 --benchmark-batches 20 --output "$RUN/benchmark_5cm"
