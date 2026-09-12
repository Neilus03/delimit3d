#!/bin/bash
#SBATCH --job-name=delimit3d-cross-eval
#SBATCH --partition=gpuhe.24h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/%x-%j.err
set -euo pipefail
: "${CONFIG:?set frozen resolved config path}"
: "${ARM:?set public or delimit3d}"
: "${PANEL:?set MO or SO}"
PY=/cluster/work/igp_psr/nedela/litept-env/bin/python
cd /cluster/home/nedela/nedela/projects/delimit3d-s3dis-kitti-matched
export PYTHONPATH="$PWD/scripts/evaluation:$PWD/src"
exec "$PY" scripts/evaluation/run_cross_dataset_matched.py --config "$CONFIG" --mode evaluate --arm "$ARM" --panel "$PANEL"
