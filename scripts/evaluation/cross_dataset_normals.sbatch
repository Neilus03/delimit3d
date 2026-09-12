#!/bin/bash
#SBATCH --job-name=delimit3d-cross-norm
#SBATCH --partition=normal.24h
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/%x-%j.err
set -euo pipefail
: "${DATASET:?set DATASET=S3DIS or KITTI360}"
: "${OFFICIAL_ROOT:?set official data root}"
: "${OUTPUT_ROOT:?set external output root}"
PY=/cluster/work/igp_psr/nedela/litept-env/bin/python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /cluster/home/nedela/nedela/projects/delimit3d-s3dis-kitti-matched
exec "$PY" scripts/evaluation/generate_cross_dataset_normals.py --dataset "$DATASET" --official-root "$OFFICIAL_ROOT" --output-root "$OUTPUT_ROOT" --scope "${SCOPE:-all}" --knn "${KNN:-30}" ${MAX_ITEMS:+--max-items "$MAX_ITEMS"}
