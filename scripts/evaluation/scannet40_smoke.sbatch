#!/usr/bin/env bash
#SBATCH --job-name=delimit3d-s40-smoke
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${DELIMIT3D_SMOKE_ROOT:?}"
: "${DELIMIT3D_SOURCE_ARCHIVE:?}"
: "${DELIMIT3D_SOURCE_SHA256:?}"
: "${DELIMIT3D_CONFIG:?}"
: "${DELIMIT3D_PYTHON:?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
printf '%s  %s\n' "$DELIMIT3D_SOURCE_SHA256" "$DELIMIT3D_SOURCE_ARCHIVE" | sha256sum -c -
code="$DELIMIT3D_SMOKE_ROOT/frozen_execution_${SLURM_JOB_ID}"
mkdir -p "$code"
tar -xf "$DELIMIT3D_SOURCE_ARCHIVE" -C "$code"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
sock="delimit3d-smoke-$SLURM_JOB_ID"
tmux -L "$sock" new-session -d -s delimit3d-scannet40-smoke -c "$code" "bash scripts/evaluation/run_scannet40_smoke.sh > '$DELIMIT3D_SMOKE_ROOT/smoke_job_${SLURM_JOB_ID}.log' 2>&1; echo \$? > '$DELIMIT3D_SMOKE_ROOT/smoke_job_${SLURM_JOB_ID}.exit'"
while tmux -L "$sock" has-session -t delimit3d-scannet40-smoke 2>/dev/null; do sleep 15; done
exit "$(cat "$DELIMIT3D_SMOKE_ROOT/smoke_job_${SLURM_JOB_ID}.exit")"
