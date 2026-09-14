#!/bin/bash
# One RTX 4090 final evaluation of the all-learnable joint ScanNet40 arm.
# The evaluator runs the official MO and SO panels sequentially and caches
# the trained encoder's dynamic validation features by scene.
#SBATCH --job-name=delimit3d-s40-joint-eval
#SBATCH --partition=gpuhe.120h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_eval3/slurm-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_eval3/slurm-%j.err

set -euo pipefail

: "${JOINT_EVAL_CONFIG:?set JOINT_EVAL_CONFIG to the frozen resolved YAML}"
: "${JOINT_EVAL_SOURCE_ARCHIVE:?set JOINT_EVAL_SOURCE_ARCHIVE to the frozen source archive}"
: "${JOINT_EVAL_SOURCE_SHA256:?set JOINT_EVAL_SOURCE_SHA256 to the frozen source archive hash}"
: "${JOINT_EVAL_PYTHON:?set JOINT_EVAL_PYTHON to the exact LitePT environment Python}"

eval_root="$(dirname "$(dirname "${JOINT_EVAL_CONFIG}")")"
printf '%s  %s\n' "${JOINT_EVAL_SOURCE_SHA256}" "${JOINT_EVAL_SOURCE_ARCHIVE}" | sha256sum -c -
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

code="${eval_root}/frozen_execution_${SLURM_JOB_ID}"
mkdir -p "${code}"
tar -xf "${JOINT_EVAL_SOURCE_ARCHIVE}" -C "${code}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONPATH="${code}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${JOINT_EVAL_PYTHON}" "${code}/scripts/evaluation/run_scannet40_joint_eval.py" \
  --config "${JOINT_EVAL_CONFIG}" --mode evaluate
