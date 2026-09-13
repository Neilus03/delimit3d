#!/bin/bash
# One RTX 4090 joint Delimit3D LitePT + AGILE3D ScanNet40 training run.
# Submit with JOINT_CONFIG, JOINT_SOURCE_ARCHIVE, JOINT_SOURCE_SHA256 and
# JOINT_PYTHON set to the frozen resolved paths.
#SBATCH --job-name=delimit3d-s40-joint
#SBATCH --partition=gpuhe.120h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_retry2/slurm-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_joint_v1_retry2/slurm-%j.err

set -euo pipefail

: "${JOINT_CONFIG:?set JOINT_CONFIG to the frozen resolved YAML}"
: "${JOINT_SOURCE_ARCHIVE:?set JOINT_SOURCE_ARCHIVE to the frozen source archive}"
: "${JOINT_SOURCE_SHA256:?set JOINT_SOURCE_SHA256 to the frozen source archive hash}"
: "${JOINT_PYTHON:?set JOINT_PYTHON to the exact LitePT environment Python}"

joint_root="$(dirname "$(dirname "${JOINT_CONFIG}")")"
printf '%s  %s\n' "${JOINT_SOURCE_SHA256}" "${JOINT_SOURCE_ARCHIVE}" | sha256sum -c -
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

code="${joint_root}/frozen_execution_${SLURM_JOB_ID}"
mkdir -p "${code}"
tar -xf "${JOINT_SOURCE_ARCHIVE}" -C "${code}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONPATH="${code}/src${PYTHONPATH:+:${PYTHONPATH}}"

command=("${JOINT_PYTHON}" "${code}/scripts/evaluation/run_scannet40_joint.py" --config "${JOINT_CONFIG}" --mode train)
if [[ -n "${JOINT_RESUME:-}" ]]; then
  command+=(--resume "${JOINT_RESUME}")
fi
exec "${command[@]}"
