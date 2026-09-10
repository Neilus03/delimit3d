#!/bin/bash
#SBATCH --job-name=delimit3d-prompt-decoder
#SBATCH --partition=gpuhe.4h
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/point_decoder_%A_%a.out
#SBATCH --error=logs/point_decoder_%A_%a.err

set -euo pipefail

: "${POINT_DECODER_CONFIG:?set POINT_DECODER_CONFIG to the frozen YAML}"
: "${POINT_DECODER_ARM:?set POINT_DECODER_ARM to public or delimit3d}"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$(dirname "${POINT_DECODER_CONFIG}")/logs"
python "${repo_root}/scripts/evaluation/run_point_decoder.py" \
  --config "${POINT_DECODER_CONFIG}" \
  --arm "${POINT_DECODER_ARM}"
