#!/usr/bin/env bash
#SBATCH --job-name=ag3d-litept-rgb3
#SBATCH --partition=gpuhe.120h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=160G
#SBATCH --time=5-00:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/%x-%j.err

# The default request is one RTX 4090.  For an RTX 3090, submit the same
# script with: sbatch --gpus=rtx_3090:1 agile3d_litept_sstar_rgb3.sbatch

set -euo pipefail

repo=/cluster/home/nedela/nedela/projects/delimit3d
python_bin=${DELIMIT3D_PYTHON:-/cluster/work/igp_psr/nedela/litept-env/bin/python}
config=${AGILE3D_LITEPT_CONFIG:-${repo}/configs/training/agile3d_litept_sstar_rgb3_scannet40.yaml}
script=${repo}/scripts/training/run_agile3d_litept_sstar_rgb3.py
mode=${1:-preflight}

test -x "${python_bin}"
test -s "${config}"
test -s "${script}"

gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader)
printf '%s\n' "${gpu_report}"
case "${gpu_report}" in
  *"RTX 3090"*|*"RTX 4090"*) ;;
  *) echo "Expected one RTX 3090 or RTX 4090, got: ${gpu_report}" >&2; exit 2 ;;
esac

export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

source "${repo}/scripts/training/stage_agile3d_litept_local.sh"
runner_args=(--config "${config}" --mode "${mode}")
if [[ -n "${AGILE3D_LITEPT_RESUME:-}" ]]; then
  cp "${AGILE3D_LITEPT_RESUME}" "${AGILE3D_LOCAL_ROOT}/resume.pth"
  runner_args+=(--resume "${AGILE3D_LOCAL_ROOT}/resume.pth")
fi
if [[ -n "${AGILE3D_LITEPT_CHECKPOINT:-}" ]]; then
  cp "${AGILE3D_LITEPT_CHECKPOINT}" "${AGILE3D_LOCAL_ROOT}/evaluate.pth"
  runner_args+=(--checkpoint "${AGILE3D_LOCAL_ROOT}/evaluate.pth")
fi

echo "Repo: ${repo}"
echo "Config: ${config}"
echo "Mode: ${mode}"
echo "Started: $(date -Is)"

exec "${python_bin}" "${script}" "${runner_args[@]}"
