#!/usr/bin/env bash
#SBATCH --job-name=art3d-agile-parts
#SBATCH --partition=gpuhe.120h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
# One scene is loaded at a time; 16 GiB total is sufficient and avoids the
# per-user QOS memory cap that held the first submission in pending.
#SBATCH --mem-per-cpu=2G
#SBATCH --time=5-00:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/delimit3d_articulate3d_agile3d_parts_eval_v1/logs/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/delimit3d_articulate3d_agile3d_parts_eval_v1/logs/%x-%j.err

set -euo pipefail

repo=/cluster/home/nedela/nedela/projects/delimit3d
python_bin=/cluster/work/igp_psr/nedela/litept-env/bin/python
config=${repo}/configs/evaluation/articulate3d_agile3d_parts_v1.yaml
script=${repo}/scripts/evaluation/run_articulate3d_parts.py

gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader)
case "${gpu_report}" in
  *"RTX 4090"*) ;;
  *) echo "Expected an RTX 4090, got: ${gpu_report}" >&2; exit 2 ;;
esac

test -s "${config}"
test -s "${script}"
export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "GPU: ${gpu_report}"
echo "Config: ${config}"
echo "Started: $(date -Is)"
exec "${python_bin}" "${script}" --config "${config}" --mode evaluate --arm all
