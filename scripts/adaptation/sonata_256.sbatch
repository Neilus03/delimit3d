#!/bin/bash
#SBATCH --job-name=sonata256
#SBATCH --partition=gpuhe.120h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/%x-%j.err
set -euo pipefail

repo=/cluster/home/nedela/nedela/projects/delimit3d
python=/cluster/work/igp_psr/nedela/litept-env/bin/python
config=${SONATA_CONFIG:-${repo}/configs/adaptation/sonata_rgbn6_256.example.yaml}
mode=${1:-preflight}
export PYTHONPATH="${repo}/src:/cluster/work/igp_psr/nedela/sonata:${PYTHONPATH:-}"

# Never silently run on an unapproved GPU type.
gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader)
printf '%s\n' "${gpu_report}"
case "${gpu_report}" in
  *"RTX 3090"*|*"RTX 4090"*|*"A100"*) ;;
  *) echo "No approved 3090/4090/A100 GPU allocated" >&2; exit 2 ;;
esac
case "${gpu_report}" in *"2080"*) echo "RTX 2080 Ti is forbidden" >&2; exit 2;; esac

# The source/checkpoint and data manifest are external and hashed by prepare.
cd "${repo}"
exec "${python}" scripts/adaptation/run_sonata_adaptation.py --config "${config}" --mode "${mode}"
