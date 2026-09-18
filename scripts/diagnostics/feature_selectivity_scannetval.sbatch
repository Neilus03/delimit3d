#!/usr/bin/env bash
#SBATCH --job-name=d3d-feat-select
#SBATCH --partition=bigmem.4h
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/delimit3d_feature_selectivity_scannetval_20260915/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/delimit3d_feature_selectivity_scannetval_20260915/%x-%j.err
set -euo pipefail

repo=/cluster/home/nedela/nedela/projects/delimit3d
python_bin=/cluster/work/igp_psr/nedela/litept-env/bin/python
run_root=/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1
output_root=/cluster/work/igp_psr/nedela/delimit3d_feature_selectivity_scannetval_20260915
mode=${1:-pilot}

export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${output_root}"
echo "hostname=$(hostname)"
echo "started=$(date -Is)"
echo "mode=${mode}"
echo "python=${python_bin}"

if [[ "${mode}" == "pilot_v3" ]]; then
  exec "${python_bin}" "${repo}/scripts/diagnostics/feature_selectivity_scannetval.py" \
    --experiment-root "${run_root}" \
    --output "${output_root}/pilot_v3_scene0568_00" \
    --scene scene0568_00 \
    --visuals \
    --bootstrap 1000
fi

if [[ "${mode}" == "pilot" || "${mode}" == "pilot_v2" ]]; then
  exec "${python_bin}" "${repo}/scripts/diagnostics/feature_selectivity_scannetval.py" \
    --experiment-root "${run_root}" \
    --output "${output_root}/pilot_v2_scene0568_00" \
    --scene scene0568_00 \
    --visuals \
    --bootstrap 1000
fi

if [[ "${mode}" == "all" ]]; then
  exec "${python_bin}" "${repo}/scripts/diagnostics/feature_selectivity_scannetval.py" \
    --experiment-root "${run_root}" \
    --output "${output_root}/all_val_scenes" \
    --visuals \
    --bootstrap 10000
fi

echo "unknown mode: ${mode}" >&2
exit 2
