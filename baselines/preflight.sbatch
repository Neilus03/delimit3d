#!/bin/bash
#SBATCH --job-name=scratch-baseline-check
#SBATCH --account=es_schin
#SBATCH --partition=gpupr.120h
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint=AMD
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=160G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/scratch_baselines_20260916/logs/check_%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/scratch_baselines_20260916/logs/check_%j.err
set -euo pipefail
source /etc/profile
set +u
module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
MASK_REPO=/cluster/home/nedela/nedela/projects/chorus-baselines
AGILE_REPO=/cluster/home/nedela/nedela/projects/delimit3d-baselines
RUN_ROOT=/cluster/work/igp_psr/nedela/scratch_baselines_20260916
source "$MASK_REPO/student/scripts/extract_litept_local_archives.sh"
export PYTHONPATH="$AGILE_REPO/src:$MASK_REPO/student:$TMPDIR/runtime/pointrope_sm80:$TMPDIR/runtime/LitePT:$TMPDIR/runtime/chorus:${PYTHONPATH:-}"
export CHORUS_SCANNET_METADATA_ROOT=/cluster/work/igp_psr/nedela/scannet_metadata
export LITEPT_MASK3D_PARITY_DECODER_TEMPLATE="$RUN_ROOT/mask3d_decoder_random_seed42.pt"
PYTHON="$VIRTUAL_ENV/bin/python"
cd "$MASK_REPO/student"
if [[ ! -f "$LITEPT_MASK3D_PARITY_DECODER_TEMPLATE" ]]; then
  "$PYTHON" scripts/create_litept_mask3d_decoder_template.py --seed 42 --num-classes 2 --eval-queries 100 --output "$LITEPT_MASK3D_PARITY_DECODER_TEMPLATE"
fi
"$PYTHON" "$AGILE_REPO/baselines/check_contract.py"
"$PYTHON" "$AGILE_REPO/scripts/evaluation/run_scannet40_scratch.py" --smoke --output "$RUN_ROOT/agile3d_smoke"
"$PYTHON" "$AGILE_REPO/scripts/evaluation/run_scannet40_scratch.py" --smoke --output "$RUN_ROOT/agile3d_smoke"
export CHORUS_OUTPUT_ROOT="$RUN_ROOT/mask3d_smoke"
"$PYTHON" scripts/run_litept_mask3d.py --config "$RUN_ROOT/mask3d_smoke.yaml"
BASELINE_STOP_AFTER_EPOCHS=2 "$PYTHON" scripts/run_litept_mask3d.py --config "$RUN_ROOT/mask3d_smoke.yaml" --resume "$RUN_ROOT/mask3d_smoke/litept_mask3d_ca_scratch_600_seed42/checkpoints/latest.pt"
"$PYTHON" "$AGILE_REPO/baselines/check_results.py"
