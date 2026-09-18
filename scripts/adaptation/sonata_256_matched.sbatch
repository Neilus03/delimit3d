#!/bin/bash
#SBATCH --job-name=sonata256m
#SBATCH --partition=gpupr.120h
#SBATCH --gpus=nvidia_a100_80gb_pcie:2
#SBATCH --constraint=AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/%x-%j.err
set -euo pipefail

repo=/cluster/home/nedela/nedela/projects/delimit3d
python=/cluster/work/igp_psr/nedela/litept-env/bin/python
artifact=/cluster/work/igp_psr/nedela/delimit3d_sonata_rgbn6_256_matched_v1
snapshot_root=$artifact/source_snapshots
archive=$snapshot_root/delimit3d-working-tree.tar
snapshot=$snapshot_root/delimit3d-working-tree
config=$snapshot/configs/adaptation/sonata_rgbn6_256_matched.example.yaml
runner=scripts/adaptation/run_sonata_matched_adaptation.py
mode=train
if [[ $# -ge 1 ]]; then
  mode=$1
fi

mkdir -p "$snapshot_root"
if [[ ! -f "$snapshot/.snapshot_complete" ]]; then
  rm -f "$archive"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$archive" \
    -C "$repo" src scripts configs
  rm -rf "$snapshot"
  mkdir -p "$snapshot"
  tar -xf "$archive" -C "$snapshot"
  touch "$snapshot/.snapshot_complete"
fi

set +u
export PYTHONPATH="$snapshot/src:/cluster/work/igp_psr/nedela/sonata:$PYTHONPATH"
set -u
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader)
printf '%s\n' "$gpu_report"
gpu_count=$(printf '%s\n' "$gpu_report" | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$gpu_count" -ne 2 ]]; then
  echo "Expected exactly two allocated GPUs, got $gpu_count" >&2
  exit 2
fi
case "$gpu_report" in
  *"RTX 3090"*|*"RTX 4090"*|*"A100"*) ;;
  *) echo "No approved 3090/4090/A100 GPU allocated" >&2; exit 2 ;;
esac
case "$gpu_report" in *"2080"*) echo "RTX 2080 Ti is forbidden" >&2; exit 2;; esac

mkdir -p "$artifact"
gpu_flat=$(printf '%s' "$gpu_report" | tr '\n' ';')
repo_head=$(git -C "$repo" rev-parse HEAD)
archive_sha=$(sha256sum "$archive" | awk '{print $1}')
started=$(date -Is)
cat > "$artifact/launch_$SLURM_JOB_ID.json" <<EOF
{"schema":"delimit3d_sonata_matched_launch/v1","slurm_job_id":"$SLURM_JOB_ID","hostname":"$(hostname)","mode":"$mode","repo_head":"$repo_head","archive":"$archive","archive_sha256":"$archive_sha","config":"$config","gpu_inventory":"$gpu_flat","started":"$started"}
EOF

cd "$snapshot"
if [[ "$mode" == "prepare" ]]; then
  exec "$python" "$runner" --config "$config" --mode prepare
fi

if [[ ! -f "$artifact/prepare_status.json" ]]; then
  "$python" "$runner" --config "$config" --mode prepare
fi

exec "$python" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$runner" --config "$config" --mode "$mode"
