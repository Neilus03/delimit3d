#!/bin/bash
#SBATCH --job-name=sonata_eval
#SBATCH --partition=gpuhe.120h
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/%x-%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/%x-%j.err
set -euo pipefail

repo=/cluster/home/nedela/nedela/projects/delimit3d
python=/cluster/work/igp_psr/nedela/litept-env/bin/python
expected_commit=b85d0c9a9caf490c53134599e4c9ec24a7a4e40d
artifact=/cluster/work/igp_psr/nedela/delimit3d_sonata_rgbn6_256_v1
archive_root=${artifact}/source_snapshots
archive=${archive_root}/delimit3d-${expected_commit}.tar
snapshot=${archive_root}/delimit3d-${expected_commit}
train_commit=3247a72fadcc16c747eaf5fdf271aae8529cc7f7
train_snapshot=${archive_root}/delimit3d-${train_commit}
config=${train_snapshot}/configs/adaptation/sonata_rgbn6_256.example.yaml

actual_commit=$(git -C "${repo}" rev-parse HEAD)
git -C "${repo}" merge-base --is-ancestor "${expected_commit}" "${actual_commit}"
mkdir -p "${archive_root}"
if [ ! -s "${archive}" ]; then
  git -C "${repo}" archive --format=tar "${expected_commit}" > "${archive}"
fi
archive_sha=$(sha256sum "${archive}" | awk '{print $1}')
if [ ! -f "${snapshot}/.snapshot_complete" ]; then
  mkdir -p "${snapshot}"
  tar -xf "${archive}" -C "${snapshot}"
  touch "${snapshot}/.snapshot_complete"
fi

export PYTHONPATH="${snapshot}/src:/cluster/work/igp_psr/nedela/sonata:${PYTHONPATH:-}"
gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader)
case "${gpu_report}" in
  *"RTX 3090"*|*"RTX 4090"*|*"A100"*) ;;
  *) echo "No approved 3090/4090/A100 GPU allocated" >&2; exit 2;;
esac
case "${gpu_report}" in *"2080"*) echo "RTX 2080 Ti is forbidden" >&2; exit 2;; esac

adapted=${artifact}/checkpoint_u000256.pt
pack=${artifact}/fixed_eval_pack
output=${artifact}/sonata_fixed_pair_u256.json
test -s "${adapted}"
test -f "${pack}/manifest.json"
test -f "${config}"
mkdir -p "${artifact}"
cat > "${artifact}/evaluation_launch_${SLURM_JOB_ID}.json" <<EOF
{"schema":"delimit3d_sonata_fixed_pair_launch/v1","slurm_job_id":"${SLURM_JOB_ID}","hostname":"$(hostname)","source_commit":"${expected_commit}","source_archive":"${archive}","source_archive_sha256":"${archive_sha}","config":"${config}","pack":"${pack}","adapted_checkpoint":"${adapted}","output":"${output}","gpu_inventory":"${gpu_report//$'\n'/; }","started":"$(date -Is)"}
EOF

cd "${snapshot}"
exec "${python}" scripts/evaluation/evaluate_sonata_fixed_pair.py \
  --config "${config}" \
  --pack-dir "${pack}" \
  --adapted-checkpoint "${adapted}" \
  --output "${output}" \
  --device cuda
