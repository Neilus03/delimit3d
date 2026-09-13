#!/bin/bash
#SBATCH --job-name=sonata256
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
expected_commit=0c44aa32c1cb174bd1e7942d4f32b34d877f9c22
artifact=/cluster/work/igp_psr/nedela/delimit3d_sonata_rgbn6_256_v1
archive_root=${artifact}/source_snapshots
archive=${archive_root}/delimit3d-${expected_commit}.tar
snapshot=${archive_root}/delimit3d-${expected_commit}
config=${snapshot}/configs/adaptation/sonata_rgbn6_256.example.yaml
mode=${1:-preflight}

# The source archive is made from the committed setup only; unrelated existing
# untracked work in the repository is never imported into a GPU job.
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

# Never silently run on an unapproved GPU type.
gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader)
printf '%s\n' "${gpu_report}"
case "${gpu_report}" in
  *"RTX 3090"*|*"RTX 4090"*|*"A100"*) ;;
  *) echo "No approved 3090/4090/A100 GPU allocated" >&2; exit 2 ;;
esac
case "${gpu_report}" in *"2080"*) echo "RTX 2080 Ti is forbidden" >&2; exit 2;; esac

# The source/checkpoint and data manifest are external and hashed by prepare.
mkdir -p "${artifact}"
cat > "${artifact}/launch_${SLURM_JOB_ID}.json" <<EOF
{"schema":"delimit3d_sonata_launch/v1","slurm_job_id":"${SLURM_JOB_ID}","hostname":"$(hostname)","mode":"${mode}","source_commit":"${expected_commit}","source_archive":"${archive}","source_archive_sha256":"${archive_sha}","checkpoint":"/cluster/work/igp_psr/nedela/sonata_weights/sonata.pth","config":"${config}","gpu_inventory":"${gpu_report//$'\n'/; }","started":"$(date -Is)"}
EOF

cd "${snapshot}"
exec "${python}" scripts/adaptation/run_sonata_adaptation.py --config "${config}" --mode "${mode}"
