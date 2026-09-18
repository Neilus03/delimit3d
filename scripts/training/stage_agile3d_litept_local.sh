#!/usr/bin/env bash
# Sourced by the batch launcher. Runtime, sources and input data stay off Lustre.
set -euo pipefail
: "${AGILE3D_LOCAL_ROOT:=${TMPDIR:?A node-local TMPDIR is required}/agile3d-litept}"
mkdir -p "${AGILE3D_LOCAL_ROOT}"
case "$(findmnt -n -o FSTYPE -T "${AGILE3D_LOCAL_ROOT}")" in
  lustre|nfs|nfs4) echo "AGILE3D_LOCAL_ROOT must use node-local storage" >&2; exit 2 ;;
esac
echo "Staging exact runtime and inputs at ${AGILE3D_LOCAL_ROOT}: $(date -Is)"
runtime_source=$(dirname "$(dirname "${python_bin}")")
# Reuse requires an explicit, successfully completed staging marker. The
# default job-specific TMPDIR is always fresh.
if [[ ! -f "${AGILE3D_LOCAL_ROOT}/runtime.ready" ]]; then
  mkdir -p "${AGILE3D_LOCAL_ROOT}/litept-env"
  rsync -a --timeout=300 --exclude=/lib/python3.12/site-packages "${runtime_source}/" "${AGILE3D_LOCAL_ROOT}/litept-env/"
  mkdir -p "${AGILE3D_LOCAL_ROOT}/litept-env/lib/python3.12/site-packages"
  # Independent packages can be copied concurrently; rsync alone serializes
  # tens of thousands of small shared-filesystem reads.
  find "${runtime_source}/lib/python3.12/site-packages" -mindepth 1 -maxdepth 1 -print0 |
    xargs -0 -P "${SLURM_CPUS_PER_TASK:-8}" -I{} rsync -a --timeout=300 "{}" "${AGILE3D_LOCAL_ROOT}/litept-env/lib/python3.12/site-packages/"
  printf '%s\n' "${runtime_source}" > "${AGILE3D_LOCAL_ROOT}/runtime.ready"
fi
test "$(cat "${AGILE3D_LOCAL_ROOT}/runtime.ready")" = "${runtime_source}"
python_bin="${AGILE3D_LOCAL_ROOT}/litept-env/bin/python"
mapfile -t input_paths < <("${python_bin}" - "${config}" <<'PY'
import os, sys, yaml
with open(sys.argv[1]) as f:
    paths = yaml.safe_load(f)["paths"]
for key in ("agile3d_root", "litept_root", "scan_folder", "train_list", "val_list"):
    print(os.path.expandvars(os.path.expanduser(paths[key])))
PY
)
test "${#input_paths[@]}" -eq 5
mkdir -p "${AGILE3D_LOCAL_ROOT}"/{AGILE3D,LitePT,ScanNet/scans,repo/src,repo/scripts/training}
rsync -a --exclude=.git --exclude=__pycache__ "${input_paths[0]}/" "${AGILE3D_LOCAL_ROOT}/AGILE3D/"
rsync -a --exclude=.git --exclude=__pycache__ --exclude=/data --exclude=/exp --exclude=/pretrained "${input_paths[1]}/" "${AGILE3D_LOCAL_ROOT}/LitePT/"
rsync -a "${input_paths[2]}/" "${AGILE3D_LOCAL_ROOT}/ScanNet/scans/"
cp "${input_paths[3]}" "${AGILE3D_LOCAL_ROOT}/ScanNet/train_list.json"
cp "${input_paths[4]}" "${AGILE3D_LOCAL_ROOT}/ScanNet/val_list.json"
rsync -a --exclude=__pycache__ "${repo}/src/" "${AGILE3D_LOCAL_ROOT}/repo/src/"
cp "${script}" "${AGILE3D_LOCAL_ROOT}/repo/scripts/training/"
cp "${config}" "${AGILE3D_LOCAL_ROOT}/config.yaml"
config="${AGILE3D_LOCAL_ROOT}/config.yaml"
script="${AGILE3D_LOCAL_ROOT}/repo/scripts/training/$(basename "${script}")"
repo="${AGILE3D_LOCAL_ROOT}/repo"
export AGILE3D_STAGED_ROOT="${AGILE3D_LOCAL_ROOT}"
export PYTHONPATH="${repo}/src:${AGILE3D_LOCAL_ROOT}/LitePT${PYTHONPATH:+:${PYTHONPATH}}"
echo "Staging finished: $(date -Is)"
