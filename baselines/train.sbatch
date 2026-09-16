#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --partition=gpupr.120h
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint=AMD
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=160G
#SBATCH --time=5-00:00:00
#SBATCH --output=/cluster/work/igp_psr/nedela/scratch_baselines_20260916/logs/train_%j.out
#SBATCH --error=/cluster/work/igp_psr/nedela/scratch_baselines_20260916/logs/train_%j.err
set -euo pipefail
ARM="${1:?Expected agile3d or mask3d}"
RUN_ROOT=/cluster/work/igp_psr/nedela/scratch_baselines_20260916
AGILE_REPO="$RUN_ROOT/source/agile3d"
MASK_REPO="$RUN_ROOT/source/mask3d"
case "$ARM" in
  agile3d) OUTPUT="$RUN_ROOT/agile3d" ;;
  mask3d) OUTPUT="$RUN_ROOT/mask3d/litept_mask3d_ca_scratch_600_seed42" ;;
  *) exit 2 ;;
esac
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/training.lock"
flock -n 9 || { echo 'Another worker owns this baseline'; exit 2; }
python3 - "$RUN_ROOT" "$OUTPUT" <<'PY'
import json,sys,hashlib
from pathlib import Path
root,output=map(Path,sys.argv[1:])
checks=json.loads((root/'preflight_passed.json').read_text())
assert checks['passed'] is True
assert hashlib.sha256((root/'mask3d_decoder_random_seed42.pt').read_bytes()).hexdigest()==checks['mask3d_decoder_file_sha256']
PY
if [[ -f "$OUTPUT/complete" ]]; then exit 0; fi
(cd "$RUN_ROOT/source" && sha256sum --quiet -c SHA256SUMS)
source /etc/profile
set +u
module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
source "$MASK_REPO/student/scripts/extract_litept_local_archives.sh"
export PYTHONPATH="$AGILE_REPO/src:$MASK_REPO/student:$TMPDIR/runtime/pointrope_sm80:$TMPDIR/runtime/LitePT:$TMPDIR/runtime/chorus:${PYTHONPATH:-}"
export CHORUS_SCANNET_METADATA_ROOT=/cluster/work/igp_psr/nedela/scannet_metadata
export LITEPT_MASK3D_PARITY_DECODER_TEMPLATE="$RUN_ROOT/mask3d_decoder_random_seed42.pt"
PYTHON="$VIRTUAL_ENV/bin/python"
if [[ "$ARM" == agile3d ]]; then
  "$PYTHON" "$AGILE_REPO/scripts/evaluation/run_scannet40_scratch.py" --output "$OUTPUT"
else
  export CHORUS_OUTPUT_ROOT="$RUN_ROOT/mask3d"
  cd "$MASK_REPO/student"
  RESUME=()
  if [[ -f "$OUTPUT/checkpoints/latest.pt" ]]; then RESUME=(--resume "$OUTPUT/checkpoints/latest.pt"); fi
  "$PYTHON" scripts/run_litept_mask3d.py --config configs/litept_mask3d_ca_scratch_600_seed42.yaml "${RESUME[@]}"
fi
COMPLETE=$("$PYTHON" - "$OUTPUT/status.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['complete']))
PY
)
if [[ "$COMPLETE" == 1 ]]; then
  touch "$OUTPUT/complete"
else
  # Only a clean, checkpointed walltime boundary schedules a continuation.
  sbatch --dependency="afterok:$SLURM_JOB_ID" --job-name="scratch-$ARM" "$AGILE_REPO/baselines/train.sbatch" "$ARM" | tee -a "$OUTPUT/continuations.log"
fi
