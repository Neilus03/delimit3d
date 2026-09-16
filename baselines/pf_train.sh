#!/bin/bash
set -euo pipefail
RUN=/home/nedela/scratch_baselines_20260916
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN/source/src:$RUN/runtime:$RUN/litept"
export LITEPT_ROOT="$RUN/litept"
unset POINTROPE_FORCE_TORCH
PYTHON=/scratch/nedela/delimit3d_scannet40_runtime_v1/bin/python
mkdir -p "$RUN/agile3d"
exec 9>"$RUN/agile3d/training.lock"
flock -n 9 || { echo 'Another baseline worker owns this run'; exit 2; }
cd "$RUN"
sha256sum --quiet -c runtime/SHA256SUMS
"$PYTHON" "$RUN/runtime/pf_verify_data.py"
# Exercise the actual 5 cm decoder/evaluation and epoch checkpoint resume.
if [[ ! -f "$RUN/runtime/pf_preflight_passed.json" ]]; then
 "$PYTHON" "$RUN/source/scripts/evaluation/run_scannet40_scratch.py" --manifest "$RUN/runtime/pf_selection_manifest.json" --grid-size .05 --smoke --output "$RUN/smoke_5cm"
 "$PYTHON" "$RUN/source/scripts/evaluation/run_scannet40_scratch.py" --manifest "$RUN/runtime/pf_selection_manifest.json" --grid-size .05 --smoke --output "$RUN/smoke_5cm"
 "$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('/home/nedela/scratch_baselines_20260916')
status=json.loads((root/'smoke_5cm/status.json').read_text())
assert status['epoch']==2
assert (root/'smoke_5cm/eval_epoch_0002.json').exists()
(root/'runtime/pf_preflight_passed.json').write_text(json.dumps({'passed':True,'grid_size':.05,'epoch_resume':2}))
PY
fi
while [[ ! -f "$RUN/agile3d/complete" ]]; do
 "$PYTHON" "$RUN/source/scripts/evaluation/run_scannet40_scratch.py" --manifest "$RUN/runtime/pf_selection_manifest.json" --grid-size .05 --output "$RUN/agile3d"
 COMPLETE=$("$PYTHON" -c "import json; print(int(json.load(open('$RUN/agile3d/status.json'))['complete']))")
 if [[ "$COMPLETE" == 1 ]]; then touch "$RUN/agile3d/complete"; fi
 # Successful 108-hour boundaries resume immediately, with the same schedule.
done
