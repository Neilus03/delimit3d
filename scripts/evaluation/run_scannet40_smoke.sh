#!/usr/bin/env bash
# Invoke inside tmux within an allocation that binds exactly one RTX4090.
set -euo pipefail
: "${DELIMIT3D_CONFIG:?frozen resolved smoke YAML required}"
: "${DELIMIT3D_PYTHON:?exact frozen environment Python required}"
: "${DELIMIT3D_SMOKE_ROOT:?external smoke artifact root required}"
if [[ -z "${TMUX:-}" ]]; then echo 'Smoke worker must run inside tmux' >&2; exit 2; fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
mkdir -p "$DELIMIT3D_SMOKE_ROOT/workers"
echo "$$" > "$DELIMIT3D_SMOKE_ROOT/workers/smoke_shell.pid"
runner=scripts/evaluation/run_scannet40_matched.py
"$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode preflight
for arm in public delimit3d; do
  "$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode cache-features --arm "$arm"
done
"$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode smoke
for arm in public delimit3d; do
  "$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode train --arm "$arm"
done
for arm in public delimit3d; do
  for panel in MO SO; do
    "$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode evaluate --arm "$arm" --panel "$panel"
  done
done
"$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode aggregate
"$DELIMIT3D_PYTHON" - <<'PY'
import hashlib,json,os,time
from pathlib import Path
r=Path(os.environ["DELIMIT3D_SMOKE_ROOT"])
p=json.loads((r/"freeze/provenance.json").read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
out={"completed":True,"time":time.time(),"host":os.uname().nodename,"source_commit":p["repo_commit"],"provenance_sha256":sha(r/"freeze/provenance.json"),"smoke_report_sha256":sha(r/"smoke_report.json"),"aggregate_sha256":sha(r/"aggregate.json")}
(r/"smoke_pipeline_complete.json").write_text(json.dumps(out)+"\n")
PY
