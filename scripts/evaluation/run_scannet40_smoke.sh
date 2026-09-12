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
"$DELIMIT3D_PYTHON" "$runner" --config "$DELIMIT3D_CONFIG" --mode smoke-pipeline
