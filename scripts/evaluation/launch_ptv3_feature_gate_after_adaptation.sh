#!/bin/sh
set -eu

REPO_ROOT=/tmp/euler_cluster_nedela_rw/home/nedela/nedela/projects/delimit3d
PYTHON=/tmp/delimit3d_sonata_pf_v1/env/bin/python
CONFIG=$REPO_ROOT/configs/evaluation/ptv3_feature_geometry_gate.example.yaml
SCRIPT=$REPO_ROOT/scripts/evaluation/evaluate_ptv3_feature_geometry.py
SESSION=delimit3d-ptv3-feature-gate

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "already running: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && exec env PYTHONPATH='$REPO_ROOT/src' '$PYTHON' '$SCRIPT' --config '$CONFIG' --mode wait-and-run --poll-seconds 60"
echo "started $SESSION"
