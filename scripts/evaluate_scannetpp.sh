#!/usr/bin/env bash
set -euo pipefail
exec python scripts/evaluation/evaluate_scannetpp.py "$@"
