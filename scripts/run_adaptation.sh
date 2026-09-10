#!/usr/bin/env bash
set -euo pipefail
exec python -m delimit3d.training.runner "$@"
