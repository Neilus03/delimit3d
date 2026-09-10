#!/usr/bin/env bash
set -euo pipefail
exec python -m delimit3d.cli.initialize_scratch "$@"
