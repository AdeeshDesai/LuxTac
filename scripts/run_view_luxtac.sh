#!/usr/bin/env bash
# Live per-taxel readout of one LuxTac alignment object: SDF depth, the driver's s,
# the peak-normalised 0-1 grid, and the matching real hold, with the gripper on the
# keyboard.
#
#   ./scripts/run_view_luxtac.sh --object peg_vert
#   ./scripts/run_view_luxtac.sh --object cuboid_horz --indent 4.0
#   ./scripts/run_view_luxtac.sh --object peg_vert --once --indent 4.5   # no keyboard
#
# One IsaacGym per GPU. Check first:  nvidia-smi --query-compute-apps=pid --format=csv
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO_DIR"
exec "$PY" scripts/view_luxtac_taxels.py "$@"
