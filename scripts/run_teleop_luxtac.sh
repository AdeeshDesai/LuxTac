#!/usr/bin/env bash
# SpaceMouse teleop with the live pad viewer; episodes are recorded.
#
#   ./scripts/run_teleop_luxtac.sh --object peg_insert --indent 6.0    # peg-in-hole
#   ./scripts/run_teleop_luxtac.sh --object usb --indent 3.0           # USB insertion
#   ./scripts/run_teleop_luxtac.sh --object peg_vert                   # an alignment object
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO_DIR/luxtac/demo"

# Pads display from a separate process: cv2.imshow deadlocks against the IsaacGym
# viewer in the same process.
"$PY" luxtac_pad_viewer.py &
VIEWER_PID=$!
trap 'kill $VIEWER_PID 2>/dev/null' EXIT

"$PY" demo_teleop_luxtac.py "$@"
