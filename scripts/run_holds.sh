#!/usr/bin/env bash
# Record a 50-frame hold per grasp at its operating step: the sweep step whose peak
# contact depth is nearest 0.55 mm. The gripper follows the same closing path as the
# sweep up to that step, so the hold starts from the same state.
#
#   ./scripts/run_holds.sh <sweepdir> <outdir> [--keys "..."] [hydra overrides]
#
# <sweepdir> is a level sweep from run_sweep.sh made with the SAME overrides.
# Default poses: the grasps compared with the real pad plus the three 135-degree mirrors.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LV=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2; mkdir -p "$OUT"
KEYS="cuboid_vert cuboid_horz cuboid_deg45 cylinder_vert cylinder_horz cylinder_deg45 ellipse_horz ellipse_deg45 peg_vert peg_horz peg_deg45 cylinder_deg135 ellipse_deg135 peg_deg135"
if [ "${1:-}" = "--keys" ]; then KEYS="$2"; shift 2; fi
cd "$REPO_DIR"
for k in $KEYS; do
    IND=$("$PY" -I -c "
import numpy as np, sys
z = np.load(sys.argv[1]); p = 'lux_' + sys.argv[2] + '__'
j = int(np.argmin(abs(z[p + 'd_max'] - 0.55)))
print(' '.join(str(x) for x in z[p + 'target_mm'][:j + 1]))" "$LV/$k.npz" "$k")
    timeout 900 "$PY" scripts/hold_luxtac_objects.py --object "$k" --frames 50 \
        --indent $IND --out "$OUT/$k.npz" \
        task.randomize.fingertip_midpoint_rot_noise="[0.0,0.0,0.0]" \
        task.randomize.fingertip_midpoint_pos_noise="[0.0,0.0,0.0]" "$@" > "$OUT/$k.log" 2>&1
    echo "$k [$IND] exit $? $(grep '^hold' "$OUT/$k.log" | cut -c1-90)"
done
echo DONE
