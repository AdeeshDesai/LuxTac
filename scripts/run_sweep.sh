#!/usr/bin/env bash
# Depth sweep of the alignment grasps: close the gripper in 7 steps and read the pad
# at each. One IsaacGym process per pose (a second sim in one process segfaults).
#
#   ./scripts/run_sweep.sh <outdir> [--keys "peg_vert peg_horz"] [hydra overrides]
#
# The arm is level (reset pose noise zeroed). Default: all 16 poses.
# One IsaacGym per GPU: check `nvidia-smi --query-compute-apps=pid --format=csv` first.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT=$(realpath -m "$1"); shift; mkdir -p "$OUT"
LEVEL=(task.randomize.fingertip_midpoint_rot_noise="[0.0,0.0,0.0]" task.randomize.fingertip_midpoint_pos_noise="[0.0,0.0,0.0]")
ORI="vert horz deg45 deg135"; KEYS=""
while [ $# -gt 0 ]; do case "$1" in
    --keys) KEYS="$2"; shift 2;;
    *) break;; esac; done
if [ -z "$KEYS" ]; then for o in cuboid cylinder ellipse peg; do for a in $ORI; do KEYS="$KEYS ${o}_${a}"; done; done; fi
cd "$REPO_DIR"
for k in $KEYS; do
    timeout 900 "$PY" scripts/align_luxtac_objects.py --object "$k" \
        --indent 1.5 2.0 2.5 3.0 3.5 4.0 4.5 --out "$OUT/$k.npz" "${LEVEL[@]}" "$@" > "$OUT/$k.log" 2>&1
    echo "$k exit $?"
done
echo DONE
