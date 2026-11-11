#!/usr/bin/env bash
# Deploy the LuxTac overlay into a luxtac-isaacgymenvs clone. Re-run it after
# changing anything under luxtac/, isaacgymenvs/ or assets/.
#
# The runtime resolves assets and task code inside the isaacgymenvs tree (asset root is
# <clone>/assets), so the files are copied there. Everything lands in luxtac-named
# files and directories; no existing file of the clone is edited.
#
#   ./scripts/deploy.sh [path-to-luxtac-isaacgymenvs]     default: thirdparty/luxtac-isaacgymenvs
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${IGE_DIR:-$REPO_DIR/thirdparty/luxtac-isaacgymenvs}}"

if [ ! -d "$DEST/isaacgymenvs" ] || [ ! -d "$DEST/assets/tacsl/yaml" ]; then
    echo "error: $DEST does not look like a luxtac-isaacgymenvs clone" >&2
    exit 1
fi
echo "Deploying LuxTac from $REPO_DIR"
echo "                 into $DEST"

# 1. Sensor model -> isaacgymenvs.tacsl_sensors.luxtac (the task classes import it there)
S="$DEST/isaacgymenvs/tacsl_sensors/luxtac"
mkdir -p "$S"
cp "$REPO_DIR"/luxtac/{__init__,luxtac_cfg,luxtac_env_mixin,luxtac_sensor,luxtac_viz,photodiode,verify_analytic}.py "$S/"
rm -rf "$S/calibration"; cp -r "$REPO_DIR/luxtac/calibration" "$S/calibration"
find "$S" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "  + isaacgymenvs/tacsl_sensors/luxtac/"

# 2. Gripper and object assets
mkdir -p "$DEST/assets/luxtac/urdf" "$DEST/assets/luxtac/mesh"
cp "$REPO_DIR"/assets/luxtac/urdf/* "$DEST/assets/luxtac/urdf/"
cp "$REPO_DIR"/assets/luxtac/mesh/* "$DEST/assets/luxtac/mesh/"
echo "  + assets/luxtac/{urdf,mesh}/"

# 3. Object table. At run time the task reads the tables from THIS repo
#    (<hydra config dir>/../../assets/tacsl/yaml); the copy in the clone is for
#    launchers that use the clone's own cfg dir.
for f in "$REPO_DIR"/assets/tacsl/yaml/luxtac_asset_info_*.yaml; do
    cp "$f" "$DEST/assets/tacsl/yaml/"; echo "  + assets/tacsl/yaml/$(basename "$f")"
done

# 4. Task class
cp "$REPO_DIR"/isaacgymenvs/tasks/tacsl/luxtac_*.py "$DEST/isaacgymenvs/tasks/tacsl/"
echo "  + isaacgymenvs/tasks/tacsl/luxtac_task_insertion.py"

# 5. Task config
if [ -d "$DEST/isaacgymenvs/cfg/task" ]; then
    for f in "$REPO_DIR"/luxtac/config/task/LuxTac*.yaml; do
        cp "$f" "$DEST/isaacgymenvs/cfg/task/"; echo "  + isaacgymenvs/cfg/task/$(basename "$f")"
    done
fi
echo "Done."
