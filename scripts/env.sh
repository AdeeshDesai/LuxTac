# Sourced by the launchers. Override any of these from the environment.
#   IGE_DIR    luxtac-isaacgymenvs clone the overlay was deployed into
#              (default: thirdparty/luxtac-isaacgymenvs)
#   CONDA_ENV  python environment created by install.sh
#              (default: the active conda env, else <conda base>/envs/luxtac)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IGE_DIR="${IGE_DIR:-$REPO_DIR/thirdparty/luxtac-isaacgymenvs}"
if [ -z "${CONDA_ENV:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "${LUXTAC_ENV_NAME:-luxtac}" ]; then
        CONDA_ENV="$CONDA_PREFIX"
    else
        for _b in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
            if [ -x "$_b/envs/${LUXTAC_ENV_NAME:-luxtac}/bin/python" ]; then CONDA_ENV="$_b/envs/${LUXTAC_ENV_NAME:-luxtac}"; break; fi
        done
    fi
fi
if [ -z "${CONDA_ENV:-}" ] || [ ! -x "$CONDA_ENV/bin/python" ]; then
    echo "error: python environment not found. Run install.sh, or set CONDA_ENV=/path/to/env" >&2
    exit 1
fi
PY="$CONDA_ENV/bin/python"
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$IGE_DIR:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -f "$IGE_DIR/isaacgymenvs/tasks/tacsl/luxtac_task_insertion.py" ]; then
    echo "error: LuxTac is not deployed in $IGE_DIR. Run: scripts/deploy.sh $IGE_DIR" >&2
    exit 1
fi
