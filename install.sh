#!/bin/bash
# LuxTac installation. The dependencies go into thirdparty/:
#
#   thirdparty/
#   ├── IsaacGym_Preview_TacSL_Package/  downloaded by this script
#   └── luxtac-isaacgymenvs/             cloned by this script
#
# Optional environment variables:
#   LUXTAC_ENV_NAME     conda environment name            (default: luxtac)
#   LUXTAC_IGE_URL      git URL or local path to clone luxtac-isaacgymenvs from
#   CI=true             no prompts; use the detected conda home

set -e  # Exit on any error

# Install into the environment only: packages in ~/.local would otherwise satisfy pip
# and leave the environment incomplete.
export PYTHONNOUSERSITE=1

echo "=========================================="
echo "LuxTac Installation Script"
echo "=========================================="

ENV_NAME="${LUXTAC_ENV_NAME:-luxtac}"
LUXTAC_IGE_URL="${LUXTAC_IGE_URL:-https://github.com/AdeeshDesai/luxtac-isaacgymenvs.git}"

# Check if conda or mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Found mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✓ Found conda"
elif [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniforge3/bin/conda"
    echo "✓ Found conda at $HOME/miniforge3"
else
    echo "⚠ conda/mamba not found. Installing Miniforge3..."
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
    "$HOME/miniforge3/bin/conda" init bash
    CONDA_CMD="$HOME/miniforge3/bin/conda"
    echo "✓ Miniforge3 installed successfully"
    echo "Please restart your terminal or run: source ~/.bashrc"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPS_DIR="$SCRIPT_DIR/thirdparty"
mkdir -p "$DEPS_DIR"
ISAACGYM_ID="13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v"

# Get conda base path so we can use it as the default env path
CONDA_BASE="$("$CONDA_CMD" info --base 2>/dev/null | tail -n 1 | awk '{print $NF}')"

# The environment is created at <miniforge_home>/envs/$ENV_NAME. Press Enter to accept
# the detected default. In CI mode (CI=true) the default is used without prompting.
get_conda_env_path() {
    local default_home="$CONDA_BASE"

    echo ""
    echo "=========================================="
    echo "Conda Environment Path"
    echo "=========================================="
    echo "Detected Miniforge/conda home: $default_home"

    if [ "${CI:-false}" = "true" ]; then
        MINIFORGE_HOME="$default_home"
        echo "CI mode: using default Miniforge home"
    else
        read -rp "Enter Miniforge home directory (press Enter to use default): " user_home
        if [ -z "$user_home" ]; then
            MINIFORGE_HOME="$default_home"
        else
            MINIFORGE_HOME="$user_home"
        fi
    fi

    CONDA_ENV_PATH="$MINIFORGE_HOME/envs/$ENV_NAME"
    echo "✓ Environment will be created at: $CONDA_ENV_PATH"
}

echo ""
echo "=========================================="
echo "Setting up LuxTac environment"
echo "=========================================="

get_conda_env_path

if [ -x "$CONDA_ENV_PATH/bin/python" ]; then
    echo "✓ Environment already exists at '$CONDA_ENV_PATH'"
else
    echo "Creating Python 3.8 environment at '$CONDA_ENV_PATH'..."
    "$CONDA_CMD" create --prefix "$CONDA_ENV_PATH" python=3.8 -y
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

echo ""
echo "=========================================="
echo "Installing IsaacGym TacSL"
echo "=========================================="

if [ ! -d "$DEPS_DIR/IsaacGym_Preview_TacSL_Package" ]; then
    TAR="$DEPS_DIR/IsaacGym_Preview_TacSL_Package.tar.gz"
    if [ ! -f "$TAR" ]; then
        echo "Downloading IsaacGym_Preview_TacSL_Package..."
        pip install -q gdown
        gdown "$ISAACGYM_ID" -O "$TAR" || true
    fi
    if [ ! -f "$TAR" ]; then
        echo "⚠ Download failed. Get the archive from:"
        echo "  https://drive.google.com/file/d/$ISAACGYM_ID/view"
        echo "Save it as $TAR and run this script again."
        exit 1
    fi
    tar -xzf "$TAR" -C "$DEPS_DIR"
fi
echo "✓ Found IsaacGym_Preview_TacSL_Package"
pip install -e "$DEPS_DIR/IsaacGym_Preview_TacSL_Package/isaacgym/python/"

echo ""
echo "=========================================="
echo "Cloning repositories"
echo "=========================================="

# IsaacGymEnvs + TacSL
if [ ! -d "$DEPS_DIR/luxtac-isaacgymenvs" ]; then
    echo "Cloning luxtac-isaacgymenvs (IsaacGymEnvs + TacSL)..."
    git clone "$LUXTAC_IGE_URL" "$DEPS_DIR/luxtac-isaacgymenvs"
else
    echo "✓ luxtac-isaacgymenvs already exists"
fi
cd "$DEPS_DIR/luxtac-isaacgymenvs"
pip install -e .

echo ""
echo "=========================================="
echo "Installing LuxTac"
echo "=========================================="

cd "$SCRIPT_DIR"
pip install -e .

echo "Installing additional dependencies..."
# numpy stays below 1.24: urdfpy and networkx (dependencies of isaacgymenvs) still use
# the np.float / np.int aliases that 1.24 removed.
pip install numpy==1.23.5 scipy==1.10.1 matplotlib==3.7.5 opencv-python==4.10.0.84 \
    pynput==1.8.1 hidapi==0.15.0 imageio==2.35.1 networkx==2.2 zarr==2.16.1 numcodecs==0.12.1

echo ""
echo "=========================================="
echo "Deploying LuxTac into luxtac-isaacgymenvs"
echo "=========================================="

bash "$SCRIPT_DIR/scripts/deploy.sh" "$DEPS_DIR/luxtac-isaacgymenvs"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $CONDA_ENV_PATH"
echo "  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}"
echo ""
echo "To verify:"
echo "  python -c \"from isaacgym import gymtorch\""
echo "  ./scripts/run_view_luxtac.sh --object peg_vert --once --headless"
echo ""
