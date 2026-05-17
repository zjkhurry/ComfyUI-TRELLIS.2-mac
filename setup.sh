#!/usr/bin/env bash
#
# Set up TRELLIS.2 for ComfyUI.
# Installs dependencies and clones repos into the plugin directory.
#

set -euo pipefail
cd "$(dirname "$0")"

echo "=== TRELLIS.2 ComfyUI — Setup ==="
echo

# ---------------------------------------------------------------------------
# Clone Git dependencies
# ---------------------------------------------------------------------------
DEPS_DIR="deps"
mkdir -p "$DEPS_DIR"

clone_dep() {
    local url="$1" dir="$2" ref="${3:-}"
    if [ ! -d "$DEPS_DIR/$dir" ]; then
        echo "Cloning $dir ..."
        git clone --depth 1 ${ref:+--branch "$ref"} "$url" "$DEPS_DIR/$dir"
    else
        echo "  $dir already cloned — skipping"
    fi
}

# utils3d needs a specific commit
if [ ! -d "$DEPS_DIR/utils3d" ]; then
    echo "Cloning utils3d ..."
    git clone https://github.com/EasternJournalist/utils3d.git "$DEPS_DIR/utils3d"
    git -C "$DEPS_DIR/utils3d" checkout 9a4eb15e4021b67b12c460c7057d642626897ec8
else
    echo "  utils3d already cloned — skipping"
fi

clone_dep https://github.com/pedronaugusto/mtlbvh.git       mtlbvh
clone_dep https://github.com/pedronaugusto/mtldiffrast.git   mtldiffrast
clone_dep https://github.com/pedronaugusto/mtlmesh.git       mtlmesh
clone_dep https://github.com/pedronaugusto/mtlgemm.git       mtlgemm
clone_dep https://github.com/pedronaugusto/trellis2-apple.git trellis2-apple

# TRELLIS.2
if [ ! -d "TRELLIS.2" ]; then
    echo "Cloning TRELLIS.2 ..."
    git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
else
    echo "  TRELLIS.2 already cloned — skipping"
fi

echo

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
echo "Installing dependencies..."
DEPS="torch torchvision torchaudio transformers accelerate huggingface_hub safetensors pillow numpy trimesh scipy tqdm easydict kornia timm imageio opencv-python-headless xatlas fast-simplification"

if command -v uv &>/dev/null; then
    PIP="uv pip install"
else
    PIP="pip3 install"
fi

$PIP $DEPS
$PIP "./deps/utils3d"

# ---------------------------------------------------------------------------
# Install Metal backends (optional, for texture baking acceleration)
# ---------------------------------------------------------------------------
if [ "${SKIP_METAL:-0}" != "1" ]; then
    export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
    echo
    echo "Installing Metal backends for texture baking (set SKIP_METAL=1 to skip)..."
    PIP_NB="$PIP --no-build-isolation"
    $PIP setuptools wheel pybind11
    $PIP_NB "./deps/mtlbvh"      || echo "  mtlbvh install failed — continuing without Metal BVH"
    $PIP_NB "./deps/mtldiffrast" || echo "  mtldiffrast install failed — continuing without Metal rasterizer"
    $PIP_NB "./deps/mtlmesh"     || echo "  mtlmesh install failed — continuing without Metal mesh ops"
    $PIP_NB "./deps/mtlgemm"     || echo "  mtlgemm install failed — baker will use lower-quality fallback"
    $PIP_NB "./deps/trellis2-apple/o-voxel" \
        || echo "  o_voxel install failed — falling back to KDTree baker"
fi

# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
echo
echo "Applying patches..."
python3 patches/mps_compat.py

echo
echo "=== Setup complete ==="
echo "Restart ComfyUI to load the TRELLIS.2 nodes."
echo
echo "Download model weights:"
echo "  huggingface-cli download microsoft/TRELLIS.2-4B --local-dir models/TRELLIS.2-4B"
