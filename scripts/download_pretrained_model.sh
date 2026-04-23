#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Allow override via MODEL_DIR, otherwise use repo-local pretrained/.
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/pretrained}"
MODEL_ZIP="$MODEL_DIR/glorie-pretrained.zip"
MODEL_PATH="$MODEL_DIR/omnidata_dpt_depth_v2.ckpt"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists. Skipping download."
else
    echo "Model not found. Downloading..."
    if ! gdown https://drive.google.com/uc?id=1oZbVPrubtaIUjRRuT8F-YjjHBW-1spKT -O "$MODEL_ZIP"; then
        echo "Failed to download model."
        exit 1
    fi
    if ! unzip "$MODEL_ZIP" -d "$MODEL_DIR"; then
        echo "Failed to unzip model."
        exit 1
    fi
    rm "$MODEL_ZIP" "$MODEL_DIR/middle_fine.pt"
    echo "Model downloaded successfully."
fi

exec "$@"
