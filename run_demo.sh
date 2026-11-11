#!/bin/bash
# =============================================================================
# Launch script for Interactive Dialect Identification & Code-Switching Web App
# Final Project: Neuro-Symbolic Italian Dialect Identification
# Authors: Luca Porfiri & Demetrio Angeloni
# =============================================================================
set -e

cd "$(dirname "$0")"

PORT=${1:-8000}
CHECKPOINTS_DIR="checkpoints"
BASELINE_URL="https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/download/v1.0.0/best_baseline_model.pt"
FUSION_URL="https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/download/v1.0.0/best_fusion_model.pt"

mkdir -p "$CHECKPOINTS_DIR"

# -----------------------------------------------------------------------------
# 1. macOS iCloud Drive Materialization (if applicable)
# -----------------------------------------------------------------------------
if command -v brctl >/dev/null 2>&1; then
    brctl download checkpoints/baseline_logreg.joblib >/dev/null 2>&1 || true
    brctl download checkpoints/fusion_feature_extractor.joblib >/dev/null 2>&1 || true
    brctl download checkpoints/best_baseline_model.pt >/dev/null 2>&1 || true
    brctl download checkpoints/best_fusion_model.pt >/dev/null 2>&1 || true
    brctl download checkpoints/cm_baseline_logreg.png >/dev/null 2>&1 || true
fi

# -----------------------------------------------------------------------------
# 2. Automated Download of Heavy Neural Model Checkpoints (>400MB each)
# -----------------------------------------------------------------------------
get_file_size() {
    if [ -f "$1" ]; then
        wc -c < "$1" | tr -d ' '
    else
        echo 0
    fi
}

MIN_SIZE=50000000  # 50 MB threshold to catch placeholders / corrupted files

BASELINE_FILE="$CHECKPOINTS_DIR/best_baseline_model.pt"
BASELINE_SIZE=$(get_file_size "$BASELINE_FILE")
if [ "$BASELINE_SIZE" -lt "$MIN_SIZE" ]; then
    echo "================================================================================"
    echo " [1/2] Downloading Baseline BERT model weights (~422 MB)..."
    echo " Source: $BASELINE_URL"
    echo "================================================================================"
    curl -L --progress-bar -o "$BASELINE_FILE" "$BASELINE_URL"
    echo " ✓ Baseline BERT checkpoint downloaded successfully."
else
    echo " ✓ Baseline BERT model weights present ($(du -h "$BASELINE_FILE" | cut -f1))."
fi

FUSION_FILE="$CHECKPOINTS_DIR/best_fusion_model.pt"
FUSION_SIZE=$(get_file_size "$FUSION_FILE")
if [ "$FUSION_SIZE" -lt "$MIN_SIZE" ]; then
    echo "================================================================================"
    echo " [2/2] Downloading Feature Fusion BERT model weights (~424 MB)..."
    echo " Source: $FUSION_URL"
    echo "================================================================================"
    curl -L --progress-bar -o "$FUSION_FILE" "$FUSION_URL"
    echo " ✓ Feature Fusion BERT checkpoint downloaded successfully."
else
    echo " ✓ Feature Fusion BERT model weights present ($(du -h "$FUSION_FILE" | cut -f1))."
fi

# -----------------------------------------------------------------------------
# 3. Locate or Setup Python Virtual Environment
# -----------------------------------------------------------------------------
PYTHON_BIN=""

check_env() {
    local py="$1"
    if [ -x "$py" ]; then
        if "$py" -c "import torch, transformers, sklearn, joblib, pandas" >/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    fi
    return 1
}

# Check existing candidate environments
if [ -n "$VIRTUAL_ENV" ] && check_env "$VIRTUAL_ENV/bin/python" >/dev/null; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif check_env ".venv_demo/bin/python" >/dev/null; then
    PYTHON_BIN=".venv_demo/bin/python"
elif check_env "/tmp/nlp_demo_env/bin/python" >/dev/null; then
    PYTHON_BIN="/tmp/nlp_demo_env/bin/python"
elif check_env ".venv/bin/python" >/dev/null; then
    PYTHON_BIN=".venv/bin/python"
fi

# If no working environment is found, create .venv_demo
if [ -z "$PYTHON_BIN" ]; then
    echo "--------------------------------------------------------------------------------"
    echo " Setting up Python virtual environment (.venv_demo)..."
    echo "--------------------------------------------------------------------------------"
    BASE_PY="python3"
    if [ -x "/opt/homebrew/bin/python3.13" ]; then
        BASE_PY="/opt/homebrew/bin/python3.13"
    elif [ -x "/opt/homebrew/bin/python3.12" ]; then
        BASE_PY="/opt/homebrew/bin/python3.12"
    elif command -v python3 >/dev/null 2>&1; then
        BASE_PY="python3"
    fi

    "$BASE_PY" -m venv .venv_demo
    PYTHON_BIN=".venv_demo/bin/python"

    echo " Installing dependencies from requirements.txt..."
    "$PYTHON_BIN" -m pip install --quiet --upgrade pip
    "$PYTHON_BIN" -m pip install --quiet -r requirements.txt
    echo " ✓ Environment ready."
fi

# -----------------------------------------------------------------------------
# 4. Launch Web Demonstration Server
# -----------------------------------------------------------------------------
echo "================================================================================"
echo " Starting Neuro-Symbolic Italian Dialect Identification Demo Server"
echo " URL: http://127.0.0.1:$PORT"
echo " Using Python: $PYTHON_BIN"
echo "================================================================================"
chmod +x "$0" 2>/dev/null || true
exec "$PYTHON_BIN" demo_app/server.py "$PORT"
