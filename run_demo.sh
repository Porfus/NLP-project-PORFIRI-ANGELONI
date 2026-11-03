#!/bin/bash
# Launch script for the Interactive Dialect Identification & Code-Switching Web App
set -e

cd "$(dirname "$0")"

PORT=${1:-8000}

# 1. Ensure iCloud placeholders in checkpoints/ and results/ are materialized locally
if command -v brctl >/dev/null 2>&1; then
    brctl download checkpoints/baseline_logreg.joblib >/dev/null 2>&1 || true
    brctl download checkpoints/fusion_feature_extractor.joblib >/dev/null 2>&1 || true
    brctl download checkpoints/best_baseline_model.pt >/dev/null 2>&1 || true
    brctl download checkpoints/best_fusion_model.pt >/dev/null 2>&1 || true
    brctl download checkpoints/cm_baseline_logreg.png >/dev/null 2>&1 || true
    brctl download results/temperature_sampling_analysis.png >/dev/null 2>&1 || true
fi

# 2. Ensure fast local Python environment (/tmp/nlp_demo_env) is ready
if [ ! -x "/tmp/nlp_demo_env/bin/python" ]; then
    echo "Setting up fast local Python environment in /tmp/nlp_demo_env..."
    if [ -x "/opt/homebrew/bin/python3.13" ]; then
        /opt/homebrew/bin/python3.13 -m venv /tmp/nlp_demo_env
    else
        python3 -m venv /tmp/nlp_demo_env
    fi
    /tmp/nlp_demo_env/bin/pip install --quiet torch transformers scikit-learn joblib pandas
fi

PYTHON_BIN="/tmp/nlp_demo_env/bin/python"

echo "Starting Dialect Identification & Code-Switching Demo Server on http://127.0.0.1:$PORT ..."
exec "$PYTHON_BIN" demo_app/server.py "$PORT"
