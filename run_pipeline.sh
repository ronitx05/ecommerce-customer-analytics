#!/usr/bin/env bash
# Runs the full pipeline: raw -> Bronze -> Silver -> Gold -> MLlib predictions
set -euo pipefail
cd "$(dirname "$0")/src"
for step in 01_ingest_bronze 02_silver 03_gold 04_ml; do
  echo "===== Running $step ====="
  python3 "$step.py"
done
echo "Pipeline complete. Launch the dashboard with: streamlit run dashboard/app.py"
