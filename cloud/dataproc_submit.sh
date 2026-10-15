#!/usr/bin/env bash
# Run the same pipeline on Google Cloud Dataproc.
# Prereqs: gcloud CLI authenticated, a GCP project with billing, and data/raw/online_retail_ii.csv
# produced locally by `python3 src/01_ingest_bronze.py` (Excel -> CSV landing).
set -euo pipefail
PROJECT_ID=${PROJECT_ID:?set PROJECT_ID}
REGION=${REGION:-asia-south1}
BUCKET=${BUCKET:-gs://${PROJECT_ID}-ecommerce-analytics}
CLUSTER=${CLUSTER:-ecommerce-cluster}
DATA_DIR="$BUCKET/data"
cd "$(dirname "$0")/.."

gcloud config set project "$PROJECT_ID"
gsutil ls "$BUCKET" >/dev/null 2>&1 || gsutil mb -l "$REGION" "$BUCKET"
gsutil cp data/raw/online_retail_ii.csv "$DATA_DIR/raw/"

gcloud dataproc clusters create "$CLUSTER" --region "$REGION" \
  --single-node --master-machine-type n2-standard-4 --image-version 2.2-debian12 || true

for step in 01_ingest_bronze 02_silver 03_gold 04_ml; do
  gcloud dataproc jobs submit pyspark "src/$step.py" --cluster "$CLUSTER" --region "$REGION" \
    --py-files src/config.py -- --data-dir="$DATA_DIR"
done

# Pull results back for the dashboard, then delete the cluster to save cost
gsutil -m cp -r "$DATA_DIR/silver" "$DATA_DIR/gold" data/
gcloud dataproc clusters delete "$CLUSTER" --region "$REGION" --quiet
