"""Shared paths and Spark session helper for the pipeline."""
import json
import os
import subprocess
import sys
import tempfile
from pyspark.sql import SparkSession

BASE_DIR = os.environ.get("PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
# Allow `--data-dir=gs://bucket/data` when submitted to Google Cloud Dataproc
for _arg in sys.argv[1:]:
    if _arg.startswith("--data-dir="):
        DATA_DIR = _arg.split("=", 1)[1]

RAW_DIR = os.path.join(DATA_DIR, "raw")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SOURCE_NAME = "UCI Online Retail II (doi:10.24432/C5CG6D)"


def get_spark(app_name):
    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def write_json(obj, path):
    """Write a small JSON report locally or to GCS (gsutil is available on Dataproc nodes)."""
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(obj, f, indent=2)
        subprocess.run(["gsutil", "cp", f.name, path], check=True)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
