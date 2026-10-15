"""Shared paths and Spark session helper for the pipeline."""
import os
from pyspark.sql import SparkSession

BASE_DIR = os.environ.get("PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

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
