"""Step 2-3: Ingest the raw UCI Online Retail II data into the Bronze layer.

Downloads the source zip (if missing), lands both Excel sheets as CSV in data/raw,
then loads them with Spark and stores near-raw Parquet with ingestion metadata.
"""
import os
import urllib.request
import zipfile

import pandas as pd
from pyspark.sql import functions as F

from config import RAW_DIR, BRONZE_DIR, DATASET_URL, SOURCE_NAME, get_spark

ZIP_PATH = os.path.join(RAW_DIR, "online_retail_ii.zip")
CSV_PATH = os.path.join(RAW_DIR, "online_retail_ii.csv")


def land_raw():
    if RAW_DIR.startswith("gs://"):  # on Dataproc the CSV is uploaded to GCS beforehand
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    if os.path.exists(CSV_PATH):
        return
    with zipfile.ZipFile(ZIP_PATH) as zf:
        xlsx = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
        zf.extract(xlsx, RAW_DIR)
    print("Converting Excel sheets to CSV (one-time, ~1-2 min)...")
    sheets = pd.read_excel(os.path.join(RAW_DIR, xlsx), sheet_name=None, dtype=str)
    frames = []
    for name, df in sheets.items():
        df["source_sheet"] = name
        frames.append(df)
    pd.concat(frames).to_csv(CSV_PATH, index=False)


def main():
    land_raw()
    spark = get_spark("bronze_ingest")
    df = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(CSV_PATH)

    # Normalise column names (e.g. "Customer ID" -> customer_id) and add lineage metadata
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    df = (
        df.withColumn("source_system", F.lit(SOURCE_NAME))
        .withColumn("source_file", F.lit(os.path.basename(CSV_PATH)))
        .withColumn("ingested_at", F.current_timestamp())
    )

    df.write.mode("overwrite").parquet(BRONZE_DIR)
    print(f"Bronze rows written: {df.count():,}")
    df.printSchema()
    spark.stop()


if __name__ == "__main__":
    main()
