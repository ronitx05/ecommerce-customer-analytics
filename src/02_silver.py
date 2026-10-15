"""Step 4: Clean the Bronze data into the Silver layer.

- casts types (quantity, price, timestamp, customer id)
- trims text and standardises descriptions / countries
- removes exact duplicates, invalid records (null customer, qty/price <= 0)
- separates cancellations (invoice starting with 'C') into their own table
"""
import json
import os

from pyspark.sql import functions as F

from config import BRONZE_DIR, SILVER_DIR, get_spark


def main():
    spark = get_spark("silver_clean")
    bronze = spark.read.parquet(BRONZE_DIR)
    report = {"bronze_rows": bronze.count()}

    typed = (
        bronze.select(
            F.trim("invoice").alias("invoice"),
            F.upper(F.trim("stockcode")).alias("stock_code"),
            F.upper(F.trim("description")).alias("description"),
            F.col("quantity").cast("int").alias("quantity"),
            F.to_timestamp("invoicedate", "yyyy-MM-dd HH:mm:ss").alias("invoice_ts"),
            F.col("price").cast("double").alias("price"),
            F.col("customer_id").cast("double").cast("int").alias("customer_id"),
            F.initcap(F.trim("country")).alias("country"),
        )
        .withColumn("is_cancelled", F.col("invoice").startswith("C"))
        .dropDuplicates()
    )
    report["after_dedup"] = typed.count()

    # Non-product stock codes (postage, manual adjustments, bank charges, etc.)
    non_product = F.col("stock_code").rlike("^(POST|DOT|M|C2|D|S|B|AMAZONFEE|BANK CHARGES|PADS|ADJUST.*|TEST.*|GIFT.*)$")

    valid = (
        typed.filter(F.col("customer_id").isNotNull() & F.col("invoice_ts").isNotNull())
        .filter(~non_product)
    )
    sales = (
        valid.filter(~F.col("is_cancelled") & (F.col("quantity") > 0) & (F.col("price") > 0))
        .withColumn("line_total", F.round(F.col("quantity") * F.col("price"), 2))
        .withColumn("invoice_date", F.to_date("invoice_ts"))
        .withColumn("year_month", F.date_format("invoice_ts", "yyyy-MM"))
        .withColumn("description", F.coalesce("description", F.lit("UNKNOWN")))
        .drop("is_cancelled")
    )
    cancellations = valid.filter(F.col("is_cancelled")).drop("is_cancelled")

    sales.write.mode("overwrite").partitionBy("year_month").parquet(os.path.join(SILVER_DIR, "sales"))
    cancellations.write.mode("overwrite").parquet(os.path.join(SILVER_DIR, "cancellations"))

    report["silver_sales_rows"] = spark.read.parquet(os.path.join(SILVER_DIR, "sales")).count()
    report["silver_cancellation_rows"] = cancellations.count()
    report["removed_rows"] = report["bronze_rows"] - report["silver_sales_rows"] - report["silver_cancellation_rows"]
    with open(os.path.join(SILVER_DIR, "quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
