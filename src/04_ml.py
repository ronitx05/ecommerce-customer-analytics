"""Steps 7-8: Spark MLlib models, predictions and validation.

1. K-Means customer segmentation on log-scaled RFM features (k chosen by silhouette).
2. ALS collaborative filtering (implicit feedback) for top-5 product recommendations,
   validated on a held-out 20% split with precision@10 / MAP@10 and RMSE.
Outputs are stored as Parquet in data/gold/predictions and metrics in data/gold/metrics.json.
"""
import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RankingEvaluator, RegressionEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import Window
from pyspark.sql import functions as F

from config import SILVER_DIR, GOLD_DIR, get_spark, write_json

PRED_DIR = os.path.join(GOLD_DIR, "predictions")
SEED = 42


def segment_customers(spark, metrics):
    rfm = spark.read.parquet(os.path.join(GOLD_DIR, "customer_rfm"))
    feats = rfm.select(
        "customer_id",
        F.log1p("recency").alias("log_recency"),
        F.log1p("frequency").alias("log_frequency"),
        F.log1p(F.greatest("monetary", F.lit(0.0))).alias("log_monetary"),
    )
    assembler = VectorAssembler(inputCols=["log_recency", "log_frequency", "log_monetary"], outputCol="raw")
    scaler = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    evaluator = ClusteringEvaluator(featuresCol="features")

    best, scores = None, {}
    for k in range(3, 7):
        model = Pipeline(stages=[assembler, scaler, KMeans(k=k, seed=SEED)]).fit(feats)
        scores[k] = round(evaluator.evaluate(model.transform(feats)), 4)
        if best is None or scores[k] > scores[best[0]]:
            best = (k, model)
    k, model = best
    print(f"KMeans silhouette by k: {scores} -> chosen k={k}")
    metrics["kmeans"] = {"silhouette_by_k": scores, "chosen_k": k, "silhouette": scores[k]}

    clustered = model.transform(feats).select("customer_id", F.col("prediction").alias("cluster"))
    joined = rfm.join(clustered, "customer_id")

    # Name clusters by average monetary value (highest = most valuable)
    profile = joined.groupBy("cluster").agg(
        F.count("*").alias("customers"),
        F.round(F.avg("recency"), 1).alias("avg_recency_days"),
        F.round(F.avg("frequency"), 1).alias("avg_frequency"),
        F.round(F.avg("monetary"), 2).alias("avg_monetary"),
    )
    names = ["High-Value Loyal", "Mid-Value Slipping", "Recent Low-Spend", "Lapsed / Lost", "Other A", "Other B"]
    ranked = profile.withColumn("value_rank", F.row_number().over(Window.orderBy(F.desc("avg_monetary"))) - 1)
    name_map = F.create_map(*[x for i, n in enumerate(names) for x in (F.lit(i), F.lit(n))])
    profile = ranked.withColumn("cluster_name", name_map[F.col("value_rank")]).drop("value_rank")

    segments = joined.join(profile.select("cluster", "cluster_name"), "cluster")
    segments.write.mode("overwrite").parquet(os.path.join(PRED_DIR, "customer_segments"))
    profile.coalesce(1).write.mode("overwrite").parquet(os.path.join(PRED_DIR, "cluster_profile"))
    profile.orderBy("avg_monetary", ascending=False).show(truncate=False)


def recommend_products(spark, metrics):
    sales = spark.read.parquet(os.path.join(SILVER_DIR, "sales"))
    # Implicit feedback: how many separate orders a customer bought the product in
    interactions = sales.groupBy("customer_id", "stock_code").agg(F.countDistinct("invoice").alias("rating"))
    indexer = StringIndexer(inputCol="stock_code", outputCol="item_id").fit(interactions)
    data = indexer.transform(interactions).withColumn("item_id", F.col("item_id").cast("int")).cache()

    als = ALS(userCol="customer_id", itemCol="item_id", ratingCol="rating", implicitPrefs=True,
              rank=20, maxIter=10, regParam=0.1, alpha=10.0, coldStartStrategy="drop", seed=SEED)

    # ---- validation on 20% hold-out ----
    train, test = data.randomSplit([0.8, 0.2], seed=SEED)
    model = als.fit(train)
    rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") \
        .evaluate(model.transform(test))
    k = 10
    recs = model.recommendForUserSubset(test.select("customer_id").distinct(), k) \
        .select("customer_id", F.col("recommendations.item_id").cast("array<double>").alias("predicted"))
    actual = test.groupBy("customer_id").agg(F.collect_set(F.col("item_id").cast("double")).alias("actual"))
    ranking_df = recs.join(actual, "customer_id")
    rank_eval = RankingEvaluator(predictionCol="predicted", labelCol="actual", k=k)
    precision = rank_eval.setMetricName("precisionAtK").evaluate(ranking_df)
    map_k = rank_eval.setMetricName("meanAveragePrecisionAtK").evaluate(ranking_df)
    hit_rate = ranking_df.withColumn("hit", F.size(F.array_intersect("predicted", "actual")) > 0) \
        .agg(F.avg(F.col("hit").cast("double"))).first()[0]
    metrics["als"] = {"rmse_implicit": round(rmse, 4), f"precision_at_{k}": round(precision, 4),
                      f"map_at_{k}": round(map_k, 4), f"hit_rate_at_{k}": round(hit_rate, 4),
                      "users": data.select("customer_id").distinct().count(),
                      "items": data.select("item_id").distinct().count(), "interactions": data.count()}
    print("ALS validation:", metrics["als"])

    # ---- final model on all data: top-5 unseen products per customer ----
    final = als.fit(data)
    top = final.recommendForAllUsers(30).select("customer_id", F.posexplode("recommendations").alias("pos", "rec")) \
        .select("customer_id", "pos", F.col("rec.item_id").alias("item_id"), F.col("rec.rating").alias("score"))
    unseen = top.join(data.select("customer_id", "item_id"), ["customer_id", "item_id"], "left_anti")
    w = Window.partitionBy("customer_id").orderBy(F.desc("score"))
    top5 = unseen.withColumn("rec_rank", F.row_number().over(w)).filter("rec_rank <= 5")

    products = spark.read.parquet(os.path.join(GOLD_DIR, "product_stats")).select("stock_code", "description")
    labels = spark.createDataFrame([(i, s) for i, s in enumerate(indexer.labels)], ["item_id", "stock_code"])
    out = top5.join(labels, "item_id").join(products, "stock_code", "left") \
        .select("customer_id", "rec_rank", "stock_code", "description", F.round("score", 4).alias("score"))
    out.write.mode("overwrite").parquet(os.path.join(PRED_DIR, "recommendations"))
    print(f"Recommendations written for {out.select('customer_id').distinct().count():,} customers")


def main():
    spark = get_spark("mllib_models")
    metrics = {}
    segment_customers(spark, metrics)
    recommend_products(spark, metrics)
    write_json(metrics, os.path.join(GOLD_DIR, "metrics.json"))
    print(json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
