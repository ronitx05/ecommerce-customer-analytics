"""Steps 5-6: PySpark / Spark SQL processing to build the Gold (business-ready) layer.

Uses Spark SQL, joins, aggregations and window functions (LAG, SUM OVER, DENSE_RANK, NTILE).
"""
import os

from config import SILVER_DIR, GOLD_DIR, get_spark


def save(df, name):
    df.coalesce(1).write.mode("overwrite").parquet(os.path.join(GOLD_DIR, name))
    print(f"gold/{name}: {df.count():,} rows")


def main():
    spark = get_spark("gold_build")
    spark.read.parquet(os.path.join(SILVER_DIR, "sales")).createOrReplaceTempView("sales")
    spark.read.parquet(os.path.join(SILVER_DIR, "cancellations")).createOrReplaceTempView("cancellations")

    save(spark.sql("""
        SELECT COUNT(DISTINCT invoice)                         AS total_orders,
               COUNT(DISTINCT customer_id)                     AS total_customers,
               COUNT(DISTINCT stock_code)                      AS total_products,
               COUNT(DISTINCT country)                         AS total_countries,
               ROUND(SUM(line_total), 2)                       AS total_revenue,
               ROUND(SUM(line_total) / COUNT(DISTINCT invoice), 2) AS avg_order_value
        FROM sales"""), "kpis")

    # Monthly revenue with month-over-month growth and running total (window functions)
    save(spark.sql("""
        WITH m AS (
            SELECT year_month, ROUND(SUM(line_total), 2) AS revenue,
                   COUNT(DISTINCT invoice) AS orders, COUNT(DISTINCT customer_id) AS customers
            FROM sales GROUP BY year_month)
        SELECT *,
               ROUND(100 * (revenue - LAG(revenue) OVER (ORDER BY year_month))
                     / LAG(revenue) OVER (ORDER BY year_month), 2) AS mom_growth_pct,
               ROUND(SUM(revenue) OVER (ORDER BY year_month), 2)   AS cumulative_revenue
        FROM m ORDER BY year_month"""), "monthly_revenue")

    save(spark.sql("""
        SELECT stock_code, FIRST(description) AS description,
               SUM(quantity) AS units_sold, ROUND(SUM(line_total), 2) AS revenue,
               COUNT(DISTINCT customer_id) AS buyers,
               DENSE_RANK() OVER (ORDER BY SUM(line_total) DESC) AS revenue_rank
        FROM sales GROUP BY stock_code"""), "product_stats")

    # Country stats joined with cancellation counts
    save(spark.sql("""
        WITH s AS (SELECT country, ROUND(SUM(line_total), 2) AS revenue,
                          COUNT(DISTINCT invoice) AS orders, COUNT(DISTINCT customer_id) AS customers
                   FROM sales GROUP BY country),
             c AS (SELECT country, COUNT(DISTINCT invoice) AS cancelled_orders
                   FROM cancellations GROUP BY country)
        SELECT s.*, COALESCE(c.cancelled_orders, 0) AS cancelled_orders,
               ROUND(100 * COALESCE(c.cancelled_orders, 0) / (s.orders + COALESCE(c.cancelled_orders, 0)), 2) AS cancel_rate_pct
        FROM s LEFT JOIN c ON s.country = c.country ORDER BY revenue DESC"""), "country_stats")

    save(spark.sql("""
        SELECT * FROM (
            SELECT country, description, ROUND(SUM(line_total), 2) AS revenue,
                   DENSE_RANK() OVER (PARTITION BY country ORDER BY SUM(line_total) DESC) AS rnk
            FROM sales GROUP BY country, description)
        WHERE rnk <= 5"""), "top_products_by_country")

    save(spark.sql("""
        SELECT date_format(invoice_ts, 'E') AS weekday, dayofweek(invoice_ts) AS weekday_num,
               hour(invoice_ts) AS hour, COUNT(DISTINCT invoice) AS orders, ROUND(SUM(line_total), 2) AS revenue
        FROM sales GROUP BY 1, 2, 3"""), "time_patterns")

    # RFM per customer + quintile scores (NTILE) + rule-based segment
    save(spark.sql("""
        WITH ref AS (SELECT DATE_ADD(MAX(invoice_date), 1) AS ref_date FROM sales),
        base AS (
            SELECT customer_id, FIRST(country) AS country,
                   DATEDIFF((SELECT ref_date FROM ref), MAX(invoice_date)) AS recency,
                   COUNT(DISTINCT invoice) AS frequency,
                   ROUND(SUM(line_total), 2) AS monetary,
                   MIN(invoice_date) AS first_purchase, MAX(invoice_date) AS last_purchase
            FROM sales GROUP BY customer_id),
        scored AS (
            SELECT *, NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
                      NTILE(5) OVER (ORDER BY frequency)    AS f_score,
                      NTILE(5) OVER (ORDER BY monetary)     AS m_score
            FROM base)
        SELECT *, CASE
                WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
                WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
                WHEN r_score >= 4 AND f_score <= 2 THEN 'New / Promising'
                WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
                WHEN r_score <= 2 AND f_score <= 2 THEN 'Hibernating'
                ELSE 'Need Attention' END AS rfm_segment
        FROM scored"""), "customer_rfm")

    spark.stop()


if __name__ == "__main__":
    main()
