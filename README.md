# E-Commerce Customer Behavior and Recommendation Analytics

Big Data Minor Project (Project #6) — PySpark medallion pipeline (Bronze → Silver → Gold), Spark SQL analytics, Spark MLlib customer segmentation and product recommendations, and a Streamlit dashboard.

## Problem Statement
Online retailers generate millions of transactions but often lack a clear view of **who their valuable customers are** and **what each customer is likely to buy next**. Using ~1M real transactions from a UK online retailer, this project:
1. Builds a scalable PySpark pipeline to ingest, clean and model the data.
2. Analyses sales trends, product and country performance, and customer behavior (RFM).
3. Segments customers with **K-Means** (Spark MLlib) on Recency/Frequency/Monetary features.
4. Recommends top-5 products per customer with **ALS collaborative filtering** (Spark MLlib).

**Target users:** marketing / CRM and merchandising teams. **Expected outcomes:** actionable customer segments, personalised recommendations, KPI dashboard.

## Dataset
- **Name:** Online Retail II — UCI Machine Learning Repository
- **Source:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Size:** 1,067,371 transactions, 8 attributes, 01/12/2009 – 09/12/2011
- **Citation:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0)

| Column | Description |
|---|---|
| Invoice | Invoice number (prefix `C` = cancellation) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units per line |
| InvoiceDate | Date & time of invoice |
| Price | Unit price (GBP) |
| Customer ID | Customer identifier |
| Country | Customer country |

## Architecture
```
UCI zip (Excel) ──► data/raw (CSV landing)
        │  01_ingest_bronze.py   (PySpark)
        ▼
Bronze  data/bronze      raw strings + source_system / source_file / ingested_at
        │  02_silver.py          (type casting, dedup, invalid & non-product rows removed, cancellations split)
        ▼
Silver  data/silver/sales (partitioned by year_month), data/silver/cancellations, quality_report.json
        │  03_gold.py            (Spark SQL, joins, aggregations, window functions)
        ▼
Gold    kpis, monthly_revenue, product_stats, country_stats, top_products_by_country, time_patterns, customer_rfm
        │  04_ml.py              (Spark MLlib: K-Means + ALS)
        ▼
Predictions  customer_segments, cluster_profile, recommendations, metrics.json
        │
        ▼
Streamlit dashboard (dashboard/app.py)      Orchestration: run_pipeline.sh · Airflow DAG · Dataproc script
```

## Big Data Technologies Used
| Requirement | Where |
|---|---|
| PySpark DataFrames (filter, joins, aggregations, partitioned Parquet) | `src/01-04` |
| Spark SQL + window functions (LAG, SUM OVER, DENSE_RANK, NTILE, ROW_NUMBER) | `src/03_gold.py`, `src/04_ml.py` |
| Spark MLlib — K-Means clustering, ALS recommendation, evaluators | `src/04_ml.py` |
| Medallion architecture (Bronze / Silver / Gold) in Parquet | `data/` |
| Google Cloud Dataproc + GCS | `cloud/dataproc_submit.sh` |
| Workflow orchestration (Apache Airflow) | `dags/ecommerce_pipeline_dag.py` |
| Dashboard (Streamlit + Plotly) | `dashboard/app.py` |

## Data Preprocessing (Silver)
| Stage | Rows |
|---|---|
| Bronze (raw) | 1,067,371 |
| After removing exact duplicates | 1,033,036 |
| Clean sales lines (valid customer, qty > 0, price > 0, real products) | 776,579 |
| Cancellation lines (stored separately) | 17,602 |

## Results
**Business KPIs:** £17.07M revenue · 36,594 orders · 5,852 customers · 4,620 products · 41 countries · AOV £466.43. Revenue is strongly seasonal, peaking every November (pre-Christmas).

**Customer segmentation (K-Means, k chosen by silhouette: k=3→0.505, 4→0.529, 5→0.510, 6→0.501):**
| Cluster | Customers | Avg recency (days) | Avg orders | Avg spend (£) |
|---|---|---|---|---|
| High-Value Loyal | 1,165 | 29 | 19.4 | 10,727 |
| Mid-Value Slipping | 1,430 | 238 | 5.0 | 1,945 |
| Recent Low-Spend | 1,283 | 28 | 3.3 | 919 |
| Lapsed / Lost | 1,974 | 388 | 1.4 | 310 |

**Recommendations (ALS, implicit feedback, 80/20 hold-out):** 5,852 users × 4,620 items, 480,973 interactions. Hit-rate@10 = **0.499** (half of customers had at least one of their actual future products in the top-10), precision@10 = 0.076, MAP@10 = 0.046. Final model recommends the top-5 *unseen* products for every customer.

**Business recommendations:** retain *High-Value Loyal* customers with loyalty perks; run win-back campaigns for *Mid-Value Slipping* customers; use ALS cross-sell recommendations for *Recent Low-Spend* customers to grow basket size; time campaigns ahead of the Q4 peak.

## How to Run
```bash
pip install -r requirements.txt          # needs Java 17+ for Spark
./run_pipeline.sh                          # downloads data, builds Bronze/Silver/Gold, trains models (~5 min)
streamlit run dashboard/app.py             # open http://localhost:8501
```
**Google Cloud Dataproc:** `PROJECT_ID=<your-project> ./cloud/dataproc_submit.sh` creates a bucket and single-node cluster, runs all four PySpark jobs against GCS (`--data-dir=gs://...`), copies results back and deletes the cluster.

**Airflow:** copy `dags/ecommerce_pipeline_dag.py` into your Airflow `dags/` folder and set `PROJECT_DIR`; the DAG runs `ingest_bronze >> clean_silver >> build_gold >> train_mllib_models` daily.

## Project Structure
```
src/config.py              paths + Spark session helper
src/01_ingest_bronze.py    ingestion -> Bronze
src/02_silver.py           cleaning -> Silver
src/03_gold.py             Spark SQL analytics -> Gold
src/04_ml.py               MLlib K-Means + ALS -> predictions
dashboard/app.py           Streamlit dashboard
dags/                      Airflow DAG
cloud/                     Dataproc submission script
run_pipeline.sh            end-to-end local run
```

## References
- Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208.
- Apache Spark documentation — Spark SQL & MLlib (K-Means, ALS): https://spark.apache.org/docs/latest/
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. *ICDM*.
- Streamlit: https://docs.streamlit.io · Apache Airflow: https://airflow.apache.org · Google Cloud Dataproc: https://cloud.google.com/dataproc
