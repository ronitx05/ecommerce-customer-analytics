"""Airflow DAG orchestrating the medallion pipeline (daily refresh)."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/ecommerce-customer-analytics"  # path where the repo is deployed

default_args = {"owner": "minor-project", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="ecommerce_customer_analytics",
    default_args=default_args,
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    tags=["pyspark", "ecommerce", "mllib"],
) as dag:
    tasks = [
        BashOperator(task_id=name, bash_command=f"cd {PROJECT_DIR}/src && python3 {script}")
        for name, script in [
            ("ingest_bronze", "01_ingest_bronze.py"),
            ("clean_silver", "02_silver.py"),
            ("build_gold", "03_gold.py"),
            ("train_mllib_models", "04_ml.py"),
        ]
    ]
    tasks[0] >> tasks[1] >> tasks[2] >> tasks[3]
