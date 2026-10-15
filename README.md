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
