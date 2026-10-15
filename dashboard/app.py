"""Streamlit dashboard reading the actual Gold / prediction outputs of the PySpark pipeline."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
GOLD = os.path.join(DATA, "gold")
PRED = os.path.join(GOLD, "predictions")

st.set_page_config(page_title="E-Commerce Customer Analytics", layout="wide")


@st.cache_data
def load(path):
    return pd.read_parquet(path)


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


st.title("E-Commerce Customer Behavior & Recommendation Analytics")
st.caption("Source: UCI Online Retail II (2009-2011) · Processed with PySpark, Spark SQL & Spark MLlib")

kpis = load(os.path.join(GOLD, "kpis")).iloc[0]
c = st.columns(5)
c[0].metric("Revenue (GBP)", f"£{kpis.total_revenue / 1e6:,.2f}M")
c[1].metric("Orders", f"{int(kpis.total_orders):,}")
c[2].metric("Customers", f"{int(kpis.total_customers):,}")
c[3].metric("Products", f"{int(kpis.total_products):,}")
c[4].metric("Avg Order Value", f"£{kpis.avg_order_value:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sales Trends", "Products & Countries", "Customer Segments",
                                        "Recommendations", "Data Quality & Model Metrics"])

with tab1:
    monthly = load(os.path.join(GOLD, "monthly_revenue")).sort_values("year_month")
    st.plotly_chart(px.line(monthly, x="year_month", y="revenue", markers=True, title="Monthly Revenue"),
                    use_container_width=True)
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(monthly, x="year_month", y="mom_growth_pct", title="Month-over-Month Growth (%)"),
                      use_container_width=True)
    col2.plotly_chart(px.line(monthly, x="year_month", y="customers", markers=True, title="Active Customers per Month"),
                      use_container_width=True)
    tp = load(os.path.join(GOLD, "time_patterns"))
    heat = tp.pivot_table(index="weekday", columns="hour", values="orders", aggfunc="sum").fillna(0)
    order = [d for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] if d in heat.index]
    st.plotly_chart(px.imshow(heat.loc[order], aspect="auto", title="Orders by Weekday and Hour",
                              color_continuous_scale="Blues"), use_container_width=True)

with tab2:
    n = st.slider("Top N products", 5, 30, 10)
    prod = load(os.path.join(GOLD, "product_stats")).nsmallest(n, "revenue_rank")
    st.plotly_chart(px.bar(prod.sort_values("revenue"), x="revenue", y="description", orientation="h",
                           title=f"Top {n} Products by Revenue"), use_container_width=True)
    cs = load(os.path.join(GOLD, "country_stats"))
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(cs.head(10), x="country", y="revenue", title="Top 10 Countries by Revenue"),
                      use_container_width=True)
    col2.plotly_chart(px.bar(cs[cs.orders >= 50].nlargest(10, "cancel_rate_pct"), x="country", y="cancel_rate_pct",
                             title="Highest Cancellation Rate (countries with 50+ orders)"), use_container_width=True)
    country = st.selectbox("Top products in country", cs.country.tolist())
    tpc = load(os.path.join(GOLD, "top_products_by_country"))
    st.dataframe(tpc[tpc.country == country].sort_values("rnk"), hide_index=True, use_container_width=True)

with tab3:
    seg = load(os.path.join(PRED, "customer_segments"))
    prof = load(os.path.join(PRED, "cluster_profile")).sort_values("avg_monetary", ascending=False)
    st.subheader("K-Means Clusters (Spark MLlib)")
    st.dataframe(prof, hide_index=True, use_container_width=True)
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.pie(prof, names="cluster_name", values="customers", title="Customers per Cluster"),
                      use_container_width=True)
    col2.plotly_chart(px.scatter(seg.sample(min(3000, len(seg)), random_state=1), x="recency", y="monetary",
                                 color="cluster_name", size="frequency", log_y=True,
                                 title="Recency vs Monetary (sample of 3,000)"), use_container_width=True)
    st.plotly_chart(px.histogram(seg, x="rfm_segment", color="cluster_name", barmode="stack",
                                 title="Rule-based RFM Segment vs K-Means Cluster"), use_container_width=True)

with tab4:
    recs = load(os.path.join(PRED, "recommendations"))
    st.subheader("Personalised Product Recommendations (ALS)")
    cust = st.selectbox("Customer ID", sorted(recs.customer_id.unique()))
    info = seg[seg.customer_id == cust].iloc[0]
    m = st.columns(4)
    m[0].metric("Cluster", info.cluster_name)
    m[1].metric("Recency (days)", int(info.recency))
    m[2].metric("Orders", int(info.frequency))
    m[3].metric("Spend", f"£{info.monetary:,.2f}")
    col1, col2 = st.columns(2)
    col1.markdown("**Top-5 recommended products (not yet purchased)**")
    col1.dataframe(recs[recs.customer_id == cust].sort_values("rec_rank")[["rec_rank", "description", "score"]],
                   hide_index=True, use_container_width=True)
    hist = pd.read_parquet(os.path.join(DATA, "silver", "sales"), columns=["customer_id", "description", "quantity"],
                           filters=[("customer_id", "==", int(cust))])
    col2.markdown("**Most purchased products (history)**")
    col2.dataframe(hist.groupby("description", as_index=False).quantity.sum().nlargest(10, "quantity"),
                   hide_index=True, use_container_width=True)

with tab5:
    col1, col2 = st.columns(2)
    col1.subheader("Silver layer data quality")
    col1.json(load_json(os.path.join(DATA, "silver", "quality_report.json")))
    col2.subheader("Model validation metrics")
    col2.json(load_json(os.path.join(GOLD, "metrics.json")))
