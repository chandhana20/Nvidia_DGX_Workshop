# Databricks notebook source

# MAGIC %md
# MAGIC # Block 5: Databricks Apps -- GPU Fleet Monitor Dashboard
# MAGIC
# MAGIC **NVIDIA DGX Cloud MLOps & GenAI Workshop** | 35 min hands-on
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this block you will be able to:
# MAGIC
# MAGIC 1. **Explain** why Databricks Apps is the ideal deployment target for data-native applications
# MAGIC 2. **Build** a Streamlit dashboard that queries GPU telemetry data via SQL warehouse
# MAGIC 3. **Deploy** the app using Databricks CLI with zero infrastructure provisioning
# MAGIC 4. **Interact** with a live GPU Fleet Monitor that surfaces cluster health, alerts, and utilization

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why Deploy on Databricks Apps?
# MAGIC
# MAGIC | Benefit | What It Means for You |
# MAGIC |---|---|
# MAGIC | **Zero Infrastructure** | No VMs, no containers to manage, no Kubernetes. Push code and Databricks handles the rest. |
# MAGIC | **Built-in Authentication** | Every app inherits workspace-level SSO and SCIM. No auth plumbing required. |
# MAGIC | **Direct Data Access** | Query Unity Catalog tables and SQL warehouses directly -- no data export, no copy jobs. |
# MAGIC | **Governed** | Row/column-level security, audit logs, and lineage carry through to the app layer. |
# MAGIC | **Shareable** | One URL, role-based access. Share a live dashboard the way you share a notebook. |
# MAGIC
# MAGIC In this lab we will deploy a **GPU Fleet Monitor** Streamlit app that reads directly from
# MAGIC the telemetry and inventory tables we built in earlier blocks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define the Streamlit App Code
# MAGIC
# MAGIC The cell below writes a complete, production-ready `app.py` to a workspace volume.
# MAGIC The app features:
# MAGIC - **KPI tiles** -- total GPUs, active alerts, average utilization, average temperature
# MAGIC - **Cluster health heatmap** (Plotly)
# MAGIC - **GPU temperature time-series chart**
# MAGIC - **Recent alerts table** with severity colour-coding
# MAGIC - **Sidebar filters** -- cluster, cloud provider, date range

# COMMAND ----------

app_py_code = r'''
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Authentication & SQL connection
# ---------------------------------------------------------------------------
from databricks.sdk import WorkspaceClient
from databricks.sql import connect as dbsql_connect

SQL_WAREHOUSE_ID = "75fd8278393d07eb"
CATALOG = "main"
SCHEMA = "mlops_genai_workshop"

@st.cache_resource(show_spinner="Connecting to Databricks...")
def get_connection():
    """Return a Databricks SQL connection using SDK auth."""
    w = WorkspaceClient()
    return dbsql_connect(
        server_hostname=w.config.host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{SQL_WAREHOUSE_ID}",
        credentials_provider=lambda: w.config.authenticate,
    )

def run_query(sql: str) -> pd.DataFrame:
    """Execute a SQL statement and return a Pandas DataFrame."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GPU Fleet Monitor",
    page_icon="🖥️",
    layout="wide",
)

st.title("🖥️ GPU Fleet Monitor")
st.caption("Real-time visibility into your NVIDIA DGX Cloud GPU fleet")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

# Date range
default_start = datetime.now() - timedelta(days=7)
default_end = datetime.now()
date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start.date(), default_end.date()),
    max_value=default_end.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start.date(), default_end.date()

# Cluster filter
clusters_df = run_query(f"""
    SELECT DISTINCT cluster_id, cluster_name
    FROM {CATALOG}.{SCHEMA}.cluster_inventory
    ORDER BY cluster_name
""")
cluster_options = ["All"] + clusters_df["cluster_name"].tolist() if not clusters_df.empty else ["All"]
selected_cluster = st.sidebar.selectbox("Cluster", cluster_options)

# Cloud provider filter
providers_df = run_query(f"""
    SELECT DISTINCT cloud_provider
    FROM {CATALOG}.{SCHEMA}.cluster_inventory
    ORDER BY cloud_provider
""")
provider_options = ["All"] + providers_df["cloud_provider"].tolist() if not providers_df.empty else ["All"]
selected_provider = st.sidebar.selectbox("Cloud Provider", provider_options)

# ---------------------------------------------------------------------------
# Build WHERE clauses
# ---------------------------------------------------------------------------
telemetry_where = [f"t.timestamp >= '{start_date}'", f"t.timestamp < '{end_date + timedelta(days=1)}'"]
alerts_where = [f"a.event_time >= '{start_date}'", f"a.event_time < '{end_date + timedelta(days=1)}'"]

if selected_cluster != "All":
    cluster_id_row = clusters_df[clusters_df["cluster_name"] == selected_cluster]
    if not cluster_id_row.empty:
        cid = cluster_id_row.iloc[0]["cluster_id"]
        telemetry_where.append(f"t.cluster_id = '{cid}'")
        alerts_where.append(f"a.cluster_id = '{cid}'")

if selected_provider != "All":
    telemetry_where.append(f"c.cloud_provider = '{selected_provider}'")
    alerts_where.append(f"c.cloud_provider = '{selected_provider}'")

telemetry_filter = " AND ".join(telemetry_where)
alerts_filter = " AND ".join(alerts_where)

# ---------------------------------------------------------------------------
# KPI Tiles
# ---------------------------------------------------------------------------
st.markdown("---")

kpi_sql = f"""
    SELECT
        COUNT(DISTINCT t.gpu_id)        AS total_gpus,
        ROUND(AVG(t.utilization_pct), 1) AS avg_utilization,
        ROUND(AVG(t.temperature_c), 1)   AS avg_temperature
    FROM {CATALOG}.{SCHEMA}.gpu_telemetry t
    JOIN {CATALOG}.{SCHEMA}.cluster_inventory c ON t.cluster_id = c.cluster_id
    WHERE {telemetry_filter}
"""

alerts_count_sql = f"""
    SELECT COUNT(*) AS active_alerts
    FROM {CATALOG}.{SCHEMA}.gpu_health_events a
    JOIN {CATALOG}.{SCHEMA}.cluster_inventory c ON a.cluster_id = c.cluster_id
    WHERE {alerts_filter}
      AND a.severity IN ('CRITICAL', 'WARNING')
"""

with st.spinner("Loading KPIs..."):
    kpi_df = run_query(kpi_sql)
    alerts_count_df = run_query(alerts_count_sql)

col1, col2, col3, col4 = st.columns(4)

if not kpi_df.empty:
    col1.metric("Total GPUs", int(kpi_df.iloc[0]["total_gpus"]))
    col3.metric("Avg Utilization", f"{kpi_df.iloc[0]['avg_utilization']}%")
    col4.metric("Avg Temperature", f"{kpi_df.iloc[0]['avg_temperature']}°C")
else:
    col1.metric("Total GPUs", "N/A")
    col3.metric("Avg Utilization", "N/A")
    col4.metric("Avg Temperature", "N/A")

if not alerts_count_df.empty:
    alert_count = int(alerts_count_df.iloc[0]["active_alerts"])
    col2.metric("Active Alerts", alert_count, delta=None)
else:
    col2.metric("Active Alerts", "N/A")

# ---------------------------------------------------------------------------
# Cluster Health Heatmap
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Cluster Health Heatmap")

heatmap_sql = f"""
    SELECT
        c.cluster_name,
        t.gpu_id,
        ROUND(AVG(t.utilization_pct), 1) AS avg_util,
        ROUND(AVG(t.temperature_c), 1) AS avg_temp
    FROM {CATALOG}.{SCHEMA}.gpu_telemetry t
    JOIN {CATALOG}.{SCHEMA}.cluster_inventory c ON t.cluster_id = c.cluster_id
    WHERE {telemetry_filter}
    GROUP BY c.cluster_name, t.gpu_id
    ORDER BY c.cluster_name, t.gpu_id
"""

with st.spinner("Loading heatmap..."):
    heatmap_df = run_query(heatmap_sql)

if not heatmap_df.empty:
    pivot_df = heatmap_df.pivot_table(
        index="cluster_name",
        columns="gpu_id",
        values="avg_util",
        aggfunc="mean",
    )
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(c) for c in pivot_df.columns],
        y=pivot_df.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="Util %"),
        hovertemplate="GPU: %{x}<br>Cluster: %{y}<br>Utilization: %{z}%<extra></extra>",
    ))
    fig_heatmap.update_layout(
        xaxis_title="GPU ID",
        yaxis_title="Cluster",
        height=max(300, len(pivot_df) * 40 + 100),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("No telemetry data available for the selected filters.")

# ---------------------------------------------------------------------------
# GPU Temperature Time Series
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("GPU Temperature Over Time")

timeseries_sql = f"""
    SELECT
        DATE_TRUNC('hour', t.timestamp) AS time_bucket,
        c.cluster_name,
        ROUND(AVG(t.temperature_c), 1) AS avg_temp,
        ROUND(MAX(t.temperature_c), 1) AS max_temp
    FROM {CATALOG}.{SCHEMA}.gpu_telemetry t
    JOIN {CATALOG}.{SCHEMA}.cluster_inventory c ON t.cluster_id = c.cluster_id
    WHERE {telemetry_filter}
    GROUP BY 1, 2
    ORDER BY 1
"""

with st.spinner("Loading temperature chart..."):
    ts_df = run_query(timeseries_sql)

if not ts_df.empty:
    fig_temp = px.line(
        ts_df,
        x="time_bucket",
        y="avg_temp",
        color="cluster_name",
        markers=True,
        labels={
            "time_bucket": "Time",
            "avg_temp": "Avg Temperature (°C)",
            "cluster_name": "Cluster",
        },
        title="Average GPU Temperature by Cluster",
    )
    # Add a threshold line at 85°C
    fig_temp.add_hline(
        y=85,
        line_dash="dash",
        line_color="red",
        annotation_text="Thermal Threshold (85°C)",
        annotation_position="top left",
    )
    fig_temp.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_temp, use_container_width=True)
else:
    st.info("No temperature data available for the selected filters.")

# ---------------------------------------------------------------------------
# Recent Alerts Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Recent Alerts")

alerts_sql = f"""
    SELECT
        a.event_time,
        c.cluster_name,
        a.gpu_id,
        a.severity,
        a.event_type,
        a.description
    FROM {CATALOG}.{SCHEMA}.gpu_health_events a
    JOIN {CATALOG}.{SCHEMA}.cluster_inventory c ON a.cluster_id = c.cluster_id
    WHERE {alerts_filter}
    ORDER BY a.event_time DESC
    LIMIT 50
"""

with st.spinner("Loading alerts..."):
    alerts_df = run_query(alerts_sql)

if not alerts_df.empty:
    def severity_color(severity):
        colors = {
            "CRITICAL": "background-color: #ff4d4d; color: white; font-weight: bold;",
            "WARNING": "background-color: #ffcc00; color: black; font-weight: bold;",
            "INFO": "background-color: #4da6ff; color: white;",
        }
        return colors.get(severity, "")

    def style_severity(row):
        return [severity_color(row["severity"]) if col == "severity" else "" for col in row.index]

    styled_alerts = alerts_df.style.apply(style_severity, axis=1)
    st.dataframe(styled_alerts, use_container_width=True, height=400)
else:
    st.info("No alerts for the selected filters.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data source: `{CATALOG}.{SCHEMA}` | "
    f"SQL Warehouse: `{SQL_WAREHOUSE_ID}` | "
    f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
'''

print(f"app.py code length: {len(app_py_code)} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Write the App File to a Workspace Volume
# MAGIC
# MAGIC We write `app.py` to a Unity Catalog volume so Databricks Apps can reference it.

# COMMAND ----------

import os

volume_path = "/Volumes/main/mlops_genai_workshop/apps/gpu_fleet_monitor"

# Create the directory structure
os.makedirs(volume_path, exist_ok=True)

# Write app.py
app_file = os.path.join(volume_path, "app.py")
with open(app_file, "w") as f:
    f.write(app_py_code)

print(f"Wrote app.py to {app_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create the `app.yaml` Configuration
# MAGIC
# MAGIC The `app.yaml` tells Databricks Apps how to run our Streamlit application --
# MAGIC the command to execute, environment variables, and resource permissions.

# COMMAND ----------

app_yaml_code = """# GPU Fleet Monitor - Databricks App Configuration
# This file tells Databricks Apps how to run the Streamlit application.

command:
  - "streamlit"
  - "run"
  - "app.py"
  - "--server.port"
  - "8501"
  - "--server.address"
  - "0.0.0.0"

env:
  - name: SQL_WAREHOUSE_ID
    value: "75fd8278393d07eb"
  - name: CATALOG
    value: "main"
  - name: SCHEMA
    value: "mlops_genai_workshop"

# Resources the app needs access to
resources:
  - name: sql-warehouse
    sql_warehouse:
      id: "75fd8278393d07eb"
      permission: "CAN_USE"
"""

app_yaml_file = os.path.join(volume_path, "app.yaml")
with open(app_yaml_file, "w") as f:
    f.write(app_yaml_code)

print(f"Wrote app.yaml to {app_yaml_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create `requirements.txt`
# MAGIC
# MAGIC Pin the Python dependencies so the deployed environment is reproducible.

# COMMAND ----------

requirements_txt = """streamlit>=1.30.0
plotly>=5.18.0
pandas>=2.0.0
databricks-sdk>=0.20.0
databricks-sql-connector>=3.0.0
"""

req_file = os.path.join(volume_path, "requirements.txt")
with open(req_file, "w") as f:
    f.write(requirements_txt)

print(f"Wrote requirements.txt to {req_file}")

# Verify all files are written
for fname in os.listdir(volume_path):
    fpath = os.path.join(volume_path, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname:25s} {size:>6,} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deploy the Databricks App
# MAGIC
# MAGIC We use the Databricks SDK to create and deploy the app programmatically.
# MAGIC
# MAGIC > **Note:** The deployment takes 2-3 minutes. The cell will poll until the app is running.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppResource, AppResourceSqlWarehouse

w = WorkspaceClient()

APP_NAME = "gpu-fleet-monitor"

# ---------------------------------------------------------------------------
# Create or update the app
# ---------------------------------------------------------------------------
print(f"Creating app '{APP_NAME}'...")

try:
    app = w.apps.create(
        name=APP_NAME,
        description="GPU Fleet Monitor - Real-time NVIDIA DGX Cloud GPU telemetry dashboard",
        resources=[
            AppResource(
                name="sql-warehouse",
                sql_warehouse=AppResourceSqlWarehouse(
                    id="75fd8278393d07eb",
                    permission="CAN_USE",
                ),
            )
        ],
    )
    print(f"App created: {app.name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"App '{APP_NAME}' already exists, proceeding to deploy...")
        app = w.apps.get(APP_NAME)
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Source Code
# MAGIC
# MAGIC Now we push the app source from the volume to the app deployment.

# COMMAND ----------

print(f"Deploying app '{APP_NAME}' from {volume_path}...")

deployment = w.apps.deploy(
    app_name=APP_NAME,
    source_code_path=volume_path,
)

print(f"Deployment started: {deployment.deployment_id}")
print(f"Status: {deployment.status.state if deployment.status else 'PENDING'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for Deployment to Complete

# COMMAND ----------

import time

MAX_WAIT_SECONDS = 300
POLL_INTERVAL = 15

print(f"Waiting for deployment to complete (timeout: {MAX_WAIT_SECONDS}s)...")

elapsed = 0
while elapsed < MAX_WAIT_SECONDS:
    app_status = w.apps.get(APP_NAME)
    state = app_status.active_deployment.status.state if app_status.active_deployment and app_status.active_deployment.status else "UNKNOWN"
    print(f"  [{elapsed:>3}s] State: {state}")

    if state in ("SUCCEEDED", "RUNNING"):
        print(f"\nApp is live!")
        print(f"URL: {app_status.url}")
        break
    elif state in ("FAILED", "CANCELLED"):
        print(f"\nDeployment failed with state: {state}")
        if app_status.active_deployment and app_status.active_deployment.status:
            print(f"Message: {app_status.active_deployment.status.message}")
        break

    time.sleep(POLL_INTERVAL)
    elapsed += POLL_INTERVAL
else:
    print(f"\nTimed out after {MAX_WAIT_SECONDS}s. Check the Apps UI for status.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Verify the Deployment
# MAGIC
# MAGIC Let's confirm the app is running and retrieve its URL.

# COMMAND ----------

app_info = w.apps.get(APP_NAME)

print("=" * 60)
print(f"  App Name:    {app_info.name}")
print(f"  App URL:     {app_info.url}")
print(f"  Status:      {app_info.status.state if app_info.status else 'N/A'}")
print(f"  Creator:     {app_info.creator}")
print("=" * 60)

if app_info.url:
    displayHTML(f"""
    <div style="padding: 20px; background: #1a1a2e; border-radius: 10px; text-align: center;">
        <h2 style="color: #76b900;">GPU Fleet Monitor is Live!</h2>
        <p style="color: #ccc; font-size: 16px;">Click below to open your dashboard:</p>
        <a href="{app_info.url}" target="_blank"
           style="display: inline-block; padding: 12px 24px; background: #76b900; color: #000;
                  text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 18px;">
            Open GPU Fleet Monitor
        </a>
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Deploy via Databricks CLI
# MAGIC
# MAGIC If you prefer CLI-based deployment, you can run these commands from a terminal:
# MAGIC
# MAGIC ```bash
# MAGIC # Navigate to the app source directory
# MAGIC cd /Volumes/main/mlops_genai_workshop/apps/gpu_fleet_monitor
# MAGIC
# MAGIC # Create the app
# MAGIC databricks apps create gpu-fleet-monitor \
# MAGIC   --description "GPU Fleet Monitor - NVIDIA DGX Cloud GPU telemetry dashboard"
# MAGIC
# MAGIC # Deploy the app
# MAGIC databricks apps deploy gpu-fleet-monitor \
# MAGIC   --source-code-path /Volumes/main/mlops_genai_workshop/apps/gpu_fleet_monitor
# MAGIC
# MAGIC # Check deployment status
# MAGIC databricks apps get gpu-fleet-monitor
# MAGIC
# MAGIC # View logs if needed
# MAGIC databricks apps logs gpu-fleet-monitor
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC Uncomment and run the cell below to delete the app when you are done.

# COMMAND ----------

# # Uncomment to delete the app
# w.apps.delete(APP_NAME)
# print(f"App '{APP_NAME}' deleted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this block you:
# MAGIC
# MAGIC | Step | What you did |
# MAGIC |------|-------------|
# MAGIC | 1 | Authored a complete Streamlit app with KPIs, heatmap, time series, and alerts |
# MAGIC | 2 | Wrote source files (`app.py`, `app.yaml`, `requirements.txt`) to a UC volume |
# MAGIC | 3 | Created and deployed the app using the Databricks Python SDK |
# MAGIC | 4 | Verified the live URL and opened the GPU Fleet Monitor dashboard |
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC - **Zero infrastructure** -- no Docker, no Kubernetes, no port forwarding
# MAGIC - **Governed by default** -- Unity Catalog permissions flow through to the app
# MAGIC - **Built-in auth** -- workspace SSO, no API keys to manage
# MAGIC - **Direct SQL access** -- query the same warehouse used by your notebooks and dashboards
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next:** Return to the workshop guide for the wrap-up and Q&A session.
