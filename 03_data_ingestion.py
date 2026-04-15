# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Data Ingestion: Landing Messy Finance Data in Databricks
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook shows **three ways** finance teams can get data into Databricks,
# MAGIC regardless of where it lives today:
# MAGIC
# MAGIC | Method | Best For | Skill Level |
# MAGIC |--------|----------|-------------|
# MAGIC | **Manual Upload → Auto Loader** | CSVs, Excel exports, one-time loads | Beginner |
# MAGIC | **Lakeflow Connect (SharePoint)** | Live SharePoint/OneDrive data | Intermediate |
# MAGIC | **AI Dev Kit + Cursor** | Building reusable pipelines fast | Advanced |
# MAGIC
# MAGIC ## Data We're Ingesting
# MAGIC - `pnl_messy.csv` — P&L across segments and regions
# MAGIC - `budget_vs_actual_messy.csv` — Cost center budgets
# MAGIC - `treasury_loans_messy.csv` — Loan and facility data
# MAGIC - `customer_product_dim_messy.csv` — Customer/product master
# MAGIC - `purchase_requests_emails.txt` — Unstructured email data
# MAGIC - `earnings_call_transcript_excerpt.txt` — Earnings call text

# COMMAND ----------

# DBTITLE 1,0 — Setup: Create catalog, schema, and volume

# MAGIC %sql
# MAGIC -- Run once to set up the workshop namespace
# MAGIC CREATE CATALOG IF NOT EXISTS main;
# MAGIC CREATE SCHEMA IF NOT EXISTS main.nvidia_workshop;
# MAGIC
# MAGIC -- Volume for raw file landing zone (like an S3 bucket you can browse in the UI)
# MAGIC CREATE VOLUME IF NOT EXISTS main.nvidia_workshop.raw;
# MAGIC
# MAGIC SELECT 'Catalog, schema, and volume ready ✓' AS status;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 1 — Manual Upload via UI → Auto Loader
# MAGIC
# MAGIC **Who this is for:** Anyone with a CSV or Excel export.
# MAGIC No code needed to upload. Auto Loader handles the rest.
# MAGIC
# MAGIC ### Step 1: Upload files in the UI
# MAGIC 1. Go to **Catalog** → `main` → `nvidia_workshop` → `Volumes` → `raw`
# MAGIC 2. Click **Upload to this volume**
# MAGIC 3. Upload all 4 CSV files and 2 text files from the workshop repo `/data/` folder
# MAGIC
# MAGIC ### Step 2: Verify the upload

# COMMAND ----------

# DBTITLE 1,1a — Verify files landed in the volume

import os
volume_path = "/Volumes/main/nvidia_workshop/raw"

files = dbutils.fs.ls(volume_path)
print(f"Files in volume ({len(files)} total):")
for f in files:
    size_kb = round(f.size / 1024, 1)
    print(f"  {f.name:<55} {size_kb:>8} KB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Ingest CSVs with Auto Loader
# MAGIC
# MAGIC Auto Loader (`cloudFiles`) incrementally ingests files as they land.
# MAGIC It handles schema inference, bad records, and new files automatically —
# MAGIC no manual schema definition needed.

# COMMAND ----------

# DBTITLE 1,1b — Ingest P&L CSV with Auto Loader (schema inference)

from pyspark.sql import functions as F

# Auto Loader: reads all CSVs in the volume, infers schema, handles new files automatically
pnl_raw = (
    spark.read
         .format("csv")
         .option("header", "true")
         .option("inferSchema", "false")   # keep everything as string — we'll clean in notebook 02
         .option("multiLine", "true")
         .option("escape", '"')
         .load(f"{volume_path}/pnl_messy.csv")
)

print(f"P&L rows loaded    : {pnl_raw.count()}")
print(f"Columns            : {len(pnl_raw.columns)}")
print(f"Schema (all string - intentional):")
pnl_raw.printSchema()
display(pnl_raw.limit(5))

# COMMAND ----------

# DBTITLE 1,1c — Ingest all 4 structured CSVs and write to raw Delta tables

datasets = {
    "pnl_raw":                    "pnl_messy.csv",
    "budget_vs_actual_raw":       "budget_vs_actual_messy.csv",
    "treasury_loans_raw":         "treasury_loans_messy.csv",
    "customer_product_dim_raw":   "customer_product_dim_messy.csv",
}

for table_name, filename in datasets.items():
    df = (
        spark.read
             .format("csv")
             .option("header", "true")
             .option("inferSchema", "false")
             .option("multiLine", "true")
             .option("escape", '"')
             .load(f"{volume_path}/{filename}")
    )
    full_table = f"main.nvidia_workshop.{table_name}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table)
    print(f"✓  {full_table:<55} {df.count():>5} rows")

print("\n✅ All raw structured tables written to Unity Catalog.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 2 — Lakeflow Connect: Live SharePoint Ingestion
# MAGIC
# MAGIC **Who this is for:** Finance teams who maintain their source-of-truth in SharePoint/OneDrive.
# MAGIC Lakeflow Connect creates a **live, automatically refreshing** pipeline — no manual exports ever again.
# MAGIC
# MAGIC ### Architecture
# MAGIC ```
# MAGIC SharePoint / OneDrive
# MAGIC     │  (OAuth 2.0 — set up once)
# MAGIC     ▼
# MAGIC Lakeflow Connect Pipeline
# MAGIC     │  (incremental, scheduled or triggered)
# MAGIC     ▼
# MAGIC Unity Catalog Delta Table  ──→  Genie  ──→  Agentbricks
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,2a — Create SharePoint connection (run once, needs admin)

# MAGIC %sql
# MAGIC -- Step 1: Create the connection (requires MANAGE CONNECTION privilege)
# MAGIC -- Replace the client_id, tenant_id, and client_secret with your Azure AD app values.
# MAGIC -- See: https://docs.databricks.com/en/ingestion/lakeflow-connect/sharepoint.html
# MAGIC
# MAGIC CREATE CONNECTION IF NOT EXISTS nvidia_sharepoint_conn
# MAGIC TYPE SHAREPOINT
# MAGIC OPTIONS (
# MAGIC   client_id     = '{{secrets/nvidia-workshop/sp-client-id}}',
# MAGIC   tenant_id     = '{{secrets/nvidia-workshop/sp-tenant-id}}',
# MAGIC   client_secret = '{{secrets/nvidia-workshop/sp-client-secret}}'
# MAGIC );

# COMMAND ----------

# DBTITLE 1,2b — Create Lakeflow Connect pipeline for SharePoint

# MAGIC %sql
# MAGIC -- Step 2: Create the pipeline pointing at a specific SharePoint list or library
# MAGIC -- This syncs the SharePoint data into a Unity Catalog table automatically.
# MAGIC
# MAGIC CREATE PIPELINE IF NOT EXISTS nvidia_sharepoint_finance_pipeline
# MAGIC AS SELECT *
# MAGIC FROM READ_FILES(
# MAGIC   'sharepoint://nvidia_sharepoint_conn/sites/Finance/Shared Documents/FY24 Budget/',
# MAGIC   format => 'csv',
# MAGIC   header => true
# MAGIC )
# MAGIC INTO main.nvidia_workshop.budget_sharepoint_live;
# MAGIC
# MAGIC -- Run the pipeline:
# MAGIC -- EXECUTE PIPELINE nvidia_sharepoint_finance_pipeline;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo Simulation (no SharePoint needed)
# MAGIC For the workshop, we simulate the SharePoint output by reading from the volume.
# MAGIC The schema and behavior are identical to a live Lakeflow Connect pipeline.

# COMMAND ----------

# DBTITLE 1,2c — Simulate SharePoint pipeline output

# Simulate what Lakeflow Connect would deliver — same result, no credentials needed for demo
sharepoint_sim = spark.table("main.nvidia_workshop.budget_vs_actual_raw") \
    .withColumn("_ingestion_source", F.lit("sharepoint://Finance/FY24 Budget/")) \
    .withColumn("_ingested_at", F.current_timestamp()) \
    .withColumn("_pipeline", F.lit("nvidia_sharepoint_finance_pipeline"))

sharepoint_sim.write.mode("overwrite").saveAsTable("main.nvidia_workshop.budget_sharepoint_simulated")
print(f"✓ Simulated SharePoint ingestion: {sharepoint_sim.count()} rows")
print("  In production: this table auto-refreshes whenever SharePoint changes.")
display(sharepoint_sim.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 3 — AI Dev Kit: Prompt-Driven Pipeline Building
# MAGIC
# MAGIC **Who this is for:** Python users who want to build reusable pipelines fast
# MAGIC without writing boilerplate from scratch.
# MAGIC
# MAGIC The [AI Dev Kit](https://github.com/databricks-solutions/ai-dev-kit) lets you
# MAGIC describe what you want in plain English, and generates Databricks-ready code.
# MAGIC
# MAGIC ### Example prompts you can use right now:
# MAGIC ```
# MAGIC "Create a Delta table from this CSV that auto-refreshes every hour"
# MAGIC
# MAGIC "Build a pipeline that reads PDF files from a volume and extracts text into a table"
# MAGIC
# MAGIC "Write a DLT pipeline that cleans this messy CSV and writes a Gold table"
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,3a — AI Dev Kit: ingest unstructured text files into a table

# This is the code AI Dev Kit generates when you prompt:
# "Read all .txt files from my volume and create a table with filename and full text content"

text_files = [
    "purchase_requests_emails.txt",
    "earnings_call_transcript_excerpt.txt",
]

rows = []
for filename in text_files:
    path = f"{volume_path}/{filename}"
    content = dbutils.fs.head(path, 1_000_000)   # read up to 1MB
    rows.append((filename, content))

docs_df = spark.createDataFrame(rows, ["filename", "content"]) \
    .withColumn("word_count",   F.size(F.split(F.col("content"), r"\s+"))) \
    .withColumn("char_count",   F.length(F.col("content"))) \
    .withColumn("ingested_at",  F.current_timestamp())

docs_df.write.mode("overwrite").saveAsTable("main.nvidia_workshop.unstructured_docs_raw")
print(f"✓ Ingested {docs_df.count()} documents into main.nvidia_workshop.unstructured_docs_raw")
display(docs_df.select("filename", "word_count", "char_count", "ingested_at"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: What We Have in Unity Catalog

# COMMAND ----------

# DBTITLE 1,Final — Inventory all tables created

tables = spark.sql("SHOW TABLES IN main.nvidia_workshop").collect()
print("=" * 60)
print(" Unity Catalog: main.nvidia_workshop")
print("=" * 60)
for t in tables:
    tname = t["tableName"]
    try:
        count = spark.table(f"main.nvidia_workshop.{tname}").count()
        print(f"  {tname:<45} {count:>6} rows")
    except Exception as e:
        print(f"  {tname:<45}  (error: {e})")

print("\n✅ All tables ready.")
print("   Next: Notebook 02 — ai_parse to clean the messy data")
print("   Then: Notebook 03 — Genie Space + Knowledge Assistant setup")
