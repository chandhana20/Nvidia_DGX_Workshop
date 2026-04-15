# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — AI Parse: Extracting Structure from Messy Finance Data
# MAGIC
# MAGIC ## The Problem
# MAGIC Finance teams deal with data that is:
# MAGIC - **Inconsistent formats**: dates as "Q1 2024", "01/01/2024", "Jan-2024" in the same column
# MAGIC - **Mixed currencies**: "$1.2M", "EUR 1200000", "¥ 850K" — no standard unit
# MAGIC - **Buried in text**: vendor names, amounts, and cost centers inside email bodies
# MAGIC - **Duplicate entries**: same invoice from 3 ERP systems with slightly different values
# MAGIC - **Null chaos**: "", "N/A", "TBD", "NULL", "-" all meaning the same thing
# MAGIC
# MAGIC ## What We'll Do
# MAGIC Use Databricks `ai_parse()` and `ai_extract()` to turn this chaos into clean, typed tables.
# MAGIC
# MAGIC | Step | Input | Output |
# MAGIC |------|-------|--------|
# MAGIC | 1 | Raw P&L CSV with mixed formats | Normalized Delta table |
# MAGIC | 2 | Email text blobs | Structured JSON (vendor, amount, cost center, urgency) |
# MAGIC | 3 | Loan status field variants | Canonical status enum |
# MAGIC | 4 | Mixed currency strings | USD-normalized float |
# MAGIC | 5 | Customer name variants | Canonical entity + confidence |

# COMMAND ----------

# DBTITLE 1,Setup — Load raw messy data

import re
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Load the messy P&L
pnl_raw = spark.read.option("header", True).csv(
    "/Volumes/main/nvidia_workshop/raw/pnl_messy.csv"
)

print(f"Raw P&L rows: {pnl_raw.count()}")
print(f"Columns: {pnl_raw.columns}")
display(pnl_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Normalize Revenue Column with `ai_parse()`
# MAGIC
# MAGIC The `Revenue` column contains values like:
# MAGIC - `"$22,100,000"` → 22100000.0
# MAGIC - `"22.1M"` → 22100000.0
# MAGIC - `"EUR 18500000"` → needs FX conversion flag
# MAGIC - `"N/A"`, `""`, `"-"` → NULL
# MAGIC
# MAGIC We use `ai_parse()` to extract the numeric value and currency in one step.

# COMMAND ----------

# DBTITLE 1,Step 1 — Extract numeric value + currency from messy Revenue string

pnl_parsed = pnl_raw.withColumn(
    "revenue_extracted",
    F.expr("""
        ai_parse(Revenue, 'Extract the numeric dollar amount and currency code.
        Return JSON: {"amount": <float or null>, "currency": <3-letter ISO code or "UNKNOWN">}
        Examples:
        "$22,100,000" -> {"amount": 22100000.0, "currency": "USD"}
        "22.1M" -> {"amount": 22100000.0, "currency": "USD"}
        "EUR 1200000" -> {"amount": 1200000.0, "currency": "EUR"}
        "¥ 850K" -> {"amount": 850000.0, "currency": "JPY"}
        "N/A" -> {"amount": null, "currency": "UNKNOWN"}
        "-" -> {"amount": null, "currency": "UNKNOWN"}
        "TBD" -> {"amount": null, "currency": "UNKNOWN"}')
    """)
).withColumn(
    "revenue_amount",   F.get_json_object(F.col("revenue_extracted"), "$.amount").cast("double")
).withColumn(
    "revenue_currency", F.get_json_object(F.col("revenue_extracted"), "$.currency")
)

display(
    pnl_parsed.select("Revenue", "revenue_amount", "revenue_currency").limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Normalize Date/Period Column
# MAGIC
# MAGIC The `Period` column has 4 different formats. We need a single canonical `YYYY-QN` format.

# COMMAND ----------

# DBTITLE 1,Step 2 — Standardize period format

pnl_dated = pnl_parsed.withColumn(
    "period_normalized",
    F.expr("""
        ai_parse(Period, 'Convert this period string to canonical format YYYY-QN.
        Examples:
        "Q2 2024" -> "2024-Q2"
        "2024-Q3" -> "2024-Q3"
        "Jan-2023" -> "2023-Q1"
        "Apr-2024" -> "2024-Q2"
        "Jul-2023" -> "2023-Q3"
        "Oct-2024" -> "2024-Q4"
        "01/01/2024" -> "2024-Q1"
        "04/01/2024" -> "2024-Q2"
        Return only the canonical string, nothing else.')
    """)
)

display(
    pnl_dated.select("Period", "period_normalized").distinct().orderBy("period_normalized").limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Canonicalize Business Segment Names
# MAGIC
# MAGIC "Data Center", "datacenter", "DC", "Data Ctr.", "DATA CENTER" → all mean the same thing.
# MAGIC Use `ai_parse()` to map to one of 4 canonical values.

# COMMAND ----------

# DBTITLE 1,Step 3 — Canonical segment names

pnl_segmented = pnl_dated.withColumn(
    "segment_canonical",
    F.expr("""
        ai_parse(Business_Segment, 'Map this business segment name to exactly one of these
        canonical values: ["Data Center", "Gaming", "Automotive", "Professional Visualization"].
        Be flexible with abbreviations and casing.
        Examples:
        "DC" -> "Data Center"
        "datacenter" -> "Data Center"
        "Data Ctr." -> "Data Center"
        "Gming" -> "Gaming"
        "AUTO" -> "Automotive"
        "ProViz" -> "Professional Visualization"
        "Pro-Viz" -> "Professional Visualization"
        Return only the canonical string.')
    """)
)

display(
    pnl_segmented
    .select("Business_Segment", "segment_canonical")
    .distinct()
    .orderBy("segment_canonical")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Normalize Loan Status Field
# MAGIC
# MAGIC The treasury loan file has 15+ variants of the same 5 statuses.

# COMMAND ----------

# DBTITLE 1,Step 4 — Canonical loan status

loans_raw = spark.read.option("header", True).csv(
    "/Volumes/main/nvidia_workshop/raw/treasury_loans_messy.csv"
)

loans_clean = loans_raw.withColumn(
    "status_canonical",
    F.expr("""
        ai_parse(Loan_Status, 'Map this loan status to exactly one of:
        ["Active", "In Default", "Paid Off", "Deferred", "Under Review", "Unknown"].
        Examples:
        "ACTIVE" -> "Active"
        "active" -> "Active"
        "DEFAULT" -> "In Default"
        "in default" -> "In Default"
        "PAID_OFF" -> "Paid Off"
        "paid-off" -> "Paid Off"
        "DEFERRED" -> "Deferred"
        "REVIEW" -> "Under Review"
        "Under Review" -> "Under Review"
        "" -> "Unknown"
        "N/A" -> "Unknown"
        Return only the canonical string.')
    """)
)

display(
    loans_clean.select("Loan_Status", "status_canonical").distinct().orderBy("status_canonical")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Information Extraction from Email Text
# MAGIC
# MAGIC The most powerful use case: pull structured fields out of unstructured email bodies.
# MAGIC This is what replaces **5 hours of manual line-by-line reformatting** for purchase requests.

# COMMAND ----------

# DBTITLE 1,Step 5 — Extract purchase request fields from email body

# Load email text as a single-column DataFrame
emails_df = spark.createDataFrame([
    (1, open("/Volumes/main/nvidia_workshop/raw/purchase_requests_emails.txt").read().split("="*92)[1]),
    (2, open("/Volumes/main/nvidia_workshop/raw/purchase_requests_emails.txt").read().split("="*92)[3]),
    (3, open("/Volumes/main/nvidia_workshop/raw/purchase_requests_emails.txt").read().split("="*92)[5]),
    (4, open("/Volumes/main/nvidia_workshop/raw/purchase_requests_emails.txt").read().split("="*92)[7]),
], ["email_id", "email_body"])

extracted = emails_df.withColumn(
    "parsed_fields",
    F.expr("""
        ai_extract(email_body, array(
            'vendor_name',
            'product_description',
            'quantity',
            'total_amount_usd',
            'cost_center',
            'requested_by',
            'urgency_level',
            'shipping_address',
            'approval_status'
        ))
    """)
).select(
    "email_id",
    F.get_json_object("parsed_fields", "$.vendor_name").alias("vendor_name"),
    F.get_json_object("parsed_fields", "$.product_description").alias("product_description"),
    F.get_json_object("parsed_fields", "$.quantity").alias("quantity"),
    F.get_json_object("parsed_fields", "$.total_amount_usd").alias("total_amount_usd"),
    F.get_json_object("parsed_fields", "$.cost_center").alias("cost_center"),
    F.get_json_object("parsed_fields", "$.requested_by").alias("requested_by"),
    F.get_json_object("parsed_fields", "$.urgency_level").alias("urgency_level"),
    F.get_json_object("parsed_fields", "$.approval_status").alias("approval_status"),
)

print("Structured purchase request data extracted from raw emails:")
display(extracted)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Deduplicate with Fuzzy Entity Matching
# MAGIC
# MAGIC The customer dimension has "Dell Technologies", "Dell Tech", "DELL", "Dell Inc." — all the same entity.
# MAGIC Use `ai_similarity_score()` to cluster duplicates.

# COMMAND ----------

# DBTITLE 1,Step 6 — Fuzzy customer name deduplication

customers_raw = spark.read.option("header", True).csv(
    "/Volumes/main/nvidia_workshop/raw/customer_product_dim_messy.csv"
)

# Get distinct customer names and canonicalize
customers_canonical = customers_raw.select("Customer_Name").distinct().withColumn(
    "canonical_name",
    F.expr("""
        ai_parse(Customer_Name, 'Map this company name to its canonical legal entity name.
        Known mappings:
        Any variant of "Dell" -> "Dell Technologies Inc."
        Any variant of "Microsoft" or "MSFT" -> "Microsoft Corporation"
        Any variant of "Meta" or "Facebook" -> "Meta Platforms Inc."
        Any variant of "Amazon" or "AWS" -> "Amazon Web Services Inc."
        Any variant of "Google" or "Alphabet" -> "Alphabet Inc."
        Any variant of "Cisco" or "CSC" -> "Cisco Systems Inc."
        Any variant of "HPE" or "Hewlett Packard" -> "Hewlett Packard Enterprise"
        Any variant of "Oracle" or "ORCL" -> "Oracle Corporation"
        Return only the canonical name.')
    """)
)

print("Customer name canonicalization:")
display(customers_canonical.orderBy("canonical_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Write Clean Tables to Unity Catalog
# MAGIC
# MAGIC Everything clean, typed, and ready for Genie and the Knowledge Assistant.

# COMMAND ----------

# DBTITLE 1,Step 7 — Write clean tables

# Final clean P&L
pnl_final = pnl_segmented.select(
    F.col("period_normalized").alias("period"),
    F.col("segment_canonical").alias("business_segment"),
    "Region",
    F.col("revenue_amount").alias("revenue_usd"),
    F.col("revenue_currency"),
    "GL_Account_Code",
    "Source_System",
).filter(F.col("revenue_amount").isNotNull())

pnl_final.write.mode("overwrite").saveAsTable("main.nvidia_workshop.pnl_clean")
print(f"✓ Wrote {pnl_final.count()} rows to main.nvidia_workshop.pnl_clean")

# Final clean loans
loans_final = loans_clean.select(
    "Loan_ID", "Counterparty", "Loan_Type",
    F.col("status_canonical").alias("loan_status"),
    "Currency", "Maturity_Date", "Covenant_Breach", "Notes"
)
loans_final.write.mode("overwrite").saveAsTable("main.nvidia_workshop.treasury_loans_clean")
print(f"✓ Wrote {loans_final.count()} rows to main.nvidia_workshop.treasury_loans_clean")

print("\n✅ Clean tables ready for Genie Space and Knowledge Assistant ingestion.")
