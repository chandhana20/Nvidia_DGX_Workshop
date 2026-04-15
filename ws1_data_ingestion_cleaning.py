# Databricks notebook source

# MAGIC %md
# MAGIC # Workshop 1 — Ingest & Clean Financial Documents (30 min)
# MAGIC
# MAGIC ## Use Case: Financial Due Diligence
# MAGIC You're evaluating whether to invest in or acquire a company. Your first task:
# MAGIC get the raw financial filings into a clean, queryable format.
# MAGIC
# MAGIC ## What We'll Do
# MAGIC | Step | Time | What | Output |
# MAGIC |------|------|------|--------|
# MAGIC | 1 | 5 min | Upload PDFs to Volume | Raw files in Unity Catalog |
# MAGIC | 2 | 10 min | Parse PDFs into text | `parsed_filings` table |
# MAGIC | 3 | 10 min | Extract structured financials with AI | `company_financials` table |
# MAGIC | 4 | 5 min | Verify and explore | Clean data ready for Genie |
# MAGIC
# MAGIC ## Documents Available
# MAGIC **164 PDFs** across 7 companies (NVIDIA, Apple, Amazon, Google, Meta, Microsoft, Tesla):
# MAGIC - 10-K annual filings, 10-Q quarterly filings
# MAGIC - Earnings releases, call transcripts, annual reports

# COMMAND ----------

# DBTITLE 1,Config

CATALOG = "main"
SCHEMA = "fins_due_diligence"
VOLUME = "raw_filings"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# COMMAND ----------

# DBTITLE 1,Step 0 — Create schema and volume

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS main.fins_due_diligence;
# MAGIC CREATE VOLUME IF NOT EXISTS main.fins_due_diligence.raw_filings;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Upload PDFs to Volume
# MAGIC
# MAGIC **Option A (UI):** Go to Catalog → main → fins_due_diligence → Volumes → raw_filings → Upload
# MAGIC
# MAGIC **Option B (Code):** Run the cell below if the PDFs are in a repo/workspace path.
# MAGIC
# MAGIC We organize files into subfolders by document type: `10K/`, `10Q/`, `Earning Releases/`, etc.

# COMMAND ----------

# DBTITLE 1,1a — Upload from workspace (adjust path to your repo)

import os

# Point this to wherever you cloned the PDFs
LOCAL_DATA_DIR = "/Workspace/Repos/ai-dev-day/data"

DOC_FOLDERS = ["10K", "10Q", "Earning Releases", "Call Transcripts", "Annual Report"]

uploaded = 0
for folder in DOC_FOLDERS:
    src_dir = os.path.join(LOCAL_DATA_DIR, folder)
    if not os.path.exists(src_dir):
        print(f"  Skip (not found): {src_dir}")
        continue
    dbutils.fs.mkdirs(f"{VOLUME_PATH}/{folder}")
    for f in os.listdir(src_dir):
        if f.lower().endswith(".pdf"):
            dbutils.fs.cp(f"file:{os.path.join(src_dir, f)}", f"{VOLUME_PATH}/{folder}/{f}")
            uploaded += 1

print(f"Uploaded {uploaded} PDFs to {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,1b — Verify upload

for folder in DOC_FOLDERS:
    try:
        files = [f for f in dbutils.fs.ls(f"{VOLUME_PATH}/{folder}") if f.name.endswith(".pdf")]
        print(f"  {folder:<25} {len(files):>3} PDFs")
    except Exception:
        print(f"  {folder:<25}   0 PDFs (folder missing)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Parse PDFs into Text
# MAGIC
# MAGIC Read all PDFs as binary, extract text with `pypdf`, and tag each document
# MAGIC with metadata: **company ticker**, **document type**, and **filing period**.

# COMMAND ----------

# DBTITLE 1,2a — Install pypdf

# MAGIC %pip install pypdf -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,2b — Parse PDFs and tag metadata

import re
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

CATALOG = "main"
SCHEMA = "fins_due_diligence"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_filings"

# --- UDF: extract text from PDF bytes ---
def extract_pdf_text(content: bytes) -> str:
    from pypdf import PdfReader
    from io import BytesIO
    try:
        reader = PdfReader(BytesIO(content))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        return f"[PARSE_ERROR] {e}"

extract_udf = F.udf(extract_pdf_text, StringType())


# --- UDF: infer company ticker from filename ---
TICKER_KEYWORDS = {
    "nvidia": "NVDA", "nvda": "NVDA",
    "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN",
    "google": "GOOGL", "goog": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "microsoft": "MSFT", "msft": "MSFT",
    "tesla": "TSLA", "tsla": "TSLA",
}

def infer_ticker(filename: str) -> str:
    fn = filename.lower().replace("-", "").replace("_", "")
    for kw, ticker in sorted(TICKER_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in fn:
            return ticker
    return "UNKNOWN"

ticker_udf = F.udf(infer_ticker, StringType())


# --- UDF: infer doc type from path ---
DOC_TYPE_MAP = {
    "10K": "10-K", "10Q": "10-Q",
    "Annual Report": "Annual Report",
    "Earning Releases": "Earnings Release",
    "Call Transcripts": "Earnings Call Transcript",
}

def infer_doc_type(path: str) -> str:
    for folder, dtype in DOC_TYPE_MAP.items():
        if f"/{folder}/" in path:
            return dtype
    return "Other"

doc_type_udf = F.udf(infer_doc_type, StringType())


# --- UDF: infer filing period from filename ---
def infer_period(filename: str) -> str:
    fn = filename.upper()
    # FY25Q4
    m = re.search(r'FY(\d{2})Q(\d)', fn)
    if m: return f"20{m.group(1)}-Q{m.group(2)}"
    # F1Q26 (NVIDIA fiscal format)
    m = re.search(r'F(\d)Q(\d{2})', fn)
    if m: return f"20{m.group(2)}-Q{m.group(1)}"
    # Q125, Q225
    m = re.search(r'Q(\d)(\d{2})[^0-9]', fn)
    if m: return f"20{m.group(2)}-Q{m.group(1)}"
    # Q1-2025 or Q1 2025
    m = re.search(r'Q(\d)[- ]?(\d{4})', fn)
    if m: return f"{m.group(2)}-Q{m.group(1)}"
    # 2024-Q3
    m = re.search(r'(\d{4})[- ]?Q(\d)', fn)
    if m: return f"{m.group(1)}-Q{m.group(2)}"
    # 20231231
    m = re.search(r'(\d{4})(1231|0930|0630|0331)', fn)
    if m:
        qmap = {"0331": "Q1", "0630": "Q2", "0930": "Q3", "1231": "Q4"}
        return f"{m.group(1)}-{qmap[m.group(2)]}"
    # Just a year
    m = re.search(r'20(2[0-9])', fn)
    if m: return f"20{m.group(1)}-FY"
    return "UNKNOWN"

period_udf = F.udf(infer_period, StringType())


# --- Read and parse ---
raw = spark.read.format("binaryFile").option("pathGlobFilter", "*.pdf").load(f"{VOLUME_PATH}/*/*.pdf")

parsed = (
    raw
    .withColumn("text_content", extract_udf(F.col("content")))
    .withColumn("filename", F.element_at(F.split("path", "/"), -1))
    .withColumn("ticker", ticker_udf("filename"))
    .withColumn("doc_type", doc_type_udf("path"))
    .withColumn("filing_period", period_udf("filename"))
    .withColumn("doc_id", F.sha2("path", 256))
    .withColumn("text_length", F.length("text_content"))
    .withColumn("page_count", F.lit(None).cast("int"))  # placeholder
    .withColumn("parsed_at", F.current_timestamp())
    .select(
        "doc_id", "filename", "ticker", "doc_type", "filing_period",
        "text_content", "text_length", "path", "parsed_at"
    )
)

parsed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.parsed_filings"
)

count = spark.table(f"{CATALOG}.{SCHEMA}.parsed_filings").count()
print(f"Parsed {count} documents into {CATALOG}.{SCHEMA}.parsed_filings")

# COMMAND ----------

# DBTITLE 1,2c — Quick look at what we parsed

display(
    spark.table(f"{CATALOG}.{SCHEMA}.parsed_filings")
    .groupBy("ticker", "doc_type")
    .agg(F.count("*").alias("docs"), F.avg("text_length").cast("int").alias("avg_chars"))
    .orderBy("ticker", "doc_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Extract Structured Financials with AI
# MAGIC
# MAGIC This is the magic: `ai_extract()` reads raw text and pulls out typed financial fields.
# MAGIC No regex, no custom parsing — the LLM handles messy formats, page breaks, and table layouts.
# MAGIC
# MAGIC **Two approaches shown side by side:**
# MAGIC - **Direct (this notebook):** Write the `ai_extract()` SQL yourself
# MAGIC - **AI Dev Kit (Claude Code):** Prompt: *"Extract revenue, net income, and EPS from the parsed_filings table using ai_extract"*

# COMMAND ----------

# DBTITLE 1,3a — Extract key financial metrics from 10-K and 10-Q filings

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE main.fins_due_diligence.company_financials AS
# MAGIC
# MAGIC SELECT
# MAGIC   doc_id,
# MAGIC   ticker,
# MAGIC   doc_type,
# MAGIC   filing_period,
# MAGIC   filename,
# MAGIC
# MAGIC   -- AI extracts structured fields from raw text
# MAGIC   ai_extract(
# MAGIC     SUBSTRING(text_content, 1, 8000),
# MAGIC     array(
# MAGIC       'total_revenue_usd',
# MAGIC       'net_income_usd',
# MAGIC       'earnings_per_share_diluted',
# MAGIC       'gross_margin_percent',
# MAGIC       'operating_income_usd',
# MAGIC       'cash_and_equivalents_usd',
# MAGIC       'total_debt_usd',
# MAGIC       'free_cash_flow_usd',
# MAGIC       'revenue_growth_yoy_percent'
# MAGIC     )
# MAGIC   ) AS extracted,
# MAGIC
# MAGIC   current_timestamp() AS extracted_at
# MAGIC
# MAGIC FROM main.fins_due_diligence.parsed_filings
# MAGIC WHERE doc_type IN ('10-K', '10-Q')
# MAGIC   AND text_content NOT LIKE '[PARSE_ERROR]%';

# COMMAND ----------

# DBTITLE 1,3b — Flatten extracted JSON into typed columns

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE main.fins_due_diligence.company_financials_clean AS
# MAGIC
# MAGIC SELECT
# MAGIC   doc_id,
# MAGIC   ticker,
# MAGIC   doc_type,
# MAGIC   filing_period,
# MAGIC   filename,
# MAGIC   CAST(extracted.total_revenue_usd AS DOUBLE) AS revenue,
# MAGIC   CAST(extracted.net_income_usd AS DOUBLE) AS net_income,
# MAGIC   CAST(extracted.earnings_per_share_diluted AS DOUBLE) AS eps_diluted,
# MAGIC   CAST(extracted.gross_margin_percent AS DOUBLE) AS gross_margin_pct,
# MAGIC   CAST(extracted.operating_income_usd AS DOUBLE) AS operating_income,
# MAGIC   CAST(extracted.cash_and_equivalents_usd AS DOUBLE) AS cash_equivalents,
# MAGIC   CAST(extracted.total_debt_usd AS DOUBLE) AS total_debt,
# MAGIC   CAST(extracted.free_cash_flow_usd AS DOUBLE) AS free_cash_flow,
# MAGIC   CAST(extracted.revenue_growth_yoy_percent AS DOUBLE) AS revenue_yoy_growth,
# MAGIC   extracted_at
# MAGIC FROM main.fins_due_diligence.company_financials;

# COMMAND ----------

# DBTITLE 1,3c — Extract earnings call sentiment

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE main.fins_due_diligence.call_transcript_insights AS
# MAGIC
# MAGIC SELECT
# MAGIC   doc_id,
# MAGIC   ticker,
# MAGIC   filing_period,
# MAGIC   filename,
# MAGIC
# MAGIC   ai_extract(
# MAGIC     SUBSTRING(text_content, 1, 10000),
# MAGIC     array(
# MAGIC       'management_sentiment_positive_neutral_negative',
# MAGIC       'top_growth_drivers_mentioned',
# MAGIC       'key_risks_discussed',
# MAGIC       'capex_and_investment_outlook',
# MAGIC       'ai_strategy_commentary',
# MAGIC       'analyst_top_concern'
# MAGIC     )
# MAGIC   ) AS insights,
# MAGIC
# MAGIC   current_timestamp() AS extracted_at
# MAGIC
# MAGIC FROM main.fins_due_diligence.parsed_filings
# MAGIC WHERE doc_type = 'Earnings Call Transcript'
# MAGIC   AND text_content NOT LIKE '[PARSE_ERROR]%';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Verify the Output
# MAGIC
# MAGIC We now have two clean tables ready for Genie:
# MAGIC - `company_financials_clean` — structured metrics per company per quarter
# MAGIC - `call_transcript_insights` — qualitative signals from earnings calls

# COMMAND ----------

# DBTITLE 1,4a — Company financials at a glance

# MAGIC %sql
# MAGIC SELECT ticker, filing_period, doc_type,
# MAGIC        revenue, net_income, eps_diluted, gross_margin_pct, free_cash_flow
# MAGIC FROM main.fins_due_diligence.company_financials_clean
# MAGIC ORDER BY ticker, filing_period DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,4b — Transcript sentiment at a glance

# MAGIC %sql
# MAGIC SELECT ticker, filing_period,
# MAGIC        insights.management_sentiment_positive_neutral_negative AS sentiment,
# MAGIC        insights.top_growth_drivers_mentioned AS growth_drivers,
# MAGIC        insights.analyst_top_concern AS analyst_concern
# MAGIC FROM main.fins_due_diligence.call_transcript_insights
# MAGIC ORDER BY ticker, filing_period DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,4c — Table inventory

# MAGIC %sql
# MAGIC SHOW TABLES IN main.fins_due_diligence;

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Dev Kit Approach
# MAGIC
# MAGIC Everything above can be generated by giving Claude Code these prompts:
# MAGIC
# MAGIC ```
# MAGIC Prompt 1: "Read all PDFs from /Volumes/main/fins_due_diligence/raw_filings,
# MAGIC extract text, tag with company ticker and doc type, write to a Delta table."
# MAGIC
# MAGIC Prompt 2: "Use ai_extract to pull revenue, net income, EPS, gross margin,
# MAGIC and free cash flow from each parsed filing. Write clean typed columns."
# MAGIC
# MAGIC Prompt 3: "Extract management sentiment and growth drivers from the earnings
# MAGIC call transcripts using ai_extract."
# MAGIC ```
# MAGIC
# MAGIC **Key takeaway:** AI Dev Kit generates the same code in ~2 minutes that
# MAGIC would take 30+ minutes to write from scratch.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Notebook 2 — Build a Genie Space on this data
