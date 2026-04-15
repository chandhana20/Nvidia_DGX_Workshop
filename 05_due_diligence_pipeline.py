# Databricks notebook source

# MAGIC %md
# MAGIC # 05 — Financial Due Diligence Data Pipeline
# MAGIC
# MAGIC ## Use Case
# MAGIC Build an AI-powered pipeline that ingests public financial documents (10-K, 10-Q,
# MAGIC earnings releases, call transcripts, annual reports) and produces:
# MAGIC 1. **Parsed document text** stored in Delta tables
# MAGIC 2. **Chunked documents** ready for Vector Search (RAG)
# MAGIC 3. **Structured financial metrics** extracted by AI
# MAGIC 4. **Due diligence scoring** for invest/acquire decisions
# MAGIC
# MAGIC ## Companies Covered
# MAGIC NVIDIA, Apple, Amazon, Alphabet (Google), Meta, Microsoft, Tesla
# MAGIC
# MAGIC ## Document Types
# MAGIC | Type | Count | Contains |
# MAGIC |------|-------|----------|
# MAGIC | 10-K (Annual Filing) | 16 | Full-year financials, risks, MD&A |
# MAGIC | 10-Q (Quarterly Filing) | 40 | Quarterly updates |
# MAGIC | Earnings Releases | 76 | Revenue, EPS, guidance |
# MAGIC | Call Transcripts | 25 | Management commentary, analyst Q&A |
# MAGIC | Annual Reports | 7 | Strategy, shareholder letters |

# COMMAND ----------

# DBTITLE 1,0 — Configuration

CATALOG = "main"
SCHEMA = "fins_due_diligence"
VOLUME_RAW = "raw_filings"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_RAW}"

# Company ticker mapping for metadata tagging
COMPANY_TICKERS = {
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "apple": "AAPL",
    "aapl": "AAPL",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "google": "GOOGL",
    "goog": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "tesla": "TSLA",
    "tsla": "TSLA",
}

# Document type mapping based on folder names
DOC_TYPE_MAP = {
    "10K": "10-K",
    "10Q": "10-Q",
    "Annual Report": "Annual Report",
    "Earning Releases": "Earnings Release",
    "Call Transcripts": "Earnings Call Transcript",
}

print(f"Pipeline target: {CATALOG}.{SCHEMA}")
print(f"Volume path: {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,1 — Create catalog, schema, and volume
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS main;
# MAGIC CREATE SCHEMA IF NOT EXISTS main.fins_due_diligence;
# MAGIC CREATE VOLUME IF NOT EXISTS main.fins_due_diligence.raw_filings;
# MAGIC SELECT 'Schema and volume ready' AS status;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Upload PDFs to Volume
# MAGIC
# MAGIC Upload the 164 PDF documents from `ai-dev-day/data/` into the volume,
# MAGIC preserving the folder structure (10K/, 10Q/, etc.) as metadata.
# MAGIC
# MAGIC > **If files are already uploaded**, skip this cell and go to Step 2.

# COMMAND ----------

# DBTITLE 1,1a — Upload PDFs to Volume (run once)

import os

# Local path where the PDFs live (adjust if running outside of repo context)
LOCAL_DATA_DIR = "/Workspace/Repos/ai-dev-day/data"  # Adjust to your repo path

# Alternative: if files are on your local machine, upload them manually to the volume
# via Catalog UI: Catalog → main → fins_due_diligence → Volumes → raw_filings → Upload

uploaded = 0
skipped = 0

for folder in DOC_TYPE_MAP.keys():
    folder_path = os.path.join(LOCAL_DATA_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"  Folder not found: {folder_path} — skipping (upload manually via UI)")
        skipped += 1
        continue

    # Create subfolder in volume
    volume_subfolder = f"{VOLUME_PATH}/{folder}"
    dbutils.fs.mkdirs(volume_subfolder)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue
        src = os.path.join(folder_path, filename)
        dst = f"{volume_subfolder}/{filename}"
        # Only upload if not already present
        try:
            dbutils.fs.ls(dst)
        except Exception:
            dbutils.fs.cp(f"file:{src}", dst)
            uploaded += 1

print(f"Uploaded: {uploaded} files | Skipped folders: {skipped}")
print(f"If folders were skipped, upload PDFs manually to {VOLUME_PATH}/<folder>/")

# COMMAND ----------

# DBTITLE 1,1b — Verify files in volume

file_counts = {}
for folder in DOC_TYPE_MAP.keys():
    try:
        files = dbutils.fs.ls(f"{VOLUME_PATH}/{folder}")
        pdfs = [f for f in files if f.name.lower().endswith(".pdf")]
        file_counts[folder] = len(pdfs)
    except Exception:
        file_counts[folder] = 0

print("=" * 50)
print(" Volume Inventory: raw_filings")
print("=" * 50)
total = 0
for folder, count in file_counts.items():
    doc_type = DOC_TYPE_MAP[folder]
    print(f"  {doc_type:<30} {count:>4} PDFs")
    total += count
print(f"  {'TOTAL':<30} {total:>4} PDFs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Parse PDFs into Text
# MAGIC
# MAGIC Read all PDFs as binary files with Spark, then extract text using PyPDF2.
# MAGIC We tag each document with metadata: company ticker, document type, and filing period.

# COMMAND ----------

# DBTITLE 1,2a — Install PDF parsing library

# MAGIC %pip install pypdf
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,2b — Read all PDFs as binary and extract text

import re
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField

# Re-declare after Python restart
CATALOG = "main"
SCHEMA = "fins_due_diligence"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_filings"

COMPANY_TICKERS = {
    "nvidia": "NVDA", "nvda": "NVDA",
    "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN",
    "google": "GOOGL", "goog": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "microsoft": "MSFT", "msft": "MSFT",
    "tesla": "TSLA", "tsla": "TSLA",
}

DOC_TYPE_MAP = {
    "10K": "10-K",
    "10Q": "10-Q",
    "Annual Report": "Annual Report",
    "Earning Releases": "Earnings Release",
    "Call Transcripts": "Earnings Call Transcript",
}


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF binary content using pypdf."""
    from pypdf import PdfReader
    from io import BytesIO

    try:
        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        return f"[PARSE_ERROR] {str(e)}"


extract_text_udf = F.udf(extract_text_from_pdf, StringType())

# Read all PDFs from volume as binary
raw_pdfs = (
    spark.read
    .format("binaryFile")
    .option("recursiveFileLookup", "false")
    .option("pathGlobFilter", "*.pdf")
    .load(f"{VOLUME_PATH}/*/*.pdf")
)

# Extract text
parsed_docs = raw_pdfs.withColumn(
    "text_content", extract_text_udf(F.col("content"))
).select(
    F.col("path"),
    F.col("modificationTime").alias("file_modified"),
    F.length(F.col("content")).alias("file_size_bytes"),
    F.col("text_content"),
    F.length(F.col("text_content")).alias("text_length"),
)

print(f"Parsed {parsed_docs.count()} PDF documents")
display(parsed_docs.select("path", "file_size_bytes", "text_length").limit(10))

# COMMAND ----------

# DBTITLE 1,2c — Tag documents with company, doc_type, filing_period metadata

def infer_ticker(filename: str) -> str:
    """Infer company ticker from filename."""
    fn = filename.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key, ticker in sorted(COMPANY_TICKERS.items(), key=lambda x: -len(x[0])):
        if key.replace("-", "").replace("_", "") in fn:
            return ticker
    return "UNKNOWN"


def infer_doc_type(path: str) -> str:
    """Infer document type from parent folder name."""
    for folder, doc_type in DOC_TYPE_MAP.items():
        if f"/{folder}/" in path or f"/{folder.replace(' ', '%20')}/" in path:
            return doc_type
    return "Other"


def infer_filing_period(filename: str) -> str:
    """Extract filing period from filename patterns like Q1-2025, FY25, 2024-Q3, etc."""
    fn = filename.upper()

    # Pattern: FY25Q4, FY25Q1, etc.
    m = re.search(r'FY(\d{2})Q(\d)', fn)
    if m:
        year = 2000 + int(m.group(1))
        return f"{year}-Q{m.group(2)}"

    # Pattern: Q125, Q225, Q325, Q425 (Qn + 2-digit year)
    m = re.search(r'Q(\d)(\d{2})[^0-9]', fn)
    if m:
        year = 2000 + int(m.group(2))
        return f"{year}-Q{m.group(1)}"

    # Pattern: Q1-2025, Q2-2024, etc.
    m = re.search(r'Q(\d)[- ]?(\d{4})', fn)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"

    # Pattern: 2024-Q3, 2025-Q1
    m = re.search(r'(\d{4})[- ]?Q(\d)', fn)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"

    # Pattern: F1Q26, F2Q25 (NVIDIA fiscal quarter format)
    m = re.search(r'F(\d)Q(\d{2})', fn)
    if m:
        year = 2000 + int(m.group(2))
        return f"{year}-Q{m.group(1)}"

    # Pattern: 20231231, 20241231 (date-based, infer Q4)
    m = re.search(r'(\d{4})(1231|0930|0630|0331)', fn)
    if m:
        year = int(m.group(1))
        quarter_map = {"0331": "Q1", "0630": "Q2", "0930": "Q3", "1231": "Q4"}
        return f"{year}-{quarter_map[m.group(2)]}"

    # Pattern: just a year like 2024, 2025
    m = re.search(r'20(2[0-9])', fn)
    if m:
        return f"20{m.group(1)}-FY"

    return "UNKNOWN"


infer_ticker_udf = F.udf(infer_ticker, StringType())
infer_doc_type_udf = F.udf(infer_doc_type, StringType())
infer_filing_period_udf = F.udf(infer_filing_period, StringType())

# Extract just the filename from the full path
tagged_docs = parsed_docs.withColumn(
    "filename", F.element_at(F.split(F.col("path"), "/"), -1)
).withColumn(
    "ticker", infer_ticker_udf(F.col("filename"))
).withColumn(
    "doc_type", infer_doc_type_udf(F.col("path"))
).withColumn(
    "filing_period", infer_filing_period_udf(F.col("filename"))
).withColumn(
    "doc_id", F.sha2(F.col("path"), 256)
).withColumn(
    "parsed_at", F.current_timestamp()
)

print("Document metadata tagging complete:")
display(
    tagged_docs
    .select("filename", "ticker", "doc_type", "filing_period", "text_length")
    .orderBy("ticker", "doc_type", "filing_period")
    .limit(30)
)

# COMMAND ----------

# DBTITLE 1,2d — Write parsed documents to Delta table

tagged_docs.select(
    "doc_id", "filename", "path", "ticker", "doc_type", "filing_period",
    "text_content", "text_length", "file_size_bytes", "file_modified", "parsed_at"
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.parsed_documents"
)

row_count = spark.table(f"{CATALOG}.{SCHEMA}.parsed_documents").count()
print(f"Wrote {row_count} documents to {CATALOG}.{SCHEMA}.parsed_documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Chunk Documents for Vector Search
# MAGIC
# MAGIC Split each document into overlapping chunks of ~1000 characters with 200-char overlap.
# MAGIC Each chunk retains the parent document's metadata for filtered retrieval.

# COMMAND ----------

# DBTITLE 1,3a — Chunk documents with sliding window

from pyspark.sql.types import ArrayType

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text(text: str) -> list:
    """Split text into overlapping chunks."""
    if not text or text.startswith("[PARSE_ERROR]"):
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip tiny trailing fragments
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


chunk_udf = F.udf(chunk_text, ArrayType(StringType()))

docs_df = spark.table(f"{CATALOG}.{SCHEMA}.parsed_documents")

chunked = docs_df.withColumn(
    "chunks", chunk_udf(F.col("text_content"))
).select(
    "doc_id", "filename", "ticker", "doc_type", "filing_period",
    F.posexplode("chunks").alias("chunk_index", "chunk_text")
).withColumn(
    "chunk_id", F.sha2(F.concat(F.col("doc_id"), F.lit("_"), F.col("chunk_index").cast("string")), 256)
)

chunk_count = chunked.count()
print(f"Created {chunk_count} chunks from {docs_df.count()} documents")
print(f"Average chunks per document: {chunk_count / max(docs_df.count(), 1):.1f}")

display(
    chunked.groupBy("doc_type").agg(
        F.count("*").alias("total_chunks"),
        F.countDistinct("doc_id").alias("documents"),
    ).orderBy("doc_type")
)

# COMMAND ----------

# DBTITLE 1,3b — Write chunks to Delta table (Vector Search source)

chunked.select(
    "chunk_id", "doc_id", "chunk_index", "chunk_text",
    "ticker", "doc_type", "filing_period", "filename"
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.document_chunks"
)

print(f"Wrote {chunk_count} chunks to {CATALOG}.{SCHEMA}.document_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Create Vector Search Index
# MAGIC
# MAGIC Set up a Databricks Vector Search endpoint and a **Delta Sync Index**
# MAGIC that automatically keeps the index in sync as chunks are updated.

# COMMAND ----------

# DBTITLE 1,4a — Create Vector Search endpoint (run once)

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

VS_ENDPOINT = "fins_due_diligence_vs"
VS_INDEX = f"{CATALOG}.{SCHEMA}.document_chunks_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_chunks"

# Create endpoint if it doesn't exist
existing_endpoints = [ep.name for ep in w.vector_search_endpoints.list_endpoints()]
if VS_ENDPOINT not in existing_endpoints:
    print(f"Creating Vector Search endpoint: {VS_ENDPOINT}")
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print("Endpoint creation initiated — may take a few minutes.")
else:
    print(f"Endpoint already exists: {VS_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,4b — Enable Change Data Feed on source table

spark.sql(f"""
    ALTER TABLE {SOURCE_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"Change Data Feed enabled on {SOURCE_TABLE}")

# COMMAND ----------

# DBTITLE 1,4c — Create Delta Sync Vector Search Index

from databricks.sdk.service.catalog import VectorSearchIndex

# Check if index already exists
existing_indexes = []
try:
    existing_indexes = [
        idx.name for idx in w.vector_search_indexes.list_indexes(VS_ENDPOINT)
    ]
except Exception:
    pass

if VS_INDEX not in existing_indexes:
    print(f"Creating Vector Search index: {VS_INDEX}")
    w.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type="DELTA_SYNC",
        delta_sync_index_spec={
            "source_table": SOURCE_TABLE,
            "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [
                {
                    "name": "chunk_text",
                    "embedding_model_endpoint_name": "databricks-gte-large-en",
                }
            ],
        },
    )
    print("Index creation initiated — will sync automatically.")
else:
    print(f"Index already exists: {VS_INDEX}")

# COMMAND ----------

# DBTITLE 1,4d — Check index sync status

import time

for _ in range(3):
    try:
        idx_info = w.vector_search_indexes.get_index(VS_INDEX)
        print(f"Index: {VS_INDEX}")
        print(f"Status: {idx_info.status}")
        break
    except Exception as e:
        print(f"Waiting for index... ({e})")
        time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Extract Structured Financial Metrics with AI
# MAGIC
# MAGIC Use `ai_extract()` to pull key financial data points from each document.
# MAGIC We extract different fields depending on document type:
# MAGIC - **10-K / 10-Q**: Revenue, net income, EPS, margins, balance sheet items
# MAGIC - **Earnings Releases**: Revenue, EPS (actual vs estimate), guidance
# MAGIC - **Call Transcripts**: Sentiment, key themes, forward-looking statements
# MAGIC - **Annual Reports**: Strategic priorities, risk highlights

# COMMAND ----------

# DBTITLE 1,5a — Extract financial metrics from 10-K and 10-Q filings

# Work with the first 4000 chars of each document (ai_extract has token limits)
filings_df = spark.table(f"{CATALOG}.{SCHEMA}.parsed_documents").filter(
    F.col("doc_type").isin("10-K", "10-Q")
).withColumn(
    "text_excerpt", F.substring(F.col("text_content"), 1, 8000)
)

financial_metrics = filings_df.withColumn(
    "extracted",
    F.expr("""
        ai_extract(text_excerpt, array(
            'total_revenue_usd',
            'net_income_usd',
            'earnings_per_share_diluted',
            'gross_margin_percent',
            'operating_income_usd',
            'total_assets_usd',
            'total_liabilities_usd',
            'cash_and_equivalents_usd',
            'total_debt_usd',
            'operating_cash_flow_usd',
            'capital_expenditures_usd',
            'free_cash_flow_usd',
            'revenue_yoy_growth_percent',
            'fiscal_year',
            'fiscal_quarter'
        ))
    """)
).select(
    "doc_id", "ticker", "doc_type", "filing_period", "filename",
    F.get_json_object("extracted", "$.total_revenue_usd").alias("revenue"),
    F.get_json_object("extracted", "$.net_income_usd").alias("net_income"),
    F.get_json_object("extracted", "$.earnings_per_share_diluted").alias("eps_diluted"),
    F.get_json_object("extracted", "$.gross_margin_percent").alias("gross_margin_pct"),
    F.get_json_object("extracted", "$.operating_income_usd").alias("operating_income"),
    F.get_json_object("extracted", "$.total_assets_usd").alias("total_assets"),
    F.get_json_object("extracted", "$.total_liabilities_usd").alias("total_liabilities"),
    F.get_json_object("extracted", "$.cash_and_equivalents_usd").alias("cash_equivalents"),
    F.get_json_object("extracted", "$.total_debt_usd").alias("total_debt"),
    F.get_json_object("extracted", "$.operating_cash_flow_usd").alias("operating_cash_flow"),
    F.get_json_object("extracted", "$.capital_expenditures_usd").alias("capex"),
    F.get_json_object("extracted", "$.free_cash_flow_usd").alias("free_cash_flow"),
    F.get_json_object("extracted", "$.revenue_yoy_growth_percent").alias("revenue_yoy_growth"),
    F.get_json_object("extracted", "$.fiscal_year").alias("fiscal_year"),
    F.get_json_object("extracted", "$.fiscal_quarter").alias("fiscal_quarter"),
    F.current_timestamp().alias("extracted_at"),
)

financial_metrics.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.financial_metrics"
)

print(f"Extracted metrics from {financial_metrics.count()} filings")
display(financial_metrics.limit(10))

# COMMAND ----------

# DBTITLE 1,5b — Extract earnings guidance and estimates from earnings releases

earnings_df = spark.table(f"{CATALOG}.{SCHEMA}.parsed_documents").filter(
    F.col("doc_type") == "Earnings Release"
).withColumn(
    "text_excerpt", F.substring(F.col("text_content"), 1, 8000)
)

earnings_data = earnings_df.withColumn(
    "extracted",
    F.expr("""
        ai_extract(text_excerpt, array(
            'quarterly_revenue_usd',
            'quarterly_eps_diluted',
            'revenue_guidance_next_quarter_usd',
            'revenue_beat_or_miss',
            'eps_beat_or_miss',
            'segment_data_center_revenue_usd',
            'segment_gaming_revenue_usd',
            'segment_cloud_revenue_usd',
            'key_highlights',
            'fiscal_quarter_reported'
        ))
    """)
).select(
    "doc_id", "ticker", "doc_type", "filing_period", "filename",
    F.get_json_object("extracted", "$.quarterly_revenue_usd").alias("quarterly_revenue"),
    F.get_json_object("extracted", "$.quarterly_eps_diluted").alias("quarterly_eps"),
    F.get_json_object("extracted", "$.revenue_guidance_next_quarter_usd").alias("revenue_guidance"),
    F.get_json_object("extracted", "$.revenue_beat_or_miss").alias("revenue_beat_miss"),
    F.get_json_object("extracted", "$.eps_beat_or_miss").alias("eps_beat_miss"),
    F.get_json_object("extracted", "$.segment_data_center_revenue_usd").alias("seg_data_center"),
    F.get_json_object("extracted", "$.segment_gaming_revenue_usd").alias("seg_gaming"),
    F.get_json_object("extracted", "$.segment_cloud_revenue_usd").alias("seg_cloud"),
    F.get_json_object("extracted", "$.key_highlights").alias("key_highlights"),
    F.get_json_object("extracted", "$.fiscal_quarter_reported").alias("fiscal_quarter"),
    F.current_timestamp().alias("extracted_at"),
)

earnings_data.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.earnings_data"
)

print(f"Extracted earnings data from {earnings_data.count()} releases")
display(earnings_data.limit(10))

# COMMAND ----------

# DBTITLE 1,5c — Extract sentiment and themes from earnings call transcripts

transcripts_df = spark.table(f"{CATALOG}.{SCHEMA}.parsed_documents").filter(
    F.col("doc_type") == "Earnings Call Transcript"
).withColumn(
    "text_excerpt", F.substring(F.col("text_content"), 1, 10000)
)

transcript_analysis = transcripts_df.withColumn(
    "extracted",
    F.expr("""
        ai_extract(text_excerpt, array(
            'overall_management_sentiment',
            'key_growth_drivers_mentioned',
            'key_risks_mentioned',
            'capex_outlook',
            'ai_strategy_commentary',
            'competitive_positioning_commentary',
            'analyst_concerns_raised',
            'forward_looking_revenue_signals',
            'management_confidence_level'
        ))
    """)
).select(
    "doc_id", "ticker", "doc_type", "filing_period", "filename",
    F.get_json_object("extracted", "$.overall_management_sentiment").alias("mgmt_sentiment"),
    F.get_json_object("extracted", "$.key_growth_drivers_mentioned").alias("growth_drivers"),
    F.get_json_object("extracted", "$.key_risks_mentioned").alias("risks_mentioned"),
    F.get_json_object("extracted", "$.capex_outlook").alias("capex_outlook"),
    F.get_json_object("extracted", "$.ai_strategy_commentary").alias("ai_strategy"),
    F.get_json_object("extracted", "$.competitive_positioning_commentary").alias("competitive_position"),
    F.get_json_object("extracted", "$.analyst_concerns_raised").alias("analyst_concerns"),
    F.get_json_object("extracted", "$.forward_looking_revenue_signals").alias("revenue_signals"),
    F.get_json_object("extracted", "$.management_confidence_level").alias("mgmt_confidence"),
    F.current_timestamp().alias("extracted_at"),
)

transcript_analysis.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.transcript_analysis"
)

print(f"Analyzed {transcript_analysis.count()} earnings call transcripts")
display(transcript_analysis.limit(10))

# COMMAND ----------

# DBTITLE 1,5d — Extract risk factors and strategic priorities from annual reports

annual_reports_df = spark.table(f"{CATALOG}.{SCHEMA}.parsed_documents").filter(
    F.col("doc_type") == "Annual Report"
).withColumn(
    "text_excerpt", F.substring(F.col("text_content"), 1, 10000)
)

risk_analysis = annual_reports_df.withColumn(
    "extracted",
    F.expr("""
        ai_extract(text_excerpt, array(
            'top_3_risk_factors',
            'strategic_priorities',
            'r_and_d_spending_usd',
            'employee_count',
            'geographic_revenue_breakdown',
            'major_acquisitions_mentioned',
            'regulatory_risks',
            'supply_chain_risks',
            'customer_concentration_risk'
        ))
    """)
).select(
    "doc_id", "ticker", "doc_type", "filing_period", "filename",
    F.get_json_object("extracted", "$.top_3_risk_factors").alias("top_risks"),
    F.get_json_object("extracted", "$.strategic_priorities").alias("strategic_priorities"),
    F.get_json_object("extracted", "$.r_and_d_spending_usd").alias("r_and_d_spend"),
    F.get_json_object("extracted", "$.employee_count").alias("employee_count"),
    F.get_json_object("extracted", "$.geographic_revenue_breakdown").alias("geo_revenue"),
    F.get_json_object("extracted", "$.major_acquisitions_mentioned").alias("acquisitions"),
    F.get_json_object("extracted", "$.regulatory_risks").alias("regulatory_risks"),
    F.get_json_object("extracted", "$.supply_chain_risks").alias("supply_chain_risks"),
    F.get_json_object("extracted", "$.customer_concentration_risk").alias("customer_concentration"),
    F.current_timestamp().alias("extracted_at"),
)

risk_analysis.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.risk_analysis"
)

print(f"Analyzed {risk_analysis.count()} annual reports")
display(risk_analysis.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Due Diligence Scoring View
# MAGIC
# MAGIC Create a unified view that aggregates all extracted data into a
# MAGIC per-company due diligence scorecard.

# COMMAND ----------

# DBTITLE 1,6a — Create due diligence summary view

spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.due_diligence_scorecard AS

    WITH latest_metrics AS (
        SELECT
            ticker,
            FIRST_VALUE(revenue) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_revenue,
            FIRST_VALUE(net_income) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_net_income,
            FIRST_VALUE(eps_diluted) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_eps,
            FIRST_VALUE(gross_margin_pct) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_gross_margin,
            FIRST_VALUE(cash_equivalents) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_cash,
            FIRST_VALUE(total_debt) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_debt,
            FIRST_VALUE(free_cash_flow) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_fcf,
            FIRST_VALUE(revenue_yoy_growth) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_revenue_growth,
            FIRST_VALUE(filing_period) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_period,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS rn
        FROM {CATALOG}.{SCHEMA}.financial_metrics
        WHERE revenue IS NOT NULL
    ),

    doc_coverage AS (
        SELECT
            ticker,
            COUNT(*) AS total_documents,
            COUNT(DISTINCT doc_type) AS doc_types_covered,
            MIN(filing_period) AS earliest_period,
            MAX(filing_period) AS latest_period
        FROM {CATALOG}.{SCHEMA}.parsed_documents
        WHERE ticker != 'UNKNOWN'
        GROUP BY ticker
    ),

    latest_sentiment AS (
        SELECT
            ticker,
            FIRST_VALUE(mgmt_sentiment) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS latest_sentiment,
            FIRST_VALUE(growth_drivers) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS growth_drivers,
            FIRST_VALUE(risks_mentioned) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS key_risks,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS rn
        FROM {CATALOG}.{SCHEMA}.transcript_analysis
    )

    SELECT
        m.ticker,
        m.latest_period,
        m.latest_revenue,
        m.latest_net_income,
        m.latest_eps,
        m.latest_gross_margin,
        m.latest_cash,
        m.latest_debt,
        m.latest_fcf,
        m.latest_revenue_growth,
        s.latest_sentiment AS mgmt_sentiment,
        s.growth_drivers,
        s.key_risks,
        d.total_documents,
        d.doc_types_covered,
        d.earliest_period AS data_coverage_from,
        d.latest_period AS data_coverage_to
    FROM latest_metrics m
    LEFT JOIN latest_sentiment s ON m.ticker = s.ticker AND s.rn = 1
    LEFT JOIN doc_coverage d ON m.ticker = d.ticker
    WHERE m.rn = 1
    ORDER BY m.ticker
""")

print("Due diligence scorecard view created.")
display(spark.table(f"{CATALOG}.{SCHEMA}.due_diligence_scorecard"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Test Vector Search Retrieval
# MAGIC
# MAGIC Verify the RAG pipeline works by querying the Vector Search index.

# COMMAND ----------

# DBTITLE 1,7a — Query Vector Search index

VS_INDEX = f"{CATALOG}.{SCHEMA}.document_chunks_index"
VS_ENDPOINT = "fins_due_diligence_vs"

results = w.vector_search_indexes.query_index(
    index_name=VS_INDEX,
    columns=["chunk_text", "ticker", "doc_type", "filing_period", "filename"],
    query_text="What are NVIDIA's main risk factors and competitive threats?",
    num_results=5,
    filters_json='{"ticker": "NVDA"}',
)

print("Top 5 results for: 'NVIDIA risk factors and competitive threats'")
print("=" * 70)
for i, row in enumerate(results.result.data_array):
    print(f"\n--- Result {i+1} ({row[1]} | {row[2]} | {row[3]}) ---")
    print(row[0][:300] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Pipeline Output Tables
# MAGIC
# MAGIC | Table | Description | Rows |
# MAGIC |-------|-------------|------|
# MAGIC | `parsed_documents` | Full text extracted from each PDF | 164 |
# MAGIC | `document_chunks` | Overlapping text chunks for Vector Search | ~10K+ |
# MAGIC | `document_chunks_index` | Vector Search index (auto-synced) | ~10K+ |
# MAGIC | `financial_metrics` | Structured financials from 10-K/10-Q | ~56 |
# MAGIC | `earnings_data` | Revenue, EPS, guidance from earnings releases | ~76 |
# MAGIC | `transcript_analysis` | Sentiment and themes from call transcripts | ~25 |
# MAGIC | `risk_analysis` | Risk factors from annual reports | ~7 |
# MAGIC | `due_diligence_scorecard` | Unified per-company scorecard (VIEW) | 7 |

# COMMAND ----------

# DBTITLE 1,Final — Inventory all tables

tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
print("=" * 60)
print(f" Unity Catalog: {CATALOG}.{SCHEMA}")
print("=" * 60)
for t in tables:
    tname = t["tableName"]
    try:
        count = spark.table(f"{CATALOG}.{SCHEMA}.{tname}").count()
        print(f"  {tname:<40} {count:>8} rows")
    except Exception as e:
        print(f"  {tname:<40}  (view or error)")

print(f"\nPipeline complete. Ready for due diligence analysis.")
print(f"Next: Use the API routes or a Genie Space to query this data.")
