# Databricks notebook source

# MAGIC %md
# MAGIC # Block 4: Vector Search, RAG & Knowledge Assistants
# MAGIC **NVIDIA DGX Cloud MLOps & GenAI Workshop**
# MAGIC
# MAGIC | Duration | Focus Area |
# MAGIC |----------|-----------|
# MAGIC | 50 min | Hands-on lab |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this notebook you will be able to:
# MAGIC
# MAGIC 1. **Create a Vector Search endpoint** and Delta Sync index on GPU operations documentation
# MAGIC 2. **Perform similarity search** against your indexed documents
# MAGIC 3. **Build a Knowledge Assistant** ("GPU Operations Advisor") backed by your document volume
# MAGIC 4. **Create a Multi-Agent Supervisor** ("DGX Operations Center") that routes queries to the right tool
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - A Databricks workspace with Unity Catalog enabled
# MAGIC - Access to Foundation Model APIs (for embeddings)
# MAGIC - Catalog `main` and schema `mlops_genai_workshop` created (from Notebook 01)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part A -- Vector Search Setup (15 min)
# MAGIC
# MAGIC We will:
# MAGIC 1. Generate synthetic GPU runbook PDFs and upload them to a UC Volume
# MAGIC 2. Parse the PDFs into a Delta table with chunked text
# MAGIC 3. Create a Vector Search endpoint
# MAGIC 4. Create a Delta Sync index with `databricks-gte-large-en` embeddings
# MAGIC 5. Query the index with similarity search

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Configuration

# COMMAND ----------

# Workshop configuration
CATALOG = "main"
SCHEMA = "mlops_genai_workshop"
VOLUME = "docs"

# Vector Search settings
VS_ENDPOINT_NAME = "dgx_workshop_vs_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.gpu_runbook_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.gpu_runbook_chunks"
EMBEDDING_MODEL = "databricks-gte-large-en"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create the Volume and Generate Synthetic GPU Runbook PDFs
# MAGIC
# MAGIC In a real engagement you would upload actual GPU runbook PDFs via the UI or CLI.
# MAGIC Here we generate synthetic documents so the workshop is self-contained.

# COMMAND ----------

# Ensure schema and volume exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

# Generate synthetic GPU runbook PDFs
# Install fpdf2 for PDF generation (lightweight, pure-Python)
%pip install fpdf2 --quiet
dbutils.library.restartPython()

# COMMAND ----------

from fpdf import FPDF
import os, textwrap

CATALOG = "main"
SCHEMA = "mlops_genai_workshop"
VOLUME = "docs"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Define synthetic runbook content
runbooks = {
    "GPU_Thermal_Management_Runbook.pdf": [
        ("GPU Thermal Management Runbook", ""),
        ("1. Overview", (
            "This runbook covers thermal management procedures for NVIDIA A100 and H100 GPUs "
            "deployed in DGX Cloud environments. Proper thermal management is critical to "
            "maintaining GPU reliability and performance."
        )),
        ("2. Temperature Thresholds", (
            "Normal operating range: 30C - 75C\n"
            "Warning threshold: 75C - 85C\n"
            "Critical threshold: Above 85C\n"
            "Emergency shutdown: Above 95C\n\n"
            "When GPU temperature exceeds 85C the recommended procedure is:\n"
            "1. Immediately reduce workload by 50% using nvidia-smi power limit commands\n"
            "2. Check ambient data center temperature and CRAC unit status\n"
            "3. Verify fan speeds are at maximum RPM using ipmi tools\n"
            "4. Inspect for blocked airflow or dust accumulation in the chassis\n"
            "5. If temperature does not drop below 80C within 10 minutes, initiate graceful "
            "   workload migration to standby nodes\n"
            "6. File a thermal incident ticket with severity P2 or higher"
        )),
        ("3. Thermal Throttling Procedure", (
            "When thermal throttling is detected (clock frequency reduced by more than 15%):\n"
            "1. Log the throttling event with timestamp, GPU ID, and clock frequencies\n"
            "2. Check nvidia-smi for current thermal readings across all GPUs in the node\n"
            "3. Verify the cooling system is operational (fan tach sensors, coolant flow)\n"
            "4. Reduce the power limit to 300W for A100 or 500W for H100 temporarily\n"
            "5. Schedule a maintenance window within 48 hours for physical inspection\n"
            "6. Review thermal paste application if throttling persists after cooling fix"
        )),
        ("4. Preventive Maintenance", (
            "Monthly: Check thermal paste condition on GPU heat sinks\n"
            "Quarterly: Clean dust filters and inspect fan assemblies\n"
            "Bi-annually: Full thermal audit including infrared imaging\n"
            "Annually: Replace thermal interface material on all GPUs"
        )),
    ],
    "GPU_Memory_Diagnostics_Runbook.pdf": [
        ("GPU Memory Diagnostics Runbook", ""),
        ("1. Overview", (
            "This runbook provides procedures for diagnosing and resolving GPU memory errors "
            "on NVIDIA A100 and H100 GPUs. ECC (Error Correcting Code) memory is used in all "
            "DGX Cloud deployments to detect and correct memory errors."
        )),
        ("2. ECC Memory Error Types", (
            "Single-Bit Errors (SBE): Correctable by ECC. Logged but do not cause failures.\n"
            "Double-Bit Errors (DBE): Uncorrectable. Will cause GPU reset or application crash.\n"
            "Row Remapping Events: Hardware remaps a faulty memory row to a spare row.\n\n"
            "Monitoring: Use nvidia-smi --query-gpu=ecc.errors.corrected.aggregate.total,"
            "ecc.errors.uncorrected.aggregate.total --format=csv"
        )),
        ("3. Diagnosing ECC Memory Errors on A100 GPUs", (
            "Step 1: Run nvidia-smi -q -d ECC to get detailed ECC error counts\n"
            "Step 2: Check if errors are volatile (since last reset) or aggregate (lifetime)\n"
            "Step 3: If SBE count exceeds 100 in 24 hours, schedule proactive replacement\n"
            "Step 4: If any DBE is detected, immediately:\n"
            "  a. Drain workloads from the affected GPU\n"
            "  b. Run dcgmi diag -r 3 for a full diagnostic\n"
            "  c. Check row remapping status with nvidia-smi -q -d ROW_REMAPPER\n"
            "  d. If row remapping is exhausted, the GPU must be RMA'd\n"
            "Step 5: For intermittent errors, reset ECC counters and monitor for 48 hours\n"
            "Step 6: Document all findings in the GPU health tracking system"
        )),
        ("4. Memory Stress Testing", (
            "Use DCGM diagnostics for comprehensive memory testing:\n"
            "  dcgmi diag -r 3 (Level 3 = full memory test, takes ~15 minutes per GPU)\n"
            "  dcgmi diag -r 2 (Level 2 = quick memory test, takes ~2 minutes per GPU)\n\n"
            "For targeted testing use cuda_memtest or gpu-burn utilities.\n"
            "Always run memory tests after any GPU replacement or firmware update."
        )),
    ],
    "DGX_Cloud_SLA_Runbook.pdf": [
        ("DGX Cloud SLA and Availability Runbook", ""),
        ("1. Overview", (
            "This document defines the SLA requirements and availability targets for "
            "NVIDIA DGX Cloud GPU infrastructure."
        )),
        ("2. SLA Requirements for DGX Cloud GPU Availability", (
            "Tier 1 (Production ML Training): 99.9% monthly uptime\n"
            "Tier 2 (Development/Testing): 99.5% monthly uptime\n"
            "Tier 3 (Batch Processing): 99.0% monthly uptime\n\n"
            "Uptime is measured per-node, calculated as:\n"
            "  (Total minutes - Unplanned downtime minutes) / Total minutes * 100\n\n"
            "Planned maintenance windows are excluded from SLA calculations.\n"
            "Standard maintenance windows: Sunday 02:00-06:00 UTC\n\n"
            "SLA credits:\n"
            "  99.0% - 99.9%: 10% credit\n"
            "  95.0% - 99.0%: 25% credit\n"
            "  Below 95.0%: 50% credit"
        )),
        ("3. Incident Response Times", (
            "P1 (Total outage): Response within 15 minutes, resolution target 4 hours\n"
            "P2 (Degraded performance): Response within 30 minutes, resolution target 8 hours\n"
            "P3 (Single GPU failure): Response within 2 hours, resolution target 24 hours\n"
            "P4 (Informational): Response within 8 hours, resolution target 5 business days"
        )),
        ("4. Availability Monitoring", (
            "GPU health checks run every 60 seconds via DCGM\n"
            "NVLink and NVSwitch health verified every 5 minutes\n"
            "Network fabric (InfiniBand) checked every 30 seconds\n"
            "Storage (Lustre/GPFS) availability checked every 60 seconds\n\n"
            "Automated failover triggers when a node is unreachable for 3 consecutive health checks."
        )),
    ],
}


def create_pdf(title_and_sections, filepath):
    """Generate a simple PDF from a list of (heading, body) tuples."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for heading, body in title_and_sections:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        if body:
            pdf.set_font("Helvetica", "", 11)
            for line in body.split("\n"):
                pdf.multi_cell(0, 6, line)
                pdf.ln(1)

    pdf.output(filepath)


os.makedirs(VOLUME_PATH, exist_ok=True)
for filename, sections in runbooks.items():
    path = os.path.join(VOLUME_PATH, filename)
    create_pdf(sections, path)
    print(f"Created: {path}")

print(f"\nAll runbook PDFs written to {VOLUME_PATH}")

# COMMAND ----------

# Verify the uploaded files
display(dbutils.fs.ls(f"dbfs:{VOLUME_PATH}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Parse PDFs into a Delta Table with Chunked Text
# MAGIC
# MAGIC Vector Search requires a Delta table as source. We parse each PDF, split the text
# MAGIC into manageable chunks, and write to a Delta table.

# COMMAND ----------

%pip install pypdf --quiet
dbutils.library.restartPython()

# COMMAND ----------

from pypdf import PdfReader
import os

CATALOG = "main"
SCHEMA = "mlops_genai_workshop"
VOLUME = "docs"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.gpu_runbook_chunks"

CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# Process all PDFs in the volume
rows = []
chunk_id = 0
for filename in os.listdir(VOLUME_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(VOLUME_PATH, filename)
        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": chunk_id,
                "doc_name": filename,
                "chunk_index": i,
                "content": chunk.strip(),
            })
            chunk_id += 1

print(f"Parsed {len(rows)} chunks from {len(os.listdir(VOLUME_PATH))} PDFs")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

schema = StructType([
    StructField("chunk_id", IntegerType(), False),
    StructField("doc_name", StringType(), False),
    StructField("chunk_index", IntegerType(), False),
    StructField("content", StringType(), False),
])

df = spark.createDataFrame(rows, schema=schema)

# Enable Change Data Feed (required for Delta Sync indexes)
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(SOURCE_TABLE)

display(spark.table(SOURCE_TABLE))
print(f"\nSource table '{SOURCE_TABLE}' created with {df.count()} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Create a Vector Search Endpoint
# MAGIC
# MAGIC A Vector Search **endpoint** is the compute resource that hosts your indexes.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
import time

w = WorkspaceClient()

VS_ENDPOINT_NAME = "dgx_workshop_vs_endpoint"

# Check if endpoint already exists
existing_endpoints = [ep.name for ep in w.vector_search_endpoints.list_endpoints()]

if VS_ENDPOINT_NAME not in existing_endpoints:
    print(f"Creating Vector Search endpoint: {VS_ENDPOINT_NAME} ...")
    w.vector_search_endpoints.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD,
    )
else:
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists.")

# COMMAND ----------

# Poll until the endpoint is ONLINE
print(f"Waiting for endpoint '{VS_ENDPOINT_NAME}' to become ONLINE ...")

while True:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT_NAME)
    status = ep.endpoint_status
    state = status.state.value if status and status.state else "UNKNOWN"
    print(f"  Status: {state}")

    if state == "ONLINE":
        print("Endpoint is ONLINE!")
        break
    elif state in ("FAILED",):
        raise RuntimeError(f"Endpoint provisioning failed: {status.message}")

    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Create a Delta Sync Index
# MAGIC
# MAGIC A **Delta Sync** index automatically keeps the vector index in sync with
# MAGIC the source Delta table. We use the `databricks-gte-large-en` embedding model
# MAGIC hosted on Foundation Model APIs.

# COMMAND ----------

from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)

VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.gpu_runbook_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.gpu_runbook_chunks"
EMBEDDING_MODEL = "databricks-gte-large-en"

# Check if index already exists
existing_indexes = [
    idx.name for idx in w.vector_search_indexes.list_indexes(name=VS_ENDPOINT_NAME)
]

if VS_INDEX_NAME not in existing_indexes:
    print(f"Creating Delta Sync index: {VS_INDEX_NAME} ...")
    w.vector_search_indexes.create_index(
        name=VS_INDEX_NAME,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="chunk_id",
        index_type="DELTA_SYNC",
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_TABLE,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name=EMBEDDING_MODEL,
                )
            ],
        ),
    )
    print("Index creation initiated.")
else:
    print(f"Index '{VS_INDEX_NAME}' already exists.")

# COMMAND ----------

# Poll until the index is ONLINE
print(f"Waiting for index '{VS_INDEX_NAME}' to become ONLINE ...")
print("(This may take 5-10 minutes while embeddings are computed.)\n")

while True:
    idx = w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
    state = idx.status.ready if idx.status else False
    message = idx.status.message if idx.status else ""
    print(f"  Ready: {state} | Message: {message}")

    if state:
        print("\nIndex is ONLINE and ready for queries!")
        break

    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Query the Vector Search Index
# MAGIC
# MAGIC Now let's test similarity search against our GPU runbook documents.

# COMMAND ----------

# Similarity search
results = w.vector_search_indexes.query_index(
    index_name=VS_INDEX_NAME,
    columns=["chunk_id", "doc_name", "content"],
    query_text="What should I do when GPU temperature exceeds 85 degrees?",
    num_results=3,
)

print("Query: 'What should I do when GPU temperature exceeds 85 degrees?'\n")
print("=" * 80)
for row in results.result.data_array:
    chunk_id, doc_name, content = row[0], row[1], row[2]
    print(f"\n[{doc_name}] (chunk {chunk_id})")
    print("-" * 40)
    print(content[:500])
    print()

# COMMAND ----------

# Try another query
results = w.vector_search_indexes.query_index(
    index_name=VS_INDEX_NAME,
    columns=["chunk_id", "doc_name", "content"],
    query_text="ECC memory error diagnosis A100",
    num_results=3,
)

print("Query: 'ECC memory error diagnosis A100'\n")
print("=" * 80)
for row in results.result.data_array:
    chunk_id, doc_name, content = row[0], row[1], row[2]
    print(f"\n[{doc_name}] (chunk {chunk_id})")
    print("-" * 40)
    print(content[:500])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part B -- Knowledge Assistant (20 min)
# MAGIC
# MAGIC We will create a **Knowledge Assistant** ("GPU Operations Advisor") that uses
# MAGIC our GPU runbook volume as its knowledge source. The KA handles document Q&A
# MAGIC natively -- no manual chunking or embedding setup required (it uses managed RAG).
# MAGIC
# MAGIC > **Note:** Knowledge Assistants use UC Volumes directly as a knowledge source.
# MAGIC > The Vector Search index we built in Part A is a great learning exercise;
# MAGIC > the KA in Part B shows how Databricks simplifies this into a managed experience.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create the Knowledge Assistant

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    App,
)
import time

w = WorkspaceClient()

CATALOG = "main"
SCHEMA = "mlops_genai_workshop"
VOLUME = "docs"

# -------------------------------------------------------------------
# PARTICIPANT TODO: After creation, fill in the KA name if different
# -------------------------------------------------------------------
KA_NAME = "gpu-operations-advisor"
KA_DISPLAY_NAME = "GPU Operations Advisor"

KA_INSTRUCTIONS = (
    "You are a GPU operations expert. Answer questions about GPU troubleshooting, "
    "maintenance procedures, and performance optimization using only the provided "
    "documentation. If the answer is not in the documentation, say so clearly. "
    "Always cite the source document name when possible."
)

# COMMAND ----------

# Create the Knowledge Assistant using the Databricks SDK
# Knowledge Assistants are managed via the serving endpoints / apps API

from databricks.sdk.service.apps import (
    App,
    AppResource,
    AppResourceServingEndpoint,
)

# --- Create Knowledge Assistant via the AI/BI or Agent API ---
# Note: The exact SDK method depends on your workspace version.
# Below is the pattern using the Databricks REST API for KA creation.

import requests
import json

# Get the workspace URL and token from the Workspace Client
host = w.config.host
headers = {
    "Authorization": f"Bearer {w.config.token}",
    "Content-Type": "application/json",
}

# Knowledge Assistant creation payload
ka_payload = {
    "display_name": KA_DISPLAY_NAME,
    "name": KA_NAME,
    "description": "Knowledge Assistant for GPU operations, troubleshooting, and maintenance runbooks.",
    "instructions": KA_INSTRUCTIONS,
    "knowledge_sources": [
        {
            "type": "VOLUME",
            "volume": {
                "catalog_name": CATALOG,
                "schema_name": SCHEMA,
                "volume_name": VOLUME,
            },
        }
    ],
}

# Create the Knowledge Assistant
response = requests.post(
    f"{host}/api/2.0/knowledge-assistants",
    headers=headers,
    json=ka_payload,
)

if response.status_code == 200:
    ka_result = response.json()
    print(f"Knowledge Assistant created successfully!")
    print(json.dumps(ka_result, indent=2))
    KA_ID = ka_result.get("id", "")
elif response.status_code == 409:
    print(f"Knowledge Assistant '{KA_NAME}' already exists. Fetching details...")
    # List existing KAs to get the ID
    list_resp = requests.get(
        f"{host}/api/2.0/knowledge-assistants",
        headers=headers,
    )
    if list_resp.status_code == 200:
        for ka in list_resp.json().get("knowledge_assistants", []):
            if ka.get("name") == KA_NAME:
                KA_ID = ka["id"]
                print(f"Found existing KA with ID: {KA_ID}")
                break
else:
    print(f"Error creating KA: {response.status_code}")
    print(response.text)
    print("\n--- Alternative: Create via UI ---")
    print(f"1. Navigate to: {host}/ml/knowledge-assistants")
    print(f"2. Click 'Create Knowledge Assistant'")
    print(f"3. Name: '{KA_DISPLAY_NAME}'")
    print(f"4. Add Knowledge Source: Volume -> {CATALOG}.{SCHEMA}.{VOLUME}")
    print(f"5. Set instructions (see KA_INSTRUCTIONS variable above)")
    KA_ID = "<FILL_IN_AFTER_MANUAL_CREATION>"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Wait for the Knowledge Assistant to be Ready
# MAGIC
# MAGIC The KA needs to ingest and index the documents from the volume. This typically
# MAGIC takes a few minutes.

# COMMAND ----------

# Poll until KA is ready
if KA_ID and KA_ID != "<FILL_IN_AFTER_MANUAL_CREATION>":
    print(f"Waiting for Knowledge Assistant '{KA_NAME}' to be ready ...")

    while True:
        resp = requests.get(
            f"{host}/api/2.0/knowledge-assistants/{KA_ID}",
            headers=headers,
        )
        if resp.status_code == 200:
            ka_status = resp.json()
            state = ka_status.get("state", "UNKNOWN")
            print(f"  State: {state}")
            if state in ("ACTIVE", "ONLINE", "READY"):
                print("Knowledge Assistant is ready!")
                break
            elif state in ("FAILED", "ERROR"):
                print(f"KA provisioning failed: {ka_status}")
                break
        else:
            print(f"  Polling error: {resp.status_code}")

        time.sleep(30)
else:
    print("Skipping poll -- fill in KA_ID after manual creation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Test the Knowledge Assistant
# MAGIC
# MAGIC Let's ask our GPU Operations Advisor the three test questions.

# COMMAND ----------

def ask_knowledge_assistant(question, ka_id=KA_ID):
    """Send a question to the Knowledge Assistant and print the response."""
    print(f"Q: {question}")
    print("-" * 60)

    response = requests.post(
        f"{host}/api/2.0/knowledge-assistants/{ka_id}/chat",
        headers=headers,
        json={"messages": [{"role": "user", "content": question}]},
    )

    if response.status_code == 200:
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
        print(f"A: {answer}")

        # Show sources if available
        sources = result.get("choices", [{}])[0].get("message", {}).get("context", {}).get("documents", [])
        if sources:
            print(f"\nSources:")
            for src in sources[:3]:
                print(f"  - {src.get('doc_name', 'Unknown')} (score: {src.get('score', 'N/A')})")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    print("=" * 60)
    print()

# COMMAND ----------

# Test Question 1
ask_knowledge_assistant(
    "What is the recommended procedure when GPU temperature exceeds 85C?"
)

# COMMAND ----------

# Test Question 2
ask_knowledge_assistant(
    "How do I diagnose ECC memory errors on A100 GPUs?"
)

# COMMAND ----------

# Test Question 3
ask_knowledge_assistant(
    "What are the SLA requirements for DGX Cloud GPU availability?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part C -- Multi-Agent Supervisor (15 min)
# MAGIC
# MAGIC Now we bring it all together with a **Multi-Agent Supervisor (MAS)** called
# MAGIC "DGX Operations Center". This supervisor routes incoming queries to the
# MAGIC appropriate tool:
# MAGIC
# MAGIC | Tool | Type | Purpose |
# MAGIC |------|------|---------|
# MAGIC | DGX Fleet Analytics | Genie Space | Query structured fleet telemetry data via SQL |
# MAGIC | GPU Operations Advisor | Knowledge Assistant | Answer questions from GPU runbook documents |
# MAGIC
# MAGIC The supervisor decides which tool to invoke based on the nature of the question.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Configuration
# MAGIC
# MAGIC Fill in the IDs from your previous work. The Genie Space should have been
# MAGIC created in Notebook 02 (DGX Fleet Analytics).

# COMMAND ----------

# -------------------------------------------------------------------
# PARTICIPANT TODO: Fill in these values from your workspace
# -------------------------------------------------------------------

# From Notebook 02 -- Genie Space for structured fleet data
GENIE_SPACE_ID = "<FILL_IN_YOUR_GENIE_SPACE_ID>"  # e.g., "01f23abc..."

# From Part B above -- Knowledge Assistant for runbook Q&A
KA_NAME_FOR_MAS = KA_NAME  # "gpu-operations-advisor"

# Multi-Agent Supervisor settings
MAS_NAME = "dgx-operations-center"
MAS_DISPLAY_NAME = "DGX Operations Center"

print(f"Genie Space ID: {GENIE_SPACE_ID}")
print(f"Knowledge Assistant: {KA_NAME_FOR_MAS}")
print(f"MAS Name: {MAS_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create the Multi-Agent Supervisor

# COMMAND ----------

# Multi-Agent Supervisor creation payload
mas_payload = {
    "display_name": MAS_DISPLAY_NAME,
    "name": MAS_NAME,
    "description": (
        "Central operations hub for DGX Cloud fleet management. "
        "Routes queries to the appropriate tool: Genie Space for fleet "
        "analytics data, or Knowledge Assistant for runbook procedures."
    ),
    "instructions": (
        "You are the DGX Operations Center supervisor. Your job is to route "
        "questions to the right tool:\n\n"
        "1. For questions about CURRENT fleet status, GPU counts, metrics, "
        "   telemetry data, or anything that requires querying a database, "
        "   use the 'DGX Fleet Analytics' Genie tool.\n\n"
        "2. For questions about PROCEDURES, runbooks, troubleshooting steps, "
        "   maintenance guides, or SLA documentation, use the "
        "   'GPU Operations Advisor' Knowledge Assistant.\n\n"
        "3. For questions that need BOTH data AND procedures (e.g., 'we have "
        "   X GPUs with issue Y, what should we do?'), query the Genie tool "
        "   first to get the data, then consult the Knowledge Assistant for "
        "   the recommended procedure. Combine both answers.\n\n"
        "Always explain which tool(s) you are consulting and why."
    ),
    "tools": [
        {
            "type": "GENIE_SPACE",
            "genie_space": {
                "space_id": GENIE_SPACE_ID,
                "display_name": "DGX Fleet Analytics",
                "description": (
                    "Query structured DGX Cloud fleet data including GPU utilization, "
                    "temperatures, memory errors, node status, and historical metrics. "
                    "Use this for any question about current or historical fleet STATUS."
                ),
            },
        },
        {
            "type": "KNOWLEDGE_ASSISTANT",
            "knowledge_assistant": {
                "name": KA_NAME_FOR_MAS,
                "display_name": "GPU Operations Advisor",
                "description": (
                    "GPU operations runbook knowledge base. Contains procedures for "
                    "thermal management, ECC memory diagnostics, SLA requirements, "
                    "and maintenance schedules. Use this for HOW-TO and PROCEDURE questions."
                ),
            },
        },
    ],
}

# Create the Multi-Agent Supervisor
response = requests.post(
    f"{host}/api/2.0/multi-agent-supervisors",
    headers=headers,
    json=mas_payload,
)

if response.status_code == 200:
    mas_result = response.json()
    print("Multi-Agent Supervisor created successfully!")
    print(json.dumps(mas_result, indent=2))
    MAS_ID = mas_result.get("id", "")
elif response.status_code == 409:
    print(f"MAS '{MAS_NAME}' already exists.")
    list_resp = requests.get(
        f"{host}/api/2.0/multi-agent-supervisors",
        headers=headers,
    )
    if list_resp.status_code == 200:
        for mas in list_resp.json().get("multi_agent_supervisors", []):
            if mas.get("name") == MAS_NAME:
                MAS_ID = mas["id"]
                print(f"Found existing MAS with ID: {MAS_ID}")
                break
else:
    print(f"Error creating MAS: {response.status_code}")
    print(response.text)
    print(f"\n--- Alternative: Create via UI ---")
    print(f"1. Navigate to: {host}/ml/multi-agent-supervisors")
    print(f"2. Click 'Create Multi-Agent Supervisor'")
    print(f"3. Name: '{MAS_DISPLAY_NAME}'")
    print(f"4. Add tools: Genie Space ('{GENIE_SPACE_ID}') + KA ('{KA_NAME_FOR_MAS}')")
    MAS_ID = "<FILL_IN_AFTER_MANUAL_CREATION>"

# COMMAND ----------

# Wait for MAS to be ready
if MAS_ID and MAS_ID != "<FILL_IN_AFTER_MANUAL_CREATION>":
    print(f"Waiting for MAS '{MAS_NAME}' to be ready ...")

    while True:
        resp = requests.get(
            f"{host}/api/2.0/multi-agent-supervisors/{MAS_ID}",
            headers=headers,
        )
        if resp.status_code == 200:
            mas_status = resp.json()
            state = mas_status.get("state", "UNKNOWN")
            print(f"  State: {state}")
            if state in ("ACTIVE", "ONLINE", "READY"):
                print("Multi-Agent Supervisor is ready!")
                break
            elif state in ("FAILED", "ERROR"):
                print(f"MAS provisioning failed: {mas_status}")
                break
        else:
            print(f"  Polling error: {resp.status_code}")

        time.sleep(30)
else:
    print("Skipping poll -- fill in MAS_ID or GENIE_SPACE_ID first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Test Routing -- Genie Tool
# MAGIC
# MAGIC This question is about **current fleet data** and should route to the Genie Space.

# COMMAND ----------

def ask_supervisor(question, mas_id=MAS_ID):
    """Send a question to the Multi-Agent Supervisor and print the response."""
    print(f"Q: {question}")
    print("-" * 60)

    response = requests.post(
        f"{host}/api/2.0/multi-agent-supervisors/{mas_id}/chat",
        headers=headers,
        json={"messages": [{"role": "user", "content": question}]},
    )

    if response.status_code == 200:
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
        print(f"A: {answer}")

        # Show which tools were used
        tool_calls = result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        if tool_calls:
            print(f"\nTools used:")
            for tc in tool_calls:
                print(f"  - {tc.get('function', {}).get('name', 'Unknown')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    print("=" * 60)
    print()

# COMMAND ----------

# Test 1: Should route to GENIE (structured data query)
ask_supervisor(
    "How many GPUs are currently showing thermal warnings?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Test Routing -- Knowledge Assistant Tool
# MAGIC
# MAGIC This question is about **procedures from documentation** and should route to the KA.

# COMMAND ----------

# Test 2: Should route to KNOWLEDGE ASSISTANT (runbook procedure)
ask_supervisor(
    "What is the runbook procedure for thermal throttling?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Test Routing -- BOTH Tools
# MAGIC
# MAGIC This question needs **data from Genie** (how many GPUs have warnings) AND
# MAGIC **procedures from the KA** (what to do about it). The supervisor should
# MAGIC consult both tools and combine the answers.

# COMMAND ----------

# Test 3: Should route to BOTH tools (data + procedure)
ask_supervisor(
    "We have 5 GPUs showing thermal warnings -- what should we do?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary & Key Takeaways
# MAGIC
# MAGIC | Component | What We Built | Key Concept |
# MAGIC |-----------|--------------|-------------|
# MAGIC | **Vector Search** | Index on GPU runbook chunks | Delta Sync auto-updates embeddings as source data changes |
# MAGIC | **Knowledge Assistant** | "GPU Operations Advisor" | Managed RAG -- just point at a Volume, no manual chunking needed |
# MAGIC | **Multi-Agent Supervisor** | "DGX Operations Center" | Intelligent routing between structured data (Genie) and unstructured docs (KA) |
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC                    +---------------------------+
# MAGIC                    |   DGX Operations Center   |
# MAGIC                    |   (Multi-Agent Supervisor)|
# MAGIC                    +----------+----------------+
# MAGIC                               |
# MAGIC              +----------------+----------------+
# MAGIC              |                                 |
# MAGIC   +----------v----------+         +-----------v-----------+
# MAGIC   | DGX Fleet Analytics |         | GPU Operations Advisor|
# MAGIC   |   (Genie Space)     |         | (Knowledge Assistant) |
# MAGIC   +----------+----------+         +-----------+-----------+
# MAGIC              |                                 |
# MAGIC   +----------v----------+         +-----------v-----------+
# MAGIC   | Structured Tables   |         |   UC Volume (PDFs)    |
# MAGIC   | (GPU telemetry)     |         |   (GPU Runbooks)      |
# MAGIC   +---------------------+         +-----------------------+
# MAGIC ```
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC - **Notebook 04**: Model deployment and A/B testing on DGX Cloud
# MAGIC - Try adding more knowledge sources (Confluence, SharePoint) to the KA
# MAGIC - Experiment with custom instructions to improve routing accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC Uncomment and run the cells below to clean up resources created in this notebook.

# COMMAND ----------

# # Delete Vector Search Index
# w.vector_search_indexes.delete_index(index_name=VS_INDEX_NAME)
# print(f"Deleted index: {VS_INDEX_NAME}")

# COMMAND ----------

# # Delete Vector Search Endpoint
# w.vector_search_endpoints.delete_endpoint(endpoint_name=VS_ENDPOINT_NAME)
# print(f"Deleted endpoint: {VS_ENDPOINT_NAME}")

# COMMAND ----------

# # Delete source table
# spark.sql(f"DROP TABLE IF EXISTS {SOURCE_TABLE}")
# print(f"Dropped table: {SOURCE_TABLE}")

# COMMAND ----------

# # Note: Knowledge Assistants and Multi-Agent Supervisors can be deleted via UI
# # or via the REST API:
# # requests.delete(f"{host}/api/2.0/knowledge-assistants/{KA_ID}", headers=headers)
# # requests.delete(f"{host}/api/2.0/multi-agent-supervisors/{MAS_ID}", headers=headers)
