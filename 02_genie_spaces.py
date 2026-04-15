# Databricks notebook source

# MAGIC %md
# MAGIC # Genie Spaces: Natural Language SQL for GPU Fleet Analytics
# MAGIC
# MAGIC *Turn plain-English questions into governed SQL over your DGX telemetry data.*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC - Create a **Genie Space** with curated tables and domain instructions via the Databricks SDK
# MAGIC - Query Genie **programmatically** through the Conversation API
# MAGIC - Understand how Genie fits into **MCP-style agentic architectures**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Build a Genie Space

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Initialize the Databricks SDK

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print(f"Connected to: {w.config.host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Define Genie Space Configuration
# MAGIC
# MAGIC Specify the tables, instructions, and sample questions that make up the Genie Space.

# COMMAND ----------

CATALOG = "main"
SCHEMA = "mlops_genai_workshop"
SPACE_NAME = "DGX Fleet Analytics"

tables = [
    f"{CATALOG}.{SCHEMA}.gpu_telemetry",
    f"{CATALOG}.{SCHEMA}.cluster_inventory",
    f"{CATALOG}.{SCHEMA}.ml_job_runs",
]

genie_instructions = """
You are a GPU fleet analytics assistant for an NVIDIA DGX Cloud environment.

## Domain Context
- **gpu_telemetry** contains per-GPU metrics sampled every 60 seconds: utilization_pct (0-100), memory_used_gb, temperature_celsius, ecc_errors (cumulative single-bit ECC error count), power_draw_watts, and gpu_clock_mhz.
- **cluster_inventory** describes each DGX cluster: cluster_id, cluster_name, cloud_provider (AWS | GCP | Azure), region, gpu_model (e.g. A100, H100), gpu_count, and status (active | maintenance | decommissioned).
- **ml_job_runs** records every ML training/inference job: job_id, job_name, cluster_id, start_time, end_time, gpu_hours_consumed, framework (PyTorch | JAX | TensorFlow), status (completed | failed | running), and cost_usd.

## Join Patterns
- Join gpu_telemetry to cluster_inventory on cluster_id to get cluster metadata for telemetry data.
- Join ml_job_runs to cluster_inventory on cluster_id to enrich job data with hardware info.

## Unit Clarifications
- GPU utilization is a percentage (0-100). "High utilization" means > 80%.
- gpu_hours_consumed is the total GPU-hours a job used (gpus * wall_clock_hours).
- cost_usd is the billed cost of the job.
- ECC errors are cumulative; use SUM or MAX depending on whether you want total or peak.
- Temperature is in Celsius. Throttling typically begins above 83 C.

## Conventions
- When asked about "this week", use the last 7 days from the current date.
- Default to ordering results by the most relevant metric descending.
- If a question is ambiguous, prefer the interpretation most useful to an MLOps engineer.
""".strip()

sample_questions = [
    "Which cluster has the most GPU errors this week?",
    "What is the average GPU utilization across all AWS clusters?",
    "Show me the top 10 most expensive ML jobs by GPU-hours",
]

print("Configuration ready.")
print(f"  Space name : {SPACE_NAME}")
print(f"  Tables     : {len(tables)}")
print(f"  Sample Qs  : {len(sample_questions)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Create the Genie Space
# MAGIC
# MAGIC Programmatic equivalent of **New > Genie Space** in the UI -- reproducible and version-controllable.

# COMMAND ----------

from databricks.sdk.service.dashboards import GenieSpace, GenieTableIdentifier

table_identifiers = [GenieTableIdentifier(table_name=t) for t in tables]

space = w.genie.create(
    title=SPACE_NAME,
    description="Natural-language analytics for NVIDIA DGX GPU fleet: telemetry, inventory, and ML job data.",
    table_identifiers=table_identifiers,
    instructions=genie_instructions,
    sample_questions=sample_questions,
)

genie_space_id = space.space_id
print(f"Genie Space created!")
print(f"  Space ID : {genie_space_id}")
print(f"  Title    : {space.title}")
print(f"  URL      : {w.config.host}/genie/rooms/{genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC > Save your Space ID -- you will need it in Part B: `genie_space_id = "<your-id>"`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: Genie Conversation API
# MAGIC
# MAGIC Interact with the Genie Space programmatically -- the foundation for agentic workflows where an LLM delegates SQL questions to Genie.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Set Your Genie Space ID

# COMMAND ----------

# Uncomment if starting from Part B directly:
# genie_space_id = "<your-genie-space-id>"

print(f"Using Genie Space: {genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Helper -- Poll for Genie Results
# MAGIC
# MAGIC The Conversation API is asynchronous, so we poll until the response is ready.

# COMMAND ----------

import time


def poll_genie_response(client, space_id, conversation_id, message_id, timeout=120, poll_interval=3):
    """Poll a Genie message until it completes or times out.

    Returns the completed GenieMessage object. Raises TimeoutError or
    RuntimeError on failure.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Genie did not respond within {timeout}s")

        message = client.genie.get_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        status = message.status
        if status == "COMPLETED":
            print(f"Response ready ({elapsed:.1f}s)")
            return message
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Genie message {status}: {message}")
        else:
            print(f"  Status: {status} ... ({elapsed:.1f}s elapsed)")
            time.sleep(poll_interval)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Start a Conversation

# COMMAND ----------

question_1 = "Which cluster has the most GPU errors this week?"

response = w.genie.start_conversation(
    space_id=genie_space_id,
    content=question_1,
)

conversation_id = response.conversation_id
message_id = response.message_id

print(f"Conversation started!")
print(f"  Conversation ID : {conversation_id}")
print(f"  Message ID      : {message_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Poll and Retrieve the Result

# COMMAND ----------

message = poll_genie_response(w, genie_space_id, conversation_id, message_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Extract SQL and Query Result
# MAGIC
# MAGIC A completed Genie message contains attachments with the generated SQL and tabular data.

# COMMAND ----------

def extract_genie_sql(message):
    """Extract the generated SQL from a Genie message's attachments."""
    if message.attachments:
        for attachment in message.attachments:
            if attachment.query and attachment.query.query:
                return attachment.query.query
    return None


def get_genie_query_result(client, space_id, conversation_id, message_id):
    """Fetch the query result from a completed Genie message."""
    message = client.genie.get_message(
        space_id=space_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    if message.attachments:
        for attachment in message.attachments:
            if attachment.query and attachment.query.query:
                result = client.genie.get_message_query_result(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    attachment_id=attachment.id,
                )
                return result
    return None


sql = extract_genie_sql(message)
if sql:
    print("Generated SQL:\n")
    print(sql)
else:
    print("No SQL found in response.")

# COMMAND ----------

result = get_genie_query_result(w, genie_space_id, conversation_id, message_id)

if result:
    print(f"Columns: {[col.name for col in result.statement_response.manifest.schema.columns]}")
    print(f"Row count: {len(result.statement_response.result.data_array)}")
    print("\nFirst 5 rows:")
    for row in result.statement_response.result.data_array[:5]:
        print(row)
else:
    print("No query result available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Multi-Turn Follow-Up Questions
# MAGIC
# MAGIC Genie conversations support multi-turn dialogue -- follow-up questions share context.

# COMMAND ----------

questions = [
    "What is the average GPU utilization across all AWS clusters?",
    "Show me the top 10 most expensive ML jobs by GPU-hours",
]

for q in questions:
    print(f"\n{'='*70}")
    print(f"Question: {q}")
    print("=" * 70)

    followup = w.genie.create_message(
        space_id=genie_space_id,
        conversation_id=conversation_id,
        content=q,
    )

    msg = poll_genie_response(
        w, genie_space_id, conversation_id, followup.id
    )

    sql = extract_genie_sql(msg)
    if sql:
        print(f"\nGenerated SQL:\n{sql}")

    result = get_genie_query_result(
        w, genie_space_id, conversation_id, followup.id
    )
    if result:
        print(f"\nRows returned: {len(result.statement_response.result.data_array)}")
        for row in result.statement_response.result.data_array[:5]:
            print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Discussion: MCP-Style Patterns with Genie
# MAGIC
# MAGIC The Model Context Protocol (MCP) lets LLM agents call external tools in a standardized way.
# MAGIC The Genie Conversation API is a natural fit -- an agent delegates SQL questions to Genie
# MAGIC rather than writing SQL itself.
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC +----------------+       +----------------+       +----------------+       +----------------+
# MAGIC |                |       |                |       |                |       |                |
# MAGIC |  Claude Code   | ----> |   MCP Server   | ----> |   Genie API    | ----> | SQL Warehouse  |
# MAGIC |  (LLM Agent)   |       |  (Tool Host)   |       | (NL-to-SQL)    |       |  (Execution)   |
# MAGIC |                |       |                |       |                |       |                |
# MAGIC +----------------+       +----------------+       +----------------+       +----------------+
# MAGIC         |                        |                        |                        |
# MAGIC    User asks a               Exposes                 Translates              Runs query,
# MAGIC    natural language          `ask_genie`             question to             returns
# MAGIC    question                  as a tool               SQL                     tabular data
# MAGIC ```
# MAGIC
# MAGIC ### Tool-Building Workflow
# MAGIC
# MAGIC | Step | Action | Detail |
# MAGIC |------|--------|--------|
# MAGIC | 1 | **Create Genie Space** | Curate tables + instructions (Part A) |
# MAGIC | 2 | **Wrap in a tool function** | Python function calling `start_conversation` / `get_message_query_result` |
# MAGIC | 3 | **Register as MCP tool** | Expose via an MCP server (e.g., `mcp` Python SDK or FastAPI) |
# MAGIC | 4 | **Connect your agent** | Point Claude Code, LangChain, or any MCP client at your server |
# MAGIC
# MAGIC Genie generates SQL scoped to the tables and instructions you define (no hallucinated table names), the LLM agent never writes raw SQL, every conversation is logged for audit, and you can compose Genie tools with vector search, model serving, and job triggers in a single agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cleanup (Optional)

# COMMAND ----------

# w.genie.delete(space_id=genie_space_id)
# print(f"Genie Space {genie_space_id} deleted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC | What We Did | How |
# MAGIC |---|---|
# MAGIC | Created a Genie Space with GPU fleet tables | `w.genie.create()` via SDK |
# MAGIC | Added domain instructions and sample questions | Natural-language context in space config |
# MAGIC | Queried Genie programmatically | `start_conversation()` + `poll_genie_response()` |
# MAGIC | Extracted generated SQL and result data | `get_message_query_result()` via attachments |
# MAGIC | Discussed MCP integration pattern | Genie as an MCP tool in agentic architectures |
# MAGIC
# MAGIC **Next up: Notebook 03 -- Building an MCP Server**
