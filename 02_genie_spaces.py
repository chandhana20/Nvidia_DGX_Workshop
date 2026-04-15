# Databricks notebook source

# MAGIC %md
# MAGIC # Block 3: Genie Spaces -- Natural Language SQL for GPU Fleet Analytics
# MAGIC
# MAGIC **NVIDIA DGX Cloud MLOps & GenAI Workshop**
# MAGIC
# MAGIC | Detail | Value |
# MAGIC |---|---|
# MAGIC | Duration | 40 minutes hands-on |
# MAGIC | Prerequisite | Block 2 (Data Foundation) complete -- tables in `main.mlops_genai_workshop` |
# MAGIC | Outcome | A working Genie Space + programmatic Genie Conversation API access |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this block you will be able to:
# MAGIC
# MAGIC 1. **Create a Genie Space** with curated tables and domain-specific instructions using the Databricks SDK
# MAGIC 2. **Interact with Genie programmatically** via the Conversation API (start conversations, poll for results, extract SQL)
# MAGIC 3. **Understand MCP-style patterns** where Genie acts as a "tool" in an agentic architecture

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part A -- Build a Genie Space (20 min)
# MAGIC
# MAGIC In this section we will:
# MAGIC - Create a Genie Space called **"DGX Fleet Analytics"**
# MAGIC - Register three tables from `main.mlops_genai_workshop`
# MAGIC - Provide natural-language instructions so Genie understands GPU fleet terminology
# MAGIC - Define sample questions for end users

# COMMAND ----------

# MAGIC %md
# MAGIC ## A.1 -- Initialize the Databricks SDK

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print(f"Connected to: {w.config.host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A.2 -- Define Genie Space Configuration
# MAGIC
# MAGIC We specify the tables, instructions, and sample questions that make up the Genie Space.

# COMMAND ----------

# ---- Configuration ----

CATALOG = "main"
SCHEMA = "mlops_genai_workshop"
SPACE_NAME = "DGX Fleet Analytics"

# Tables to include in the Genie Space
tables = [
    f"{CATALOG}.{SCHEMA}.gpu_telemetry",
    f"{CATALOG}.{SCHEMA}.cluster_inventory",
    f"{CATALOG}.{SCHEMA}.ml_job_runs",
]

# Natural-language instructions that give Genie domain context
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

# Sample questions that appear in the Genie Space UI
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
# MAGIC ## A.3 -- Create the Genie Space
# MAGIC
# MAGIC We use the Databricks SDK to create the space programmatically. This is equivalent to clicking
# MAGIC **New > Genie Space** in the workspace UI, but is reproducible and version-controllable.

# COMMAND ----------

from databricks.sdk.service.dashboards import GenieSpace, GenieTableIdentifier

# Build table identifiers
table_identifiers = [GenieTableIdentifier(table_name=t) for t in tables]

# Create the Genie Space
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
# MAGIC > **Save your Space ID!** You will need it in Part B. Copy it from the output above or from the URL.
# MAGIC >
# MAGIC > `genie_space_id = "<your-genie-space-id>"`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part B -- Genie Conversation API (20 min)
# MAGIC
# MAGIC Now we will interact with the Genie Space **programmatically**. This is the foundation for
# MAGIC building agentic workflows where an LLM can ask Genie questions on behalf of a user.

# COMMAND ----------

# MAGIC %md
# MAGIC ## B.1 -- Set Your Genie Space ID
# MAGIC
# MAGIC If you created the space in Part A above, the variable `genie_space_id` is already set.
# MAGIC Otherwise, paste your Space ID below.

# COMMAND ----------

# Uncomment and set this if you are starting from Part B directly:
# genie_space_id = "<your-genie-space-id>"

print(f"Using Genie Space: {genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B.2 -- Helper: Poll for Genie Results
# MAGIC
# MAGIC The Genie Conversation API is **asynchronous**. After starting a conversation or sending a
# MAGIC follow-up message, we need to poll until the response is ready.

# COMMAND ----------

import time


def poll_genie_response(client, space_id, conversation_id, message_id, timeout=120, poll_interval=3):
    """
    Poll a Genie message until it completes or times out.

    Parameters
    ----------
    client : WorkspaceClient
        Authenticated Databricks SDK client.
    space_id : str
        The Genie Space ID.
    conversation_id : str
        The conversation ID returned by start_conversation or create_message.
    message_id : str
        The message ID returned by start_conversation or create_message.
    timeout : int
        Maximum seconds to wait before raising a TimeoutError.
    poll_interval : int
        Seconds between polling attempts.

    Returns
    -------
    message : GenieMessage
        The completed Genie message object.
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
# MAGIC ## B.3 -- Start a Conversation
# MAGIC
# MAGIC Let's ask Genie our first question programmatically.

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
# MAGIC ## B.4 -- Poll and Retrieve the Result

# COMMAND ----------

message = poll_genie_response(w, genie_space_id, conversation_id, message_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## B.5 -- Extract the SQL and Query Result
# MAGIC
# MAGIC A completed Genie message contains **attachments**. The SQL attachment holds the generated
# MAGIC query, and we can fetch the actual data result separately.

# COMMAND ----------

def extract_genie_sql(message):
    """Extract the generated SQL from a Genie message's attachments."""
    if message.attachments:
        for attachment in message.attachments:
            if attachment.query and attachment.query.query:
                return attachment.query.query
    return None


def get_genie_query_result(client, space_id, conversation_id, message_id):
    """Fetch the query result DataFrame from a completed Genie message."""
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


# Extract and display the SQL
sql = extract_genie_sql(message)
if sql:
    print("Generated SQL:")
    print("-" * 60)
    print(sql)
    print("-" * 60)
else:
    print("No SQL found in response.")

# COMMAND ----------

# Fetch the actual query result
result = get_genie_query_result(w, genie_space_id, conversation_id, message_id)

if result:
    print(f"\nQuery result columns: {[col.name for col in result.statement_response.manifest.schema.columns]}")
    print(f"Row count: {len(result.statement_response.result.data_array)}")
    print("\nFirst 5 rows:")
    for row in result.statement_response.result.data_array[:5]:
        print(row)
else:
    print("No query result available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B.6 -- Send Additional Questions in the Same Conversation
# MAGIC
# MAGIC Genie conversations support multi-turn dialogue. Follow-up questions share context.

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
# MAGIC # Discussion: MCP-Style Patterns with Genie
# MAGIC
# MAGIC ## Genie as a "Tool" in an MCP Server
# MAGIC
# MAGIC The Model Context Protocol (MCP) lets LLM agents call external **tools** in a standardized way.
# MAGIC The Genie Conversation API is a natural fit: an agent can delegate SQL questions to Genie
# MAGIC rather than writing SQL itself.
# MAGIC
# MAGIC ### Architecture Pattern
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
# MAGIC ### Quick Tool-Building Workflow (4 Steps)
# MAGIC
# MAGIC | Step | Action | Detail |
# MAGIC |------|--------|--------|
# MAGIC | 1 | **Create Genie Space** | Curate tables + instructions (as we did in Part A) |
# MAGIC | 2 | **Wrap in a tool function** | Write a Python function that calls `start_conversation` / `get_message_query_result` |
# MAGIC | 3 | **Register as MCP tool** | Expose the function via an MCP server (e.g., using `mcp` Python SDK or FastAPI) |
# MAGIC | 4 | **Connect your agent** | Point Claude Code, LangChain, or any MCP client at your server |
# MAGIC
# MAGIC ### Why This Matters for MLOps Engineers
# MAGIC
# MAGIC - **Governed SQL**: Genie generates SQL scoped to the tables and instructions you define -- no hallucinated table names
# MAGIC - **No SQL in the agent**: The LLM agent never writes raw SQL; it delegates to a domain expert (Genie)
# MAGIC - **Audit trail**: Every Genie conversation is logged with the generated SQL and results
# MAGIC - **Composable**: Combine Genie tools with other MCP tools (vector search, model serving, job triggers) in a single agent

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC Uncomment the cell below to delete the Genie Space created in this notebook.

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
# MAGIC | Created a Genie Space with GPU fleet tables | Databricks SDK `w.genie.create()` |
# MAGIC | Added domain instructions and sample questions | Natural-language context in the space config |
# MAGIC | Started a conversation programmatically | `w.genie.start_conversation()` |
# MAGIC | Polled for async results | Custom `poll_genie_response()` helper |
# MAGIC | Extracted SQL and result data | `get_message_query_result()` via attachments |
# MAGIC | Discussed MCP integration pattern | Genie as an MCP tool in agentic architectures |
# MAGIC
# MAGIC **Next up: Block 4 -- Building an MCP Server**
