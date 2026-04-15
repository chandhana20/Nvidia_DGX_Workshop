# Databricks notebook source

# MAGIC %md
# MAGIC # Block 2: GenAI Foundations on Databricks
# MAGIC **NVIDIA DGX Cloud MLOps & GenAI Workshop**
# MAGIC
# MAGIC | Detail | Value |
# MAGIC |---|---|
# MAGIC | Duration | 40 minutes hands-on |
# MAGIC | Schema | `main.mlops_genai_workshop` |
# MAGIC | Cluster | Shared / Serverless SQL Warehouse |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this notebook you will be able to:
# MAGIC
# MAGIC 1. **Use built-in AI Functions** (`ai_classify`, `ai_extract`, `ai_summarize`) to enrich structured data with LLM intelligence — no model deployment required
# MAGIC 2. **Invoke Claude directly from SQL** using `ai_query` with Anthropic's `claude-sonnet-4-5-20250929` model via Databricks Foundation Model APIs
# MAGIC 3. **Generate and execute SQL programmatically** by combining the Anthropic Python SDK with Spark SQL for natural-language-to-SQL workflows
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Part A — AI Functions in SQL (20 min)
# MAGIC
# MAGIC Databricks **AI Functions** let you call large language models directly inside SQL queries. They run on Databricks-managed infrastructure — no endpoints to deploy, no tokens to manage.
# MAGIC
# MAGIC We will work with three functions:
# MAGIC
# MAGIC | Function | Purpose |
# MAGIC |---|---|
# MAGIC | `ai_classify` | Categorize free-text into a fixed set of labels |
# MAGIC | `ai_extract` | Pull structured fields out of unstructured text |
# MAGIC | `ai_summarize` | Condense text into a short summary |

# COMMAND ----------

# MAGIC %md
# MAGIC ## A.1 — `ai_classify`: Root-Cause Classification of GPU Health Events
# MAGIC
# MAGIC Our `gpu_health_events` table contains free-text descriptions of GPU incidents across DGX clusters. We want to classify each event into one of five root-cause categories so downstream dashboards and alerting can aggregate by cause.
# MAGIC
# MAGIC **Categories:** `hardware_failure`, `thermal_throttle`, `memory_error`, `software_bug`, `network_issue`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Preview the raw GPU health events
# MAGIC SELECT event_id, cluster_id, gpu_id, event_description, severity, event_timestamp
# MAGIC FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Classify each GPU health event by root cause using ai_classify
# MAGIC SELECT
# MAGIC   event_id,
# MAGIC   cluster_id,
# MAGIC   gpu_id,
# MAGIC   severity,
# MAGIC   event_description,
# MAGIC   ai_classify(
# MAGIC     event_description,
# MAGIC     ARRAY('hardware_failure', 'thermal_throttle', 'memory_error', 'software_bug', 'network_issue')
# MAGIC   ) AS root_cause
# MAGIC FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint A.1
# MAGIC
# MAGIC Notice how `ai_classify` returns exactly one of the five labels for every row — no prompt engineering required. This is ideal for building governed classification pipelines you can materialize into Gold tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A.2 — `ai_extract`: Structured Extraction from ML Job Descriptions
# MAGIC
# MAGIC The `ml_job_runs` table has a free-text `description` column that engineers fill in when submitting training jobs. We want to extract four structured fields from each description:
# MAGIC
# MAGIC | Field | Example |
# MAGIC |---|---|
# MAGIC | `framework` | PyTorch, TensorFlow, JAX |
# MAGIC | `model_size` | 7B, 13B, 70B |
# MAGIC | `dataset` | ImageNet, Common Crawl, custom |
# MAGIC | `objective` | fine-tuning, pre-training, RLHF |

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Preview the ML job runs
# MAGIC SELECT job_id, job_name, description, status, cluster_id, start_time
# MAGIC FROM main.mlops_genai_workshop.ml_job_runs
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Extract structured metadata from free-text job descriptions
# MAGIC SELECT
# MAGIC   job_id,
# MAGIC   job_name,
# MAGIC   description,
# MAGIC   ai_extract(
# MAGIC     description,
# MAGIC     ARRAY('framework', 'model_size', 'dataset', 'objective')
# MAGIC   ) AS extracted_metadata
# MAGIC FROM main.mlops_genai_workshop.ml_job_runs
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint A.2
# MAGIC
# MAGIC `ai_extract` returns a JSON struct with the requested keys. When a field cannot be determined from the text, the model returns `null` for that key. You can explode this struct into columns for downstream joins and aggregations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A.3 — `ai_summarize`: Cluster Health Summaries
# MAGIC
# MAGIC Operations teams need a quick roll-up of what happened on each cluster. We will group events by `cluster_id`, concatenate the descriptions, and ask `ai_summarize` to produce a concise summary.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summarize GPU health events per cluster
# MAGIC WITH cluster_events AS (
# MAGIC   SELECT
# MAGIC     cluster_id,
# MAGIC     concat_ws(' | ', collect_list(event_description)) AS combined_events,
# MAGIC     count(*) AS event_count
# MAGIC   FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC   GROUP BY cluster_id
# MAGIC )
# MAGIC SELECT
# MAGIC   cluster_id,
# MAGIC   event_count,
# MAGIC   ai_summarize(combined_events, 80) AS cluster_health_summary
# MAGIC FROM cluster_events
# MAGIC ORDER BY event_count DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint A.3
# MAGIC
# MAGIC `ai_summarize` accepts an optional max-word-count parameter (we used 80 words). This keeps summaries tight for dashboards and Slack alerts while capturing the key themes across all events on a cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part B — Invoking Claude from SQL (10 min)
# MAGIC
# MAGIC Databricks Foundation Model APIs give you access to frontier models — including Anthropic Claude — through a unified `ai_query` interface. No API keys, no external network calls. The model is served inside your Databricks environment.
# MAGIC
# MAGIC In this section we send **critical GPU health events** to Claude Sonnet 4.5 and ask it to analyze the situation and recommend remediation actions.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Analyze critical GPU health events with Claude and recommend actions
# MAGIC SELECT
# MAGIC   event_id,
# MAGIC   cluster_id,
# MAGIC   gpu_id,
# MAGIC   severity,
# MAGIC   event_description,
# MAGIC   ai_query(
# MAGIC     'claude-sonnet-4-5-20250929',
# MAGIC     CONCAT(
# MAGIC       'You are an expert NVIDIA DGX systems engineer. Analyze the following critical GPU health event and provide:\n',
# MAGIC       '1. A brief root-cause analysis (2-3 sentences)\n',
# MAGIC       '2. Immediate recommended action\n',
# MAGIC       '3. Preventive measure for the future\n\n',
# MAGIC       'Event: ', event_description, '\n',
# MAGIC       'Severity: ', severity, '\n',
# MAGIC       'GPU ID: ', gpu_id, '\n',
# MAGIC       'Cluster ID: ', cluster_id
# MAGIC     )
# MAGIC   ) AS claude_analysis
# MAGIC FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC WHERE severity = 'critical'
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint B
# MAGIC
# MAGIC Key takeaways:
# MAGIC - `ai_query` calls Claude Sonnet 4.5 directly — no endpoint provisioning, no external API key
# MAGIC - You can embed rich prompts using `CONCAT` and reference any column in your table
# MAGIC - This pattern is powerful for enrichment pipelines: run an LLM over every row and materialize the results into a Silver or Gold table
# MAGIC - All calls are governed by Unity Catalog permissions — the same access controls that protect your tables also protect your AI function calls

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part C — Python-to-SQL Generation (10 min)
# MAGIC
# MAGIC In this final section we combine the **Anthropic Python SDK** with **Spark SQL** to build a natural-language-to-SQL pipeline:
# MAGIC
# MAGIC 1. Define the table schemas so the model has context
# MAGIC 2. Send a natural language question to Claude
# MAGIC 3. Extract the generated SQL
# MAGIC 4. Execute it with `spark.sql()`
# MAGIC
# MAGIC This pattern is the foundation for text-to-SQL agents, Genie-like experiences, and self-service analytics copilots.

# COMMAND ----------

# Install the Anthropic SDK (already available on most Databricks runtimes with ML)
%pip install anthropic --quiet
dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import re

# ------------------------------------------------------------------
# Table schemas — we give the model full context so it generates
# accurate SQL without hallucinating column names.
# ------------------------------------------------------------------
TABLE_SCHEMAS = """
### Table: main.mlops_genai_workshop.gpu_health_events
Columns:
  - event_id (STRING): Unique identifier for the health event
  - cluster_id (STRING): DGX cluster identifier
  - gpu_id (STRING): Individual GPU identifier within the cluster
  - event_description (STRING): Free-text description of the GPU event
  - severity (STRING): Event severity level (critical, warning, info)
  - gpu_temperature_celsius (DOUBLE): GPU temperature at the time of the event
  - gpu_utilization_pct (DOUBLE): GPU utilization percentage at the time of the event
  - memory_used_gb (DOUBLE): GPU memory used in GB
  - event_timestamp (TIMESTAMP): When the event occurred

### Table: main.mlops_genai_workshop.ml_job_runs
Columns:
  - job_id (STRING): Unique identifier for the ML job
  - job_name (STRING): Human-readable job name
  - description (STRING): Free-text description of the training job
  - status (STRING): Job status (running, completed, failed)
  - cluster_id (STRING): DGX cluster the job ran on
  - gpu_count (INT): Number of GPUs allocated to the job
  - start_time (TIMESTAMP): Job start time
  - end_time (TIMESTAMP): Job end time
  - duration_minutes (DOUBLE): Total job duration in minutes

### Table: main.mlops_genai_workshop.cluster_metrics
Columns:
  - cluster_id (STRING): DGX cluster identifier
  - metric_timestamp (TIMESTAMP): When the metric was recorded
  - avg_gpu_temperature_celsius (DOUBLE): Average GPU temperature across all GPUs in the cluster
  - avg_gpu_utilization_pct (DOUBLE): Average GPU utilization across all GPUs in the cluster
  - total_memory_used_gb (DOUBLE): Total GPU memory used across the cluster
  - active_job_count (INT): Number of active jobs running on the cluster
"""

# ------------------------------------------------------------------
# The natural language question we want to answer
# ------------------------------------------------------------------
QUESTION = "Which GPU clusters had the highest average temperature in the last 24 hours?"

print(f"Question: {QUESTION}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate SQL from natural language using Claude

# COMMAND ----------

# Initialize the Anthropic client
# On Databricks, the ANTHROPIC_API_KEY is typically set via the Foundation Model API
# or through workspace secrets
client = anthropic.Anthropic()

# Build the prompt with schema context
prompt = f"""You are a Databricks SQL expert. Given the following table schemas, generate a single
Databricks SQL query that answers the user's question. Return ONLY the SQL query — no explanation,
no markdown code fences, no commentary.

{TABLE_SCHEMAS}

Question: {QUESTION}

Important:
- Use only the tables and columns listed above
- Use current_timestamp() for the current time in Databricks SQL
- Return results ordered by the relevant metric descending
- Limit to the top 10 results
"""

# Call Claude to generate the SQL
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

generated_sql = response.content[0].text.strip()

# Clean up any accidental markdown fences
generated_sql = re.sub(r"^```(?:sql)?\s*", "", generated_sql)
generated_sql = re.sub(r"\s*```$", "", generated_sql)

print("Generated SQL:")
print("-" * 60)
print(generated_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute the generated SQL

# COMMAND ----------

# Execute the LLM-generated SQL using Spark
result_df = spark.sql(generated_sql)
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint C
# MAGIC
# MAGIC What just happened:
# MAGIC 1. We provided Claude with the **exact table schemas** — this is the single most important step for accurate SQL generation
# MAGIC 2. Claude generated a syntactically correct Databricks SQL query targeting the right tables and columns
# MAGIC 3. We executed the SQL directly with `spark.sql()` and displayed the results
# MAGIC
# MAGIC **Production considerations:**
# MAGIC - Always validate generated SQL before execution (syntax check, table/column allow-list)
# MAGIC - Use parameterized prompts and few-shot examples for higher accuracy
# MAGIC - Wrap execution in a try/except to handle generation errors gracefully
# MAGIC - Consider Databricks **Genie Spaces** for a fully managed text-to-SQL experience with governance built in

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC | Section | What You Learned |
# MAGIC |---|---|
# MAGIC | **Part A** | `ai_classify`, `ai_extract`, `ai_summarize` — zero-deployment AI functions for classification, extraction, and summarization directly in SQL |
# MAGIC | **Part B** | `ai_query` with Claude Sonnet 4.5 — invoke frontier models from SQL for complex reasoning and recommendations |
# MAGIC | **Part C** | Python SDK + Spark SQL — build natural-language-to-SQL pipelines using Claude as the generation engine |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Block 3:** We will deploy custom models on NVIDIA DGX Cloud with MLflow and Mosaic AI Model Serving
# MAGIC - Try modifying the prompts in Part B to generate different output formats (JSON, Markdown tables)
# MAGIC - Experiment with `ai_query` parameters like `max_tokens` and `temperature` for different use cases
