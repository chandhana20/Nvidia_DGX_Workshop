# Databricks notebook source
# MAGIC %md
# MAGIC # GenAI Foundations on Databricks
# MAGIC *Built-in AI functions, Claude via SQL, and natural-language-to-SQL -- all without deploying a model.*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **What you'll build:**
# MAGIC - Classify, extract, and summarize GPU telemetry using SQL AI functions
# MAGIC - Invoke Claude Sonnet 4.5 directly from SQL with `ai_query`
# MAGIC - Generate and execute Spark SQL from natural language using the Anthropic Python SDK

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: AI Functions in SQL
# MAGIC
# MAGIC `ai_classify`, `ai_extract`, and `ai_summarize` call LLMs inside SQL queries on Databricks-managed infrastructure. No endpoints or tokens to manage.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Preview raw GPU health events
# MAGIC SELECT event_id, cluster_id, gpu_id, event_description, severity, event_timestamp
# MAGIC FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Classify each event into a root-cause category
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

# MAGIC %sql
# MAGIC -- Preview ML job runs with free-text descriptions
# MAGIC SELECT job_id, job_name, description, status, cluster_id, start_time
# MAGIC FROM main.mlops_genai_workshop.ml_job_runs
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Extract structured metadata (framework, model_size, dataset, objective) from descriptions
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

# MAGIC %sql
# MAGIC -- Summarize all GPU health events per cluster (max 80 words)
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
# MAGIC ## Part B: Invoking Claude from SQL
# MAGIC
# MAGIC `ai_query` gives you access to frontier models like Claude Sonnet 4.5 through a unified SQL interface -- no API keys, no external network calls.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Send critical GPU events to Claude for root-cause analysis and remediation recommendations
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
# MAGIC All `ai_query` calls are governed by Unity Catalog permissions -- the same access controls that protect your tables also protect your AI function calls.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Natural-Language-to-SQL with the Anthropic Python SDK
# MAGIC
# MAGIC Combine Claude with `spark.sql()` to turn plain English into executable queries. This pattern underpins text-to-SQL agents and self-service analytics copilots.

# COMMAND ----------

# Install the Anthropic SDK (already available on most Databricks runtimes with ML)
%pip install anthropic --quiet
dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import re

# Table schemas -- full context so the model generates accurate SQL
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

QUESTION = "Which GPU clusters had the highest average temperature in the last 24 hours?"
print(f"Question: {QUESTION}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate SQL from natural language

# COMMAND ----------

client = anthropic.Anthropic()

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
print(generated_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute the generated SQL

# COMMAND ----------

result_df = spark.sql(generated_sql)
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In production, validate generated SQL before execution (syntax checks, column allow-lists) and consider Databricks Genie Spaces for a fully managed text-to-SQL experience with built-in governance.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC | Section | Capability |
# MAGIC |---|---|
# MAGIC | **Part A** | `ai_classify`, `ai_extract`, `ai_summarize` -- zero-deployment LLM functions in SQL |
# MAGIC | **Part B** | `ai_query` with Claude Sonnet 4.5 -- frontier model reasoning from SQL |
# MAGIC | **Part C** | Anthropic Python SDK + Spark SQL -- natural-language-to-SQL generation |
# MAGIC
# MAGIC **Next notebook:** Deploy custom models on NVIDIA DGX Cloud with MLflow and Mosaic AI Model Serving.