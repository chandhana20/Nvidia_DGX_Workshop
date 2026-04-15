# Databricks notebook source

# MAGIC %md
# MAGIC # End-to-End GPU Fleet Operations Pipeline
# MAGIC *Live demo: full pipeline from telemetry spike to automated remediation.*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC | Step | Component | What Happens |
# MAGIC |------|-----------|--------------|
# MAGIC | 1 | **Data Ingestion** | Simulate a GPU telemetry spike on `dgx-aws-03` |
# MAGIC | 2 | **Feature Engineering** | Update the feature table with fresh aggregations |
# MAGIC | 3 | **Model Serving** | Score with `gpu-anomaly-detector` endpoint |
# MAGIC | 4 | **Genie Space** | Natural language query on anomaly status |
# MAGIC | 5 | **Knowledge Assistant** | Retrieve runbook procedures |
# MAGIC | 6 | **Supervisor Agent** | Orchestrate remediation recommendation |
# MAGIC | 7 | **Monitoring** | Observe drift in temperature distributions |
# MAGIC | 8 | **Retraining** | Automated retraining trigger concept |
# MAGIC
# MAGIC Instructor note: Run each cell top-to-bottom. Pause at each section header to explain before executing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

import requests
import json
import time
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    IntegerType,
)

# Update these before the demo
CATALOG = "nvidia_dgx_workshop"
SCHEMA = "gpu_ops"

GENIE_SPACE_ID = "<YOUR_GENIE_SPACE_ID>"          # e.g. "01f0abcd..."
KA_NAME = "<YOUR_KNOWLEDGE_ASSISTANT_NAME>"        # e.g. "gpu-runbook-assistant"
SUPERVISOR_AGENT_NAME = "<YOUR_SUPERVISOR_AGENT>"  # e.g. "gpu-fleet-supervisor"
MODEL_SERVING_ENDPOINT = "gpu-anomaly-detector"

# Databricks host and token (auto-populated in notebooks)
DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

print(f"Workspace:  https://{DATABRICKS_HOST}")
print(f"Catalog:    {CATALOG}")
print(f"Schema:     {SCHEMA}")
print(f"Endpoint:   {MODEL_SERVING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Simulate GPU Telemetry Spike
# MAGIC
# MAGIC Talking points: In production, telemetry streams via Kafka/Kinesis through Auto Loader. Here we inject anomalous readings for cluster `dgx-aws-03` -- temperatures above 90C and utilization near 100% -- to trigger the full pipeline.

# COMMAND ----------

import random
import uuid

now = datetime.utcnow()

# Generate anomalous GPU telemetry for 5 GPUs on dgx-aws-03
anomalous_records = []
for i in range(5):
    gpu_id = f"gpu-a100-{17 + i:03d}"
    for minute_offset in range(5):
        anomalous_records.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": now - timedelta(minutes=minute_offset),
                "cluster_id": "dgx-aws-03",
                "gpu_id": gpu_id,
                "gpu_model": "A100-SXM4-80GB",
                "temperature_celsius": round(random.uniform(89.0, 97.0), 1),
                "gpu_utilization_pct": round(random.uniform(94.0, 99.5), 1),
                "memory_utilization_pct": round(random.uniform(88.0, 96.0), 1),
                "power_draw_watts": round(random.uniform(380.0, 420.0), 1),
                "fan_speed_pct": round(random.uniform(90.0, 100.0), 1),
                "ecc_errors_single_bit": random.randint(0, 3),
                "ecc_errors_double_bit": 0,
                "pcie_throughput_gbps": round(random.uniform(20.0, 25.0), 1),
            }
        )

schema = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("timestamp", TimestampType(), False),
        StructField("cluster_id", StringType(), False),
        StructField("gpu_id", StringType(), False),
        StructField("gpu_model", StringType(), False),
        StructField("temperature_celsius", DoubleType(), False),
        StructField("gpu_utilization_pct", DoubleType(), False),
        StructField("memory_utilization_pct", DoubleType(), False),
        StructField("power_draw_watts", DoubleType(), False),
        StructField("fan_speed_pct", DoubleType(), False),
        StructField("ecc_errors_single_bit", IntegerType(), False),
        StructField("ecc_errors_double_bit", IntegerType(), False),
        StructField("pcie_throughput_gbps", DoubleType(), False),
    ]
)

df_anomalous = spark.createDataFrame(anomalous_records, schema=schema)

df_anomalous.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.gpu_telemetry")

print(f"Inserted {len(anomalous_records)} anomalous telemetry records for dgx-aws-03")
display(
    df_anomalous.select(
        "gpu_id",
        "cluster_id",
        "temperature_celsius",
        "gpu_utilization_pct",
        "power_draw_watts",
    )
    .orderBy("gpu_id")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Update Feature Table
# MAGIC
# MAGIC Talking points: In production this runs as a scheduled Lakeflow Declarative Pipeline. We aggregate raw telemetry into per-GPU rolling features (avg/max/stddev over 15-min windows) and publish the feature table to Unity Catalog for training and online serving.

# COMMAND ----------

feature_df = spark.sql(f"""
    SELECT
        gpu_id,
        cluster_id,
        gpu_model,
        COUNT(*)                                  AS reading_count,
        ROUND(AVG(temperature_celsius), 2)        AS avg_temperature,
        ROUND(MAX(temperature_celsius), 2)        AS max_temperature,
        ROUND(STDDEV(temperature_celsius), 2)     AS stddev_temperature,
        ROUND(AVG(gpu_utilization_pct), 2)        AS avg_utilization,
        ROUND(MAX(gpu_utilization_pct), 2)        AS max_utilization,
        ROUND(AVG(memory_utilization_pct), 2)     AS avg_memory_utilization,
        ROUND(AVG(power_draw_watts), 2)           AS avg_power_draw,
        ROUND(MAX(power_draw_watts), 2)           AS max_power_draw,
        SUM(ecc_errors_single_bit)                AS total_ecc_single_bit,
        SUM(ecc_errors_double_bit)                AS total_ecc_double_bit,
        ROUND(AVG(fan_speed_pct), 2)              AS avg_fan_speed,
        ROUND(AVG(pcie_throughput_gbps), 2)       AS avg_pcie_throughput,
        MIN(timestamp)                            AS window_start,
        MAX(timestamp)                            AS window_end,
        current_timestamp()                       AS computed_at
    FROM {CATALOG}.{SCHEMA}.gpu_telemetry
    WHERE timestamp >= current_timestamp() - INTERVAL 15 MINUTES
    GROUP BY gpu_id, cluster_id, gpu_model
""")

feature_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.gpu_health_features"
)

print("Feature table updated:")
display(
    feature_df.filter(F.col("cluster_id") == "dgx-aws-03")
    .select(
        "gpu_id",
        "cluster_id",
        "avg_temperature",
        "max_temperature",
        "avg_utilization",
        "max_utilization",
        "avg_power_draw",
        "reading_count",
    )
    .orderBy("gpu_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Score with Model Serving Endpoint
# MAGIC
# MAGIC Talking points: The `gpu-anomaly-detector` endpoint hosts an MLflow model returning an anomaly score (0-1) and a boolean flag. Model Serving provides sub-100ms latency with autoscaling. Watch for `gpu-a100-017` through `gpu-a100-021` to be flagged.

# COMMAND ----------

# Collect features for the anomalous GPUs
scoring_df = (
    feature_df.filter(F.col("cluster_id") == "dgx-aws-03")
    .select(
        "gpu_id",
        "cluster_id",
        "avg_temperature",
        "max_temperature",
        "stddev_temperature",
        "avg_utilization",
        "max_utilization",
        "avg_memory_utilization",
        "avg_power_draw",
        "max_power_draw",
        "total_ecc_single_bit",
        "total_ecc_double_bit",
        "avg_fan_speed",
        "avg_pcie_throughput",
    )
    .toPandas()
)

records = scoring_df.drop(columns=["gpu_id", "cluster_id"]).to_dict(orient="records")
payload = {"dataframe_records": records}

url = f"https://{DATABRICKS_HOST}/serving-endpoints/{MODEL_SERVING_ENDPOINT}/invocations"

print(f"Scoring {len(records)} GPUs against endpoint: {MODEL_SERVING_ENDPOINT}")
print(f"POST {url}\n")

response = requests.post(url, headers=HEADERS, json=payload, timeout=30)

if response.status_code == 200:
    predictions = response.json().get("predictions", [])
    for idx, pred in enumerate(predictions):
        gpu_id = scoring_df.iloc[idx]["gpu_id"]
        cluster = scoring_df.iloc[idx]["cluster_id"]
        score = pred.get("anomaly_score", pred)
        is_anomaly = pred.get("is_anomaly", score > 0.7 if isinstance(score, (int, float)) else None)
        status = "ANOMALY DETECTED" if is_anomaly else "Normal"
        print(f"  {gpu_id} ({cluster}): score={score:.3f} --> {status}")
else:
    print(f"Endpoint returned {response.status_code}: {response.text}")
    print("If the endpoint is not deployed, this is expected in a workshop setting.")
    print("The anomaly detection logic would flag these GPUs based on the elevated readings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Query Genie Space -- Natural Language Analytics
# MAGIC
# MAGIC Talking points: Genie Spaces let non-technical users ask questions in plain English. Behind the scenes, Genie generates SQL against Unity Catalog tables. This is how an ops lead checks fleet status without writing SQL.

# COMMAND ----------

def query_genie(space_id, question, max_wait_seconds=60):
    """Query a Genie Space and poll for results."""
    base_url = f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}"

    start_resp = requests.post(
        f"{base_url}/conversations",
        headers=HEADERS,
        json={"content": question},
        timeout=30,
    )
    start_resp.raise_for_status()
    conversation = start_resp.json()
    conversation_id = conversation["conversation_id"]
    message_id = conversation["message_id"]

    poll_url = f"{base_url}/conversations/{conversation_id}/messages/{message_id}"
    for _ in range(max_wait_seconds // 3):
        time.sleep(3)
        poll_resp = requests.get(poll_url, headers=HEADERS, timeout=30)
        poll_resp.raise_for_status()
        msg = poll_resp.json()
        status = msg.get("status", "")
        if status in ("COMPLETED", "COMPLETE"):
            return msg
        elif status in ("FAILED", "ERROR"):
            return msg
    return {"status": "TIMEOUT", "message": "Genie did not respond in time."}


genie_question = (
    "How many GPUs on cluster dgx-aws-03 are showing anomalies? "
    "What is the average temperature and utilization for those GPUs?"
)

print(f"Asking Genie: \"{genie_question}\"\n")

genie_result = query_genie(GENIE_SPACE_ID, genie_question)

if genie_result.get("status") in ("COMPLETED", "COMPLETE"):
    for attachment in genie_result.get("attachments", []):
        if "query" in attachment:
            print("Generated SQL:")
            print(attachment["query"].get("query", ""))
            print()
        if "text" in attachment:
            print("Genie Answer:")
            print(attachment["text"].get("content", ""))
else:
    print(f"Genie status: {genie_result.get('status', 'UNKNOWN')}")
    print(json.dumps(genie_result, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Query Knowledge Assistant -- Runbook Retrieval
# MAGIC
# MAGIC Talking points: The Knowledge Assistant uses RAG backed by a Vector Search index over GPU operations runbooks. It finds relevant procedures and summarizes them -- replacing the 2 AM hunt through Confluence for the right runbook.

# COMMAND ----------

def query_knowledge_assistant(ka_name, question):
    """Query a Knowledge Assistant (Compound AI agent)."""
    url = f"https://{DATABRICKS_HOST}/serving-endpoints/{ka_name}/invocations"
    payload = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


ka_question = (
    "What is the runbook procedure for thermal throttling on A100 GPUs? "
    "Include steps for immediate mitigation and escalation criteria."
)

print(f"Asking Knowledge Assistant ({KA_NAME}):")
print(f"\"{ka_question}\"\n")

ka_result = query_knowledge_assistant(KA_NAME, ka_question)

if "choices" in ka_result:
    answer = ka_result["choices"][0]["message"]["content"]
    print("Knowledge Assistant Response:")
    print(answer)
elif "output" in ka_result:
    print("Knowledge Assistant Response:")
    print(ka_result["output"])
else:
    print(json.dumps(ka_result, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Query Supervisor Agent -- Orchestrated Remediation
# MAGIC
# MAGIC Talking points: The Supervisor Agent (Multi-Agent System) coordinates the Genie Agent for analytics, the Knowledge Assistant for runbooks, and custom tools for cluster capacity and job migration. It synthesizes everything into one actionable recommendation -- the single pane of glass for GPU fleet ops.

# COMMAND ----------

def query_supervisor_agent(agent_name, question):
    """Query the Supervisor (Multi-Agent System) endpoint."""
    url = f"https://{DATABRICKS_HOST}/serving-endpoints/{agent_name}/invocations"
    payload = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


supervisor_question = (
    "5 GPUs are showing thermal anomalies on cluster dgx-aws-03. "
    "The anomaly detector flagged gpu-a100-017 through gpu-a100-021 with scores above 0.85. "
    "Average temperature is 93C and utilization is 97%. "
    "Runbook says: reduce workload, check cooling system, escalate if temps exceed 95C. "
    "Current utilization on dgx-aws-03 is 97% -- recommend migrating 2 jobs to dgx-gcp-01 "
    "which is at 42% utilization. "
    "What is the recommended action plan?"
)

print(f"Asking Supervisor Agent ({SUPERVISOR_AGENT_NAME}):")
print(f"\"{supervisor_question}\"\n")

supervisor_result = query_supervisor_agent(SUPERVISOR_AGENT_NAME, supervisor_question)

if "choices" in supervisor_result:
    answer = supervisor_result["choices"][0]["message"]["content"]
    print("Supervisor Agent Recommendation:")
    print(answer)
elif "output" in supervisor_result:
    print("Supervisor Agent Recommendation:")
    print(supervisor_result["output"])
else:
    print(json.dumps(supervisor_result, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Monitoring -- Drift Detection
# MAGIC
# MAGIC Talking points: Lakehouse Monitoring tracks statistical drift in the feature table. The temperature distribution is shifting higher -- a leading indicator. Profile metrics are stored historically, and alerts fire in production when drift exceeds a threshold.

# COMMAND ----------

drift_df = spark.sql(f"""
    SELECT
        column_name,
        window_start,
        window_end,
        ROUND(mean, 2)            AS mean_value,
        ROUND(stddev, 2)          AS stddev_value,
        ROUND(min, 2)             AS min_value,
        ROUND(max, 2)             AS max_value,
        ROUND(percent_change, 2)  AS pct_change_from_baseline
    FROM {CATALOG}.{SCHEMA}.gpu_health_features_profile_metrics
    WHERE column_name IN ('avg_temperature', 'max_temperature', 'avg_utilization', 'avg_power_draw')
    ORDER BY window_end DESC, column_name
    LIMIT 20
""")

print("Drift Metrics -- Feature Distribution Changes:")
display(drift_df)

# COMMAND ----------

# Temperature distribution: baseline vs current window
print("Temperature Distribution: Baseline vs Current Window")

temp_comparison = spark.sql(f"""
    SELECT
        'baseline'  AS period,
        ROUND(AVG(avg_temperature), 2)    AS mean_temp,
        ROUND(STDDEV(avg_temperature), 2) AS stddev_temp,
        ROUND(MAX(max_temperature), 2)    AS peak_temp,
        COUNT(*)                          AS gpu_count
    FROM {CATALOG}.{SCHEMA}.gpu_health_features
    WHERE cluster_id != 'dgx-aws-03'

    UNION ALL

    SELECT
        'current'   AS period,
        ROUND(AVG(avg_temperature), 2)    AS mean_temp,
        ROUND(STDDEV(avg_temperature), 2) AS stddev_temp,
        ROUND(MAX(max_temperature), 2)    AS peak_temp,
        COUNT(*)                          AS gpu_count
    FROM {CATALOG}.{SCHEMA}.gpu_health_features
    WHERE cluster_id = 'dgx-aws-03'
""")

display(temp_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Automated Retraining Trigger (Concept)
# MAGIC
# MAGIC Talking points: When drift exceeds a threshold, a Databricks Workflow Job fires automatically. The pipeline pulls latest labeled data, trains a new model version with MLflow, registers it in Unity Catalog, validates against a holdout set, and promotes to Champion. Model Serving auto-deploys the new version. This closes the loop: drift, detection, retraining, deployment -- all automated.
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC Lakehouse Monitor (drift detected)
# MAGIC         |
# MAGIC         v
# MAGIC Databricks Workflow Job (triggered via webhook/alert)
# MAGIC         |
# MAGIC         +---> Pull latest features from gpu_health_features
# MAGIC         +---> Train new model (XGBoost / PyTorch on DGX)
# MAGIC         +---> Log to MLflow, register in Unity Catalog
# MAGIC         +---> Validate against holdout set
# MAGIC         +---> Promote to Champion alias
# MAGIC         |
# MAGIC         v
# MAGIC Model Serving Endpoint (auto-deploys new Champion)
# MAGIC ```
# MAGIC
# MAGIC ### Example DAB Config
# MAGIC
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     gpu_anomaly_retrain:
# MAGIC       name: "gpu-anomaly-retrain"
# MAGIC       triggers:
# MAGIC         - type: webhook
# MAGIC           condition: "drift_score > 0.15"
# MAGIC       tasks:
# MAGIC         - task_key: retrain
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/04_model_training
# MAGIC           existing_cluster_id: ${var.dgx_cluster_id}
# MAGIC ```
# MAGIC
# MAGIC In production, the Lakehouse Monitor alert fires a webhook that triggers this job. The workshop showed the manual flow; the automation is identical.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Customer Use Cases
# MAGIC
# MAGIC The pipeline above is a template for any domain combining real-time ingestion, ML-powered detection, natural language analytics, knowledge retrieval, agent orchestration, and continuous monitoring. Below are five industry examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 1: Telecom -- Network Anomaly Detection
# MAGIC
# MAGIC Detect network degradation across thousands of cell towers before customers are impacted. RAN telemetry (signal strength, handover rates, packet loss) streams at 500K events/sec through Auto Loader. An Isolation Forest + LSTM ensemble trained on DGX scores towers in real time, while a Supervisor Agent correlates anomalies with weather data to recommend load balancing or field dispatch.
# MAGIC
# MAGIC | Pipeline Component | Telecom Mapping |
# MAGIC |---|---|
# MAGIC | Feature Engineering | Rolling 5-min per-tower aggregations (avg signal, handover failures) |
# MAGIC | Genie Space | NOC analysts: "Which Dallas towers degraded in the last hour?" |
# MAGIC | Knowledge Assistant | RAG over network ops runbooks for escalation paths |
# MAGIC | Monitoring | Drift on signal distributions; seasonal retrain triggers |
# MAGIC | Impact | 40% MTTD reduction, 25% fewer customer-impacting outages |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 2: Financial Services -- Due Diligence Automation
# MAGIC
# MAGIC Accelerate M&A due diligence by automating document review and risk identification. SEC filings, contracts, and financial statements are batch-ingested and parsed with Docling/LlamaParse. A fine-tuned Llama 3.1 70B on DGX classifies contract clauses and scores risk, while a Supervisor Agent aggregates financial analysis, legal flags, and market sentiment into a draft deal memo.
# MAGIC
# MAGIC | Pipeline Component | FinServ Mapping |
# MAGIC |---|---|
# MAGIC | Feature Engineering | Entity extraction, sentiment scoring, financial ratio computation |
# MAGIC | Genie Space | Analysts: "Revenue trend for TargetCo over the last 8 quarters?" |
# MAGIC | Knowledge Assistant | RAG over M&A playbooks for red flags and working capital criteria |
# MAGIC | Monitoring | Extraction accuracy tracking; retrain on new document formats |
# MAGIC | Impact | Due diligence cycle from 6 weeks to 10 days |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 3: Manufacturing -- Predictive Maintenance
# MAGIC
# MAGIC Predict equipment failures on production lines before unplanned downtime. IoT sensor data (vibration, temperature, pressure) streams from 100K sensors every 5 seconds. Survival analysis and gradient-boosted trees estimate remaining useful life per machine, while a Supervisor Agent cross-references predictions with parts inventory and technician schedules to generate optimized maintenance plans.
# MAGIC
# MAGIC | Pipeline Component | Manufacturing Mapping |
# MAGIC |---|---|
# MAGIC | Feature Engineering | Time-domain and frequency-domain features (FFT on vibration), rolling failure indicators |
# MAGIC | Genie Space | Plant managers: "Which Line 3 machines fail within 48 hours?" |
# MAGIC | Knowledge Assistant | RAG over maintenance SOPs for bearing replacement procedures |
# MAGIC | Monitoring | Drift on vibration distributions; retrain when new machine types added |
# MAGIC | Impact | 35% reduction in unplanned downtime, 20% lower maintenance costs |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 4: Healthcare -- Clinical Document Analysis
# MAGIC
# MAGIC Extract structured insights from unstructured clinical notes to improve care coordination and coding accuracy. EHR notes, discharge summaries, and radiology reports stream via HL7/FHIR through Lakeflow Connect. A fine-tuned BioMistral on DGX performs clinical NLP and ICD-10 code suggestion, while a Supervisor Agent correlates extracted diagnoses with coding suggestions and flags discrepancies for human review.
# MAGIC
# MAGIC | Pipeline Component | Healthcare Mapping |
# MAGIC |---|---|
# MAGIC | Feature Engineering | NER for conditions/medications, negation detection, temporal relation extraction |
# MAGIC | Genie Space | Clinical ops: "Patients discharged last week with primary CHF diagnosis?" |
# MAGIC | Knowledge Assistant | RAG over clinical guidelines for CMS admission criteria |
# MAGIC | Monitoring | Extraction F1 scores; retrain on new terminology or guidelines |
# MAGIC | Impact | 50% less manual chart review, 15% coding accuracy improvement |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 5: Retail -- Demand Forecasting + LLM Commentary
# MAGIC
# MAGIC Improve demand forecast accuracy and generate executive-ready commentary on forecast drivers. POS transactions stream in real time alongside daily batch loads for weather, social sentiment, and competitor pricing. A Temporal Fusion Transformer on DGX produces SKU-store-day forecasts, while a Supervisor Agent combines outputs with inventory positions and the promotional calendar to generate weekly buy recommendations with LLM narrative.
# MAGIC
# MAGIC | Pipeline Component | Retail Mapping |
# MAGIC |---|---|
# MAGIC | Feature Engineering | Lag features, promotional lift, holiday indicators, price elasticity per SKU-store |
# MAGIC | Genie Space | Merchandising: "Categories trending above forecast in Southeast this week?" |
# MAGIC | Knowledge Assistant | RAG over merchandising playbooks for markdown strategies |
# MAGIC | Monitoring | MAPE by category; retrain on new product lines or demand shifts |
# MAGIC | Impact | 20% overstock reduction, 12% forecast accuracy gain |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Recap and Resources
# MAGIC
# MAGIC ## What We Built
# MAGIC
# MAGIC | Step | Component | Databricks Service |
# MAGIC |------|-----------|-------------------|
# MAGIC | 1 | Data Ingestion | Auto Loader, Delta Lake |
# MAGIC | 2 | Feature Engineering | DLT / SQL, Feature Table |
# MAGIC | 3 | Anomaly Detection | Model Serving, MLflow, Unity Catalog |
# MAGIC | 4 | Natural Language Analytics | Genie Space |
# MAGIC | 5 | Knowledge Retrieval | Knowledge Assistant, Vector Search |
# MAGIC | 6 | Agent Orchestration | Supervisor Agent (Multi-Agent System) |
# MAGIC | 7-8 | Monitoring and Retraining | Lakehouse Monitor, Drift Detection |
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC | Resource | Link |
# MAGIC |----------|------|
# MAGIC | Mosaic AI Agent Framework | https://docs.databricks.com/en/generative-ai/agent-framework/index.html |
# MAGIC | Lakehouse Monitoring | https://docs.databricks.com/en/lakehouse-monitoring/index.html |
# MAGIC | Model Serving | https://docs.databricks.com/en/machine-learning/model-serving/index.html |
# MAGIC | NVIDIA DGX Cloud on Databricks | https://www.databricks.com/partners/nvidia |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Thank you for attending the NVIDIA DGX Cloud MLOps and GenAI Workshop.**
# MAGIC
# MAGIC Questions? Reach out to your Databricks and NVIDIA account teams.
