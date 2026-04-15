# Databricks notebook source

# MAGIC %md
# MAGIC # Block 7: End-to-End GPU Fleet Operations Pipeline
# MAGIC ### NVIDIA DGX Cloud MLOps & GenAI Workshop
# MAGIC
# MAGIC **Duration:** 15 minutes (live demo)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **What you'll see in this demo:**
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
# MAGIC > **Instructor note:** Run each cell top-to-bottom. Pause at each section header to explain what's happening before executing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

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

# ---------------------------------------------------------------------------
# Placeholder variables -- update these before the demo
# ---------------------------------------------------------------------------
CATALOG = "nvidia_dgx_workshop"
SCHEMA = "gpu_ops"

GENIE_SPACE_ID = "<YOUR_GENIE_SPACE_ID>"          # e.g. "01f0abcd..."
KA_NAME = "<YOUR_KNOWLEDGE_ASSISTANT_NAME>"        # e.g. "gpu-runbook-assistant"
SUPERVISOR_AGENT_NAME = "<YOUR_SUPERVISOR_AGENT>"  # e.g. "gpu-fleet-supervisor"
MODEL_SERVING_ENDPOINT = "gpu-anomaly-detector"

# Databricks host & token (auto-populated in notebooks)
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
# MAGIC ---
# MAGIC ## Step 1: Simulate GPU Telemetry Spike
# MAGIC
# MAGIC **Talking points:**
# MAGIC - In production, telemetry streams in via Kafka / Kinesis through Auto Loader.
# MAGIC - Here we inject **anomalous readings** for cluster `dgx-aws-03` to trigger the full pipeline.
# MAGIC - Notice the temperatures above 90C and utilization near 100% -- these will be flagged as anomalies.

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
# MAGIC ---
# MAGIC ## Step 2: Update Feature Table
# MAGIC
# MAGIC **Talking points:**
# MAGIC - In production this runs as a scheduled Lakeflow Declarative Pipeline (DLT).
# MAGIC - We aggregate the raw telemetry into per-GPU rolling features: avg/max/stddev over 15-min windows.
# MAGIC - The feature table is published to Unity Catalog for both training and online serving.

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
# MAGIC ---
# MAGIC ## Step 3: Score with Model Serving Endpoint
# MAGIC
# MAGIC **Talking points:**
# MAGIC - The `gpu-anomaly-detector` endpoint hosts an MLflow model trained on historical GPU telemetry.
# MAGIC - It returns an anomaly score (0-1) and a boolean flag.
# MAGIC - Model Serving gives us sub-100ms latency with autoscaling -- critical for real-time fleet ops.
# MAGIC - Watch for `gpu-a100-017` through `gpu-a100-021` on `dgx-aws-03` to show as anomalies.

# COMMAND ----------

# Collect features for the GPUs we just inserted
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

# Build the payload
records = scoring_df.drop(columns=["gpu_id", "cluster_id"]).to_dict(orient="records")
payload = {"dataframe_records": records}

# Call the Model Serving endpoint
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
# MAGIC ---
# MAGIC ## Step 4: Query Genie Space -- Natural Language Analytics
# MAGIC
# MAGIC **Talking points:**
# MAGIC - Genie Spaces let non-technical users ask questions in plain English.
# MAGIC - Behind the scenes, Genie generates SQL against our Unity Catalog tables.
# MAGIC - This is how an ops team lead would check on fleet status without writing SQL.

# COMMAND ----------

def query_genie(space_id, question, max_wait_seconds=60):
    """Query a Genie Space and poll for results."""
    base_url = f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}"

    # Start conversation
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

    # Poll for result
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

# Display the Genie response
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
# MAGIC ---
# MAGIC ## Step 5: Query Knowledge Assistant -- Runbook Retrieval
# MAGIC
# MAGIC **Talking points:**
# MAGIC - The Knowledge Assistant is backed by a Vector Search index over our GPU operations runbooks.
# MAGIC - It uses RAG to find the most relevant procedures and summarize them.
# MAGIC - This replaces hunting through Confluence/SharePoint for the right runbook at 2 AM.

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

# Extract and display the response
if "choices" in ka_result:
    answer = ka_result["choices"][0]["message"]["content"]
    print("Knowledge Assistant Response:")
    print("-" * 60)
    print(answer)
elif "output" in ka_result:
    print("Knowledge Assistant Response:")
    print("-" * 60)
    print(ka_result["output"])
else:
    print(json.dumps(ka_result, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Query Supervisor Agent -- Orchestrated Remediation
# MAGIC
# MAGIC **Talking points:**
# MAGIC - The Supervisor Agent (Multi-Agent System) coordinates:
# MAGIC   - **Genie Agent** for real-time fleet analytics
# MAGIC   - **Knowledge Assistant** for runbook procedures
# MAGIC   - **Custom tools** for cluster capacity lookups and job migration
# MAGIC - It synthesizes all the information into a single actionable recommendation.
# MAGIC - This is the "single pane of glass" for GPU fleet operations.

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

# Extract and display the response
if "choices" in supervisor_result:
    answer = supervisor_result["choices"][0]["message"]["content"]
    print("Supervisor Agent Recommendation:")
    print("=" * 60)
    print(answer)
elif "output" in supervisor_result:
    print("Supervisor Agent Recommendation:")
    print("=" * 60)
    print(supervisor_result["output"])
else:
    print(json.dumps(supervisor_result, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: Monitoring -- Drift Detection
# MAGIC
# MAGIC **Talking points:**
# MAGIC - Lakehouse Monitoring tracks statistical drift in our feature table.
# MAGIC - We can see the temperature distribution shifting higher -- a leading indicator.
# MAGIC - The profile metrics table stores historical statistics for every monitored column.
# MAGIC - In production, alerts fire when drift exceeds a threshold.

# COMMAND ----------

# Query the Lakehouse Monitor drift metrics for the feature table
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

# Show temperature distribution comparison
print("Temperature Distribution: Baseline vs Current Window")
print("-" * 55)

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
# MAGIC ---
# MAGIC ## Step 8: Automated Retraining Trigger (Concept)
# MAGIC
# MAGIC **Talking points:**
# MAGIC - When drift exceeds a threshold, a Databricks **Workflow Job** is triggered automatically.
# MAGIC - The retraining pipeline:
# MAGIC   1. Pulls the latest labeled data from the feature table
# MAGIC   2. Trains a new model version using MLflow Experiment Tracking
# MAGIC   3. Registers the model in **Unity Catalog** as a new version
# MAGIC   4. Runs validation against a holdout set
# MAGIC   5. Promotes to the `Champion` alias if validation passes (via Model Registry)
# MAGIC   6. The Model Serving endpoint auto-deploys the new champion
# MAGIC - This closes the loop: **data drift --> detection --> retraining --> deployment** -- all automated.
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
# MAGIC ### Key Config (Databricks Asset Bundle)
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
# MAGIC > **In production**, the alert from Lakehouse Monitor fires a webhook that triggers this job.
# MAGIC > For this workshop, we showed the manual flow -- but the automation is identical.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ---
# MAGIC # Customer Use Cases: Where This Pattern Applies
# MAGIC
# MAGIC The GPU Fleet Operations pipeline we just demonstrated is a **template** for any domain where you need:
# MAGIC - Real-time data ingestion and feature engineering
# MAGIC - ML-powered anomaly detection or prediction
# MAGIC - Natural language analytics (Genie)
# MAGIC - Knowledge retrieval (RAG / Knowledge Assistant)
# MAGIC - Orchestrated decision-making (Supervisor Agent)
# MAGIC - Continuous monitoring and automated retraining
# MAGIC
# MAGIC Below are five industry examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 1: Telecom -- Network Anomaly Detection
# MAGIC
# MAGIC | Aspect | Details |
# MAGIC |--------|---------|
# MAGIC | **Problem** | Detect network degradation across thousands of cell towers before customers are impacted |
# MAGIC | **Data** | RAN telemetry (signal strength, handover rates, packet loss), CDRs, weather data |
# MAGIC | **Ingestion** | Auto Loader streaming from Kafka topics, 500K events/sec |
# MAGIC | **Feature Engineering** | Rolling 5-min aggregations per tower: avg signal, anomaly rate, handover failures |
# MAGIC | **Model** | Isolation Forest + LSTM ensemble trained on DGX, served via Model Serving |
# MAGIC | **Genie Space** | NOC analysts ask: "Which towers in the Dallas metro have degraded signal in the last hour?" |
# MAGIC | **Knowledge Assistant** | RAG over network ops runbooks: "What is the escalation path for RAN congestion?" |
# MAGIC | **Supervisor Agent** | Correlates tower anomalies with weather data, recommends load balancing or dispatches field crew |
# MAGIC | **Monitoring** | Drift detection on signal distributions; auto-retrain when seasonal patterns shift |
# MAGIC | **Business Impact** | 40% reduction in mean time to detect (MTTD), 25% fewer customer-impacting outages |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 2: Financial Services -- Due Diligence Automation
# MAGIC
# MAGIC | Aspect | Details |
# MAGIC |--------|---------|
# MAGIC | **Problem** | Accelerate M&A due diligence by automating document review and risk identification |
# MAGIC | **Data** | SEC filings, financial statements, legal contracts, news feeds, internal CRM notes |
# MAGIC | **Ingestion** | Batch ingestion of PDFs via Auto Loader + unstructured data parsing (Docling / LlamaParse) |
# MAGIC | **Feature Engineering** | Entity extraction, sentiment scoring, financial ratio computation across 10K/10Q filings |
# MAGIC | **Model** | Fine-tuned LLM (Llama 3.1 70B on DGX) for contract clause classification and risk scoring |
# MAGIC | **Genie Space** | Analysts ask: "What is the revenue trend for TargetCo over the last 8 quarters?" |
# MAGIC | **Knowledge Assistant** | RAG over internal M&A playbooks: "What are the red flags for working capital adjustments?" |
# MAGIC | **Supervisor Agent** | Aggregates financial analysis, legal risk flags, and market sentiment into a deal memo draft |
# MAGIC | **Monitoring** | Track extraction accuracy; retrain when new document formats appear |
# MAGIC | **Business Impact** | Due diligence cycle reduced from 6 weeks to 10 days; analysts review AI-generated summaries instead of raw docs |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 3: Manufacturing -- Predictive Maintenance
# MAGIC
# MAGIC | Aspect | Details |
# MAGIC |--------|---------|
# MAGIC | **Problem** | Predict equipment failures on production lines before unplanned downtime occurs |
# MAGIC | **Data** | IoT sensor data (vibration, temperature, pressure, RPM), maintenance logs, ERP work orders |
# MAGIC | **Ingestion** | Streaming from IoT Edge gateways via Event Hubs / Kinesis, 100K sensors reporting every 5 sec |
# MAGIC | **Feature Engineering** | Time-domain and frequency-domain features (FFT on vibration signals), rolling failure indicators |
# MAGIC | **Model** | Survival analysis + gradient-boosted trees for remaining useful life (RUL) prediction |
# MAGIC | **Genie Space** | Plant managers ask: "Which machines on Line 3 are predicted to fail in the next 48 hours?" |
# MAGIC | **Knowledge Assistant** | RAG over maintenance SOPs: "What is the procedure for bearing replacement on CNC Mill Model X?" |
# MAGIC | **Supervisor Agent** | Cross-references RUL predictions with parts inventory and technician schedules, generates optimized maintenance plan |
# MAGIC | **Monitoring** | Drift on vibration frequency distributions; retrain when new machine types are added to the line |
# MAGIC | **Business Impact** | 35% reduction in unplanned downtime, 20% decrease in maintenance costs through optimized scheduling |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 4: Healthcare -- Clinical Document Analysis
# MAGIC
# MAGIC | Aspect | Details |
# MAGIC |--------|---------|
# MAGIC | **Problem** | Extract structured insights from unstructured clinical notes to improve care coordination and coding accuracy |
# MAGIC | **Data** | EHR clinical notes, discharge summaries, radiology reports, pathology reports, ICD/CPT code mappings |
# MAGIC | **Ingestion** | HL7/FHIR event streams via Lakeflow Connect, batch ingestion of historical notes |
# MAGIC | **Feature Engineering** | NER for conditions, medications, procedures; negation detection; temporal relation extraction |
# MAGIC | **Model** | Fine-tuned biomedical LLM (BioMistral on DGX) for clinical NLP; ICD-10 code suggestion model |
# MAGIC | **Genie Space** | Clinical ops team asks: "How many patients discharged last week had a primary diagnosis of CHF?" |
# MAGIC | **Knowledge Assistant** | RAG over clinical guidelines: "What are the CMS criteria for inpatient admission for pneumonia?" |
# MAGIC | **Supervisor Agent** | Correlates extracted diagnoses with coding suggestions, flags discrepancies for human review |
# MAGIC | **Monitoring** | Track extraction F1 scores; retrain when new clinical terminology or guidelines are published |
# MAGIC | **Business Impact** | 50% reduction in manual chart review time, 15% improvement in coding accuracy, faster prior auth |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case 5: Retail -- Demand Forecasting + LLM Commentary
# MAGIC
# MAGIC | Aspect | Details |
# MAGIC |--------|---------|
# MAGIC | **Problem** | Improve demand forecasting accuracy and generate executive-ready commentary on forecast drivers |
# MAGIC | **Data** | POS transactions, inventory levels, promotions calendar, weather, social media sentiment, competitor pricing |
# MAGIC | **Ingestion** | Real-time POS streams via Kafka, daily batch loads for external signals (weather, social) |
# MAGIC | **Feature Engineering** | Lag features, promotional lift factors, holiday indicators, price elasticity estimates per SKU-store |
# MAGIC | **Model** | Hierarchical time-series model (Temporal Fusion Transformer on DGX) for SKU-store-day forecasts |
# MAGIC | **Genie Space** | Merchandising team asks: "Which categories are trending above forecast in the Southeast region this week?" |
# MAGIC | **Knowledge Assistant** | RAG over merchandising playbooks: "What is the markdown strategy for seasonal overstock?" |
# MAGIC | **Supervisor Agent** | Combines forecast outputs with inventory positions and promotional calendar, generates weekly buy recommendation with LLM narrative |
# MAGIC | **Monitoring** | Forecast accuracy (MAPE) by category; retrain when new product lines launch or demand patterns shift |
# MAGIC | **Business Impact** | 20% reduction in overstock waste, 12% improvement in forecast accuracy, automated weekly executive summaries |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ---
# MAGIC # Recap & Resources
# MAGIC
# MAGIC ## What We Built Today
# MAGIC
# MAGIC ```
# MAGIC Raw Telemetry --> Feature Engineering --> Anomaly Detection --> Analytics & Retrieval --> Agent Orchestration --> Monitoring & Retraining
# MAGIC      |                  |                      |                      |                         |                       |
# MAGIC  Auto Loader      DLT / SQL             Model Serving          Genie + KA              Supervisor Agent        Lakehouse Monitor
# MAGIC  Delta Lake       Feature Table          MLflow + UC            Vector Search           Multi-Agent System      Drift Detection
# MAGIC ```
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 1. **Unified Platform** -- Data, ML, and GenAI all on one lakehouse. No data copies, no integration tax.
# MAGIC 2. **GPU-Native** -- DGX Cloud clusters for training, Model Serving for inference. Use GPUs where they matter.
# MAGIC 3. **Governed by Default** -- Unity Catalog for data, features, models, and agents. One lineage graph.
# MAGIC 4. **Natural Language First** -- Genie Spaces and Knowledge Assistants make insights accessible to everyone.
# MAGIC 5. **Agents, Not Just Models** -- Supervisor Agents orchestrate across tools and knowledge sources for real decisions.
# MAGIC 6. **Closed-Loop Operations** -- Monitoring detects drift, triggers retraining, deploys new models automatically.
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC | Resource | Link |
# MAGIC |----------|------|
# MAGIC | Databricks Documentation | https://docs.databricks.com |
# MAGIC | MLflow Documentation | https://mlflow.org/docs/latest |
# MAGIC | NVIDIA DGX Cloud on Databricks | https://www.databricks.com/partners/nvidia |
# MAGIC | Mosaic AI Agent Framework | https://docs.databricks.com/en/generative-ai/agent-framework/index.html |
# MAGIC | Genie Spaces | https://docs.databricks.com/en/genie/index.html |
# MAGIC | Lakehouse Monitoring | https://docs.databricks.com/en/lakehouse-monitoring/index.html |
# MAGIC | Model Serving | https://docs.databricks.com/en/machine-learning/model-serving/index.html |
# MAGIC | Unity Catalog | https://docs.databricks.com/en/data-governance/unity-catalog/index.html |
# MAGIC
# MAGIC ## Workshop Notebooks
# MAGIC
# MAGIC | # | Notebook | Topic |
# MAGIC |---|----------|-------|
# MAGIC | 01 | Platform Setup | Catalog, schema, cluster config |
# MAGIC | 02 | Data Ingestion | Auto Loader, streaming, Delta Lake |
# MAGIC | 03 | Feature Engineering | DLT pipelines, feature tables |
# MAGIC | 04 | Model Training | MLflow, DGX training, Unity Catalog registry |
# MAGIC | 05 | GenAI Agents | Genie, Knowledge Assistant, Supervisor Agent |
# MAGIC | **06** | **End-to-End Demo** | **This notebook -- the full pipeline in action** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Thank you for attending the NVIDIA DGX Cloud MLOps & GenAI Workshop!**
# MAGIC
# MAGIC Questions? Reach out to your Databricks and NVIDIA account teams.
