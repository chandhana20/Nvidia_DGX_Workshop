# Databricks notebook source
# MAGIC %md
# MAGIC # NVIDIA DGX Cloud -- MLOps & GenAI Workshop
# MAGIC
# MAGIC *Build an end-to-end GPU fleet monitoring pipeline on Databricks.*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **What you'll build in this notebook:**
# MAGIC - A Unity Catalog schema and volumes for all workshop assets
# MAGIC - Synthetic telemetry for a 30-cluster, multi-cloud DGX fleet (A100 / H100 / H200)
# MAGIC - Health events, ML job history, anomaly labels, and a joined training dataset
# MAGIC - Exploratory queries to validate the generated data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Create Schema and Volumes
# MAGIC
# MAGIC All workshop assets live under `main.mlops_genai_workshop`. Two volumes hold documentation and app code for later notebooks.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS main.mlops_genai_workshop;

# COMMAND ----------

# DBTITLE 1,Set active schema
# MAGIC %sql
# MAGIC USE SCHEMA main.mlops_genai_workshop;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS main.mlops_genai_workshop.docs;
# MAGIC CREATE VOLUME IF NOT EXISTS main.mlops_genai_workshop.app_code;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Generate Synthetic GPU Fleet Data
# MAGIC
# MAGIC All data uses deterministic seeds for reproducibility. No external libraries required.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Cluster Inventory (~30 rows)
# MAGIC
# MAGIC 30 clusters across AWS, Azure, GCP, and Oracle with mixed GPU types.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    TimestampType, BooleanType
)
import random
import hashlib
from datetime import datetime, timedelta

random.seed(42)

cloud_providers = ["AWS", "Azure", "GCP", "Oracle"]
regions_by_cloud = {
    "AWS": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
    "Azure": ["eastus", "westus2", "westeurope", "southeastasia"],
    "GCP": ["us-central1", "us-east4", "europe-west1", "asia-southeast1"],
    "Oracle": ["us-ashburn-1", "us-phoenix-1", "uk-london-1", "ap-tokyo-1"],
}
gpu_types = ["A100", "H100", "H200"]
gpu_counts = [8, 16, 32, 64]
statuses = ["active", "maintenance", "scaling"]

cluster_rows = []
for i in range(30):
    cloud = cloud_providers[i % len(cloud_providers)]
    region = regions_by_cloud[cloud][i % len(regions_by_cloud[cloud])]
    cloud_short = cloud.lower()
    region_short = region.replace("-", "").replace("_", "")[:6]
    cluster_id = f"dgx-{cloud_short}-{region_short}-{str(i + 1).zfill(2)}"
    gpu_type = gpu_types[i % len(gpu_types)]
    gpu_count = gpu_counts[i % len(gpu_counts)]
    r = random.random()
    status = "active" if r < 0.8 else ("maintenance" if r < 0.9 else "scaling")
    cluster_rows.append((cluster_id, cloud, region, gpu_type, gpu_count, status))

cluster_schema = StructType([
    StructField("cluster_id", StringType(), False),
    StructField("cloud_provider", StringType(), False),
    StructField("region", StringType(), False),
    StructField("gpu_type", StringType(), False),
    StructField("gpu_count", IntegerType(), False),
    StructField("status", StringType(), False),
])

df_clusters = spark.createDataFrame(cluster_rows, schema=cluster_schema)
df_clusters.write.mode("overwrite").saveAsTable("main.mlops_genai_workshop.cluster_inventory")
print(f"cluster_inventory: {df_clusters.count()} rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. GPU Telemetry (~50K rows)
# MAGIC
# MAGIC Point-in-time readings per GPU. Normal temps 65-85C; ~5% injected anomalies at 90-105C with elevated error counts.

# COMMAND ----------

import random
from datetime import datetime, timedelta

random.seed(42)

cluster_data = df_clusters.collect()

gpu_registry = []
for row in cluster_data:
    gpu_type_lower = row["gpu_type"].lower()
    for g in range(row["gpu_count"]):
        gpu_id = f"gpu-{gpu_type_lower}-{str(len(gpu_registry) + 1).zfill(4)}"
        gpu_registry.append((gpu_id, row["cluster_id"], row["gpu_type"]))

print(f"Total GPUs in fleet: {len(gpu_registry)}")

NUM_TELEMETRY = 50000
BASE_TIME = datetime(2025, 1, 1, 0, 0, 0)
MAX_HOURS = 30 * 24

telemetry_rows = []
rng = random.Random(42)

for i in range(NUM_TELEMETRY):
    gpu_id, cluster_id, gpu_type = gpu_registry[i % len(gpu_registry)]
    ts = BASE_TIME + timedelta(hours=rng.uniform(0, MAX_HOURS))
    is_anomaly = rng.random() < 0.05

    if is_anomaly:
        temp = round(rng.uniform(90.0, 105.0), 1)
        utilization = round(rng.uniform(95.0, 100.0), 1)
        memory_max = 80.0 if gpu_type == "A100" else (80.0 if gpu_type == "H100" else 80.0)
        memory_used = round(rng.uniform(memory_max * 0.9, memory_max), 1)
        power = round(rng.uniform(380.0, 450.0), 1)
        error_count = rng.choices([0, 1, 2, 3, 4, 5], weights=[10, 20, 30, 20, 15, 5])[0]
    else:
        temp = round(rng.uniform(65.0, 85.0), 1)
        utilization = round(rng.uniform(10.0, 95.0), 1)
        memory_max = 80.0
        memory_used = round(rng.uniform(5.0, memory_max * 0.85), 1)
        power = round(rng.uniform(200.0, 380.0), 1)
        error_count = rng.choices([0, 1, 2, 3, 4, 5], weights=[70, 15, 8, 4, 2, 1])[0]

    telemetry_rows.append((
        gpu_id, cluster_id, ts, temp, utilization, memory_used, power, error_count
    ))

telemetry_schema = StructType([
    StructField("gpu_id", StringType(), False),
    StructField("cluster_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("temp_celsius", DoubleType(), False),
    StructField("utilization_pct", DoubleType(), False),
    StructField("memory_used_gb", DoubleType(), False),
    StructField("power_watts", DoubleType(), False),
    StructField("error_count", IntegerType(), False),
])

df_telemetry = spark.createDataFrame(telemetry_rows, schema=telemetry_schema)
df_telemetry.write.mode("overwrite").saveAsTable("main.mlops_genai_workshop.gpu_telemetry")
print(f"gpu_telemetry: {df_telemetry.count()} rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. GPU Health Events (~2K rows)
# MAGIC
# MAGIC Realistic error events modeled on common DGX Cloud failure modes: NVLink errors, ECC faults, thermal throttling, Xid errors, etc.

# COMMAND ----------

rng = random.Random(42)

event_types = ["warning", "error", "critical"]
event_type_weights = [50, 35, 15]

gpu_error_descriptions = {
    "warning": [
        "GPU temperature approaching thermal limit",
        "NVLink bandwidth degradation detected on link 3",
        "ECC single-bit error corrected in HBM2e bank 7",
        "GPU clock throttled due to power cap",
        "PCIe link speed downgraded to Gen4 x8",
        "Memory utilization sustained above 90% for 15 minutes",
        "Fan speed increased to compensate for ambient temperature",
        "GPU driver timeout recovered successfully",
        "Intermittent NVSwitch port flap detected",
        "CUDA compute preemption latency above threshold",
    ],
    "error": [
        "Xid 79: GPU has fallen off the bus - attempting recovery",
        "ECC double-bit uncorrectable error in GPU SRAM",
        "NVLink fatal error: link 2 training failure",
        "GPU firmware reported unrecoverable SBE count exceeded",
        "InfiniBand HCA link down on mlx5_0 port 1",
        "CUDA out of memory during kernel launch",
        "PCIe AER: corrected error received, multiple downstream",
        "GPU power brake assertion - power delivery anomaly",
        "NVSwitch non-fatal error: trunk link CRC mismatch",
        "DMA engine timeout on copy H2D channel 4",
    ],
    "critical": [
        "Xid 94: Contained ECC error - GPU page retirement triggered",
        "NVLink fatal: all 12 links down on GPU - node isolation required",
        "GPU thermal emergency shutdown at 110C - immediate cooldown required",
        "Uncorrectable ECC error storm: 47 errors in 60 seconds - hardware RMA needed",
        "System fabric manager lost communication with GPU - baseboard reset required",
        "InfiniBand network partition detected - cluster interconnect severed",
        "GPU power delivery failure - VRM fault code 0x3F",
        "Xid 48: Double-bit ECC error in register file - GPU halted",
        "NVSwitch fatal: switch ASIC non-responsive after 3 reset attempts",
        "BMC reported chassis intrusion event - physical inspection required",
    ],
}

NUM_EVENTS = 2000
event_rows = []

for i in range(NUM_EVENTS):
    event_id = f"evt-{str(i + 1).zfill(5)}"
    gpu_id, cluster_id, _ = gpu_registry[rng.randint(0, len(gpu_registry) - 1)]
    event_type = rng.choices(event_types, weights=event_type_weights)[0]
    description = rng.choice(gpu_error_descriptions[event_type])
    ts = BASE_TIME + timedelta(hours=rng.uniform(0, MAX_HOURS))
    resolve_prob = {"warning": 0.9, "error": 0.6, "critical": 0.3}
    resolved = rng.random() < resolve_prob[event_type]

    event_rows.append((event_id, gpu_id, event_type, description, ts, resolved))

event_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("gpu_id", StringType(), False),
    StructField("event_type", StringType(), False),
    StructField("description", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("resolved", BooleanType(), False),
])

df_events = spark.createDataFrame(event_rows, schema=event_schema)
df_events.write.mode("overwrite").saveAsTable("main.mlops_genai_workshop.gpu_health_events")
print(f"gpu_health_events: {df_events.count()} rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. ML Job Runs (~5K rows)
# MAGIC
# MAGIC Training and inference jobs across PyTorch, TensorFlow, and JAX. Free-text descriptions are used for GenAI classification later.

# COMMAND ----------

rng = random.Random(42)

frameworks = ["PyTorch", "TensorFlow", "JAX"]
model_types = ["LLM", "Vision", "RL", "Tabular"]
job_statuses = ["completed", "running", "failed"]
job_status_weights = [70, 15, 15]

job_descriptions_by_type = {
    "LLM": [
        "Fine-tuning Llama-3 70B on proprietary customer support transcripts with LoRA rank 16",
        "Continued pre-training of Mistral-7B on domain-specific medical literature corpus",
        "RLHF alignment training for GPT-NeoX 20B with DPO loss on human preference data",
        "Distributed FSDP training of 13B parameter code generation model on The Stack v2",
        "Evaluation sweep of quantized LLM variants (GPTQ 4-bit vs AWQ) on MMLU benchmark",
        "Speculative decoding inference optimization for Falcon-180B serving pipeline",
        "Multi-node Megatron-LM training of 65B parameter model with tensor parallelism degree 8",
        "Knowledge distillation from Llama-3 405B teacher to 8B student model",
        "SFT training on instruction-following dataset with 500K curated examples",
        "Prefix-tuning experiment for domain adaptation of Phi-3 on financial documents",
    ],
    "Vision": [
        "Training DINO v2 ViT-L/14 on satellite imagery dataset for land use classification",
        "Fine-tuning Segment Anything Model (SAM) on industrial defect detection dataset",
        "Multi-GPU training of 3D medical image segmentation model (nnU-Net) on CT scans",
        "Object detection model (RT-DETR) training on autonomous driving perception dataset",
        "Video understanding model training with VideoMAE on Kinetics-700 action recognition",
        "Training Stable Diffusion XL LoRA on branded product image generation dataset",
        "Depth estimation model training (Depth Anything v2) on indoor scene dataset",
        "OCR model fine-tuning (TrOCR) on handwritten document recognition corpus",
        "Image classification ensemble training for manufacturing quality inspection",
        "Pose estimation model (ViTPose) training on multi-person sports tracking dataset",
    ],
    "RL": [
        "PPO training of robotic manipulation policy in Isaac Sim environment",
        "Multi-agent RL training for warehouse logistics optimization with 64 agents",
        "DreamerV3 world model training on Atari-100K benchmark suite",
        "MAPPO training for autonomous drone swarm coordination in simulated airspace",
        "Offline RL (Decision Transformer) training on historical trading data",
        "Reward model training for RLHF pipeline using human comparison judgments",
        "SAC continuous control policy training for industrial process optimization",
        "Curiosity-driven exploration experiment with RND on procedural game environments",
        "Multi-objective RL for chip placement optimization using circuit netlists",
        "Sim-to-real transfer policy training for quadruped locomotion on rough terrain",
    ],
    "Tabular": [
        "XGBoost hyperparameter sweep for customer churn prediction (10K trial Optuna search)",
        "LightGBM training on 500M row click-through rate prediction dataset",
        "TabNet deep learning model training on structured healthcare claims data",
        "Feature engineering pipeline + CatBoost training for fraud detection model",
        "AutoML benchmark comparison (H2O vs AutoGluon vs FLAML) on OpenML regression suite",
        "Gradient boosted tree ensemble for real-time bidding price prediction",
        "Time-series forecasting with temporal fusion transformer on energy demand data",
        "Multi-task learning model training for simultaneous LTV and conversion prediction",
        "Federated learning simulation for credit scoring across 12 institutional datasets",
        "Neural architecture search for tabular data using DARTS on census income dataset",
    ],
}

NUM_JOBS = 5000
job_rows = []
cluster_ids = [row["cluster_id"] for row in cluster_data]

for i in range(NUM_JOBS):
    job_id = f"job-{str(i + 1).zfill(5)}"
    cluster_id = rng.choice(cluster_ids)
    framework = rng.choice(frameworks)
    model_type = rng.choice(model_types)
    description = rng.choice(job_descriptions_by_type[model_type])
    start_time = BASE_TIME + timedelta(hours=rng.uniform(0, MAX_HOURS))

    duration_hours_map = {"LLM": (2, 72), "Vision": (1, 24), "RL": (4, 96), "Tabular": (0.5, 8)}
    min_h, max_h = duration_hours_map[model_type]
    duration_hours = round(rng.uniform(min_h, max_h), 2)
    end_time = start_time + timedelta(hours=duration_hours)

    status = rng.choices(job_statuses, weights=job_status_weights)[0]
    if status == "running":
        end_time = None
        gpu_hours = round(duration_hours * 0.5, 2)
    elif status == "failed":
        fail_fraction = rng.uniform(0.1, 0.7)
        end_time = start_time + timedelta(hours=duration_hours * fail_fraction)
        gpu_hours = round(duration_hours * fail_fraction, 2)
    else:
        gpu_hours = round(duration_hours, 2)

    cost_per_hour = rng.uniform(2.0, 4.5)
    cost_usd = round(gpu_hours * cost_per_hour, 2)

    job_rows.append((
        job_id, cluster_id, framework, model_type, description,
        start_time, end_time, gpu_hours, status, cost_usd
    ))

job_schema = StructType([
    StructField("job_id", StringType(), False),
    StructField("cluster_id", StringType(), False),
    StructField("framework", StringType(), False),
    StructField("model_type", StringType(), False),
    StructField("description", StringType(), False),
    StructField("start_time", TimestampType(), False),
    StructField("end_time", TimestampType(), True),
    StructField("gpu_hours", DoubleType(), False),
    StructField("status", StringType(), False),
    StructField("cost_usd", DoubleType(), False),
])

df_jobs = spark.createDataFrame(job_rows, schema=job_schema)
df_jobs.write.mode("overwrite").saveAsTable("main.mlops_genai_workshop.ml_job_runs")
print(f"ml_job_runs: {df_jobs.count()} rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2e. GPU Anomaly Labels (~5K rows)
# MAGIC
# MAGIC Binary labels for 1-hour GPU windows. ~5% positive rate to simulate realistic class imbalance.

# COMMAND ----------

rng = random.Random(42)

NUM_ANOMALY_LABELS = 5000
anomaly_rows = []
labeled_gpus = [gpu_registry[i] for i in range(min(200, len(gpu_registry)))]

for i in range(NUM_ANOMALY_LABELS):
    gpu_id, _, _ = labeled_gpus[i % len(labeled_gpus)]
    window_start = BASE_TIME + timedelta(hours=rng.uniform(0, MAX_HOURS))
    is_anomaly = 1 if rng.random() < 0.05 else 0
    anomaly_rows.append((gpu_id, window_start, is_anomaly))

anomaly_schema = StructType([
    StructField("gpu_id", StringType(), False),
    StructField("window_start", TimestampType(), False),
    StructField("is_anomaly", IntegerType(), False),
])

df_anomaly_labels = spark.createDataFrame(anomaly_rows, schema=anomaly_schema)
df_anomaly_labels.write.mode("overwrite").saveAsTable("main.mlops_genai_workshop.gpu_anomaly_labels")
print(f"gpu_anomaly_labels: {df_anomaly_labels.count()} rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2f. Training Dataset (Joined Features + Labels)
# MAGIC
# MAGIC Joins telemetry to anomaly labels by matching each label's 1-hour window, then aggregates features per GPU per window.

# COMMAND ----------

df_tel = spark.table("main.mlops_genai_workshop.gpu_telemetry")
df_labels = spark.table("main.mlops_genai_workshop.gpu_anomaly_labels")

df_labels_with_end = df_labels.withColumn("window_end", F.col("window_start") + F.expr("INTERVAL 1 HOUR"))

df_joined = df_labels_with_end.alias("l").join(
    df_tel.alias("t"),
    (F.col("t.gpu_id") == F.col("l.gpu_id")) &
    (F.col("t.timestamp") >= F.col("l.window_start")) &
    (F.col("t.timestamp") < F.col("l.window_end")),
    "left"
)

df_training = df_joined.groupBy(
    F.col("l.gpu_id").alias("gpu_id"),
    F.col("l.window_start"),
    F.col("l.is_anomaly"),
).agg(
    F.count("t.timestamp").alias("reading_count"),
    F.avg("t.temp_celsius").alias("avg_temp_celsius"),
    F.max("t.temp_celsius").alias("max_temp_celsius"),
    F.min("t.temp_celsius").alias("min_temp_celsius"),
    F.stddev("t.temp_celsius").alias("std_temp_celsius"),
    F.avg("t.utilization_pct").alias("avg_utilization_pct"),
    F.max("t.utilization_pct").alias("max_utilization_pct"),
    F.avg("t.memory_used_gb").alias("avg_memory_used_gb"),
    F.max("t.memory_used_gb").alias("max_memory_used_gb"),
    F.avg("t.power_watts").alias("avg_power_watts"),
    F.max("t.power_watts").alias("max_power_watts"),
    F.sum("t.error_count").alias("total_error_count"),
    F.max("t.error_count").alias("max_error_count"),
)

df_training.write.mode("overwrite").saveAsTable("main.mlops_genai_workshop.training_dataset")
print(f"training_dataset: {df_training.count()} rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: Data Exploration
# MAGIC
# MAGIC Validate the generated data with summary queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Cluster Inventory

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM main.mlops_genai_workshop.cluster_inventory ORDER BY cluster_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   cloud_provider,
# MAGIC   COUNT(*) AS cluster_count,
# MAGIC   SUM(gpu_count) AS total_gpus,
# MAGIC   COLLECT_SET(gpu_type) AS gpu_types
# MAGIC FROM main.mlops_genai_workshop.cluster_inventory
# MAGIC GROUP BY cloud_provider
# MAGIC ORDER BY cluster_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Telemetry Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_readings,
# MAGIC   COUNT(DISTINCT gpu_id) AS unique_gpus,
# MAGIC   COUNT(DISTINCT cluster_id) AS unique_clusters,
# MAGIC   ROUND(AVG(temp_celsius), 1) AS avg_temp,
# MAGIC   ROUND(MAX(temp_celsius), 1) AS max_temp,
# MAGIC   ROUND(AVG(utilization_pct), 1) AS avg_utilization,
# MAGIC   ROUND(AVG(power_watts), 1) AS avg_power,
# MAGIC   SUM(error_count) AS total_errors
# MAGIC FROM main.mlops_genai_workshop.gpu_telemetry;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CASE WHEN temp_celsius > 90 THEN 'anomalous' ELSE 'normal' END AS reading_type,
# MAGIC   COUNT(*) AS count,
# MAGIC   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM main.mlops_genai_workshop.gpu_telemetry), 2) AS pct
# MAGIC FROM main.mlops_genai_workshop.gpu_telemetry
# MAGIC GROUP BY CASE WHEN temp_celsius > 90 THEN 'anomalous' ELSE 'normal' END;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Health Events

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   event_type,
# MAGIC   COUNT(*) AS event_count,
# MAGIC   SUM(CASE WHEN resolved THEN 1 ELSE 0 END) AS resolved_count,
# MAGIC   ROUND(SUM(CASE WHEN resolved THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS resolution_rate_pct
# MAGIC FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC GROUP BY event_type
# MAGIC ORDER BY event_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   event_type,
# MAGIC   description,
# MAGIC   COUNT(*) AS occurrences
# MAGIC FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC GROUP BY event_type, description
# MAGIC ORDER BY occurrences DESC
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3d. ML Job Runs

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   framework,
# MAGIC   model_type,
# MAGIC   COUNT(*) AS job_count,
# MAGIC   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
# MAGIC   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
# MAGIC   SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running,
# MAGIC   ROUND(SUM(gpu_hours), 1) AS total_gpu_hours,
# MAGIC   ROUND(SUM(cost_usd), 2) AS total_cost_usd
# MAGIC FROM main.mlops_genai_workshop.ml_job_runs
# MAGIC GROUP BY framework, model_type
# MAGIC ORDER BY total_cost_usd DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT job_id, cluster_id, framework, model_type, gpu_hours, cost_usd, status
# MAGIC FROM main.mlops_genai_workshop.ml_job_runs
# MAGIC ORDER BY cost_usd DESC
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3e. Anomaly Labels

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   is_anomaly,
# MAGIC   COUNT(*) AS label_count,
# MAGIC   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM main.mlops_genai_workshop.gpu_anomaly_labels), 2) AS pct
# MAGIC FROM main.mlops_genai_workshop.gpu_anomaly_labels
# MAGIC GROUP BY is_anomaly
# MAGIC ORDER BY is_anomaly;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3f. Training Dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_windows,
# MAGIC   SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) AS anomaly_windows,
# MAGIC   ROUND(AVG(reading_count), 1) AS avg_readings_per_window,
# MAGIC   ROUND(AVG(avg_temp_celsius), 1) AS overall_avg_temp,
# MAGIC   ROUND(AVG(total_error_count), 2) AS avg_total_errors
# MAGIC FROM main.mlops_genai_workshop.training_dataset;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   is_anomaly,
# MAGIC   COUNT(*) AS window_count,
# MAGIC   ROUND(AVG(avg_temp_celsius), 1) AS avg_temp,
# MAGIC   ROUND(AVG(max_temp_celsius), 1) AS avg_max_temp,
# MAGIC   ROUND(AVG(avg_utilization_pct), 1) AS avg_util,
# MAGIC   ROUND(AVG(avg_power_watts), 1) AS avg_power,
# MAGIC   ROUND(AVG(total_error_count), 2) AS avg_errors
# MAGIC FROM main.mlops_genai_workshop.training_dataset
# MAGIC WHERE reading_count > 0
# MAGIC GROUP BY is_anomaly
# MAGIC ORDER BY is_anomaly;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: Verify All Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'cluster_inventory' AS table_name, COUNT(*) AS row_count FROM main.mlops_genai_workshop.cluster_inventory
# MAGIC UNION ALL
# MAGIC SELECT 'gpu_telemetry', COUNT(*) FROM main.mlops_genai_workshop.gpu_telemetry
# MAGIC UNION ALL
# MAGIC SELECT 'gpu_health_events', COUNT(*) FROM main.mlops_genai_workshop.gpu_health_events
# MAGIC UNION ALL
# MAGIC SELECT 'ml_job_runs', COUNT(*) FROM main.mlops_genai_workshop.ml_job_runs
# MAGIC UNION ALL
# MAGIC SELECT 'gpu_anomaly_labels', COUNT(*) FROM main.mlops_genai_workshop.gpu_anomaly_labels
# MAGIC UNION ALL
# MAGIC SELECT 'training_dataset', COUNT(*) FROM main.mlops_genai_workshop.training_dataset
# MAGIC ORDER BY table_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete
# MAGIC
# MAGIC | Asset | Type | Description |
# MAGIC |-------|------|-------------|
# MAGIC | `cluster_inventory` | Table | 30 DGX Cloud clusters across AWS, Azure, GCP, Oracle |
# MAGIC | `gpu_telemetry` | Table | ~50K GPU readings with ~5% anomaly injection |
# MAGIC | `gpu_health_events` | Table | ~2K warning / error / critical events |
# MAGIC | `ml_job_runs` | Table | ~5K training jobs across PyTorch, TensorFlow, JAX |
# MAGIC | `gpu_anomaly_labels` | Table | ~5K binary labels (~5% positive rate) |
# MAGIC | `training_dataset` | Table | Joined feature + label dataset for ML training |
# MAGIC | `docs` | Volume | Documentation and reference materials |
# MAGIC | `app_code` | Volume | Application code artifacts |
# MAGIC
# MAGIC **Next:** Proceed to Block 2 for feature engineering and anomaly detection model training.
