# Databricks notebook source

# MAGIC %md
# MAGIC # Advanced MLOps on Databricks
# MAGIC *From feature engineering to production monitoring -- the complete GPU anomaly detection lifecycle*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC - **Part A** -- Build a GPU health feature table with time-series primary keys
# MAGIC - **Part B** -- Train a LightGBM anomaly detector, register in Unity Catalog
# MAGIC - **Part C** -- Deploy to a serverless serving endpoint with scale-to-zero
# MAGIC - **Part D** -- Attach Lakehouse Monitoring for drift detection and custom alerts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.1: Create Feature Table
# MAGIC Hourly GPU health features with a `TIMESERIES` primary key for point-in-time lookups.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS main.mlops_genai_workshop.gpu_health_features (
# MAGIC   gpu_id            STRING       NOT NULL,
# MAGIC   window_start      TIMESTAMP    NOT NULL,
# MAGIC   avg_temp          DOUBLE,
# MAGIC   max_temp          DOUBLE,
# MAGIC   avg_utilization   DOUBLE,
# MAGIC   avg_memory_pct    DOUBLE,
# MAGIC   avg_power         DOUBLE,
# MAGIC   error_count_1h    LONG,
# MAGIC   error_count_24h   LONG,
# MAGIC   temp_variance     DOUBLE,
# MAGIC   util_trend        DOUBLE,
# MAGIC   PRIMARY KEY (gpu_id, window_start TIMESERIES)
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Hourly GPU health features for anomaly detection'
# MAGIC TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.2: Populate Feature Table
# MAGIC Aggregate raw telemetry into hourly windows with rolling error counts and utilization trends.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

telemetry_df = spark.table("main.mlops_genai_workshop.gpu_telemetry")

# Hourly aggregation
hourly_features = (
    telemetry_df
    .withColumn("window_start", F.date_trunc("hour", F.col("timestamp")))
    .groupBy("gpu_id", "window_start")
    .agg(
        F.avg("temp_celsius").alias("avg_temp"),
        F.max("temp_celsius").alias("max_temp"),
        F.avg("utilization_pct").alias("avg_utilization"),
        F.avg(F.col("memory_used_gb") / 80.0 * 100).alias("avg_memory_pct"),
        F.avg("power_watts").alias("avg_power"),
        F.sum("error_count").alias("error_count_1h"),
        F.variance("temp_celsius").alias("temp_variance"),
    )
)

# Rolling 24-hour error count
window_24h = (
    Window
    .partitionBy("gpu_id")
    .orderBy(F.col("window_start").cast("long"))
    .rangeBetween(-23 * 3600, 0)
)

hourly_features = hourly_features.withColumn(
    "error_count_24h", F.sum("error_count_1h").over(window_24h)
)

# Utilization trend (slope over last 6 hours)
window_6h = (
    Window
    .partitionBy("gpu_id")
    .orderBy("window_start")
    .rowsBetween(-5, 0)
)

hourly_features = hourly_features.withColumn(
    "util_trend",
    F.col("avg_utilization") - F.first("avg_utilization").over(window_6h)
)

# Merge into feature table
hourly_features.createOrReplaceTempView("new_features")

spark.sql("""
    MERGE INTO main.mlops_genai_workshop.gpu_health_features AS target
    USING new_features AS source
    ON target.gpu_id = source.gpu_id AND target.window_start = source.window_start
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print("Feature table populated successfully.")
display(spark.table("main.mlops_genai_workshop.gpu_health_features").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: Train and Register Model
# MAGIC LightGBM binary classifier for GPU anomaly detection, logged to MLflow and registered in Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.1: Prepare Training Data

# COMMAND ----------

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

mlflow.set_registry_uri("databricks-uc")

# Build training data by joining the feature table with anomaly labels
features_df = spark.table("main.mlops_genai_workshop.gpu_health_features")
labels_df = spark.table("main.mlops_genai_workshop.gpu_anomaly_labels")

training_spark_df = features_df.join(
    labels_df,
    on=["gpu_id", "window_start"],
    how="inner"
).na.fill(0)

training_df = training_spark_df.toPandas()

feature_columns = [
    "avg_temp",
    "max_temp",
    "avg_utilization",
    "avg_memory_pct",
    "avg_power",
    "error_count_1h",
    "error_count_24h",
    "temp_variance",
    "util_trend",
]

X = training_df[feature_columns]
y = training_df["is_anomaly"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Anomaly rate (train): {y_train.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.2: Train with LightGBM and Log to MLflow

# COMMAND ----------

model_name = "main.mlops_genai_workshop.gpu_anomaly_detector"
mlflow.set_experiment("/Users/" + spark.sql("SELECT current_user()").first()[0] + "/gpu-anomaly-detection-workshop")

with mlflow.start_run(run_name="gpu_anomaly_lgbm") as run:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 6,
        "class_weight": "balanced",
        "random_state": 42,
    }
    mlflow.log_params(params)

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)],
    )

    y_pred = clf.predict(X_test)

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    mlflow.lightgbm.log_model(
        clf,
        artifact_path="model",
        registered_model_name=model_name,
        input_example=X_test.head(5),
    )

    print(f"\nModel registered: {model_name}")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.3: Set the Champion Alias

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Get the latest version just registered
latest_version = client.search_model_versions(
    f"name='{model_name}'",
    order_by=["version_number DESC"],
    max_results=1,
)[0].version

client.set_registered_model_alias(
    name=model_name,
    alias="Champion",
    version=latest_version,
)

print(f"Set alias 'Champion' on version {latest_version} of {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Model Serving Endpoint
# MAGIC Deploy the Champion model to a serverless endpoint with scale-to-zero and inference auto-capture.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.1: Create the Serving Endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
)
import time

w = WorkspaceClient()

endpoint_name = "gpu-anomaly-detector"
served_model_name = "main.mlops_genai_workshop.gpu_anomaly_detector"

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=served_model_name,
            entity_version=str(latest_version),
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name="main",
        schema_name="mlops_genai_workshop",
        enabled=True,
    ),
)

# Create or update
try:
    existing = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists -- updating configuration...")
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=endpoint_config.served_entities,
        auto_capture_config=endpoint_config.auto_capture_config,
    )
except Exception:
    print(f"Creating endpoint '{endpoint_name}'...")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=endpoint_config,
    )

# Wait for ready state
print("Waiting for endpoint to be ready (this may take a few minutes)...")

for i in range(60):
    status = w.serving_endpoints.get(endpoint_name)
    state = status.state.ready
    if str(state) == "READY":
        print(f"Endpoint '{endpoint_name}' is READY.")
        break
    time.sleep(10)
    if i % 6 == 0:
        print(f"  ...still waiting ({i * 10}s elapsed, state={state})")
else:
    print("WARNING: Endpoint did not reach READY state within 10 minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.2: Test the Serving Endpoint

# COMMAND ----------

import json

sample_records = X_test.head(3).to_dict(orient="records")
payload = {"dataframe_records": sample_records}

print("Request payload:")
print(json.dumps(payload, indent=2))

response = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=sample_records,
)

print("\nPredictions:")
print(json.dumps(response.as_dict(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D: Lakehouse Monitoring and Drift Detection
# MAGIC Attach a Quality Monitor to the inference table for continuous data/prediction drift tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D.1: Create the Lakehouse Monitor

# COMMAND ----------

from databricks.sdk.service.catalog import (
    MonitorTimeSeries,
    MonitorMetric,
    MonitorMetricType,
)

predictions_table = "main.mlops_genai_workshop.model_predictions"
baseline_table = "main.mlops_genai_workshop.model_monitoring_baseline"

try:
    monitor = w.quality_monitors.create(
        table_name=predictions_table,
        time_series=MonitorTimeSeries(
            timestamp_col="prediction_timestamp",
            granularities=["1 hour", "1 day"],
        ),
        baseline_table_name=baseline_table,
        slicing_exprs=[
            "cluster_id",
            "gpu_type",
        ],
        assets_dir=f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/monitoring/gpu_anomaly",
        output_schema_name="main.mlops_genai_workshop",
    )
    print(f"Monitor created on '{predictions_table}'")
    print(f"  Granularities : 1 hour, 1 day")
    print(f"  Baseline table: {baseline_table}")
    print(f"  Slicing exprs : cluster_id, gpu_type")
except Exception as e:
    if "MONITOR_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
        print(f"Monitor already exists on '{predictions_table}' -- skipping creation.")
        monitor = w.quality_monitors.get(table_name=predictions_table)
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### D.2: Refresh the Monitor and View Drift Metrics

# COMMAND ----------

print("Triggering monitor refresh...")
w.quality_monitors.run_refresh(table_name=predictions_table)
print("Refresh initiated. It may take a few minutes to complete.\n")

profile_table = f"{predictions_table}_profile_metrics"
drift_table = f"{predictions_table}_drift_metrics"

print(f"Profile metrics table: {profile_table}")
print(f"Drift metrics table  : {drift_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   window,
# MAGIC   column_name,
# MAGIC   data_type,
# MAGIC   percent_distinct,
# MAGIC   avg,
# MAGIC   stddev,
# MAGIC   percent_null
# MAGIC FROM main.mlops_genai_workshop.model_predictions_profile_metrics
# MAGIC WHERE column_name IN ('avg_temp', 'avg_utilization', 'prediction')
# MAGIC ORDER BY window.start DESC, column_name
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   window,
# MAGIC   column_name,
# MAGIC   chi_squared_test.statistic AS chi2_stat,
# MAGIC   chi_squared_test.pvalue    AS chi2_pvalue,
# MAGIC   ks_test.statistic          AS ks_stat,
# MAGIC   ks_test.pvalue             AS ks_pvalue
# MAGIC FROM main.mlops_genai_workshop.model_predictions_drift_metrics
# MAGIC WHERE drift_type = 'CONSECUTIVE'
# MAGIC ORDER BY window.start DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ### D.3: Custom Business Metric -- Anomaly Rate Alert (> 5%)

# COMMAND ----------

try:
    monitor_updated = w.quality_monitors.update(
        table_name=predictions_table,
        time_series=MonitorTimeSeries(
            timestamp_col="prediction_timestamp",
            granularities=["1 hour", "1 day"],
        ),
        baseline_table_name=baseline_table,
        slicing_exprs=[
            "cluster_id",
            "gpu_type",
        ],
        custom_metrics=[
            MonitorMetric(
                name="anomaly_rate",
                input_columns=[":table"],
                definition="avg(CASE WHEN prediction = 1 THEN 1.0 ELSE 0.0 END)",
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                output_data_type="DOUBLE",
            ),
        ],
    )
    print("Custom metric 'anomaly_rate' added to monitor.")
except Exception as e:
    print(f"Note: {e}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   window,
# MAGIC   slice_key,
# MAGIC   slice_value,
# MAGIC   anomaly_rate,
# MAGIC   CASE
# MAGIC     WHEN anomaly_rate > 0.05 THEN 'ALERT: Anomaly rate exceeds 5%!'
# MAGIC     ELSE 'OK'
# MAGIC   END AS alert_status
# MAGIC FROM main.mlops_genai_workshop.model_predictions_profile_metrics
# MAGIC WHERE anomaly_rate IS NOT NULL
# MAGIC ORDER BY window.start DESC, slice_key, slice_value
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC | Part | What We Built | Key Databricks Features |
# MAGIC |------|---------------|------------------------|
# MAGIC | **A** | GPU health feature table with hourly aggregations | Unity Catalog, Feature Store, TIMESERIES PK |
# MAGIC | **B** | LightGBM anomaly detection model | MLflow, UC Model Registry, Champion alias |
# MAGIC | **C** | Serverless model serving endpoint | Model Serving, scale-to-zero, auto-capture |
# MAGIC | **D** | Lakehouse monitoring with drift detection | Quality Monitors, slicing, custom metrics |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Set up SQL alerts on the anomaly rate metric to trigger PagerDuty/Slack notifications
# MAGIC - Build a Databricks AI/BI dashboard to visualize drift trends over time
# MAGIC - Deploy a Challenger model alongside the Champion for A/B testing
# MAGIC - Connect the feature table to real-time DGX Cloud telemetry via Lakeflow Connect
