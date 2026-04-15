# NVIDIA DGX Cloud Workshop: MLOps & Generative AI on Databricks

**Date:** Wednesday, April 15, 2026 | 9:00 AM - 5:00 PM  
**Instructor:** Chandhana Padmanabhan  
**Audience:** NVIDIA DGX Cloud developers (setup completed on Tuesday with Andrew/Esha)  
**Workspace:** `e2-demo-field-eng.cloud.databricks.com` (profile: `e2-demo-west`)  
**Prerequisite:** Claude Code + Databricks CLI authenticated (verified Tuesday)  
**SQL Warehouse:** `75fd8278393d07eb` (Shared, 2X-Large)

---

## Philosophy

Every section has **two build paths**:
1. **Prompt-Driven** — Use Claude Code to generate and deploy via natural language
2. **Genie/Code-Driven** — Use Genie Spaces, SQL, and Python directly

"We do not leave until the code runs."

---

## Unity Catalog Schema Plan

```
Catalog: main
Schema:  main.mlops_genai_workshop   <-- NEW (all workshop assets live here)

Tables to create:
  -- GPU Fleet Data (synthetic, from Andrew's DGX data or generated fresh) --
  main.mlops_genai_workshop.gpu_telemetry          -- ~50K rows: gpu_id, cluster_id, timestamp, temp_celsius, utilization_pct, memory_used_gb, power_watts, error_count
  main.mlops_genai_workshop.gpu_health_events       -- ~2K rows: event_id, gpu_id, event_type (warning/error/critical), description, timestamp
  main.mlops_genai_workshop.cluster_inventory        -- ~30 rows: cluster_id, cloud_provider, region, gpu_type, gpu_count, status
  main.mlops_genai_workshop.ml_job_runs             -- ~5K rows: job_id, cluster_id, framework, model_type, start_time, end_time, gpu_hours, status, cost_usd

  -- MLOps Feature/Training Tables --
  main.mlops_genai_workshop.gpu_health_features     -- Feature table for ML: aggregated GPU metrics per device per hour
  main.mlops_genai_workshop.gpu_anomaly_labels      -- Label table: gpu_id, timestamp, is_anomaly (0/1)
  main.mlops_genai_workshop.training_dataset        -- Joined features + labels for model training
  main.mlops_genai_workshop.model_predictions       -- Batch inference output
  main.mlops_genai_workshop.model_monitoring_baseline -- Baseline for monitoring

  -- GenAI Tables --
  main.mlops_genai_workshop.ai_classified_events    -- Output of ai_classify() on GPU events
  main.mlops_genai_workshop.ai_extracted_insights   -- Output of ai_extract() on job descriptions
  main.mlops_genai_workshop.claude_sql_results      -- Output of Claude-generated SQL via ai_query()

Volumes:
  main.mlops_genai_workshop.docs                    -- GPU runbooks, incident reports (PDFs for RAG)
  main.mlops_genai_workshop.app_code                -- App deployment files

Genie Space:
  "DGX Fleet Analytics" — over gpu_telemetry, cluster_inventory, ml_job_runs

Knowledge Assistant:
  "GPU Operations Advisor" — over docs volume (runbooks, incident reports)

Multi-Agent Supervisor:
  "DGX Operations Center" — routes to Genie (quantitative) + KA (qualitative)
```

---

## Agenda: Time Block Detail

### Block 1: Welcome & Environment Check
| | |
|---|---|
| **Time** | 9:00 - 9:30 (30 min) |
| **Type** | Setup + Interactive |
| **Notebook** | `00_setup_and_explore.py` |

**Learning Objectives:**
- Confirm Claude Code + Databricks connectivity
- Create workshop schema and verify SQL warehouse
- Explore the GPU fleet dataset

**Content:**
1. Quick intro: what we're building today (5 min lecture)
2. Run setup notebook: create schema, load synthetic GPU data (10 min hands-on)
3. Explore data with SQL — basic queries on gpu_telemetry, cluster_inventory (10 min)
4. Buffer / troubleshoot stragglers (5 min)

**Two Build Paths:**
- **Prompt-driven:** "Create a schema main.mlops_genai_workshop and load synthetic GPU telemetry data for 30 clusters"
- **Code-driven:** Run the setup notebook cells directly

---

### Block 2: Generative AI on Databricks — The Foundation
| | |
|---|---|
| **Time** | 9:30 - 10:30 (60 min) |
| **Type** | Lecture (20 min) + Hands-on (40 min) |
| **Notebook** | `01_genai_foundations.py` |

**Learning Objectives:**
- Understand Mosaic AI and Foundation Model APIs
- Use inline AI functions in SQL (ai_query, ai_classify, ai_extract, ai_summarize, ai_similarity)
- Invoke Claude from SQL using ai_query()
- Generate SQL from Python using Claude, then execute it

**Slide Deck (20 min):**
1. Mosaic AI platform overview — what's included
2. Foundation Model APIs: pay-per-token, provisioned throughput, external models
3. AI Functions in SQL — the 6 built-in functions
4. Claude as an external model on Databricks
5. Architecture: Python orchestration layer generating SQL

**Hands-on (40 min):**

*Part A — AI Functions in SQL (20 min):*
```sql
-- Classify GPU health events by severity
SELECT event_id, description,
  ai_classify(description, ARRAY('hardware_failure', 'thermal_throttle', 'memory_error', 'software_bug', 'network_issue')) as root_cause
FROM main.mlops_genai_workshop.gpu_health_events
LIMIT 20;

-- Extract structured info from free-text job descriptions
SELECT job_id,
  ai_extract(description, ARRAY('framework', 'model_size', 'dataset', 'objective')) as extracted
FROM main.mlops_genai_workshop.ml_job_runs
WHERE description IS NOT NULL
LIMIT 10;

-- Summarize cluster health over the past week
SELECT cluster_id,
  ai_summarize(CONCAT_WS('; ', COLLECT_LIST(description))) as health_summary
FROM main.mlops_genai_workshop.gpu_health_events
GROUP BY cluster_id;
```

*Part B — Invoking Claude from SQL (10 min):*
```sql
-- Use Claude via ai_query with an external model endpoint
SELECT
  gpu_id,
  description,
  ai_query(
    'claude-sonnet-4-5-20250929',
    CONCAT('You are a GPU operations expert. Analyze this GPU health event and recommend an action: ', description)
  ) as claude_recommendation
FROM main.mlops_genai_workshop.gpu_health_events
WHERE event_type = 'critical'
LIMIT 5;
```

*Part C — Python-to-SQL Generation (10 min):*
```python
# Use Claude to generate SQL, then execute it on Databricks
import anthropic

client = anthropic.Anthropic()
question = "Which GPU clusters had the highest average temperature in the last 24 hours?"

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": f"""
    Given these tables:
    - main.mlops_genai_workshop.gpu_telemetry (gpu_id, cluster_id, timestamp, temp_celsius, utilization_pct, memory_used_gb, power_watts)
    - main.mlops_genai_workshop.cluster_inventory (cluster_id, cloud_provider, region, gpu_type, gpu_count)

    Write a Databricks SQL query to answer: {question}
    Return ONLY the SQL, no explanation.
    """}]
)

sql_query = response.content[0].text
result = spark.sql(sql_query)
display(result)
```

**Two Build Paths:**
- **Prompt-driven:** "Use Claude Code to write a notebook that classifies all GPU health events using ai_classify and generates remediation advice using ai_query with Claude"
- **Code-driven:** Follow notebook cells step by step

---

### BREAK: 10:30 - 10:45 (15 min)

---

### Block 3: Genie Spaces & Programmatic Access
| | |
|---|---|
| **Time** | 10:45 - 12:00 (75 min) |
| **Type** | Demo (15 min) + Hands-on (40 min) + Discussion (20 min) |
| **Notebook** | `02_genie_spaces.py` |

**Learning Objectives:**
- Build a Genie Space over GPU fleet data
- Use the Genie Conversation API programmatically
- Understand MCP-style patterns over Genie
- Quick tool-building for engineers using MCP + Genie

**Slide Deck (10 min):**
1. What is Genie? Natural language SQL for any dataset
2. Genie Conversation API — REST access for programmatic use
3. MCP architecture: how Genie can act as a "tool" in an MCP server
4. Quick tool-building: Claude Code + Databricks MCP = instant tooling

**Hands-on (40 min):**

*Part A — Build a Genie Space (20 min):*
1. Create Genie Space: "DGX Fleet Analytics"
2. Add tables: `gpu_telemetry`, `cluster_inventory`, `ml_job_runs`
3. Add instructions: GPU fleet context, join patterns, unit clarifications
4. Add sample questions:
   - "Which cluster has the most GPU errors this week?"
   - "What is the average GPU utilization across all AWS clusters?"
   - "Show me the top 10 most expensive ML jobs by GPU-hours"
5. Test with natural language queries

*Part B — Genie Conversation API (20 min):*
```python
# Programmatic access to Genie via REST API
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Start a Genie conversation
genie_space_id = "<your-genie-space-id>"

# Ask a question programmatically
conversation = w.genie.start_conversation(
    space_id=genie_space_id,
    content="Which GPU clusters have utilization below 50% but are running expensive jobs?"
)

# Get the response (includes generated SQL + results)
result = w.genie.get_message_query_result(
    space_id=genie_space_id,
    conversation_id=conversation.conversation_id,
    message_id=conversation.message_id
)
print(result)
```

**Discussion: MCP-Style Patterns (20 min):**
- Genie as a "tool" in an MCP server — each Genie Space becomes a queryable tool
- Pattern: Claude Code asks a question -> MCP server calls Genie API -> returns structured data
- Demo: Databricks MCP server with Genie tool integration
- When this makes sense vs. direct SQL
- Quick tool-building workflow for engineers:
  1. Load data into Unity Catalog
  2. Create Genie Space (2 min)
  3. Expose via MCP or API (5 min)
  4. Claude Code can now query your data conversationally

**Two Build Paths:**
- **Prompt-driven:** "Create a Genie Space called DGX Fleet Analytics over my GPU telemetry, cluster inventory, and ML job runs tables. Add sample questions about cluster utilization and job costs."
- **Code-driven:** Step through UI + API calls manually

---

### LUNCH: 12:00 - 1:00 (60 min)

---

### Block 4: Vector Search, RAG & Knowledge Assistants
| | |
|---|---|
| **Time** | 1:00 - 2:00 (60 min) |
| **Type** | Lecture (10 min) + Hands-on (50 min) |
| **Notebook** | `03_vector_search_rag.py` |

**Learning Objectives:**
- Create a Vector Search endpoint and index
- Build a Knowledge Assistant for GPU operations documentation
- Combine Genie + KA in a Multi-Agent Supervisor

**Slide Deck (10 min):**
1. Vector Search architecture: embedding model -> vector index -> similarity search
2. Knowledge Assistants: managed RAG with source citations
3. Multi-Agent Supervisor: routing quantitative vs. qualitative questions
4. Customer use case: ops teams using agents for incident triage

**Hands-on (50 min):**

*Part A — Vector Search Setup (15 min):*
- Upload GPU runbook PDFs to `main.mlops_genai_workshop.docs` volume
- Create Vector Search endpoint (or use existing)
- Index the documents with `databricks-gte-large-en` embeddings
- Query the index with similarity search

*Part B — Knowledge Assistant (20 min):*
1. Create KA: "GPU Operations Advisor"
2. Add knowledge source: `main.mlops_genai_workshop.docs` volume
3. Instructions: "You are a GPU operations expert. Answer questions about GPU troubleshooting, maintenance procedures, and performance optimization using only the provided documentation."
4. Sync & test:
   - "What is the recommended procedure when GPU temperature exceeds 85C?"
   - "How do I diagnose ECC memory errors on A100 GPUs?"
   - "What are the SLA requirements for DGX Cloud GPU availability?"

*Part C — Multi-Agent Supervisor (15 min):*
1. Create MAS: "DGX Operations Center"
2. Add tools:
   - Genie Space "DGX Fleet Analytics" (for data questions)
   - Knowledge Assistant "GPU Operations Advisor" (for documentation questions)
3. Test routing:
   - "How many GPUs are currently showing thermal warnings?" -> routes to Genie
   - "What is the runbook procedure for thermal throttling?" -> routes to KA
   - "We have 5 GPUs showing thermal warnings — what should we do and how many are affected across clusters?" -> routes to BOTH

**Two Build Paths:**
- **Prompt-driven:** "Create a Knowledge Assistant called GPU Operations Advisor over the docs volume in my workshop schema, then create a Multi-Agent Supervisor that combines it with the DGX Fleet Analytics Genie Space"
- **Code-driven:** Step through AgentBricks UI

---

### Block 5: Databricks Apps — Why and How
| | |
|---|---|
| **Time** | 2:00 - 2:45 (45 min) |
| **Type** | Lecture (10 min) + Hands-on (35 min) |
| **Notebook** | `04_databricks_apps.py` |

**Learning Objectives:**
- Understand why deploying apps on Databricks makes sense
- Deploy a GPU Fleet Monitor dashboard app
- Understand auth, SQL warehouse connectivity, sharing patterns

**Slide Deck (10 min):**
1. Why apps on Databricks?
   - Zero infrastructure: no K8s, no Docker, no cloud accounts
   - Built-in auth: SSO, service principals, user-level permissions
   - Direct data access: Unity Catalog, SQL warehouses, ML endpoints
   - Governed: audit logs, lineage, permissions all built in
   - Shareable: URL-based access for anyone in the org
2. App types: Streamlit, Gradio, FastAPI, Dash, Flask
3. Architecture: app runtime -> SQL warehouse -> Unity Catalog
4. Real-world patterns: dashboards, chatbots, ML model frontends, internal tools

**Hands-on (35 min):**

*Deploy a GPU Fleet Monitor App:*
- Streamlit app that reads from `gpu_telemetry`, `cluster_inventory`, `gpu_health_events`
- Features:
  - KPI tiles: total GPUs, active alerts, avg utilization, avg temperature
  - Cluster health heatmap
  - GPU temperature time series chart
  - Recent alerts table with severity color-coding
  - Filterable by cluster, cloud provider, date range
- Deploy via `databricks apps create` + `databricks apps deploy`
- Share the URL with neighbors

**Two Build Paths:**
- **Prompt-driven:** "Build me a Streamlit Databricks App that monitors GPU fleet health with KPI tiles, a temperature heatmap by cluster, and a recent alerts table. Use the gpu_telemetry and gpu_health_events tables in main.mlops_genai_workshop."
- **Code-driven:** Review provided app.py, update config, deploy via CLI

---

### BREAK: 2:45 - 3:00 (15 min)

---

### Block 6: Advanced MLOps
| | |
|---|---|
| **Time** | 3:00 - 4:15 (75 min) |
| **Type** | Lecture (15 min) + Hands-on (60 min) |
| **Notebook** | `05_advanced_mlops.py` |

**Learning Objectives:**
- Build feature tables with Unity Catalog constraints
- Train a GPU anomaly detection model with MLflow tracking
- Register models with Champion/Challenger pattern
- Deploy to a Model Serving endpoint
- Set up Lakehouse Monitoring with drift detection

**Slide Deck (15 min):**
1. MLOps lifecycle: features -> train -> register -> validate -> serve -> monitor -> retrain
2. Unity Catalog as the ML backbone: feature tables, model registry, lineage
3. Champion/Challenger pattern for safe rollouts
4. Model Serving: auto-scaling, A/B testing, scale-to-zero
5. Lakehouse Monitoring: data drift, prediction drift, custom business metrics
6. Automated retraining triggers via Databricks Workflows

**Source:** Advanced track from `dbdemos mlops-end2end` (notebooks 01-08), adapted for GPU health use case.

**Hands-on (60 min):**

*Part A — Feature Engineering (15 min):*
```python
# Create a feature table with UC constraints
spark.sql("""
CREATE OR REPLACE TABLE main.mlops_genai_workshop.gpu_health_features (
  gpu_id STRING NOT NULL,
  window_start TIMESTAMP NOT NULL,
  avg_temp DOUBLE,
  max_temp DOUBLE,
  avg_utilization DOUBLE,
  avg_memory_pct DOUBLE,
  avg_power DOUBLE,
  error_count_1h INT,
  error_count_24h INT,
  temp_variance DOUBLE,
  util_trend DOUBLE,
  CONSTRAINT pk PRIMARY KEY (gpu_id, window_start TIMESERIES)
) USING DELTA
COMMENT 'Hourly aggregated GPU health features for anomaly detection'
""")

# Populate features from raw telemetry
spark.sql("""
INSERT INTO main.mlops_genai_workshop.gpu_health_features
SELECT
  gpu_id,
  date_trunc('HOUR', timestamp) as window_start,
  AVG(temp_celsius) as avg_temp,
  MAX(temp_celsius) as max_temp,
  AVG(utilization_pct) as avg_utilization,
  AVG(memory_used_gb / 80.0 * 100) as avg_memory_pct,
  AVG(power_watts) as avg_power,
  SUM(error_count) as error_count_1h,
  NULL as error_count_24h,
  STDDEV(temp_celsius) as temp_variance,
  NULL as util_trend
FROM main.mlops_genai_workshop.gpu_telemetry
GROUP BY gpu_id, date_trunc('HOUR', timestamp)
""")
```

*Part B — Train & Register Model (15 min):*
```python
import mlflow
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

mlflow.set_experiment("/Users/{user}/gpu-anomaly-detection")

# Load training data
df = spark.table("main.mlops_genai_workshop.training_dataset").toPandas()
X = df.drop(columns=["gpu_id", "window_start", "is_anomaly"])
y = df["is_anomaly"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="gpu_anomaly_lgbm"):
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    mlflow.log_metrics({
        "f1_score": f1_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds)
    })
    mlflow.sklearn.log_model(model, "model")

# Register to Unity Catalog
best_run = mlflow.search_runs(order_by=["metrics.f1_score DESC"]).iloc[0]
model_uri = f"runs:/{best_run.run_id}/model"
mlflow.register_model(model_uri, "main.mlops_genai_workshop.gpu_anomaly_detector")

# Set Champion alias
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("main.mlops_genai_workshop.gpu_anomaly_detector", "Champion", 1)
```

*Part C — Model Serving Endpoint (15 min):*
```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

w.serving_endpoints.create(
    name="gpu-anomaly-detector",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name="main.mlops_genai_workshop.gpu_anomaly_detector",
                entity_version="1",
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
)

# Test the endpoint
import requests
response = w.serving_endpoints.query(
    name="gpu-anomaly-detector",
    dataframe_records=[{
        "avg_temp": 82.5, "max_temp": 91.0, "avg_utilization": 95.2,
        "avg_memory_pct": 88.0, "avg_power": 380.0, "error_count_1h": 3,
        "error_count_24h": 15, "temp_variance": 8.5, "util_trend": 0.05
    }]
)
print(response)
```

*Part D — Lakehouse Monitoring + Drift Detection (15 min):*
```python
from databricks.sdk.service.catalog import MonitorTimeSeries

# Create a monitor on the predictions table
w.quality_monitors.create(
    table_name="main.mlops_genai_workshop.model_predictions",
    time_series=MonitorTimeSeries(
        timestamp_col="prediction_timestamp",
        granularities=["1 hour", "1 day"]
    ),
    baseline_table_name="main.mlops_genai_workshop.model_monitoring_baseline",
    slicing_exprs=["cluster_id", "gpu_type"],
    output_schema_name="main.mlops_genai_workshop"
)

# View drift metrics
drift_df = spark.table("main.mlops_genai_workshop.model_predictions_drift_metrics")
display(drift_df.orderBy("window_start"))

# Custom business metric: alert if anomaly rate > 5%
from pyspark.sql import functions as F
anomaly_rate = spark.table("main.mlops_genai_workshop.model_predictions") \
    .groupBy("cluster_id") \
    .agg(F.avg("prediction").alias("anomaly_rate")) \
    .filter("anomaly_rate > 0.05")
display(anomaly_rate)
```

**Two Build Paths:**
- **Prompt-driven:** "Use Claude Code to build an end-to-end MLOps pipeline: create a feature table from gpu_telemetry, train a LightGBM anomaly detector, register it in Unity Catalog, deploy to a serving endpoint, and set up monitoring"
- **Code-driven:** Follow notebook cells sequentially

---

### Block 7: Customer Use Cases & Putting It Together
| | |
|---|---|
| **Time** | 4:15 - 4:45 (30 min) |
| **Type** | Lecture (15 min) + Live Demo (15 min) |
| **Notebook** | `06_end_to_end_demo.py` |

**Learning Objectives:**
- See how all pieces connect in a production workflow
- Understand real customer use cases across industries
- See automated retraining triggered by drift

**Customer Use Cases (15 min lecture):**
1. **Telecom — Network anomaly detection:** Feature Store + Model Serving + real-time scoring on network telemetry (similar to your GPU monitoring)
2. **Financial Services — Due diligence automation:** Multi-agent supervisors routing questions to Genie (quantitative) + Knowledge Assistants (qualitative) for investment analysis
3. **Manufacturing — Predictive maintenance:** Streaming sensor data -> feature engineering -> ML model -> automated alerts (direct analogy to GPU health monitoring)
4. **Healthcare — Clinical document analysis:** ai_extract + Knowledge Assistants for parsing unstructured medical records
5. **Retail — Demand forecasting with LLM commentary:** MLOps pipeline + Claude generating executive summaries of forecast deviations

**Live Demo: GPU Fleet Operations Pipeline (15 min):**
- Show the complete flow end-to-end:
  1. New GPU telemetry arrives (simulated)
  2. Feature table updates automatically
  3. Anomaly detection model scores in real-time
  4. Genie Space shows current fleet status
  5. Knowledge Assistant provides runbook guidance
  6. Supervisor Agent triages: "5 GPUs showing thermal anomalies on cluster dgx-aws-03. Runbook says: reduce workload, check cooling. Current utilization is 97% — recommend migrating 2 jobs to dgx-gcp-01 which is at 42% utilization."
  7. Monitoring dashboard shows drift metrics

---

### Block 8: Wrap-Up & Q&A
| | |
|---|---|
| **Time** | 4:45 - 5:00 (15 min) |
| **Type** | Discussion |

**Content:**
1. Recap: what we built today (2 min)
2. Resources: AI Dev Kit repo, dbdemos, Databricks documentation (3 min)
3. Next steps for NVIDIA's DGX Cloud team (5 min)
4. Open Q&A (5 min)

---

## Notebook Sequence

| # | Notebook | Duration | Type | Key Datasets |
|---|----------|----------|------|-------------|
| 00 | `00_setup_and_explore.py` | 30 min | Setup | Creates schema, loads all synthetic data |
| 01 | `01_genai_foundations.py` | 40 min | Hands-on | gpu_health_events, ml_job_runs |
| 02 | `02_genie_spaces.py` | 40 min | Hands-on | gpu_telemetry, cluster_inventory, ml_job_runs |
| 03 | `03_vector_search_rag.py` | 50 min | Hands-on | docs volume (PDFs), gpu_health_events |
| 04 | `04_databricks_apps.py` | 35 min | Hands-on | gpu_telemetry, gpu_health_events, cluster_inventory |
| 05 | `05_advanced_mlops.py` | 60 min | Hands-on | gpu_health_features, training_dataset, model_predictions |
| 06 | `06_end_to_end_demo.py` | 15 min | Demo | All tables |

---

## Datasets to Prepare

### 1. GPU Telemetry (synthetic, ~50K rows)
Generate using Faker/random:
- `gpu_id`: STRING (e.g., "gpu-a100-001")
- `cluster_id`: STRING (e.g., "dgx-aws-us-east-01")
- `timestamp`: TIMESTAMP (last 7 days, 5-min intervals)
- `temp_celsius`: DOUBLE (65-95, with anomaly spikes)
- `utilization_pct`: DOUBLE (0-100)
- `memory_used_gb`: DOUBLE (0-80 for A100)
- `power_watts`: DOUBLE (200-400)
- `error_count`: INT (0-5, heavy zero-weighted)

### 2. GPU Health Events (synthetic, ~2K rows)
- `event_id`: STRING
- `gpu_id`: STRING
- `event_type`: STRING (warning, error, critical)
- `description`: STRING (realistic GPU error descriptions)
- `timestamp`: TIMESTAMP
- `resolved`: BOOLEAN

### 3. Cluster Inventory (synthetic, ~30 rows)
- `cluster_id`: STRING
- `cloud_provider`: STRING (AWS, Azure, GCP, Oracle)
- `region`: STRING
- `gpu_type`: STRING (A100, H100, H200)
- `gpu_count`: INT
- `status`: STRING (active, maintenance, scaling)

### 4. ML Job Runs (synthetic, ~5K rows)
- `job_id`: STRING
- `cluster_id`: STRING
- `framework`: STRING (PyTorch, TensorFlow, JAX)
- `model_type`: STRING (LLM, Vision, RL, Tabular)
- `description`: STRING (free text)
- `start_time`, `end_time`: TIMESTAMP
- `gpu_hours`: DOUBLE
- `status`: STRING (completed, running, failed)
- `cost_usd`: DOUBLE

### 5. GPU Runbook PDFs (5-10 synthetic documents)
- GPU Thermal Management Guide
- ECC Memory Error Troubleshooting
- DGX Cloud SLA Reference
- GPU Performance Optimization Best Practices
- Incident Response Procedures

### 6. Anomaly Labels (synthetic, ~5K rows)
- `gpu_id`: STRING
- `window_start`: TIMESTAMP
- `is_anomaly`: INT (0 or 1, ~5% positive rate)

---

## Slide Deck Outline

### Opening Slides (5 min)
1. Title: "MLOps & Generative AI on Databricks"
2. Agenda overview
3. What you'll build today (architecture diagram)

### Block 2 Slides: GenAI Foundations (20 min)
4. Mosaic AI platform map
5. Foundation Model APIs: 3 deployment modes
6. External models: Claude, GPT-4, Gemini on Databricks
7. AI Functions in SQL: the 6 functions with examples
8. ai_query() deep dive: invoking Claude from SQL
9. Pattern: Python generates SQL via LLM -> executes on warehouse
10. Architecture: LLM + SQL warehouse + Unity Catalog

### Block 3 Slides: Genie Spaces (10 min)
11. What is Genie? (screenshot)
12. Genie Conversation API: REST endpoints
13. MCP architecture diagram: Claude Code -> MCP Server -> Genie API -> SQL Warehouse
14. Quick tool-building workflow (4 steps)

### Block 4 Slides: Vector Search & RAG (10 min)
15. Vector Search architecture
16. Knowledge Assistant: managed RAG
17. Multi-Agent Supervisor: routing logic
18. Customer use case: ops team incident triage

### Block 5 Slides: Apps (10 min)
19. Why deploy on Databricks? (comparison table vs. traditional)
20. App architecture: runtime -> warehouse -> UC
21. Supported frameworks
22. Sharing and governance

### Block 6 Slides: Machine Learning & MLOps (15 min lecture, 12 slides)

**Source Mapping:** Curated from 3 internal Databricks decks:
- **Deck A:** "MLOps end2end" (`1W2Cxv...`)
- **Deck B:** "[Field-demos]-MLOps end2end - Churn" (`1qY8d1...`)
- **Deck C:** "Databricks Overview (ML)" (`1IZMXFoJ...`)

| Slide # | Title | Source | Content & Talking Points |
|---------|-------|--------|-------------------------|
| **25** | Why Are Companies Struggling with ML? | Deck A, Slide 2 / Deck B, Slide 4 | Three root causes: (1) Data not ML-ready — siloed, fragmented. (2) ML hard to productionize — lifecycle is patched together, doesn't scale. (3) Low team productivity — no collaboration medium. **Hook:** "87% of ML projects never make it to production." Tie to NVIDIA: "You're building the GPUs the world trains on — but operationalizing ML on those GPUs is the unsolved piece." |
| **26** | Full ML Lifecycle: Data Ingest to Deployment | Deck A, Slide 5 / Deck C, Slide 7 | The 3-layer framework: **DataOps** (data prep, versioning, featurization) -> **ModelOps** (training, tracking, registry, serving, monitoring) -> **DevOps** (automation, governance). Show how each maps to a Databricks product. **Callout:** "Everything we built in the GenAI blocks this morning — Genie, Knowledge Assistants, Apps — these are the serving layer. Now we go upstream to see how the model gets there." |
| **27** | Databricks for Data Science & ML | Deck A, Slide 4 / Deck C, Slide 4 | Platform overview slide: Collaborative Notebooks, Feature Store, MLflow Tracking, Model Registry, AutoML, Model Serving, Lakehouse Monitoring. Position as "one platform for the entire lifecycle" vs. patching 8 tools together. |
| **28** | Unity Catalog: Governance for Data AND AI | Deck C, Slide 8 | UC governs tables, files, AND models. Discovery, Access Control, Auditing, Lineage, Monitoring — all unified. **Key point:** "When you register a model in UC, it inherits the same governance as the data it was trained on. That's the Lakehouse advantage." Show lineage from `gpu_telemetry` -> `gpu_health_features` -> `gpu_anomaly_detector` model. |
| **29** | Feature Store: From Raw Data to ML-Ready Features | Deck C, Slide 11 | Feature Registry (discoverability, versioning, lineage), Online + Offline Feature Serving, prevention of training/serving skew. **Workshop tie-in:** "We built `gpu_health_features` with a PRIMARY KEY constraint — that's a UC Feature Table. It auto-versions, tracks lineage, and can serve features online to your model endpoint." |
| **30** | MLflow: Experiment Tracking & Model Registry | Deck C, Slide 12 / Deck A, Slide 11 | Two halves: (1) **Tracking** — auto-log params, metrics, artifacts for every run. (2) **Registry** — Champion/Challenger aliases (replaces old Staging/Production). Show the experiment URL from our workshop run. **Demo moment:** Open the MLflow experiment in the browser and show the logged GPU anomaly model. |
| **31** | MLOps Framework: CI/CD/CT/CM | Deck A, Slide 10 | The 4 continuous loops: **CP** (Continuous Pipeline — data ingestion + quality), **CI** (Continuous Integration — feature eng + training + HPO), **CD** (Continuous Deployment — validation + promotion + serving), **CT** (Continuous Training — drift-triggered retraining), **CM** (Continuous Monitoring — metrics + alerts). Map each to a Databricks product. **Callout:** "This is where the money is. Most teams get CI working but never close the CT + CM loop." |
| **32** | End-to-End MLOps Workflow (v3) | Deck B, Slide 39 | The most current workflow diagram (v3 with monitoring): Data Prep -> UC Feature Tables -> Model Training (MLflow) -> UC Model Registry -> Champion/Challenger -> Model Validation Job -> Batch Inference -> Model Serving -> Inference Tables -> Lakehouse Monitoring -> Drift Detection & Retrain. **This is the "money slide."** Walk through each step and map to what we're building in the hands-on. |
| **33** | Model Serving: From Notebook to Production | Deck C, Slide 14 (adapted) / Deck B, Slide 39 detail | Serving endpoint architecture: Scale-to-zero, auto-scaling, CPU/GPU, A/B testing with traffic splitting (80/20 Champion vs. Challenger). REST API access. **Workshop tie-in:** "We'll deploy our GPU anomaly detector as a serving endpoint. It scales to zero when idle, wakes up on the first request, and you can A/B test Champion vs. Challenger." |
| **34** | Lakehouse Monitoring & Drift Detection | Deck B, Slide 39 detail / Deck A, Slide 9 | Three monitoring targets: (1) **Data drift** — input feature distributions shift. (2) **Prediction drift** — model outputs change. (3) **Label drift** — ground truth changes. Auto-generated profile + drift metrics tables, custom business metrics (e.g., "alert if anomaly rate > 5%"). Monitoring dashboard in Catalog Explorer. **Key stat:** "Jensen-Shannon distance > 0.2 = significant drift = trigger retraining." |
| **35** | Automated Retraining & Webhook Flow | Deck B, Slide 29 / Deck A, Slide 22 | The automation loop: Monitoring detects drift -> triggers Databricks Workflow -> retrains model -> registers new version -> validation job (schema, accuracy, business KPIs) -> approve/reject -> promote to Champion or notify DS. Webhook/Slack integration for human-in-the-loop. **Callout:** "This is the CT loop from slide 31. It's what separates a demo from production." |
| **36** | Production MLOps Architecture | New (composite) | Full architecture diagram combining everything: GPU Telemetry (streaming) -> Feature Table -> Training Pipeline -> UC Model Registry (@Champion) -> Model Serving Endpoint -> Monitoring -> Drift Alert -> Retrain Workflow. Overlay the GenAI layer: Genie Space for querying, Knowledge Assistant for runbooks, Supervisor Agent for triage. **Closing message:** "One platform. From raw GPU telemetry to production-serving anomaly detection to AI-powered operations — all governed by Unity Catalog." |

**Delivery Notes for Block 6 Slides:**
- Spend 1 min max per slide — these are scene-setting before the 60-min hands-on
- Slides 25-28 are "why" slides (2 min each, 8 min total)
- Slides 29-31 are "what" slides (1.5 min each, 4.5 min total)
- Slide 32 is the "money slide" — pause, walk through slowly (2 min)
- Slides 33-36 are "how" slides (1 min each, preview what they'll build hands-on)
- Total: ~15 min lecture, then straight into the notebook

### Block 7 Slides: Customer Use Cases (15 min)
37. Telecom: network anomaly detection
38. Financial Services: due diligence automation
39. Manufacturing: predictive maintenance
40. Healthcare: clinical document analysis
41. Retail: demand forecasting + LLM commentary

### Closing Slides (5 min)
42. Recap: what we built
43. Resources and next steps
44. Q&A

---

## Setup Steps (Pre-Workshop)

### Night Before (Chandhana):
1. Verify SQL warehouse `75fd8278393d07eb` is running
2. Create schema `main.mlops_genai_workshop`
3. Generate and load all synthetic datasets
4. Upload GPU runbook PDFs to volume
5. Pre-create Genie Space (as fallback)
6. Pre-create Knowledge Assistant (as fallback — indexing takes 15-30 min)
7. Pre-create Multi-Agent Supervisor (as fallback)
8. Test all notebook cells end-to-end
9. Prepare slide deck
10. Verify Claude external model endpoint is configured

### Morning Of:
1. Arrive 30 min early
2. Verify warehouse is warm
3. Test Claude Code -> Databricks connectivity
4. Have fallback pre-built resources ready
5. Share notebook repo link with attendees

---

## What's Lecture vs. Hands-On

| Block | Lecture | Hands-On | Buffer/Q&A |
|-------|---------|----------|------------|
| Block 1: Setup | 5 min | 20 min | 5 min |
| Block 2: GenAI | 20 min | 40 min | — |
| Block 3: Genie | 10 min + 20 min discussion | 40 min | 5 min |
| Block 4: RAG | 10 min | 50 min | — |
| Block 5: Apps | 10 min | 35 min | — |
| Block 6: MLOps | 15 min | 60 min | — |
| Block 7: Use Cases | 15 min | 15 min (demo) | — |
| Block 8: Wrap-Up | 5 min | — | 10 min |
| **Totals** | **110 min (1h50m)** | **260 min (4h20m)** | **20 min** |
| Breaks/Lunch | | | **90 min** |

**Total: 480 min = 8 hours (9 AM - 5 PM)**

---

## Demo Flow (E2E, for Block 7)

```
[Simulated GPU Telemetry Spike]
       |
       v
[Feature Table Auto-Update]
       |
       v
[Model Serving Endpoint] --> [Anomaly Detected: gpu-a100-017, cluster dgx-aws-03]
       |
       v
[Genie Space Query] --> "5 GPUs on dgx-aws-03 showing anomalies. Avg temp: 91C. Utilization: 97%"
       |
       v
[Knowledge Assistant] --> "Runbook: Reduce workload, check cooling fans, verify airflow"
       |
       v
[Supervisor Agent] --> Combined response with data + recommended actions
       |
       v
[Monitoring Dashboard] --> Drift metrics show temperature distribution shifting
       |
       v
[Retraining Trigger] --> "Drift detected, auto-retraining initiated"
```

---

## APPROVAL CHECKPOINT

**Please review this plan and confirm:**

1. **Agenda timing** — Does the 8-hour flow with breaks work?
2. **Topics covered** — All requested topics are mapped to specific blocks
3. **Schema plan** — `main.mlops_genai_workshop` with GPU fleet use case
4. **Notebook sequence** — 7 notebooks (00-06)
5. **Datasets** — Synthetic GPU fleet data (consistent with Andrew's DGX theme)
6. **Slide outline** — 35 slides across all lecture blocks
7. **Two build paths** — Every hands-on block has prompt-driven AND code-driven options
8. **MLOps depth** — Feature store, model registry, serving, monitoring, drift detection, retraining

**What I'll do after your approval:**
1. Generate the synthetic datasets (Python scripts)
2. Write all 7 complete Databricks notebooks with runnable code
3. Create the setup SQL
4. Build the slide deck outline as a Google Doc
5. Test the complete flow against the workspace
