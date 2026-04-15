# NVIDIA DGX Cloud Workshop — Demo Script

**Date:** April 15, 2026 | **Instructor:** Chandhana Padmanabhan
**Workspace:** e2-demo-west | **Schema:** main.mlops_genai_workshop

---

## Pre-Workshop Checklist (Night Before)

- [ ] SQL warehouse `75fd8278393d07eb` is running
- [ ] Schema `main.mlops_genai_workshop` exists with 8 tables
- [ ] Docs volume has 5 GPU runbooks uploaded
- [ ] Genie Space "DGX Fleet Analytics" created (ID: `01f13870422d1b8c97e2d988c2367782`)
- [ ] Knowledge Assistant "GPU Operations Advisor" created via AgentBricks UI
- [ ] Multi-Agent Supervisor "DGX Operations Center" created via AgentBricks UI
- [ ] Vector Search endpoint `mlops-genai-workshop-vs` is ONLINE
- [ ] MLflow experiment exists: `/Users/chandhana.padmanabhan@databricks.com/gpu-anomaly-detection-workshop`
- [ ] Model `main.mlops_genai_workshop.gpu_anomaly_detector` v2 @Champion registered
- [ ] Start Shared Autoscaling Americas cluster (`0730-172948-runts698`)
- [ ] Test all notebooks end-to-end on the cluster
- [ ] Slide deck loaded and tested

## Morning Of (8:30 AM)

- [ ] Arrive 30 min early
- [ ] Verify warehouse is warm (run `SELECT 1`)
- [ ] Verify cluster is running
- [ ] Open slide deck in presenter mode
- [ ] Have fallback pre-built resources ready
- [ ] Share notebook repo link in Slack/chat

---

## BLOCK 1: Setup & Data Exploration (9:00 - 9:20)

**Notebook:** `00_setup_and_explore.py`

### Script:

> "Good morning everyone. Yesterday with Andrew and Esha you got your Databricks + Claude Code environments set up. Today we're going to build a complete MLOps and AI pipeline on your DGX Cloud fleet data."

**Demo Steps:**
1. Open the notebook, show the schema:
   ```sql
   USE CATALOG main;
   USE SCHEMA mlops_genai_workshop;
   SHOW TABLES;
   ```
2. Quick data exploration:
   ```sql
   SELECT * FROM cluster_inventory LIMIT 5;
   SELECT cloud_provider, COUNT(*) as clusters, SUM(gpu_count) as total_gpus
   FROM cluster_inventory GROUP BY 1;
   ```
3. Show GPU telemetry patterns:
   ```sql
   SELECT cluster_id, ROUND(AVG(temp_celsius),1) as avg_temp,
     ROUND(AVG(utilization_pct),1) as avg_util,
     SUM(error_count) as total_errors
   FROM gpu_telemetry GROUP BY 1 ORDER BY total_errors DESC LIMIT 10;
   ```
4. Show health events:
   ```sql
   SELECT event_type, COUNT(*) FROM gpu_health_events GROUP BY 1;
   SELECT * FROM gpu_health_events WHERE event_type = 'critical' LIMIT 3;
   ```

> "We have 30 clusters across 4 cloud providers, 50K telemetry readings, and 2K health events. Let's start by building the ML pipeline."

---

## BLOCK 2: ML Lecture (9:20 - 10:00)

**Slides 2-11 (9 ML slides)**

Walk through at ~4 min/slide:
- **Slide 3** (Why struggling): Hit the "87% fail" stat hard. Tie to NVIDIA: "You build the world's training hardware — let's fix the ops side."
- **Slide 4** (Lifecycle): Explain 3 layers. "This morning we build ModelOps. This afternoon we add the GenAI serving layer."
- **Slide 8** (CI/CD/CT/CM): This is the conceptual anchor. "CI is easy. CT + CM is where the money is."
- **Slide 10** (Money Slide): Walk through slowly. "Every step in this diagram maps to a cell in your notebook."
- **Slide 11** (GPU use case): Set the scene for hands-on.

---

## BREAK (10:00 - 10:15)

---

## BLOCK 3: Hands-On ML Lab 1 (10:15 - 11:00)

**Notebook:** `05_advanced_mlops.py` — Parts A + B

### Lab 1: Feature Engineering (20 min)

> "Let's build our feature table. We're going to aggregate raw GPU telemetry into hourly windows with statistical features."

**Demo Steps:**
1. Show the CREATE TABLE with PRIMARY KEY constraint:
   ```sql
   CREATE OR REPLACE TABLE gpu_health_features (
     gpu_id STRING NOT NULL,
     window_start TIMESTAMP NOT NULL,
     avg_temp DOUBLE, max_temp DOUBLE, ...
     CONSTRAINT pk PRIMARY KEY (gpu_id, window_start)
   ) USING DELTA
   ```
2. Populate features:
   ```sql
   INSERT INTO gpu_health_features
   SELECT gpu_id, date_trunc('HOUR', timestamp) as window_start,
     AVG(temp_celsius) as avg_temp, MAX(temp_celsius) as max_temp, ...
   FROM gpu_telemetry GROUP BY 1, 2
   ```
3. Show the result: "5,380 hourly feature rows. The PK constraint makes this a UC Feature Table."
4. Create anomaly labels + training dataset

**Prompt-driven alternative:**
> "Use Claude Code: 'Create an hourly feature table from gpu_telemetry with avg/max/min temp, utilization, power, error counts, and a PK constraint. Then create anomaly labels where high temp + errors = anomaly.'"

### Lab 2: Model Training (20 min)

> "Now let's train a model and register it in Unity Catalog."

**Demo Steps:**
1. Show MLflow experiment creation
2. Train GradientBoosting with class balancing:
   ```python
   with mlflow.start_run(run_name="gpu_anomaly_gbt_v1"):
       pipeline.fit(X_train, y_train)
       mlflow.sklearn.log_model(pipeline, "model")
   ```
3. Show MLflow UI: params, metrics, artifacts
4. Register model: `mlflow.register_model(uri, "main.mlops_genai_workshop.gpu_anomaly_detector")`
5. Set Champion alias
6. **Open the UC Model Registry in the browser** — show lineage back to feature table

---

## BLOCK 4: Hands-On ML Lab 2 (11:00 - 11:45)

**Notebook:** `05_advanced_mlops.py` — Parts C + D

### Lab 3: Model Serving (20 min)

> "Our model is registered. Let's deploy it as a REST endpoint."

**Demo Steps:**
1. Create serving endpoint (show the SDK code)
2. Wait for endpoint to be ready (~3-5 min)
3. Test with a sample that should be anomalous:
   ```python
   response = w.serving_endpoints.query(
     name="gpu-anomaly-detector",
     dataframe_records=[{"avg_temp": 91.0, "max_temp": 95.0, "error_count_1h": 5, ...}]
   )
   ```
4. Run batch inference: save 5,380 predictions to `model_predictions`
5. Show: "83 anomalies detected across the fleet"

### Lab 4: Monitoring + Drift (20 min)

> "The model is serving. But how do we know when it's going stale?"

**Demo Steps:**
1. Create Lakehouse Monitor on `model_predictions`
2. Show auto-generated profile_metrics and drift_metrics tables
3. Custom business metric:
   ```sql
   SELECT cluster_id, AVG(prediction) as anomaly_rate
   FROM model_predictions GROUP BY 1 HAVING anomaly_rate > 0.05
   ```
4. Discuss: "When JS distance > 0.2, a Databricks Workflow triggers retraining."

### ML Wrap-Up & Q&A (15 min)

> "Recap: We built the complete MLOps pipeline in 90 minutes. Feature table with PK constraint, model trained and tracked in MLflow, registered in UC with Champion alias, deployed to a serving endpoint, and monitored for drift. All governed by Unity Catalog."

---

## LUNCH (12:00 - 12:45)

---

## BLOCK 5: GenAI Lecture (12:45 - 1:00)

**Slides 22-23 (2 GenAI slides)**

> "This morning we built the ML backbone. Now let's add the intelligence layer — AI functions, Genie Spaces, and agents that make your data conversational."

- **Slide 22**: Mosaic AI platform overview, AI Functions in SQL
- **Slide 23**: Why GenAI on the Lakehouse? Data stays in place, SQL analysts become AI users.

---

## BLOCK 6: Hands-On AI Lab 1 (1:00 - 1:45)

**Notebooks:** `01_genai_foundations.py` + `02_genie_spaces.py`

### AI Functions in SQL (20 min)

> "Let's classify those 2,000 GPU health events without writing any Python."

**Demo Steps:**
1. ai_classify:
   ```sql
   SELECT event_id, description,
     ai_classify(description, ARRAY('hardware_failure', 'thermal_throttle',
       'memory_error', 'software_bug', 'network_issue')) as root_cause
   FROM gpu_health_events WHERE event_type = 'critical' LIMIT 5;
   ```
   **Money moment:** Show that it correctly identifies "thermal_throttle" vs "hardware_failure" vs "memory_error"

2. ai_extract:
   ```sql
   SELECT job_id, description,
     ai_extract(description, ARRAY('framework', 'model_name', 'dataset')) as extracted
   FROM ml_job_runs LIMIT 5;
   ```
   **Money moment:** It pulls "Llama-3 70B", "LoRA", "internal code corpus" from free text

3. ai_query for remediation:
   ```sql
   SELECT gpu_id, description,
     ai_query('databricks-meta-llama-3-3-70b-instruct',
       CONCAT('Analyze this GPU event and recommend action: ', description)
     ) as remediation
   FROM gpu_health_events WHERE event_type = 'critical' LIMIT 3;
   ```
   **Money moment:** "The LLM generates specific remediation steps — replace PSU, check cooling, migrate workloads."

### Genie Space (25 min)

> "Now let's make all of this queryable in plain English."

**Demo Steps:**
1. Show existing Genie Space "DGX Fleet Analytics" (pre-built)
2. Ask: "Which cluster has the most GPU errors this week?"
3. Ask: "Compare GPU utilization by cloud provider"
4. Ask: "Show me the top 5 most expensive ML jobs by GPU-hours"
5. Show the generated SQL for each answer
6. **API Demo:** Query Genie programmatically:
   ```python
   w.genie.start_conversation(space_id="01f13870...", content="Which GPUs have temp above 90?")
   ```
7. Discuss MCP pattern: "Every Genie Space is now a programmable tool for Claude Code"

---

## BREAK (2:00 - 2:15)

---

## BLOCK 7: Hands-On AI Lab 2 (2:15 - 3:30)

**Notebooks:** `03_vector_search_rag.py` + `04_databricks_apps.py`

### Knowledge Assistant (25 min)

> "Genie answers data questions. But what about document questions? 'What does the runbook say?'"

**Demo Steps:**
1. Show the GPU Operations Advisor KA (pre-built)
2. Test: "What should I do when GPU temperature exceeds 85C?"
   - Show source citations from Thermal Management Guide
3. Test: "How do I diagnose ECC memory errors?"
   - Show citations from ECC Troubleshooting Guide
4. Test guardrail: "What is IBM's stock price?" — should decline

### Multi-Agent Supervisor (15 min)

> "Now the magic: combine Genie + KA into one agent that routes intelligently."

**Demo Steps:**
1. Show DGX Operations Center MAS (pre-built)
2. Quantitative: "How many GPUs show thermal warnings?" -> routes to Genie
3. Qualitative: "What's the runbook for thermal throttling?" -> routes to KA
4. Complex: "We have thermal warnings on dgx-aws-03. How many GPUs are affected and what should we do?" -> routes to BOTH, synthesizes

### Databricks App (20 min)

> "Let's put a dashboard on this for the ops team."

**Demo Steps:**
1. Show the Streamlit app code (app.py)
2. Deploy: `databricks apps create gpu-fleet-monitor`
3. Open the URL — show KPI tiles, temperature chart, alerts table
4. Share URL: "Anyone in the org can see this, no Databricks account needed to USE it"

### MCP Patterns Discussion (15 min)

> "Every Genie Space, every KA, every serving endpoint — they're all API-accessible. That means Claude Code can use them as tools."

Show the MCP architecture: Claude Code -> MCP Server -> Genie API -> SQL Warehouse

---

## BLOCK 8: E2E Demo + Closing (3:30 - 4:00+)

### Live Demo: GPU Fleet Operations Pipeline (20 min)

> "Let's see everything work together, end to end."

**Demo Flow:**
1. Simulate a telemetry spike (INSERT new rows with high temps)
2. Show feature table updating
3. Query serving endpoint — "anomaly detected"
4. Genie: "5 GPUs on dgx-aws-03 showing anomalies. Avg temp: 91C"
5. KA: "Thermal Management Guide: reduce workload, check cooling fans"
6. Supervisor: "5 GPUs affected on dgx-aws-03. Runbook recommends reducing workload and checking cooling. Current utilization is 97% — suggest migrating 2 jobs to dgx-gcp-01 at 42% utilization."
7. Show monitoring: drift metrics
8. "If this drift persists, the retraining workflow triggers automatically."

### Wrap-Up (10 min)

> "Today we went from raw GPU telemetry to a production ML pipeline to an AI-powered operations center — all on one platform, all governed by Unity Catalog."

**Recap what we built:**
- 8 tables (73K+ rows)
- Feature table with PK constraint
- GPU anomaly detector @Champion in UC
- Model serving endpoint + monitoring
- AI Functions in SQL
- Genie Space + Knowledge Assistant + Supervisor Agent
- Databricks App

**Next Steps:**
- Deploy first agent to production (30 days)
- Close the monitoring + retraining loop (60 days)
- Scale to 3+ automated workflows (90 days)

> "Any questions?"

---

## Fallback Resource IDs

| Resource | ID |
|----------|-----|
| Schema | main.mlops_genai_workshop |
| Genie Space | 01f13870422d1b8c97e2d988c2367782 |
| VS Endpoint | mlops-genai-workshop-vs |
| MLflow Experiment | /Users/chandhana.padmanabhan@databricks.com/gpu-anomaly-detection-workshop |
| Model | main.mlops_genai_workshop.gpu_anomaly_detector v2 @Champion |
| SQL Warehouse | 75fd8278393d07eb |
| Cluster | 0730-172948-runts698 (Shared Autoscaling Americas) |
