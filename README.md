# NVIDIA DGX Cloud Workshop: MLOps & Generative AI on Databricks

**Date:** April 15, 2026 | **Duration:** 9:00 AM - 5:00 PM

## Workshop Structure

| Time | Block | Type |
|------|-------|------|
| 9:00 - 9:20 | Setup & Data Exploration | Hands-on |
| 9:20 - 10:00 | ML Lecture (9 slides) | Lecture |
| 10:15 - 11:45 | ML Labs: Features, Training, Serving, Monitoring | Hands-on |
| 12:45 - 1:00 | GenAI Lecture (2 slides) | Lecture |
| 1:00 - 3:30 | AI Labs: AI Functions, Genie, RAG, Apps, Agents | Hands-on |
| 3:30 - 4:00 | E2E Demo + Wrap-Up | Demo |

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 00 | `00_setup_and_explore.py` | Schema creation, synthetic GPU fleet data (50K rows), data exploration |
| 01 | `01_genai_foundations.py` | ai_classify, ai_extract, ai_summarize, ai_query, Python-to-SQL generation |
| 02 | `02_genie_spaces.py` | Genie Space creation, Conversation API, MCP patterns |
| 03 | `03_vector_search_rag.py` | Vector Search, Knowledge Assistants, Multi-Agent Supervisors |
| 04 | `04_databricks_apps.py` | Streamlit GPU Fleet Monitor app deployment |
| 05 | `05_advanced_mlops.py` | Feature engineering, MLflow training, UC Model Registry, serving, monitoring |
| 06 | `06_end_to_end_demo.py` | Full E2E pipeline demo combining all components |

## Use Case

GPU anomaly detection on a synthetic NVIDIA DGX Cloud fleet:
- 30 clusters across AWS, Azure, GCP, Oracle
- A100, H100, H200 GPUs running LLM, Vision, RL, and Tabular workloads
- 50K telemetry rows, 2K health events, 5K ML job runs
- Schema: `main.mlops_genai_workshop`

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Claude Code + Databricks CLI authenticated
- SQL Warehouse running
- ML Runtime 15.4+ cluster (for MLflow notebooks)
