# Databricks notebook source

# MAGIC %md
# MAGIC # Workshop 3 — Agent Bricks: Due Diligence Agent (45 min)
# MAGIC
# MAGIC ## What We'll Build
# MAGIC A **Supervisor Agent** that combines a **Knowledge Assistant** (RAG over financial PDFs)
# MAGIC and a **Genie Space** (SQL over structured financials) to answer due diligence questions.
# MAGIC
# MAGIC ```
# MAGIC User: "Should we consider acquiring NVIDIA?"
# MAGIC
# MAGIC Supervisor Agent routes to:
# MAGIC   → Knowledge Assistant (agent): retrieves from 10-K, transcripts, earnings PDFs
# MAGIC   → Genie Space (agent): queries revenue, margins, cash, debt from Delta tables
# MAGIC   → Synthesizes a due diligence brief
# MAGIC ```
# MAGIC
# MAGIC ## Key Insight: RAG = Knowledge Assistant = Agent
# MAGIC In Agent Bricks, you **don't** manually build Vector Search, chunking, or retrieval.
# MAGIC A Knowledge Assistant IS a RAG agent — you point it at a Volume of PDFs and it handles:
# MAGIC - Document parsing and chunking
# MAGIC - Embedding and Vector Search index creation
# MAGIC - Retrieval and LLM-powered answering
# MAGIC
# MAGIC ## Steps
# MAGIC | Step | Time | What | API Call |
# MAGIC |------|------|------|----------|
# MAGIC | 1 | 5 min | Create Knowledge Assistant on PDFs | `manage_ka` |
# MAGIC | 2 | 5 min | Verify Genie Space from Notebook 2 | `get_genie` |
# MAGIC | 3 | 10 min | Create Supervisor Agent (KA + Genie) | `manage_mas` |
# MAGIC | 4 | 10 min | Wait for provisioning, test | Query the agent |
# MAGIC | 5 | 15 min | Hands-on: run due diligence queries | Interactive |
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Notebook 1 completed (PDFs uploaded to Volume)
# MAGIC - Notebook 2 completed (Genie Space created)

# COMMAND ----------

# DBTITLE 1,Config

CATALOG = "main"
SCHEMA = "fins_due_diligence"
VOLUME = "raw_filings"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# From Notebook 2 — replace with your Genie Space ID
GENIE_SPACE_ID = "01f136fb247a1ec39f3d22534bd406d1"  # Pre-built backup

# COMMAND ----------

# DBTITLE 1,Install dependencies

# MAGIC %pip install databricks-sdk -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-declare config

CATALOG = "main"
SCHEMA = "fins_due_diligence"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_filings"
GENIE_SPACE_ID = "01f136fb247a1ec39f3d22534bd406d1"  # Pre-built backup

DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create Knowledge Assistant
# MAGIC
# MAGIC This is the entire RAG pipeline in **one API call**. Agent Bricks will:
# MAGIC 1. Read all PDFs from the Volume
# MAGIC 2. Parse and chunk the documents
# MAGIC 3. Create embeddings and a Vector Search index
# MAGIC 4. Deploy a serving endpoint that retrieves and answers
# MAGIC
# MAGIC No manual chunking. No manual Vector Search setup. No LLM wiring.

# COMMAND ----------

# DBTITLE 1,1a — Create Knowledge Assistant (one call = entire RAG pipeline)

from databricks.sdk import WorkspaceClient
import requests, json

w = WorkspaceClient()

# The KA just needs: a name, the volume path, and instructions
KA_NAME = "Due Diligence Document Analyst"
KA_DESCRIPTION = "Answers questions about financial filings, earnings calls, and annual reports for 7 public companies (NVIDIA, Apple, Amazon, Google, Meta, Microsoft, Tesla)."
KA_INSTRUCTIONS = """You are a financial due diligence analyst.
You have access to SEC filings (10-K, 10-Q), earnings releases,
earnings call transcripts, and annual reports.

When answering:
- Cite the specific document and filing period for every fact
- Distinguish between historical data and forward-looking statements
- Flag material risks and uncertainties
- If comparing companies, present findings in a structured format
- If the answer is not in the documents, say so clearly
"""

# Create the KA via MCP tool (or API)
# Using the REST API directly for notebook compatibility
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Create KA tile
ka_payload = {
    "display_name": KA_NAME,
    "description": KA_DESCRIPTION,
    "knowledge_base": {
        "vector_store_config": {
            "data_sources": [{
                "type": "VOLUME",
                "volume_path": VOLUME_PATH,
            }]
        }
    },
    "instruction": KA_INSTRUCTIONS,
}

ka_resp = requests.post(
    f"https://{DATABRICKS_HOST}/api/2.0/agent-bricks/knowledge-assistants",
    headers=headers,
    json=ka_payload,
)

if ka_resp.status_code == 200:
    ka_data = ka_resp.json()
    KA_TILE_ID = ka_data.get("tile_id") or ka_data.get("id")
    print(f"Knowledge Assistant created!")
    print(f"  Name: {KA_NAME}")
    print(f"  Tile ID: {KA_TILE_ID}")
    print(f"  Volume: {VOLUME_PATH}")
    print(f"  Status: Provisioning... (2-5 minutes)")
else:
    print(f"Error: {ka_resp.status_code}")
    print(ka_resp.text[:500])
    print("\nIf the KA already exists, use the tile_id from the UI.")
    KA_TILE_ID = "YOUR_KA_TILE_ID"  # Fallback — paste from UI

# COMMAND ----------

# MAGIC %md
# MAGIC ### What just happened?
# MAGIC
# MAGIC That single API call triggered Agent Bricks to:
# MAGIC
# MAGIC ```
# MAGIC Volume: /Volumes/main/fins_due_diligence/raw_filings/
# MAGIC   ├── 10K/ (16 PDFs)           ──→  Parsed, chunked, embedded
# MAGIC   ├── 10Q/ (40 PDFs)           ──→  Vector Search index created
# MAGIC   ├── Earning Releases/ (76)   ──→  Serving endpoint deployed
# MAGIC   ├── Call Transcripts/ (25)   ──→  Ready for queries
# MAGIC   └── Annual Report/ (7)       ──→  All automatic
# MAGIC ```
# MAGIC
# MAGIC **Compare to the manual approach** (which would require):
# MAGIC - pypdf UDF to extract text
# MAGIC - Chunking function with overlap logic
# MAGIC - Delta table for chunks
# MAGIC - Vector Search endpoint creation
# MAGIC - Delta Sync index with embedding model
# MAGIC - LLM retrieval function
# MAGIC - Prompt engineering for answers
# MAGIC
# MAGIC Agent Bricks handles all of that in one call.

# COMMAND ----------

# DBTITLE 1,1b — Check KA provisioning status

import time

for attempt in range(10):
    ka_status = requests.get(
        f"https://{DATABRICKS_HOST}/api/2.0/agent-bricks/knowledge-assistants/{KA_TILE_ID}",
        headers=headers,
    )
    if ka_status.status_code == 200:
        status_data = ka_status.json()
        endpoint_status = status_data.get("endpoint_status", "UNKNOWN")
        print(f"  Attempt {attempt+1}: {endpoint_status}")
        if endpoint_status == "ONLINE":
            KA_ENDPOINT = status_data.get("endpoint_name", "")
            print(f"\n  Knowledge Assistant is ONLINE!")
            print(f"  Endpoint: {KA_ENDPOINT}")
            break
    time.sleep(30)
else:
    print("  Still provisioning — continue with other steps. It will be ready shortly.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Verify Genie Space
# MAGIC
# MAGIC We created the Genie Space in Notebook 2. Let's confirm it's ready.

# COMMAND ----------

# DBTITLE 1,2a — Verify Genie Space

genie_resp = requests.get(
    f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}",
    headers=headers,
)

if genie_resp.status_code == 200:
    genie_data = genie_resp.json()
    print(f"Genie Space: {genie_data.get('title', 'N/A')}")
    print(f"Space ID: {GENIE_SPACE_ID}")
    print(f"Tables: {len(genie_data.get('table_identifiers', []))}")
    print("Ready for Supervisor Agent!")
else:
    print(f"Genie Space not found ({genie_resp.status_code}).")
    print("Create it in Notebook 2 or via UI first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Create Supervisor Agent
# MAGIC
# MAGIC The Supervisor Agent orchestrates between the KA and Genie.
# MAGIC It routes questions based on intent:
# MAGIC - Numbers, metrics, comparisons → **Genie**
# MAGIC - Documents, risks, commentary → **Knowledge Assistant**
# MAGIC - Full due diligence → **Both**

# COMMAND ----------

# DBTITLE 1,3a — Create Supervisor Agent (KA + Genie)

MAS_NAME = "Due Diligence Supervisor"
MAS_DESCRIPTION = "Financial due diligence agent that combines structured financial data analysis with document-based research across 7 public companies."

MAS_INSTRUCTIONS = """You are a senior financial due diligence analyst evaluating companies
as potential investment or acquisition targets.

ROUTING RULES:
- Financial NUMBERS (revenue, EPS, margins, cash, debt, growth, rankings) → financial_analyst (Genie)
- DOCUMENTS and QUALITATIVE info (risk factors, management commentary, strategy,
  earnings call statements, 10-K disclosures, analyst concerns) → document_researcher (KA)
- Full due diligence assessment → call BOTH agents, then synthesize

When performing a full due diligence assessment, structure your response as:
1. **Financial Profile** — key metrics and trends (from Genie)
2. **Growth Drivers** — what's fueling the business (from documents)
3. **Risk Factors** — material risks from filings and transcripts (from documents)
4. **Management Signal** — confidence level from earnings calls (from documents)
5. **Assessment** — data-driven summary combining all findings

COMPANIES AVAILABLE: NVDA, AAPL, AMZN, GOOGL, META, MSFT, TSLA

Always cite the source of each data point (table or document)."""

mas_payload = {
    "display_name": MAS_NAME,
    "description": MAS_DESCRIPTION,
    "instruction": MAS_INSTRUCTIONS,
    "agents": [
        {
            "name": "document_researcher",
            "description": "Searches financial documents (10-K filings, 10-Q filings, earnings releases, earnings call transcripts, annual reports) to answer qualitative questions about risk factors, management commentary, strategic priorities, analyst concerns, and forward-looking guidance.",
            "ka_tile_id": KA_TILE_ID,
        },
        {
            "name": "financial_analyst",
            "description": "Queries structured financial data tables to answer quantitative questions about revenue, net income, EPS, gross margin, operating income, cash, debt, free cash flow, and revenue growth. Can compare across companies and show trends.",
            "genie_space_id": GENIE_SPACE_ID,
        },
    ],
    "examples": [
        {
            "question": "What was NVIDIA's revenue last quarter?",
            "guideline": "Route to financial_analyst — this is a metrics question."
        },
        {
            "question": "What are NVIDIA's top risk factors from the 10-K?",
            "guideline": "Route to document_researcher — this requires document retrieval."
        },
        {
            "question": "Perform a due diligence assessment of NVIDIA as an acquisition target.",
            "guideline": "Route to BOTH agents. Use financial_analyst for metrics, document_researcher for risks and commentary. Synthesize into a structured brief."
        },
    ],
}

mas_resp = requests.post(
    f"https://{DATABRICKS_HOST}/api/2.0/agent-bricks/supervisor-agents",
    headers=headers,
    json=mas_payload,
)

if mas_resp.status_code == 200:
    mas_data = mas_resp.json()
    MAS_TILE_ID = mas_data.get("tile_id") or mas_data.get("id")
    print(f"Supervisor Agent created!")
    print(f"  Name: {MAS_NAME}")
    print(f"  Tile ID: {MAS_TILE_ID}")
    print(f"  Agents: document_researcher (KA) + financial_analyst (Genie)")
    print(f"  Status: Provisioning...")
else:
    print(f"Error: {mas_resp.status_code}")
    print(mas_resp.text[:500])
    MAS_TILE_ID = "YOUR_MAS_TILE_ID"

# COMMAND ----------

# DBTITLE 1,3b — Wait for Supervisor Agent to come online

for attempt in range(10):
    mas_status = requests.get(
        f"https://{DATABRICKS_HOST}/api/2.0/agent-bricks/supervisor-agents/{MAS_TILE_ID}",
        headers=headers,
    )
    if mas_status.status_code == 200:
        status_data = mas_status.json()
        endpoint_status = status_data.get("endpoint_status", "UNKNOWN")
        print(f"  Attempt {attempt+1}: {endpoint_status}")
        if endpoint_status == "ONLINE":
            MAS_ENDPOINT = status_data.get("endpoint_name", "")
            print(f"\n  Supervisor Agent is ONLINE!")
            print(f"  Endpoint: {MAS_ENDPOINT}")
            break
    time.sleep(30)
else:
    print("  Still provisioning. Check the AI Playground in the UI to test once ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Test the Agent
# MAGIC
# MAGIC Query the Supervisor Agent endpoint. It will route to the right sub-agent.

# COMMAND ----------

# DBTITLE 1,4a — Helper: query the supervisor agent

def ask_agent(question: str, endpoint: str = None) -> str:
    """Send a question to the Supervisor Agent endpoint."""
    ep = endpoint or MAS_ENDPOINT
    resp = requests.post(
        f"https://{DATABRICKS_HOST}/serving-endpoints/{ep}/invocations",
        headers=headers,
        json={
            "input": [{"role": "user", "content": question}],
            "max_tokens": 2000,
        },
    )
    if resp.status_code == 200:
        output = resp.json().get("output", [])
        # Extract the final assistant response
        for item in reversed(output):
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text" and len(c.get("text", "")) > 100:
                            return c["text"]
                elif isinstance(content, str) and len(content) > 100:
                    return content
        return str(output)[:2000]
    return f"Error {resp.status_code}: {resp.text[:300]}"

# COMMAND ----------

# DBTITLE 1,4b — Test: Route to Genie (structured data)

answer = ask_agent(
    "Compare NVIDIA, Apple, and Microsoft on revenue, gross margin, and free cash flow."
)
print(answer)

# COMMAND ----------

# DBTITLE 1,4c — Test: Route to Knowledge Assistant (documents)

answer = ask_agent(
    "What are NVIDIA's top 3 risk factors from their most recent 10-K filing?"
)
print(answer)

# COMMAND ----------

# DBTITLE 1,4d — Test: Route to BOTH (full due diligence)

answer = ask_agent("""
Perform a due diligence assessment of NVIDIA as an acquisition target.
Include financial metrics, growth drivers, risk factors, and management
sentiment from the most recent earnings call.
""")
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Hands-On: Run Your Own Due Diligence Queries
# MAGIC
# MAGIC Try these in the **AI Playground** (UI) or by calling `ask_agent()` below.

# COMMAND ----------

# DBTITLE 1,5a — Your turn: pick a question and run it

# Uncomment one (or write your own) and run this cell:

# answer = ask_agent("Which of the 7 companies has the strongest balance sheet?")
# answer = ask_agent("What supply chain risks does NVIDIA disclose in their 10-K?")
# answer = ask_agent("Compare NVIDIA and Google as AI investment targets — financials and strategy.")
# answer = ask_agent("Summarize analyst concerns from NVIDIA's latest earnings call.")
# answer = ask_agent("If we were acquiring for AI infrastructure exposure, NVIDIA or Amazon?")
# answer = ask_agent("What is Tesla's debt-to-cash ratio and what risks does management highlight?")
# answer = ask_agent("Rank all 7 companies by revenue growth and flag any with declining margins.")

# print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## More Questions to Try
# MAGIC
# MAGIC **Single-company deep dive:**
# MAGIC - "Give me a complete financial profile of Apple including risks and growth outlook"
# MAGIC - "What did Microsoft's CFO say about cloud margins on the latest earnings call?"
# MAGIC
# MAGIC **Cross-company comparison:**
# MAGIC - "Compare Meta and Google on revenue, margins, and AI strategy commentary"
# MAGIC - "Which company has the lowest debt and highest free cash flow?"
# MAGIC
# MAGIC **Investment thesis:**
# MAGIC - "Build a bull case and bear case for investing in NVIDIA based on the filings"
# MAGIC - "What are the 3 biggest risks across all 7 companies?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Dev Kit Approach
# MAGIC
# MAGIC The entire agent stack can be built with three prompts in Claude Code:
# MAGIC
# MAGIC ```
# MAGIC Prompt 1: "Create a Knowledge Assistant called 'Due Diligence Document Analyst'
# MAGIC pointed at /Volumes/main/fins_due_diligence/raw_filings. It should answer
# MAGIC questions about financial filings, earnings calls, and annual reports."
# MAGIC
# MAGIC Prompt 2: "Create a Supervisor Agent called 'Due Diligence Supervisor' that
# MAGIC combines the Knowledge Assistant with the Genie Space. Route numbers questions
# MAGIC to Genie and document questions to the KA. For full due diligence assessments,
# MAGIC use both and structure as: Financial Profile, Growth Drivers, Risk Factors,
# MAGIC Management Signal, Assessment."
# MAGIC ```
# MAGIC
# MAGIC That's it. Two prompts = full due diligence agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────┐
# MAGIC │          Due Diligence Supervisor Agent          │
# MAGIC │            (Agent Bricks — MAS)                  │
# MAGIC │                                                  │
# MAGIC │  Routes: numbers → Genie | docs → KA | both     │
# MAGIC └──────────────────┬───────────────────────────────┘
# MAGIC                    │
# MAGIC        ┌───────────┴────────────┐
# MAGIC        ▼                        ▼
# MAGIC ┌──────────────┐    ┌─────────────────────────┐
# MAGIC │ Genie Space  │    │  Knowledge Assistant    │
# MAGIC │ (SQL agent)  │    │     (RAG agent)         │
# MAGIC │              │    │                         │
# MAGIC │ NL → SQL     │    │  Parses PDFs            │
# MAGIC │ over Delta   │    │  Chunks + embeds        │
# MAGIC │ tables       │    │  Vector Search          │
# MAGIC │              │    │  Retrieves + answers    │
# MAGIC └──────┬───────┘    └────────────┬────────────┘
# MAGIC        │                         │
# MAGIC        ▼                         ▼
# MAGIC  Delta Tables             Volume: raw_filings/
# MAGIC  ─ financials_clean       ─ 10K/ (16 PDFs)
# MAGIC  ─ transcript_insights    ─ 10Q/ (40 PDFs)
# MAGIC                           ─ Earning Releases/ (76)
# MAGIC                           ─ Call Transcripts/ (25)
# MAGIC                           ─ Annual Report/ (7)
# MAGIC ```
# MAGIC
# MAGIC **Total API calls to build this:** 3 (create KA, create Genie, create MAS)
# MAGIC **Total lines of infrastructure code:** 0 — Agent Bricks handles it all.
