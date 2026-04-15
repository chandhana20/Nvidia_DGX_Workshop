# Databricks notebook source

# MAGIC %md
# MAGIC # 03 — Genie Space + Knowledge Assistant + Supervisor Agent
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook wires together the three components of the NVIDIA Finance AI Assistant:
# MAGIC
# MAGIC ```
# MAGIC User Question
# MAGIC      │
# MAGIC      ▼
# MAGIC ┌─────────────────────────────────┐
# MAGIC │     Supervisor Agent            │
# MAGIC │  (routes based on intent)       │
# MAGIC └──────────┬──────────────────────┘
# MAGIC            │
# MAGIC    ┌───────┴────────┐
# MAGIC    ▼                ▼
# MAGIC ┌──────────┐  ┌──────────────────────┐
# MAGIC │  Genie   │  │  Knowledge Assistant  │
# MAGIC │(SQL/     │  │  (PDFs, emails,       │
# MAGIC │ tables)  │  │   transcripts)        │
# MAGIC └──────────┘  └──────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC | Component | What it answers | Data source |
# MAGIC |-----------|----------------|-------------|
# MAGIC | **Genie** | "Show me Q3 revenue by segment" | Clean Delta tables |
# MAGIC | **Knowledge Assistant** | "What were the CFO's margin comments?" | PDFs, emails, transcripts |
# MAGIC | **Supervisor** | Routes any question to the right tool | Both |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 — Genie Space Setup
# MAGIC
# MAGIC ### What is Genie?
# MAGIC Genie lets anyone ask natural language questions about data in Delta tables —
# MAGIC no SQL, no Python, no dashboards to configure.
# MAGIC
# MAGIC ### Tables we'll add to our Genie Space
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `main.nvidia_workshop.pnl_clean` | Revenue, COGS, margins by segment/region/period |
# MAGIC | `main.nvidia_workshop.budget_vs_actual_raw` | Budget vs actual by cost center |
# MAGIC | `main.nvidia_workshop.treasury_loans_clean` | Loan facilities and status |
# MAGIC | `main.nvidia_workshop.customer_product_dim_raw` | Customer/product transactions |
# MAGIC
# MAGIC ### How to create the Genie Space (UI walkthrough)
# MAGIC 1. Go to **Databricks → Genie** in the left nav
# MAGIC 2. Click **New Genie Space**
# MAGIC 3. Name it: `NVIDIA Finance Assistant`
# MAGIC 4. Add tables: search for `main.nvidia_workshop.*`
# MAGIC 5. Add instructions (see cell below)
# MAGIC 6. Click **Save and test**

# COMMAND ----------

# DBTITLE 1,1a — Genie Space instructions (paste into the UI)

GENIE_INSTRUCTIONS = """
You are a financial analyst assistant for NVIDIA Finance.

TABLES AVAILABLE:
- pnl_clean: Revenue, COGS, gross profit, OpEx, net income by business_segment, region, and period (YYYY-QN format). Revenue is in USD as a float.
- budget_vs_actual_raw: Budget vs actual spend by cost center and quarter. Variance can be negative (over budget) or positive (under budget).
- treasury_loans_clean: Active loan facilities with counterparty, loan type, outstanding balance, maturity date, and canonical loan_status.
- customer_product_dim_raw: Individual product transactions with customer_name, product_name, quantity, unit_price_USD.

BUSINESS SEGMENTS (canonical names):
- Data Center
- Gaming
- Automotive
- Professional Visualization

PERIOD FORMAT: All periods are YYYY-QN (e.g., "2024-Q1", "2024-Q2").

COMMON QUESTIONS TO HANDLE:
- Revenue trends by segment and region
- Budget vs actual variance analysis (which cost centers are over/under budget?)
- Loan exposure by counterparty, status, or currency
- Top customers by transaction value
- QoQ and YoY growth comparisons

IMPORTANT RULES:
- Always specify units in your answers (USD millions, %)
- If asked about China/export controls, note this data is US GAAP consolidated and does not show geographic breakdowns
- For budget questions, a NEGATIVE variance means over budget; POSITIVE means under budget
"""

print("Genie Space Instructions:")
print("-" * 60)
print(GENIE_INSTRUCTIONS)
print("-" * 60)
print("\nCopy the above into your Genie Space 'Instructions' field in the UI.")

# COMMAND ----------

# DBTITLE 1,1b — Test questions for your Genie Space demo

DEMO_QUESTIONS = [
    # Beginner questions (any accountant can ask these)
    "What was total Data Center revenue in 2024?",
    "Which business segment had the highest gross profit in Q4 2024?",
    "Show me all cost centers that are over budget this year.",
    "Which loans are currently in default or under review?",

    # Intermediate questions
    "Compare Q1 vs Q4 2024 revenue for Gaming across all regions.",
    "What is the total outstanding loan balance by counterparty?",
    "Which cost centers have the largest unfavorable variance this quarter?",
    "Show me top 10 customers by total transaction value.",

    # Power user questions
    "Calculate QoQ revenue growth rate by segment for 2024.",
    "What percentage of loans have covenant breaches?",
    "Which regions are underperforming relative to budget?",
    "Show me the trend in gross margin for Data Center over the last 6 quarters.",
]

print("Demo questions to run live in your Genie Space:\n")
for i, q in enumerate(DEMO_QUESTIONS, 1):
    prefix = "🟢" if i <= 4 else "🟡" if i <= 8 else "🔴"
    print(f"  {prefix} {i:>2}. {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2 — Knowledge Assistant Setup
# MAGIC
# MAGIC ### What is the Knowledge Assistant?
# MAGIC The Knowledge Assistant lets users chat with **unstructured documents** —
# MAGIC PDFs, emails, transcripts — using natural language.
# MAGIC Under the hood it uses Vector Search to find relevant chunks, then an LLM to answer.
# MAGIC
# MAGIC ### Documents we'll add
# MAGIC | Document | Use Case |
# MAGIC |----------|----------|
# MAGIC | `earnings_call_transcript_excerpt.txt` | IR teams, CFO office |
# MAGIC | `purchase_requests_emails.txt` | AP/procurement teams |
# MAGIC | NVIDIA 10-K PDF (upload separately) | All finance teams |

# COMMAND ----------

# DBTITLE 1,2a — Prepare documents table for Vector Search indexing

from pyspark.sql import functions as F

# Load the raw documents table we created in notebook 01
docs_df = spark.table("main.nvidia_workshop.unstructured_docs_raw")

# Chunk documents into 500-word segments for better retrieval
# (Vector Search works better with focused chunks than entire documents)
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word chunks for Vector Search."""
    if text is None:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

chunk_udf = F.udf(chunk_text, "array<string>")

docs_chunked = (
    docs_df
    .withColumn("chunks", chunk_udf(F.col("content")))
    .withColumn("chunk", F.explode(F.col("chunks")))
    .withColumn("chunk_id", F.monotonically_increasing_id())
    .select(
        F.col("chunk_id").cast("string").alias("id"),
        "filename",
        "chunk",
        F.current_timestamp().alias("indexed_at"),
    )
)

docs_chunked.write.mode("overwrite").saveAsTable("main.nvidia_workshop.docs_for_vector_search")
print(f"✓ Created {docs_chunked.count()} chunks for Vector Search indexing")
display(docs_chunked.limit(5))

# COMMAND ----------

# DBTITLE 1,2b — Create Vector Search endpoint and index (run once)

# MAGIC %sql
# MAGIC -- Step 1: Create the Vector Search endpoint (takes ~5 minutes first time)
# MAGIC -- In UI: Compute → Vector Search → Create Endpoint
# MAGIC -- Or via SQL:
# MAGIC
# MAGIC -- CREATE VECTOR SEARCH ENDPOINT nvidia_finance_vs_endpoint;

# COMMAND ----------

# Using Python SDK for Vector Search index creation
from databricks.vector_search.client import VectorSearchClient

VS_ENDPOINT   = "nvidia_finance_vs_endpoint"
SOURCE_TABLE  = "main.nvidia_workshop.docs_for_vector_search"
INDEX_NAME    = "main.nvidia_workshop.docs_vector_index"

vsc = VectorSearchClient()

# Create a Delta Sync index — automatically re-indexes when the source table changes
try:
    index = vsc.create_delta_sync_index(
        endpoint_name   = VS_ENDPOINT,
        index_name      = INDEX_NAME,
        source_table_name = SOURCE_TABLE,
        pipeline_type   = "TRIGGERED",           # or "CONTINUOUS" for real-time
        primary_key     = "id",
        embedding_source_column = "chunk",       # column to embed
        embedding_model_endpoint_name = "databricks-gte-large-en",  # built-in embedding model
    )
    print(f"✓ Vector Search index created: {INDEX_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"ℹ Index already exists: {INDEX_NAME}")
        index = vsc.get_index(VS_ENDPOINT, INDEX_NAME)
    else:
        raise e

# COMMAND ----------

# DBTITLE 1,2c — Test Vector Search retrieval

# Wait for index to be ready, then test a retrieval
results = index.similarity_search(
    query_text = "What was NVIDIA's gross margin guidance for Q1?",
    columns    = ["id", "filename", "chunk"],
    num_results = 3,
)

print("Top 3 chunks retrieved for: 'What was NVIDIA's gross margin guidance for Q1?'\n")
for i, r in enumerate(results.get("result", {}).get("data_array", []), 1):
    print(f"Result {i} — Source: {r[1]}")
    print(f"  {r[2][:300]}...")
    print()

# COMMAND ----------

# DBTITLE 1,2d — Knowledge Assistant UI setup instructions

KA_SETUP = """
KNOWLEDGE ASSISTANT SETUP (UI Steps):
======================================
1. Go to Databricks → Mosaic AI → Knowledge Assistants
2. Click "Create Knowledge Assistant"
3. Name: "NVIDIA Finance Knowledge Assistant"
4. Vector Search Index: main.nvidia_workshop.docs_vector_index
5. Instructions (paste below):
-------------------------------
You are a financial knowledge assistant for NVIDIA Finance.
You have access to earnings call transcripts, internal emails, and purchase requests.

When answering:
- Cite the source document for every fact
- For financial figures, always include the time period
- For emails, summarize the key action items
- If you cannot find the answer in the documents, say so clearly — do not guess

Common use cases:
- "What did the CFO say about gross margins on the earnings call?"
- "Summarize the open purchase requests for IT infrastructure"
- "What are the reconciliation items from the Q1 close email?"
- "What export control risks are mentioned?"
-------------------------------
6. Click Save → Test with sample questions below
"""

print(KA_SETUP)

# COMMAND ----------

# DBTITLE 1,2e — Test questions for Knowledge Assistant demo

KA_DEMO_QUESTIONS = [
    # Earnings call
    "What was NVIDIA's Q4 FY2024 revenue and how does it compare to the prior year?",
    "What did Jensen Huang say about Blackwell demand?",
    "What are the gross margin expectations for Q1?",
    "What did analysts ask about China export controls?",

    # Purchase requests
    "What are the open purchase requests and their total value?",
    "Who submitted the H100 purchase request and what cost center should it hit?",
    "Are there any shipping address discrepancies in the purchase requests?",
    "What invoices are currently unpaid according to the emails?",

    # Q1 close email
    "What are the reconciliation items for Q1 close?",
    "What is the Dell invoice discrepancy and how large is it?",
    "Which legacy Mellanox invoices are not yet migrated?",
    "When is the JPMorgan loan covenant certificate due?",
]

print("Knowledge Assistant demo questions:\n")
for i, q in enumerate(KA_DEMO_QUESTIONS, 1):
    source = "📊 Earnings" if i <= 4 else "📧 Emails" if i <= 8 else "📋 Q1 Close"
    print(f"  {source}  {i:>2}. {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3 — Supervisor Agent: Routing Genie + Knowledge Assistant
# MAGIC
# MAGIC The supervisor agent is the "front door" — it takes any finance question,
# MAGIC decides whether it needs structured data (Genie) or document lookup (KA),
# MAGIC and routes accordingly.
# MAGIC
# MAGIC ```
# MAGIC "What was Q3 revenue?"          → Genie  (structured query)
# MAGIC "What did the CFO say about X?" → KA     (document retrieval)
# MAGIC "Compare actual vs 10-K data"   → Both   (multi-step)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,3a — Define Genie tool for the agent

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Genie API client
import requests, json

GENIE_SPACE_ID = "YOUR_GENIE_SPACE_ID"   # replace after creating in UI
DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def query_genie(question: str) -> str:
    """
    Query the Genie Space with a natural language question.
    Returns the answer as a string.
    """
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    # Start conversation
    start_resp = requests.post(
        f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
        headers=headers,
        json={"content": question},
    )
    conversation = start_resp.json()
    conversation_id = conversation["conversation_id"]
    message_id      = conversation["message_id"]

    # Poll for result
    import time
    for _ in range(30):
        result_resp = requests.get(
            f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}"
            f"/conversations/{conversation_id}/messages/{message_id}",
            headers=headers,
        )
        result = result_resp.json()
        if result.get("status") == "COMPLETED":
            attachments = result.get("attachments", [])
            for att in attachments:
                if "text" in att:
                    return att["text"]["content"]
                if "query" in att:
                    return f"Query result: {att['query'].get('description', '')}"
            return "No result returned from Genie."
        time.sleep(2)
    return "Genie query timed out."

print("✓ Genie tool defined: query_genie(question)")

# COMMAND ----------

# DBTITLE 1,3b — Define Knowledge Assistant tool for the agent

def query_knowledge_assistant(question: str, num_results: int = 5) -> str:
    """
    Query the Vector Search index + LLM to answer questions from documents.
    Returns a cited answer string.
    """
    from databricks.vector_search.client import VectorSearchClient
    from databricks_genai import ChatClient

    # Step 1: Retrieve relevant chunks
    vsc   = VectorSearchClient()
    index = vsc.get_index(VS_ENDPOINT, INDEX_NAME)

    results = index.similarity_search(
        query_text  = question,
        columns     = ["id", "filename", "chunk"],
        num_results = num_results,
    )

    chunks = results.get("result", {}).get("data_array", [])
    if not chunks:
        return "No relevant documents found."

    # Build context string
    context = "\n\n".join([
        f"[Source: {r[1]}]\n{r[2]}"
        for r in chunks
    ])

    # Step 2: Ask the LLM with retrieved context
    llm = ChatClient(model="databricks-meta-llama-3-3-70b-instruct")
    prompt = f"""You are a financial analyst. Answer the question using only the context below.
Cite the source document for each fact you use. If the answer is not in the context, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = llm.chat(messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

print("✓ Knowledge Assistant tool defined: query_knowledge_assistant(question)")

# COMMAND ----------

# DBTITLE 1,3c — Build the Supervisor Agent with intent routing

from databricks_langchain import ChatDatabricks
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Wrap functions as LangChain tools
@tool
def genie_tool(question: str) -> str:
    """
    Use this tool when the user asks about numbers, metrics, trends, or anything
    that requires querying structured finance data tables.
    Examples: revenue, budget vs actual, loan balances, cost center spend,
    quarterly comparisons, top customers.
    """
    return query_genie(question)

@tool
def knowledge_assistant_tool(question: str) -> str:
    """
    Use this tool when the user asks about documents, reports, emails, transcripts,
    or anything qualitative.
    Examples: earnings call commentary, CFO statements, purchase request details,
    email reconciliation items, 10-K risk factors, analyst questions.
    """
    return query_knowledge_assistant(question)

# Supervisor LLM
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

# System prompt for the supervisor
SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial AI assistant for NVIDIA Finance.
You have two tools:
1. genie_tool — for structured data questions (numbers, tables, trends)
2. knowledge_assistant_tool — for document questions (reports, emails, transcripts)

ROUTING RULES:
- Numbers, metrics, trends, comparisons → genie_tool
- Documents, comments, qualitative insights, emails → knowledge_assistant_tool
- Complex questions needing both → call BOTH tools and synthesize the answer

Always be concise. Cite sources. If unsure which tool to use, try knowledge_assistant_tool first."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Build the agent
tools = [genie_tool, knowledge_assistant_tool]
agent = create_tool_calling_agent(llm, tools, SUPERVISOR_PROMPT)
supervisor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

print("✓ Supervisor agent ready.")
print("  Tools: genie_tool + knowledge_assistant_tool")
print("  Model: databricks-meta-llama-3-3-70b-instruct")

# COMMAND ----------

# DBTITLE 1,3d — Test the supervisor agent live

# Run each of these to show the routing in action during the demo
test_questions = [
    # Routes to Genie
    "What was the total Data Center revenue in Q4 2024?",

    # Routes to Knowledge Assistant
    "What did the CFO say about gross margin sustainability on the earnings call?",

    # Routes to both
    "How does NVIDIA's actual Q4 gross margin compare to what the CFO guided on the call?",

    # Practical finance use case
    "Summarize the open purchase requests and tell me the total budget impact.",
]

for q in test_questions[:1]:   # run one at a time during demo
    print(f"\n{'='*60}")
    print(f"QUESTION: {q}")
    print('='*60)
    result = supervisor.invoke({"input": q})
    print(f"\nANSWER:\n{result['output']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4 — Deploy as a Databricks App
# MAGIC
# MAGIC Once the supervisor works in the notebook, deploy it as a shareable web app
# MAGIC that any finance user can access — no Databricks account needed.

# COMMAND ----------

# DBTITLE 1,4a — app.py for Databricks Apps deployment

APP_CODE = '''
import gradio as gr
from agent import supervisor   # import from 03_genie_ka_supervisor_agent

def answer(question, history):
    result = supervisor.invoke({"input": question})
    return result["output"]

demo = gr.ChatInterface(
    fn=answer,
    title="NVIDIA Finance AI Assistant",
    description="Ask anything about NVIDIA Finance data — revenue, budgets, loans, or documents.",
    examples=[
        "What was Q4 2024 Data Center revenue?",
        "Which cost centers are over budget?",
        "What did the CFO say about margins on the earnings call?",
        "Summarize the open purchase requests",
        "Are there any loans in default?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
'''

# Write to app directory
import os
os.makedirs("/Workspace/Users/chandhana.padmanabhan@databricks.com/nvidia_finance_app", exist_ok=True)

with open("/Workspace/Users/chandhana.padmanabhan@databricks.com/nvidia_finance_app/app.py", "w") as f:
    f.write(APP_CODE)

print("✓ app.py written.")
print("\nTo deploy:")
print("  1. databricks apps create nvidia-finance-assistant")
print("  2. databricks apps deploy nvidia-finance-assistant \\")
print("       --source-code-path /Workspace/Users/chandhana.padmanabhan@databricks.com/nvidia_finance_app")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Full Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                  NVIDIA Finance AI Assistant                    │
# MAGIC │                   (Databricks App — Gradio UI)                  │
# MAGIC └───────────────────────────┬─────────────────────────────────────┘
# MAGIC                             │
# MAGIC                             ▼
# MAGIC                   ┌─────────────────┐
# MAGIC                   │ Supervisor Agent │  (LLaMA 3.3 70B)
# MAGIC                   │  intent routing  │
# MAGIC                   └────────┬────────┘
# MAGIC                            │
# MAGIC              ┌─────────────┴─────────────┐
# MAGIC              ▼                           ▼
# MAGIC    ┌──────────────────┐      ┌──────────────────────────┐
# MAGIC    │   Genie Space    │      │   Knowledge Assistant    │
# MAGIC    │  (NL → SQL)      │      │   (Vector Search + LLM)  │
# MAGIC    └────────┬─────────┘      └───────────┬──────────────┘
# MAGIC             │                            │
# MAGIC             ▼                            ▼
# MAGIC    ┌──────────────────┐      ┌──────────────────────────┐
# MAGIC    │  Delta Tables    │      │   Vector Search Index    │
# MAGIC    │  pnl_clean       │      │   (chunked docs)         │
# MAGIC    │  budget_*        │      │                          │
# MAGIC    │  treasury_*      │      │   earnings_call.txt      │
# MAGIC    │  customer_dim    │      │   purchase_emails.txt    │
# MAGIC    └────────┬─────────┘      │   NVIDIA 10-K.pdf        │
# MAGIC             │                └──────────────────────────┘
# MAGIC             ▼
# MAGIC    ┌──────────────────┐
# MAGIC    │  Unity Catalog   │  ← ingested via Auto Loader
# MAGIC    │  (main.nvidia_   │     Lakeflow Connect (SharePoint)
# MAGIC    │   workshop.*)    │     AI Dev Kit / Cursor
# MAGIC    └──────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Total build time in workshop:** ~90 minutes for Track 3 (Expert)
# MAGIC **What NVIDIA Finance gets:** A production-ready AI assistant over their own data.
