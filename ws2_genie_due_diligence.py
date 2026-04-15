# Databricks notebook source

# MAGIC %md
# MAGIC # Workshop 2 — Genie Space for Due Diligence (30 min)
# MAGIC
# MAGIC ## What We'll Build
# MAGIC A Genie Space that lets anyone on the finance team ask natural language
# MAGIC questions about target companies — no SQL required.
# MAGIC
# MAGIC **Example questions:**
# MAGIC - "Compare NVIDIA and Apple revenue over the last 4 quarters"
# MAGIC - "Which company has the strongest free cash flow?"
# MAGIC - "Show me gross margin trends for all 7 companies"
# MAGIC - "Rank companies by revenue growth rate"
# MAGIC
# MAGIC ## Steps
# MAGIC | Step | Time | What |
# MAGIC |------|------|------|
# MAGIC | 1 | 5 min | Create Genie Space via API |
# MAGIC | 2 | 5 min | Add tables and instructions |
# MAGIC | 3 | 10 min | Hands-on: ask due diligence questions |
# MAGIC | 4 | 10 min | Add custom SQL examples for complex queries |
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Notebook 1 completed (tables exist in `main.fins_due_diligence`)

# COMMAND ----------

# DBTITLE 1,Config

CATALOG = "main"
SCHEMA = "fins_due_diligence"

# Verify tables exist
tables = [r.tableName for r in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()]
print(f"Available tables: {tables}")

assert "company_financials_clean" in tables, "Run Notebook 1 first!"
assert "call_transcript_insights" in tables, "Run Notebook 1 first!"
print("All prerequisite tables found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create the Genie Space
# MAGIC
# MAGIC **Two approaches:**
# MAGIC - **UI (recommended for workshop):** Genie → New Genie Space
# MAGIC - **API (shown below):** Programmatic creation
# MAGIC
# MAGIC ### UI Walkthrough
# MAGIC 1. Left nav → **Genie**
# MAGIC 2. Click **New Genie Space**
# MAGIC 3. Name: `Due Diligence Analyst`
# MAGIC 4. Add tables (Step 2 below)
# MAGIC 5. Paste instructions (Step 2 below)
# MAGIC 6. Save and test

# COMMAND ----------

# DBTITLE 1,1a — Create Genie Space via API

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

GENIE_TITLE = "Due Diligence Analyst"
GENIE_DESCRIPTION = "Ask financial due diligence questions about target companies. Covers NVIDIA, Apple, Amazon, Google, Meta, Microsoft, and Tesla."

# Create the Genie Space
genie = w.genie.create(
    space_id=None,  # auto-generate
    title=GENIE_TITLE,
    description=GENIE_DESCRIPTION,
    table_identifiers=[
        f"{CATALOG}.{SCHEMA}.company_financials_clean",
        f"{CATALOG}.{SCHEMA}.call_transcript_insights",
        f"{CATALOG}.{SCHEMA}.parsed_filings",
    ],
)

GENIE_SPACE_ID = genie.space_id
print(f"Genie Space created: {GENIE_TITLE}")
print(f"Space ID: {GENIE_SPACE_ID}")
print(f"URL: Open Genie in the left nav and look for '{GENIE_TITLE}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configure Instructions and Table Descriptions
# MAGIC
# MAGIC Good instructions make Genie dramatically better. Paste these into your
# MAGIC Genie Space settings or set them via API.

# COMMAND ----------

# DBTITLE 1,2a — Genie Space instructions (paste into the UI)

GENIE_INSTRUCTIONS = """
You are a financial due diligence analyst. Your job is to help evaluate
whether a company is a good investment or acquisition target.

TABLES:
- company_financials_clean: Quarterly and annual financial metrics per company.
  Columns: ticker, filing_period (YYYY-QN or YYYY-FY), revenue, net_income,
  eps_diluted, gross_margin_pct, operating_income, cash_equivalents, total_debt,
  free_cash_flow, revenue_yoy_growth. All dollar values in USD.

- call_transcript_insights: AI-extracted signals from earnings call transcripts.
  Columns: ticker, filing_period, insights (JSON with management_sentiment,
  growth_drivers, key_risks, capex_outlook, ai_strategy, analyst_concerns).

- parsed_filings: Full text of all parsed documents.
  Columns: ticker, doc_type (10-K, 10-Q, Earnings Release, Earnings Call Transcript,
  Annual Report), filing_period, text_content.

COMPANIES: NVDA (NVIDIA), AAPL (Apple), AMZN (Amazon), GOOGL (Google/Alphabet),
META (Meta), MSFT (Microsoft), TSLA (Tesla)

DUE DILIGENCE FRAMEWORK:
When comparing companies, consider:
1. Revenue scale and growth trajectory
2. Profitability (gross margin, operating margin, net margin)
3. Cash generation (free cash flow, cash on hand)
4. Financial health (debt levels, current ratio)
5. Management confidence (from transcript sentiment)

RULES:
- Always show dollar amounts in billions (e.g., $39.3B not $39,331,000,000)
- When comparing companies, use tables or ranked lists
- If a value is NULL, say "not available" — do not guess
- For period comparisons, show the most recent quarters first
- When asked "should we invest", provide data-driven analysis — not a recommendation
"""

print("Genie Space Instructions (copy into the UI):")
print("=" * 60)
print(GENIE_INSTRUCTIONS)

# COMMAND ----------

# DBTITLE 1,2b — Add sample SQL queries to Genie Space

SAMPLE_QUERIES = {
    "Company revenue comparison": """
        SELECT ticker, filing_period, revenue
        FROM main.fins_due_diligence.company_financials_clean
        WHERE doc_type = '10-Q'
        ORDER BY filing_period DESC, revenue DESC
    """,

    "Peer gross margin ranking": """
        SELECT ticker,
               ROUND(AVG(gross_margin_pct), 1) AS avg_gross_margin,
               ROUND(AVG(revenue_yoy_growth), 1) AS avg_revenue_growth
        FROM main.fins_due_diligence.company_financials_clean
        WHERE filing_period >= '2024-Q1'
        GROUP BY ticker
        ORDER BY avg_gross_margin DESC
    """,

    "Financial health scorecard": """
        SELECT ticker,
               MAX(filing_period) AS latest_period,
               FIRST_VALUE(cash_equivalents) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS cash,
               FIRST_VALUE(total_debt) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS debt,
               FIRST_VALUE(free_cash_flow) OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS fcf
        FROM main.fins_due_diligence.company_financials_clean
        WHERE cash_equivalents IS NOT NULL
        GROUP BY ticker, filing_period, cash_equivalents, total_debt, free_cash_flow
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY filing_period DESC) = 1
        ORDER BY cash DESC
    """,

    "Management sentiment tracker": """
        SELECT ticker, filing_period,
               insights.management_sentiment_positive_neutral_negative AS sentiment,
               insights.top_growth_drivers_mentioned AS growth_drivers
        FROM main.fins_due_diligence.call_transcript_insights
        ORDER BY filing_period DESC, ticker
    """,
}

print("Sample SQL queries to add to Genie Space as examples:\n")
for name, sql in SAMPLE_QUERIES.items():
    print(f"--- {name} ---")
    print(sql.strip())
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Hands-On: Ask Due Diligence Questions
# MAGIC
# MAGIC Open the Genie Space and try these questions. They progress from simple
# MAGIC lookups to complex cross-company analysis.

# COMMAND ----------

# DBTITLE 1,3a — Demo questions by difficulty

DEMO_QUESTIONS = {
    "Beginner (single company lookup)": [
        "What was NVIDIA's revenue last quarter?",
        "Show me Apple's EPS over the last 4 quarters.",
        "What is Microsoft's gross margin?",
        "How much cash does Amazon have on hand?",
    ],
    "Intermediate (comparisons)": [
        "Compare revenue for NVIDIA, Apple, and Microsoft in the most recent quarter.",
        "Which company has the highest gross margin?",
        "Rank all 7 companies by free cash flow.",
        "Show quarterly revenue growth trends for NVIDIA and Tesla.",
    ],
    "Advanced (due diligence analysis)": [
        "Which company has the best combination of revenue growth and gross margin?",
        "Compare the debt-to-cash ratio across all companies.",
        "Show me companies where management sentiment was positive and revenue growth exceeded 20%.",
        "If I had to pick 3 companies with the strongest financial profile, which would they be?",
    ],
}

for level, questions in DEMO_QUESTIONS.items():
    print(f"\n{level}")
    print("-" * 50)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Verify Genie Answers with Direct SQL
# MAGIC
# MAGIC For the workshop, run these SQL queries to validate Genie's answers.
# MAGIC This teaches participants to **trust but verify** AI-generated queries.

# COMMAND ----------

# DBTITLE 1,4a — Revenue comparison: latest quarter

# MAGIC %sql
# MAGIC -- Verify: "Compare revenue across all companies"
# MAGIC WITH latest AS (
# MAGIC   SELECT ticker, filing_period, revenue,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS rn
# MAGIC   FROM main.fins_due_diligence.company_financials_clean
# MAGIC   WHERE revenue IS NOT NULL AND doc_type = '10-Q'
# MAGIC )
# MAGIC SELECT ticker, filing_period, ROUND(revenue / 1e9, 2) AS revenue_billions
# MAGIC FROM latest
# MAGIC WHERE rn = 1
# MAGIC ORDER BY revenue DESC;

# COMMAND ----------

# DBTITLE 1,4b — Gross margin ranking

# MAGIC %sql
# MAGIC -- Verify: "Which company has the highest gross margin?"
# MAGIC WITH latest AS (
# MAGIC   SELECT ticker, filing_period, gross_margin_pct,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS rn
# MAGIC   FROM main.fins_due_diligence.company_financials_clean
# MAGIC   WHERE gross_margin_pct IS NOT NULL
# MAGIC )
# MAGIC SELECT ticker, filing_period, ROUND(gross_margin_pct, 1) AS gross_margin_pct
# MAGIC FROM latest WHERE rn = 1
# MAGIC ORDER BY gross_margin_pct DESC;

# COMMAND ----------

# DBTITLE 1,4c — Financial health: cash vs debt

# MAGIC %sql
# MAGIC -- Verify: "Compare debt-to-cash ratio"
# MAGIC WITH latest AS (
# MAGIC   SELECT ticker, filing_period, cash_equivalents, total_debt, free_cash_flow,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY filing_period DESC) AS rn
# MAGIC   FROM main.fins_due_diligence.company_financials_clean
# MAGIC   WHERE cash_equivalents IS NOT NULL
# MAGIC )
# MAGIC SELECT ticker, filing_period,
# MAGIC        ROUND(cash_equivalents / 1e9, 1) AS cash_B,
# MAGIC        ROUND(total_debt / 1e9, 1) AS debt_B,
# MAGIC        ROUND(free_cash_flow / 1e9, 1) AS fcf_B,
# MAGIC        ROUND(CASE WHEN cash_equivalents > 0 THEN total_debt / cash_equivalents ELSE NULL END, 2) AS debt_to_cash
# MAGIC FROM latest WHERE rn = 1
# MAGIC ORDER BY debt_to_cash ASC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Dev Kit Approach
# MAGIC
# MAGIC To create this entire Genie Space with Claude Code:
# MAGIC ```
# MAGIC Prompt: "Create a Genie Space called 'Due Diligence Analyst' on the tables
# MAGIC in main.fins_due_diligence. Add instructions that frame it as a financial
# MAGIC due diligence tool for evaluating investment targets. Include sample SQL
# MAGIC for revenue comparison, margin ranking, and financial health scoring."
# MAGIC ```
# MAGIC
# MAGIC Claude Code will generate the API calls, instructions, and sample queries
# MAGIC in one shot.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Notebook 3 — Build an Agent Bricks Due Diligence Agent
