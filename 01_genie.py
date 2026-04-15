# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Genie: Natural Language Analytics over Finance Data
# MAGIC
# MAGIC ## Overview
# MAGIC Genie lets anyone on the finance team ask plain English questions
# MAGIC about data in Delta tables — no SQL, no dashboards to configure.
# MAGIC
# MAGIC | Component | What it answers | Data source |
# MAGIC |-----------|----------------|-------------|
# MAGIC | **Genie** | "Show me Q3 revenue by segment" | Clean Delta tables |
# MAGIC
# MAGIC ## Tables available in this Genie Space
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `main.nvidia_workshop.pnl_clean` | Revenue, COGS, margins by segment/region/period |
# MAGIC | `main.nvidia_workshop.budget_vs_actual_raw` | Budget vs actual by cost center |
# MAGIC | `main.nvidia_workshop.treasury_loans_clean` | Loan facilities and status |
# MAGIC | `main.nvidia_workshop.customer_product_dim_raw` | Customer/product transactions |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create the Genie Space (UI walkthrough)
# MAGIC
# MAGIC 1. Go to **Databricks → Genie** in the left nav
# MAGIC 2. Click **New Genie Space**
# MAGIC 3. Name it: `NVIDIA Finance Assistant`
# MAGIC 4. Add tables: search for `main.nvidia_workshop.*`
# MAGIC 5. Paste the instructions from the next cell into the **Instructions** field
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

# DBTITLE 1,1b — Demo questions to run in your Genie Space

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
