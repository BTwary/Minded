---
title: Minded — Autonomous Analytical Intelligence OS
author: Minded
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
---

# Minded — Autonomous Analytical Intelligence OS
### Real Human Data Analyst Level AI — Relational & Deterministic (hosted demo)

> **Hosted demonstration — not private.** Files you upload are processed on the Hugging Face Space infrastructure. Do **not** upload confidential or sensitive data. For private data, use the local Windows application, which runs fully on your machine. This demo calls no external AI API.

This Space provides an interactive live demonstration of **Minded (AA-OS)**, an enterprise-grade autonomous data analyst operating with mathematical and statistical rigor.

It executes the **canonical Minded analytical kernel** (`apps.api.src.ai.runtime.InvestigationRuntime` -> `InvestigationController`). It does not run an ungrounded LLM chatbot and does not call external cloud APIs.

---

## 🌟 Key Capabilities Demonstrated

1. **Multi-Table Relational Question Solving:**
   - Evaluates complex questions across multiple related tables (e.g. `orders`, `customers`, `products`).
   - Discovers join paths, verifies grain safety to prevent fan-out multiplication, and aggregates dimensions autonomously.

2. **Enterprise Multi-Data-Type Ingestion:**
   - Coerces and sanitizes real-world enterprise messiness: accounting parentheses `(1,234.50)`, currency prefixes (`$`, `€`, `£`), missing tokens (`#N/A`, `nil`, `-999`), and European decimal formats.

3. **Human-Analyst Level Direct Answers:**
   - Produces numbers-first answers with exact metrics, percentage shares, ranked segment comparisons, and 95% confidence intervals.

4. **Prescriptive Strategic Playbook (Decision Recommendations):**
   - Automatically sizes opportunities, computes expected utility, models downside risks, win probabilities, and defines operational preconditions.

5. **Statistical Evidence & Epistemic Audit Trail:**
   - Formal hypothesis testing (Bayesian posteriors, p-values, Cohen's d / effect sizes) backed by tamper-proof cryptographic manifest hashes.

---

## 🚀 Interactive Demo Scenarios

Click any scenario button at the top of the interface:
- **🏢 Scenario 1: Multi-Table Relational (B2B SaaS)**
  - Tables: `customers.csv`, `orders.csv`, `products.csv`
  - Question: *"Which customer segment generated the highest revenue from the premium product category?"*
- **📊 Scenario 2: Messy Financials & Dirty Types**
  - Table: `quarterly_financials_messy.csv` (contains dirty numbers, currencies, accounting parentheses)
  - Question: *"What is the total net profit by division?"*
- **📈 Scenario 3: SaaS Retention & Churn Cohorts**
  - Table: `user_retention_cohorts.csv`
  - Question: *"What is the average monthly spend by tier?"*

Or upload your own CSV / Parquet files!

---

## 🔒 Security & Privacy Policy

- **Hosted demo, not a private environment:** Uploaded files are processed on the Hugging Face Space, so the Space host can technically access them. Do not upload confidential or sensitive enterprise data; use the local Windows application for private data.
- **No external AI:** This demo uses the deterministic analytical engine only and calls no external AI API or telemetry endpoint.
- **Temporary processing:** Uploads are handled in temporary storage for the current investigation and are not deliberately retained by the app.
- **Resource Limits:** Up to 4 datasets, 50 MB per file, 500,000 rows, 250 columns.
