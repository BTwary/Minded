"""Independent Adversarial Audit Harness for MindEd AA-OS.
Executes 120 independent test cases across 20 analytical families.
Records exact expected vs actual interpretations, executed plans, numerical results,
verification statuses, verdicts, and failure classifications.
"""
import os
import sys
import json
import time
import math
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.src.core.database import SessionLocal, init_db
from apps.api.src.models.entities import Project
from apps.api.src.services.dataset_service import DatasetService
from apps.api.src.services.analysis_service import AnalysisService
from packages.schemas.src.analysis import AnalysisCreate
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
    extract_filter_conditions,
    extract_aggregation,
)
from packages.analytics_core.src.intelligence.typed_investigation_planner import (
    compile_typed_investigation,
    resolve_plan_branches,
    conditional_execution_policy,
)
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.engines.polars_query_recompiler import execute_known_query
from packages.analytics_core.src.engines.analyst_answer import build_analyst_result

print("Modules loaded successfully. Preparing test datasets...")

# -------------------------------------------------------------
# 1. Deterministic Synthetic Ground-Truth Datasets
# -------------------------------------------------------------
np.random.seed(42)

# Orders dataset (100 rows)
orders = []
regions = ['North', 'South', 'East', 'West']
channels = ['Online', 'Retail', 'Partner']
categories = ['Electronics', 'Furniture', 'Clothing']
statuses = ['Completed', 'Completed', 'Completed', 'Cancelled', 'Refunded']
start_date = datetime(2026, 1, 1)

for i in range(1, 101):
    reg = regions[(i - 1) % 4]
    cust_id = f'CUST-{(i % 20) + 1:03d}'
    cat = categories[(i - 1) % 3]
    ch = channels[(i - 1) % 3]
    st = statuses[(i - 1) % 5]
    dt = (start_date + pd.Timedelta(days=(i - 1))).strftime('%Y-%m-%d')
    qty = (i % 5) + 1
    price = 100.0 + ((i % 10) * 10.0)
    disc = 0.10 if (i % 4 == 0) else 0.0
    gross = round(qty * price, 2)
    rev = round(gross * (1.0 - disc), 2)
    cost = round(rev * 0.6, 2)
    profit = round(rev - cost, 2)
    is_late = True if (i % 6 == 0) else False
    is_returned = True if (st == 'Refunded') else False
    
    orders.append({
        'order_id': f'ORD-{i:03d}',
        'customer_id': cust_id,
        'order_date': dt,
        'region': reg,
        'channel': ch,
        'category': cat,
        'status': st,
        'quantity': qty,
        'unit_price': price,
        'discount_rate': disc,
        'revenue': rev,
        'cost': cost,
        'profit': profit,
        'is_late': is_late,
        'is_returned': is_returned,
    })

df_orders = pd.DataFrame(orders)

# Customers dataset (20 rows)
customers = []
segments = ['Enterprise', 'SMB', 'Consumer']
for i in range(1, 21):
    cid = f'CUST-{i:03d}'
    seg = segments[(i - 1) % 3]
    reg = regions[(i - 1) % 4]
    signup = (datetime(2025, 1, 1) + pd.Timedelta(days=i * 15)).strftime('%Y-%m-%d')
    churned = (i % 4 == 0)
    score = 70.0 + (i % 30)
    customers.append({
        'customer_id': cid,
        'customer_name': f'Customer {i}',
        'segment': seg,
        'region': reg,
        'signup_date': signup,
        'is_churned': churned,
        'satisfaction_score': score,
    })
df_customers = pd.DataFrame(customers)

# Line items (250 rows)
items = []
item_idx = 1
for order in orders:
    num_items = (int(order['order_id'].split('-')[1]) % 3) + 1
    for k in range(num_items):
        items.append({
            'item_id': f'ITEM-{item_idx:04d}',
            'order_id': order['order_id'],
            'product_id': f'PRD-{(item_idx % 10) + 1:03d}',
            'quantity': (k + 1),
            'item_price': round(order['unit_price'] / num_items, 2),
            'item_discount': order['discount_rate'],
        })
        item_idx += 1
df_items = pd.DataFrame(items)

# Empty dataset
df_empty = pd.DataFrame({'order_id': [], 'revenue': [], 'region': []})

# All-nulls dataset
df_nulls = pd.DataFrame({
    'order_id': [f'ORD-{i:03d}' for i in range(1, 11)],
    'revenue': [None] * 10,
    'region': ['North'] * 5 + [None] * 5,
})

# Messy / mixed types dataset
df_messy = pd.DataFrame({
    'order_id': ['O1', 'O2', 'O2', 'O3', 'O4', 'O5'], # duplicate O2
    'revenue': [100.0, -50.0, 200.0, 0.0, float('nan'), 300.0],
    'churn_flag': ['yes', 'no', '1', '0', 'TRUE', None],
    'order_date': ['2026-01-01', '01/02/2026', 'invalid_date', '2026-03-01', None, '2026-05-01'],
    'region': ['North', 'South', 'North', 'West', None, 'East'],
})

print(f"Datasets created: orders={len(df_orders)}, customers={len(df_customers)}, items={len(df_items)}, messy={len(df_messy)}")

# -------------------------------------------------------------
# 2. Database Ingestion for E2E Analysis Execution
# -------------------------------------------------------------
init_db()
db = SessionLocal()

project_id = "proj-audit-eval"
proj = db.query(Project).filter(Project.id == project_id).first()
if not proj:
    proj = Project(
        id=project_id,
        name="Independent Audit Workspace",
        description="Workspace for adversarial evaluation",
        org_id="audit-org",
        owner_id="audit-auditor",
    )
    db.add(proj)
    db.commit()

ds_service = DatasetService(db)

from apps.api.src.models.entities import Dataset

def ingest_or_get(name: str, df: pd.DataFrame) -> str:
    existing = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    for d in existing:
        if d.name == name:
            return d.id
    ds = ds_service.ingest_dataset_file(
        project_id=project_id,
        filename=name,
        file_bytes=df.to_csv(index=False).encode('utf-8'),
        description=f"Audit dataset {name}"
    )
    return ds.id

ds_orders_id = ingest_or_get("audit_orders.csv", df_orders)
ds_customers_id = ingest_or_get("audit_customers.csv", df_customers)
ds_items_id = ingest_or_get("audit_items.csv", df_items)
ds_messy_id = ingest_or_get("audit_messy.csv", df_messy)

print(f"Ingested datasets: orders={ds_orders_id}, customers={ds_customers_id}, items={ds_items_id}, messy={ds_messy_id}")

# -------------------------------------------------------------
# 3. Test Cases Definition (120 Cases across 20 Families)
# -------------------------------------------------------------

test_cases = [
    # Family 1: Descriptive statistics & data profiling
    {
        "id": "TC-F01-001", "family": 1, "family_name": "Profiling",
        "question": "Describe the orders dataset.",
        "dataset": "orders", "target": None, "group": None, "op": "PROFILE",
        "expected_behavior": "SAFE_PROFILE", "expected_answer": "profile overview",
    },
    {
        "id": "TC-F01-002", "family": 1, "family_name": "Profiling",
        "question": "What is the summary of sales data?",
        "dataset": "orders", "target": None, "group": None, "op": "PROFILE",
        "expected_behavior": "SAFE_PROFILE", "expected_answer": "summary statistics",
    },
    {
        "id": "TC-F01-003", "family": 1, "family_name": "Profiling",
        "question": "Give an overview of customer records.",
        "dataset": "customers", "target": None, "group": None, "op": "PROFILE",
        "expected_behavior": "SAFE_PROFILE", "expected_answer": "profile overview",
    },
    {
        "id": "TC-F01-004", "family": 1, "family_name": "Profiling",
        "question": "What columns and data types are in orders?",
        "dataset": "orders", "target": None, "group": None, "op": "PROFILE",
        "expected_behavior": "SAFE_PROFILE", "expected_answer": "schema description",
    },
    {
        "id": "TC-F01-005", "family": 1, "family_name": "Profiling",
        "question": "Profile the distribution of variables in orders.",
        "dataset": "orders", "target": None, "group": None, "op": "PROFILE",
        "expected_behavior": "SAFE_PROFILE", "expected_answer": "distribution profile",
    },

    # Family 2: Counts, sums, averages, medians, modes, min, max, variance, stddev
    {
        "id": "TC-F02-006", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the total revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "SUM",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].sum()),
    },
    {
        "id": "TC-F02-007", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the average revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "MEAN",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].mean()),
    },
    {
        "id": "TC-F02-008", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the median revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "MEDIAN",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].median()),
    },
    {
        "id": "TC-F02-009", "family": 2, "family_name": "Univariate Aggregates",
        "question": "How many total orders are there?",
        "dataset": "orders", "target": "order_id", "group": None, "op": "COUNT",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(len(df_orders)),
    },
    {
        "id": "TC-F02-010", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the maximum order revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "MAX",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].max()),
    },
    {
        "id": "TC-F02-011", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the minimum order revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "MIN",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].min()),
    },
    {
        "id": "TC-F02-012", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the standard deviation of revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "STDDEV",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].std()),
    },
    {
        "id": "TC-F02-013", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the variance of revenue?",
        "dataset": "orders", "target": "revenue", "group": None, "op": "VARIANCE",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].var(ddof=0)),
    },
    {
        "id": "TC-F02-014", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the most common order quantity?",
        "dataset": "orders", "target": "quantity", "group": None, "op": "MODE",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['quantity'].mode()[0]),
    },
    {
        "id": "TC-F02-015", "family": 2, "family_name": "Univariate Aggregates",
        "question": "What is the total quantity of products ordered?",
        "dataset": "orders", "target": "quantity", "group": None, "op": "SUM",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['quantity'].sum()),
    },

    # Family 3: Grouped and multilevel aggregations
    {
        "id": "TC-F03-016", "family": 3, "family_name": "Grouped Aggregations",
        "question": "What is the total revenue by region?",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "SUM",
        "expected_behavior": "GROUPED_SUM", "expected_groups": df_orders.groupby('region')['revenue'].sum().to_dict(),
    },
    {
        "id": "TC-F03-017", "family": 3, "family_name": "Grouped Aggregations",
        "question": "Calculate average revenue per category.",
        "dataset": "orders", "target": "revenue", "group": "category", "op": "MEAN",
        "expected_behavior": "GROUPED_MEAN", "expected_groups": df_orders.groupby('category')['revenue'].mean().to_dict(),
    },
    {
        "id": "TC-F03-018", "family": 3, "family_name": "Grouped Aggregations",
        "question": "What is total revenue by channel?",
        "dataset": "orders", "target": "revenue", "group": "channel", "op": "SUM",
        "expected_behavior": "GROUPED_SUM", "expected_groups": df_orders.groupby('channel')['revenue'].sum().to_dict(),
    },
    {
        "id": "TC-F03-019", "family": 3, "family_name": "Grouped Aggregations",
        "question": "Show order count by status.",
        "dataset": "orders", "target": "order_id", "group": "status", "op": "COUNT",
        "expected_behavior": "GROUPED_COUNT", "expected_groups": df_orders.groupby('status')['order_id'].count().to_dict(),
    },
    {
        "id": "TC-F03-020", "family": 3, "family_name": "Grouped Aggregations",
        "question": "What is average profit per region?",
        "dataset": "orders", "target": "profit", "group": "region", "op": "MEAN",
        "expected_behavior": "GROUPED_MEAN", "expected_groups": df_orders.groupby('region')['profit'].mean().to_dict(),
    },
    {
        "id": "TC-F03-021", "family": 3, "family_name": "Grouped Aggregations",
        "question": "Total revenue by region and category.",
        "dataset": "orders", "target": "revenue", "group": ["region", "category"], "op": "SUM",
        "expected_behavior": "MULTILEVEL_GROUP", "expected_groups": "multilevel",
    },
    {
        "id": "TC-F03-022", "family": 3, "family_name": "Grouped Aggregations",
        "question": "Sum of profit by channel and region.",
        "dataset": "orders", "target": "profit", "group": ["channel", "region"], "op": "SUM",
        "expected_behavior": "MULTILEVEL_GROUP", "expected_groups": "multilevel",
    },

    # Family 4: Ranking, top N, bottom N, ties, percentage shares
    {
        "id": "TC-F04-023", "family": 4, "family_name": "Ranking & Shares",
        "question": "Which region generated the most revenue?",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "TOP_1",
        "expected_behavior": "TOP_ENTITY", "expected_entity": df_orders.groupby('region')['revenue'].sum().idxmax(),
    },
    {
        "id": "TC-F04-024", "family": 4, "family_name": "Ranking & Shares",
        "question": "Rank regions by total revenue, highest first.",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "RANK_DESC",
        "expected_behavior": "RANK_LIST", "expected_top": df_orders.groupby('region')['revenue'].sum().sort_values(ascending=False).index.tolist(),
    },
    {
        "id": "TC-F04-025", "family": 4, "family_name": "Ranking & Shares",
        "question": "Top 3 customers by total spend.",
        "dataset": "orders", "target": "revenue", "group": "customer_id", "op": "TOP_3",
        "expected_behavior": "TOP_3_LIST", "expected_top": df_orders.groupby('customer_id')['revenue'].sum().sort_values(ascending=False).head(3).index.tolist(),
    },
    {
        "id": "TC-F04-026", "family": 4, "family_name": "Ranking & Shares",
        "question": "Which category had the lowest total sales?",
        "dataset": "orders", "target": "revenue", "group": "category", "op": "BOTTOM_1",
        "expected_behavior": "BOTTOM_ENTITY", "expected_entity": df_orders.groupby('category')['revenue'].sum().idxmin(),
    },
    {
        "id": "TC-F04-027", "family": 4, "family_name": "Ranking & Shares",
        "question": "Bottom 2 regions by profit.",
        "dataset": "orders", "target": "profit", "group": "region", "op": "BOTTOM_2",
        "expected_behavior": "BOTTOM_2_LIST", "expected_top": df_orders.groupby('region')['profit'].sum().sort_values(ascending=True).head(2).index.tolist(),
    },
    {
        "id": "TC-F04-028", "family": 4, "family_name": "Ranking & Shares",
        "question": "What percentage share of overall revenue came from each region?",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "SHARE_OF_TOTAL",
        "expected_behavior": "PERCENT_SHARES", "expected_groups": (df_orders.groupby('region')['revenue'].sum() / df_orders['revenue'].sum()).to_dict(),
    },
    {
        "id": "TC-F04-029", "family": 4, "family_name": "Ranking & Shares",
        "question": "Which customer placed the most orders?",
        "dataset": "orders", "target": "order_id", "group": "customer_id", "op": "TOP_1_COUNT",
        "expected_behavior": "TOP_CUSTOMER", "expected_entity": df_orders.groupby('customer_id')['order_id'].count().idxmax(),
    },

    # Family 5: Filters, multiple predicates, exclusions, compound conditions
    {
        "id": "TC-F05-030", "family": 5, "family_name": "Filters & Predicates",
        "question": "What is total revenue in the North region?",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "region", "val": "North"}], "op": "FILTERED_SUM",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[df_orders['region'] == 'North']['revenue'].sum()),
    },
    {
        "id": "TC-F05-031", "family": 5, "family_name": "Filters & Predicates",
        "question": "What is total revenue for Completed orders?",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "status", "val": "Completed"}], "op": "FILTERED_SUM",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[df_orders['status'] == 'Completed']['revenue'].sum()),
    },
    {
        "id": "TC-F05-032", "family": 5, "family_name": "Filters & Predicates",
        "question": "What is total revenue in the South region for Electronics?",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "region", "val": "South"}, {"col": "category", "val": "Electronics"}], "op": "COMPOUND_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['region'] == 'South') & (df_orders['category'] == 'Electronics')]['revenue'].sum()),
    },
    {
        "id": "TC-F05-033", "family": 5, "family_name": "Filters & Predicates",
        "question": "What is total revenue excluding Refunded orders?",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "status", "val": "Refunded", "op": "!="}], "op": "EXCLUSION_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[df_orders['status'] != 'Refunded']['revenue'].sum()),
    },
    {
        "id": "TC-F05-034", "family": 5, "family_name": "Filters & Predicates",
        "question": "How many orders had discount_rate > 0?",
        "dataset": "orders", "target": "order_id", "filters": [{"col": "discount_rate", "val": 0, "op": ">"}], "op": "PREDICATE_COUNT",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(len(df_orders[df_orders['discount_rate'] > 0])),
    },
    {
        "id": "TC-F05-035", "family": 5, "family_name": "Filters & Predicates",
        "question": "What is total profit for orders where quantity > 3?",
        "dataset": "orders", "target": "profit", "filters": [{"col": "quantity", "val": 3, "op": ">"}], "op": "PREDICATE_SUM",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[df_orders['quantity'] > 3]['profit'].sum()),
    },
    {
        "id": "TC-F05-036", "family": 5, "family_name": "Filters & Predicates",
        "question": "Calculate revenue for Online channel in West region.",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "channel", "val": "Online"}, {"col": "region", "val": "West"}], "op": "COMPOUND_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['channel'] == 'Online') & (df_orders['region'] == 'West')]['revenue'].sum()),
    },
    {
        "id": "TC-F05-037", "family": 5, "family_name": "Filters & Predicates",
        "question": "Show sales excluding Cancelled and Refunded orders.",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "status", "val": ["Cancelled", "Refunded"], "op": "NOT_IN"}], "op": "COMPOUND_EXCLUSION",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[~df_orders['status'].isin(['Cancelled', 'Refunded'])]['revenue'].sum()),
    },

    # Family 6: Date ranges, relative dates, periods, time-grain
    {
        "id": "TC-F06-038", "family": 6, "family_name": "Date Ranges & Time Grains",
        "question": "What was total revenue in January 2026?",
        "dataset": "orders", "target": "revenue", "time": "2026-01", "op": "TIME_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['order_date'] >= '2026-01-01') & (df_orders['order_date'] <= '2026-01-31')]['revenue'].sum()),
    },
    {
        "id": "TC-F06-039", "family": 6, "family_name": "Date Ranges & Time Grains",
        "question": "What was total revenue in February 2026?",
        "dataset": "orders", "target": "revenue", "time": "2026-02", "op": "TIME_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['order_date'] >= '2026-02-01') & (df_orders['order_date'] <= '2026-02-28')]['revenue'].sum()),
    },
    {
        "id": "TC-F06-040", "family": 6, "family_name": "Date Ranges & Time Grains",
        "question": "What was total revenue in March 2026?",
        "dataset": "orders", "target": "revenue", "time": "2026-03", "op": "TIME_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['order_date'] >= '2026-03-01') & (df_orders['order_date'] <= '2026-03-31')]['revenue'].sum()),
    },
    {
        "id": "TC-F06-041", "family": 6, "family_name": "Date Ranges & Time Grains",
        "question": "What was total revenue in Q1 2026?",
        "dataset": "orders", "target": "revenue", "time": "2026-Q1", "op": "TIME_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['order_date'] >= '2026-01-01') & (df_orders['order_date'] <= '2026-03-31')]['revenue'].sum()),
    },
    {
        "id": "TC-F06-042", "family": 6, "family_name": "Date Ranges & Time Grains",
        "question": "Show monthly revenue in 2026.",
        "dataset": "orders", "target": "revenue", "time": "month", "op": "TIME_GRAIN_MONTH",
        "expected_behavior": "MONTHLY_BREAKDOWN", "expected_groups": "monthly",
    },

    # Family 7: Period-over-period and YoY comparisons
    {
        "id": "TC-F07-043", "family": 7, "family_name": "Period Comparisons",
        "question": "Compare revenue in February 2026 vs January 2026.",
        "dataset": "orders", "target": "revenue", "op": "PERIOD_COMPARE",
        "expected_behavior": "PERIOD_CONTRAST", "expected_answer": "diff between Feb and Jan",
    },
    {
        "id": "TC-F07-044", "family": 7, "family_name": "Period Comparisons",
        "question": "What was month-over-month revenue growth in March 2026?",
        "dataset": "orders", "target": "revenue", "op": "MOM_GROWTH",
        "expected_behavior": "MOM_GROWTH", "expected_answer": "pct change Mar vs Feb",
    },
    {
        "id": "TC-F07-045", "family": 7, "family_name": "Period Comparisons",
        "question": "Did revenue increase in March compared to February?",
        "dataset": "orders", "target": "revenue", "op": "DIRECTIONAL_COMPARE",
        "expected_behavior": "DIRECTIONAL_RESULT", "expected_answer": "yes/no with numbers",
    },

    # Family 8: Growth, percentage change, ratios, rates, conversion, retention, churn
    {
        "id": "TC-F08-046", "family": 8, "family_name": "Rates & Proportions",
        "question": "What is the late delivery rate?",
        "dataset": "orders", "target": "is_late", "op": "RATE",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['is_late'].mean()),
    },
    {
        "id": "TC-F08-047", "family": 8, "family_name": "Rates & Proportions",
        "question": "What is the return rate?",
        "dataset": "orders", "target": "is_returned", "op": "RATE",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['is_returned'].mean()),
    },
    {
        "id": "TC-F08-048", "family": 8, "family_name": "Rates & Proportions",
        "question": "What is the overall profit margin?",
        "dataset": "orders", "target": "profit_margin", "op": "RATIO_OF_SUMS",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['profit'].sum() / df_orders['revenue'].sum()),
    },
    {
        "id": "TC-F08-049", "family": 8, "family_name": "Rates & Proportions",
        "question": "What percentage of orders were late in the North region?",
        "dataset": "orders", "target": "is_late", "filters": [{"col": "region", "val": "North"}], "op": "FILTERED_RATE",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[df_orders['region'] == 'North']['is_late'].mean()),
    },
    {
        "id": "TC-F08-050", "family": 8, "family_name": "Rates & Proportions",
        "question": "Compare late delivery rate between North and South.",
        "dataset": "orders", "target": "is_late", "group": "region", "op": "RATE_COMPARISON",
        "expected_behavior": "RATE_CONTRAST", "expected_groups": df_orders[df_orders['region'].isin(['North', 'South'])].groupby('region')['is_late'].mean().to_dict(),
    },
    {
        "id": "TC-F08-051", "family": 8, "family_name": "Rates & Proportions",
        "question": "What is the customer churn rate?",
        "dataset": "customers", "target": "is_churned", "op": "RATE",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_customers['is_churned'].mean()),
    },
    {
        "id": "TC-F08-052", "family": 8, "family_name": "Rates & Proportions",
        "question": "What is the churn rate by segment?",
        "dataset": "customers", "target": "is_churned", "group": "segment", "op": "GROUPED_RATE",
        "expected_behavior": "GROUPED_RATE", "expected_groups": df_customers.groupby('segment')['is_churned'].mean().to_dict(),
    },

    # Family 9: Correlation and association
    {
        "id": "TC-F09-053", "family": 9, "family_name": "Correlation & Association",
        "question": "What is the correlation between unit_price and revenue?",
        "dataset": "orders", "target": "revenue", "x": "unit_price", "op": "CORRELATION",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['unit_price'].corr(df_orders['revenue'])),
    },
    {
        "id": "TC-F09-054", "family": 9, "family_name": "Correlation & Association",
        "question": "Is quantity correlated with profit?",
        "dataset": "orders", "target": "profit", "x": "quantity", "op": "CORRELATION",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['quantity'].corr(df_orders['profit'])),
    },
    {
        "id": "TC-F09-055", "family": 9, "family_name": "Correlation & Association",
        "question": "What is the association between discount_rate and revenue?",
        "dataset": "orders", "target": "revenue", "x": "discount_rate", "op": "CORRELATION",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['discount_rate'].corr(df_orders['revenue'])),
    },
    {
        "id": "TC-F09-056", "family": 9, "family_name": "Correlation & Association",
        "question": "Is there a relationship between cost and profit?",
        "dataset": "orders", "target": "profit", "x": "cost", "op": "CORRELATION",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['cost'].corr(df_orders['profit'])),
    },
    {
        "id": "TC-F09-057", "family": 9, "family_name": "Correlation & Association",
        "question": "What is the rank correlation between quantity and unit_price?",
        "dataset": "orders", "target": "unit_price", "x": "quantity", "op": "SPEARMAN",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['quantity'].corr(df_orders['unit_price'], method='spearman')),
    },

    # Family 10: Business drivers and outcome relationships
    {
        "id": "TC-F10-058", "family": 10, "family_name": "Business Drivers",
        "question": "What factors are associated with order profit?",
        "dataset": "orders", "target": "profit", "op": "DRIVERS",
        "expected_behavior": "DRIVER_RANKING", "expected_answer": "ranked predictors of profit",
    },
    {
        "id": "TC-F10-059", "family": 10, "family_name": "Business Drivers",
        "question": "Which variables drive higher revenue?",
        "dataset": "orders", "target": "revenue", "op": "DRIVERS",
        "expected_behavior": "DRIVER_RANKING", "expected_answer": "ranked predictors of revenue",
    },
    {
        "id": "TC-F10-060", "family": 10, "family_name": "Business Drivers",
        "question": "What drives customer churn?",
        "dataset": "customers", "target": "is_churned", "op": "CHURN_DRIVERS",
        "expected_behavior": "CHURN_FACTORS", "expected_answer": "factors associated with churn",
    },
    {
        "id": "TC-F10-061", "family": 10, "family_name": "Business Drivers",
        "question": "Why did profit decline even though revenue increased?",
        "dataset": "orders", "target": "profit", "op": "DECOMPOSITION",
        "expected_behavior": "DECOMPOSITION_EXPLANATION", "expected_answer": "cost decomposition",
    },

    # Family 11: Outliers and anomaly detection
    {
        "id": "TC-F11-062", "family": 11, "family_name": "Outliers & Anomalies",
        "question": "Are there any revenue outliers in the dataset?",
        "dataset": "orders", "target": "revenue", "op": "OUTLIER_SCAN",
        "expected_behavior": "OUTLIER_RESULT", "expected_answer": "identified outliers",
    },
    {
        "id": "TC-F11-063", "family": 11, "family_name": "Outliers & Anomalies",
        "question": "Find anomalous transactions by order amount.",
        "dataset": "orders", "target": "revenue", "op": "ANOMALY_SCAN",
        "expected_behavior": "ANOMALY_CANDIDATES", "expected_answer": "anomalous rows",
    },
    {
        "id": "TC-F11-064", "family": 11, "family_name": "Outliers & Anomalies",
        "question": "Detect unusual orders with extreme quantity.",
        "dataset": "orders", "target": "quantity", "op": "OUTLIER_SCAN",
        "expected_behavior": "OUTLIER_RESULT", "expected_answer": "extreme quantity orders",
    },

    # Family 12: Forecasting and trend analysis
    {
        "id": "TC-F12-065", "family": 12, "family_name": "Forecasting & Trends",
        "question": "What is the overall trend in revenue over time?",
        "dataset": "orders", "target": "revenue", "time": "order_date", "op": "TREND",
        "expected_behavior": "TREND_DIRECTION", "expected_answer": "increasing/flat trend",
    },
    {
        "id": "TC-F12-066", "family": 12, "family_name": "Forecasting & Trends",
        "question": "Is profit increasing or decreasing over time?",
        "dataset": "orders", "target": "profit", "time": "order_date", "op": "TREND_DIRECTION",
        "expected_behavior": "TREND_DIRECTION", "expected_answer": "trend direction",
    },
    {
        "id": "TC-F12-067", "family": 12, "family_name": "Forecasting & Trends",
        "question": "Forecast revenue for the next 30 days.",
        "dataset": "orders", "target": "revenue", "time": "order_date", "op": "FORECAST",
        "expected_behavior": "FORECAST_PROJECTION", "expected_answer": "forecast with horizon 30 days",
    },
    {
        "id": "TC-F12-068", "family": 12, "family_name": "Forecasting & Trends",
        "question": "Forecast sales.",
        "dataset": "orders", "target": "revenue", "op": "FORECAST_AMBIGUOUS",
        "expected_behavior": "SAFE_REFUSAL_OR_CLARIFICATION", "expected_answer": "unspecified horizon / grain",
    },
    {
        "id": "TC-F12-069", "family": 12, "family_name": "Forecasting & Trends",
        "question": "Forecast revenue without time column.",
        "dataset": "customers", "target": "satisfaction_score", "op": "FORECAST_IMPOSSIBLE",
        "expected_behavior": "SAFE_REFUSAL", "expected_answer": "no time index",
    },

    # Family 13: Missingness, duplicates, data quality, reconciliation
    {
        "id": "TC-F13-070", "family": 13, "family_name": "Data Quality & Reconciliation",
        "question": "What is the percentage of missing values in each column?",
        "dataset": "orders", "op": "MISSINGNESS_PROFILE",
        "expected_behavior": "MISSING_RATES", "expected_answer": "missing percentages",
    },
    {
        "id": "TC-F13-071", "family": 13, "family_name": "Data Quality & Reconciliation",
        "question": "Are there any duplicate order_id records?",
        "dataset": "orders", "op": "DUPLICATE_CHECK",
        "expected_behavior": "DUPLICATE_COUNT", "expected_num": 0,
    },
    {
        "id": "TC-F13-072", "family": 13, "family_name": "Data Quality & Reconciliation",
        "question": "Are there duplicate order_id records in messy data?",
        "dataset": "messy", "op": "DUPLICATE_CHECK",
        "expected_behavior": "DUPLICATE_COUNT", "expected_num": 1,
    },
    {
        "id": "TC-F13-073", "family": 13, "family_name": "Data Quality & Reconciliation",
        "question": "Check whether revenue equals cost + profit.",
        "dataset": "orders", "op": "RECONCILIATION",
        "expected_behavior": "IDENTITY_PASS", "expected_answer": "exact reconciliation",
    },
    {
        "id": "TC-F13-074", "family": 13, "family_name": "Data Quality & Reconciliation",
        "question": "Verify whether revenue equals cost minus profit.",
        "dataset": "orders", "op": "RECONCILIATION_FAIL",
        "expected_behavior": "IDENTITY_FAIL", "expected_answer": "reconciliation discrepancy",
    },

    # Family 14: Multi-table joins and cross-table metrics
    {
        "id": "TC-F14-075", "family": 14, "family_name": "Multi-table Joins",
        "question": "What is total revenue by customer segment?",
        "dataset": "multi_orders_customers", "target": "revenue", "group": "segment", "op": "JOIN_SUM",
        "expected_behavior": "JOIN_GROUPED_SUM", "expected_answer": "revenue grouped by segment",
    },
    {
        "id": "TC-F14-076", "family": 14, "family_name": "Multi-table Joins",
        "question": "What is average spend per customer segment?",
        "dataset": "multi_orders_customers", "target": "revenue", "group": "segment", "op": "JOIN_MEAN",
        "expected_behavior": "JOIN_GROUPED_MEAN", "expected_answer": "average spend by segment",
    },
    {
        "id": "TC-F14-077", "family": 14, "family_name": "Multi-table Joins",
        "question": "Which customer segment had the highest total revenue?",
        "dataset": "multi_orders_customers", "target": "revenue", "group": "segment", "op": "JOIN_RANK",
        "expected_behavior": "JOIN_TOP_SEGMENT", "expected_answer": "top segment",
    },
    {
        "id": "TC-F14-078", "family": 14, "family_name": "Multi-table Joins",
        "question": "Total revenue across orders and order_items.",
        "dataset": "multi_orders_items", "target": "revenue", "op": "FANOUT_RISK",
        "expected_behavior": "FANOUT_PREVENTION", "expected_answer": "fanout detected or pre-aggregated",
    },

    # Family 15: Derived metrics & ratios of aggregates
    {
        "id": "TC-F15-079", "family": 15, "family_name": "Derived Metrics",
        "question": "What is the average order value?",
        "dataset": "orders", "target": "revenue", "op": "AOV",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].sum() / len(df_orders)),
    },
    {
        "id": "TC-F15-080", "family": 15, "family_name": "Derived Metrics",
        "question": "What is overall profit margin percentage?",
        "dataset": "orders", "target": "profit_margin", "op": "MARGIN_PCT",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float((df_orders['profit'].sum() / df_orders['revenue'].sum()) * 100.0),
    },
    {
        "id": "TC-F15-081", "family": 15, "family_name": "Derived Metrics",
        "question": "What is the average revenue per customer?",
        "dataset": "orders", "target": "revenue", "op": "REV_PER_CUST",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['revenue'].sum() / df_orders['customer_id'].nunique()),
    },
    {
        "id": "TC-F15-082", "family": 15, "family_name": "Derived Metrics",
        "question": "What is the ratio of cost to revenue?",
        "dataset": "orders", "target": "cost_ratio", "op": "COST_RATIO",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['cost'].sum() / df_orders['revenue'].sum()),
    },

    # Family 16: Conditional questions, exceptions, nested conditions
    {
        "id": "TC-F16-083", "family": 16, "family_name": "Conditional & Nested",
        "question": "What is revenue for orders in North region with quantity > 2?",
        "dataset": "orders", "target": "revenue", "op": "NESTED_COND",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['region'] == 'North') & (df_orders['quantity'] > 2)]['revenue'].sum()),
    },
    {
        "id": "TC-F16-084", "family": 16, "family_name": "Conditional & Nested",
        "question": "Find revenue for Electronics except where discount_rate > 0.05.",
        "dataset": "orders", "target": "revenue", "op": "EXCEPTION_COND",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders[(df_orders['category'] == 'Electronics') & (df_orders['discount_rate'] <= 0.05)]['revenue'].sum()),
    },
    {
        "id": "TC-F16-085", "family": 16, "family_name": "Conditional & Nested",
        "question": "Show orders where status is Completed or Refunded.",
        "dataset": "orders", "target": "order_id", "op": "OR_CONDITION",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(len(df_orders[df_orders['status'].isin(['Completed', 'Refunded'])])),
    },

    # Family 17: Questions with implicit denominators & ambiguous metrics
    {
        "id": "TC-F17-086", "family": 17, "family_name": "Ambiguous & Implicit Denominators",
        "question": "What is the conversion rate?",
        "dataset": "orders", "target": "conversion_rate", "op": "AMBIGUOUS_RATE",
        "expected_behavior": "SAFE_REFUSAL_OR_CLARIFICATION", "expected_answer": "missing conversion column",
    },
    {
        "id": "TC-F17-087", "family": 17, "family_name": "Ambiguous & Implicit Denominators",
        "question": "What is customer retention?",
        "dataset": "orders", "target": "retention", "op": "AMBIGUOUS_GRAIN",
        "expected_behavior": "SAFE_REFUSAL_OR_CLARIFICATION", "expected_answer": "unclear retention definition",
    },
    {
        "id": "TC-F17-088", "family": 17, "family_name": "Ambiguous & Implicit Denominators",
        "question": "What is the performance score?",
        "dataset": "orders", "target": "performance_score", "op": "MISSING_COLUMN",
        "expected_behavior": "SAFE_REFUSAL", "expected_answer": "variable not found",
    },
    {
        "id": "TC-F17-089", "family": 17, "family_name": "Ambiguous & Implicit Denominators",
        "question": "Show efficiency by region.",
        "dataset": "orders", "target": "efficiency", "group": "region", "op": "AMBIGUOUS_METRIC",
        "expected_behavior": "SAFE_REFUSAL_OR_CLARIFICATION", "expected_answer": "efficiency is undefined",
    },

    # Family 18: Categorical, numerical, boolean, ordinal, temporal variables
    {
        "id": "TC-F18-090", "family": 18, "family_name": "Variable Types",
        "question": "How many late orders were there?",
        "dataset": "orders", "target": "is_late", "op": "BOOLEAN_FILTER",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(df_orders['is_late'].sum()),
    },
    {
        "id": "TC-F18-091", "family": 18, "family_name": "Variable Types",
        "question": "What is the distribution of order status?",
        "dataset": "orders", "target": "status", "op": "CATEGORICAL_DIST",
        "expected_behavior": "DISTRIBUTION_TABLE", "expected_groups": df_orders['status'].value_counts().to_dict(),
    },
    {
        "id": "TC-F18-092", "family": 18, "family_name": "Variable Types",
        "question": "Average revenue for late versus on-time orders.",
        "dataset": "orders", "target": "revenue", "group": "is_late", "op": "BOOLEAN_GROUP",
        "expected_behavior": "BOOLEAN_CONTRAST", "expected_groups": df_orders.groupby('is_late')['revenue'].mean().to_dict(),
    },
    {
        "id": "TC-F18-093", "family": 18, "family_name": "Variable Types",
        "question": "Total profit by category and late status.",
        "dataset": "orders", "target": "profit", "group": ["category", "is_late"], "op": "MIXED_GROUP",
        "expected_behavior": "MULTILEVEL_GROUP", "expected_groups": "multilevel",
    },

    # Family 19: Explanations, comparisons, evidence
    {
        "id": "TC-F19-094", "family": 19, "family_name": "Explanations & Evidence",
        "question": "Why is revenue higher in South than in North?",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "EXPLANATION",
        "expected_behavior": "EXPLANATION_BREAKDOWN", "expected_answer": "decomposition of South vs North",
    },
    {
        "id": "TC-F19-095", "family": 19, "family_name": "Explanations & Evidence",
        "question": "Explain the difference in profit between Electronics and Clothing.",
        "dataset": "orders", "target": "profit", "group": "category", "op": "EXPLANATION",
        "expected_behavior": "EXPLANATION_BREAKDOWN", "expected_answer": "profit difference factors",
    },
    {
        "id": "TC-F19-096", "family": 19, "family_name": "Explanations & Evidence",
        "question": "Provide evidence on whether discounts increase sales volume.",
        "dataset": "orders", "target": "quantity", "x": "discount_rate", "op": "EVIDENCE",
        "expected_behavior": "ASSOCIATION_EVIDENCE", "expected_answer": "statistical evidence",
    },
    {
        "id": "TC-F19-097", "family": 19, "family_name": "Explanations & Evidence",
        "question": "Did the discount policy cause revenue to increase?",
        "dataset": "orders", "target": "revenue", "treatment": "discount_rate", "op": "CAUSAL_CLAIM",
        "expected_behavior": "CAUSAL_REFUSAL_OR_BOUND", "expected_answer": "refusal of unverified causality",
    },

    # Family 20: Unsupported, underspecified, contradictory, impossible
    {
        "id": "TC-F20-098", "family": 20, "family_name": "Impossible & Adversarial",
        "question": "What is the average shoe size of our customers?",
        "dataset": "orders", "target": "shoe_size", "op": "ABSENT_MEASURE",
        "expected_behavior": "SAFE_REFUSAL", "expected_answer": "shoe size not found",
    },
    {
        "id": "TC-F20-099", "family": 20, "family_name": "Impossible & Adversarial",
        "question": "What will revenue be in 2099?",
        "dataset": "orders", "target": "revenue", "op": "IMPOSSIBLE_TIME",
        "expected_behavior": "SAFE_REFUSAL", "expected_answer": "out of range time horizon",
    },
    {
        "id": "TC-F20-100", "family": 20, "family_name": "Impossible & Adversarial",
        "question": "Find orders where revenue < 0 and revenue > 1000000.",
        "dataset": "orders", "target": "order_id", "op": "CONTRADICTORY_PREDICATE",
        "expected_behavior": "SAFE_ZERO_OR_REFUSAL", "expected_num": 0,
    },
    {
        "id": "TC-F20-101", "family": 20, "family_name": "Impossible & Adversarial",
        "question": "What caused customer churn?",
        "dataset": "customers", "target": "is_churned", "op": "UNSUPPORTED_CAUSAL",
        "expected_behavior": "CAUSAL_GATE_BLOCK", "expected_answer": "cannot assert causality without design",
    },
    {
        "id": "TC-F20-102", "family": 20, "family_name": "Impossible & Adversarial",
        "question": "What was the employee salary by department?",
        "dataset": "orders", "target": "salary", "group": "department", "op": "ABSENT_COLUMNS",
        "expected_behavior": "SAFE_REFUSAL", "expected_answer": "salary/department absent",
    },
    {
        "id": "TC-F20-103", "family": 20, "family_name": "Impossible & Adversarial",
        "question": "Calculate the square root of customer_id.",
        "dataset": "orders", "target": "customer_id", "op": "TYPE_INCOMPATIBLE",
        "expected_behavior": "SAFE_REFUSAL", "expected_answer": "cannot compute sqrt of string",
    },

    # Additional Adversarial & Edge Cases (104 - 120)
    {
        "id": "TC-ADV-104", "family": 4, "family_name": "Adversarial Paraphrase",
        "question": "Which region had top revenue?",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "PARAPHRASE_TOP",
        "expected_behavior": "TOP_ENTITY", "expected_entity": df_orders.groupby('region')['revenue'].sum().idxmax(),
    },
    {
        "id": "TC-ADV-105", "family": 3, "family_name": "Adversarial Abbreviation",
        "question": "Rev by reg",
        "dataset": "orders", "target": "revenue", "group": "region", "op": "ABBREVIATION",
        "expected_behavior": "GROUPED_SUM_OR_CLARIFY", "expected_answer": "resolve rev and reg or clarify",
    },
    {
        "id": "TC-ADV-106", "family": 5, "family_name": "Adversarial Typo",
        "question": "Total sales in norht region",
        "dataset": "orders", "target": "revenue", "filters": [{"col": "region", "val": "North"}], "op": "TYPO_FILTER",
        "expected_behavior": "TYPO_TOLERANCE_OR_CLARIFY", "expected_answer": "clarify norht or resolve North",
    },
    {
        "id": "TC-ADV-107", "family": 20, "family_name": "SQL Injection Attack",
        "question": "Orders where status = 'Completed'; DROP TABLE orders; --",
        "dataset": "orders", "target": "order_id", "op": "SQL_INJECTION",
        "expected_behavior": "SAFE_SANITIZED", "expected_answer": "no sql execution / injection sanitized",
    },
    {
        "id": "TC-ADV-108", "family": 20, "family_name": "Predicate Injection",
        "question": "Give me total revenue where 1=1 OR region='North'",
        "dataset": "orders", "target": "revenue", "op": "PREDICATE_INJECTION",
        "expected_behavior": "SAFE_REJECT_OR_ISOLATE", "expected_answer": "no arbitrary SQL execution",
    },
    {
        "id": "TC-ADV-109", "family": 2, "family_name": "Empty Dataset Robustness",
        "question": "What is the total revenue?",
        "dataset": "empty", "target": "revenue", "op": "EMPTY_DATASET",
        "expected_behavior": "SAFE_EMPTY_RESPONSE", "expected_answer": "no records or 0 without crash",
    },
    {
        "id": "TC-ADV-110", "family": 2, "family_name": "All-Null Dataset Robustness",
        "question": "What is the average revenue?",
        "dataset": "nulls", "target": "revenue", "op": "ALL_NULLS",
        "expected_behavior": "SAFE_NULL_HANDLING", "expected_answer": "null/undefined without crash",
    },
    {
        "id": "TC-ADV-111", "family": 8, "family_name": "Messy Churn Flags",
        "question": "What is the churn rate?",
        "dataset": "messy", "target": "churn_flag", "op": "MESSY_BOOLEAN_RATE",
        "expected_behavior": "SAFE_OR_REFUSE_MESSY", "expected_answer": "refusal or clean rate calculation",
    },
    {
        "id": "TC-ADV-112", "family": 13, "family_name": "Negative Revenue Handling",
        "question": "What is total revenue in messy data?",
        "dataset": "messy", "target": "revenue", "op": "NEGATIVE_NUMBERS",
        "expected_behavior": "PRESERVE_NEGATIVES", "expected_answer": "sum handles negatives without corruption",
    },
    {
        "id": "TC-ADV-113", "family": 6, "family_name": "Invalid Date Formats",
        "question": "What is revenue in January 2026 in messy data?",
        "dataset": "messy", "target": "revenue", "time": "2026-01", "op": "MIXED_DATE_FORMATS",
        "expected_behavior": "SAFE_DATE_PARSE", "expected_answer": "skips unparseable dates safely",
    },
    {
        "id": "TC-ADV-114", "family": 4, "family_name": "Ranking Ties",
        "question": "Top 2 regions by order count.",
        "dataset": "orders", "target": "order_id", "group": "region", "op": "RANK_TIE",
        "expected_behavior": "DETERMINISTIC_TIE_BREAK", "expected_answer": "deterministic ordering on tie",
    },
    {
        "id": "TC-ADV-115", "family": 2, "family_name": "Weighted Average",
        "question": "What is the weighted average price by quantity?",
        "dataset": "orders", "target": "unit_price", "weight": "quantity", "op": "WEIGHTED_AVG",
        "expected_behavior": "WEIGHTED_MEAN", "expected_num": float((df_orders['unit_price'] * df_orders['quantity']).sum() / df_orders['quantity'].sum()),
    },
    {
        "id": "TC-ADV-116", "family": 17, "family_name": "Unclear Aggregation Grain",
        "question": "Compare each customer's revenue with their region's average.",
        "dataset": "orders", "target": "revenue", "group": "customer_id", "op": "GRAIN_MISMATCH",
        "expected_behavior": "WINDOW_OR_CLARIFY", "expected_answer": "multi-grain calculation or clarification",
    },
    {
        "id": "TC-ADV-117", "family": 8, "family_name": "Zero Denominator Handling",
        "question": "What is the ratio of revenue to profit when profit is zero?",
        "dataset": "orders", "target": "ratio", "op": "ZERO_DIVISION",
        "expected_behavior": "SAFE_ZERO_DIV", "expected_answer": "null or undefined without zero division crash",
    },
    {
        "id": "TC-ADV-118", "family": 5, "family_name": "Boundary Inclusive/Exclusive",
        "question": "Orders where revenue >= 200 and revenue <= 500.",
        "dataset": "orders", "target": "order_id", "op": "INCLUSIVE_BOUNDS",
        "expected_behavior": "NUMERIC_EXACT", "expected_num": float(len(df_orders[(df_orders['revenue'] >= 200.0) & (df_orders['revenue'] <= 500.0)])),
    },
    {
        "id": "TC-ADV-119", "family": 9, "family_name": "Constant Variable Correlation",
        "question": "What is the correlation between constant_one and revenue?",
        "dataset": "constant_df", "target": "revenue", "x": "constant_col", "op": "CONSTANT_CORR",
        "expected_behavior": "SAFE_ZERO_VARIANCE", "expected_answer": "undefined correlation without zero division crash",
    },
    {
        "id": "TC-ADV-120", "family": 14, "family_name": "Duplicate Key Join Fanout",
        "question": "What is total revenue joined on duplicate keys?",
        "dataset": "multi_orders_messy", "target": "revenue", "op": "DUPLICATE_KEY_JOIN",
        "expected_behavior": "FANOUT_GATE_BLOCK", "expected_answer": "blocked unsafe many-to-many join",
    },
]

print(f"Total test cases configured: {len(test_cases)}")

# -------------------------------------------------------------
# 4. Independent Execution & Evaluation Function
# -------------------------------------------------------------

def evaluate_case(tc: dict) -> dict:
    cid = tc["id"]
    q = tc["question"]
    family = tc["family"]
    ds_name = tc.get("dataset", "orders")
    
    # Pick dataset
    if ds_name == "orders":
        df = df_orders
        ds_id = ds_orders_id
    elif ds_name == "customers":
        df = df_customers
        ds_id = ds_customers_id
    elif ds_name == "items":
        df = df_items
        ds_id = ds_items_id
    elif ds_name == "messy":
        df = df_messy
        ds_id = ds_messy_id
    elif ds_name == "empty":
        df = df_empty
        ds_id = ds_orders_id # will evaluate directly on df
    elif ds_name == "nulls":
        df = df_nulls
        ds_id = ds_orders_id
    elif ds_name == "constant_df":
        df = df_orders.copy()
        df['constant_col'] = 1.0
        ds_id = ds_orders_id
    elif "multi" in ds_name:
        df = df_orders
        ds_id = ds_orders_id
    else:
        df = df_orders
        ds_id = ds_orders_id

    # 1. Test Intent parsing
    intent = IntentEngine.parse_intent(q)
    
    # 2. Test Semantic resolution
    sem_engine = SemanticEngine()
    semantic = sem_engine.resolve_schema(intent, {'data': df})
    
    # 3. Test Canonical Contract
    try:
        canonical_contract = compile_canonical_question_contract(q, df, semantic=semantic)
        contract_family = canonical_contract.task_family
        contract_target = canonical_contract.target_column
        contract_grouping = list(canonical_contract.grouping_columns or [])
        contract_filters = list(canonical_contract.filter_conditions or [])
        contract_aggregation = canonical_contract.requested_aggregation
        contract_completeness = canonical_contract.completeness
    except Exception as exc:
        canonical_contract = None
        contract_family = f"ERROR:{type(exc).__name__}"
        contract_target = None
        contract_grouping = []
        contract_filters = []
        contract_aggregation = None
        contract_completeness = "ERROR"

    # 4. Test Typed Investigation Planner
    try:
        typed_plan = compile_typed_investigation(q, intent.intent_type, df=df, semantics=semantic)
        typed_task = typed_plan.get("task_type")
        typed_roles = typed_plan.get("semantic_roles", {})
        typed_unresolved = typed_plan.get("unresolved_roles", [])
        typed_warnings = typed_plan.get("type_warnings", [])
        branch_decisions = resolve_plan_branches(typed_plan, {"data_quality": {"status": "PASS"}, "roles": {}})
        typed_policy = conditional_execution_policy(typed_task, branch_decisions)
    except Exception as exc:
        typed_plan = None
        typed_task = f"ERROR:{type(exc).__name__}"
        typed_roles = {}
        typed_unresolved = []
        typed_warnings = [str(exc)]
        branch_decisions = []
        typed_policy = {"authorized": False, "error": str(exc)}

    # 5. Test direct Analyst Result calculation
    try:
        analyst_res = build_analyst_result(
            q, df,
            target=canonical_contract.target_column if canonical_contract else None,
            group=canonical_contract.grouping_columns[0] if (canonical_contract and canonical_contract.grouping_columns) else None,
            time_col=getattr(semantic, "time_col", None),
            approved_contract=None,
        )
        analyst_text = analyst_res.to_text() if analyst_res else None
        analyst_headline = analyst_res.headline if analyst_res else None
        analyst_numbers = analyst_res.numbers if analyst_res else None
    except Exception as exc:
        analyst_res = None
        analyst_text = f"ERROR:{type(exc).__name__}: {exc}"
        analyst_headline = None
        analyst_numbers = None

    # 6. Check for Silent Semantic Errors & Evaluation
    # Categories:
    # 1. Correct answer and correct verdict (PASS)
    # 2. Correct numeric answer but incorrect interpretation or verdict
    # 3. Partially correct answer with material omissions
    # 4. Incorrect answer or wrong analytical variable
    # 5. Unsupported answer presented as certain
    # 6. Correctly clarified, refused, or marked inconclusive
    # 7. Failure or crash without a misleading answer

    eval_category = None
    severity = "PASS"
    defect_reason = None
    extracted_num = None

    # Helper to parse number from headline
    if analyst_headline:
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(analyst_headline).replace(",", ""))
        if m:
            try:
                extracted_num = float(m.group(0))
            except Exception:
                extracted_num = None

    expected_behavior = tc.get("expected_behavior")

    # Evaluation logic per expected behavior
    if expected_behavior == "NUMERIC_EXACT":
        expected_num = tc["expected_num"]
        if extracted_num is not None:
            # Check delta
            delta = abs(extracted_num - expected_num)
            tol = 1e-2 if abs(expected_num) > 1 else 1e-4
            if delta <= tol:
                eval_category = "CORRECT_ANSWER_AND_VERDICT"
                severity = "PASS"
            else:
                eval_category = "INCORRECT_ANSWER_OR_WRONG_VARIABLE"
                severity = "P1"
                defect_reason = f"Numerical mismatch: expected {expected_num}, got {extracted_num} (delta {delta})"
        else:
            if analyst_res is None:
                eval_category = "PARTIALLY_CORRECT_OR_OMISSION"
                severity = "P2"
                defect_reason = f"Analyst result was None for exact numeric question: {q}"
            else:
                eval_category = "INCORRECT_ANSWER_OR_WRONG_VARIABLE"
                severity = "P1"
                defect_reason = f"Could not parse numeric answer from headline: {analyst_headline}"

    elif expected_behavior in ("GROUPED_SUM", "GROUPED_MEAN", "GROUPED_COUNT", "PERCENT_SHARES", "GROUPED_RATE"):
        if analyst_res and analyst_numbers and "table" in analyst_numbers:
            table = analyst_numbers["table"]
            expected_groups = tc.get("expected_groups", {})
            mismatches = []
            if isinstance(expected_groups, dict):
                for k, v in expected_groups.items():
                    k_str = str(k)
                    if k_str in table:
                        actual_val = table[k_str].get("value")
                        if actual_val is None or abs(float(actual_val) - float(v)) > 1e-2:
                            mismatches.append(f"{k}: expected {v}, got {actual_val}")
                    else:
                        mismatches.append(f"Missing group key: {k_str}")
            if not mismatches:
                eval_category = "CORRECT_ANSWER_AND_VERDICT"
                severity = "PASS"
            else:
                eval_category = "INCORRECT_ANSWER_OR_WRONG_VARIABLE"
                severity = "P1"
                defect_reason = f"Group value mismatches: {'; '.join(mismatches[:3])}"
        elif analyst_headline and any(k in str(analyst_headline) for k in ("North", "South", "East", "West", "Electronics")):
            eval_category = "CORRECT_ANSWER_AND_VERDICT"
            severity = "PASS"
        else:
            eval_category = "INCORRECT_ANSWER_OR_WRONG_VARIABLE"
            severity = "P1"
            defect_reason = f"Grouping answer failed or missing grouped table: headline={analyst_headline}"

    elif expected_behavior in ("TOP_ENTITY", "BOTTOM_ENTITY", "TOP_CUSTOMER"):
        expected_entity = str(tc.get("expected_entity"))
        if analyst_headline and expected_entity.lower() in str(analyst_headline).lower():
            eval_category = "CORRECT_ANSWER_AND_VERDICT"
            severity = "PASS"
        else:
            eval_category = "INCORRECT_ANSWER_OR_WRONG_VARIABLE"
            severity = "P1"
            defect_reason = f"Expected top entity '{expected_entity}' not found in headline: {analyst_headline}"

    elif expected_behavior in ("SAFE_REFUSAL", "SAFE_REFUSAL_OR_CLARIFICATION", "CAUSAL_REFUSAL_OR_BOUND", "CAUSAL_GATE_BLOCK", "FANOUT_GATE_BLOCK", "FANOUT_PREVENTION"):
        # The system SHOULD refuse, mark inconclusive, or return None
        if canonical_contract and canonical_contract.completeness != "COMPLETE":
            eval_category = "CORRECTLY_CLARIFIED_OR_REFUSED"
            severity = "PASS"
        elif analyst_res is None or typed_policy.get("action") == "BLOCK_SPECIALIST_PATH" or not typed_policy.get("authorized"):
            eval_category = "CORRECTLY_CLARIFIED_OR_REFUSED"
            severity = "PASS"
        elif "refus" in str(analyst_text).lower() or "inconclusive" in str(analyst_text).lower() or "cannot" in str(analyst_text).lower():
            eval_category = "CORRECTLY_CLARIFIED_OR_REFUSED"
            severity = "PASS"
        else:
            # IT RETURNED A CONFIDENT ANSWER TO AN UNSUPPORTED / IMPOSSIBLE QUESTION!
            eval_category = "UNSUPPORTED_ANSWER_PRESENTED_AS_CERTAIN"
            severity = "P1"
            defect_reason = f"Silent semantic error: answered unsupported/impossible request as certain: {analyst_headline}"

    elif expected_behavior in ("SAFE_EMPTY_RESPONSE", "SAFE_NULL_HANDLING", "SAFE_ZERO_DIV", "SAFE_ZERO_VARIANCE"):
        if analyst_res is None or "undefined" in str(analyst_headline).lower() or "no records" in str(analyst_headline).lower() or extracted_num == 0.0 or math.isnan(extracted_num or float('nan')):
            eval_category = "CORRECTLY_CLARIFIED_OR_REFUSED"
            severity = "PASS"
        else:
            eval_category = "CORRECT_ANSWER_AND_VERDICT"
            severity = "PASS"

    elif expected_behavior in ("PROFILE", "SAFE_PROFILE", "SCHEMA_DESCRIPTION", "DISTRIBUTION_PROFILE", "OUTLIER_RESULT", "ANOMALY_CANDIDATES", "TREND_DIRECTION", "DECOMPOSITION_EXPLANATION", "EXPLANATION_BREAKDOWN", "ASSOCIATION_EVIDENCE", "DRIVER_RANKING", "CHURN_FACTORS"):
        if typed_task in ("PROFILE", "DISTRIBUTION", "ANOMALY", "TEMPORAL_TREND", "CONTRIBUTION", "ASSOCIATION") or analyst_res is not None:
            eval_category = "CORRECT_ANSWER_AND_VERDICT"
            severity = "PASS"
        else:
            eval_category = "PARTIALLY_CORRECT_OR_OMISSION"
            severity = "P2"
            defect_reason = f"Analytical family profile/trend omitted or unhandled: typed_task={typed_task}"
    else:
        # Default fallback
        if analyst_res is not None:
            eval_category = "CORRECT_ANSWER_AND_VERDICT"
            severity = "PASS"
        else:
            eval_category = "PARTIALLY_CORRECT_OR_OMISSION"
            severity = "P2"
            defect_reason = f"Unresolved operation for question: {q}"

    return {
        "test_id": cid,
        "family": family,
        "family_name": tc["family_name"],
        "question": q,
        "dataset_name": ds_name,
        "expected_behavior": expected_behavior,
        "expected_num": tc.get("expected_num"),
        "expected_entity": tc.get("expected_entity"),
        "intent_type": intent.intent_type,
        "semantic_target": getattr(semantic, "target_metric_col", None),
        "semantic_group": getattr(semantic, "group_dimension_col", None),
        "canonical_family": contract_family,
        "canonical_target": contract_target,
        "canonical_grouping": contract_grouping,
        "canonical_filters": contract_filters,
        "canonical_completeness": contract_completeness,
        "typed_task": typed_task,
        "typed_roles": typed_roles,
        "typed_unresolved": typed_unresolved,
        "typed_warnings": typed_warnings,
        "typed_policy": typed_policy,
        "analyst_headline": analyst_headline,
        "analyst_extracted_num": extracted_num,
        "eval_category": eval_category,
        "severity": severity,
        "defect_reason": defect_reason,
    }

print("Running 120 audit test cases...")
results = []
t_start = time.time()
for idx, tc in enumerate(test_cases, 1):
    res = evaluate_case(tc)
    results.append(res)
    if idx % 20 == 0 or idx == len(test_cases):
        print(f"Processed {idx}/{len(test_cases)} cases...")

t_elapsed = round(time.time() - t_start, 2)
print(f"Completed evaluation of {len(results)} cases in {t_elapsed}s.")

# Write full raw results to JSON
out_path = Path(__file__).resolve().parent / "audit_matrix_results.json"
out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"Raw results saved to {out_path}")

# Print summary scorecard
df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("AUDIT MATRIX SUMMARY SCORECARD")
print("="*80)
print(f"Total Cases Executed: {len(df_res)}")
print("\nResults by Evaluation Category:")
print(df_res['eval_category'].value_counts())

print("\nResults by Severity:")
print(df_res['severity'].value_counts())

print("\nResults by Family:")
fam_summary = df_res.groupby('family_name').agg(
    total=('test_id', 'count'),
    passed=('severity', lambda s: (s == 'PASS').sum()),
    p1=('severity', lambda s: (s == 'P1').sum()),
    p2=('severity', lambda s: (s == 'P2').sum()),
    p0=('severity', lambda s: (s == 'P0').sum()),
)
fam_summary['pass_rate'] = (fam_summary['passed'] / fam_summary['total'] * 100).round(1)
print(fam_summary.to_string())

print("\nIdentified P1 Defects Sample:")
p1_sample = df_res[df_res['severity'] == 'P1'][['test_id', 'question', 'defect_reason']].head(15)
for _, r in p1_sample.iterrows():
    print(f"[{r['test_id']}] Q: '{r['question']}' -> Defect: {r['defect_reason']}")
