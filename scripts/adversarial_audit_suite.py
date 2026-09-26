"""Rigorous Automated Adversarial Audit Suite with Executable Assertion Proofs."""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from apps.api.src.ai.orchestrator import AutonomousOrchestrator
from apps.api.src.ai.providers.factory import get_ai_provider
from packages.analytics_core.src.validation.validator import IndependentValidator
from packages.shared.src.enums import AnalysisStatus, ValidationStatus

provider = get_ai_provider("mock")
orch = AutonomousOrchestrator(ai_provider=provider)

print("=" * 60)
print("RUNNING ADVERSARIAL AUDIT SUITE WITH EXECUTABLE ASSERTIONS")
print("=" * 60)

# -------------------------------------------------------------
# TEST 1: UNKNOWN DATASETS TEST
# -------------------------------------------------------------
print("\n--- TEST 1: UNKNOWN DATASETS ---")

# Dataset A: Marketing
marketing_df = pd.DataFrame({
    'campaign_code': [f'CMP-{i:03d}' for i in range(100)],
    'launch_day': pd.date_range('2025-01-01', periods=100, freq='D'),
    'channel_name': np.random.choice(['Search', 'Social', 'Display', 'Email'], size=100),
    'spend_usd': np.random.uniform(500, 5000, size=100),
    'impressions': np.random.randint(10000, 100000, size=100),
    'click_count': np.random.randint(200, 5000, size=100),
    'conversion_count': np.random.randint(10, 300, size=100),
    'revenue_generated': np.random.uniform(1000, 20000, size=100),
})

res_a = orch.run_investigation("Why did revenue_generated decline?", "proj-test", {"marketing": marketing_df})
assert res_a.status == AnalysisStatus.COMPLETED, f"Marketing dataset failed: {res_a.status}"
assert len(res_a.hypotheses) >= 1, "Marketing dataset generated zero hypotheses"
assert res_a.direct_answer is not None, "Marketing dataset direct answer is None"
print(f" -> Marketing Dataset PASSED ({len(res_a.hypotheses)} hypotheses evaluated)")

# Dataset B: SaaS
saas_df = pd.DataFrame({
    'account_key': [f'ACC-{i:04d}' for i in range(200)],
    'signup_timestamp': pd.date_range('2024-01-01', periods=200, freq='W'),
    'plan_name': np.random.choice(['Starter', 'Pro', 'Enterprise'], size=200),
    'monthly_fee': np.random.choice([29, 99, 499], size=200),
    'login_count': np.random.randint(1, 100, size=200),
    'support_tickets': np.random.randint(0, 10, size=200),
    'last_active_timestamp': pd.date_range('2025-01-01', periods=200, freq='D'),
    'renewal_status': np.random.choice(['Active', 'Churned', 'Pending'], size=200),
})

res_b = orch.run_investigation("Which accounts are likely to churn?", "proj-test", {"saas": saas_df})
assert res_b.status == AnalysisStatus.COMPLETED, f"SaaS dataset failed: {res_b.status}"
assert len(res_b.steps) >= 3, "SaaS dataset failed to execute multi-step investigation"
print(f" -> SaaS Dataset PASSED ({len(res_b.steps)} execution steps)")

# Dataset C: Manufacturing
mfg_df = pd.DataFrame({
    'production_day': pd.date_range('2025-06-01', periods=150, freq='D'),
    'factory_code': np.random.choice(['FAC-East', 'FAC-West', 'FAC-North'], size=150),
    'machine_code': np.random.choice(['M-01', 'M-02', 'M-03', 'M-04'], size=150),
    'units_produced': np.random.randint(500, 2000, size=150),
    'defect_count': np.random.randint(0, 50, size=150),
    'downtime_minutes': np.random.randint(0, 180, size=150),
    'energy_consumption': np.random.uniform(1000, 5000, size=150),
})

res_c = orch.run_investigation("Forecast energy_consumption trajectory", "proj-test", {"manufacturing": mfg_df})
assert res_c.status == AnalysisStatus.COMPLETED, f"Mfg dataset failed: {res_c.status}"
assert "energy_consumption" in res_c.direct_answer.lower() or "forecast" in res_c.main_finding.lower(), "Mfg forecasting output invalid"
print(" -> Manufacturing Dataset PASSED (Forecast generated)")

# -------------------------------------------------------------
# TEST 2: SCHEMA MUTATION TEST
# -------------------------------------------------------------
print("\n--- TEST 2: SCHEMA MUTATION ON SALES DATA ---")
sales_df = pd.read_csv("data/seed/sales.csv")
mutated_sales = sales_df.rename(columns={
    'order_date': 'transaction_timestamp',
    'customer_id': 'buyer_key',
    'product_id': 'item_code',
    'region': 'territory',
    'quantity': 'units',
    'unit_price': 'price_each',
    'revenue': 'net_sales',
    'cost': 'acquisition_cost',
    'profit': 'gross_margin',
})

res_mut = orch.run_investigation("Why did net_sales fall?", "proj-test", {"sales_mutated": mutated_sales})
assert res_mut.status == AnalysisStatus.COMPLETED, "Mutated schema investigation failed"
assert len(res_mut.evidence) >= 1, "Mutated schema failed to produce validated evidence"
assert any(e.validation_status == ValidationStatus.PASSED for e in res_mut.evidence), "Evidence failed tolerance validation"
assert any("Region A" in e.statement or "71.7%" in e.statement for e in res_mut.evidence), "Mutated schema localized driver mismatch"
print(f" -> Schema Mutation PASSED (Isolated driver on mutated column names with validated evidence)")

# -------------------------------------------------------------
# TEST 3: UNKNOWN QUESTIONS TEST
# -------------------------------------------------------------
print("\n--- TEST 3: UNKNOWN QUESTIONS ---")
unknown_qs = [
    "Explore this dataset and tell me what deserves investigation",
    "What changed most significantly over time?",
    "What are the biggest risks visible in this data?",
    "Find evidence that contradicts the obvious explanation",
]

for uq in unknown_qs:
    res_u = orch.run_investigation(uq, "proj-test", {"sales": sales_df})
    assert res_u.status == AnalysisStatus.COMPLETED, f"Failed on unknown query: {uq}"
    assert len(res_u.steps) >= 3, f"Incomplete steps on query: {uq}"
    assert res_u.direct_answer is not None, f"Empty direct answer on: {uq}"
    print(f" -> Query '{uq[:35]}...' PASSED ({len(res_u.steps)} steps)")

# -------------------------------------------------------------
# TEST 4: FAULT INJECTION ON VALIDATOR
# -------------------------------------------------------------
print("\n--- TEST 4: FAULT INJECTION ON VALIDATOR ---")
validator = IndependentValidator()

v_exact = validator.validate_numerical_claim("Exact", 100.0, 100.0, relative_tolerance=0.01)
assert v_exact.status == ValidationStatus.PASSED, "Exact numerical match failed validation"

v_1pct = validator.validate_numerical_claim("1% dev", 100.0, 101.0, relative_tolerance=0.01)
assert v_1pct.status == ValidationStatus.PASSED, "1% deviation inside tolerance failed"

v_10pct = validator.validate_numerical_claim("10% dev", 100.0, 110.0, relative_tolerance=0.01)
assert v_10pct.status == ValidationStatus.FAILED, "10% deviation outside tolerance was not rejected"

v_50pct = validator.validate_numerical_claim("50% dev", 100.0, 150.0, relative_tolerance=0.01)
assert v_50pct.status == ValidationStatus.FAILED, "50% deviation outside tolerance was not rejected"

print(" -> Validator Fault Injection PASSED (Strict tolerance boundary enforcement proven)")

# -------------------------------------------------------------
# TEST 5: COUNTER-EXAMPLE HYPOTHESIS TEST
# -------------------------------------------------------------
print("\n--- TEST 5: COUNTER-EXAMPLE HYPOTHESIS TEST ---")
dates = pd.date_range('2026-01-01', periods=100, freq='D')
price_collapse_df = pd.DataFrame({
    'order_id': [f'ORD-{i}' for i in range(100)],
    'order_date': dates,
    'region': ['Region A' for _ in dates],
    'unit_price': [100.0 if d.month == 2 else 50.0 for d in dates],  # Price drops 50% in Mar
    'quantity': [10 for _ in range(100)],                             # Volume constant
    'revenue': [1000.0 if d.month == 2 else 500.0 for d in dates],   # Revenue drops 50% in Mar
})

res_price_col = orch.run_investigation("Why did revenue fall in March?", "proj-test", {"sales": price_collapse_df})
assert res_price_col.status == AnalysisStatus.COMPLETED, "Price collapse run failed"
assert any(h.id == "HYP-01" and h.status == "supported" for h in res_price_col.hypotheses), (
    "HYP-01 (Pricing) was not supported when price collapsed by 50%"
)
print(" -> Counter-Example Pricing Test PASSED (HYP-01 supported on genuine price collapse data)")

print("\n" + "=" * 60)
print("ALL 5 ADVERSARIAL AUDIT TESTS PASSED WITH 100% ASSERTION INTEGRITY")
print("=" * 60)
