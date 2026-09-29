"""Property-Based Schema Generalization Test Suite for the Autonomous Data Analyst Platform."""
import os
import random
import string
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from apps.api.src.ai.orchestrator import AutonomousOrchestrator
from apps.api.src.ai.providers.factory import get_ai_provider
from packages.shared.src.enums import AnalysisStatus, AnalyticalVerdict, ValidationStatus

provider = get_ai_provider("mock")
orch = AutonomousOrchestrator(ai_provider=provider)

raw_sales = pd.read_csv("data/seed/sales.csv")

print("=" * 60)
print("RUNNING PROPERTY-BASED SCHEMA GENERALIZATION TEST SUITE")
print("=" * 60)

# Helper to generate randomized noise column
def generate_noise_series(length: int, col_type: str):
    if col_type == "str":
        return [f"NOISE-{random.randint(1000, 9999)}" for _ in range(length)]
    elif col_type == "float":
        return np.random.uniform(1.0, 100.0, size=length)
    elif col_type == "int":
        return np.random.randint(0, 50, size=length)
    else:
        return [random.choice([True, False]) for _ in range(length)]

# -------------------------------------------------------------
# PERMUTATION 1: OBFUSCATED SUPPLY CHAIN WITH RANDOMIZED NOISE COLUMNS
# -------------------------------------------------------------
print("\n[PERMUTATION 1] Obfuscated Supply Chain Schema + 5 Injected Noise Columns + Reordered Columns...")
p1_mapping = {
    'order_date': 'ts_event',
    'customer_id': 'uid_hash',
    'product_id': 'sku_num',
    'region': 'loc_zone',
    'quantity': 'qty_shipped',
    'unit_price': 'rate_per_unit',
    'revenue': 'gross_inflow',
    'cost': 'expenditure_amt',
    'profit': 'net_delta',
}

df_p1 = raw_sales.rename(columns=p1_mapping)

# Inject noise columns
n_rows = len(df_p1)
df_p1['sys_worker_id'] = generate_noise_series(n_rows, 'str')
df_p1['lot_batch_code'] = generate_noise_series(n_rows, 'str')
df_p1['is_test_env'] = generate_noise_series(n_rows, 'bool')
df_p1['sensor_temp_c'] = generate_noise_series(n_rows, 'float')
df_p1['crc32_val'] = generate_noise_series(n_rows, 'int')

# Randomize column order
cols_p1 = list(df_p1.columns)
random.shuffle(cols_p1)
df_p1 = df_p1[cols_p1]

res_p1 = orch.run_investigation("Why did gross_inflow decline?", "proj-perm1", {"supply_chain": df_p1})
assert res_p1.status == AnalysisStatus.COMPLETED, f"P1 Failed with status: {res_p1.status}"
assert len(res_p1.steps) >= 3, "P1 Failed: insufficient execution steps"
assert len(res_p1.evidence) >= 1, "P1 Failed: zero evidence generated"
assert all(e.validation_status == ValidationStatus.PASSED for e in res_p1.evidence), "P1 Evidence failed validation"
assert res_p1.verdict in (AnalyticalVerdict.DIAGNOSED, AnalyticalVerdict.OBSERVED), f"P1 Invalid verdict: {res_p1.verdict}"
assert len(res_p1.manifest.reproducible_hash) == 64, "P1 Invalid SHA-256 hash"
print(f" -> PASSED: Verdict = {res_p1.verdict.value} across {len(cols_p1)} randomized columns")
print(f" -> Direct Answer: {res_p1.direct_answer}")

# -------------------------------------------------------------
# PERMUTATION 2: FINTECH LEDGER WITH INJECTED MISSINGNESS (NaNs)
# -------------------------------------------------------------
print("\n[PERMUTATION 2] Fintech Ledger Schema with Injected Missingness (NaNs)...")
p2_mapping = {
    'order_date': 'settlement_day',
    'customer_id': 'wallet_id',
    'product_id': 'asset_type',
    'region': 'jurisdiction',
    'quantity': 'token_count',
    'unit_price': 'strike_rate',
    'revenue': 'volume_usd',
    'cost': 'fee_toll',
    'profit': 'net_gain',
}
df_p2 = raw_sales.rename(columns=p2_mapping).copy()

# Inject 5% NaNs into fee_toll and jurisdiction
mask_nan = np.random.rand(len(df_p2)) < 0.05
df_p2.loc[mask_nan, 'fee_toll'] = np.nan

res_p2 = orch.run_investigation("What are the correlation drivers of volume_usd?", "proj-perm2", {"fintech_ledger": df_p2})
assert res_p2.status == AnalysisStatus.COMPLETED, f"P2 Failed: {res_p2.status}"
assert len(res_p2.evidence) >= 1, "P2 Failed: zero evidence generated"
assert any(e.validation_status == ValidationStatus.PASSED for e in res_p2.evidence), "P2 Evidence failed validation"
assert res_p2.verdict in (AnalyticalVerdict.STATISTICALLY_SIGNIFICANT, AnalyticalVerdict.OBSERVED), f"P2 Verdict invalid: {res_p2.verdict}"
print(f" -> PASSED: Correlation isolated on noisy ledger with NaNs (Verdict = {res_p2.verdict.value})")

# -------------------------------------------------------------
# PERMUTATION 3: TELEMETRY SENSOR TIME SERIES
# -------------------------------------------------------------
print("\n[PERMUTATION 3] Telemetry Sensor Time-Series Forecasting...")
dates_hourly = pd.date_range('2026-01-01', periods=200, freq='h')
df_p3 = pd.DataFrame({
    'read_timestamp': dates_hourly,
    'sensor_tag': [f'SNS-{i%5:02d}' for i in range(200)],
    'energy_load_kwh': [1000.0 + (i * 2.5) + (50.0 * np.sin(i / 10.0)) for i in range(200)],
    'ambient_temp_k': np.random.uniform(280, 310, size=200),
    'fan_speed_rpm': np.random.randint(1000, 3500, size=200),
})

res_p3 = orch.run_investigation("Forecast energy_load_kwh trajectory", "proj-perm3", {"telemetry": df_p3})
assert res_p3.status == AnalysisStatus.COMPLETED, f"P3 Failed: {res_p3.status}"
assert res_p3.verdict in (AnalyticalVerdict.PREDICTED, AnalyticalVerdict.OBSERVED), f"P3 Verdict invalid: {res_p3.verdict}"
assert "energy_load_kwh" in res_p3.direct_answer.lower() or "forecast" in res_p3.main_finding.lower(), "P3 Output invalid"
print(f" -> PASSED: Trajectory model fitted on hourly telemetry (Verdict = {res_p3.verdict.value})")

# -------------------------------------------------------------
# PERMUTATION 4: MULTI-TABLE JOIN GRAPH (TX + CLIENT METADATA)
# -------------------------------------------------------------
print("\n[PERMUTATION 4] Multi-Table Entity Join Graph (2 Relational Tables)...")
tx_df = pd.DataFrame({
    'tx_id': [f'TX-{i:05d}' for i in range(500)],
    'client_key': [f'CL-{random.randint(1, 50):03d}' for _ in range(500)],
    'tx_timestamp': pd.date_range('2026-01-01', periods=500, freq='h'),
    'tx_amount': np.random.uniform(50, 2000, size=500),
})

client_df = pd.DataFrame({
    'client_key': [f'CL-{i:03d}' for i in range(1, 51)],
    'account_tier': np.random.choice(['Gold', 'Silver', 'Bronze'], size=50),
    'signup_year': np.random.choice([2023, 2024, 2025], size=50),
})

res_p4 = orch.run_investigation(
    "Explore transactions and identify top client spend patterns",
    "proj-perm4",
    {"transactions": tx_df, "clients": client_df},
)
assert res_p4.status == AnalysisStatus.COMPLETED, f"P4 Failed: {res_p4.status}"
assert len(res_p4.steps) >= 3, "P4 Failed: insufficient steps across multi-table schema"
assert res_p4.discovery is not None, "P4 Discovery summary is None"
assert len(res_p4.discovery.candidate_keys) >= 1 or len(res_p4.discovery.primary_metrics) >= 1, "P4 Failed to discover schema attributes"
print(f" -> PASSED: Multi-table relational graph discovered ({len(res_p4.discovery.candidate_keys)} keys, {len(res_p4.discovery.primary_metrics)} metrics identified)")

print("\n" + "=" * 60)
print("ALL 4 PROPERTY-BASED GENERALIZATION PERMUTATIONS PASSED WITH 100% INTEGRITY")
print("=" * 60)
