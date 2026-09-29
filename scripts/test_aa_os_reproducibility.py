"""Reproducibility, Manifest Integrity, and Deterministic Hash Verification Suite for AA-OS."""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from apps.api.src.ai.orchestrator import AutonomousOrchestrator
from apps.api.src.ai.providers.factory import get_ai_provider
from packages.analytics_core.src.validation.validator import IndependentValidator
from packages.shared.src.enums import AnalysisStatus, ValidationStatus

provider = get_ai_provider("mock")
orch = AutonomousOrchestrator(ai_provider=provider)

sales_df = pd.read_csv("data/seed/sales.csv")

print("=" * 60)
print("RUNNING AA-OS REPRODUCIBILITY & CANONICAL MANIFEST VERIFICATION")
print("=" * 60)

# 1. Dual identical run determinism
print("\n[TEST 1] Dual identical investigation execution & hash comparison...")
q = "Why did revenue fall in March?"
res1 = orch.run_investigation(q, "proj-det", {"sales": sales_df.copy()})
res2 = orch.run_investigation(q, "proj-det", {"sales": sales_df.copy()})

assert res1.status == AnalysisStatus.COMPLETED, f"Run 1 failed: {res1.status}"
assert res2.status == AnalysisStatus.COMPLETED, f"Run 2 failed: {res2.status}"

# Check hash determinism
hash1 = res1.manifest.reproducible_hash
hash2 = res2.manifest.reproducible_hash
assert len(hash1) == 64, f"Hash 1 is not valid SHA-256: {hash1}"
assert len(hash2) == 64, f"Hash 2 is not valid SHA-256: {hash2}"
assert hash1 == hash2, f"Determinism failure: Hash 1 ({hash1}) != Hash 2 ({hash2})"
assert res1.verdict == res2.verdict, "Verdict mismatch across identical runs"
assert len(res1.steps) == len(res2.steps), "Step count mismatch across identical runs"
assert res1.direct_answer == res2.direct_answer, "Direct answer mismatch across identical runs"
print(f" -> PASSED: Dual execution produced 100% identical SHA-256 fingerprint ({hash1})")

# 2. Data perturbation changes the hash
print("\n[TEST 2] Single-row data perturbation modifies canonical hash...")
sales_perturbed = sales_df.copy()
sales_perturbed.loc[0, 'revenue'] = sales_perturbed.loc[0, 'revenue'] + 500.0

res_pert = orch.run_investigation(q, "proj-det", {"sales": sales_perturbed})
hash_pert = res_pert.manifest.reproducible_hash
assert hash_pert != hash1, f"Data perturbation failed to modify SHA-256 hash: {hash_pert} == {hash1}"
print(f" -> PASSED: Row-level perturbation altered hash ({hash_pert} != {hash1})")

# 3. Secondary validator catches corrupted calculation
print("\n[TEST 3] Independent validator catches corrupted numerical calculation...")
validator = IndependentValidator()
# Corrupt a 1,800,974.25 claim to 1,500,000.00 (16.7% deviation)
v_corrupted = validator.validate_numerical_claim(
    claim_name="Region B Revenue Contraction",
    expected_value=1800974.25,
    actual_value=1500000.00,
    relative_tolerance=0.01,
)
assert v_corrupted.status == ValidationStatus.FAILED, f"Validator failed to reject corrupted calculation: {v_corrupted.status}"
print(f" -> PASSED: 16.7% corrupted numerical claim successfully caught and failed by validator.")

print("\n" + "=" * 60)
print("ALL REPRODUCIBILITY & CANONICAL MANIFEST TESTS PASSED WITH 100% INTEGRITY")
print("=" * 60)
