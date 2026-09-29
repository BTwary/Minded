"""Fail-closed verification-contract equivalence proofs."""
from packages.analytics_core.src.engines.verification_contract import VerificationContract, compare_verification_contracts


def test_same_estimand_population_grain_and_parameters_are_comparable():
    a = VerificationContract("mean revenue", "customers_active", "customer", "mean")
    b = VerificationContract("mean revenue", "customers_active", "customer", "mean")
    assert compare_verification_contracts(a, b) == (True, [])


def test_different_estimand_is_incomparable_even_when_numeric_value_matches():
    a = VerificationContract("mean revenue", "customers_active", "customer", "mean")
    b = VerificationContract("sum revenue", "customers_active", "customer", "sum")
    comparable, mismatches = compare_verification_contracts(a, b)
    assert comparable is False
    assert "estimand" in mismatches
    assert "parameterization" in mismatches


def test_population_or_grain_mismatch_is_incomparable():
    a = VerificationContract("conversion rate", "all_customers", "customer", "rate")
    b = VerificationContract("conversion rate", "paid_customers", "customer", "rate")
    comparable, mismatches = compare_verification_contracts(a, b)
    assert comparable is False
    assert mismatches == ["population"]
