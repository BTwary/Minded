"""Deterministic analytical identity proofs."""
from packages.analytics_core.src.intelligence.analytical_identity import FinalAnalyticalContract


def _contract():
    return FinalAnalyticalContract(
        dataset_identity="dataset-abc",
        problem_class="CORRELATIONAL",
        canonical_task="CORRELATION",
        objective="COMPARE",
        target_column="annual_sales",
        predictor_columns=("marketing_spend", "price"),
        selected_method_code="association_numeric",
        claim_ceiling="STATISTICALLY_SIGNIFICANT",
        verification_regime="independent_correlation_recompute",
        method_family="CORRELATION",
        bindings=(("OUTCOME", "sales", "annual_sales"), ("EXPLANATORY_VARIABLE", "sales", "marketing_spend"), ("EXPLANATORY_VARIABLE", "sales", "price")),
    )


def test_same_contract_is_byte_identical():
    a = _contract()
    b = _contract()
    assert a.analytical_identity == b.analytical_identity
    assert a.estimand_identity == b.estimand_identity
    assert a.semantic_binding_identity == b.semantic_binding_identity


def test_runtime_failure_cannot_mutate_identity_inputs():
    before = _contract().analytical_identity
    try:
        raise RuntimeError("transient decide failure")
    except RuntimeError:
        pass
    after = _contract().analytical_identity
    assert after == before
