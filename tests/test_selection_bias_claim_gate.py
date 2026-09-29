from packages.analytics_core.src.governance.claim_gate import admit_positive_claim


def test_selection_indicators_block_positive_claims():
    result = admit_positive_claim(
        verdict_type="STATISTICALLY_SIGNIFICANT",
        directly_tested=True,
        verified_evidence=True,
        selection_bias_indicators=["active/survivorship field present"],
    )
    assert not result.allowed
    assert "selection_bias_indicators_present" in result.reasons


def test_selection_indicators_do_not_block_descriptive_observation():
    result = admit_positive_claim(
        verdict_type="OBSERVED",
        directly_tested=True,
        verified_evidence=True,
        selection_bias_indicators=["active/survivorship field present"],
    )
    assert result.allowed


def test_no_selection_indicators_preserves_positive_claim():
    result = admit_positive_claim(
        verdict_type="STATISTICALLY_SIGNIFICANT",
        directly_tested=True,
        verified_evidence=True,
        selection_bias_indicators=[],
    )
    assert result.allowed
