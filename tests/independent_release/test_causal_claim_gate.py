from packages.analytics_core.src.governance.claim_gate import admit_positive_claim


def test_identified_causal_claim_requires_effect_estimate():
    r = admit_positive_claim(
        verdict_type="DIAGNOSED",
        directly_tested=True,
        verified_evidence=True,
        requested_causal=True,
        causal_identifiable=True,
        causal_effect_estimated=False,
    )
    assert not r.allowed
    assert "causal_effect_estimate_missing" in r.reasons


def test_identified_causal_claim_can_pass_when_effect_estimate_exists():
    r = admit_positive_claim(
        verdict_type="DIAGNOSED",
        directly_tested=True,
        verified_evidence=True,
        requested_causal=True,
        causal_identifiable=True,
        causal_effect_estimated=True,
        method_claim_ceiling="DIAGNOSED",
    )
    assert r.allowed


def test_observational_claim_is_unaffected_by_causal_estimator_flag():
    r = admit_positive_claim(
        verdict_type="DIAGNOSED",
        directly_tested=True,
        verified_evidence=True,
        requested_causal=False,
        causal_identifiable=False,
        causal_effect_estimated=False,
    )
    assert r.allowed
