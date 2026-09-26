from packages.analytics_core.src.engines.verdict import VerdictEngine


def test_final_verdict_cannot_claim_stopping_when_authoritative_engine_refused():
    result = VerdictEngine.evaluate_verdict(
        leading_hypothesis_code="H1",
        leading_hypothesis_posterior=0.90,
        initial_entropy=1.0,
        final_entropy=0.60,
        all_verifications_passed=True,
        adversarial_attack_survived=True,
        leading_hypothesis_directly_tested=True,
        authoritative_stopping_met=False,
        authoritative_stopping_reason="COUNTER_HYPOTHESIS_NOT_EVALUATED",
    )
    assert result.stopping_criteria_met is False
    assert "COUNTER_HYPOTHESIS_NOT_EVALUATED" in result.stopping_rationale


def test_authoritative_stop_is_reflected_when_engine_allows_it():
    result = VerdictEngine.evaluate_verdict(
        leading_hypothesis_code="H1",
        leading_hypothesis_posterior=0.90,
        all_verifications_passed=True,
        adversarial_attack_survived=True,
        leading_hypothesis_directly_tested=True,
        authoritative_stopping_met=True,
        authoritative_stopping_reason="SUFFICIENTLY_RESOLVED",
    )
    assert result.stopping_criteria_met is True
