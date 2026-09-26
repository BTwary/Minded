from packages.analytics_core.src.governance.claim_gate import admit_positive_claim
from packages.analytics_core.src.engines.verdict import VerdictEngine


def test_material_unvalidated_assumption_blocks_claim():
    r = admit_positive_claim(verdict_type='DIAGNOSED', directly_tested=True, verified_evidence=True, unvalidated_high_risk_assumptions=1)
    assert not r.allowed
    assert 'material_high_risk_assumptions_unvalidated' in r.reasons


def test_observed_remains_available_under_assumption_risk():
    r = admit_positive_claim(verdict_type='OBSERVED', directly_tested=True, verified_evidence=True, unvalidated_high_risk_assumptions=1)
    assert r.allowed is False  # universal gate still requires the stronger scientific proof path


def test_verdict_cannot_ignore_material_assumption_risk():
    v = VerdictEngine.evaluate_verdict(leading_hypothesis_code='H1', leading_hypothesis_posterior=.9, all_verifications_passed=True, adversarial_attack_survived=True, leading_hypothesis_directly_tested=True, unvalidated_high_risk_assumptions=1)
    assert v.verdict_type == 'INCONCLUSIVE'
