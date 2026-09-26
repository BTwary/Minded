import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.analytics_core.src.governance.claim_gate import admit_positive_claim

def test_full_scope_allows_positive_claim_when_other_requirements_hold():
    r = admit_positive_claim(verdict_type="DIAGNOSED", directly_tested=True, verified_evidence=True, full_scope=True, sampling_policy="NONE")
    assert r.allowed is True

def test_sampled_analysis_blocks_strong_positive_claim():
    r = admit_positive_claim(verdict_type="DIAGNOSED", directly_tested=True, verified_evidence=True, full_scope=False, sampling_policy="STRATIFIED", sampling_reason="resource bounded")
    assert r.allowed is False
    assert "sampled_analysis_scope" in r.reasons

def test_descriptive_observation_survives_sampling_ceiling():
    r = admit_positive_claim(verdict_type="OBSERVED", directly_tested=True, verified_evidence=True, full_scope=False, sampling_policy="RANDOM")
    assert r.allowed is True
