import os, sys
sys.path.insert(0, os.path.abspath("."))
from packages.analytics_core.src.execution.state_machine import AnalyticalPhase
from packages.analytics_core.src.intelligence.contract_authority import set_phase

def test_contract_phase_vocab_has_authoritative_lifecycle():
    assert AnalyticalPhase.QUESTION_UNDERSTANDING
    assert AnalyticalPhase.SEMANTIC_MODEL
    assert AnalyticalPhase.DATA_READINESS
    assert AnalyticalPhase.CONTRACT_COMPILED
    assert AnalyticalPhase.HYPOTHESIS_FORMATION
    assert AnalyticalPhase.METHOD_SELECTION
    assert AnalyticalPhase.EXPERIMENT_SELECTION
    assert AnalyticalPhase.EXPERIMENT_EXECUTION
    assert AnalyticalPhase.VERIFICATION
    assert AnalyticalPhase.BELIEF_UPDATE
    assert AnalyticalPhase.REPLANNING
    assert AnalyticalPhase.STOPPING
    assert AnalyticalPhase.VERDICT
    assert AnalyticalPhase.DECISION_GUIDANCE
    assert AnalyticalPhase.FINALIZING

def test_replan_state_function_exists():
    assert callable(set_phase)
