from pathlib import Path

CONTROLLER = Path(__file__).parents[1] / 'packages/analytics_core/src/runtime/controller.py'


def test_specialist_path_is_fail_closed_until_canonical_transition_exists():
    text = CONTROLLER.read_text()
    start = text.index('# SPECIALIST AUTHORITY GATE')
    end = text.index('# Build World Model AST', start)
    block = text[start:end]
    assert 'specialist_requires_canonical_transition' in block
    assert 'InvestigationVerdict(' not in block
    assert 'EvidenceVerification(' not in block
    assert 'complete_execution' not in block
    assert '_complete_inconclusive(' in block


def test_specialist_path_cannot_claim_counter_hypothesis_or_scientific_stop():
    text = CONTROLLER.read_text()
    start = text.index('# SPECIALIST AUTHORITY GATE')
    end = text.index('# Build World Model AST', start)
    block = text[start:end]
    assert 'counter_hypothesis_refuted=True' not in block
    assert 'StoppingEngine.evaluate_stopping(' not in block
