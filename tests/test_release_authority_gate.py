from pathlib import Path

ROOT = Path(__file__).parents[1]
CONTROLLER = ROOT / 'packages/analytics_core/src/runtime/controller.py'
SPECIAL = ROOT / 'packages/analytics_core/src/intelligence/universal_specialized.py'


def test_specialists_cannot_persist_as_authoritative_verdicts():
    text = CONTROLLER.read_text()
    start = text.index('# SPECIALIST AUTHORITY GATE')
    end = text.index('# Build World Model AST', start)
    block = text[start:end]
    assert 'InvestigationVerdict(' not in block
    assert 'EvidenceVerification(' not in block
    assert 'confidence_score=float(special' not in block
    assert 'complete_execution' not in block
    assert 'specialist_requires_canonical_transition' in block


def test_prediction_never_guesses_positive_class():
    text = SPECIAL.read_text()
    assert 'positive_class: Any = None' in text
    assert 'if positive_class is None:' in text
    assert 'TARGET_SEMANTICS_UNRESOLVED' in text


def test_no_specialist_confidence_from_auc():
    text = SPECIAL.read_text()
    assert '0.5 + 0.5 * max(float(best_auc)' not in text
    assert '"confidence": None' in text
