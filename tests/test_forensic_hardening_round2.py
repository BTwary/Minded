from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / 'packages/analytics_core/src/engines/verification.py'
TRANS = ROOT / 'packages/analytics_core/src/intelligence/transition.py'
PRED = ROOT / 'packages/analytics_core/src/intelligence/prediction_engine.py'
CTRL = ROOT / 'packages/analytics_core/src/runtime/controller.py'


def test_verification_failures_have_no_fabricated_measurements():
    text = VER.read_text(encoding='utf-8')
    assert 'secondary_value=0.0' not in text
    assert 'observed_delta_pct=1.0' not in text
    assert 'secondary_value=None' in text
    assert 'observed_delta_pct=None' in text


def test_prediction_has_no_epistemic_confidence_defaults():
    text = PRED.read_text(encoding='utf-8')
    assert 'confidence: Optional[float] = None' in text
    assert 'confidence=0.75' not in text
    assert 'confidence=0.70' not in text
    assert 'confidence=min(0.9' not in text
    assert 'confidence=min(0.85' not in text


def test_statistical_failure_never_becomes_eta_15():
    text = TRANS.read_text(encoding='utf-8')
    assert 'eta_sq = 15.0' not in text
    assert 'eta_sq = 25.0' not in text
    assert 'eta_sq = None' in text
    assert '"status": "NUMERICAL_FAILURE"' in text


def test_churn_semantic_binding_is_explicit():
    text = TRANS.read_text(encoding='utf-8')
    assert 'getattr(semantic, "churn_event_col", None)' in text
    assert 'resolved_churn_event = None' not in text


def test_unsupported_specialists_fail_closed():
    text = CTRL.read_text(encoding='utf-8')
    assert 'specialist_requires_canonical_transition' in text
    assert 'has no canonical transition executor; ' in text
    assert 'no authoritative specialist result was produced.' in text
