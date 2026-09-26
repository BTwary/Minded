from pathlib import Path

SPECIAL = Path(__file__).parents[1] / "packages/analytics_core/src/intelligence/universal_specialized.py"


def test_specialists_do_not_emit_authoritative_verdict_or_confidence_fields():
    text = SPECIAL.read_text()
    forbidden = [
        '"verdict": "OBSERVED"',
        '"confidence": 0.85',
        '"confidence": 0.90',
        'confidence = 0.95 if pct',
        'confidence = max(0.0, min(1.0',
    ]
    for marker in forbidden:
        assert marker not in text


def test_specialist_validation_defaults_to_unverified():
    text = SPECIAL.read_text()
    assert '"validation": "UNVERIFIED"' in text
