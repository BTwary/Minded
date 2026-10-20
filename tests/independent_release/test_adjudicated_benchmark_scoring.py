import pytest
from scripts.summarize_adjudicated_benchmark import score_cases


def test_benchmark_reports_four_outcome_classes_separately():
    cases = [
        {"id": "a", "answerable": True, "expected": {"value": 4}},
        {"id": "b", "answerable": True, "expected": {"value": 7}},
        {"id": "c", "answerable": False, "expected": None},
        {"id": "d", "answerable": False, "expected": None},
    ]
    results = [
        {"id": "a", "outcome": "ANSWER", "actual": {"value": 4}},
        {"id": "b", "outcome": "SAFE_REFUSAL", "reason": "ambiguous"},
        {"id": "c", "outcome": "SAFE_REFUSAL", "reason": "unsupported"},
        {"id": "d", "outcome": "ANSWER", "actual": {"value": 2}},
    ]
    report = score_cases(cases, results, expected_count=4)
    assert report["counts"] == {
        "correct_answers": 1, "wrong_answers": 1,
        "safe_refusals": 1, "answerable_refusals": 1,
    }


def test_benchmark_refuses_missing_or_duplicate_cases():
    with pytest.raises(ValueError, match="exactly 2"):
        score_cases([], [], expected_count=2)
    cases = [{"id": "a", "answerable": True, "expected": 1}, {"id": "a", "answerable": True, "expected": 1}]
    with pytest.raises(ValueError, match="Duplicate"):
        score_cases(cases, [{"id": "a", "outcome": "ANSWER", "actual": 1}], expected_count=2)


def test_benchmark_refuses_incomplete_result_set():
    cases = [{"id": "a", "answerable": True, "expected": 1}, {"id": "b", "answerable": False, "expected": None}]
    with pytest.raises(ValueError, match="ID mismatch"):
        score_cases(cases, [{"id": "a", "outcome": "ANSWER", "actual": 1}], expected_count=2)
