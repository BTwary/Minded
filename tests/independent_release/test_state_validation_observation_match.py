"""Regression test: `validate_state`'s "prediction marked resolved but no
observation" check (DEFECT-013-adjacent) must require a resolved
prediction's target_experiment_id to match a *real* raw_observations entry,
not just be non-empty.

This exact fix was applied once before (see prior session notes /
BUGFIXES_2026-09-09_v4.md) but the check had reverted to its weaker,
pre-fix form -- `observation_exp_ids` was computed and then never
consulted. Verified here directly against both a failing case (stale
target_experiment_id with no matching observation) and a passing case
(target_experiment_id that matches a real raw_observations entry).
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packages.schemas.src.analysis import (
    CanonicalInvestigationState,
    EvidenceLedgerSchema,
    HypothesisSchema,
    PredictionSchema,
    PredictionStatus,
    RawObservationRecord,
)
from packages.analytics_core.src.intelligence.state_validation import validate_state


def _base_state(**overrides):
    hyp = HypothesisSchema(id="HYP-01", statement="stmt", rationale="why", priority=0.5)
    kwargs = dict(
        investigation_id="INV-1",
        original_question="q",
        hypotheses=[hyp],
        evidence_ledger=EvidenceLedgerSchema(investigation_id="INV-1"),
        predictions=[],
        raw_observations=[],
    )
    kwargs.update(overrides)
    return CanonicalInvestigationState(**kwargs)


def test_resolved_prediction_with_stale_target_experiment_id_fails_closed():
    pred = PredictionSchema(
        prediction_id="PRED-01",
        hypothesis_id="HYP-01",
        statement="stmt",
        status=PredictionStatus.SUPPORTED,
        target_experiment_id="EXP-DOES-NOT-EXIST",
        actual_observed_result=None,
    )
    state = _base_state(predictions=[pred], raw_observations=[])
    report = validate_state(state, runtime_hypotheses={"HYP-01": object()}, runtime_predictions={"PRED-01": object()})
    codes = [e.error_code for e in report.errors]
    assert "prediction_marked_resolved_but_no_observation" in codes, (
        "resolved prediction with a target_experiment_id that matches no real "
        "observation must be flagged, not silently accepted"
    )


def test_resolved_prediction_with_real_observation_passes():
    obs = RawObservationRecord(observation_id="OBS-01", experiment_id="EXP-REAL")
    pred = PredictionSchema(
        prediction_id="PRED-02",
        hypothesis_id="HYP-01",
        statement="stmt",
        status=PredictionStatus.SUPPORTED,
        target_experiment_id="EXP-REAL",
        actual_observed_result=None,
    )
    state = _base_state(predictions=[pred], raw_observations=[obs])
    report = validate_state(state, runtime_hypotheses={"HYP-01": object()}, runtime_predictions={"PRED-02": object()})
    codes = [e.error_code for e in report.errors]
    assert "prediction_marked_resolved_but_no_observation" not in codes, (
        "resolved prediction whose target_experiment_id matches a real "
        "observation must not be flagged"
    )


def test_resolved_prediction_with_inline_observed_result_passes():
    pred = PredictionSchema(
        prediction_id="PRED-03",
        hypothesis_id="HYP-01",
        statement="stmt",
        status=PredictionStatus.REFUTED,
        target_experiment_id=None,
        actual_observed_result={"value": 1.0},
    )
    state = _base_state(predictions=[pred], raw_observations=[])
    report = validate_state(state, runtime_hypotheses={"HYP-01": object()}, runtime_predictions={"PRED-03": object()})
    codes = [e.error_code for e in report.errors]
    assert "prediction_marked_resolved_but_no_observation" not in codes


if __name__ == "__main__":
    test_resolved_prediction_with_stale_target_experiment_id_fails_closed()
    test_resolved_prediction_with_real_observation_passes()
    test_resolved_prediction_with_inline_observed_result_passes()
    print("PASS: resolved-prediction observation check requires a real matching observation.")
