"""v20-C4.2 (detection increment): proves
MethodSelectionEngine.detect_canonical_plan_role_disagreement correctly
flags a real disagreement between plan.semantics (the
UniversalQuestionCompiler's own binding object, which
admissibility_for_plan/gate_method_for_plan/roles_for_plan still consume
directly) and decision.estimand (the canonical role identity C3/C4/C4.1
hardened), and that it stays purely detection-only: it never changes
method_selection_scores/selected_method_code and never blocks execution.

See docs/AAOS_V20C4_2_CONTRACT_AUTHORITY_AUDIT.md for why this is scoped
as detection-only rather than a reconciliation.
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.method_selection import (
    EstimandSpec, MethodFamily, MethodSelectionDecision, MethodSelectionEngine,
    ObjectiveType, ProblemClass,
)
from packages.schemas.src.analysis import AggregationType, MetricRef, GrainRef


def _decision_with_estimand(target_column=None, predictor_columns=None, comparison_dimension=None, time_column=None):
    est = EstimandSpec(
        metric_ref=MetricRef(name=target_column or "revenue", table="orders", column=target_column or "revenue", aggregation=AggregationType.SUM),
        metric_definition=None,
        unit_of_analysis=GrainRef(table="orders", keys=[target_column or "revenue"]),
        comparison_dimension=comparison_dimension,
        time_column=time_column,
        target_column=target_column,
        predictor_columns=predictor_columns or [],
    )
    return MethodSelectionDecision(
        problem_class=ProblemClass.CORRELATIONAL, objective=ObjectiveType.COMPARE,
        estimand=est, admissible_families={MethodFamily.CORRELATION}, max_verdict_tier=None,
    )


def _plan(target_column=None, explanatory_columns=None, grouping_columns=None, time_column=None):
    semantics = types.SimpleNamespace(
        target_column=target_column, explanatory_columns=explanatory_columns or [],
        grouping_columns=grouping_columns or [], time_column=time_column,
    )
    return types.SimpleNamespace(task="ASSOCIATION", semantics=semantics)


class TestC42DetectionIncrement(unittest.TestCase):
    def test_no_conflicts_when_plan_and_estimand_agree(self):
        decision = _decision_with_estimand(target_column="sales", predictor_columns=["price"])
        plan = _plan(target_column="sales", explanatory_columns=["price"])
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertEqual(conflicts, [])

    def test_flags_target_column_disagreement(self):
        decision = _decision_with_estimand(target_column="price", predictor_columns=["sales"])
        plan = _plan(target_column="sales", explanatory_columns=["price"])
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertTrue(any("target_column" in c for c in conflicts))

    def test_flags_predictor_columns_disagreement(self):
        decision = _decision_with_estimand(target_column="sales", predictor_columns=["discount"])
        plan = _plan(target_column="sales", explanatory_columns=["price"])
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertTrue(any("predictor_columns" in c for c in conflicts))

    def test_flags_time_column_disagreement(self):
        decision = _decision_with_estimand(target_column="revenue", time_column="order_date")
        plan = _plan(target_column="revenue", time_column="ship_date")
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertTrue(any("time_column" in c for c in conflicts))

    def test_no_conflict_when_estimand_field_unset(self):
        # An unset estimand field (None / empty) is not itself a
        # disagreement -- only two actually-resolved, differing values are.
        decision = _decision_with_estimand(target_column="sales")
        plan = _plan(target_column="sales", explanatory_columns=["price"])
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertEqual(conflicts, [])

    def test_detection_never_mutates_the_decision(self):
        decision = _decision_with_estimand(target_column="price", predictor_columns=["sales"])
        plan = _plan(target_column="sales", explanatory_columns=["price"])
        before = (decision.selected_method_code, dict(decision.method_selection_scores), decision.estimand.target_column)
        MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        after = (decision.selected_method_code, dict(decision.method_selection_scores), decision.estimand.target_column)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
