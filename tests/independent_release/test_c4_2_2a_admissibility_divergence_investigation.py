"""v20-C4.2.2a: Admissibility/Scoring Divergence Investigation (no production
code change).

Answers, empirically, the question an external review of the C4.2.2
scoping doc raised: the C4.2 detector (detect_canonical_plan_role_
disagreement) only flags a conflict when BOTH plan.semantics and
decision.estimand are populated and disagree -- it says nothing about the
populated-vs-empty case (e.g. plan predictor=[] vs estimand
predictor=["price"]). Since admissibility_for_plan/gate_method_for_plan/
roles_for_plan read ONLY plan.semantics, "zero detector conflicts across
615+ tests" does not by itself prove plan.semantics and decision.estimand
are functionally equivalent for method selection purposes.

This file makes NO changes to select_for_plan(), roles_for_plan(),
gate_method_for_plan(), MethodRegistry.admissibility_for_plan(), or
detect_canonical_plan_role_disagreement() -- it is investigation-only,
exactly as the reviewed scoping decision requires.

Method: for each case, run the REAL pipeline (IntentEngine -> SemanticEngine
-> UniversalQuestionCompiler -> decide()) to get a genuine plan/semantic/df,
then call select_for_plan() TWICE against the *same* plan/semantic/df/quality
-- once with the unmodified decision, once with decision.estimand
deliberately overridden to diverge from plan.semantics in exactly one role
-- and diff selected_method_code/method_selection_scores/method_selection_
reasons between the two calls. Any difference would have to come from
decision.estimand's influence, since plan/semantic/df/quality are held
identical. This isolates the variable the review asked about.
"""
import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import (
    MethodRegistry, MethodSelectionEngine, gate_method_for_plan, roles_for_plan,
)
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal


def _df(n=120, seed=99):
    rng = np.random.RandomState(seed)
    price = rng.uniform(5, 50, n)
    annual_sales = 200 - 2 * price + rng.normal(0, 5, n)
    region = rng.choice(["East", "West"], n)
    order_date = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"price": price, "annual_sales": annual_sales, "region": region, "order_date": order_date})


def _pipeline(question, df):
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics)
    decision = MethodSelectionEngine.decide(intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles)
    return intent, semantic, plan, decision


def _run_select_for_plan(decision, plan, semantic, df):
    d = dataclasses.replace(decision) if dataclasses.is_dataclass(decision) else decision
    return MethodSelectionEngine.select_for_plan(d, plan, semantic, None, df)


class TestCaseA_PlanPredictorEmptyEstimandPopulated(unittest.TestCase):
    """plan.semantics.explanatory_columns = [] ; decision.estimand.predictor_columns = ["price"]."""

    def test_selected_method_unaffected_by_estimand_only_predictor(self):
        df = _df()
        question = "Is price associated with annual_sales?"
        intent, semantic, plan, decision = _pipeline(question, df)
        baseline = _run_select_for_plan(decision, plan, df=df, semantic=semantic)

        # Arrange the divergence: strip the compiler's own explanatory
        # columns while decision.estimand keeps its resolved predictor.
        arranged_plan_semantics = dataclasses.replace(plan.semantics, explanatory_columns=[])
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        arranged_decision = dataclasses.replace(decision)
        self.assertEqual(arranged_decision.estimand.predictor_columns, ["price"])

        result = MethodSelectionEngine.select_for_plan(arranged_decision, arranged_plan, semantic, None, df)

        # Corroborate at the gate_method_for_plan level directly too.
        self.assertEqual(gate_method_for_plan(plan, semantic, df), "association_numeric")
        self.assertIsNone(gate_method_for_plan(arranged_plan, semantic, df))

        # This is the actual finding: admissibility_for_plan/gate_method_for_plan/
        # roles_for_plan read ONLY plan.semantics, never decision.estimand --
        # so an emptied plan.semantics.explanatory_columns changes the
        # SELECTED METHOD (loses the "exposure" role, association_numeric
        # becomes inadmissible) regardless of what decision.estimand still
        # correctly says. decision.estimand's correct value has zero power
        # to prevent this.
        self.assertEqual(baseline.selected_method_code, "association_numeric")
        self.assertNotEqual(result.selected_method_code, baseline.selected_method_code)
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(arranged_decision, arranged_plan)
        # The detector does NOT catch this, exactly as the review predicted:
        # plan side is empty, so the "both populated" gate suppresses it.
        self.assertEqual(conflicts, [])


class TestCaseB_PlanTimeNoneEstimandPopulated(unittest.TestCase):
    """plan.semantics.time_column = None ; decision.estimand.time_column = "order_date"."""

    def test_forecast_role_lost_when_plan_time_column_cleared(self):
        df = _df()
        question = "Forecast annual_sales over time"
        intent, semantic, plan, decision = _pipeline(question, df)
        if decision.estimand.time_column != "order_date" or plan.semantics.time_column != "order_date":
            self.skipTest(f"compiler/canonical did not resolve order_date for this phrasing (plan={plan.semantics.time_column!r}, estimand={decision.estimand.time_column!r}); not this pass's scope to fix compiler resolution")

        baseline_roles = roles_for_plan(plan, semantic, df)
        arranged_plan_semantics = dataclasses.replace(plan.semantics, time_column=None)
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        arranged_roles = roles_for_plan(arranged_plan, semantic, df)

        self.assertIn("time", baseline_roles)
        self.assertNotIn("time", arranged_roles)
        # decision.estimand.time_column is untouched and still correct --
        # roles_for_plan never looks at it.
        self.assertEqual(decision.estimand.time_column, "order_date")
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, arranged_plan)
        self.assertEqual(conflicts, [])


class TestCaseC_PlanGroupingEmptyEstimandPopulated(unittest.TestCase):
    """plan.semantics.grouping_columns = [] ; decision.estimand.comparison_dimension = "region"."""

    def test_grouping_role_lost_when_plan_grouping_cleared(self):
        df = _df()
        question = "Compare annual_sales between regions"
        intent, semantic, plan, decision = _pipeline(question, df)
        if not plan.semantics.grouping_columns or decision.estimand.comparison_dimension != "region":
            self.skipTest(f"compiler/canonical did not resolve region as a comparison dimension for this phrasing (plan={plan.semantics.grouping_columns!r}, estimand={decision.estimand.comparison_dimension!r})")

        baseline_roles = roles_for_plan(plan, semantic, df)
        arranged_plan_semantics = dataclasses.replace(plan.semantics, grouping_columns=[])
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        arranged_roles = roles_for_plan(arranged_plan, semantic, df)

        self.assertIn("grouping", baseline_roles)
        self.assertNotIn("grouping", arranged_roles)
        self.assertEqual(decision.estimand.comparison_dimension, "region")


class TestCaseD_PlanTargetNoneEstimandPopulated(unittest.TestCase):
    """plan.semantics.target_column = None ; decision.estimand.target_column = "annual_sales"."""

    def test_outcome_role_and_method_lost_when_plan_target_cleared(self):
        df = _df()
        question = "Is price associated with annual_sales?"
        intent, semantic, plan, decision = _pipeline(question, df)
        baseline = _run_select_for_plan(decision, plan, df=df, semantic=semantic)

        arranged_plan_semantics = dataclasses.replace(plan.semantics, target_column=None)
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        result = MethodSelectionEngine.select_for_plan(dataclasses.replace(decision), arranged_plan, semantic, None, df)

        self.assertEqual(baseline.selected_method_code, "association_numeric")
        self.assertNotEqual(result.selected_method_code, baseline.selected_method_code)
        # decision.estimand.target_column is untouched and still correct.
        self.assertEqual(decision.estimand.target_column, "annual_sales")


class TestCaseE_SameNameTypeDisagreement(unittest.TestCase):
    """plan.semantics names a real column, but its declared identity
    disagrees with the dataframe's actual type -- does admissibility_for_plan
    notice, or trust the label?"""

    def test_target_declared_but_actually_high_cardinality_non_numeric_column(self):
        df = _df().assign(customer_name=[f"Cust-{i}" for i in range(len(_df()))])
        question = "Is price associated with annual_sales?"
        intent, semantic, plan, decision = _pipeline(question, df)
        # Point target_column at a high-cardinality, non-numeric column
        # (not a valid binary categorical target and not numeric).
        arranged_plan_semantics = dataclasses.replace(plan.semantics, target_column="customer_name")
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        ok, gate = MethodRegistry.admissibility_for_plan(arranged_plan, semantic, None, df)
        # Finding: same fail-open-to-generic-battery pattern as the
        # nonexistent-column case below -- `code` stays unset because the
        # target is neither numeric nor binary, so it falls through to
        # generic_diagnostic_battery and reports ok=True. Note the
        # asymmetry with roles: "outcome" IS added here (the column
        # genuinely exists in df, unlike the nonexistent-column case),
        # even though the actual method selected treats it as generic
        # rather than as a real numeric/binary outcome.
        self.assertTrue(ok)
        self.assertEqual(gate["method"], "generic_diagnostic_battery")
        self.assertIn("outcome", gate["roles"])

    def test_2level_categorical_target_is_correctly_treated_as_valid_binary_outcome(self):
        # Sanity contrast for the above: a genuinely 2-level categorical
        # column (e.g. region East/West) is NOT a "type disagreement" --
        # it is a legitimate binary target, and admissibility_for_plan
        # correctly routes it to association_categorical_binary. Verified
        # explicitly so the finding above isn't misread as "any non-numeric
        # target falls back to generic" -- only non-binary, non-numeric ones do.
        df = _df()
        question = "Is price associated with annual_sales?"
        intent, semantic, plan, decision = _pipeline(question, df)
        arranged_plan_semantics = dataclasses.replace(plan.semantics, target_column="region")
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        ok, gate = MethodRegistry.admissibility_for_plan(arranged_plan, semantic, None, df)
        self.assertTrue(ok)
        self.assertEqual(gate["method"], "association_categorical_binary")

    def test_target_column_not_present_in_dataframe_at_all(self):
        df = _df()
        question = "Is price associated with annual_sales?"
        intent, semantic, plan, decision = _pipeline(question, df)
        arranged_plan_semantics = dataclasses.replace(plan.semantics, target_column="nonexistent_column")
        arranged_plan = dataclasses.replace(plan, semantics=arranged_plan_semantics)
        # Must not raise -- admissibility_for_plan is a policy gate that has
        # to survive a plan naming a column absent from the actual data.
        try:
            ok, gate = MethodRegistry.admissibility_for_plan(arranged_plan, semantic, None, df)
        except Exception as exc:
            self.fail(f"admissibility_for_plan raised on a target_column absent from df: {exc!r}")
        # Actual, verified finding (not the initial hypothesis): this does
        # NOT get rejected. `code` stays unset when target isn't in
        # df.columns (the ASSOCIATION branch's inner `if target and target
        # in df.columns` never fires), and admissibility_for_plan's own
        # pre-existing "Fail closed: an executable plan is never outside
        # method policy" rule (`if not code: code = "generic_diagnostic_battery"`)
        # then makes it trivially admissible via the always-available
        # fallback battery -- not by validating the named target at all.
        # This is a legitimate, already-documented, intentional design
        # choice in the source (retreat to the broad diagnostic battery
        # rather than hard-reject), not a newly discovered defect. The
        # thing worth naming explicitly: `ok=True` here does not mean "the
        # requested ASSOCIATION analysis is admissible" -- it means "we
        # gave up on ASSOCIATION and fell back to the generic battery,"
        # and the gate's own payload still reports `task: "ASSOCIATION"`,
        # which could read as more specific than what will actually run.
        self.assertTrue(ok)
        self.assertEqual(gate["method"], "generic_diagnostic_battery")
        self.assertNotIn("outcome", gate["roles"])


class TestCaseF_TaskProblemClassDisagreement(unittest.TestCase):
    """plan.task and decision.problem_class disagree, outside the two
    specializations (SURVIVAL_CHURN, CAUSAL) select_for_plan() already
    guards. What actually happens to decision.problem_class?"""

    def test_non_specialized_problem_class_is_silently_overwritten_by_plan_task(self):
        df = _df()
        question = "Is price associated with annual_sales?"
        intent, semantic, plan, decision = _pipeline(question, df)
        self.assertEqual(plan.task, "ASSOCIATION")
        original_problem_class = decision.problem_class

        # Arrange decision.problem_class to say something plan.task disagrees
        # with, outside the SURVIVAL_CHURN/CAUSAL carve-outs (e.g. FORECASTING).
        from packages.analytics_core.src.engines.method_selection import ProblemClass
        arranged_decision = dataclasses.replace(decision, problem_class=ProblemClass.FORECASTING)
        result = MethodSelectionEngine.select_for_plan(arranged_decision, plan, semantic, None, df)

        # Finding: select_for_plan's own 1:1 canonical_task mapping silently
        # overwrites decision.problem_class back to whatever plan.task
        # dictates for any non-churn/non-causal disagreement -- there is no
        # ANALYTICAL_AUTHORITY_CONFLICT-style hard stop for this case at the
        # select_for_plan() level itself (that guard lives one layer up, in
        # controller.py, and only compares canonical_task vs analysis_plan.task
        # -- i.e. it checks select_for_plan() did its own job consistently,
        # not whether decide()'s original problem_class agreed with the plan).
        self.assertEqual(result.problem_class, ProblemClass.CORRELATIONAL)
        self.assertNotEqual(result.problem_class, ProblemClass.FORECASTING)
        self.assertNotEqual(original_problem_class, ProblemClass.FORECASTING)  # decide() itself agreed with plan.task originally


if __name__ == "__main__":
    unittest.main()
