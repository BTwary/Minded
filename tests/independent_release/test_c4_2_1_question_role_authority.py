"""v20-C4.2.1: explicit question roles participate in canonical semantic
resolution, and a REQUESTED variable is never silently replaced by a
DISCOVERED one.

Origin: the C4.2 detector fired on real traffic for
"Does plan tier affect cancellation rate?":

    plan.semantics.explanatory_columns  = ['plan_tier']
    decision.estimand.predictor_columns = ['tenure_months']

Root cause (verified against the persisted experiment record, not inferred):
IntentEngine labels "affect"/"influence" questions CORRELATION, and the
CORRELATION branch of MethodSelectionEngine.decide() bound the predictor to
SemanticEngine's discovered secondary_metric_col (tenure_months) without ever
seeing the question's own role. The only experiment that executed was
`SELECT cancellation_event, tenure_months ...` -- a churn-vs-tenure Pearson
test that says nothing about plan tier -- stamped with the method code
`association_categorical_binary`. The equivalent phrasing "Which plan tier has
higher cancellation rate?" (CHURN intent) on identical data ran the correct
plan_tier x vintage_year stratified experiments.

These tests pin the fixed behavior AND the deliberate limits of the fix.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import (
    EstimandSpec, MethodFamily, MethodSelectionEngine, ProblemClass,
)
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.semantic_resolution_builder import build_canonical_semantic_resolution
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal

QUESTION = "Does plan tier affect cancellation rate?"


def _confounded_df() -> pd.DataFrame:
    """Same Simpson's-paradox dataset test_defect_015 #4 uses."""
    rng = np.random.RandomState(7777)
    rows, cid = [], 1
    for plan, vintage, n, hazard in [
        ("Starter", "Vintage-2023", 400, 0.30), ("Starter", "Vintage-2025", 100, 0.05),
        ("Growth", "Vintage-2023", 100, 0.30), ("Growth", "Vintage-2025", 400, 0.05),
    ]:
        for _ in range(n):
            churn = int(rng.uniform() < hazard)
            rows.append({
                "account_id": f"ACC-{cid:05d}", "plan_tier": plan, "vintage_year": vintage,
                "tenure_months": int(rng.uniform(6, 36)), "cancellation_event": churn,
                "right_censored": 1 - churn,
            })
            cid += 1
    return pd.DataFrame(rows)


def _pipeline(question: str, df: pd.DataFrame, *, with_roles: bool = True):
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics) if with_roles else None
    decision = MethodSelectionEngine.decide(
        intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles,
    )
    return intent, semantic, plan, decision


class TestQuestionRoleProposal(unittest.TestCase):
    def test_only_question_referenced_columns_are_explicit(self):
        p = QuestionRoleProposal(
            explanatory_columns=["plan_tier", "tenure_months"],
            referenced_columns=["plan_tier", "cancellation_event"],
            target_column="cancellation_event",
        )
        self.assertEqual(p.explicit_explanatory_columns(), ["plan_tier"])
        self.assertEqual(p.inferred_explanatory_columns(), ["tenure_months"])

    def test_target_is_never_its_own_predictor(self):
        p = QuestionRoleProposal(["y", "x"], ["y", "x"], target_column="y")
        self.assertEqual(p.explicit_explanatory_columns(), ["x"])

    def test_missing_plan_semantics_yields_empty_proposal_not_error(self):
        p = QuestionRoleProposal.from_plan_semantics(None)
        self.assertEqual(p.explicit_explanatory_columns(), [])


class TestCanonicalRequestedVsDiscovered(unittest.TestCase):
    def setUp(self):
        self.df = _confounded_df()
        intent = IntentEngine.parse_intent(QUESTION, available_columns=list(self.df.columns))
        self.semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": self.df})
        self.plan = UniversalQuestionCompiler.compile(QUESTION, semantic=self.semantic, df=self.df)

    def test_compiler_names_plan_tier_for_the_discovered_case(self):
        self.assertEqual(list(self.plan.semantics.explanatory_columns), ["plan_tier"])

    def test_requested_and_discovered_are_distinct_fields(self):
        canon = build_canonical_semantic_resolution(
            self.semantic, "CORRELATION",
            question_roles=QuestionRoleProposal.from_plan_semantics(self.plan.semantics),
        )
        self.assertEqual(canon.requested_explanatory_columns(), ["plan_tier"])
        # The discovered secondary metric is still visible, but is NOT the
        # requested predictor.
        self.assertEqual(canon.secondary_metric_column(), "tenure_months")
        self.assertNotIn("tenure_months", canon.requested_explanatory_columns())

    def test_legacy_callers_without_roles_are_unchanged(self):
        canon = build_canonical_semantic_resolution(self.semantic, "CORRELATION")
        self.assertIsNone(canon.requested_explanatory)
        self.assertEqual(canon.requested_explanatory_columns(), [])
        self.assertEqual(canon.secondary_metric_column(), "tenure_months")

    def test_inferred_only_column_is_not_authoritative(self):
        roles = QuestionRoleProposal(
            explanatory_columns=["tenure_months"], referenced_columns=["cancellation_event"],
            target_column="cancellation_event",
        )
        canon = build_canonical_semantic_resolution(self.semantic, "CORRELATION", question_roles=roles)
        self.assertEqual(canon.requested_explanatory_columns(), [])
        self.assertEqual(
            list(canon.requested_explanatory.provenance["inferred_not_authoritative"]), ["tenure_months"],
        )


class TestDecideHonorsRequestedPredictor(unittest.TestCase):
    """The audit's most important regression, on the exact discovered case."""

    def setUp(self):
        self.df = _confounded_df()

    def test_intent_is_still_correlation_so_this_exercises_the_broken_branch(self):
        intent, *_ = _pipeline(QUESTION, self.df)
        self.assertEqual(intent.intent_type, "CORRELATION")

    def test_requested_predictor_is_plan_tier_not_tenure(self):
        _, _, plan, decision = _pipeline(QUESTION, self.df)
        self.assertEqual(list(plan.semantics.explanatory_columns), ["plan_tier"])
        self.assertEqual(decision.estimand.predictor_columns, ["plan_tier"])
        self.assertNotIn("tenure_months", decision.estimand.predictor_columns)

    def test_tenure_is_recorded_as_discovered_not_as_the_answer(self):
        _, _, _, decision = _pipeline(QUESTION, self.df)
        self.assertEqual(decision.estimand.discovered_columns, ["tenure_months"])

    def test_confounder_candidates_remain_available(self):
        _, semantic, _, decision = _pipeline(QUESTION, self.df)
        self.assertIn("vintage_year", list(semantic.churn_confounder_cols or []))
        self.assertIsNotNone(decision.estimand.churn_bindings)

    def test_not_routed_to_numeric_correlation_family(self):
        _, _, _, decision = _pipeline(QUESTION, self.df)
        self.assertNotEqual(decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertNotEqual(decision.method_family, MethodFamily.CORRELATION)

    def test_routes_to_same_family_as_equivalent_churn_phrasing(self):
        _, _, _, affect = _pipeline(QUESTION, self.df)
        _, _, _, which = _pipeline("Which plan tier has higher cancellation rate?", self.df)
        self.assertEqual(affect.problem_class, which.problem_class)
        self.assertEqual(affect.method_family, which.method_family)

    def test_no_plan_vs_estimand_disagreement_remains(self):
        _, _, plan, decision = _pipeline(QUESTION, self.df)
        self.assertEqual(MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan), [])

    def test_without_question_roles_legacy_behavior_is_preserved(self):
        """Callers that do not pass question_roles (older call sites) keep
        exactly the previous behavior -- the disagreement is still detectable
        there, which is the point of leaving the C4.2 detector in place."""
        _, _, plan, decision = _pipeline(QUESTION, self.df, with_roles=False)
        self.assertEqual(decision.estimand.predictor_columns, ["tenure_months"])
        self.assertTrue(MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan))


class TestFixIsDeliberatelyNarrow(unittest.TestCase):
    def test_numeric_requested_predictor_still_uses_correlation(self):
        df = pd.DataFrame({
            "price": [10.0, 12.0, 9.0, 15.0, 11.0, 13.0, 14.0, 10.5, 9.5, 12.5] * 3,
            "sales": [100, 120, 90, 150, 110, 130, 140, 105, 95, 125] * 3,
        })
        _, _, _, decision = _pipeline("Is price associated with sales?", df)
        self.assertEqual(decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertEqual(decision.method_family, MethodFamily.CORRELATION)
        self.assertEqual(decision.estimand.discovered_columns, [])


class TestPearsonGuardFailsClosedOnCategoricalPredictor(unittest.TestCase):
    def test_no_correlation_experiment_over_a_categorical_column(self):
        df = _confounded_df()
        _, semantic, _, decision = _pipeline(QUESTION, df)
        # Force the hazardous shape directly: a canonical decision that carries
        # a categorical predictor into the correlation experiment builder.
        est = EstimandSpec(target_column="cancellation_event", predictor_columns=["plan_tier"])
        fake = type("D", (), {"estimand": est})()
        hyp = PredictiveHypothesis(
            id="HYP-01", hypothesis_code="HYP-01", claim="c", mechanism="m",
            predicted_observables_if_true=["a"], predicted_observables_if_false=["b"],
            falsification_criteria="f", required_assumptions=[], prior_probability=0.5,
            posterior_probability=0.5, target_metric="cancellation_event",
            secondary_metric="plan_tier", claim_type="ASSOCIATION",
        )
        out = ExperimentSynthesizer._synthesize_correlation_experiments([hyp], semantic, decision=fake)
        self.assertEqual(out, [])


class TestEndToEndExecutedExperimentTestsRequestedVariable(unittest.TestCase):
    """The strongest form of the audit's assertion: 'Experiment uses plan_tier'.
    Checked against what was actually persisted and executed, because the
    pre-existing test_defect_015 #4 could not catch this -- it accepts the
    bare word 'inconclusive', which every inconclusive verdict contains."""

    def test_executed_experiments_test_plan_tier_not_tenure(self):
        from apps.api.src.core.database import SessionLocal
        from apps.api.src.models.entities import Experiment, InvestigationEvent
        from tests.independent_release.test_defect_015_forensic_completion import _run_forensic_investigation

        inv = _run_forensic_investigation(_confounded_df(), QUESTION)
        db = SessionLocal()
        try:
            exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
            self.assertTrue(exps, "no experiments executed")
            sqls = [str((e.arguments_json or {}).get("sql", "")) for e in exps]
            self.assertTrue(all("plan_tier" in q for q in sqls), sqls)
            self.assertFalse(any("tenure_months" in q for q in sqls), sqls)
            self.assertFalse(any(e.tool_name == "scipy_correlation" for e in exps))
            self.assertTrue(any("vintage_year" in q for q in sqls), "confounder stratification missing")

            events = db.query(InvestigationEvent).filter(InvestigationEvent.investigation_id == inv.id).all()
            self.assertFalse(
                [e for e in events if e.event_type == "investigation.canonical_plan_role_disagreement"],
                "role disagreement still recorded",
            )
            ms = [e for e in events if e.event_type == "investigation.method_selection"]
            self.assertTrue(ms)
            est = ms[-1].event_payload_json["estimand"]
            self.assertEqual(est["predictor_columns"], ["plan_tier"])
            self.assertEqual(est["discovered_columns"], ["tenure_months"])
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
