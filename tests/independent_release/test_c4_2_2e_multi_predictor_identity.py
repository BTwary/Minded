"""v20-C4.2.2e: Multi-Predictor Analytical Identity and Pairwise Execution.

Verification suite for C4.2.2e closing the full analytical chain for
multi-predictor natural language questions:
  Question
  -> Lossless QuestionRoleProposal.requested_explanatory
  -> CanonicalSemanticResolution.requested_explanatory_columns()
  -> EstimandSpec.predictor_columns
  -> Pairwise PredictiveHypotheses (HYP-01, HYP-02, ...)
  -> Pairwise CandidateExperiments (EXP-CORR-01, EXP-CORR-02, ...)
  -> Deterministic calculations & dual-engine verification
  -> Evidence ledger claims with matching source columns
"""
import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import duckdb
import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import (
    EstimandSpec, MethodFamily, MethodSelectionEngine, ProblemClass,
)
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.engines.evidence_ledger import EvidenceLedger
from packages.schemas.src.analysis import EpistemicClaimType
from packages.analytics_core.src.intelligence.experiment_synthesizer import (
    CandidateExperiment, ExperimentSynthesizer,
)
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    HypothesisSynthesizer, PredictiveHypothesis,
)
from packages.analytics_core.src.intelligence.semantic_resolution_builder import (
    build_canonical_semantic_resolution,
)
from packages.analytics_core.src.intelligence.transition import ScientificTransitionService
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.schemas.src.semantic_resolution_contract import (
    CanonicalSemanticResolution, QuestionRoleProposal,
)


def _multi_df(n: int = 150, seed: int = 42):
    rng = np.random.RandomState(seed)
    price = rng.uniform(10, 100, n)
    marketing_spend = rng.uniform(200, 5000, n)
    discount = rng.uniform(0, 0.3, n)
    annual_sales = 500 - 2.5 * price + 0.05 * marketing_spend - 50 * discount + rng.normal(0, 5, n)
    region = rng.choice(["North", "South", "East", "West"], size=n)
    category = rng.choice(["Electronics", "Apparel", "Home"], size=n)
    return pd.DataFrame({
        "price": price,
        "marketing_spend": marketing_spend,
        "discount": discount,
        "annual_sales": annual_sales,
        "region": region,
        "category": category,
    })


def _compile_and_decide(question: str, df: pd.DataFrame):
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics)
    decision = MethodSelectionEngine.decide(
        intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles,
    )
    return intent, semantic, plan, roles, decision


class TestMultiPredictorQuestionIntelligence(unittest.TestCase):
    """Verifies that multi-predictor bivariate association phrasing is protected
    from erroneous clause splitting and preserves all named predictors."""

    def test_multi_predictor_not_split_into_multiple_clauses(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        self.assertNotIn("compound_question_requires_clause_preserving_analysis", plan.unresolved_questions)
        self.assertEqual(plan.semantics.explanatory_columns, ["price", "marketing_spend"])

    def test_three_predictors_preserved_in_plan_semantics(self):
        df = _multi_df()
        q = "Are price, marketing_spend, and discount associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        self.assertEqual(plan.semantics.explanatory_columns, ["price", "marketing_spend", "discount"])


class TestMultiPredictorContractIdentity(unittest.TestCase):
    """Verifies lossless propagation through QuestionRoleProposal,
    CanonicalSemanticResolution, and EstimandSpec."""

    def test_role_proposal_preserves_multiple_predictors(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        _, _, plan, roles, _ = _compile_and_decide(q, df)
        self.assertEqual(list(roles.requested_explanatory), ["price", "marketing_spend"])
        self.assertEqual(roles.deterministic_requested_explanatory(), ["marketing_spend", "price"])

    def test_canonical_resolution_preserves_multiple_predictors(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        _, semantic, _, roles, _ = _compile_and_decide(q, df)
        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        self.assertEqual(canonical.requested_explanatory_columns(), ["price", "marketing_spend"])
        self.assertEqual(canonical.deterministic_requested_explanatory(), ["marketing_spend", "price"])

    def test_estimand_spec_preserves_multiple_predictors(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        _, _, _, _, decision = _compile_and_decide(q, df)
        est = decision.estimand
        self.assertEqual(est.target_column, "annual_sales")
        self.assertEqual(est.predictor_columns, ["price", "marketing_spend"])
        self.assertEqual(est.deterministic_predictor_columns(), ["marketing_spend", "price"])


class TestMultiPredictorHypothesisSynthesis(unittest.TestCase):
    """Verifies that 2 predictors produce 2 pairwise hypotheses (HYP-01, HYP-02)."""

    def test_two_pairwise_hypotheses_synthesized(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        self.assertEqual(len(hyps), 2)
        h1, h2 = hyps[0], hyps[1]
        self.assertEqual(h1.hypothesis_code, "HYP-01")
        self.assertEqual(h1.target_metric, "annual_sales")
        self.assertEqual(h1.secondary_metric, "price")
        self.assertIn("price", h1.claim)
        self.assertIn("annual_sales", h1.claim)

        self.assertEqual(h2.hypothesis_code, "HYP-02")
        self.assertEqual(h2.target_metric, "annual_sales")
        self.assertEqual(h2.secondary_metric, "marketing_spend")
        self.assertIn("marketing_spend", h2.claim)
        self.assertIn("annual_sales", h2.claim)

        self.assertFalse(h1.is_counter_hypothesis)
        self.assertFalse(h2.is_counter_hypothesis)


class TestMultiPredictorExperimentSynthesis(unittest.TestCase):
    """Verifies that 2 predictors produce 2 pairwise experiments (EXP-CORR-01, EXP-CORR-02)."""

    def test_two_pairwise_experiments_synthesized(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )
        self.assertEqual(len(exps), 2)
        e1, e2 = exps[0], exps[1]

        self.assertEqual(e1.code, "EXP-CORR-01")
        self.assertEqual(e1.target_hypothesis_code, "HYP-01")
        self.assertEqual(e1.metrics, ["annual_sales", "price"])
        self.assertIn("price", e1.query_sql)
        self.assertIn("annual_sales", e1.query_sql)
        self.assertNotIn("marketing_spend", e1.query_sql)

        self.assertEqual(e2.code, "EXP-CORR-02")
        self.assertEqual(e2.target_hypothesis_code, "HYP-02")
        self.assertEqual(e2.metrics, ["annual_sales", "marketing_spend"])
        self.assertIn("marketing_spend", e2.query_sql)
        self.assertIn("annual_sales", e2.query_sql)
        self.assertNotIn("price", e2.query_sql)


class TestThreePredictorScenario(unittest.TestCase):
    """Verifies scalability to 3 predictors (3 requested -> 3 hyps -> 3 exps)."""

    def test_three_predictors_end_to_end_synthesis(self):
        df = _multi_df()
        q = "Are price, marketing_spend, and discount associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        self.assertEqual(decision.estimand.predictor_columns, ["price", "marketing_spend", "discount"])

        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        self.assertEqual(len(hyps), 3)
        self.assertEqual([h.secondary_metric for h in hyps], ["price", "marketing_spend", "discount"])
        self.assertEqual([h.hypothesis_code for h in hyps], ["HYP-01", "HYP-02", "HYP-03"])

        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )
        self.assertEqual(len(exps), 3)
        self.assertEqual([e.code for e in exps], ["EXP-CORR-01", "EXP-CORR-02", "EXP-CORR-03"])
        self.assertEqual(exps[0].metrics, ["annual_sales", "price"])
        self.assertEqual(exps[1].metrics, ["annual_sales", "marketing_spend"])
        self.assertEqual(exps[2].metrics, ["annual_sales", "discount"])


class TestSinglePredictorBackwardCompatibility(unittest.TestCase):
    """Verifies that 1 predictor produces the exact legacy 1-hypothesis-pair (H1, H0)
    and 1-experiment (EXP-CORR) structure."""

    def test_single_predictor_backward_compatibility(self):
        df = _multi_df()
        q = "Is price associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        self.assertEqual(decision.estimand.predictor_columns, ["price"])

        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        self.assertEqual(len(hyps), 2)
        self.assertEqual(hyps[0].hypothesis_code, "HYP-01")
        self.assertFalse(hyps[0].is_counter_hypothesis)
        self.assertEqual(hyps[1].hypothesis_code, "HYP-02")
        self.assertTrue(hyps[1].is_counter_hypothesis)

        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0].code, "EXP-CORR")
        self.assertEqual(exps[0].metrics, ["annual_sales", "price"])


class TestDuplicatePredictorDeduplication(unittest.TestCase):
    """Verifies that duplicate explicit predictors are deduplicated deterministically."""

    def test_duplicate_predictors_deduplicate(self):
        df = _multi_df()
        q = "Are price and price associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        self.assertEqual(decision.estimand.predictor_columns, ["price"])

        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        self.assertEqual(len(hyps), 2)  # H1 and H0 for single predictor price

        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0].code, "EXP-CORR")


class TestInvalidPredictorGating(unittest.TestCase):
    """Verifies fail-closed gating:
    - One invalid predictor does not suppress valid predictors.
    - All invalid predictors fail closed."""

    def test_invalid_predictor_does_not_suppress_valid_predictor(self):
        df = _multi_df()
        q = "Are price and region associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        # "price" is valid numeric, "region" is categorical
        self.assertEqual(decision.estimand.predictor_columns, ["price"])
        self.assertTrue(any("region" in lim for lim in decision.estimand.limitations))

        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        # Because only 1 valid predictor remains, synthesizes single predictor pair
        self.assertEqual(len(hyps), 2)
        self.assertEqual(hyps[0].secondary_metric, "price")

        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0].code, "EXP-CORR")
        self.assertEqual(exps[0].metrics, ["annual_sales", "price"])

    def test_all_invalid_predictors_fail_closed(self):
        df = _multi_df()
        q = "Are region and category associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        self.assertEqual(decision.estimand.predictor_columns, [])
        self.assertEqual(decision.problem_class, ProblemClass.DIAGNOSTIC)
        self.assertTrue(any("not a valid predictor" in lim.lower() for lim in decision.estimand.limitations))


class TestRealDuckDBExecutionAndVerification(unittest.TestCase):
    """Executes the 2 synthesized experiments against DuckDB on the actual data,
    verifies deterministic statistical calculation and evidence trace creation."""

    def test_duckdb_execution_and_transition_verification(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )

        con = duckdb.connect(":memory:")
        con.register("data_table", df)

        # Run EXP-CORR-01 (price)
        exp1 = exps[0]
        res1_df = con.execute(exp1.query_sql).fetchdf()
        self.assertEqual(list(res1_df.columns), ["annual_sales", "price"])
        self.assertEqual(len(res1_df), 150)

        state_mgr1 = InvestigationStateManager("INV-TEST-1")
        for h in hyps:
            state_mgr1.create_hypothesis(h)

        step1 = ScientificTransitionService.apply_post_execution_transition(
            state_mgr=state_mgr1,
            experiment_id=exp1.code,
            experiment_code=exp1.code,
            target_prediction_ids=[],
            primary_df=df,
            result_df=res1_df,
            target_metric_col=exp1.target_metric,
            aggregation_type=exp1.aggregation_type,
            primary_value=0.0,
            target_hypothesis_code=exp1.target_hypothesis_code,
            semantic=semantic,
        )
        self.assertIsNotNone(step1.correlation_r)
        self.assertIsNotNone(step1.correlation_p_value)
        self.assertEqual(step1.correlation_n, 150)
        # Negative association between price and sales:
        self.assertLess(step1.correlation_r, 0)
        self.assertLess(step1.correlation_p_value, 0.05)

        # Run EXP-CORR-02 (marketing_spend)
        exp2 = exps[1]
        res2_df = con.execute(exp2.query_sql).fetchdf()
        self.assertEqual(list(res2_df.columns), ["annual_sales", "marketing_spend"])
        self.assertEqual(len(res2_df), 150)

        state_mgr2 = InvestigationStateManager("INV-TEST-2")
        for h in hyps:
            state_mgr2.create_hypothesis(h)

        step2 = ScientificTransitionService.apply_post_execution_transition(
            state_mgr=state_mgr2,
            experiment_id=exp2.code,
            experiment_code=exp2.code,
            target_prediction_ids=[],
            primary_df=df,
            result_df=res2_df,
            target_metric_col=exp2.target_metric,
            aggregation_type=exp2.aggregation_type,
            primary_value=0.0,
            target_hypothesis_code=exp2.target_hypothesis_code,
            semantic=semantic,
        )
        self.assertIsNotNone(step2.correlation_r)
        self.assertIsNotNone(step2.correlation_p_value)
        self.assertEqual(step2.correlation_n, 150)
        # Positive association between marketing_spend and sales:
        self.assertGreater(step2.correlation_r, 0)
        self.assertLess(step2.correlation_p_value, 0.05)

        con.close()

    def test_evidence_ledger_records_distinct_predictor_columns(self):
        """Verifies that evidence claims record the exact predictor in source_columns."""
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, q, intent=intent, canonical_semantics=canonical, decision=decision,
        )
        exps = ExperimentSynthesizer.synthesize_candidate_experiments(
            hyps, semantic, decision=decision,
        )

        ledger = EvidenceLedger("INV-TEST-LEDGER")
        canonical_target_col = canonical.outcome_column()

        for exp in exps:
            src_cols = list(dict.fromkeys([
                c for c in (
                    [canonical_target_col]
                    + list(getattr(exp, "metrics", []) or [])
                ) if c
            ]))
            ledger.record_claim(
                claim_statement=f"Association test for {exp.metrics}",
                claim_type=EpistemicClaimType.ASSOCIATION,
                source_experiment_id=exp.code,
                source_datasets=["data_table"],
                source_columns=src_cols,
                row_count_evaluated=150,
                computation_proof={"primary_value": 0.0},
            )

        claims = ledger.get_all_claims()
        self.assertEqual(len(claims), 2)
        claim1 = next(c for c in claims if c.source_experiment_id == "EXP-CORR-01")
        claim2 = next(c for c in claims if c.source_experiment_id == "EXP-CORR-02")

        self.assertEqual(claim1.source_columns, ["annual_sales", "price"])
        self.assertEqual(claim2.source_columns, ["annual_sales", "marketing_spend"])


class TestNoJointMultivariableInScope(unittest.TestCase):
    """Verifies that joint multivariable modeling remains strictly out of scope
    for C4.2.2e and that Option A pairwise identity is maintained."""

    def test_no_joint_multivariable_method_exists(self):
        from packages.analytics_core.src.engines.method_selection import MethodRegistry
        capabilities = getattr(MethodRegistry, "CAPABILITIES", {})
        multivariable_codes = [
            c for c in capabilities.keys()
            if "multivariable" in c.lower() or "multiple_regression" in c.lower() or "joint_regression" in c.lower()
        ]
        self.assertEqual(multivariable_codes, [])

    def test_select_for_plan_executes_association_numeric(self):
        df = _multi_df()
        q = "Are price and marketing_spend associated with annual_sales?"
        intent, semantic, plan, roles, decision = _compile_and_decide(q, df)
        result = MethodSelectionEngine.select_for_plan(dataclasses.replace(decision), plan, semantic, None, df)
        self.assertEqual(result.analytical_task_status, "IMPLEMENTED")
        self.assertEqual(result.selected_method_code, "association_numeric")


if __name__ == "__main__":
    unittest.main()
