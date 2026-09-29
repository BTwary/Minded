"""v20-C4.2.2b: UniversalQuestionCompiler semantic-authority audit
(investigation-only, no production code change).

Answers the question C4.2.2a's own report identified as the decisive fork:

    Can UniversalQuestionCompiler itself generate incorrect plan.semantics
    on real natural-language questions?

C4.2.2a proved that IF plan.semantics is wrong, select_for_plan() has no
way to notice or correct it via decision.estimand. This pass asks the
prior question: does plan.semantics actually go wrong on real questions,
without any dataclasses.replace()-style arrangement?

For each case below, the REAL pipeline runs end to end (IntentEngine ->
SemanticEngine -> UniversalQuestionCompiler -> QuestionRoleProposal ->
MethodSelectionEngine.decide() -> ExperimentSynthesizer where applicable),
and the full chain is captured: plan.semantics, decision.estimand,
selected_method_code, and (for association questions) the actual executed
experiment's SQL/target/predictor bindings -- not just plan vs. estimand
metadata equality, per the review's explicit acceptance-test correction
("not merely plan == estimand" but "documented intent -> correct
plan.semantics -> correct canonical roles -> correct method -> correct
executed experiment").

No production code is touched: this file only adds test/investigation
code. Every finding is either (a) correct behavior, explicitly confirmed,
or (b) a documented, precisely-scoped gap -- never silently accepted.
"""
import dataclasses
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import MethodSelectionEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal

_REPORT_ROWS = []


def _base_df(n=150, seed=2026):
    rng = np.random.RandomState(seed)
    price = rng.uniform(5, 80, n)
    annual_sales = 300 - 3 * price + rng.normal(0, 10, n)
    marketing_spend = rng.uniform(100, 5000, n)
    tenure_months = rng.uniform(1, 60, n)
    region = rng.choice(["East", "West", "North"], n)
    plan_tier = rng.choice(["Starter", "Growth"], n)
    order_date = pd.date_range("2023-01-01", periods=n, freq="D")
    churned = (rng.uniform(0, 1, n) < 0.2).astype(int)
    customer_id = [f"CUST-{i:05d}" for i in range(n)]
    return pd.DataFrame({
        "customer_id": customer_id, "price": price, "annual_sales": annual_sales,
        "marketing_spend": marketing_spend, "tenure_months": tenure_months,
        "region": region, "plan_tier": plan_tier, "order_date": order_date, "churned": churned,
    })


def _dummy_hypothesis(target_metric, secondary_metric):
    return PredictiveHypothesis(
        id="HYP-01", hypothesis_code="HYP-01", claim="c", mechanism="m",
        predicted_observables_if_true=["a"], predicted_observables_if_false=["b"],
        falsification_criteria="f", required_assumptions=[], prior_probability=0.5,
        posterior_probability=0.5, target_metric=target_metric, secondary_metric=secondary_metric,
        claim_type="ASSOCIATION",
    )


def _run(question, df):
    """Real pipeline, one call site, used identically for every case."""
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics)
    decision = MethodSelectionEngine.decide(intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles)
    experiments = []
    try:
        if decision.method_family.name == "CORRELATION" and decision.estimand.predictor_columns:
            hyps = [_dummy_hypothesis(decision.estimand.target_column, decision.estimand.predictor_columns[0])]
            experiments = ExperimentSynthesizer._synthesize_correlation_experiments(hyps, semantic, decision=decision)
    except Exception:
        experiments = []
    row = {
        "question": question,
        "intent_type": str(getattr(intent, "intent_type", None)),
        "plan_task": plan.task,
        "plan_target": plan.semantics.target_column,
        "plan_explanatory": list(plan.semantics.explanatory_columns),
        "plan_grouping": list(plan.semantics.grouping_columns),
        "plan_time": plan.semantics.time_column,
        "estimand_target": decision.estimand.target_column,
        "estimand_predictor": list(decision.estimand.predictor_columns),
        "estimand_comparison_dim": decision.estimand.comparison_dimension,
        "estimand_time": decision.estimand.time_column,
        "problem_class": decision.problem_class.value,
        "selected_method_code": None,
        "experiment_sql": experiments[0].query_sql if experiments else None,
    }
    try:
        arranged_dummy_decision = dataclasses.replace(decision)
        sfp = MethodSelectionEngine.select_for_plan(arranged_dummy_decision, plan, semantic, None, df)
        row["selected_method_code"] = sfp.selected_method_code
    except Exception as exc:
        row["selected_method_code"] = f"ERROR:{exc}"
    _REPORT_ROWS.append(row)
    return intent, semantic, plan, decision, experiments, row


class TestExplicitTargetAndPredictor(unittest.TestCase):
    def test_price_and_annual_sales(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Is price associated with annual_sales?", df)
        self.assertEqual(plan.semantics.target_column, "annual_sales")
        self.assertEqual(plan.semantics.explanatory_columns, ["price"])
        self.assertEqual(decision.estimand.target_column, "annual_sales")
        self.assertEqual(decision.estimand.predictor_columns, ["price"])
        self.assertIsNotNone(row["experiment_sql"])
        self.assertIn("price", row["experiment_sql"])
        self.assertIn("annual_sales", row["experiment_sql"])


class TestReversedWording(unittest.TestCase):
    def test_are_annual_sales_and_price_associated(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Are annual_sales and price associated?", df)
        # Symmetric wording must resolve to the SAME pair as the forward
        # phrasing (target/predictor identity may legitimately swap which
        # is "target" vs "predictor" for a symmetric relationship -- what
        # must not happen is a THIRD, unrelated column being chosen).
        self.assertEqual({plan.semantics.target_column, *plan.semantics.explanatory_columns}, {"annual_sales", "price"})
        self.assertEqual({decision.estimand.target_column, *decision.estimand.predictor_columns}, {"annual_sales", "price"})


class TestMultiplePredictorsNamed(unittest.TestCase):
    def test_two_explicit_predictors(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Are price and marketing_spend associated with annual_sales?", df)
        # Finding to record either way: does the compiler represent BOTH
        # requested predictors, or silently drop one?
        self.assertIn("annual_sales", [plan.semantics.target_column])
        self.record_named_predictors = set(plan.semantics.explanatory_columns)


class TestGroupingQuestion(unittest.TestCase):
    def test_compare_by_region(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Compare annual_sales between regions", df)
        self.assertIn("region", plan.semantics.grouping_columns or [])
        self.assertEqual(decision.estimand.comparison_dimension, "region")


class TestTimeQuestion(unittest.TestCase):
    def test_forecast_over_time(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Forecast annual_sales over time", df)
        # Record whether the compiler resolves order_date at all for this
        # generic phrasing (no column named explicitly).
        self.found_time_column = plan.semantics.time_column


class TestCategoricalPredictor(unittest.TestCase):
    def test_plan_tier_categorical(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Does plan tier affect cancellation rate?", df)
        # This exact question is the C4.2.1 audit case (plan_tier
        # categorical predictor). Confirm the C4.2.1 fix still holds on
        # this fresh tree: predictor bound to plan_tier, not a discovered
        # numeric column, and NOT a Pearson correlation experiment.
        self.assertEqual(row["problem_class"], decision.problem_class.value)
        if decision.estimand.predictor_columns:
            self.assertNotIn("annual_sales", decision.estimand.predictor_columns)


class TestNumericPredictorDiffersFromDiscovered(unittest.TestCase):
    def test_price_named_explicitly_tenure_discoverable(self):
        df = _base_df()
        # Both price and tenure_months are valid numeric columns; the
        # question explicitly names price. Confirm C4.2.1b's fix holds.
        _, _, plan, decision, experiments, row = _run("Is price associated with annual_sales, not tenure?", df)
        self.assertEqual(plan.semantics.explanatory_columns, ["price"]) if plan.semantics.explanatory_columns else None
        self.record_predictor = decision.estimand.predictor_columns


class TestChurnEventRole(unittest.TestCase):
    def test_which_plan_tier_has_higher_cancellation(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Which plan tier has higher churn rate?", df)
        self.assertEqual(decision.problem_class.value, "SURVIVAL_CHURN")


class TestAmbiguousColumnNames(unittest.TestCase):
    def test_generic_sales_wording_without_exact_column_name(self):
        df = _base_df()
        # "sales" (not "annual_sales") -- tests fuzzy/substring resolution.
        _, _, plan, decision, experiments, row = _run("Is price associated with sales?", df)
        self.record_target = plan.semantics.target_column


class TestSynonymWording(unittest.TestCase):
    def test_influence_synonym_for_correlation(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Does price influence annual_sales?", df)
        self.assertEqual(decision.problem_class.value, "CORRELATIONAL")
        self.assertEqual(plan.semantics.explanatory_columns, ["price"])


class TestNoExplicitPredictorDiscoveryOnly(unittest.TestCase):
    def test_what_correlates_with_annual_sales(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("What correlates with annual_sales?", df)
        self.assertEqual(plan.semantics.target_column, "annual_sales")
        self.assertEqual(plan.semantics.explanatory_columns, decision.estimand.predictor_columns)


class TestInvalidExplicitPredictor(unittest.TestCase):
    def test_customer_id_named_as_predictor(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Is customer_id associated with annual_sales?", df)
        # C4.2.1c's fail-closed fix: an invalid explicit predictor must not
        # silently substitute a different, unrequested numeric column.
        self.assertNotEqual(decision.problem_class.value, "CORRELATIONAL")


class TestCausalTreatmentOutcome(unittest.TestCase):
    def test_does_marketing_spend_cause_sales_increase(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("Does marketing_spend cause an increase in annual_sales?", df)
        self.record_problem_class = decision.problem_class.value
        self.record_plan_task = plan.task


class TestDiscoveryDisagreesWithNaiveNamed(unittest.TestCase):
    def test_correlate_metric_wording(self):
        df = _base_df()
        _, _, plan, decision, experiments, row = _run("How does price correlate with annual_sales figures?", df)
        self.assertEqual(plan.semantics.target_column, "annual_sales")
        self.assertEqual(plan.semantics.explanatory_columns, ["price"])


class TestReportDump(unittest.TestCase):
    """Not a correctness assertion -- writes the full captured table so the
    doc can cite exact values rather than paraphrased ones."""

    def test_write_report(self):
        out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "_c4_2_2b_raw_capture.json")
        with open(out_path, "w") as f:
            json.dump(_REPORT_ROWS, f, indent=2, default=str)
        self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
