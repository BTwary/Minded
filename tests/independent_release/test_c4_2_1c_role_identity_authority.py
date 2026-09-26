"""v20-C4.2.1b + C4.2.1c: does a question-explicit NUMERIC predictor remain
authoritative when SemanticEngine's discovered secondary_metric_col points
to a different numeric column -- through method selection, the executed
experiment, AND the hypothesis's own narrative text?

C4.2.1b scope: does the EXECUTED experiment test the requested predictor?
Does NOT add a blocking disagreement guard (detect_canonical_plan_role_
disagreement stays detection-only), does not touch select_for_plan()/
plan.semantics, durable InvestigationContract authority, MethodRegistry
scoring, causal calibration, or DEFECT-005/007.

C4.2.1c scope (added after an external audit of the C4.2.1b delivery found
two real remaining gaps in that pass): (1) an INVALID explicit predictor
must fail closed, not silently substitute a different discovered predictor
and answer a different question; (2) the hypothesis's own persisted claim
text must name the same predictor the executed experiment actually tests.

Background (see docs/AAOS_V20C4_2_1_QUESTION_ROLE_AUTHORITY.md, "Deliberately
NOT done"): C4.2.1 fixed the CATEGORICAL requested-predictor case (a question
naming a categorical variable is rerouted away from Pearson correlation
entirely -- see _bind_requested_predictor). It explicitly left the NUMERIC
case unchanged: _select_correlational bound predictor_columns to whatever
SemanticEngine discovered (secondary_metric_col), never to the question's
own requested_explanatory_columns(). That is the gap C4.2.1b measured and
closed for the valid-predictor case.

Root cause (C4.2.1b), verified directly against the source (not inferred):
in MethodSelectionEngine._select_correlational,

    est.predictor_columns = [secondary] if secondary else []

read only canonical.secondary_metric_column() -- the DISCOVERED variable --
and never consulted canonical.requested_explanatory_columns() at all. A
question naming one numeric predictor explicitly, on a dataset where
SemanticEngine's own secondary-metric heuristic happens to discover a
*different* numeric column, would silently test the discovered column
instead of the one the question asked about.

Fix (C4.2.1b): _select_correlational now checks for a single, type-valid
(numeric, non-categorical, not the target) requested predictor and -- only
then -- prefers it over the discovered secondary metric, recording the
discovered column in estimand.discovered_columns instead. Mirrors the
already-shipped _bind_requested_predictor pattern for the categorical case.

Gap found in C4.2.1b's own delivery, by external audit: an INVALID explicit
predictor (e.g. a non-numeric column named explicitly) was silently
discarded in favor of the DISCOVERED secondary metric -- meaning a question
naming an invalid predictor could still get a CORRELATIONAL verdict testing
some other, unrequested variable. Fixed in C4.2.1c: `decide()`'s CORRELATION
dispatch now fails closed to the diagnostic family (reusing the same reroute
already used for "no resolved secondary metric") whenever there is exactly
one requested predictor and it is not a valid numeric column and not the
resolved semantic dimension -- rather than silently answering a different
question.

Gap found in C4.2.1b's own delivery, self-identified while verifying
assertion 8 (actual SQL/result binding): HypothesisSynthesizer built its own,
role-less CanonicalSemanticResolution, so the persisted hypothesis
`statement` text could name the discovered predictor even when the executed
experiment correctly tested the requested one, for the same investigation.
Fixed in C4.2.1c: the production call site now passes
`decision.estimand.predictor_columns[0]` straight into
`_synthesize_correlation_hypotheses` as an authoritative override.
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
    EstimandSpec, MethodFamily, MethodSelectionEngine, ProblemClass,
)
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.semantic_resolution_builder import build_canonical_semantic_resolution
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal

QUESTION_1 = "Is price associated with annual_sales?"
QUESTION_2 = "Are annual_sales and price associated?"


def _price_sales_tenure_df(n: int = 120, seed: int = 4242) -> pd.DataFrame:
    """price = requested predictor, annual_sales = target, tenure_months =
    a third valid numeric column that plays no role in the question at all.
    Column names are chosen (annual_sales, not sales) so the real compiler's
    own tie-break resolves target=annual_sales / predictor=price for this
    exact phrasing -- verified empirically, not assumed; see
    docs/AAOS_V20C4_2_1B_NUMERIC_REQUESTED_ROLE_AUTHORITY.md for the
    alternative naming ("sales") where the same tie-break instead resolves
    target=price / predictor=sales, an unrelated pre-existing compiler
    quirk this pass does not touch.
    """
    rng = np.random.RandomState(seed)
    price = rng.uniform(5, 50, n)
    annual_sales = 200 - 2 * price + rng.normal(0, 5, n)
    tenure_months = rng.uniform(1, 60, n)
    return pd.DataFrame({"price": price, "annual_sales": annual_sales, "tenure_months": tenure_months})


def _pipeline(question: str, df: pd.DataFrame, *, with_roles: bool = True):
    """Real production path: IntentEngine -> SemanticEngine ->
    UniversalQuestionCompiler -> QuestionRoleProposal -> MethodSelectionEngine.decide()."""
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics) if with_roles else None
    return intent, semantic, plan, roles


def _decide(intent, semantic, question, df, roles):
    return MethodSelectionEngine.decide(
        intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles,
    )


def _arrange_discovered_divergence(semantic, discovered_col: str):
    """Pins the exact disagreement the C4.2.1b brief specifies
    (secondary_metric_col pointing at a column the question never named),
    on top of the real SemanticEngine resolution -- rather than hunting for
    a fragile natural-language phrasing that happens to reproduce it via
    SemanticEngine's own heuristic (which, for THIS phrasing, correctly
    discovers the named column -- see TestFixIsDeliberatelyNarrow in
    test_c4_2_1_question_role_authority.py for that non-divergent case).
    SemanticResolution is a plain (non-frozen) dataclass; this is the same
    "arrange the semantic state" instruction the C4.2.1b brief itself gives
    for the primary regression, not a synthetic decision object -- every
    other step (compiler roles, decide(), estimand, experiment synthesis)
    still runs for real.
    """
    return dataclasses.replace(semantic, secondary_metric_col=discovered_col)


def _hypothesis(target_metric: str, secondary_metric: str) -> PredictiveHypothesis:
    return PredictiveHypothesis(
        id="HYP-01", hypothesis_code="HYP-01", claim="c", mechanism="m",
        predicted_observables_if_true=["a"], predicted_observables_if_false=["b"],
        falsification_criteria="f", required_assumptions=[], prior_probability=0.5,
        posterior_probability=0.5, target_metric=target_metric, secondary_metric=secondary_metric,
        claim_type="ASSOCIATION",
    )


class TestPrimaryRegressionNumericRequestedPredictor(unittest.TestCase):
    """"Is price associated with annual_sales?" with SemanticEngine's own
    secondary_metric_col arranged to point at tenure_months instead --
    the exact scenario the C4.2.1b brief specifies."""

    def setUp(self):
        self.df = _price_sales_tenure_df()
        self.intent, semantic, self.plan, self.roles = _pipeline(QUESTION_1, self.df)
        # Sanity: confirm the compiler names price as the requested
        # explanatory variable BEFORE arranging the divergence, so the
        # divergence below is real, not an artifact of the arrangement.
        self.assertEqual(self.roles.explicit_explanatory_columns(), ["price"])
        self.semantic = _arrange_discovered_divergence(semantic, "tenure_months")
        self.decision = _decide(self.intent, self.semantic, QUESTION_1, self.df, self.roles)

    def test_1_question_role_proposal_identifies_price_as_requested_explanatory(self):
        self.assertEqual(self.roles.explicit_explanatory_columns(), ["price"])

    def test_2_canonical_resolution_preserves_price_as_requested_role(self):
        canon = build_canonical_semantic_resolution(self.semantic, "CORRELATION", question_roles=self.roles)
        self.assertEqual(canon.requested_explanatory_columns(), ["price"])
        # The discovered secondary metric is still visible, but distinct.
        self.assertEqual(canon.secondary_metric_column(), "tenure_months")
        self.assertNotIn("tenure_months", canon.requested_explanatory_columns())

    def test_3_estimand_predictor_columns_is_requested_price(self):
        self.assertEqual(self.decision.estimand.predictor_columns, ["price"])

    def test_4_selected_method_is_correlation(self):
        self.assertEqual(self.decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertEqual(self.decision.method_family, MethodFamily.CORRELATION)

    def test_5_generated_sql_references_price(self):
        hyp = _hypothesis(target_metric=self.decision.estimand.target_column, secondary_metric="price")
        fake_decision = type("D", (), {"estimand": self.decision.estimand})()
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], self.semantic, decision=fake_decision,
        )
        self.assertTrue(candidates)
        self.assertIn("price", candidates[0].query_sql)

    def test_6_generated_sql_does_not_silently_substitute_tenure_months(self):
        hyp = _hypothesis(target_metric=self.decision.estimand.target_column, secondary_metric="price")
        fake_decision = type("D", (), {"estimand": self.decision.estimand})()
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], self.semantic, decision=fake_decision,
        )
        self.assertNotIn("tenure_months", candidates[0].query_sql)

    def test_7_tenure_months_remains_available_as_discovered_not_as_the_answer(self):
        self.assertEqual(self.decision.estimand.discovered_columns, ["tenure_months"])
        self.assertNotIn("tenure_months", self.decision.estimand.predictor_columns)

    def test_8_verified_against_actual_sql_result_binding_not_only_metadata(self):
        """Assertion 8: for a correlation experiment, verify the actual
        SQL/result binding, not only estimand metadata -- executed here
        against the real dataframe via the real query builder + scipy
        correlation the experiment declares (tool_name), independent of
        whatever estimand.predictor_columns merely claims."""
        hyp = _hypothesis(target_metric=self.decision.estimand.target_column, secondary_metric="price")
        fake_decision = type("D", (), {"estimand": self.decision.estimand})()
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], self.semantic, decision=fake_decision,
        )
        cand = candidates[0]
        self.assertEqual(cand.tool_name, "scipy_correlation")
        self.assertEqual(set(cand.metrics), {self.decision.estimand.target_column, "price"})
        # Execute the declared SQL for real against an in-memory duckdb
        # table, exactly the shape the runtime executor would run, to
        # confirm the paired observations really are (target, price) and
        # not (target, tenure_months).
        import duckdb
        con = duckdb.connect()
        con.register("data_table", self.df)
        result = con.execute(cand.query_sql).fetchdf()
        con.close()
        self.assertEqual(set(result.columns), {self.decision.estimand.target_column, "price"})
        self.assertEqual(len(result), len(self.df))

    def test_9_fails_against_pre_c4_2_1b_behavior(self):
        """Pre-fix behavior is exactly the legacy no-question-roles path:
        predictor_columns is bound to the discovered secondary metric.
        Confirms this test suite actually discriminates the fix -- if roles
        are withheld (the pre-C4.2.1b shape at this call site, since the
        override added in this pass only fires when a valid requested
        predictor is present), the old, broken predictor comes back."""
        legacy_decision = _decide(self.intent, self.semantic, QUESTION_1, self.df, None)
        self.assertEqual(legacy_decision.estimand.predictor_columns, ["tenure_months"])
        self.assertNotEqual(legacy_decision.estimand.predictor_columns, self.decision.estimand.predictor_columns)


class TestSecondReversedWordOrder(unittest.TestCase):
    """"Are annual_sales and price associated?" -- same canonical analytical
    identity as QUESTION_1 should result."""

    def setUp(self):
        self.df = _price_sales_tenure_df()
        self.intent, semantic, self.plan, self.roles = _pipeline(QUESTION_2, self.df)
        self.assertEqual(self.roles.explicit_explanatory_columns(), ["price"])
        self.semantic = _arrange_discovered_divergence(semantic, "tenure_months")
        self.decision = _decide(self.intent, self.semantic, QUESTION_2, self.df, self.roles)

    def test_predictor_role_matches_first_phrasing(self):
        self.assertEqual(self.decision.estimand.predictor_columns, ["price"])

    def test_target_role_matches_first_phrasing(self):
        self.assertEqual(self.decision.estimand.target_column, "annual_sales")

    def test_estimand_identity_matches_first_phrasing(self):
        first_intent, first_semantic, first_plan, first_roles = _pipeline(QUESTION_1, self.df)
        first_semantic = _arrange_discovered_divergence(first_semantic, "tenure_months")
        first_decision = _decide(first_intent, first_semantic, QUESTION_1, self.df, first_roles)
        self.assertEqual(first_decision.estimand.target_column, self.decision.estimand.target_column)
        self.assertEqual(first_decision.estimand.predictor_columns, self.decision.estimand.predictor_columns)
        self.assertEqual(first_decision.estimand.discovered_columns, self.decision.estimand.discovered_columns)

    def test_selected_method_matches_first_phrasing(self):
        self.assertEqual(self.decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertEqual(self.decision.method_family, MethodFamily.CORRELATION)

    def test_experiment_bindings_match_first_phrasing(self):
        hyp = _hypothesis(target_metric=self.decision.estimand.target_column, secondary_metric="price")
        fake_decision = type("D", (), {"estimand": self.decision.estimand})()
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], self.semantic, decision=fake_decision,
        )
        self.assertIn("price", candidates[0].query_sql)
        self.assertNotIn("tenure_months", candidates[0].query_sql)


class TestThirdDeliberateDisagreement(unittest.TestCase):
    """Direct construction (not derived from a phrasing that happens to
    agree): SemanticResolution.target_metric_col='sales',
    secondary_metric_col='tenure_months' disagree with a QuestionRoleProposal
    naming target_column='sales', explanatory_columns=['price']. The
    canonical result MUST prefer the explicit requested role. Deliberately
    NOT "fixed" by making SemanticResolution agree -- the disagreement is
    the point."""

    def setUp(self):
        self.df = pd.DataFrame({
            "price": [10.0, 12.0, 9.0, 15.0, 11.0, 13.0, 14.0, 10.5, 9.5, 12.5] * 4,
            "sales": [100, 120, 90, 150, 110, 130, 140, 105, 95, 125] * 4,
            "tenure_months": [6, 8, 10, 12, 14, 16, 18, 20, 22, 24] * 4,
        })
        self.intent = IntentEngine.parse_intent("Is price associated with sales?", available_columns=list(self.df.columns))
        raw_semantic = SemanticEngine.resolve_schema_static(self.intent, {"data_table": self.df})
        # Arrange the exact disagreement the brief specifies, independent of
        # whatever the real compiler happened to resolve for this phrasing.
        self.semantic = dataclasses.replace(
            raw_semantic, target_metric_col="sales", secondary_metric_col="tenure_months",
        )
        self.roles = QuestionRoleProposal(
            explanatory_columns=["price"], referenced_columns=["sales", "price"], target_column="sales",
        )

    def test_target_is_sales(self):
        canon = build_canonical_semantic_resolution(self.semantic, "CORRELATION", question_roles=self.roles)
        self.assertEqual(canon.outcome_column(), "sales")

    def test_predictor_is_price_not_tenure_months(self):
        decision = _decide(self.intent, self.semantic, "Is price associated with sales?", self.df, self.roles)
        self.assertEqual(decision.estimand.target_column, "sales")
        self.assertEqual(decision.estimand.predictor_columns, ["price"])
        self.assertNotIn("tenure_months", decision.estimand.predictor_columns)
        self.assertEqual(decision.estimand.discovered_columns, ["tenure_months"])

    def test_semantic_resolution_itself_was_not_mutated_to_agree(self):
        # The disagreement is still directly observable on the raw semantic
        # object -- proving the fix did not "cheat" by silently overwriting
        # SemanticResolution.secondary_metric_col to make the two agree.
        self.assertEqual(self.semantic.secondary_metric_col, "tenure_months")
        self.assertNotEqual(self.semantic.secondary_metric_col, "price")


class TestFourthInvalidExplicitNumericRole(unittest.TestCase):
    """A question-role proposal naming a non-numeric/incompatible predictor
    for a numeric correlation method must NOT be blindly trusted.

    v20-C4.2.1c: the original C4.2.1b version of this test accepted a
    silent fallback to the discovered secondary metric here. An external
    audit of that pass correctly flagged this as analytically dangerous:
    the user explicitly asked about `customer_name`; silently answering a
    correlation between `annual_sales` and `price` instead (a variable the
    user never mentioned) answers a *different question* without saying so.
    Expected behavior is now genuinely fail-closed: the explicit role is
    proposed, semantic/type validation rejects it, and routing reroutes to
    the diagnostic family (the same reroute this function already uses two
    lines below for "correlation intent lacks a resolved secondary metric"
    -- no new policy invented) rather than silently substituting a
    different predictor and returning a CORRELATIONAL verdict for it.
    """

    def setUp(self):
        rng = np.random.RandomState(99)
        n = 60
        price = rng.uniform(5, 50, n)
        annual_sales = 200 - 2 * price + rng.normal(0, 5, n)
        tenure_months = rng.uniform(1, 60, n)
        # A second, legitimate categorical column so the primary dimension
        # does not trivially resolve to the sole categorical column present
        # (an unrelated, pre-existing dimension-resolution quirk on
        # single-categorical-column datasets that is out of scope here --
        # see the "not modified" list in the module docstring).
        region = rng.choice(["East", "West"], n)
        customer_name = [f"cust-{i}" for i in range(n)]
        self.df = pd.DataFrame({
            "price": price, "annual_sales": annual_sales, "tenure_months": tenure_months,
            "region": region, "customer_name": customer_name,
        })
        self.intent = IntentEngine.parse_intent(QUESTION_1, available_columns=list(self.df.columns))
        self.semantic = SemanticEngine.resolve_schema_static(self.intent, {"data_table": self.df})
        # Confirm the discovered secondary metric is the legitimate "price"
        # here (unarranged) -- this test is about an INVALID explicit role,
        # not about the requested-vs-discovered divergence covered above.
        self.assertEqual(self.semantic.secondary_metric_col, "price")
        self.bad_roles = QuestionRoleProposal(
            explanatory_columns=["customer_name"],
            referenced_columns=["annual_sales", "customer_name"],
            target_column="annual_sales",
        )

    def test_invalid_explicit_role_is_not_promoted_to_predictor(self):
        decision = _decide(self.intent, self.semantic, QUESTION_1, self.df, self.bad_roles)
        self.assertNotIn("customer_name", decision.estimand.predictor_columns)

    def test_invalid_explicit_role_is_not_silently_answered_with_a_different_predictor(self):
        """The heart of the audit finding: `price` must NOT quietly become
        the answer to a question that named `customer_name`. Falling back
        to ANY other predictor the user did not ask about is itself the
        defect, independent of which discovered column it happens to be."""
        decision = _decide(self.intent, self.semantic, QUESTION_1, self.df, self.bad_roles)
        self.assertEqual(decision.estimand.predictor_columns, [])
        self.assertNotEqual(decision.problem_class, ProblemClass.CORRELATIONAL)

    def test_fails_closed_to_diagnostic_with_an_explanatory_rationale(self):
        decision = _decide(self.intent, self.semantic, QUESTION_1, self.df, self.bad_roles)
        self.assertEqual(decision.problem_class, ProblemClass.DIAGNOSTIC)
        self.assertIn("customer_name", decision.rationale)
        self.assertIn("not a valid predictor", decision.rationale)

    def test_no_correlation_experiment_would_be_synthesized_over_a_string_column(self):
        est = EstimandSpec(target_column="annual_sales", predictor_columns=["customer_name"])
        fake_decision = type("D", (), {"estimand": est})()
        hyp = _hypothesis(target_metric="annual_sales", secondary_metric="customer_name")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], self.semantic, decision=fake_decision,
        )
        self.assertEqual(candidates, [])


class TestFifthDiscoveryOnlyNoExplicitPredictor(unittest.TestCase):
    """When the question does NOT explicitly name a predictor, existing
    discovery behavior (SemanticEngine's secondary_metric_col) may still be
    used. Proves the new authority rule does not destroy exploratory
    discovery."""

    def setUp(self):
        self.df = _price_sales_tenure_df()
        self.intent = IntentEngine.parse_intent(
            "What correlates with annual_sales?", available_columns=list(self.df.columns),
        )
        self.semantic = SemanticEngine.resolve_schema_static(self.intent, {"data_table": self.df})
        self.no_predictor_roles = QuestionRoleProposal(
            explanatory_columns=[], referenced_columns=["annual_sales"], target_column="annual_sales",
        )

    def test_no_requested_predictor_is_named(self):
        self.assertEqual(self.no_predictor_roles.explicit_explanatory_columns(), [])

    def test_discovery_still_supplies_a_predictor(self):
        decision = _decide(self.intent, self.semantic, "What correlates with annual_sales?", self.df, self.no_predictor_roles)
        self.assertEqual(decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertTrue(decision.estimand.predictor_columns)
        self.assertEqual(decision.estimand.predictor_columns, [self.semantic.secondary_metric_col])
        # Nothing was "discovered but suppressed" here -- there was no
        # requested role to disagree with the discovered one.
        self.assertEqual(decision.estimand.discovered_columns, [])


class TestNoConceptCollapse(unittest.TestCase):
    """requested predictor / discovered candidate / confounder must never be
    silently collapsed into each other. Reuses the C4.2.1 categorical
    scenario (plan_tier requested, tenure_months discovered, vintage_year
    confounder) to confirm the numeric-branch fix does not disturb that
    already-shipped distinction."""

    def _confounded_df(self) -> pd.DataFrame:
        rng = np.random.RandomState(7777)
        rows, cid = [], 1
        for plan, vintage, n, hazard in [
            ("Starter", "Vintage-2023", 100, 0.30), ("Starter", "Vintage-2025", 30, 0.05),
            ("Growth", "Vintage-2023", 30, 0.30), ("Growth", "Vintage-2025", 100, 0.05),
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

    def test_confounder_is_never_promoted_to_predictor_or_discovered(self):
        df = self._confounded_df()
        question = "Does plan tier affect cancellation rate?"
        intent, semantic, plan, roles = _pipeline(question, df)
        decision = _decide(intent, semantic, question, df, roles)
        self.assertEqual(decision.estimand.predictor_columns, ["plan_tier"])
        self.assertEqual(decision.estimand.discovered_columns, ["tenure_months"])
        self.assertIn("vintage_year", list(semantic.churn_confounder_cols or []))
        # vintage_year is neither the requested predictor nor recorded as a
        # "discovered" variable -- it is tracked separately as a confounder.
        self.assertNotIn("vintage_year", decision.estimand.predictor_columns)
        self.assertNotIn("vintage_year", decision.estimand.discovered_columns)


class TestStaticAuthorityAudit(unittest.TestCase):
    """After implementation: grep for requested_explanatory / target_column
    / secondary_metric_col / predictor_columns and verify no production path
    silently converts a requested explicit predictor into the discovered
    secondary metric without an explicit semantic-validity check."""

    def test_select_correlational_gates_the_override_on_validity(self):
        import inspect
        from packages.analytics_core.src.engines import method_selection
        src = inspect.getsource(method_selection.MethodSelectionEngine._select_correlational)
        self.assertIn("requested_explanatory_columns", src)
        self.assertIn("is_valid_numeric", src)
        self.assertIn("available_numeric_cols", src)
        # The discovered-secondary assignment line is still present (the
        # pre-existing fallback for the no-valid-requested-predictor case),
        # but it must precede a validity-gated override, not stand alone.
        self.assertIn("est.predictor_columns = [secondary] if secondary else []", src)
        self.assertIn("if is_valid_numeric and candidate != secondary:", src)

    def test_bind_requested_predictor_categorical_path_untouched(self):
        import inspect
        from packages.analytics_core.src.engines import method_selection
        src = inspect.getsource(method_selection.MethodSelectionEngine._bind_requested_predictor)
        self.assertIn("requested_explanatory_columns", src)


class TestDetectorMeasurementNumericScenarios(unittest.TestCase):
    """The C4.2 detector (detect_canonical_plan_role_disagreement) stays
    detection-only in this pass. Measured directly on the new numeric
    scenarios: with the fix applied, plan.semantics.explanatory_columns and
    decision.estimand.predictor_columns agree again (both name the question's
    requested predictor), same as the C4.2.1 categorical fix already
    achieved for its scenario."""

    def test_no_disagreement_for_the_primary_regression_scenario(self):
        df = _price_sales_tenure_df()
        intent, semantic, plan, roles = _pipeline(QUESTION_1, df)
        semantic = _arrange_discovered_divergence(semantic, "tenure_months")
        decision = _decide(intent, semantic, QUESTION_1, df, roles)
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertEqual(conflicts, [])

    def test_disagreement_still_detectable_without_question_roles(self):
        """Legacy call sites (no question_roles threaded) still exhibit the
        detectable disagreement -- confirms the detector itself was not
        weakened by this pass, only the authoritative path was fixed."""
        df = _price_sales_tenure_df()
        intent, semantic, plan, roles = _pipeline(QUESTION_1, df)
        semantic = _arrange_discovered_divergence(semantic, "tenure_months")
        decision = _decide(intent, semantic, QUESTION_1, df, None)
        conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(decision, plan)
        self.assertTrue(conflicts)


class TestFifthCHypothesisTextRoleIdentityMatchesExperiment(unittest.TestCase):
    """v20-C4.2.1c: closes the hypothesis-text gap the C4.2.1b audit found
    (and flagged in this file's original TestKnownRemainingGapHypothesisText
    IsNotRoleAware, which pinned it as known-but-not-yet-fixed).

    HypothesisSynthesizer.synthesize_competing_hypotheses previously built
    its own CanonicalSemanticResolution (via _build_canonical_semantics),
    which was never given question_roles -- unlike
    MethodSelectionEngine.decide(), which controller.py DOES thread
    QuestionRoleProposal into. So the EXECUTED experiment SQL correctly
    tested the requested predictor while the HYPOTHESIS's own persisted
    `statement` text could still name the discovered variable instead, for
    the same investigation.

    Fix: the production call site now passes
    `decision.estimand.predictor_columns[0]` -- the single authoritative
    predictor MethodSelectionEngine.decide() already resolved -- straight
    into `_synthesize_correlation_hypotheses` as `predictor_column`,
    overriding `canonical.secondary_metric_column()` for hypothesis-text
    purposes exactly as method_selection.py already does for the estimand.
    No new canonical/role-threading machinery added; reuses the existing
    authoritative value.
    """

    def test_hypothesis_statement_and_experiment_sql_now_name_the_same_predictor(self):
        import dataclasses as _dc
        from apps.api.src.core.database import SessionLocal
        from apps.api.src.models.entities import Experiment, Hypothesis
        from tests.independent_release.test_defect_015_forensic_completion import _run_forensic_investigation
        import packages.analytics_core.src.engines.semantic as semmod

        df = _price_sales_tenure_df()
        orig = semmod.SemanticEngine.resolve_schema

        def _patched(self, intent, datasets_map, **kw):
            res = orig(self, intent, datasets_map, **kw)
            return _dc.replace(res, secondary_metric_col="tenure_months")

        semmod.SemanticEngine.resolve_schema = _patched
        try:
            inv = _run_forensic_investigation(df, QUESTION_1)
        finally:
            semmod.SemanticEngine.resolve_schema = orig

        db = SessionLocal()
        try:
            exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
            sqls = [str((e.arguments_json or {}).get("sql", "")) for e in exps]
            self.assertTrue(any("price" in q and "tenure_months" not in q for q in sqls), sqls)

            hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
            statements = [h.statement or "" for h in hyps]
            self.assertTrue(statements)
            # Fixed: every hypothesis statement for this investigation now
            # names the requested predictor (price), never the discovered,
            # question-unmentioned one (tenure_months).
            for s in statements:
                self.assertIn("price", s)
                self.assertNotIn("tenure_months", s)
        finally:
            db.close()

    def test_legacy_no_decision_call_sites_are_unaffected(self):
        """Callers that never supply a MethodSelectionDecision at all (the
        `else` branch in synthesize_competing_hypotheses) keep their prior
        behavior exactly -- predictor_column stays unset, falling back to
        canonical.secondary_metric_column(), unchanged by this pass."""
        from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer

        df = _price_sales_tenure_df()
        intent, semantic, plan, roles = _pipeline(QUESTION_1, df)
        semantic = _arrange_discovered_divergence(semantic, "tenure_months")
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, QUESTION_1, intent=intent, decision=None,
        )
        self.assertTrue(hyps)
        self.assertTrue(any("tenure_months" in h.claim for h in hyps))


if __name__ == "__main__":
    unittest.main()
