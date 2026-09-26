"""v20-C4 Phase 4 regression suite: CanonicalSemanticResolution wired as the
authority upstream of MethodSelectionEngine.

Scope (deliberately narrow -- see docs/AAOS_V20C4_CANONICAL_SEMANTIC_AUTHORITY.md
Phase 4 section): this only proves the authority-transfer chain

    CanonicalSemanticResolution -> MethodSelectionEngine.decide() -> EstimandSpec

is real, not just a parameter rename. It does NOT touch ExperimentSynthesizer,
UniversalQuestionCompiler, or the controller -- those are Phase 5/6/7.

Before this change, MethodSelectionEngine.decide() never constructed a
CanonicalSemanticResolution at all; EstimandSpec.target_column /
comparison_dimension / time_column / predictor_columns / churn_bindings were
read directly off the raw SemanticResolution inside MethodSelectionEngine._estimand
and friends. This suite pins the new behavior: decide() now builds canonical
once, before any routing decision, and every _select_* branch derives those
EstimandSpec fields from canonical accessors instead.

Run with: python -m unittest tests/independent_release/test_c4_phase4_method_selection_canonical_authority.py
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines import method_selection as method_selection_module
from packages.analytics_core.src.engines.method_selection import (
    MethodSelectionEngine, ProblemClass, MethodFamily,
)
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType
from packages.schemas.src.semantic_binding import SemanticBindingSet
from packages.schemas.src.semantic_resolution_contract import (
    CanonicalSemanticResolution, ResolvedSemanticField, SemanticCandidateSet,
)
from packages.schemas.src.semantic_role import ResolutionStatus, SemanticRole


def _semantic(
    metric="revenue",
    group=None,
    time_col=None,
    secondary=None,
    churn_event=None,
    available_categorical=None,
    churn_confounders=None,
    churn_exposure=None,
    churn_censored=None,
    churn_outcome_available=True,
):
    return SemanticResolution(
        primary_dataset_name="orders",
        target_metric_col=metric,
        group_dimension_col=group,
        time_col=time_col,
        table_grain="record_level",
        available_numeric_cols=[metric],
        available_categorical_cols=list(available_categorical or ([group] if group else [])),
        world_model=None,
        metric_definition=MetricDefinition(
            name=metric, table_name="orders", source_columns=[metric], semantic_type="sum_measure",
            aggregation_type=AggregationType.SUM, is_additive=True, valid_aggregations=[AggregationType.SUM],
            semantic_resolution_status="RESOLVED",
        ),
        direction_hint="unspecified",
        secondary_metric_col=secondary,
        churn_event_col=churn_event,
        churn_exposure_col=churn_exposure,
        churn_censored_col=churn_censored,
        churn_confounder_cols=list(churn_confounders or []),
        churn_outcome_available=churn_outcome_available,
    )


class TestCanonicalIsActuallyConsulted(unittest.TestCase):
    """Proves decide() reads through canonical, not just that the two happen
    to agree (which parity alone could never distinguish from the old
    behavior)."""

    def test_deliberate_disagreement_canonical_wins(self):
        # churn_event_col makes the canonical outcome field churn-relevant
        # (see semantic_resolution_builder._resolve_outcome_field), so it
        # resolves to "price" -- deliberately different from
        # semantic.target_metric_col ("sales") -- even though this question
        # routes to CORRELATIONAL, not CHURN. Before C4 this scenario could
        # not even be constructed: target_column was read straight off
        # semantic.target_metric_col with no canonical object anywhere in
        # the call chain.
        semantic = _semantic(metric="sales", secondary="discount", churn_event="price")
        intent = IntentEngine.parse_intent("Is there a correlation between sales and discount?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertEqual(decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertFalse(decision.fallback_used)
        # The old resolver's answer would have been "sales". Canonical's
        # answer is "price". Asserting "price" is the only way to prove
        # canonical -- not semantic -- is now the source of truth.
        self.assertEqual(decision.estimand.target_column, "price")
        self.assertNotEqual(decision.estimand.target_column, semantic.target_metric_col)

    def test_ordinary_case_still_agrees_with_semantic(self):
        # No churn relevance here, so canonical.outcome_column() and
        # semantic.target_metric_col are guaranteed identical by
        # construction (_resolve_outcome_field's non-churn branch passes
        # target_metric_col straight through). This is the common-case
        # parity guarantee the disagreement test above depends on for its
        # contrast to mean anything.
        semantic = _semantic(metric="revenue", secondary="cost")
        intent = IntentEngine.parse_intent("Is there a correlation between revenue and cost?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertEqual(decision.estimand.target_column, "revenue")
        self.assertEqual(decision.estimand.predictor_columns, ["cost"])


class TestAmbiguityStaysAmbiguous(unittest.TestCase):
    """Canonical must never silently pick a dimension decide() itself never
    asked for; a resolver that guesses defeats the point of this phase."""

    def test_ambiguous_dimension_candidates_do_not_get_guessed(self):
        semantic = _semantic(
            metric="revenue", secondary="cost",
            group=None, available_categorical=["region", "channel"],
        )
        intent = IntentEngine.parse_intent("Is there a correlation between revenue and cost?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertIsNone(decision.estimand.comparison_dimension)

    def test_single_candidate_dimension_resolves_through_canonical(self):
        # This is a genuine, intended behavior improvement from wiring
        # canonical in: previously comparison_dimension was a bare read of
        # semantic.group_dimension_col, so a single unambiguous categorical
        # candidate that was never explicitly bound went unused. Canonical
        # already applies the same 0/1/>1 fail-closed rule
        # resolve_group_dimension() does (see semantic_resolution_builder),
        # so decide() now benefits from it too instead of only
        # HypothesisSynthesizer.
        semantic = _semantic(
            metric="revenue", secondary="cost",
            group=None, available_categorical=["region"],
        )
        intent = IntentEngine.parse_intent("Is there a correlation between revenue and cost?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertEqual(decision.estimand.comparison_dimension, "region")


class TestChurnBindingsSourcedFromCanonical(unittest.TestCase):
    def test_churn_bindings_match_canonical_exposure_censored_confounders(self):
        semantic = _semantic(
            metric="revenue",
            churn_event="churned",
            churn_exposure="tenure_months",
            churn_censored="is_censored",
            churn_confounders=["plan_type", "region"],
        )
        intent = IntentEngine.parse_intent("Why are customers churning?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertEqual(decision.problem_class, ProblemClass.SURVIVAL_CHURN)
        self.assertEqual(decision.estimand.churn_bindings["event_col"], "churned")
        self.assertEqual(decision.estimand.churn_bindings["exposure_col"], "tenure_months")
        self.assertEqual(decision.estimand.churn_bindings["censored_col"], "is_censored")
        self.assertEqual(
            sorted(decision.estimand.churn_bindings["confounder_cols"]),
            ["plan_type", "region"],
        )


class TestForecastTimeColumnSourcedFromCanonical(unittest.TestCase):
    def test_time_column_matches_canonical_time_variable(self):
        semantic = _semantic(metric="revenue", time_col="order_date")
        intent = IntentEngine.parse_intent("Forecast next month's revenue")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertEqual(decision.problem_class, ProblemClass.FORECASTING)
        self.assertEqual(decision.estimand.time_column, "order_date")


class TestCanonicalConstructionFailsClosed(unittest.TestCase):
    """If canonical cannot be built, decide() must not silently fall back to
    reading semantic.* directly and continuing as if nothing happened --
    that would just reintroduce the second, ungoverned resolution path this
    phase removes. It must fail closed via the module's existing
    fallback() path instead."""

    def test_canonical_builder_exception_fails_closed_without_reading_semantic(self):
        # A perfectly ordinary, well-formed semantic object -- the point is
        # to isolate the canonical-construction step specifically (not any
        # other attribute access further down decide()), by making
        # build_canonical_semantic_resolution itself raise.
        semantic = _semantic(metric="revenue", secondary="cost")
        intent = IntentEngine.parse_intent("Is there a correlation between revenue and cost?")

        with mock.patch.object(
            method_selection_module,
            "build_canonical_semantic_resolution",
            side_effect=RuntimeError("boom"),
        ):
            decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertTrue(decision.fallback_used)
        self.assertIn("canonical_construction_failed", decision.rationale)
        self.assertIn("RuntimeError", decision.rationale)
        # Fail-closed means no routing decision was made and no EstimandSpec
        # field was populated from semantic.* as a substitute -- it's the
        # plain fallback() estimand, not a hand-rolled one.
        self.assertIsNone(decision.estimand.target_column)
        self.assertEqual(decision.problem_class, ProblemClass.FALLBACK)


class TestEstimandDirectCallBackwardCompatibleWithoutCanonical(unittest.TestCase):
    """_estimand() is called directly (with no canonical) by several
    pre-existing unit test suites that build a decision by hand to test
    other EstimandSpec fields in isolation. Confirms that path is
    unaffected -- canonical is additive, not a required argument."""

    def test_estimand_without_canonical_still_reads_semantic_directly(self):
        semantic = _semantic(metric="revenue", group="region", time_col="order_date")
        est = MethodSelectionEngine._estimand(semantic, None)
        self.assertEqual(est.target_column, "revenue")
        self.assertEqual(est.comparison_dimension, "region")
        self.assertEqual(est.time_column, "order_date")


def _resolved(role, value, source):
    return ResolvedSemanticField(
        role=role, value=value, status=ResolutionStatus.RESOLVED,
        candidates=[value], provenance={"source": source},
    )


def _unresolved(role, source):
    return ResolvedSemanticField(
        role=role, value=None, status=ResolutionStatus.UNRESOLVED,
        candidates=[], provenance={"source": source},
    )


def _canonical(
    outcome=None, dimension=None, time=None, secondary_metric=None,
    exposure=None, censored=None, confounders=None,
):
    """Hand-build a CanonicalSemanticResolution directly, bypassing
    semantic_resolution_builder.build_canonical_semantic_resolution
    entirely. This is the only way to construct a canonical resolution that
    deliberately disagrees with (or resolves something absent from) the raw
    SemanticResolution passed alongside it in these tests -- the real
    builder derives every field from that same SemanticResolution, so it
    can never produce a disagreement by itself (see the "known limitation"
    in docs/AAOS_V20C3_CANONICAL_EXPERIMENT_CONTRACT.md: there is currently
    no independent second resolver upstream of it). These tests stand in
    for that eventual second resolver to prove decide()'s routing and
    EstimandSpec population genuinely read canonical rather than merely
    happening to agree with semantic by construction.
    """
    return CanonicalSemanticResolution(
        bindings=SemanticBindingSet(bindings=[]),
        outcome=outcome or _unresolved(SemanticRole.OUTCOME, "semantic.target_metric_col"),
        dimension=dimension or _unresolved(SemanticRole.GROUPING_DIMENSION, "semantic.group_dimension_col"),
        time=time or _unresolved(SemanticRole.TIME_VARIABLE, "semantic.time_col"),
        secondary_metric=secondary_metric or _unresolved(SemanticRole.EXPLANATORY_VARIABLE, "semantic.secondary_metric_col"),
        exposure=exposure or _unresolved(SemanticRole.EXPOSURE, "semantic.churn_exposure_col"),
        censored=censored or _unresolved(SemanticRole.CENSORED_TIME, "semantic.churn_censored_col"),
        confounders=confounders or SemanticCandidateSet(
            role=SemanticRole.EXPLANATORY_VARIABLE, purpose="churn_confounders",
            columns=[], provenance={"source": "semantic.churn_confounder_cols"},
        ),
    )


class TestC41RoutingCannotBeSuppressedByStaleSemanticFields(unittest.TestCase):
    """STEP 3.A/B/C + STEP 4's required adversarial tests: dispatch/estimand
    population must follow canonical even when the raw SemanticResolution
    disagrees with or lacks the corresponding field entirely."""

    def test_A_correlation_routes_from_canonical_even_when_semantic_secondary_is_none(self):
        # semantic.secondary_metric_col = None; canonical secondary/predictor
        # binding = valid ("price"). The canonical CORRELATION path must
        # still be selected.
        semantic = _semantic(metric="sales", secondary=None)
        intent = IntentEngine.parse_intent("Is there a correlation between sales and price?")
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "sales", "semantic.target_metric_col"),
            secondary_metric=_resolved(SemanticRole.EXPLANATORY_VARIABLE, "price", "semantic.secondary_metric_col"),
        )
        with mock.patch.object(
            method_selection_module, "build_canonical_semantic_resolution", return_value=canonical,
        ):
            decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertFalse(decision.fallback_used)
        self.assertEqual(decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertEqual(decision.method_family, MethodFamily.CORRELATION)
        self.assertEqual(decision.estimand.predictor_columns, ["price"])

    def test_B_forecast_routes_from_canonical_even_when_semantic_time_col_is_none(self):
        # semantic.time_col = None; canonical time binding = valid
        # ("order_date"). The canonical FORECAST path must still be
        # selected.
        semantic = _semantic(metric="revenue", time_col=None)
        intent = IntentEngine.parse_intent("Forecast next month's revenue")
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "revenue", "semantic.target_metric_col"),
            time=_resolved(SemanticRole.TIME_VARIABLE, "order_date", "semantic.time_col"),
        )
        with mock.patch.object(
            method_selection_module, "build_canonical_semantic_resolution", return_value=canonical,
        ):
            decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertFalse(decision.fallback_used)
        self.assertEqual(decision.problem_class, ProblemClass.FORECASTING)
        self.assertEqual(decision.method_family, MethodFamily.FORECAST)
        self.assertEqual(decision.estimand.time_column, "order_date")

    def test_C_churn_routing_uses_canonical_event_even_when_semantic_event_col_is_stale(self):
        # semantic.churn_event_col = stale/None; canonical churn event =
        # "actual_event". The canonical churn path must use the canonical
        # event, both for routing (intent_type == "CHURN" always reaches
        # _select_churn regardless) and for the resolved event_col itself.
        semantic = _semantic(metric="revenue", churn_event=None)
        intent = IntentEngine.parse_intent("Why are customers churning?")
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "actual_event", "semantic.churn_event_col"),
        )
        with mock.patch.object(
            method_selection_module, "build_canonical_semantic_resolution", return_value=canonical,
        ):
            decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertFalse(decision.fallback_used)
        self.assertEqual(decision.problem_class, ProblemClass.SURVIVAL_CHURN)
        self.assertEqual(decision.estimand.churn_bindings["event_col"], "actual_event")
        self.assertNotEqual(decision.estimand.churn_bindings["event_col"], semantic.churn_event_col)

    def test_root_cause_gate_also_uses_canonical_not_stale_semantic_churn_field(self):
        # Same disagreement as test_C, but exercised through the
        # ROOT_CAUSE -> churn gate in decide() (the other of the two
        # `getattr(semantic, "churn_event_col", ...)` reads this phase
        # removed), not the direct intent_type == "CHURN" path.
        semantic = _semantic(metric="revenue", churn_event=None)
        intent = IntentEngine.parse_intent("Why are customers churning?")
        # Force ROOT_CAUSE classification regardless of IntentEngine's own
        # churn-keyword handling, to isolate the gate under test.
        intent.intent_type = "ROOT_CAUSE"
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "actual_event", "semantic.churn_event_col"),
        )
        with mock.patch.object(
            method_selection_module, "build_canonical_semantic_resolution", return_value=canonical,
        ):
            decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertFalse(decision.fallback_used)
        self.assertEqual(decision.problem_class, ProblemClass.SURVIVAL_CHURN)
        self.assertEqual(decision.estimand.churn_bindings["event_col"], "actual_event")


class TestC41MetricIdentityConsistency(unittest.TestCase):
    """STEP 3.D: a deliberate disagreement between semantic.target_metric_col
    and the canonical outcome must be reflected consistently across every
    estimand identity field that derives from it -- not just target_column."""

    def test_target_column_metric_ref_and_unit_all_reference_canonical_value(self):
        semantic = _semantic(metric="sales", secondary="discount")
        intent = IntentEngine.parse_intent("Is there a correlation between sales and discount?")
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "price", "semantic.churn_event_col"),
            secondary_metric=_resolved(SemanticRole.EXPLANATORY_VARIABLE, "discount", "semantic.secondary_metric_col"),
        )
        with mock.patch.object(
            method_selection_module, "build_canonical_semantic_resolution", return_value=canonical,
        ):
            decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)

        self.assertFalse(decision.fallback_used)
        est = decision.estimand
        # None of these should still say "sales" -- every one of them must
        # consistently reference canonical's "price".
        self.assertEqual(est.target_column, "price")
        self.assertEqual(est.metric_ref.column, "price")
        self.assertEqual(est.metric_ref.name, "price")
        self.assertIn("price", est.unit_of_analysis.keys)
        self.assertNotIn("sales", est.unit_of_analysis.keys)


class TestC41ChurnRelevanceDistinguishesOrdinaryFromChurnOutcomes(unittest.TestCase):
    """STEP 3.E/F: is_churn_relevant() must tell a churn-flavored outcome
    apart from an ordinary resolved target -- including when the churn
    outcome is itself still unresolved."""

    def test_E_ordinary_resolved_target_is_not_churn_relevant(self):
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "revenue", "semantic.target_metric_col"),
        )
        self.assertFalse(canonical.is_churn_relevant())

    def test_E_genuine_churn_relevant_resolved_outcome_is_churn_relevant(self):
        canonical = _canonical(
            outcome=_resolved(SemanticRole.OUTCOME, "churned", "semantic.churn_event_col"),
        )
        self.assertTrue(canonical.is_churn_relevant())

    def test_F_unresolved_churn_outcome_is_still_churn_relevant_not_ordinary(self):
        # A churn-flavored question whose event column could not be
        # identified (UNRESOLVED) must still register as churn-relevant --
        # outcome.is_resolved() is False here (as it would also be for a
        # field that was simply never populated), so is_churn_relevant()
        # must NOT be implemented in terms of is_resolved(); it must check
        # provenance, which correctly distinguishes "churn, unresolved"
        # from "not churn at all".
        canonical = _canonical(
            outcome=_unresolved(SemanticRole.OUTCOME, "semantic.churn_event_col"),
        )
        self.assertIsNone(canonical.outcome_column())
        self.assertTrue(canonical.is_churn_relevant())

        # Contrast: a field that was never populated at all (ordinary
        # question, no churn signal anywhere) is UNRESOLVED for a
        # completely different reason and must NOT be misread as churn.
        ordinary_unresolved = _canonical(
            outcome=_unresolved(SemanticRole.OUTCOME, "semantic.target_metric_col"),
        )
        self.assertIsNone(ordinary_unresolved.outcome_column())
        self.assertFalse(ordinary_unresolved.is_churn_relevant())



if __name__ == "__main__":
    unittest.main()
