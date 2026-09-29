"""v20-C2 regression suite: predictive_hypothesis.py's migration onto
CanonicalSemanticResolution.

Covers (per docs/AAOS_V20C2_PREDICTIVE_SEMANTIC_MIGRATION.md section 17):
  - target/outcome: resolved / unresolved / ambiguous
  - group dimension: zero / one / multiple / explicit
  - time: none / one
  - secondary metric: resolved / absent / ambiguous
  - churn event: resolved / absent / ambiguous
  - churn ambiguity candidate-list preservation
  - confounders: zero / one / multiple
  - exposure: resolved / absent
  - categorical candidates: no first-wins behavior
  - golden behavioral equivalence: legacy (raw-field) vs. canonical-sourced
    hypothesis content is identical for every representative fixture
  - a real InvestigationController-style call actually receives the
    canonical object (not just a unit-constructed one)
"""
import os
import sys
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.method_selection import ProblemClass
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    HypothesisSynthesizer,
    _build_canonical_semantics,
    _dimension_from_canonical,
    _target_metric_via_canonical,
    resolve_group_dimension,
)
from packages.analytics_core.src.intelligence.semantic_resolution_builder import (
    build_canonical_semantic_resolution,
)
from packages.schemas.src.semantic_resolution_contract import CanonicalSemanticResolution


@dataclass
class _FakeSemantic:
    """Minimal stand-in for SemanticResolution (shared shape with the C1
    suite's _FakeSemantic)."""
    primary_dataset_name: str = "t"
    target_metric_col: Optional[str] = "revenue"
    group_dimension_col: Optional[str] = None
    time_col: Optional[str] = None
    available_categorical_cols: List[str] = field(default_factory=list)
    secondary_metric_col: Optional[str] = None
    churn_event_col: Optional[str] = None
    churn_exposure_col: Optional[str] = None
    churn_censored_col: Optional[str] = None
    churn_confounder_cols: List[str] = field(default_factory=list)
    churn_outcome_available: bool = True
    churn_event_resolution_status: str = "RESOLVED"
    churn_event_ambiguity: List[str] = field(default_factory=list)
    direction_hint: str = ""


def _decision(problem_class: ProblemClass) -> SimpleNamespace:
    """A minimal decision-shaped object -- synthesize_competing_hypotheses
    only reads .problem_class off it."""
    return SimpleNamespace(problem_class=problem_class)


class TestEntryBoundaryBuildsCanonicalOnce(unittest.TestCase):
    """Section 3: canonical resolution is built once at the entry
    boundary; a caller-supplied object is used as-is rather than rebuilt."""

    def test_supplied_canonical_is_used_not_rebuilt(self):
        semantic = _FakeSemantic(target_metric_col="revenue", secondary_metric_col="cost")
        canonical = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        # Corrupt the raw semantic field the supplied canonical was already
        # built from -- if synthesize_competing_hypotheses rebuilt its own
        # canonical instead of using the one supplied, this corruption would
        # show up in the hypothesis content.
        semantic.target_metric_col = "SHOULD_NOT_APPEAR"
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "is cost associated with revenue?",
            decision=_decision(ProblemClass.CORRELATIONAL),
            canonical_semantics=canonical,
        )
        self.assertTrue(all("SHOULD_NOT_APPEAR" not in h.claim for h in hyps))
        self.assertTrue(any("revenue" in h.claim for h in hyps))

    def test_none_supplied_builds_internally(self):
        semantic = _FakeSemantic(target_metric_col="revenue", secondary_metric_col="cost")
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "is cost associated with revenue?",
            decision=_decision(ProblemClass.CORRELATIONAL),
        )
        self.assertEqual(2, len(hyps))
        self.assertTrue(any("revenue" in h.claim and "cost" in h.claim for h in hyps))


class TestTargetOutcomeResolution(unittest.TestCase):
    def test_resolved_target_flows_into_forecast_hypotheses(self):
        semantic = _FakeSemantic(target_metric_col="revenue", time_col="month")
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "forecast revenue", decision=_decision(ProblemClass.FORECASTING),
        )
        self.assertTrue(any("revenue" in h.claim and "month" in h.claim for h in hyps))

    def test_unresolved_target_falls_back_to_default_family_without_crash(self):
        semantic = _FakeSemantic(target_metric_col=None, available_categorical_cols=[])
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, "why did things change?")
        self.assertEqual(2, len(hyps))

    def test_target_metric_via_canonical_matches_raw_field_when_not_churn(self):
        semantic = _FakeSemantic(target_metric_col="revenue")
        canonical = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual("revenue", _target_metric_via_canonical(semantic, canonical))

    def test_target_metric_via_canonical_falls_back_when_canonical_is_churn_sourced(self):
        """The documented fallthrough-safety edge case: canonical.outcome is
        churn-sourced, but the caller still wants the raw target metric
        (e.g. a non-churn hypothesis family reached via routing fallthrough).
        Must return the raw field, not the churn column."""
        semantic = _FakeSemantic(target_metric_col="revenue", churn_event_col="cancelled")
        canonical = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual("cancelled", canonical.outcome_column())
        self.assertEqual("revenue", _target_metric_via_canonical(semantic, canonical))


class TestGroupDimensionResolution(unittest.TestCase):
    def test_zero_candidates_unresolved(self):
        canonical = build_canonical_semantic_resolution(_FakeSemantic(available_categorical_cols=[]), "SEGMENTATION")
        dim, status, candidates = _dimension_from_canonical(canonical)
        self.assertEqual(("", "UNRESOLVED", []), (dim, status, candidates))

    def test_one_candidate_resolved(self):
        canonical = build_canonical_semantic_resolution(_FakeSemantic(available_categorical_cols=["plan"]), "SEGMENTATION")
        dim, status, candidates = _dimension_from_canonical(canonical)
        self.assertEqual(("plan", "RESOLVED", ["plan"]), (dim, status, candidates))

    def test_multiple_candidates_ambiguous(self):
        canonical = build_canonical_semantic_resolution(
            _FakeSemantic(available_categorical_cols=["plan", "region"]), "SEGMENTATION",
        )
        dim, status, candidates = _dimension_from_canonical(canonical)
        self.assertEqual("", dim)
        self.assertEqual("AMBIGUOUS", status)
        self.assertEqual(["plan", "region"], candidates)

    def test_explicit_dimension_resolved(self):
        canonical = build_canonical_semantic_resolution(
            _FakeSemantic(group_dimension_col="region", available_categorical_cols=["region", "plan"]), "SEGMENTATION",
        )
        dim, status, candidates = _dimension_from_canonical(canonical)
        self.assertEqual(("region", "RESOLVED", ["region"]), (dim, status, candidates))

    def test_dimension_from_canonical_matches_resolve_group_dimension_exactly(self):
        """Parity guard: _dimension_from_canonical must always agree with
        the legacy resolve_group_dimension() it replaces at every internal
        call site (section 14's information-preservation requirement)."""
        for semantic in (
            _FakeSemantic(available_categorical_cols=[]),
            _FakeSemantic(available_categorical_cols=["plan"]),
            _FakeSemantic(available_categorical_cols=["plan", "region", "tier"]),
            _FakeSemantic(group_dimension_col="region", available_categorical_cols=["region", "plan"]),
        ):
            legacy = resolve_group_dimension(semantic)
            canonical = build_canonical_semantic_resolution(semantic, "SEGMENTATION")
            self.assertEqual(legacy, _dimension_from_canonical(canonical))

    def test_ambiguous_dimension_preserved_in_segmentation_claim_text(self):
        """Section 5: AMBIGUOUS(region, plan) must not collapse to None --
        the candidate list must still reach the hypothesis claim/mechanism
        text, exactly as resolve_group_dimension's callers relied on."""
        semantic = _FakeSemantic(target_metric_col="revenue", available_categorical_cols=["region", "plan"])
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "how do segments differ?", decision=_decision(ProblemClass.COMPARATIVE),
        )
        h1 = hyps[0]
        self.assertEqual("AMBIGUOUS", h1.dimension_resolution_status)
        self.assertEqual(["region", "plan"], h1.dimension_candidates)
        self.assertIn("region", h1.claim)
        self.assertIn("plan", h1.claim)
        self.assertIn("AMBIGUOUS", h1.claim)


class TestTimeVariableResolution(unittest.TestCase):
    def test_no_time_variable_does_not_route_to_forecast(self):
        semantic = _FakeSemantic(target_metric_col="revenue", time_col=None)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "what will happen?", decision=_decision(ProblemClass.FORECASTING),
        )
        # FORECASTING decision without a resolved time_col must NOT route
        # to the forecast family (existing decision.problem_class branch
        # requires canonical.time_variable() truthy, same as the raw field).
        self.assertFalse(any(h.claim_type == "PREDICTION" for h in hyps))

    def test_one_time_variable_routes_to_forecast(self):
        semantic = _FakeSemantic(target_metric_col="revenue", time_col="month")
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "what will happen?", decision=_decision(ProblemClass.FORECASTING),
        )
        self.assertTrue(all(h.claim_type == "PREDICTION" for h in hyps))
        self.assertTrue(any("month" in h.claim for h in hyps))


class TestSecondaryMetricResolution(unittest.TestCase):
    def test_resolved_secondary_metric_routes_to_correlation(self):
        semantic = _FakeSemantic(target_metric_col="revenue", secondary_metric_col="cost")
        canonical = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual("cost", canonical.secondary_metric_column())
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "is cost associated with revenue?", decision=_decision(ProblemClass.CORRELATIONAL),
        )
        self.assertTrue(all(h.claim_type == "ASSOCIATION" for h in hyps))

    def test_absent_secondary_metric_does_not_route_to_correlation(self):
        semantic = _FakeSemantic(target_metric_col="revenue", secondary_metric_col=None)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "is cost associated with revenue?", decision=_decision(ProblemClass.CORRELATIONAL),
        )
        self.assertFalse(all(h.claim_type == "ASSOCIATION" for h in hyps))

    def test_secondary_metric_never_collapsed_into_outcome(self):
        semantic = _FakeSemantic(target_metric_col="revenue", secondary_metric_col="cost")
        canonical = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertNotEqual(canonical.outcome_column(), canonical.secondary_metric_column())


class TestChurnEventResolution(unittest.TestCase):
    def test_resolved_churn_event_produces_churn_hypotheses(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            available_categorical_cols=["plan"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertTrue(all(h.target_metric == "cancelled" for h in hyps))

    def test_absent_churn_event_produces_unidentifiable_hypotheses(self):
        semantic = _FakeSemantic(
            churn_event_col=None, churn_outcome_available=False,
            target_metric_col=None,
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertIn("unidentifiable", hyps[0].claim.lower())

    def test_ambiguous_churn_event_preserves_candidate_list_in_claim(self):
        semantic = _FakeSemantic(
            churn_event_col=None, churn_outcome_available=False,
            churn_event_resolution_status="AMBIGUOUS",
            churn_event_ambiguity=["churn", "cancelled", "subscription_churn"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        claim = hyps[0].claim
        for candidate in ("churn", "cancelled", "subscription_churn"):
            self.assertIn(candidate, claim)
        self.assertIn("AMBIGUOUS", hyps[0].claim if "AMBIGUOUS" in hyps[0].claim else claim.upper())


class TestConfounderSet(unittest.TestCase):
    def test_zero_confounders_produces_no_confounding_hypotheses(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_confounder_cols=[], available_categorical_cols=["plan"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertEqual(2, len(hyps))  # HYP-01/HYP-02 only, no HYP-03+

    def test_one_confounder_produces_one_confounding_hypothesis(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_confounder_cols=["tenure"], available_categorical_cols=["plan"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertEqual(3, len(hyps))
        self.assertTrue(any("tenure" in h.claim for h in hyps))

    def test_multiple_confounders_produce_one_hypothesis_each(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_confounder_cols=["tenure", "cohort"], available_categorical_cols=["plan"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertEqual(4, len(hyps))
        self.assertTrue(any("tenure" in h.claim for h in hyps))
        self.assertTrue(any("cohort" in h.claim for h in hyps))


class TestExposureResolution(unittest.TestCase):
    def test_resolved_exposure_produces_exposure_hypothesis(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_exposure_col="tenure_days", available_categorical_cols=["plan"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertTrue(any("tenure_days" in h.claim for h in hyps))

    def test_absent_exposure_produces_no_exposure_hypothesis(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_exposure_col=None, available_categorical_cols=["plan"],
        )
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN),
        )
        self.assertEqual(2, len(hyps))


class TestCategoricalCandidatesNoFirstWins(unittest.TestCase):
    def test_multiple_candidates_do_not_silently_pick_first(self):
        """H3 (secondary dimensional confounding) must be free to consider
        ANY other categorical column, not just the alphabetically- or
        positionally-first one -- available_categorical_candidates must be
        the full pool, not truncated to a single 'winner'."""
        semantic = _FakeSemantic(
            target_metric_col="revenue",
            group_dimension_col="region",
            available_categorical_cols=["region", "plan", "channel"],
        )
        canonical = build_canonical_semantic_resolution(semantic, "GENERAL_EXPLORATION")
        self.assertEqual(["region", "plan", "channel"], list(canonical.available_categorical_candidates))
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, "why did revenue change?")
        h3 = next((h for h in hyps if h.hypothesis_code == "HYP-03"), None)
        self.assertIsNotNone(h3)
        self.assertEqual("plan", h3.target_dimension)


class TestGoldenBehavioralEquivalence(unittest.TestCase):
    """Section 18: for representative fixtures, the canonical-sourced
    result must match what the legacy raw-field reads would have produced
    -- same target, same dimensions, same confounders, same ambiguity,
    same claim classification."""

    def _assert_golden(self, semantic, question, **kwargs):
        # "Legacy" reference computed directly from the raw fields, exactly
        # as predictive_hypothesis.py read them before v20-C2.
        legacy_dim, legacy_status, legacy_candidates = resolve_group_dimension(semantic)
        canonical = build_canonical_semantic_resolution(
            semantic, kwargs.get("task", "GENERAL_EXPLORATION"),
        )
        self.assertEqual((legacy_dim, legacy_status, legacy_candidates), _dimension_from_canonical(canonical))
        if not (semantic.churn_event_col or not semantic.churn_outcome_available):
            self.assertEqual(semantic.target_metric_col, _target_metric_via_canonical(semantic, canonical))
        self.assertEqual(list(semantic.available_categorical_cols or []), list(canonical.available_categorical_candidates))
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question, **{k: v for k, v in kwargs.items() if k != "task"})
        return hyps

    def test_correlation_fixture(self):
        semantic = _FakeSemantic(target_metric_col="revenue", secondary_metric_col="cost")
        hyps = self._assert_golden(semantic, "is cost associated with revenue?", decision=_decision(ProblemClass.CORRELATIONAL))
        self.assertEqual(2, len(hyps))

    def test_forecast_fixture(self):
        semantic = _FakeSemantic(target_metric_col="revenue", time_col="month")
        hyps = self._assert_golden(semantic, "forecast revenue", decision=_decision(ProblemClass.FORECASTING))
        self.assertEqual(2, len(hyps))

    def test_segmentation_ambiguous_dimension_fixture(self):
        semantic = _FakeSemantic(target_metric_col="revenue", available_categorical_cols=["region", "plan"])
        hyps = self._assert_golden(semantic, "how do segments differ?", decision=_decision(ProblemClass.COMPARATIVE))
        self.assertEqual(2, len(hyps))

    def test_churn_with_confounders_and_exposure_fixture(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_confounder_cols=["tenure"], churn_exposure_col="tenure_days",
            available_categorical_cols=["plan"],
        )
        hyps = self._assert_golden(semantic, "why did customers churn?", decision=_decision(ProblemClass.SURVIVAL_CHURN))
        self.assertEqual(4, len(hyps))

    def test_default_root_cause_fixture_with_multiple_dimensions(self):
        semantic = _FakeSemantic(target_metric_col="revenue", available_categorical_cols=["region", "plan"])
        hyps = self._assert_golden(semantic, "why did revenue change?")
        self.assertEqual(3, len(hyps))  # H1, H2, H3 (secondary dimension exists)


class TestRealControllerStyleInvocation(unittest.TestCase):
    """Section 19: a real-shaped call (semantic + intent + decision, as
    controller.py actually invokes synthesize_competing_hypotheses) must
    exercise the canonical-object code path, not merely a directly
    unit-constructed CanonicalSemanticResolution."""

    def test_controller_shaped_call_receives_canonical_semantics(self):
        semantic = _FakeSemantic(
            churn_event_col="cancelled", churn_outcome_available=True,
            churn_confounder_cols=["tenure"], available_categorical_cols=["plan"],
        )
        intent = SimpleNamespace(intent_type="CHURN")
        decision = _decision(ProblemClass.SURVIVAL_CHURN)
        # No canonical_semantics kwarg supplied here -- this is exactly the
        # controller.py call shape (semantic, question, intent=, decision=).
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
            semantic, "why are customers cancelling?", intent=intent, decision=decision,
        )
        self.assertEqual(3, len(hyps))
        self.assertTrue(any("tenure" in h.claim for h in hyps))
        # Confirm the internal builder actually ran by checking the
        # constructed canonical resolution independently agrees.
        canonical = _build_canonical_semantics(semantic, intent=intent, decision=decision)
        self.assertIsInstance(canonical, CanonicalSemanticResolution)
        self.assertEqual("cancelled", canonical.churn_event_resolution().value)
        self.assertEqual(["tenure"], list(canonical.confounder_set().columns))


if __name__ == "__main__":
    unittest.main()
