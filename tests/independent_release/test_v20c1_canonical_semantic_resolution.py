"""v20-C1 regression suite: the richer CanonicalSemanticResolution model
(ResolvedSemanticField / SemanticCandidateSet / CanonicalSemanticResolution)
and its builder, build_canonical_semantic_resolution().

This is deliberately NOT yet wired into predictive_hypothesis.py or
experiment_synthesizer.py (that is v20-C2/C3) -- these tests only cover the
new API surface in isolation, including parity with the existing
resolve_group_dimension() fail-closed behavior it generalizes.
"""
import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.intelligence.predictive_hypothesis import resolve_group_dimension
from packages.analytics_core.src.intelligence.semantic_resolution_builder import build_canonical_semantic_resolution
from packages.schemas.src.semantic_role import ResolutionStatus, SemanticRole


@dataclass
class _FakeSemantic:
    """Minimal stand-in for SemanticResolution carrying only the fields
    these tests exercise."""
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


class TestDimensionResolutionParityWithExistingHelper(unittest.TestCase):
    """The new ResolvedSemanticField.dimension must reproduce
    resolve_group_dimension()'s exact fail-closed decision in every case,
    since it is a direct generalization of that function's logic."""

    def _assert_parity(self, semantic):
        expected_val, expected_status, expected_candidates = resolve_group_dimension(semantic)
        canon = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual(expected_status, canon.dimension.status.value)
        self.assertEqual(list(expected_candidates), list(canon.dimension.candidates))
        self.assertEqual(expected_val or None, canon.primary_dimension())

    def test_explicit_group_dimension_resolved(self):
        self._assert_parity(_FakeSemantic(group_dimension_col="region", available_categorical_cols=["region", "segment"]))

    def test_single_candidate_resolved(self):
        self._assert_parity(_FakeSemantic(group_dimension_col=None, available_categorical_cols=["segment"]))

    def test_zero_candidates_unresolved(self):
        self._assert_parity(_FakeSemantic(group_dimension_col=None, available_categorical_cols=[]))

    def test_multiple_candidates_ambiguous(self):
        self._assert_parity(_FakeSemantic(group_dimension_col=None, available_categorical_cols=["region", "segment", "plan"]))


class TestOutcomeFieldChurnAwareness(unittest.TestCase):
    def test_resolved_churn_event_produces_outcome_binding(self):
        semantic = _FakeSemantic(target_metric_col=None, churn_event_col="churned", churn_event_resolution_status="RESOLVED")
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertEqual("churned", canon.outcome_column())
        self.assertEqual(SemanticRole.OUTCOME, canon.outcome.role)
        self.assertEqual(["churned"], list(canon.outcome.candidates))

    def test_ambiguous_churn_event_preserves_candidate_list(self):
        """This is exactly the case build_binding_set()'s flat OUTCOME
        binding cannot represent -- churn_event_ambiguity is a whole
        candidate list, not a single column, and previously had nowhere
        to live once the semantic model was flattened."""
        semantic = _FakeSemantic(
            target_metric_col=None, churn_event_col=None,
            churn_event_resolution_status="AMBIGUOUS",
            churn_event_ambiguity=["churned", "cancelled", "subscription_churn"],
        )
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertIsNone(canon.outcome_column())
        self.assertEqual(ResolutionStatus.AMBIGUOUS, canon.outcome.status)
        self.assertEqual(["churned", "cancelled", "subscription_churn"], list(canon.outcome.candidates))

    def test_unresolved_churn_outcome_not_available(self):
        semantic = _FakeSemantic(target_metric_col=None, churn_event_col=None, churn_outcome_available=False)
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertIsNone(canon.outcome_column())
        self.assertEqual(ResolutionStatus.UNRESOLVED, canon.outcome.status)

    def test_claimed_resolved_status_without_column_fails_closed(self):
        """A defensive case: if churn_event_resolution_status somehow says
        RESOLVED but churn_event_col is empty, this must not be trusted as
        resolved."""
        semantic = _FakeSemantic(target_metric_col=None, churn_event_col=None, churn_event_resolution_status="RESOLVED", churn_outcome_available=False)
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertIsNone(canon.outcome_column())
        self.assertEqual(ResolutionStatus.UNRESOLVED, canon.outcome.status)

    def test_non_churn_investigation_uses_target_metric_as_outcome(self):
        semantic = _FakeSemantic(target_metric_col="revenue")
        canon = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual("revenue", canon.outcome_column())
        self.assertEqual(SemanticRole.OUTCOME, canon.outcome.role)

    def test_forecast_task_uses_target_role(self):
        semantic = _FakeSemantic(target_metric_col="revenue")
        canon = build_canonical_semantic_resolution(semantic, "FORECAST")
        self.assertEqual("revenue", canon.outcome_column())
        self.assertEqual(SemanticRole.TARGET, canon.outcome.role)


class TestConfounderSetIsACollectionNotASingleValue(unittest.TestCase):
    def test_confounders_preserved_as_named_purpose_set(self):
        semantic = _FakeSemantic(churn_confounder_cols=["cohort", "tenure_band"])
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertEqual("CONFOUNDER", canon.confounder_set().purpose)
        self.assertEqual(["cohort", "tenure_band"], list(canon.confounder_set().columns))
        self.assertFalse(canon.confounder_set().is_empty())

    def test_no_confounders_is_a_normal_empty_set_not_an_error(self):
        semantic = _FakeSemantic(churn_confounder_cols=[])
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertTrue(canon.confounder_set().is_empty())


class TestAvailableCategoricalCandidatesDistinctFromDimensionCandidates(unittest.TestCase):
    def test_full_pool_preserved_even_when_dimension_explicitly_resolved(self):
        """resolve_group_dimension() (and this builder's dimension field)
        report candidates=[group_dimension_col] once a dimension is
        explicitly resolved -- losing the fact that other categorical
        columns existed in the dataset. available_categorical_candidates
        keeps that full pool regardless of how the dimension resolved."""
        semantic = _FakeSemantic(group_dimension_col="region", available_categorical_cols=["region", "segment", "plan"])
        canon = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual(["region"], list(canon.dimension.candidates))
        self.assertEqual(["region", "segment", "plan"], list(canon.available_categorical_candidates))


class TestExposureAndCensoredFields(unittest.TestCase):
    def test_exposure_and_censored_resolved_when_present(self):
        semantic = _FakeSemantic(churn_exposure_col="tenure_days", churn_censored_col="is_censored")
        canon = build_canonical_semantic_resolution(semantic, "SURVIVAL_CHURN")
        self.assertEqual("tenure_days", canon.exposure_column())
        self.assertEqual("is_censored", canon.censored_column())

    def test_exposure_and_censored_unresolved_when_absent(self):
        semantic = _FakeSemantic()
        canon = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertIsNone(canon.exposure_column())
        self.assertIsNone(canon.censored_column())


class TestBindingsFieldStillCanonicalForExecution(unittest.TestCase):
    """CanonicalSemanticResolution.bindings must remain a real
    SemanticBindingSet, usable exactly as validate_experiment_against_contract
    already expects -- this model adds richness, it does not replace or
    bypass the v20-A/B1 execution-facing binding set."""

    def test_bindings_field_is_a_real_semantic_binding_set(self):
        semantic = _FakeSemantic(target_metric_col="revenue", group_dimension_col="region", available_categorical_cols=["region"])
        canon = build_canonical_semantic_resolution(semantic, "ASSOCIATION")
        self.assertEqual("revenue", canon.bindings.projected_target_column())
        self.assertEqual("region", canon.bindings.projected_group_dimension())

    def test_reuses_a_precomputed_binding_set_instead_of_rebuilding(self):
        from packages.analytics_core.src.intelligence.semantic_binding_builder import build_binding_set
        semantic = _FakeSemantic(target_metric_col="revenue")
        precomputed = build_binding_set(semantic, "ASSOCIATION")
        canon = build_canonical_semantic_resolution(semantic, "ASSOCIATION", bindings=precomputed)
        self.assertIs(precomputed, canon.bindings)


if __name__ == "__main__":
    unittest.main()
