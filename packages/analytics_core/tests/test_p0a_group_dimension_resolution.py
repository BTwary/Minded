"""Regression tests for P0-A (defect019 audit, section 2).

Covers the removal of the semantic "segment" fallback in
predictive_hypothesis.py (_synthesize_churn_hypotheses /
_synthesize_segmentation_hypotheses) and adversarial_attacker.py.

Required behavior per the audit instruction:
  - explicit resolved dimension          -> use it
  - exactly one valid candidate          -> optionally resolve it
  - zero candidates                      -> unresolved
  - multiple candidates                  -> ambiguous
  - never invent "segment"

Five scenarios required by the instruction are exercised directly against
resolve_group_dimension: zero dimensions, one dimension, two equally
plausible dimensions, an explicit user/resolved dimension, and conflicting
(3+) dimensions.
"""
import pandas as pd
import pytest

from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    HypothesisSynthesizer,
    PredictiveHypothesis,
    resolve_group_dimension,
)
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker


def _semantic(group_dimension_col=None, available_categorical_cols=None, **kwargs):
    defaults = dict(
        primary_dataset_name="customers",
        target_metric_col="monthly_spend",
        group_dimension_col=group_dimension_col,
        time_col=None,
        table_grain="row",
        available_numeric_cols=["monthly_spend"],
        available_categorical_cols=available_categorical_cols or [],
    )
    defaults.update(kwargs)
    return SemanticResolution(**defaults)


# ---------------------------------------------------------------------------
# resolve_group_dimension: the single authoritative rule
# ---------------------------------------------------------------------------

class TestResolveGroupDimension:
    def test_zero_dimensions_is_unresolved_not_segment(self):
        dim, status, candidates = resolve_group_dimension(_semantic(available_categorical_cols=[]))
        assert dim == ""
        assert status == "UNRESOLVED"
        assert candidates == []
        assert dim != "segment"

    def test_one_dimension_is_resolved(self):
        dim, status, candidates = resolve_group_dimension(_semantic(available_categorical_cols=["plan"]))
        assert dim == "plan"
        assert status == "RESOLVED"
        assert candidates == ["plan"]

    def test_two_equally_plausible_dimensions_is_ambiguous_not_first_wins(self):
        dim, status, candidates = resolve_group_dimension(
            _semantic(available_categorical_cols=["region", "plan"])
        )
        assert dim == ""
        assert status == "AMBIGUOUS"
        assert set(candidates) == {"region", "plan"}
        assert dim != "segment"

    def test_explicit_user_dimension_is_resolved_and_takes_priority(self):
        dim, status, candidates = resolve_group_dimension(
            _semantic(group_dimension_col="acquisition_channel", available_categorical_cols=["region", "plan"])
        )
        assert dim == "acquisition_channel"
        assert status == "RESOLVED"
        assert candidates == ["acquisition_channel"]

    def test_conflicting_multiple_dimensions_is_ambiguous(self):
        dim, status, candidates = resolve_group_dimension(
            _semantic(available_categorical_cols=["region", "plan", "acquisition_channel"])
        )
        assert dim == ""
        assert status == "AMBIGUOUS"
        assert set(candidates) == {"region", "plan", "acquisition_channel"}

    def test_never_returns_literal_segment_fallback(self):
        # A dataset that happens to have NO column literally named "segment"
        # must never end up with target_dimension == "segment" regardless of
        # how many/few candidates exist.
        for cols in ([], ["a"], ["a", "b"], ["a", "b", "c"]):
            dim, _, _ = resolve_group_dimension(_semantic(available_categorical_cols=cols))
            assert dim != "segment"


# ---------------------------------------------------------------------------
# _synthesize_churn_hypotheses: real dataset holding a genuine "segment"
# column must not be silently swept up by a fabricated fallback
# ---------------------------------------------------------------------------

class TestChurnHypothesisDimension:
    def test_ambiguous_dimension_does_not_silently_pick_real_segment_column(self):
        """If the dataset happens to contain a REAL column literally named
        'segment' alongside other equally-plausible categorical columns, the
        engine must not silently prefer/invent it -- ambiguity must remain
        explicit so a downstream consumer cannot mistake this for a
        deliberate semantic match."""
        semantic = _semantic(
            available_categorical_cols=["segment", "region"],
            churn_event_col="churn",
            churn_outcome_available=True,
        )
        hyps = HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, "Why are customers churning?")
        h1 = next(h for h in hyps if h.hypothesis_code == "HYP-01")
        assert h1.dimension_resolution_status == "AMBIGUOUS"
        # target_dimension must be empty, not fabricated -- even though
        # "segment" happens to be a real, legitimate candidate here, it must
        # not be chosen implicitly over "region".
        assert h1.target_dimension == ""
        assert set(h1.dimension_candidates) == {"segment", "region"}

    def test_zero_candidate_dimension_is_unresolved(self):
        semantic = _semantic(
            available_categorical_cols=[],
            churn_event_col="churn",
            churn_outcome_available=True,
        )
        hyps = HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, "Why are customers churning?")
        h1 = next(h for h in hyps if h.hypothesis_code == "HYP-01")
        assert h1.dimension_resolution_status == "UNRESOLVED"
        assert h1.target_dimension == ""

    def test_single_candidate_dimension_is_resolved(self):
        semantic = _semantic(
            available_categorical_cols=["region"],
            churn_event_col="churn",
            churn_outcome_available=True,
        )
        hyps = HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, "Why are customers churning?")
        h1 = next(h for h in hyps if h.hypothesis_code == "HYP-01")
        assert h1.dimension_resolution_status == "RESOLVED"
        assert h1.target_dimension == "region"

    def test_ambiguous_churn_outcome_column_is_distinguished_from_absent(self):
        """Section 8: when churn_event_resolution_status is AMBIGUOUS (not
        just absent), the generated unidentifiability hypothesis must say so
        explicitly rather than reading identically to the genuinely-absent
        case."""
        semantic = _semantic(
            available_categorical_cols=["region"],
            churn_event_col=None,
            churn_outcome_available=False,
            churn_event_resolution_status="AMBIGUOUS",
            churn_event_ambiguity=["churn", "is_churn", "cancelled"],
        )
        hyps = HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, "Why are customers churning?")
        h1 = next(h for h in hyps if h.hypothesis_code == "HYP-01")
        assert "AMBIGUOUS" in h1.claim.upper() or "ambiguous" in h1.claim.lower()
        assert "churn" in h1.claim and "is_churn" in h1.claim and "cancelled" in h1.claim

    def test_genuinely_absent_churn_outcome_is_distinguished_from_ambiguous(self):
        semantic = _semantic(
            available_categorical_cols=["region"],
            churn_event_col=None,
            churn_outcome_available=False,
            churn_event_resolution_status="UNRESOLVED",
            churn_event_ambiguity=[],
        )
        hyps = HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, "Why are customers churning?")
        h1 = next(h for h in hyps if h.hypothesis_code == "HYP-01")
        assert "No churn/cancellation outcome column is recorded" in h1.claim
        assert "AMBIGUOUS" not in h1.claim.upper()


class TestSegmentationHypothesisDimension:
    def test_ambiguous_segmentation_dimension_is_explicit(self):
        semantic = _semantic(available_categorical_cols=["region", "plan"])
        hyps = HypothesisSynthesizer._synthesize_segmentation_hypotheses(semantic, "How do outcomes vary by segment?")
        h1 = next(h for h in hyps if h.hypothesis_code == "HYP-01")
        assert h1.dimension_resolution_status == "AMBIGUOUS"
        assert h1.target_dimension == ""
        assert "region" in h1.claim and "plan" in h1.claim


# ---------------------------------------------------------------------------
# AdversarialAttacker: must fail closed instead of attacking a fabricated
# "segment" column (which is dangerous specifically because real datasets
# commonly DO have a column named "segment").
# ---------------------------------------------------------------------------

class TestAdversarialAttackerNoFabricatedDimension:
    def test_unresolved_target_dimension_does_not_attack_real_segment_column(self):
        # This dataset has a REAL "segment" column, unrelated to the
        # hypothesis under test (whose target_dimension was never resolved).
        # The old `leading_hyp.target_dimension or "segment"` fallback would
        # silently attack against it and produce a misleading result.
        df = pd.DataFrame({
            "segment": ["a", "b", "a", "b"] * 5,
            "monthly_spend": [10, 90, 12, 88] * 5,
        })
        hyp = PredictiveHypothesis(
            id="HYP-01", hypothesis_code="HYP-01", claim="c", mechanism="m",
            predicted_observables_if_true=[], predicted_observables_if_false=[],
            falsification_criteria="f", required_assumptions=[],
            prior_probability=1.0, posterior_probability=1.0,
            target_metric="monthly_spend", target_dimension="",  # UNRESOLVED
            dimension_resolution_status="AMBIGUOUS",
        )
        result = AdversarialAttacker.design_attack(leading_hyp=hyp, counter_hyps=[], df=df)
        # An attack that could not be constructed is NOT_APPLICABLE, never SURVIVED
        # (SURVIVED would read downstream as adversarial evidence that never existed).
        assert result.attack_status == "NOT_APPLICABLE"
        assert result.is_falsified is False
        assert "not applicable" in result.epistemic_impact.lower()
        # Must never have built a query against the real "segment" column.
        assert "segment" not in result.discriminating_test_sql

    def test_resolved_dimension_still_attacks_normally(self):
        df = pd.DataFrame({
            "region": ["east", "west", "east", "west"] * 5,
            "monthly_spend": [10, 90, 12, 88] * 5,
        })
        hyp = PredictiveHypothesis(
            id="HYP-01", hypothesis_code="HYP-01", claim="c", mechanism="m",
            predicted_observables_if_true=[], predicted_observables_if_false=[],
            falsification_criteria="f", required_assumptions=[],
            prior_probability=1.0, posterior_probability=1.0,
            target_metric="monthly_spend", target_dimension="region",
            dimension_resolution_status="RESOLVED",
        )
        result = AdversarialAttacker.design_attack(leading_hyp=hyp, counter_hyps=[], df=df)
        assert "region" in result.discriminating_test_sql
