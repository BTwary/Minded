"""Regression tests for P0-B (v17 controller hardening).

Covers:
  1. controller.py:1331 `filtered or candidates` -> fail-closed
     `compute_task_compatible_candidates`.
  2. controller.py 1573/1581/1584 `semantic.group_dimension_col or 'segment'`
     -> `build_simpsons_conditional_candidate`, using the shared v16
     `resolve_group_dimension` authority.

Both were extracted from in-loop closures into standalone, independently
testable module-level functions in `controller.py` as part of this fix, so
these tests exercise the real production code paths directly rather than
re-implementing/duplicating the logic.
"""
import pytest

from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition, AggregationType
from packages.analytics_core.src.runtime.controller import (
    compute_task_compatible_candidates,
    build_simpsons_conditional_candidate,
)


def _candidate(code: str, **kwargs) -> CandidateExperiment:
    defaults = dict(
        target_hypothesis_code="HYP-01",
        tool_name="duckdb_sql",
        query_sql="SELECT 1",
        description="d",
        aggregation_type="SUM",
    )
    defaults.update(kwargs)
    return CandidateExperiment(code=code, **defaults)


def _semantic(group_dimension_col=None, available_categorical_cols=None, metric_definition=None, **kwargs):
    defaults = dict(
        primary_dataset_name="customers",
        target_metric_col="monthly_spend",
        group_dimension_col=group_dimension_col,
        time_col=None,
        table_grain="row",
        available_numeric_cols=["monthly_spend"],
        available_categorical_cols=available_categorical_cols or [],
        metric_definition=metric_definition,
    )
    defaults.update(kwargs)
    return SemanticResolution(**defaults)


def _metric_def():
    return MetricDefinition(
        name="monthly_spend", table_name="customers", source_columns=["monthly_spend"],
        semantic_type="sum_measure", aggregation_type=AggregationType.SUM,
    )


# ---------------------------------------------------------------------------
# Test A / B — compute_task_compatible_candidates (section 1)
# ---------------------------------------------------------------------------

class TestCandidateFiltering:
    def test_a_no_compatible_candidates_returns_empty_not_all(self):
        """FORECAST task with only non-forecast candidates: must return []
        and record a diagnostic -- never fall back to executing the
        incompatible candidates."""
        candidates = [_candidate("EXP-CORR-1"), _candidate("EXP-DIAG-1")]
        filtered, diagnostic = compute_task_compatible_candidates(candidates, "FORECAST", _semantic())
        assert filtered == []
        assert diagnostic is not None
        assert diagnostic["diagnostic"] == "NO_COMPATIBLE_EXPERIMENT_CANDIDATE"
        assert diagnostic["analytical_task"] == "FORECAST"
        assert set(diagnostic["rejected_candidate_codes"]) == {"EXP-CORR-1", "EXP-DIAG-1"}
        assert diagnostic["allowed_prefixes"] == ["EXP-FORECAST"]

    def test_b_compatible_candidates_preserved_incompatible_dropped(self):
        """A mix of compatible and incompatible candidates: only the
        compatible ones survive, and no diagnostic fires."""
        candidates = [_candidate("EXP-FORECAST-BASELINE"), _candidate("EXP-CORR-1"), _candidate("EXP-FORECAST-ARIMA")]
        filtered, diagnostic = compute_task_compatible_candidates(candidates, "FORECAST", _semantic())
        assert {c.code for c in filtered} == {"EXP-FORECAST-BASELINE", "EXP-FORECAST-ARIMA"}
        assert diagnostic is None

    def test_causal_task_never_restores_non_causal_candidates(self):
        candidates = [_candidate("EXP-DESCRIBE-1"), _candidate("EXP-CORR-2")]
        filtered, diagnostic = compute_task_compatible_candidates(candidates, "CAUSAL", _semantic())
        assert filtered == []
        assert diagnostic["analytical_task"] == "CAUSAL"

    def test_task_with_no_prefix_restriction_passes_through(self):
        """Tasks that don't define keep_prefixes (e.g. DESCRIPTIVE) are
        unaffected by this filter -- verifies the fail-closed change did not
        change behavior for unrestricted tasks."""
        candidates = [_candidate("EXP-ANYTHING")]
        filtered, diagnostic = compute_task_compatible_candidates(candidates, "DESCRIPTIVE", _semantic())
        assert filtered == candidates
        assert diagnostic is None

    def test_empty_input_returns_empty_no_diagnostic(self):
        filtered, diagnostic = compute_task_compatible_candidates([], "FORECAST", _semantic())
        assert filtered == []
        assert diagnostic is None


# ---------------------------------------------------------------------------
# Test C / D / E / F / G — build_simpsons_conditional_candidate (sections 2/3)
# ---------------------------------------------------------------------------

class TestConditionalAdversarialExperiment:
    def test_c_resolved_dimension_constructs_conditional_experiment(self):
        semantic = _semantic(group_dimension_col="region", metric_definition=_metric_def())
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "plan", "HYP-01")
        assert diagnostic is None
        assert candidate is not None
        assert "region" in candidate.query_sql
        assert "plan" in candidate.query_sql
        assert candidate.dimensions[:2] == ["region", "plan"]

    def test_d_unresolved_dimension_produces_no_sql_experiment(self):
        semantic = _semantic(available_categorical_cols=[], metric_definition=_metric_def())
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "plan", "HYP-01")
        assert candidate is None
        assert diagnostic["status"] == "UNRESOLVED_PRIMARY_DIMENSION"
        assert diagnostic["candidate_dimensions"] == []

    def test_e_ambiguous_dimension_produces_no_sql_and_preserves_candidates(self):
        semantic = _semantic(available_categorical_cols=["region", "acquisition_channel"], metric_definition=_metric_def())
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "plan", "HYP-01")
        assert candidate is None
        assert diagnostic["status"] == "AMBIGUOUS_PRIMARY_DIMENSION"
        assert set(diagnostic["candidate_dimensions"]) == {"region", "acquisition_channel"}

    def test_f_real_segment_column_not_selected_when_ambiguous(self):
        """Dataset contains segment, region, plan as candidates, and the
        resolver is genuinely AMBIGUOUS: 'segment' must NOT be silently
        selected merely because it happens to be present."""
        semantic = _semantic(available_categorical_cols=["segment", "region", "plan"], metric_definition=_metric_def())
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "channel", "HYP-01")
        assert candidate is None
        assert diagnostic["status"] == "AMBIGUOUS_PRIMARY_DIMENSION"
        assert set(diagnostic["candidate_dimensions"]) == {"segment", "region", "plan"}

    def test_f_real_segment_column_not_selected_when_unresolved(self):
        """Even a dataset with zero available categorical candidates must
        not fall back to 'segment' as a placeholder."""
        semantic = _semantic(available_categorical_cols=[], metric_definition=_metric_def())
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "channel", "HYP-01")
        assert candidate is None
        assert diagnostic["status"] == "UNRESOLVED_PRIMARY_DIMENSION"

    def test_g_sql_never_contains_fabricated_segment_placeholder(self):
        """SQL-safety assertion: across resolved/unresolved/ambiguous cases,
        no constructed query_sql may contain the literal 'segment' unless it
        was the actually-resolved dimension. Inspects the CandidateExperiment
        object directly, not just a text search."""
        # Unresolved / ambiguous cases produce no candidate at all -> no SQL.
        for cols in ([], ["region", "plan"], ["segment", "region"]):
            semantic = _semantic(available_categorical_cols=cols, metric_definition=_metric_def())
            candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "channel", "HYP-01")
            assert candidate is None
            assert diagnostic is not None

        # Resolved case: 'segment' only appears when it is the real,
        # legitimately resolved column -- not as a placeholder.
        semantic_real_segment = _semantic(group_dimension_col="segment", metric_definition=_metric_def())
        candidate_real, diagnostic_real = build_simpsons_conditional_candidate(semantic_real_segment, "channel", "HYP-01")
        assert diagnostic_real is None
        assert candidate_real.dimensions[0] == "segment"  # legitimately resolved, not fabricated
        assert candidate_real.target_dimension == "segment_channel"

    def test_unresolved_metric_semantics_takes_precedence_over_dimension_check(self):
        semantic = _semantic(group_dimension_col="region", metric_definition=None)
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "plan", "HYP-01")
        assert candidate is None
        assert diagnostic["status"] == "UNRESOLVED_METRIC_SEMANTICS"

    def test_conditional_candidate_never_executed_without_resolution(self):
        """End-to-end guard: build a candidate list the way the controller
        loop would (starting empty, inserting the conditional candidate only
        if resolution succeeds) and confirm an ambiguous/unresolved
        dimension never contributes an executable candidate to the list."""
        semantic = _semantic(available_categorical_cols=["segment", "region"], metric_definition=_metric_def())
        available_candidates = [_candidate("EXP-CHURN-CRUDE")]
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "plan", "HYP-01")
        if candidate is not None:
            available_candidates.insert(0, candidate)
        assert all(c.code != f"EXP-COND-PLAN" for c in available_candidates)
        assert len(available_candidates) == 1
        assert diagnostic["status"] == "AMBIGUOUS_PRIMARY_DIMENSION"
