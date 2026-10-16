
"""Regression test for concentration experiment result-column alignment.

Historical bug:
The non-additive concentration branch reused an aggregation expression
aliased as ``total_metric`` while declaring
``primary_result_column="mean_metric"``. The declared column was therefore
missing from the query result at execution time.

This test extracts aliases from the generated SQL and verifies that every
concentration candidate's declared primary result column is actually
returned by its query. Both quoted and unquoted SQL identifiers are supported.
"""

import re

from packages.analytics_core.src.intelligence.experiment_synthesizer import (
    ExperimentSynthesizer,
)
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    PredictiveHypothesis,
)
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType


_SQL_ALIAS_PATTERN = re.compile(
    r'\bAS\s+(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*))',
    re.IGNORECASE,
)


def _extract_sql_aliases(query_sql: str) -> set[str]:
    """Return aliases declared with AS, regardless of identifier quoting."""
    aliases = set()

    for match in _SQL_ALIAS_PATTERN.finditer(query_sql):
        alias = next(group for group in match.groups() if group is not None)
        aliases.add(alias)

    return aliases


def test_concentration_non_additive_declared_column_matches_sql_alias():
    metric_def = MetricDefinition(
        name="unit_cost",
        table_name="data_table",
        source_columns=["unit_cost"],
        semantic_type="mean_measure",
        aggregation_type=AggregationType.MEAN,
        grain="row_level",
        is_additive=False,
        valid_aggregations=[AggregationType.MEAN],
        rationale="test",
        semantic_resolution_status="RESOLVED",
    )

    semantic = SemanticResolution(
        primary_dataset_name="data_table",
        target_metric_col="unit_cost",
        group_dimension_col="category",
        time_col=None,
        table_grain="row_level",
        available_numeric_cols=["unit_cost"],
        available_categorical_cols=["category"],
        metric_definition=metric_def,
    )

    hypothesis = PredictiveHypothesis(
        id="h1",
        hypothesis_code="HYP-01",
        claim="test claim",
        mechanism="test",
        predicted_observables_if_true=[],
        predicted_observables_if_false=[],
        falsification_criteria="",
        required_assumptions=[],
        prior_probability=0.5,
        posterior_probability=0.5,
        target_metric="unit_cost",
        target_dimension="category",
    )

    candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
        hypotheses=[hypothesis],
        semantic=semantic,
    )

    concentration_candidates = [
        candidate
        for candidate in candidates
        if candidate.code.startswith("EXP-CONC")
    ]

    assert concentration_candidates, "No concentration candidate generated"

    for candidate in concentration_candidates:
        aliases = _extract_sql_aliases(candidate.query_sql)

        assert candidate.primary_result_column in aliases, (
            "Declared primary result column is not returned by the query. "
            f"Candidate={candidate.code!r}, "
            f"declared_column={candidate.primary_result_column!r}, "
            f"SQL_aliases={sorted(aliases)!r}, "
            f"query_sql={candidate.query_sql!r}"
        )
