"""Regression test: the non-additive branch of the concentration experiment
must declare a primary_result_column that actually matches the alias used
in its own generated SQL.

Bug (found 2026-09-12): the non-additive branch reused the additive branch's
shared `metric_agg_expr` (always aliased ``AS total_metric``) while declaring
``primary_result_column="mean_metric"`` -- a column name that never appeared
in the query result. Any non-additive metric (MEAN/RATE/RATIO/etc.) routed
through the root-cause concentration experiment would therefore raise
``ValueError: Declared primary result column 'mean_metric' was not returned
by the query.`` in execution_provider.py at runtime.
"""
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType


def test_concentration_non_additive_declared_column_matches_sql_alias():
    metric_def = MetricDefinition(
        name="unit_cost", table_name="data_table", source_columns=["unit_cost"],
        semantic_type="mean_measure", aggregation_type=AggregationType.MEAN,
        grain="row_level", is_additive=False,
        valid_aggregations=[AggregationType.MEAN], rationale="test",
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
    hyp = PredictiveHypothesis(
        id="h1", hypothesis_code="HYP-01", claim="test claim", mechanism="test",
        predicted_observables_if_true=[], predicted_observables_if_false=[],
        falsification_criteria="", required_assumptions=[],
        prior_probability=0.5, posterior_probability=0.5,
        target_metric="unit_cost", target_dimension="category",
    )
    candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
        hypotheses=[hyp], semantic=semantic,
    )
    conc = [c for c in candidates if c.code.startswith("EXP-CONC")]
    assert conc, "no concentration candidate generated"
    c = conc[0]
    assert f"AS {c.primary_result_column}" in c.query_sql, (
        f"declared primary_result_column={c.primary_result_column!r} does not "
        f"match any alias in query_sql={c.query_sql!r}"
    )
