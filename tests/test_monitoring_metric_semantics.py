import pandas as pd

from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver


def test_monitoring_policy_does_not_sum_rate_metrics():
    # The continuous monitor now delegates metric meaning to the same canonical
    # semantics resolver used by investigations. A conversion_rate must never
    # be silently summed across periods.
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"],
        "conversion_rate": [0.10, 0.20, 0.30, 0.40],
    })
    metric = MetricSemanticsResolver.resolve("conversion_rate", df, table_name="sales")
    assert metric.aggregation_type.value in {"rate", "mean", "weighted_mean", "ratio", "proportion"}
    assert metric.aggregation_type.value != "SUM"


def test_additive_sales_metric_remains_sum_semantics():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-02-01"],
        "sales": [100.0, 120.0],
    })
    metric = MetricSemanticsResolver.resolve("sales", df, table_name="sales")
    assert metric.aggregation_type.value == "sum"
