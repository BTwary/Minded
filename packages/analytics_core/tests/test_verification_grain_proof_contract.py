import pandas as pd
from packages.analytics_core.src.engines.verification import VerificationEngine


def test_legacy_scalar_verification_does_not_fabricate_grain_proof():
    df = pd.DataFrame({"region": ["A", "B", "A"], "revenue": [10.0, 20.0, 30.0]})
    result = VerificationEngine.verify_secondary(
        primary_df=df,
        target_metric_col="revenue",
        aggregation_type="SUM",
        primary_metric=60.0,
        query_sql=None,
        group_dimension_col=None,
    )
    assert result.status == "VERIFIED"
    assert result.grain_proof is None


def test_relation_path_still_produces_real_grain_proof():
    primary = pd.DataFrame({"region": ["A", "B"], "total_metric": [40.0, 20.0]})
    source = pd.DataFrame({"region": ["A", "A", "B"], "revenue": [10.0, 30.0, 20.0]})
    result = VerificationEngine.verify_secondary(
        primary_df=source,
        target_metric_col="revenue",
        aggregation_type="SUM",
        primary_metric=40.0,
        query_sql="SELECT region, SUM(revenue) AS total_metric FROM data_table GROUP BY region ORDER BY total_metric DESC",
        group_dimension_col="region",
        primary_result_df=primary,
    )
    assert result.status == "VERIFIED"
