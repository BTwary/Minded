import pandas as pd

from packages.analytics_core.src.profiling.profiler import DataProfiler


def test_numeric_profile_contains_instant_analytics():
    df = pd.DataFrame({"income": [10, 20, 20, 30, 40]})
    profile = DataProfiler().profile_dataframe(df, "sample")
    col = profile.columns[0]

    assert col.name == "income"
    assert col.mean_value == 24.0
    assert col.mode_value == 20.0
    assert col.mode_frequency == 2
    assert col.variance == 104.0
    assert col.min_value == 10.0
    assert col.max_value == 40.0


def test_numeric_profile_reports_no_mode_when_all_values_are_unique():
    import pandas as pd
    from packages.analytics_core.src.profiling.profiler import DataProfiler

    profile = DataProfiler().profile_dataframe(pd.DataFrame({"income": [10, 20, 30]}), "sample")
    col = profile.columns[0]
    assert col.mode_value is None
    assert col.mode_frequency is None


def test_instant_profile_uses_entire_observed_column_not_profile_sample_size():
    df = pd.DataFrame({"income": list(range(1, 101))})
    profile = DataProfiler(sample_size=5).profile_dataframe(df, "sample")
    col = profile.columns[0]

    # The profile is a claim about the ingested dataset version, so the configured
    # profiler sample_size must not silently turn these first-pass statistics into
    # a sample calculation.
    assert profile.row_count == 100
    assert col.mean_value == 50.5
    assert col.min_value == 1.0
    assert col.max_value == 100.0
