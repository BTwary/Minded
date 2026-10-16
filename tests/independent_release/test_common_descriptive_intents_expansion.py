"""Regression coverage for common descriptive questions added to the typed intent registry."""
import pandas as pd
import pytest

from packages.analytics_core.src.intelligence.analytical_operations import aggregate_frame
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract


def test_mode_handles_categorical_values_and_deterministic_ties():
    frame = pd.DataFrame({"payment_method": ["card", "cash", "card", "cash", None]})
    result = aggregate_frame(frame, operation="MODE", target_column="payment_method")
    # A tie is resolved deterministically by the documented lexical/type key.
    assert result["value"] == "card"
    assert result["verification"]["status"] == "verified"
    assert aggregate_frame(pd.DataFrame({"x": ["b", "a", "b"]}), operation="MODE", target_column="x")["value"] == "b"


def test_range_quartiles_iqr_and_missing_count_use_explicit_semantics():
    frame = pd.DataFrame({"age": [1.0, 2.0, 3.0, 4.0, None]})
    assert aggregate_frame(frame, operation="RANGE", target_column="age")["value"] == 3.0
    assert aggregate_frame(frame, operation="Q1", target_column="age")["value"] == pytest.approx(1.75)
    assert aggregate_frame(frame, operation="Q3", target_column="age")["value"] == pytest.approx(3.25)
    assert aggregate_frame(frame, operation="IQR", target_column="age")["value"] == pytest.approx(1.5)
    assert aggregate_frame(frame, operation="MISSING_COUNT", target_column="age")["value"] == 1


def test_new_operations_fail_closed_on_wrong_column_type():
    frame = pd.DataFrame({"category": ["a", "b"]})
    with pytest.raises(ValueError, match="requires a numeric column"):
        aggregate_frame(frame, operation="RANGE", target_column="category")


def test_natural_language_compiler_preserves_common_new_operation_names():
    frame = pd.DataFrame({
        "payment_method": ["card", "cash", "card"],
        "age": [20.0, 30.0, None],
    })
    cases = [
        ("What is the mode of payment method?", "MODE"),
        ("What is the most common payment method?", "MODE"),
        ("Which payment method occurs most often?", "MODE"),
        ("How many age values are missing?", "MISSING_COUNT"),
        ("What is the 25th percentile of age?", "Q1"),
        ("What is the 75th percentile of age?", "Q3"),
        ("What is the range of age?", "RANGE"),
        ("Calculate the first quartile of age", "Q1"),
        ("Calculate the third quartile of age", "Q3"),
        ("Calculate the interquartile range of age", "IQR"),
        ("Count missing values in age", "MISSING_COUNT"),
    ]
    for question, expected in cases:
        contract = compile_canonical_question_contract(question, frame)
        assert contract.requested_aggregation == expected, (question, contract.to_dict())


def test_new_intents_execute_through_approved_contract_path():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result

    frame = pd.DataFrame({"payment_method": ["card", "cash", "card"], "age": [20.0, 30.0, None]})
    cases = [
        ("What is the mode of payment method?", "MODE", "card"),
        ("What is the range of age?", "RANGE", 10.0),
        ("Calculate the first quartile of age", "Q1", 22.5),
        ("Calculate the third quartile of age", "Q3", 27.5),
        ("Calculate the interquartile range of age", "IQR", 5.0),
        ("Count missing values in age", "MISSING_COUNT", 1),
    ]
    for question, expected_operation, expected_value in cases:
        contract = compile_canonical_question_contract(question, frame)
        answer = build_analyst_result(
            question, frame, target=contract.target_column,
            group=(contract.grouping_columns[0] if contract.grouping_columns else None),
            approved_contract=contract,
        )
        assert answer is not None, (question, contract.to_dict())
        assert answer.numbers["operation"] == expected_operation
        assert answer.numbers["value"] == pytest.approx(expected_value) if isinstance(expected_value, (int, float)) else answer.numbers["value"] == expected_value
        assert answer.numbers["verification"]["status"] == "verified"


def test_new_descriptive_intent_preserves_group_dimension():
    frame = pd.DataFrame({"region": ["East", "West", "East"], "age": [20.0, 30.0, 40.0]})
    contract = compile_canonical_question_contract("Calculate the interquartile range of age by region", frame)
    assert contract.requested_aggregation == "IQR"
    assert contract.target_column == "age"
    assert contract.grouping_columns == ("region",)


def test_common_data_quality_and_extreme_value_questions_compile_and_execute():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result

    frame = pd.DataFrame({
        "customer_id": [1, 1, 2, 3],
        "age": [20.0, 30.0, None, 50.0],
        "region": ["East", "West", "East", "West"],
    })
    cases = [
        ("What is the minimum value of age?", "MIN", 20.0),
        ("What is the lowest age?", "MIN", 20.0),
        ("What is the maximum value of age?", "MAX", 50.0),
        ("What is the highest age?", "MAX", 50.0),
        ("How many unique customer_id?", "COUNT_DISTINCT", 3),
        ("What is the percentage of missing values in age?", "MISSING_PERCENT", 25.0),
    ]
    for question, operation, expected in cases:
        contract = compile_canonical_question_contract(question, frame)
        assert contract.completeness == "COMPLETE", (question, contract.to_dict())
        assert contract.requested_aggregation == operation, (question, contract.to_dict())
        answer = build_analyst_result(
            question, frame, target=contract.target_column,
            group=(contract.grouping_columns[0] if contract.grouping_columns else None),
            approved_contract=contract,
        )
        assert answer is not None, (question, contract.to_dict())
        assert answer.numbers["operation"] == operation
        assert answer.numbers["value"] == pytest.approx(expected)
        assert answer.numbers["verification"]["status"] == "verified"


def test_missing_percentage_uses_the_filtered_group_population_and_handles_all_missing():
    frame = pd.DataFrame({"region": ["East", "East", "West", "West"], "age": [None, 20, None, 40]})
    result = aggregate_frame(frame, operation="MISSING_PERCENT", target_column="age", grouping_columns=("region",))
    actual = {r["group"]["region"]: r["value"] for r in result["records"]}
    assert actual == {"East": 50.0, "West": 50.0}
    all_missing = aggregate_frame(pd.DataFrame({"x": [None, None]}), operation="MISSING_PERCENT", target_column="x")
    assert all_missing["value"] == 100.0


def test_duplicate_row_questions_use_explicit_whole_row_definition():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result

    frame = pd.DataFrame({
        "order_id": [1, 1, 2, 3, 3],
        "region": ["East", "East", "West", "North", "North"],
        "amount": [10, 10, 20, 30, 30],
    })
    for question in ("How many duplicate rows are there?", "Count repeated records"):
        contract = compile_canonical_question_contract(question, frame)
        assert contract.completeness == "COMPLETE", (question, contract.to_dict())
        assert contract.requested_aggregation == "DUPLICATE_ROW_COUNT"
        answer = build_analyst_result(question, frame, target=None, group=None, approved_contract=contract)
        assert answer is not None
        assert answer.numbers["operation"] == "DUPLICATE_ROW_COUNT"
        assert answer.numbers["value"] == 2
        assert answer.numbers["verification"]["status"] == "verified"


def test_duplicate_row_count_rejects_an_unneeded_target_column():
    with pytest.raises(ValueError, match="does not accept a target column"):
        aggregate_frame(pd.DataFrame({"x": [1, 1]}), operation="DUPLICATE_ROW_COUNT", target_column="x")


def test_common_grouped_business_aggregates_accept_for_each_wording_and_reject_dropped_scope():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result

    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West"],
        "revenue": [10.0, 30.0, 20.0, 60.0],
        "age": [20.0, 40.0, 30.0, 50.0],
    })
    cases = [
        ("What is the total revenue by region?", "SUM", {"East": 40.0, "West": 80.0}),
        ("What is the total revenue for each region?", "SUM", {"East": 40.0, "West": 80.0}),
        ("Calculate the average age for each region", "MEAN", {"East": 30.0, "West": 40.0}),
        ("Show the median age by region", "MEDIAN", {"East": 30.0, "West": 40.0}),
    ]
    for question, operation, expected in cases:
        contract = compile_canonical_question_contract(question, frame)
        assert contract.completeness == "COMPLETE", (question, contract.to_dict())
        assert contract.target_column in frame.columns, (question, contract.to_dict())
        assert contract.grouping_columns == ("region",), (question, contract.to_dict())
        answer = build_analyst_result(
            question, frame, target=contract.target_column, group="region", approved_contract=contract
        )
        assert answer is not None, (question, contract.to_dict())
        assert answer.numbers["operation"] == operation
        table = answer.numbers["table"]
        actual = {key: row["value"] for key, row in table.items()}
        assert actual == expected
        assert answer.numbers["verification"]["status"] == "verified"

    # Do not silently drop a trailing time/filter condition when resolving the group phrase.
    unsafe = compile_canonical_question_contract(
        "What is the total revenue for each region in 2024?", frame
    )
    assert unsafe.completeness != "COMPLETE", unsafe.to_dict()

@pytest.mark.parametrize("operation,expected", [
    ("ABS_SUM", 11.0),
    ("POSITIVE_COUNT", 2),
    ("NEGATIVE_COUNT", 2),
    ("ZERO_COUNT", 1),
    ("NONZERO_COUNT", 4),
    ("MAD", 2.0),
])
def test_extended_numeric_distribution_aggregations(operation, expected):
    frame = pd.DataFrame({"delta": [-4.0, -1.0, 0.0, 2.0, 4.0, None]})
    result = aggregate_frame(frame, operation=operation, target_column="delta")
    assert result["value"] == pytest.approx(expected)
    assert result["verification"]["status"] == "verified"


def test_shape_statistics_and_coefficient_of_variation_have_defined_small_sample_behavior():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert aggregate_frame(frame, operation="SKEWNESS", target_column="x")["value"] == pytest.approx(0.0)
    assert aggregate_frame(frame, operation="KURTOSIS", target_column="x")["value"] == pytest.approx(-1.2)
    cv = aggregate_frame(frame, operation="COEFFICIENT_OF_VARIATION", target_column="x")["value"]
    assert cv == pytest.approx(frame.x.std(ddof=1) / abs(frame.x.mean()))
    one = pd.DataFrame({"x": [5.0]})
    assert aggregate_frame(one, operation="SKEWNESS", target_column="x")["value"] is None
    assert aggregate_frame(one, operation="KURTOSIS", target_column="x")["value"] is None


def test_coefficient_of_variation_returns_undefined_for_zero_mean():
    frame = pd.DataFrame({"x": [-1.0, 1.0]})
    assert aggregate_frame(frame, operation="COEFFICIENT_OF_VARIATION", target_column="x")["value"] is None
