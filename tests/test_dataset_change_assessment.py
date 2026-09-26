import pandas as pd

from packages.analytics_core.src.data.dataset_change_assessment import assess_dataset_change


def test_relevant_material_metric_change_triggers_review():
    before = pd.DataFrame({"id": [1, 2, 3, 4], "date": ["2026-01-01"] * 4, "revenue": [100, 100, 100, 100]})
    after = pd.DataFrame({"id": [1, 2, 3, 4, 5], "date": ["2026-02-01"] * 5, "revenue": [100, 100, 100, 100, 180]})

    result = assess_dataset_change(
        before,
        after,
        before_version=1,
        after_version=2,
        question_columns=["revenue", "date"],
        time_column="date",
        key_columns=["id"],
        task="DIAGNOSTIC",
    )

    assert result.material_change is True
    assert result.question_relevant is True
    assert result.needs_investigation is True
    assert result.status == "RELEVANT_MATERIAL_CHANGE"
    assert "revenue" in result.question_relevant_columns
    assert result.time_coverage_change is not None


def test_material_change_in_unrelated_column_is_not_question_relevant():
    before = pd.DataFrame({"id": [1, 2, 3, 4], "revenue": [100, 100, 100, 100], "note": ["a"] * 4})
    after = pd.DataFrame({"id": [1, 2, 3, 4], "revenue": [100, 100, 100, 100], "note": ["a", "b", "c", "d"]})

    result = assess_dataset_change(
        before,
        after,
        before_version=1,
        after_version=2,
        question_columns=["revenue"],
        key_columns=["id"],
        task="DESCRIPTIVE",
    )

    assert result.material_change is True
    assert result.question_relevant is False
    assert result.needs_investigation is False
    assert result.status == "MATERIAL_CHANGE_NOT_DIRECTLY_RELEVANT"


def test_small_row_count_change_does_not_trigger_materiality_by_itself():
    before = pd.DataFrame({"id": list(range(100)), "revenue": [100] * 100})
    after = pd.DataFrame({"id": list(range(101)), "revenue": [100] * 101})

    result = assess_dataset_change(
        before,
        after,
        before_version=1,
        after_version=2,
        question_columns=["revenue"],
        key_columns=["id"],
        task="DESCRIPTIVE",
    )

    assert result.material_change is False
    assert result.status == "NO_MATERIAL_CHANGE"


def test_same_logical_rows_with_reorder_are_not_a_material_version_change():
    before = pd.DataFrame({"id": [1, 2, 3], "revenue": [100, 200, 300]})
    after = pd.DataFrame({"id": [3, 1, 2], "revenue": [300, 100, 200]})

    result = assess_dataset_change(
        before,
        after,
        before_version=1,
        after_version=2,
        question_columns=["revenue"],
        key_columns=["id"],
        task="DESCRIPTIVE",
    )

    assert result.material_change is False
    assert result.needs_investigation is False
