import pandas as pd

from packages.analytics_core.src.data.dataset_diff import compare_datasets


def test_stable_key_diff_detects_added_removed_changed_rows_and_schema():
    before = pd.DataFrame({
        "id": [1, 2, 3],
        "region": ["A", "A", "B"],
        "revenue": [100.0, 200.0, 300.0],
    })
    after = pd.DataFrame({
        "id": [2, 3, 4],
        "region": ["A", "B", "C"],
        "revenue": [250.0, 300.0, 400.0],
        "channel": ["web", "store", "web"],
    })

    result = compare_datasets(before, after, key_columns=["id"])

    assert result.row_identity_mode == "STABLE_KEY"
    assert result.rows_added == 1
    assert result.rows_removed == 1
    assert result.rows_changed == 1
    assert result.schema_added == ["channel"]
    assert result.schema_removed == []
    assert "region" in result.categorical_changes


def test_without_key_uses_row_multiset_and_does_not_claim_updates():
    before = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    after = pd.DataFrame({"x": [3, 2, 4], "y": ["c", "b", "d"]})

    result = compare_datasets(before, after)

    assert result.row_identity_mode == "ROW_MULTISET"
    assert result.rows_added == 1
    assert result.rows_removed == 1
    assert result.rows_changed == 0
    assert any("No stable key" in note for note in result.notes)


def test_missingness_and_numeric_distribution_changes_are_reported():
    before = pd.DataFrame({"id": [1, 2, 3, 4], "value": [10, 10, 10, 10]})
    after = pd.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, None, 20]})

    result = compare_datasets(before, after, key_columns=["id"])

    assert result.missingness_delta["value"] > 0
    assert result.numeric_distribution_delta["value"]["mean_after"] > result.numeric_distribution_delta["value"]["mean_before"]



def test_stable_key_row_reorder_does_not_create_false_duplicates_or_changes():
    before = pd.DataFrame({
        "id": [1, 2, 3],
        "revenue": [100, 200, 300],
    })
    after = pd.DataFrame({
        "id": [3, 2, 1],
        "revenue": [300, 200, 100],
    })

    result = compare_datasets(before, after, key_columns=["id"])

    assert result.duplicate_key_counts == {}
    assert result.rows_added == 0
    assert result.rows_removed == 0
    assert result.rows_changed == 0


def test_true_duplicate_stable_keys_are_detected_by_key_multiplicity():
    before = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    after = pd.DataFrame({"id": [1, 1, 2], "value": [10, 11, 20]})

    result = compare_datasets(before, after, key_columns=["id"])

    assert result.duplicate_key_counts
    assert any("Duplicate stable keys detected" in note for note in result.notes)
    assert result.rows_added == 1
    assert result.rows_changed == 0


def test_composite_stable_key_remains_order_invariant():
    before = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "month": ["2026-01", "2026-02", "2026-01"],
        "value": [10, 20, 30],
    })
    after = pd.DataFrame({
        "customer_id": [2, 1, 1],
        "month": ["2026-01", "2026-02", "2026-01"],
        "value": [30, 20, 10],
    })

    result = compare_datasets(before, after, key_columns=["customer_id", "month"])

    assert result.duplicate_key_counts == {}
    assert result.rows_added == 0
    assert result.rows_removed == 0
    assert result.rows_changed == 0


def test_null_stable_keys_are_explicitly_excluded_from_identity_matching():
    before = pd.DataFrame({"id": [1, None], "value": [10, 20]})
    after = pd.DataFrame({"id": [1, None, None], "value": [10, 20, 30]})

    result = compare_datasets(before, after, key_columns=["id"])

    assert result.duplicate_key_counts == {}
    assert any("incomplete stable keys" in note for note in result.notes)


def test_stable_key_dtype_change_is_explicitly_reported():
    before = pd.DataFrame({"id": pd.Series([1, 2], dtype="int64"), "value": [10, 20]})
    after = pd.DataFrame({"id": pd.Series(["1", "2"], dtype="string"), "value": [10, 20]})

    result = compare_datasets(before, after, key_columns=["id"])

    assert "id" in result.schema_type_changed
    assert any("stable-key column dtypes changed" in note for note in result.notes)
