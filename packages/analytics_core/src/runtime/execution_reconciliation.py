"""Lightweight contract-to-execution reconciliation, independent of SQL engine imports."""
from typing import Any

def reconcile_approved_execution(contract, binding: Any, result: Any, *,
                                  dataset_name: str, expected_fingerprints: dict,
                                  expected_versions: dict, expected_inputs: dict) -> dict[str, Any]:
    """Reconcile the approved contract with the physical plan and execution receipt.

    This is intentionally fail-closed. A successful SQL response is not sufficient:
    metric, aggregation, grouping, effective filters, denominator semantics, dataset
    identity/version, and the exact executed SQL must all agree.
    """
    spec = binding.specification
    expected_filters = [dict(item) for item in contract.filter_conditions]
    time_range = contract.time_range
    if time_range and time_range.get("type") != "all":
        expected_filters.extend([
            {"column": contract.time_column, "operator": ">=", "value": time_range["start"]},
            {"column": contract.time_column,
             "operator": "<" if time_range.get("end_exclusive", False) else "<=",
             "value": time_range["end"]},
        ])
    comparison_dimension_present = bool(contract.grouping_columns)
    if contract.comparison_values and comparison_dimension_present:
        expected_filters.append({"column": list(contract.grouping_columns)[0],
                                 "operator": "IN", "value": list(contract.comparison_values)})
    # Keep stable order while avoiding duplicate time/comparison predicates already present.
    deduped_filters = []
    for item in expected_filters:
        if item not in deduped_filters:
            deduped_filters.append(item)

    checks = {
        "metric_matches": spec.get("target_column") == contract.target_column,
        "aggregation_matches": spec.get("aggregation") == contract.requested_aggregation or
            (str(contract.requested_aggregation).upper() == "AVG" and spec.get("aggregation") == "MEAN"),
        "grouping_matches": list(spec.get("grouping_columns") or []) == list(contract.grouping_columns),
        "filters_match": list(spec.get("filters") or []) == deduped_filters,
        "denominator_semantics_supported": not contract.numerator_column and not contract.denominator_column,
        "share_denominator_matches": bool(spec.get("share_of_total")) == bool(contract.share_of_total),
        "time_range_matches": spec.get("time_range") == contract.time_range,
        # Comparison values without a grouping dimension cannot be expressed as a predicate: fail closed.
        "comparison_scope_matches": (list(spec.get("comparison_values") or []) == list(contract.comparison_values)
                                     and (comparison_dimension_present or not contract.comparison_values)),
        "requested_outputs_match": list(spec.get("requested_outputs") or []) == list(contract.requested_outputs),
        # The executor only records a direction for RANKING tasks; for other grouped tasks the
        # contract's direction still drives ORDER BY, which executed_sql_matches verifies exactly.
        "ranking_direction_matches": (spec.get("ranking_direction") == contract.ranking_direction
            or (spec.get("ranking_direction") is None
                and getattr(contract, "task_family", None) != "RANKING")),
        "dataset_identity_matches": getattr(result, "source_datasets", None) == [dataset_name],
        "dataset_fingerprint_matches": getattr(result, "source_dataset_fingerprints", None) == expected_fingerprints,
        "dataset_version_matches": getattr(result, "source_dataset_versions", None) == expected_versions,
        "input_fingerprint_matches": getattr(result, "input_content_fingerprints", None) == expected_inputs,
        "executed_sql_matches": getattr(result, "executed_query", None) == binding.plan.to_sql(),
    }
    return {"status": "MATCHED" if all(checks.values()) else "MISMATCH",
            "checks": checks, "failed_checks": [key for key, passed in checks.items() if not passed],
            "expected_dataset": dataset_name,
            "expected_dataset_versions": expected_versions,
            "expected_dataset_fingerprints": expected_fingerprints,
            "expected_specification": spec}

