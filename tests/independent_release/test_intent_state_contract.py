"""Regression tests for the stateful canonical-intent/clarification patch loop."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from packages.analytics_core.src.governance.guided_investigation import (
    AmbiguityStatus,
    ResolverStatus,
    apply_human_guidance_to_contract,
    compute_resolved_plan_hash,
    evaluate_guided_investigation,
)
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    canonical_json_dumps,
)
from packages.analytics_core.src.intelligence.intent_state import (
    _relative_time_range,
    filter_condition_mask,
    normalize_guidance_payload,
)


def test_empty_or_acknowledgement_only_guidance_does_not_change_intent():
    df = pd.DataFrame({"revenue": [10.0, 20.0], "region": ["East", "West"]})
    base = apply_human_guidance_to_contract("Rank revenue by region", df, {})
    empty_ack = apply_human_guidance_to_contract(
        "Rank revenue by region", df, {"confirmed_assumptions": []}
    )
    reviewed_ack = apply_human_guidance_to_contract(
        "Rank revenue by region", df, {"confirmed_assumptions": ["Group by region"]}
    )
    assert base.status == empty_ack.status == reviewed_ack.status == ResolverStatus.RESOLVED
    assert base.contract.contract_hash() == empty_ack.contract.contract_hash()
    assert base.contract.contract_hash() == reviewed_ack.contract.contract_hash()
    assert "confirmed_assumptions" not in normalize_guidance_payload({"confirmed_assumptions": []})


def test_preview_never_uses_heuristic_identifier_ranking_as_clear_plan():
    df = pd.DataFrame({"order_id": ["o1", "o2"], "customer_id": ["c1", "c2"]})
    preview = evaluate_guided_investigation("Rank order_id by customer_id", df=df)
    assert preview.ambiguity_status != AmbiguityStatus.CLEAR
    assert preview.can_auto_proceed is False
    assert preview.target_metric is None
    assert preview.group_dimension is None
    assert preview.planned_calculation.startswith("Not executable")
    assert "COUNT(*)" not in preview.planned_calculation
    assert "target_column" in preview.canonical_contract.unresolved_roles
    assert "grouping_columns" in preview.canonical_contract.unresolved_roles


def test_refund_rate_binds_the_requested_outcome_not_an_unrelated_flag():
    df = pd.DataFrame({
        "refund_flag": [1, 0, 1, 0],
        "active_flag": [1, 1, 0, 1],
        "payment_method": ["card", "cash", "card", "cash"],
    })
    preview = evaluate_guided_investigation("Which payment method has the highest refund rate?", df=df)
    assert preview.ambiguity_status == AmbiguityStatus.CLEAR
    assert preview.can_auto_proceed is True
    assert preview.canonical_contract.target_column == "refund_flag"
    assert preview.canonical_contract.grouping_columns == ("payment_method",)
    assert preview.canonical_contract.requested_aggregation == "RATE"
    assert '"refund_flag"' in preview.planned_calculation
    assert '"active_flag"' not in preview.planned_calculation


def test_temporal_scope_requires_time_column_when_multiple_dates_are_plausible():
    df = pd.DataFrame({
        "resolution_time": [2.0, 3.0],  # duration, not a calendar field
        "severity": ["high", "low"],
        "created_at": ["2026-10-13 09:00:00", "2026-10-14 17:00:00"],
        "resolved_at": ["2026-10-13 10:00:00", "2026-10-14 18:00:00"],
    })
    question = "Compare mean resolution time by severity for tickets opened in the last 90 days"
    preview = evaluate_guided_investigation(question, df=df)
    assert preview.ambiguity_status == AmbiguityStatus.AMBIGUOUS_SCOPE
    assert preview.can_auto_proceed is False
    assert len(preview.clarification_option_patches) == 2
    assert "resolution_time" not in [item["patch"]["set"].get("time_column") for item in preview.clarification_option_patches]

    patch = preview.clarification_option_patches[0]["patch"]
    resolved = apply_human_guidance_to_contract(question, df, {"intent_patch": patch})
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract.time_column == "created_at"
    bounds = [c for c in resolved.contract.filter_conditions if c["column"] == "created_at"]
    assert {c["operator"] for c in bounds} == {">=", "<"}
    assert len(bounds) == 2


def test_population_clarification_patch_adds_filter_without_dropping_other_slots():
    df = pd.DataFrame({
        "revenue": [10.0, 20.0, 30.0],
        "region": ["East", "West", "East"],
        "active_flag": [1, 0, 1],
    })
    question = "Calculate median revenue by region for active customers only"
    preview = evaluate_guided_investigation(question, df=df)
    assert preview.can_auto_proceed is False
    assert preview.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert preview.planned_calculation.startswith("Not executable")
    assert "COUNT(*)" not in preview.planned_calculation
    patch = next(item["patch"] for item in preview.clarification_option_patches if "active_flag" in item["label"])
    resolved = apply_human_guidance_to_contract(question, df, {"intent_patch": patch})
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract.target_column == "revenue"
    assert resolved.contract.grouping_columns == ("region",)
    assert resolved.contract.requested_aggregation == "MEDIAN"
    assert {"column": "active_flag", "operator": "=", "value": 1} in list(resolved.contract.filter_conditions)


def test_population_filter_polarity_must_match_the_requested_population():
    df = pd.DataFrame({
        "revenue": [10.0, 20.0], "region": ["East", "West"], "active_flag": [1, 0]
    })
    question = "Calculate median revenue by region for active customers only"
    initial = apply_human_guidance_to_contract(question, df, {})
    wrong_filter = replace(
        initial.contract,
        filter_conditions=({"column": "active_flag", "operator": "=", "value": 0},),
    )
    checked = apply_human_guidance_to_contract(question, df, {}, previous_contract=wrong_filter)
    assert checked.status == ResolverStatus.NEEDS_CLARIFICATION
    assert any(item.get("reason") == "population_qualifier_not_preserved" for item in checked.contract.intent_ledger)


def test_per_customer_metric_does_not_execute_without_a_real_denominator_plan():
    df = pd.DataFrame({
        "revenue": [10.0, 20.0], "region": ["East", "West"],
        "active_flag": [1, 0], "created_at": ["2026-01-01", "2026-02-01"],
    })
    preview = evaluate_guided_investigation(
        "How much revenue per active customer did each region generate last quarter?", df=df
    )
    assert preview.can_auto_proceed is False
    assert preview.ambiguity_status != AmbiguityStatus.CLEAR
    assert any(item.get("slot") == "denominator_column" for item in preview.canonical_contract.intent_ledger)


def test_compound_question_cannot_be_approved_as_its_first_parsed_clause_only():
    df = pd.DataFrame({
        "revenue": [10.0, 20.0], "profit": [1.0, 4.0], "region": ["East", "West"]
    })
    question = "Rank revenue by region and compare profit by region"
    preview = evaluate_guided_investigation(question, df=df)
    resolution = apply_human_guidance_to_contract(question, df, {})
    assert preview.can_auto_proceed is False
    assert preview.planned_calculation.startswith("Not executable")
    assert "Split it into one analytical question per operation" in preview.clarification_prompt
    assert resolution.status == ResolverStatus.UNSUPPORTED
    assert "will not execute only the easiest-to-parse clause" in resolution.reason


def test_canonical_json_rejects_unknown_objects_instead_of_stringifying():
    class Unknown:
        def __str__(self):
            return "ambiguous-id"

    with pytest.raises(TypeError):
        canonical_json_dumps({"value": Unknown()})
    with pytest.raises(ValueError):
        canonical_json_dumps({"value": float("nan")})


def test_resolved_plan_hash_ignores_provenance_but_binds_material_filter():
    df = pd.DataFrame({"revenue": [10.0, 20.0], "region": ["East", "West"]})
    result = apply_human_guidance_to_contract("Rank revenue by region", df, {})
    contract = result.contract
    h1 = compute_resolved_plan_hash(contract, "dataset-version-hash")
    provenance_only = replace(
        contract,
        slot_provenance={**contract.slot_provenance, "target_column": "CLARIFIED"},
        slot_confidence={**contract.slot_confidence, "target_column": 0.99},
    )
    assert compute_resolved_plan_hash(provenance_only, "dataset-version-hash") == h1
    changed = replace(
        contract,
        filter_conditions=({"column": "region", "operator": "=", "value": "East"},),
    )
    assert compute_resolved_plan_hash(changed, "dataset-version-hash") != h1


def test_filter_comparison_is_typed_and_fails_closed():
    numbers = pd.Series([1, 2, 3])
    assert filter_condition_mask(numbers, ">=", 2).tolist() == [False, True, True]
    dates = pd.Series(pd.to_datetime(["2026-10-13", "2026-10-14", "2026-10-15"], utc=True))
    cutoff = "2026-10-15"
    assert filter_condition_mask(dates, "<", cutoff).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        filter_condition_mask(pd.Series(["a", "b"]), ">", "a")


def test_filter_membership_preserves_numeric_types_and_null_semantics():
    values = pd.Series([1.0, 2.0, None, 3.0])
    assert filter_condition_mask(values, "IN", [1, 3]).tolist() == [True, False, False, True]
    assert filter_condition_mask(values, "NOT IN", [1, 3]).tolist() == [False, True, False, False]
    assert filter_condition_mask(values, "IS NULL").tolist() == [False, False, True, False]


def test_relative_quarter_boundary_is_end_exclusive():
    result = _relative_time_range("last quarter")
    assert result is not None
    assert result["end_exclusive"] is True
    assert result["start"] < result["end"]


def test_freeform_answer_is_compiled_as_patch_for_pending_time_column():
    df = pd.DataFrame({
        "resolution_time": [2.0, 3.0], "severity": ["high", "low"],
        "created_at": ["2026-10-13 09:00:00", "2026-10-14 17:00:00"],
        "resolved_at": ["2026-10-13 10:00:00", "2026-10-14 18:00:00"],
    })
    question = "Compare mean resolution time by severity for tickets opened in the last 90 days"
    outcome = apply_human_guidance_to_contract(
        question, df, {"pending_intent_slot": "time_column", "custom_instruction": "Use created_at as the time field"}
    )
    assert outcome.status == ResolverStatus.RESOLVED
    assert outcome.contract.time_column == "created_at"
    assert outcome.contract.slot_provenance["time_column"] == "CLARIFIED"
    assert outcome.contract.slot_confidence["time_column"] == 1.0
    assert any(item["column"] == "created_at" and item["operator"] == "<" for item in outcome.contract.filter_conditions)


def test_freeform_answer_is_compiled_as_filter_patch_for_pending_population_slot():
    df = pd.DataFrame({"revenue": [10.0, 20.0], "region": ["East", "West"], "active_flag": [1, 0]})
    question = "Calculate median revenue by region for active customers only"
    outcome = apply_human_guidance_to_contract(
        question, df, {"pending_intent_slot": "filter_conditions", "custom_instruction": "Only active customers"}
    )
    assert outcome.status == ResolverStatus.RESOLVED
    assert {"column": "active_flag", "operator": "=", "value": 1} in list(outcome.contract.filter_conditions)


def test_freeform_answer_does_not_invent_column_mapping_when_ambiguous():
    df = pd.DataFrame({"created_at": ["2026-01-01"], "resolved_at": ["2026-01-02"]})
    outcome = apply_human_guidance_to_contract(
        "Use a date window", df,
        {"pending_intent_slot": "time_column", "custom_instruction": "Use the date"},
    )
    assert outcome.status == ResolverStatus.NEEDS_CLARIFICATION
    assert "Could not identify one unique" in (outcome.reason or "")


def test_rank_with_last_quarter_binds_temporal_range_to_dataset_and_contract():
    df = pd.DataFrame({
        "revenue": [10.0, 20.0],
        "region": ["East", "West"],
        "created_at": ["2025-09-15", "2026-10-01"],
    })
    preview = evaluate_guided_investigation("Rank revenue by region for last quarter", df=df)
    contract = preview.canonical_contract
    assert preview.can_auto_proceed is True
    assert contract.task_family == "RANKING"
    assert contract.target_column == "revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.time_column == "created_at"
    assert contract.time_range["source"] == "dataset_temporal_resolver"
    assert contract.time_range["end_exclusive"] is True
    assert {("created_at", ">=", contract.time_range["start"]),
            ("created_at", "<", contract.time_range["end"])} <= {
        (item["column"], item["operator"], item["value"])
        for item in contract.filter_conditions
    }


def test_all_data_clarification_removes_prior_time_bounds_but_keeps_other_filters():
    df = pd.DataFrame({
        "revenue": [10.0], "region": ["East"], "created_at": ["2026-01-01"], "active_flag": [1]
    })
    question = "Rank revenue by region for last quarter"
    # The initial question produces date bounds; the clarification explicitly overrides that scope.
    initial = apply_human_guidance_to_contract(question, df, {})
    assert any(item["column"] == "created_at" for item in initial.contract.filter_conditions)
    patched = apply_human_guidance_to_contract(
        question, df, {"intent_patch": {"set": {"time_range": {"type": "all", "expression": "all data"}}}},
        previous_contract=initial.contract,
    )
    assert patched.status == ResolverStatus.RESOLVED
    assert patched.contract.time_range["type"] == "all"
    assert not any(item["column"] == "created_at" and item["operator"] in {">", ">=", "<", "<="} for item in patched.contract.filter_conditions)
