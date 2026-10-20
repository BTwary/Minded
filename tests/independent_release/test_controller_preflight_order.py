"""Regression guard: locked-contract dispatch must not bypass variable preflight.

The controller supports an approved-contract fast path. Keep the fail-closed
question/schema check before that path so an explicit nonexistent column can
never be turned into a different, apparently successful analysis.
"""
from pathlib import Path

from packages.analytics_core.src.engines.variable_resolution_gate import (
    detect_unresolved_requested_variables,
)


def test_missing_snake_case_variable_is_detected_without_fallback():
    columns = {"revenue", "region", "date"}
    assert detect_unresolved_requested_variables(
        "Is revenue correlated with marketing_spend?", columns
    ) == ["marketing_spend"]


def test_controller_preflight_runs_before_locked_contract_fast_path():
    controller_path = (
        Path(__file__).resolve().parents[2]
        / "packages/analytics_core/src/runtime/controller.py"
    )
    source = controller_path.read_text(encoding="utf-8")
    preflight = source.index(
        "_missing_requested_vars = detect_unresolved_requested_variables(question, _all_available_columns)"
    )
    locked_dispatch = source.index(
        "# Approved plans dispatch once to an existing engine."
    )
    assert preflight < locked_dispatch, (
        "The locked-contract fast path bypasses missing-variable preflight; "
        "an unresolved requested column could be replaced by a fallback analysis."
    )


def test_locked_contract_applies_human_corrections_before_dispatch():
    controller_path = (
        Path(__file__).resolve().parents[2]
        / "packages/analytics_core/src/runtime/controller.py"
    )
    source = controller_path.read_text(encoding="utf-8")
    amendment = source.index("event_type=\"investigation.canonical_contract_amended\"")
    dispatch = source.index("# Approved plans dispatch once to an existing engine.")
    assert amendment < dispatch, (
        "Human-selected metric/aggregation/group corrections must amend the effective "
        "contract before the approved execution path is entered."
    )
    assert "_requested_scope_filter = _corrections.get(\"scope_filter\")" in source[:dispatch]
    assert "same_scope_filter(" in source[:dispatch], (
        "An approved scope filter must be distinguished from a newly introduced correction."
    )


def test_approved_scope_filter_is_not_mistaken_for_a_post_lock_correction():
    from packages.analytics_core.src.runtime.controller_scope import same_scope_filter

    approved = "\"customer_segment\" = 'Small Business' AND quarter = 'Q3' AND delivery_completed_flag = 1"
    same_with_formatting = "\"customer_segment\"   = 'Small Business' and quarter = 'Q3' AND delivery_completed_flag = 1"
    assert same_scope_filter(same_with_formatting, approved)


def test_new_scope_filter_after_approval_still_requires_reapproval():
    from packages.analytics_core.src.runtime.controller_scope import same_scope_filter

    approved = "region = 'West'"
    changed = "region = 'East'"
    assert not same_scope_filter(changed, approved)
    assert not same_scope_filter(changed, None)
