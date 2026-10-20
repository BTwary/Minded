"""Fixes that existed in only one of the two diverged 2026-10-17/18 branches, pinned in the merged tree.

Branch "merged_audited" (second + third pass) and branch "intent_first" each fixed things the other
lacked. These tests pin the pieces that came from "intent_first" (and the interactions with the
"merged_audited" pieces) so a later re-merge cannot silently drop either side.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
from packages.analytics_core.src.runtime.controller_scope import validate_scope_filter_expression

_spec = importlib.util.spec_from_file_location("_rr", pathlib.Path(__file__).parents[1] / "test_release_regressions.py")
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)
_rr._SCRIPT = _rr._SCRIPT.replace(
    'names = ["customers", "sales"]',
    'names = sorted(f[:-4] for f in os.listdir(tables_dir) if f.endswith(".csv"))',
)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(11)
    n = 1200
    return pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(n)],
        "customer_segment": rng.choice(["Consumer", "Small Business", "Corporate"], n),
        "sales_channel": rng.choice(["Website", "Mobile App", "Marketplace", "Retail Partner"], n),
        "region": rng.choice(["North", "South", "East", "West"], n, p=[.4, .3, .2, .1]),
        "order_status": rng.choice(["Delivered", "Returned", "Cancelled"], n, p=[.7, .15, .15]),
        "returned": rng.choice([0, 1], n, p=[.85, .15]),
        "gross_sales": np.round(rng.gamma(2.0, 150.0, n), 2),
        "discount_pct": np.round(rng.random(n) * .3, 3),
    })


def test_negated_lifecycle_word_is_not_bound_as_an_inclusion(df):
    c = compile_canonical_question_contract("What was the total gross sales for orders that were not delivered?", df)
    conds = [(f["column"], f["operator"], f["value"]) for f in c.filter_conditions]
    assert ("order_status", "!=", "Delivered") in conds
    assert ("order_status", "=", "Delivered") not in conds      # was a silent wrong answer


def test_zero_one_flag_is_bound_from_plain_words(df):
    c = compile_canonical_question_contract("Total gross sales for orders that were not returned", df)
    assert any(f["column"] == "returned" and f["operator"] == "=" and f["value"] == 0 for f in c.filter_conditions)


def test_compare_between_names_the_groups_not_a_date_range(df):
    c = compile_canonical_question_contract("Compare gross_sales between sales_channel", df)
    assert c.task_family == "GROUP_COMPARISON" and c.target_column == "gross_sales"
    assert list(c.grouping_columns) == ["sales_channel"] and "filter_instruction" not in c.unresolved_roles


def test_unmatched_population_word_asks_with_observed_values(df):
    q = "Among Enterprise customers, what is the total gross sales?"
    c = compile_canonical_question_contract(q, df)
    assert "filter_value_unmatched:Enterprise" in c.unresolved_roles
    plan = evaluate_guided_investigation(q, df=df)
    assert not plan.can_auto_proceed
    assert "Enterprise" in (plan.clarification_prompt or "")
    assert 0 < len(plan.clarification_options) <= 6
    assert all(o.startswith("customer_segment = ") for o in plan.clarification_options)


def test_generated_verb_ranks_a_bare_measure_and_keeps_qualified_phrases_out(df):
    c = compile_canonical_question_contract("Which sales_channel generated the highest gross_sales?", df)
    assert c.task_family == "RANKING" and c.target_column == "gross_sales" and c.ranking_direction == "DESC"


def test_count_ranking_still_compiles_with_the_branch_a_fix(df):
    c = compile_canonical_question_contract("Which region has the most orders?", df)
    assert c.completeness == "COMPLETE" and c.requested_aggregation == "COUNT_DISTINCT" and c.ranking_direction == "DESC"


def test_count_ranking_runs_end_to_end_and_is_independently_verified(df, tmp_path):
    d = tmp_path / "tables"
    d.mkdir()
    df.to_csv(d / "orders.csv", index=False)
    r = _rr.run((d, df), tmp_path, "Which region has the most orders?", "orders")
    assert r["status"] == "COMPLETED", r
    top = df.groupby("region").order_id.nunique().sort_values(ascending=False)
    assert top.index[0] in r["answer"][:60], (r["answer"], top)


@pytest.mark.parametrize("expr", ["region = 'North' OR 1=1", "region = 'North' OR TRUE", "1 = 1"])
def test_scope_filter_rejects_injected_constants(expr):
    with pytest.raises(ValueError):
        validate_scope_filter_expression(expr, ["region"])


def test_scope_filter_still_accepts_boolean_comparison_values():
    assert "is_active" in validate_scope_filter_expression("is_active = TRUE", ["is_active"])
