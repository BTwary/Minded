"""Review/investigation layer regressions: no silent loss of intent, direct answers for rankings."""
import tempfile
import uuid

import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.engines.analyst_answer import _execute_canonical_descriptive_contract
from packages.analytics_core.src.governance.guided_investigation import (
    AmbiguityStatus, apply_human_guidance_to_contract, evaluate_guided_investigation,
    render_contract_preview_from_contract,
)


def _sales(n=400, seed=5):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(n)],
        "region": r.choice(["North", "South", "East", "West"], n),
        "customer_segment": r.choice(["Corporate", "Consumer", "SMB"], n, p=[.5, .3, .2]),
        "revenue": r.uniform(10, 500, n).round(2),
    })


# ---- a predicate on a field the dataset lacks must never be dropped -------------------------
@pytest.mark.parametrize("q", [
    "Show total revenue for customers where loyalty_tier = Gold",
    "Show total revenue where loyaltyTier = Gold",
    "Average revenue by region where discount_rate > 0.1",
])
def test_filter_on_unavailable_column_is_not_silently_dropped(q):
    out = apply_human_guidance_to_contract(q, _sales(), {})
    assert out.status.name == "NEEDS_CLARIFICATION"
    assert out.contract.completeness == "INCOMPLETE"
    assert "not a column" in out.reason and "Nothing was calculated" in out.reason
    assert render_contract_preview_from_contract(out.contract) is None   # no executable-looking SQL


def test_filter_on_existing_column_still_resolves():
    out = apply_human_guidance_to_contract("Total revenue for region = North", _sales(), {})
    assert out.status.name == "RESOLVED" and out.contract.completeness == "COMPLETE"
    assert any(f["column"] == "region" for f in out.contract.filter_conditions)


def test_preview_says_why_for_unavailable_filter():
    plan = evaluate_guided_investigation("Show total revenue where loyalty_tier = Gold", df=_sales())
    assert not plan.can_auto_proceed and "loyalty_tier" in (plan.clarification_prompt or "")
    assert "SUM" not in plan.planned_calculation


# ---- ranking: direct answer first, evidence second ------------------------------------------
def _answer(q, df):
    out = apply_human_guidance_to_contract(q, df, {})
    assert out.status.name == "RESOLVED", out.reason
    return out.contract, _execute_canonical_descriptive_contract(out.contract, df)


@pytest.mark.parametrize("q,order,idx", [
    ("Which customer_segment has the highest total revenue?", "highest", 0),
    ("Which customer_segment has the lowest total revenue?", "lowest", -1),
    ("Highest total revenue by customer_segment?", "highest", 0),
])
def test_ranking_answer_leads_with_winner_and_verified_value(q, order, idx):
    df = _sales()
    totals = df.groupby("customer_segment").revenue.sum().sort_values(ascending=False)
    winner = totals.index[idx]
    _, ans = _answer(q, df)
    assert ans.headline.startswith(f"{winner}: ") and any(d.startswith(f"{winner} has the {order} total revenue") for d in ans.details)
    assert f"{totals[winner]:,.0f}" in ans.headline
    assert ans.numbers["top"] == winner and abs(ans.numbers["top_value"] - totals[winner]) < 1e-6
    assert ans.headline.count(";") == len(totals) - 1      # full ranked table still reported


def test_tied_leaders_are_reported_as_tied_not_arbitrarily_picked():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "v": [5, 5, 5, 5, 1], "id": range(5)})
    _, ans = _answer("Which g has the highest total v?", df)
    assert any("tied" in d for d in ans.details) and ans.numbers["leaders"] == ["a", "b"]


def test_plain_breakdown_is_not_turned_into_a_ranking_claim():
    _, ans = _answer("Show total revenue by region", _sales())
    assert not any(" has the " in d for d in ans.details)


def test_ranking_preview_orders_like_the_executor():
    out = apply_human_guidance_to_contract("Highest total revenue by customer_segment?", _sales(), {})
    assert "ORDER BY metric_value DESC" in render_contract_preview_from_contract(out.contract)
    out = apply_human_guidance_to_contract("Show total revenue by region", _sales(), {})
    assert "ORDER BY" not in render_contract_preview_from_contract(out.contract)


# ---- rephrasings reach the same contract ----------------------------------------------------
def test_paraphrases_of_a_ranking_question_share_one_contract():
    df = _sales()
    qs = ["Which customer_segment has the highest total revenue?", "Highest total revenue by customer_segment?",
          "Which customer_segment generated the highest total revenue?"]
    sig = set()
    for q in qs:
        c = apply_human_guidance_to_contract(q, df, {}).contract
        sig.add((c.target_column, c.requested_aggregation, c.grouping_columns, c.ranking_direction, tuple(map(str, c.filter_conditions))))
    assert len(sig) == 1


# ---- alias + ambiguity preserved (second-environment report items) ---------------------------
def test_refund_rate_binds_refund_flag_not_an_unrelated_flag():
    r = np.random.default_rng(1)
    df = pd.DataFrame({"payment_method": r.choice(["card", "cash"], 300), "refund_flag": r.integers(0, 2, 300),
                       "active_flag": r.integers(0, 2, 300), "amount": r.uniform(1, 9, 300)})
    c = apply_human_guidance_to_contract("Rank payment_method by refund rate", df, {}).contract
    assert c.target_column == "refund_flag"


def test_ambiguous_metric_keeps_its_specific_diagnosis():
    r = np.random.default_rng(2)
    df = pd.DataFrame({"category": r.choice(["x", "y"], 200), "gross_revenue": r.uniform(1, 9, 200),
                       "net_revenue": r.uniform(1, 9, 200), "id": range(200)})
    plan = evaluate_guided_investigation("What was our total revenue by category?", df=df)
    assert plan.ambiguity_status == AmbiguityStatus.AMBIGUOUS_METRIC and not plan.can_auto_proceed
    text = " ".join(plan.clarification_options) + (plan.clarification_prompt or "")
    assert "gross_revenue" in text and "net_revenue" in text


# ---- HTTP: /interpret must not show a heuristic calculation behind a refused question ---------
@pytest.fixture()
def api_env():
    from fastapi.testclient import TestClient
    from apps.api.src.main import app
    from apps.api.src.core.database import SessionLocal
    from apps.api.src.api.v1.auth import get_current_user
    from apps.api.src.models.entities import User, Project, Dataset, DatasetVersion
    db = SessionLocal()
    user = db.query(User).first()
    if not user:
        user = User(id=str(uuid.uuid4()), email="rl@minded.ai", full_name="RL", hashed_password="pw", is_active=True)
        db.add(user); db.commit(); db.refresh(user)
    project = Project(id=str(uuid.uuid4()), name="RL", owner_id=user.id)
    db.add(project); db.commit()
    ds = Dataset(id=str(uuid.uuid4()), project_id=project.id, name="rl_data", row_count=4, column_count=3)
    db.add(ds); db.commit()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("revenue,units,region\n100.0,2,North\n200.0,3,South\n300.0,4,North\n150.0,5,South\n")
        path = f.name
    db.add(DatasetVersion(id=str(uuid.uuid4()), dataset_id=ds.id, version_number=1, file_path=path,
                          row_count=4, column_count=3, content_hash="fp_rl_1"))
    db.commit()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        yield TestClient(app), project.id, ds.id
    finally:
        app.dependency_overrides.clear(); db.close()


def _interpret(env, q):
    client, pid, did = env
    res = client.post("/api/v1/investigations/interpret", json={"question": q, "project_id": pid, "dataset_ids": [did]})
    assert res.status_code == 200, res.text
    return res.json()


def test_route_never_shows_a_calculation_the_resolver_refused(api_env):
    body = _interpret(api_env, "Calculate growth")
    assert body["can_auto_proceed"] is False and body["plan_hash"] is None
    assert "SUM(units)" not in body["planned_calculation"]
    assert body["planned_calculation"].startswith("Not executable")


def test_route_unavailable_filter_is_blocked_with_a_readable_reason(api_env):
    body = _interpret(api_env, "Show total revenue where loyalty_tier = Gold")
    assert body["can_auto_proceed"] is False and body["plan_hash"] is None
    assert "loyalty_tier" in body["clarification_prompt"]


# ---- analyst-owned group-difference claims need two independent confirmations ------------------
from packages.analytics_core.src.governance.verification_gate import analyst_group_comparison_independently_verified as _gv

_GOOD = {"anova_p": 1e-9, "kruskal_p": 1e-8, "robust": True, "adequate_power": True}


def test_group_claim_verified_only_with_both_tests_robustness_and_power():
    assert _gv("GROUP_COMPARISON", _GOOD)
    for broken in ({"kruskal_p": 0.2}, {"anova_p": 0.2}, {"robust": None}, {"robust": False},
                   {"adequate_power": False}, {"kruskal_p": None}, {"anova_p": float("nan")}):
        assert not _gv("GROUP_COMPARISON", {**_GOOD, **broken}), broken
    assert not _gv("ASSOCIATION", _GOOD) and not _gv("GROUP_COMPARISON", None)


def test_end_to_end_null_difference_is_never_reported_as_significant():
    from tests.independent_release.test_session10_analyst_answer import _run
    r = np.random.RandomState(21)
    n = 1200
    df = pd.DataFrame({"region": r.choice(["North", "South", "East", "West"], n), "order_value": r.normal(100, 20, n)})
    verdict, answer, status = _run(df, "Does average order value differ by region?")
    assert status == "COMPLETED" and verdict != "STATISTICALLY_SIGNIFICANT"


def test_end_to_end_real_difference_is_reported_with_the_leading_group():
    from tests.independent_release.test_session10_analyst_answer import _run
    r = np.random.RandomState(21)
    n = 1200
    reg = r.choice(["North", "South", "East", "West"], n)
    df = pd.DataFrame({"region": reg, "order_value": r.normal(100, 20, n) + np.where(reg == "West", 15, 0)})
    verdict, answer, _ = _run(df, "Does average order value differ by region?")
    assert verdict == "STATISTICALLY_SIGNIFICANT" and "West" in answer
