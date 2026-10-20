"""Authority firewall (step 2): enforced COUNT / SIMPLE_AGGREGATE families.

Proves, through the real IntentEngine -> SemanticEngine -> UniversalQuestionCompiler path:
  * resolvable simple aggregates / counts are NOT refused (no false veto);
  * an unresolvable aggregate IS refused and nothing downstream re-supplies its roles;
  * non-enforced families (causal, association, "worth the most") are untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
)
from packages.analytics_core.src.intelligence.universal_question_planner import (
    UniversalQuestionCompiler,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "order_id": range(n),
            "region": rng.choice(["N", "S", "E"], n),
            "revenue": rng.random(n) * 100,
            "discount": rng.random(n),
            "quantity": rng.integers(1, 9, n),
            "customer_id": rng.integers(1, 50, n),
            "order_date": pd.date_range("2025-01-01", periods=n),
        }
    )


def _plan(question: str, df: pd.DataFrame):
    intent = IntentEngine.parse_intent(question)
    sem = SemanticEngine().resolve_schema(intent, {"orders": df})
    return UniversalQuestionCompiler.compile(question, semantic=sem, df=df)


@pytest.mark.parametrize(
    "question,target",
    [
        ("What is the average revenue?", "revenue"),
        ("What is the median quantity?", "quantity"),
        ("Total revenue by region", "revenue"),
        ("What is the total revenue?", "revenue"),
    ],
)
def test_resolvable_simple_aggregate_is_not_vetoed(df, question, target):
    contract = compile_canonical_question_contract(question, df)
    assert contract.family == "SIMPLE_AGGREGATE"
    assert contract.completeness == "COMPLETE"
    assert contract.target_column == target
    plan = _plan(question, df)
    assert "authority_veto" not in plan.semantics.semantic_evidence
    assert plan.semantics.target_column == target


def test_total_by_region_keeps_grouping(df):
    contract = compile_canonical_question_contract("Total revenue by region", df)
    assert list(contract.grouping_columns) == ["region"]


@pytest.mark.parametrize(
    "question",
    [
        "What is the average flibbertigibbet?",  # no such column
        "What is the average region?",  # non-numeric measure
        "What is the total revenue per order?",  # "total ... per order" has no unambiguous meaning
    ],
)
def test_unresolvable_simple_aggregate_is_refused_and_final(df, question):
    contract = compile_canonical_question_contract(question, df)
    assert contract.family == "SIMPLE_AGGREGATE"
    assert contract.completeness == "INCOMPLETE"
    plan = _plan(question, df)
    veto = plan.semantics.semantic_evidence.get("authority_veto")
    assert veto is not None, "refusal must be recorded on the plan"
    assert plan.semantics.target_column is None
    assert plan.semantics.canonical_roles_authoritative is False
    assert plan.semantics.resolution_confidence <= 0.2
    assert any("Enforced SIMPLE_AGGREGATE" in u for u in plan.unresolved_questions)


@pytest.mark.parametrize(
    "question",
    [
        "Did discount affect quantity?",
        "Which customers are worth the most?",
        "Why did revenue drop over time?",
    ],
)
def test_non_enforced_families_are_not_vetoed(df, question):
    contract = compile_canonical_question_contract(question, df)
    assert contract.completeness == "NOT_ENFORCED"
    plan = _plan(question, df)
    assert "authority_veto" not in plan.semantics.semantic_evidence


def test_worth_the_most_stays_unresolved(df):
    plan = _plan("Which customers are worth the most?", df)
    assert any("worth" in u.lower() for u in plan.unresolved_questions)


@pytest.mark.parametrize("question", ["How many customers are there?", "How many orders are there?"])
def test_resolvable_count_is_not_vetoed(df, question):
    contract = compile_canonical_question_contract(question, df)
    assert contract.family == "COUNT" and contract.completeness == "COMPLETE"
    assert "authority_veto" not in _plan(question, df).semantics.semantic_evidence


@pytest.mark.parametrize(
    "question", ["How many zorblats are there?", "How many widgets are there?", "How many sprockets have signed up?"]
)
def test_count_of_unknown_noun_is_incomplete_and_refused(df, question):
    contract = compile_canonical_question_contract(question, df)
    assert contract.family == "COUNT"
    assert contract.completeness == "INCOMPLETE"
    plan = _plan(question, df)
    assert "authority_veto" in plan.semantics.semantic_evidence
    assert plan.semantics.target_column is None


def test_count_with_member_style_id_column_still_resolves():
    d = pd.DataFrame({"member_id": range(10), "plan": ["a", "b"] * 5})
    contract = compile_canonical_question_contract("How many members are there?", d)
    assert contract.completeness == "COMPLETE" and contract.target_column == "member_id"


# --- Regression: step-2 false veto on cross-table ranking questions (found by
# packages/analytics_core/tests/test_m1_controller_two_table_autonomous.py) -----------------------
@pytest.mark.parametrize(
    "question",
    [
        "Which customer segment generates the highest total order amount?",
        "Which region has the lowest average revenue?",
        "Which segment has the top total revenue?",
    ],
)
def test_superlative_ranking_with_aggregate_word_is_not_a_plain_aggregate(df, question):
    """A selection/ranking question that merely contains 'total'/'average' must not be resolved
    as a grouping-free DESCRIPTIVE aggregate, nor refused as an incomplete one."""
    contract = compile_canonical_question_contract(question, df)
    assert not (contract.task_family == "DESCRIPTIVE" and not contract.grouping_columns)
    assert contract.completeness != "INCOMPLETE"
    plan = _plan(question, df)
    assert "authority_veto" not in plan.semantics.semantic_evidence


def test_plain_aggregate_must_open_the_question(df):
    """The 9b rule is anchored: an aggregate word mid-sentence does not trigger it."""
    d = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "order_id": ["a", "b", "c"]})
    mid = compile_canonical_question_contract("Which team generates the highest total amount?", d)
    assert mid.target_column is None
    opening = compile_canonical_question_contract("What is the total amount?", d)
    assert opening.target_column == "amount" and opening.completeness == "COMPLETE"


@pytest.mark.parametrize(
    "question",
    ["Does average order value differ by region?", "Is the average revenue different across regions?"],
)
def test_comparison_wording_with_aggregate_word_is_not_vetoed(df, question):
    contract = compile_canonical_question_contract(question, df)
    assert contract.completeness != "INCOMPLETE"
    assert "authority_veto" not in _plan(question, df).semantics.semantic_evidence


def test_average_per_order_resolves_only_when_order_grain_is_verified(df):
    """"average revenue per order" is the mean over rows iff order_id is unique per row (the grain is
    the order). With repeated order ids the grain is unverified and the question is not guessed."""
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    ok = compile_canonical_question_contract("What is the average revenue per order?", df)
    if df["order_id"].nunique() == len(df):
        assert ok.completeness == "COMPLETE" and ok.target_column == "revenue" and not ok.grouping_columns
        assert ok.resolution_evidence.get("row_grain", {}).get("identifier_column") == "order_id"
    repeated = df.copy()
    repeated["order_id"] = repeated["order_id"].iloc[0]
    bad = compile_canonical_question_contract("What is the average revenue per order?", repeated)
    assert not (bad.completeness == "COMPLETE" and not bad.grouping_columns and bad.target_column == "revenue" and "row_grain" in bad.resolution_evidence)
