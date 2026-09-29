"""Regression tests for P0-A (defect019 audit, section 8): churn/event
column ambiguity in semantic.py must never be resolved by column order.

Required rule:
    0 candidates  -> no resolved churn event (UNRESOLVED)
    1 candidate   -> resolved (RESOLVED)
    >1 candidates -> ambiguous unless equivalence/authority is established
                     (AMBIGUOUS)

The engine must not silently choose whichever happens to occur first in the
schema. Exercised with churn / is_churn / customer_churn / subscription_churn
/ cancelled appearing simultaneously, as specified.
"""
import pandas as pd
import pytest

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine


def _resolve(df: pd.DataFrame, question: str = "Why are customers churning?"):
    intent = IntentEngine.parse_intent(question, list(df.columns))
    return SemanticEngine().resolve_schema(intent, {"customers": df})


class TestChurnEventCandidateCounting:
    def test_zero_candidates_is_unresolved(self):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "monthly_spend": [10, 20, 30, 40],
        })
        res = _resolve(df)
        assert res.churn_event_col is None
        assert res.churn_event_resolution_status == "UNRESOLVED"
        assert res.churn_event_ambiguity == []
        assert res.churn_outcome_available is False

    def test_one_candidate_is_resolved(self):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "region": ["e", "w", "e", "w"],
            "churn": [0, 1, 0, 1],
        })
        res = _resolve(df)
        assert res.churn_event_col == "churn"
        assert res.churn_event_resolution_status == "RESOLVED"
        assert res.churn_event_ambiguity == []
        assert res.churn_outcome_available is True

    def test_multiple_simultaneous_candidates_is_ambiguous_not_first_column(self):
        """churn, is_churn, customer_churn, subscription_churn, and
        cancelled all present simultaneously -- the engine must not pick
        whichever occurs first by column position."""
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churn": [0, 1, 0, 1],
            "is_churn": [0, 1, 0, 1],
            "customer_churn": [0, 1, 0, 1],
            "subscription_churn": [0, 1, 0, 1],
            "cancelled": [0, 1, 0, 1],
        })
        res = _resolve(df)
        assert res.churn_event_col is None
        assert res.churn_event_resolution_status == "AMBIGUOUS"
        assert set(res.churn_event_ambiguity) == {
            "churn", "is_churn", "customer_churn", "subscription_churn", "cancelled",
        }
        assert res.churn_outcome_available is False

    def test_ambiguity_is_order_independent(self):
        """Reordering the ambiguous columns must not change the outcome --
        proves the resolution isn't secretly still order-sensitive."""
        cols_forward = ["customer_id", "churn", "is_churn", "cancelled"]
        cols_reversed = ["customer_id", "cancelled", "is_churn", "churn"]
        data = {"customer_id": [1, 2, 3, 4], "churn": [0, 1, 0, 1],
                "is_churn": [0, 1, 0, 1], "cancelled": [0, 1, 0, 1]}

        df_forward = pd.DataFrame({c: data[c] for c in cols_forward})
        df_reversed = pd.DataFrame({c: data[c] for c in cols_reversed})

        res_forward = _resolve(df_forward)
        res_reversed = _resolve(df_reversed)

        assert res_forward.churn_event_resolution_status == "AMBIGUOUS"
        assert res_reversed.churn_event_resolution_status == "AMBIGUOUS"
        assert set(res_forward.churn_event_ambiguity) == set(res_reversed.churn_event_ambiguity)

    def test_negative_keyword_columns_excluded_from_candidacy(self):
        """A column like churn_reason or churn_score must not count as a
        churn-event candidate (existing negative-keyword filter) -- so a
        dataset with one real event column plus metadata about it should
        still resolve cleanly, not become falsely ambiguous."""
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churn": [0, 1, 0, 1],
            "churn_reason": ["a", "b", "a", "b"],
            "churn_score": [0.1, 0.9, 0.2, 0.8],
        })
        res = _resolve(df)
        assert res.churn_event_col == "churn"
        assert res.churn_event_resolution_status == "RESOLVED"
