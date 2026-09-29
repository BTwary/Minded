import pandas as pd

from packages.analytics_core.src.intelligence.dataset_question_discovery import (
    DatasetQuestionDiscovery,
    SuggestedQuestion,
)
from packages.schemas.src.semantic_graph import EpistemicSource


def test_single_table_discovers_grounded_questions():
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "revenue_usd": [10.0, 20.0, 30.0, 40.0, 50.0],
        "region": ["East", "West", "East", "West", "East"],
        "order_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02", "2026-02-03"]),
    })
    qs = DatasetQuestionDiscovery.discover({"orders": df}, max_questions=12)
    texts = [q.question for q in qs]
    assert texts
    assert any("revenue usd" in q.lower() for q in texts)
    assert any("region" in q.lower() for q in texts)
    assert any("order date" in q.lower() for q in texts)
    assert all(q.evidence_scope for q in qs)

    # Verify that questions have utility scores and epistemic tiers
    for q in qs:
        assert isinstance(q.utility_score, float)
        assert 0.0 <= q.utility_score <= 1.0
        assert q.epistemic_tier in [
            EpistemicSource.PHYSICAL_FACT.value,
            EpistemicSource.INFERRED_SEMANTICS.value,
            EpistemicSource.USER_CONFIRMATION.value,
            EpistemicSource.EXTERNAL_DECLARATION.value,
            EpistemicSource.MODEL_HYPOTHESIS.value,
        ]
        assert q.analytical_problem_class in [
            "DESCRIPTIVE_RECORD_COUNT",
            "DESCRIPTIVE_AGGREGATION",
            "SEGMENT_COMPARISON",
            "TEMPORAL_TREND",
            "LONGITUDINAL_COMPARISON",
            "RELATIONAL_CROSS_TABLE",
            "IDENTITY_RECONCILIATION",
        ]

    # Verify utility ranking order: questions should be ordered descending by utility_score
    scores = [q.utility_score for q in qs]
    assert scores == sorted(scores, reverse=True)


def test_cross_table_question_requires_discovered_relationship():
    orders = pd.DataFrame({"order_id": [1, 2, 3, 4], "customer_id": [10, 20, 10, 30], "revenue_usd": [10., 20., 15., 30.]})
    customers = pd.DataFrame({"customer_id": [10, 20, 30], "segment": ["A", "B", "A"]})
    qs = DatasetQuestionDiscovery.discover({"orders": orders, "customers": customers}, max_questions=20)
    rel_questions = [q for q in qs if q.analytical_problem_class == "RELATIONAL_CROSS_TABLE"]
    assert len(rel_questions) > 0
    assert any("using the relationship" in q.question.lower() for q in rel_questions)
    for q in rel_questions:
        assert "orders" in q.evidence_scope
        assert "customers" in q.evidence_scope


def test_accounting_reconciliation_question_discovery():
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "quantity": [2, 5, 10],
        "unit_price": [15.0, 10.0, 50.0],
        "gross_amount": [30.0, 50.0, 500.0],
    })
    qs = DatasetQuestionDiscovery.discover({"invoices": df}, max_questions=10)
    recon_questions = [q for q in qs if q.analytical_problem_class == "IDENTITY_RECONCILIATION"]
    assert len(recon_questions) >= 1
    assert "reconcile" in recon_questions[0].question.lower()
    assert "accounting identity" in recon_questions[0].question.lower()


def test_empty_dataset_set_is_safe():
    assert DatasetQuestionDiscovery.discover({}) == []



def test_ambiguous_dimensions_and_times_never_drive_suggestions():
    df = pd.DataFrame({
        "order_id": range(1, 9),
        "revenue_usd": [10., 20., 30., 40., 50., 60., 70., 80.],
        "region": ["East", "West"] * 4,
        "segment": ["SMB", "Enterprise"] * 4,
        "order_date": pd.to_datetime(["2026-01-01"] * 8),
        "updated_at": pd.to_datetime(["2026-02-01"] * 8),
    })
    qs = DatasetQuestionDiscovery.discover({"orders": df}, max_questions=20)
    classes = {q.analytical_problem_class for q in qs}
    assert "SEGMENT_COMPARISON" not in classes
    assert "TEMPORAL_TREND" not in classes
    assert "LONGITUDINAL_COMPARISON" not in classes


def test_ambiguous_reconciliation_roles_never_choose_first_column():
    df = pd.DataFrame({
        "quantity_ordered": [1, 2, 3],
        "quantity_shipped": [1, 2, 2],
        "unit_price": [10., 20., 30.],
        "gross_amount": [10., 40., 90.],
    })
    qs = DatasetQuestionDiscovery.discover({"invoices": df}, max_questions=20)
    assert not [q for q in qs if q.analytical_problem_class == "IDENTITY_RECONCILIATION"]
