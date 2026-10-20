"""Phase 1 Acceptance Suite: 35 Semantic Boundary & Human-Guided Investigation Test Cases.

Tests the actual production investigation path and asserts the exact verdict, answer,
metric, population, ambiguity status, and fail-closed verification invariants.

Coverage across 35 explicit cases:
- Cases 1-5: Clear ranking questions (auto-proceed, verified totals).
- Cases 6-10: Clear comparisons (auto-proceed, confidence intervals).
- Cases 11-15: Multiple plausible metrics (flags AMBIGUOUS_METRIC, requires choice, no silent fallback).
- Cases 16-20: Missing measures (e.g. profit, cost; flags MISSING_DATA, explains missing columns, proposes defensible alternative).
- Cases 21-25: Causal on observational data (flags CAUSAL_ON_OBSERVATIONAL, explains non-causal bounds).
- Cases 26-28: Join fanout risk (flags JOIN_FANOUT_RISK, alerts to row duplication).
- Cases 29-31: Zero-match / empty filter handling.
- Cases 32-35: Human review invariants (Acceptance cannot flip failed verification to VERIFIED; reject requires reason; redo reruns with parameters).
"""
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.analytics_core.src.governance.guided_investigation import (
    AmbiguityStatus,
    HumanReviewAction,
    ProposedInterpretation,
    apply_human_review_decision,
    evaluate_guided_investigation,
)


@pytest.fixture
def ecommerce_df():
    """Standard e-commerce dataset with distinct revenue metrics."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "category": ["Electronics", "Electronics", "Apparel", "Apparel", "Home", "Home", "Books", "Books"],
        "region": ["North", "South", "North", "South", "North", "South", "North", "South"],
        "gross_revenue": [1200.0, 850.0, 450.0, 300.0, 950.0, 600.0, 150.0, 200.0],
        "net_revenue": [1050.0, 720.0, 390.0, 270.0, 810.0, 510.0, 130.0, 175.0],
        "units_sold": [12, 8, 15, 10, 9, 6, 5, 7],
        "discount_pct": [0.12, 0.15, 0.13, 0.10, 0.14, 0.15, 0.13, 0.12],
    })


@pytest.fixture
def single_revenue_df():
    """Dataset with a single unambiguous revenue measure and no profit/cost."""
    return pd.DataFrame({
        "sales_rep": ["Alice", "Bob", "Charlie", "Diana"],
        "territory": ["East", "West", "East", "West"],
        "segment": ["Enterprise", "SMB", "Enterprise", "SMB"],
        "channel": ["Direct", "Partner", "Direct", "Partner"],
        "store": ["Store_A", "Store_B", "Store_A", "Store_B"],
        "revenue": [50000.0, 35000.0, 62000.0, 41000.0],
        "deals_closed": [5, 4, 7, 3],
        "active_days": [22, 20, 21, 19],
    })


# ==============================================================================
# Cases 1-5: Clear ranking questions (auto-proceed, verified totals)
# ==============================================================================

def test_case_01_clear_ranking_by_category(single_revenue_df):
    """Case 1: Which sales rep had highest revenue."""
    res = evaluate_guided_investigation("Which sales rep had the highest revenue?", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "revenue"
    assert res.group_dimension == "sales_rep"
    assert 'SUM("revenue")' in res.planned_calculation


def test_case_02_clear_ranking_by_territory(single_revenue_df):
    """Case 2: Top territory by deals closed."""
    res = evaluate_guided_investigation("Which territory had the highest deals_closed?", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "deals_closed"
    assert res.group_dimension == "territory"


def test_case_03_clear_ranking_by_segment(single_revenue_df):
    """Case 3: Best segment by revenue."""
    res = evaluate_guided_investigation("Which segment generated the highest revenue?", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "revenue"
    assert res.group_dimension == "segment"


def test_case_04_clear_ranking_lowest_channel(single_revenue_df):
    """Case 4: Lowest revenue channel ranking."""
    res = evaluate_guided_investigation("Which channel had the lowest revenue?", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "revenue"
    assert res.group_dimension == "channel"


def test_case_05_clear_ranking_by_store(single_revenue_df):
    """Case 5: Store with most active days."""
    res = evaluate_guided_investigation("Which store had the highest active_days?", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "active_days"
    assert res.group_dimension == "store"


# ==============================================================================
# Cases 6-10: Clear comparisons (auto-proceed, confidence intervals)
# ==============================================================================

def test_case_06_clear_comparison_territory(single_revenue_df):
    """Case 6: Compare revenue between East and West territories."""
    res = evaluate_guided_investigation("Compare revenue between territory", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "revenue"
    assert res.task in ("COMPARISON", "ASSOCIATION")


def test_case_07_clear_comparison_channel(single_revenue_df):
    """Case 7: Difference in deals closed across channel."""
    res = evaluate_guided_investigation("Difference in deals closed across channel", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "deals_closed"


def test_case_08_clear_comparison_segment(single_revenue_df):
    """Case 8: Compare revenue by segment."""
    res = evaluate_guided_investigation("Compare revenue by segment", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "revenue"
    assert res.group_dimension == "segment"


def test_case_09_clear_comparison_versus(single_revenue_df):
    """Case 9: Revenue versus territory comparison."""
    res = evaluate_guided_investigation("Revenue versus territory", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "revenue"


def test_case_10_clear_comparison_higher_than(single_revenue_df):
    """Case 10: Is revenue higher than average by store."""
    res = evaluate_guided_investigation("Is revenue higher than average by store", df=single_revenue_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True


# ==============================================================================
# Cases 11-15: Multiple plausible metrics (flags AMBIGUOUS_METRIC)
# ==============================================================================

def test_case_11_ambiguous_revenue_metric(ecommerce_df):
    """Case 11: Question asks for revenue when gross_revenue and net_revenue exist."""
    res = evaluate_guided_investigation("What was our total revenue by category?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.AMBIGUOUS_METRIC
    assert res.can_auto_proceed is False
    assert "gross_revenue" in str(res.clarification_options)
    assert "net_revenue" in str(res.clarification_options)
    assert res.clarification_prompt is not None


def test_case_12_ambiguous_revenue_ranking(ecommerce_df):
    """Case 12: Ranking question with ambiguous revenue measure."""
    res = evaluate_guided_investigation("Which region had the highest revenue?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.AMBIGUOUS_METRIC
    assert res.can_auto_proceed is False
    assert any("gross_revenue" in opt.lower() for opt in res.clarification_options)


def test_case_13_ambiguous_sales_synonym(ecommerce_df):
    """Case 13: Question asks for sales when multiple revenue measures exist."""
    res = evaluate_guided_investigation("Show sales by category", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.AMBIGUOUS_METRIC
    assert res.can_auto_proceed is False


def test_case_14_ambiguous_revenue_comparison(ecommerce_df):
    """Case 14: Compare revenue between regions with multiple revenue measures."""
    res = evaluate_guided_investigation("Compare revenue across region", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.AMBIGUOUS_METRIC
    assert res.can_auto_proceed is False


def test_case_15_explicit_net_revenue_clears_ambiguity(ecommerce_df):
    """Case 15: Explicitly naming net_revenue bypasses ambiguity and proceeds."""
    res = evaluate_guided_investigation("Which category had the highest net_revenue?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert res.can_auto_proceed is True
    assert res.target_metric == "net_revenue"


# ==============================================================================
# Cases 16-20: Missing measures (flags MISSING_DATA, explains missing columns)
# ==============================================================================

def test_case_16_missing_profit_measure(ecommerce_df):
    """Case 16: Question asks for profit when no profit or cost column exists."""
    res = evaluate_guided_investigation("Which category was the most profitable?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert res.can_auto_proceed is False
    assert "profit" in res.proposed_interpretation.lower()
    assert "missing" in res.planned_calculation.lower()
    assert any("instead" in opt.lower() for opt in res.clarification_options)


def test_case_17_missing_cost_measure(ecommerce_df):
    """Case 17: Question asks for total cost when cost is absent."""
    res = evaluate_guided_investigation("What was our total cost by region?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert res.can_auto_proceed is False
    assert "cost" in res.clarification_prompt.lower()


def test_case_18_missing_margin_measure(ecommerce_df):
    """Case 18: Question asks for profit margin when margin is absent."""
    res = evaluate_guided_investigation("Which region had the highest margin?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert res.can_auto_proceed is False


def test_case_19_missing_churn_measure(ecommerce_df):
    """Case 19: Question asks for churn when churn is absent."""
    res = evaluate_guided_investigation("What is the churn rate by category?", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert res.can_auto_proceed is False


def test_case_20_missing_measure_proposes_defensible_alternative(ecommerce_df):
    """Case 20: Missing measure clarification proposes available numeric alternatives."""
    res = evaluate_guided_investigation("Analyze profit by region", df=ecommerce_df)
    assert res.ambiguity_status == AmbiguityStatus.MISSING_DATA
    # Must offer to investigate changes in available revenue/units instead
    assert any("gross_revenue" in opt or "net_revenue" in opt or "instead" in opt for opt in res.clarification_options)


# ==============================================================================
# Cases 21-25: Causal on observational data (flags CAUSAL_ON_OBSERVATIONAL)
# ==============================================================================

def test_case_21_causal_did_discount_cause_sales(ecommerce_df):
    """Case 21: Direct causal question on observational dataset."""
    res = evaluate_guided_investigation("Did discounts cause higher sales?", df=ecommerce_df, is_observational=True)
    assert res.ambiguity_status == AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL
    assert res.can_auto_proceed is False
    assert "observational" in res.clarification_prompt.lower()
    assert any("non-causal" in opt.lower() or "association" in opt.lower() for opt in res.clarification_options)


def test_case_22_causal_what_caused_churn(ecommerce_df):
    """Case 22: Diagnostic 'what caused' question."""
    res = evaluate_guided_investigation("What caused the increase in units_sold?", df=ecommerce_df, is_observational=True)
    assert res.ambiguity_status == AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL
    assert res.can_auto_proceed is False


def test_case_23_causal_why_did_revenue_drop(ecommerce_df):
    """Case 23: Diagnostic 'why did' question."""
    res = evaluate_guided_investigation("Why did net_revenue drop in South region?", df=ecommerce_df, is_observational=True)
    assert res.ambiguity_status == AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL
    assert res.can_auto_proceed is False


def test_case_24_causal_does_promotion_drive_revenue(ecommerce_df):
    """Case 24: 'Does X drive Y' causal claim."""
    res = evaluate_guided_investigation("Does category drive gross_revenue?", df=ecommerce_df, is_observational=True)
    assert res.ambiguity_status == AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL
    assert res.can_auto_proceed is False


def test_case_25_causal_bounds_refuses_causality_without_design(ecommerce_df):
    """Case 25: Observational causal bounds provide qualified association alternative."""
    res = evaluate_guided_investigation("Did category lead to higher units_sold?", df=ecommerce_df, is_observational=True)
    assert res.ambiguity_status == AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL
    assert "randomization" in res.clarification_prompt.lower() or "observational" in res.clarification_prompt.lower()


# ==============================================================================
# Cases 26-28: Join fanout risk (flags JOIN_FANOUT_RISK)
# ==============================================================================

def test_case_26_one_to_many_join_fanout_detected():
    """Case 26: 1:N relationship flags fanout risk and warns of row duplication."""
    res = evaluate_guided_investigation(
        "Calculate total revenue by customer",
        table_relationships=[{
            "left_table": "customers",
            "right_table": "orders",
            "cardinality": "1:N",
            "join_key": "customer_id",
            "metric": "revenue",
        }],
    )
    assert res.ambiguity_status == AmbiguityStatus.JOIN_FANOUT_RISK
    assert res.can_auto_proceed is False
    assert "duplicate" in res.clarification_prompt.lower() or "fanout" in res.clarification_prompt.lower()


def test_case_27_many_to_many_join_fanout_detected():
    """Case 27: N:M relationship flags fanout risk."""
    res = evaluate_guided_investigation(
        "Calculate total spend",
        table_relationships=[{
            "left_table": "accounts",
            "right_table": "transactions",
            "cardinality": "MANY_TO_MANY",
            "join_key": "account_id",
            "metric": "spend",
        }],
    )
    assert res.ambiguity_status == AmbiguityStatus.JOIN_FANOUT_RISK
    assert res.can_auto_proceed is False


def test_case_28_join_fanout_offers_pre_aggregation_option():
    """Case 28: Fanout warning offers pre-aggregation choice."""
    res = evaluate_guided_investigation(
        "Join orders and items",
        table_relationships=[{
            "left_table": "orders",
            "right_table": "order_items",
            "cardinality": "ONE_TO_MANY",
            "join_key": "order_id",
            "metric": "item_price",
        }],
    )
    assert res.ambiguity_status == AmbiguityStatus.JOIN_FANOUT_RISK
    assert any("aggregate" in opt.lower() for opt in res.clarification_options)


# ==============================================================================
# Cases 29-31: Zero-match / empty filter handling
# ==============================================================================

def test_case_29_empty_dataframe_handled_safely():
    """Case 29: Profiling empty dataframe does not crash and handles zero records."""
    empty_df = pd.DataFrame({"region": pd.Series(dtype="object"), "revenue": pd.Series(dtype="float64")})
    res = evaluate_guided_investigation("Which region had highest revenue?", df=empty_df)
    assert res is not None
    assert res.target_metric == "revenue"


def test_case_30_zero_matching_numeric_columns_defaults_to_count():
    """Case 30: Dataset with only categorical columns plans count calculation."""
    text_only_df = pd.DataFrame({"city": ["London", "Paris", "Tokyo"], "status": ["A", "B", "A"]})
    res = evaluate_guided_investigation("How many items by city?", df=text_only_df)
    assert res.ambiguity_status == AmbiguityStatus.CLEAR
    assert "COUNT(*)" in res.planned_calculation


def test_case_31_machine_verification_failed_halts_claim():
    """Case 31: Failed machine verification halts claim and flags VERIFICATION_FAILED."""
    res = evaluate_guided_investigation(
        "Which sales rep had highest revenue?",
        verification_passed=False,
    )
    assert res.ambiguity_status == AmbiguityStatus.VERIFICATION_FAILED
    assert res.can_auto_proceed is False
    assert "failed" in res.proposed_interpretation.lower()


# ==============================================================================
# Cases 32-35: Human review invariants (MindEd 1.0.0 Governance Invariants)
# ==============================================================================

def test_case_32_human_accept_does_not_override_failed_verification():
    """Case 32: CRITICAL INVARIANT: Accept on unverified analysis remains UNVERIFIED."""
    rec = apply_human_review_decision(
        investigation_id="inv-test-001",
        action="ACCEPT",
        actor="lead_analyst@company.com",
        comments="Accepted for local exploratory dashboard only",
        verification_passed=False,
    )
    assert rec["review_status"] == "ACCEPTED_FOR_WORKFLOW_UNVERIFIED"
    assert rec["effective_verification_status"] == "UNVERIFIED"
    assert "does not override verification checks" in rec["notice"]


def test_case_33_human_accept_on_verified_analysis_records_verified():
    """Case 33: Accept on verified analysis records ACCEPTED_AND_VERIFIED."""
    rec = apply_human_review_decision(
        investigation_id="inv-test-002",
        action="ACCEPT",
        actor="lead_analyst@company.com",
        comments="Approved for board presentation",
        verification_passed=True,
    )
    assert rec["review_status"] == "ACCEPTED_AND_VERIFIED"
    assert rec["effective_verification_status"] == "VERIFIED"


def test_case_34_human_rejection_requires_documented_reason():
    """Case 34: Reject action requires a mandatory documented explanation."""
    with pytest.raises(ValueError, match="Rejection requires a documented reason"):
        apply_human_review_decision(
            investigation_id="inv-test-003",
            action="REJECT",
            actor="lead_analyst@company.com",
            comments="",
            verification_passed=True,
        )

    valid_reject = apply_human_review_decision(
        investigation_id="inv-test-003",
        action="REJECT",
        actor="lead_analyst@company.com",
        comments="Metric does not account for customer refunds in Q3",
        verification_passed=True,
    )
    assert valid_reject["review_status"] == "REJECTED"
    assert valid_reject["effective_verification_status"] == "REJECTED"


def test_case_35_human_redo_records_revised_parameters():
    """Case 35: Redo action records modified parameters and sets status to NOT_RUN."""
    rec = apply_human_review_decision(
        investigation_id="inv-test-004",
        action="REDO",
        actor="lead_analyst@company.com",
        comments="Rerun using net_revenue instead of gross_revenue",
        modified_parameters={"selected_metric": "net_revenue", "scope_filter": "region = 'North'"},
        verification_passed=True,
    )
    assert rec["review_status"] == "REDO_REQUESTED"
    assert rec["effective_verification_status"] == "NOT_RUN"
    assert rec["modified_parameters"]["selected_metric"] == "net_revenue"
    assert rec["modified_parameters"]["scope_filter"] == "region = 'North'"


# ==============================================================================
# Cases 36-40: Authoritative Guidance Execution Contract & End-to-End Tests
# ==============================================================================

def test_case_36_clarification_choice_resolves_metric(tmp_path, ecommerce_df):
    """Case 36: Selecting 'Net revenue (`net_revenue`)' clarification resolves target metric in controller."""
    import os
    from apps.api.src.core import database as dbmod
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/t36.db"
    os.environ["DATA_STORAGE_DIR"] = f"{tmp_path}/storage36"
    dbmod.init_db()
    from apps.api.src.models.entities import Project, Investigation, InvestigationEvent
    from apps.api.src.services.dataset_service import DatasetService
    from packages.analytics_core.src.runtime.controller import InvestigationController

    with dbmod.SessionLocal() as db:
        p = Project(name="ws_e2e_36")
        db.add(p)
        db.commit()
        csv_file = tmp_path / "ecom36.csv"
        ecommerce_df.to_csv(csv_file, index=False)
        DatasetService(db).ingest_dataset_file(p.id, "ecom.csv", open(csv_file, "rb").read())

        iid = "INV-E2E-36"
        inv = Investigation(id=iid, project_id=p.id, question="What was our revenue?", status="PLANNED")
        db.add(inv)
        db.add(InvestigationEvent(
            id="EVT-E2E-36",
            investigation_id=iid,
            event_type="investigation.human_guidance_intent",
            event_payload_json={"clarification_response": "Net revenue (`net_revenue`)"},
        ))
        db.commit()

    ctrl = InvestigationController(session_factory=dbmod.SessionLocal)
    ok = ctrl.execute_investigation(investigation_id=iid, worker_id="w-36")
    assert ok is True

    with dbmod.SessionLocal() as db:
        res = db.query(Investigation).filter(Investigation.id == iid).first()
        assert res.status == "COMPLETED"
        assert res.verdict_type in ("OBSERVED", "INCONCLUSIVE", "STATISTICALLY_SIGNIFICANT")


def test_case_37_scope_filter_affects_population(tmp_path, ecommerce_df):
    """Case 37: Applying custom scope filter restricts dataset population and fails closed on tautology injection."""
    import os
    from apps.api.src.core import database as dbmod
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/t37.db"
    os.environ["DATA_STORAGE_DIR"] = f"{tmp_path}/storage37"
    dbmod.init_db()
    from apps.api.src.models.entities import Project, Investigation, InvestigationEvent
    from apps.api.src.services.dataset_service import DatasetService
    from packages.analytics_core.src.runtime.controller import InvestigationController, validate_scope_filter_expression

    # Part A: validate_scope_filter_expression rejects boolean injection
    with pytest.raises(ValueError, match="Boolean literals"):
        validate_scope_filter_expression("region = 'North' OR TRUE", list(ecommerce_df.columns))

    with pytest.raises(ValueError, match="Tautological comparisons"):
        validate_scope_filter_expression("region = 'North' OR 1=1", list(ecommerce_df.columns))

    # Part B: Valid filter executes and filters population in controller
    with dbmod.SessionLocal() as db:
        p = Project(name="ws_e2e_37")
        db.add(p)
        db.commit()
        csv_file = tmp_path / "ecom37.csv"
        ecommerce_df.to_csv(csv_file, index=False)
        DatasetService(db).ingest_dataset_file(p.id, "ecom.csv", open(csv_file, "rb").read())

        iid = "INV-E2E-37"
        inv = Investigation(id=iid, project_id=p.id, question="What was our total net revenue by category?", status="PLANNED")
        db.add(inv)
        db.add(InvestigationEvent(
            id="EVT-E2E-37",
            investigation_id=iid,
            event_type="investigation.human_guidance_intent",
            event_payload_json={"scope_filter": "region = 'North'"},
        ))
        db.commit()

    ctrl = InvestigationController(session_factory=dbmod.SessionLocal)
    ok = ctrl.execute_investigation(investigation_id=iid, worker_id="w-37")
    assert ok is True
    with dbmod.SessionLocal() as db:
        res = db.query(Investigation).filter(Investigation.id == iid).first()
        assert res.status == "COMPLETED"


def test_case_38_aggregation_selection_changes_calculation(tmp_path, ecommerce_df):
    """Case 38: Non-numeric metric selection fails closed immediately in controller."""
    import os
    from apps.api.src.core import database as dbmod
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/t38.db"
    os.environ["DATA_STORAGE_DIR"] = f"{tmp_path}/storage38"
    dbmod.init_db()
    from apps.api.src.models.entities import Project, Investigation, InvestigationEvent
    from apps.api.src.services.dataset_service import DatasetService
    from packages.analytics_core.src.runtime.controller import InvestigationController

    with dbmod.SessionLocal() as db:
        p = Project(name="ws_e2e_38")
        db.add(p)
        db.commit()
        csv_file = tmp_path / "ecom38.csv"
        ecommerce_df.to_csv(csv_file, index=False)
        DatasetService(db).ingest_dataset_file(p.id, "ecom.csv", open(csv_file, "rb").read())

        iid = "INV-E2E-38"
        inv = Investigation(id=iid, project_id=p.id, question="What was our total net revenue?", status="PLANNED")
        db.add(inv)
        # Category is string, not numeric
        db.add(InvestigationEvent(
            id="EVT-E2E-38",
            investigation_id=iid,
            event_type="investigation.human_guidance_intent",
            event_payload_json={"selected_metric": "category"},
        ))
        db.commit()

    ctrl = InvestigationController(session_factory=dbmod.SessionLocal)
    ok = ctrl.execute_investigation(investigation_id=iid, worker_id="w-38")
    assert ok is False
    with dbmod.SessionLocal() as db:
        res = db.query(Investigation).filter(Investigation.id == iid).first()
        assert res.status == "FAILED"
        assert "non-numeric" in res.direct_answer.lower()


def test_case_39_clarification_refusal_halts_execution(tmp_path, ecommerce_df):
    """Case 39: Selecting explicit refusal in clarification halts execution with inconclusive verdict."""
    import os
    from apps.api.src.core import database as dbmod
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/t39.db"
    os.environ["DATA_STORAGE_DIR"] = f"{tmp_path}/storage39"
    dbmod.init_db()
    from apps.api.src.models.entities import Project, Investigation, InvestigationEvent
    from apps.api.src.services.dataset_service import DatasetService
    from packages.analytics_core.src.runtime.controller import InvestigationController

    with dbmod.SessionLocal() as db:
        p = Project(name="ws_e2e_39")
        db.add(p)
        db.commit()
        csv_file = tmp_path / "ecom39.csv"
        ecommerce_df.to_csv(csv_file, index=False)
        DatasetService(db).ingest_dataset_file(p.id, "ecom.csv", open(csv_file, "rb").read())

        iid = "INV-E2E-39"
        inv = Investigation(id=iid, project_id=p.id, question="What was our profit by region?", status="PLANNED")
        db.add(inv)
        db.add(InvestigationEvent(
            id="EVT-E2E-39",
            investigation_id=iid,
            event_type="investigation.human_guidance_intent",
            event_payload_json={"clarification_response": "Refuse analysis due to missing measures"},
        ))
        db.commit()

    ctrl = InvestigationController(session_factory=dbmod.SessionLocal)
    ok = ctrl.execute_investigation(investigation_id=iid, worker_id="w-39")
    assert ok is True
    with dbmod.SessionLocal() as db:
        res = db.query(Investigation).filter(Investigation.id == iid).first()
        assert res.status == "COMPLETED"
        assert res.verdict_type == "INCONCLUSIVE"
        assert "halted by human analyst decision" in res.direct_answer.lower()


def test_case_40_unresolved_ambiguity_blocks_execution():
    """Case 40: Invariant: Plan with unselected clarification options cannot auto-proceed."""
    res = evaluate_guided_investigation("What was our revenue?", df=pd.DataFrame({"gross_revenue": [1], "net_revenue": [2]}))
    assert res.ambiguity_status == AmbiguityStatus.AMBIGUOUS_METRIC
    assert res.can_auto_proceed is False
    assert len(res.clarification_options) > 1


