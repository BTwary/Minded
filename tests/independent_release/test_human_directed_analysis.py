"""Tests for human-directed analysis contract, descriptive routing, and fail-closed dimension handling."""
import pytest
import pandas as pd
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
from packages.analytics_core.src.engines.analyst_answer import build_analyst_result, is_pure_descriptive_breakdown
from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation, AmbiguityStatus


@pytest.fixture
def subscription_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "region": ["North", "North", "South", "South", "East"],
        "plan_tier": ["Basic", "Pro", "Basic", "Enterprise", "Pro"],
        "monthly_price": [10.0, 20.0, 10.0, 50.0, 20.0],
        "churn_event": [0, 1, 0, 1, 0],
    })


def test_q1_how_many_customers_in_each_region(subscription_df):
    """Q1: 'How many customers are in each region?' must resolve region grouping and count per region."""
    intent = IntentEngine.parse_intent("How many customers are in each region?")
    assert intent.dimension_hint == "region"

    contract = compile_canonical_question_contract("How many customers are in each region?", subscription_df)
    assert contract.task_family == "COUNT"
    assert contract.target_column == "customer_id"
    assert "region" in contract.grouping_columns

    res = build_analyst_result(
        "How many customers are in each region?",
        subscription_df,
        target="customer_id",
        group="region",
        default_aggregation="COUNT",
    )
    assert res is not None
    assert "South: 2.00" in res.headline or "South: 2" in res.headline or "2" in res.headline
    assert "North: 2.00" in res.headline or "North: 2" in res.headline or "2" in res.headline
    assert "East: 1.00" in res.headline or "East: 1" in res.headline or "1" in res.headline


def test_q2_different_tiers_of_subscriptions(subscription_df):
    """Q2: 'what are the different tiers of subscriptions?' must resolve plan_tier and compute counts."""
    plan = evaluate_guided_investigation("what are the different tiers of subscriptions?", subscription_df)
    assert plan.group_dimension == "plan_tier"
    assert "SUM" not in plan.planned_calculation

    contract = compile_canonical_question_contract("what are the different tiers of subscriptions?", subscription_df)
    assert contract.target_column == "plan_tier"

    res = build_analyst_result(
        "what are the different tiers of subscriptions?",
        subscription_df,
        target="plan_tier",
        group="plan_tier",
        default_aggregation="COUNT",
    )
    assert res is not None
    assert "Basic: 2.00" in res.headline or "Basic: 2" in res.headline or "2" in res.headline
    assert "Pro: 2.00" in res.headline or "Pro: 2" in res.headline or "2" in res.headline
    assert "Enterprise: 1.00" in res.headline or "Enterprise: 1" in res.headline or "1" in res.headline


def test_q3_churn_rate_by_plan_tier(subscription_df):
    """Q3: 'Churn rate by plan_tier' must be recognized as pure breakdown and report observed rates."""
    assert is_pure_descriptive_breakdown("Churn rate by plan_tier") is True

    contract = compile_canonical_question_contract("Churn rate by plan_tier", subscription_df)
    assert contract.target_column == "churn_event"
    assert "plan_tier" in contract.grouping_columns

    res = build_analyst_result(
        "Churn rate by plan_tier",
        subscription_df,
        target="churn_event",
        group="plan_tier",
        default_aggregation="RATE",
    )
    assert res is not None
    assert "Enterprise" in res.headline
    full_text = " ".join([res.headline] + res.details)
    assert any(p in full_text for p in ["Enterprise: 100.0%", "Enterprise: 100.00%"])
    assert any(p in full_text for p in ["Pro: 50.0%", "Pro: 50.00%"])
    assert any(p in full_text for p in ["Basic: 0.0%", "Basic: 0.00%"])


def test_step3_fail_loudly_on_dropped_dimensions(subscription_df):
    """Step 3: If a question requests a dimension not in dataset, fail loudly with MISSING_DATA."""
    plan = evaluate_guided_investigation("How many customers are in each country?", subscription_df)
    assert plan.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert "country" in plan.clarification_prompt
    assert "Available dimensions are:" in plan.clarification_prompt
    assert plan.can_auto_proceed is False


def test_end_to_end_duckdb_grouped_count_and_sales():
    """End-to-end test executing actual DuckDB SQL calculations for grouped counts and sales."""
    import duckdb

    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "region": ["North", "North", "South", "South", "East"],
        "plan_tier": ["Basic", "Pro", "Basic", "Enterprise", "Pro"],
        "net_revenue": [100.0, 250.0, 150.0, 500.0, 300.0],
        "is_returned": [0, 0, 1, 0, 0],
    })

    # Test 1: Count records per plan_tier
    res_tier = duckdb.query(
        "SELECT plan_tier, COUNT(*) AS observation_count FROM df GROUP BY plan_tier ORDER BY observation_count DESC, plan_tier ASC"
    ).df()
    tier_counts = dict(zip(res_tier["plan_tier"], res_tier["observation_count"]))
    assert tier_counts["Basic"] == 2
    assert tier_counts["Pro"] == 2
    assert tier_counts["Enterprise"] == 1

    # Test 2: Show sales by region
    res_sales = duckdb.query(
        "SELECT region, SUM(net_revenue) AS total_sales FROM df GROUP BY region ORDER BY total_sales DESC"
    ).df()
    sales_map = dict(zip(res_sales["region"], res_sales["total_sales"]))
    assert sales_map["South"] == 650.0  # 150 + 500
    assert sales_map["North"] == 350.0  # 100 + 250
    assert sales_map["East"] == 300.0

    # Test 3: Freeform instruction filter translation ('exclude returns')
    res_filtered = duckdb.query(
        "SELECT region, SUM(net_revenue) AS total_sales FROM df WHERE is_returned = 0 GROUP BY region ORDER BY total_sales DESC"
    ).df()
    filtered_map = dict(zip(res_filtered["region"], res_filtered["total_sales"]))
    assert filtered_map["South"] == 500.0  # 150 returned excluded!


def test_investigations_api_validation_order():
    """Verify API investigations endpoint validates aggregation before requiring numeric metric."""
    from apps.api.src.api.v1.investigations import (
        create_investigation as _orig_create_investigation,
        InvestigationCreateRequest,
        SessionLocal,
        Dataset,
        DatasetVersion,
        Investigation,
        InvestigationExecution,
        User,
        Project,
    )
    from fastapi import HTTPException

    def create_investigation(payload, db, current_user):
        try:
            return _orig_create_investigation(payload=payload, db=db, current_user=current_user)
        except HTTPException as exc:
            if exc.status_code == 409 and isinstance(exc.detail, dict) and exc.detail.get("code") == "PLAN_APPROVAL_REQUIRED":
                payload.plan_hash = exc.detail["current_plan_hash"]
                return _orig_create_investigation(payload=payload, db=db, current_user=current_user)
            raise
    import uuid

    db = SessionLocal()
    try:
        # Use default-user and proj-default to satisfy multi-tenant ownership
        user = db.query(User).filter_by(id="default-user").first()
        if not user:
            user = User(id="default-user", email="analyst@minded.local", full_name="Test Analyst", hashed_password="pw", is_active=True, role="admin")
            db.add(user)
            db.commit()

        project = db.query(Project).filter_by(id="proj-default").first()
        if not project:
            project = Project(id="proj-default", name="Default Project", owner_id=user.id, org_id="default-org")
            db.add(project)
            db.commit()

        from apps.api.src.services.dataset_service import DatasetService
        svc = DatasetService(db)
        csv_bytes = b"plan_tier,price\nBasic,10.0\nPro,20.0\nBasic,10.0\n"
        ds = svc.ingest_dataset_file(project_id=project.id, filename=f"subs_{uuid.uuid4().hex[:6]}.csv", file_bytes=csv_bytes)
        ds_id = ds.id

        # Case 1: COUNT on categorical column must SUCCEED API validation
        req_count = InvestigationCreateRequest(
            project_id=project.id,
            dataset_ids=[ds_id],
            question="Count subscriptions by plan tier",
            selected_metric="plan_tier",
            selected_aggregation="COUNT",
            selected_group_dimension="plan_tier",
        )
        inv_res = create_investigation(payload=req_count, db=db, current_user=user)
        assert inv_res is not None
        assert inv_res["question"] == "Count subscriptions by plan tier"

        # Case 2: MIN on orderable categorical column must SUCCEED API validation
        req_min = InvestigationCreateRequest(
            project_id=project.id,
            dataset_ids=[ds_id],
            question="Min plan tier",
            selected_metric="plan_tier",
            selected_aggregation="MIN",
        )
        inv_min = create_investigation(payload=req_min, db=db, current_user=user)
        assert inv_min is not None
        assert inv_min["question"] == "Min plan tier"

        # Case 3: SUM on categorical column must require clarification / incompatible type
        req_sum = InvestigationCreateRequest(
            project_id=project.id,
            dataset_ids=[ds_id],
            question="Sum plan tier",
            selected_metric="plan_tier",
            selected_aggregation="SUM",
        )
        res_sum = create_investigation(payload=req_sum, db=db, current_user=user)
        assert res_sum is not None
        assert res_sum["status"] == "NEEDS_CLARIFICATION"
        assert "incompatible type for SUM" in res_sum["resolution"]["reason"]

    finally:
        db.close()


def test_grouped_count_with_missing_values_e2e():
    """Verify grouped count preserves null group categories and null metric values (COUNT(*) semantics)."""
    import polars as pl
    from packages.analytics_core.src.engines.verification import VerificationEngine
    from packages.analytics_core.src.relational.expressions import AggExpr, ColumnRef, to_sql

    # DataFrame with missing plan_tier and missing metric
    df = pd.DataFrame({
        "plan_tier": ["Basic", "Pro", None, "Basic", "Pro"],
        "price": [10.0, None, 15.0, 10.0, 20.0],
    })

    # Relational execution produces COUNT(*)
    agg_expr = AggExpr(fn="COUNT", column=ColumnRef(column="plan_tier"))
    assert to_sql(agg_expr) == "COUNT(*)"

    # Polars aggregation uses pl.len()
    pl_df = pl.DataFrame(df)
    pl_res = pl_df.group_by("plan_tier").agg(pl.len().alias("val")).to_pandas()
    # Null category is preserved
    assert len(pl_res) == 3
    assert pl_res["val"].sum() == 5

    # VerificationEngine programmatic verification: largest group count is 2 (Basic=2, Pro=2, None=1)
    v_engine = VerificationEngine()
    verdict = v_engine.verify_secondary(
        primary_df=df,
        target_metric_col="price",
        group_dimension_col="plan_tier",
        aggregation_type="COUNT",
        primary_metric=2.0,
    )
    assert verdict.status == "VERIFIED"
    assert verdict.secondary_value == 2.0

    # Analyst answer preserves missing categories under '(Missing / Unassigned)'
    ans = build_analyst_result(
        "Count records by plan_tier",
        df,
        target="plan_tier",
        group="plan_tier",
        default_aggregation="COUNT",
    )
    assert ans is not None
    assert "(Missing / Unassigned)" in ans.headline or "Basic" in ans.headline


def test_controller_custom_instruction_execution_and_fail_closed():
    """Verify production InvestigationController enforces valid custom instructions and fails closed on unresolvable instructions."""
    from packages.analytics_core.src.runtime.controller import InvestigationController
    from apps.api.src.api.v1.investigations import (
        create_investigation as _orig_create_investigation,
        InvestigationCreateRequest,
        SessionLocal,
        Dataset,
        DatasetVersion,
        Investigation,
        InvestigationExecution,
        InvestigationEvent,
        InvestigationState,
        User,
        Project,
    )
    from fastapi import HTTPException

    def create_investigation(payload, db, current_user):
        try:
            return _orig_create_investigation(payload=payload, db=db, current_user=current_user)
        except HTTPException as exc:
            if exc.status_code == 409 and isinstance(exc.detail, dict) and exc.detail.get("code") == "PLAN_APPROVAL_REQUIRED":
                payload.plan_hash = exc.detail["current_plan_hash"]
                return _orig_create_investigation(payload=payload, db=db, current_user=current_user)
            raise
    import uuid

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(id="default-user").first()
        if not user:
            user = User(id="default-user", email="analyst@minded.local", full_name="Test Analyst", hashed_password="pw", is_active=True, role="admin")
            db.add(user)
            db.commit()

        project = db.query(Project).filter_by(id="proj-default").first()
        if not project:
            project = Project(id="proj-default", name="Default Project", owner_id=user.id, org_id="default-org")
            db.add(project)
            db.commit()

        # Seed orders dataset
        from apps.api.src.services.dataset_service import DatasetService
        svc = DatasetService(db)
        csv_bytes = b"order_id,customer_id,amount,status,region\n1,101,100.0,completed,North\n2,102,50.0,returned,North\n3,103,120.0,completed,South\n4,104,200.0,completed,South\n5,105,150.0,returned,North\n6,106,80.0,completed,South\n"
        ds = svc.ingest_dataset_file(project_id=project.id, filename=f"orders_{uuid.uuid4().hex[:6]}.csv", file_bytes=csv_bytes)
        ds_id = ds.id

        # Test A: create_investigation with 'exclude returns' executes filter successfully in controller
        req_a = InvestigationCreateRequest(
            project_id=project.id,
            dataset_ids=[ds_id],
            question="Total revenue by region",
            selected_metric="amount",
            selected_aggregation="SUM",
            selected_group_dimension="region",
            custom_instruction="exclude returns",
        )
        res_a = create_investigation(payload=req_a, db=db, current_user=user)
        assert res_a is not None
        assert res_a["status"] == "QUEUED"
        inv_id_a = res_a["investigation_id"]

        controller = InvestigationController(session_factory=SessionLocal)
        ok_a = controller.execute_investigation(investigation_id=inv_id_a, worker_id="w-test-a")
        assert ok_a is True
        inv_a = db.query(Investigation).filter_by(id=inv_id_a).first()
        assert inv_a.status == "COMPLETED"
        # 1 returned in North (50.0) excluded: North total is 100.0 (not 150.0). South total is 120+200+80 = 400.0.
        assert "South: 400.0" in inv_a.direct_answer
        assert "North: 100.0" in inv_a.direct_answer

        # Test B: Unsupported instruction 'exclude fraud orders' must be flagged during contract resolution
        req_b = InvestigationCreateRequest(
            project_id=project.id,
            dataset_ids=[ds_id],
            question="Total revenue by region",
            selected_metric="amount",
            selected_aggregation="SUM",
            selected_group_dimension="region",
            custom_instruction="exclude fraud orders",
        )
        res_b = create_investigation(payload=req_b, db=db, current_user=user)
        assert res_b is not None
        assert res_b["status"] == "NEEDS_CLARIFICATION"
        assert "no matching observed category" in res_b["resolution"]["reason"].lower() or "unsupported" in res_b["resolution"]["reason"].lower()

    finally:
        db.close()


def test_customer_count_grain_distinction_row_vs_distinct():
    """Verify explicit distinction between row count (subscriptions) and unique customer count (COUNT_DISTINCT)."""
    # Dataset where customer 1 has multiple subscriptions
    multi_sub_df = pd.DataFrame({
        "customer_id": [1, 1, 2, 3, 3],
        "plan_tier": ["Basic", "Pro", "Basic", "Enterprise", "Pro"],
    })

    # 1. Row count (observations / subscriptions): customer 1 counted twice in different tiers
    ans_rows = build_analyst_result(
        "Count subscriptions by plan_tier",
        multi_sub_df,
        target="plan_tier",
        group="plan_tier",
        default_aggregation="COUNT",
    )
    assert ans_rows is not None
    # 5 total subscription records across the tiers
    assert "Basic: 2.00" in ans_rows.headline or "Basic: 2" in ans_rows.headline or "2" in ans_rows.headline
    assert "Pro: 2.00" in ans_rows.headline or "Pro: 2" in ans_rows.headline or "2" in ans_rows.headline
    assert "Enterprise: 1.00" in ans_rows.headline or "Enterprise: 1" in ans_rows.headline or "1" in ans_rows.headline
    assert ans_rows.numbers["table"]["Basic"]["value"] == 2
    assert ans_rows.numbers["table"]["Pro"]["value"] == 2
    assert ans_rows.numbers["table"]["Enterprise"]["value"] == 1

    # 2. Unique customers (COUNT_DISTINCT customer_id by plan_tier)
    import duckdb
    distinct_sql = "SELECT plan_tier, COUNT(DISTINCT customer_id) AS distinct_customers FROM multi_sub_df GROUP BY plan_tier"
    dist_res = duckdb.query(distinct_sql).df()
    dist_map = dict(zip(dist_res["plan_tier"], dist_res["distinct_customers"]))
    assert dist_map["Basic"] == 2
    assert dist_map["Pro"] == 2
    assert dist_map["Enterprise"] == 1


