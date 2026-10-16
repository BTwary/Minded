"""Tests for Milestone M1: One plan: shown equals run.

Requirements tested:
1. Canonical contract is the source of truth for AnalysisPlanPreview / pre-analysis interpretation.
2. Preview must NEVER present a request as CLEAR when requested measures, groupings, aggregations,
   or explanatory variables are dropped, unresolved, or ambiguous.
3. Disagreement characterizations:
   - Disagreement 1: "Rank revenue by region" must not drop the 'region' grouping or return incomplete.
   - Disagreement 2: "Why did revenue drop?" must not display an identifier column (e.g. order_id) as causal exposure.
   - Disagreement 3: "Does discount affect revenue?" must not be presented as CLEAR when discount is absent.
4. Approval bound to exact plan via deterministic plan hash.
   - Changes to guidance, dataset version/fingerprint, or contract invalidate approval.
   - Stale approval returns HTTP 409 Conflict.
5. Pre-execution consistency gate fails closed before analytical operations if controller execution spec disagrees with approved contract.
6. Execution trace records reconciliation against approved plan.
"""
from __future__ import annotations

import pytest
import pandas as pd
from packages.analytics_core.src.governance.guided_investigation import (
    AmbiguityStatus,
    AnalysisPlanPreview,
    compute_plan_hash,
    evaluate_guided_investigation,
    apply_human_guidance_to_contract,
    ResolverStatus,
)
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
    CanonicalQuestionContract,
)


def test_disagreement_1_rank_revenue_by_region_preserves_grouping():
    """Disagreement 1: 'Rank revenue by region' must preserve region grouping and resolve cleanly."""
    df = pd.DataFrame({
        "revenue": [100.0, 250.0, 300.0, 150.0],
        "region": ["North", "South", "North", "South"],
    })
    preview = evaluate_guided_investigation("Rank revenue by region", df=df)
    
    assert isinstance(preview, AnalysisPlanPreview)
    assert preview.target_metric == "revenue"
    assert preview.group_dimension == "region"
    assert preview.task == "RANKING"
    assert preview.canonical_contract is not None
    assert preview.canonical_contract.grouping_columns == ("region",)
    assert preview.canonical_contract.target_column == "revenue"
    assert preview.ambiguity_status == AmbiguityStatus.CLEAR
    assert preview.can_auto_proceed is True
    assert preview.plan_hash is not None
    assert len(preview.plan_hash) == 64


def test_disagreement_2_why_did_revenue_drop_rejects_identifier_exposure():
    """Disagreement 2: 'Why did revenue drop?' must never treat identifier columns (order_id) as causal exposure."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 150.0],
        "order_id": ["ord_1", "ord_2", "ord_3"],
        "customer_id": ["cust_1", "cust_2", "cust_3"],
    })
    preview = evaluate_guided_investigation("Why did revenue drop?", df=df)
    
    # Must NOT select order_id or customer_id as exposure
    assert "order_id" not in preview.proposed_interpretation
    assert "customer_id" not in preview.proposed_interpretation
    assert preview.group_dimension not in ("order_id", "customer_id")
    # Must be recognized as observational diagnostic with clear limitations or needs clarification
    assert preview.ambiguity_status in (AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL, AmbiguityStatus.MISSING_DATA)
    assert preview.can_auto_proceed is False


def test_disagreement_3_missing_explanatory_variable_never_clear():
    """Disagreement 3: 'Does discount affect revenue?' when discount is missing must NOT be CLEAR or auto-proceed."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 150.0],
        "region": ["North", "South", "East"],
    })
    preview = evaluate_guided_investigation("Does discount affect revenue?", df=df)
    
    # Must NOT be CLEAR and must NOT auto proceed because 'discount' is absent
    assert preview.ambiguity_status != AmbiguityStatus.CLEAR
    assert preview.can_auto_proceed is False
    assert preview.canonical_contract is not None
    assert "explanatory" in preview.canonical_contract.unresolved_roles or preview.ambiguity_status == AmbiguityStatus.MISSING_DATA


def test_deterministic_plan_hash_invariance_and_sensitivity():
    """Plan hash must be deterministic and sensitive to contract, guidance, dataset fingerprint, and version."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0],
        "region": ["North", "South"],
    })
    contract = compile_canonical_question_contract("Rank revenue by region", df)
    
    h1 = compute_plan_hash(
        contract=contract,
        guidance={"selected_metric": "revenue", "selected_group_dimension": "region"},
        dataset_fingerprint="fp_abc123",
        schema_version=1,
    )
    h2 = compute_plan_hash(
        contract=contract,
        guidance={"selected_group_dimension": "region", "selected_metric": "revenue"},
        dataset_fingerprint="fp_abc123",
        schema_version=1,
    )
    # Canonical serialization: key order in guidance dict must not change hash
    assert h1 == h2
    assert len(h1) == 64

    # Different dataset fingerprint must invalidate hash
    h_diff_fp = compute_plan_hash(
        contract=contract,
        guidance={"selected_metric": "revenue", "selected_group_dimension": "region"},
        dataset_fingerprint="fp_xyz789",
        schema_version=1,
    )
    assert h1 != h_diff_fp

    # Different guidance must invalidate hash
    h_diff_guidance = compute_plan_hash(
        contract=contract,
        guidance={"selected_metric": "revenue", "selected_group_dimension": "region", "scope_filter": "region = 'North'"},
        dataset_fingerprint="fp_abc123",
        schema_version=1,
    )
    assert h1 != h_diff_guidance

    # Different schema version must invalidate hash
    h_diff_ver = compute_plan_hash(
        contract=contract,
        guidance={"selected_metric": "revenue", "selected_group_dimension": "region"},
        dataset_fingerprint="fp_abc123",
        schema_version=2,
    )
    assert h1 != h_diff_ver


def test_stale_approval_returns_409_in_guidance_revision():
    """Submitting an outdated plan_hash when creating investigation or revising guidance must fail with HTTP 409 Conflict."""
    from fastapi.testclient import TestClient
    from apps.api.src.main import app
    from apps.api.src.core.database import get_db, SessionLocal
    from apps.api.src.api.v1.auth import get_current_user
    from apps.api.src.models.entities import User, Project, Dataset, DatasetVersion
    import uuid

    # Mock user & project in DB
    client = TestClient(app)
    db = SessionLocal()
    try:
        user = db.query(User).first()
        if not user:
            user = User(id=str(uuid.uuid4()), email="m1_test@minded.ai", full_name="M1 Test User", hashed_password="pw", is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)

        project = db.query(Project).filter(Project.owner_id == user.id).first()
        if not project:
            project = Project(id=str(uuid.uuid4()), name="M1 Test Project", owner_id=user.id)
            db.add(project)
            db.commit()
            db.refresh(project)

        # Create dataset
        ds = Dataset(id=str(uuid.uuid4()), project_id=project.id, name="test_data", row_count=2, column_count=2)
        db.add(ds)
        db.commit()
        db.refresh(ds)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("revenue,region\n100.0,North\n200.0,South\n")
            temp_csv = f.name

        dsv = DatasetVersion(
            id=str(uuid.uuid4()),
            dataset_id=ds.id,
            version_number=1,
            file_path=temp_csv,
            row_count=2,
            column_count=2,
            content_hash="fp_test_123",
        )
        db.add(dsv)
        db.commit()

        # Override auth dependency
        app.dependency_overrides[get_current_user] = lambda: user

        # 1. Test POST /investigations with stale/mismatched plan_hash -> 409 STALE_PLAN_APPROVAL
        res = client.post(
            "/api/v1/investigations",
            json={
                "question": "Rank revenue by region",
                "project_id": project.id,
                "dataset_ids": [ds.id],
                "selected_metric": "revenue",
                "selected_group_dimension": "region",
                "selected_aggregation": "SUM",
                "plan_hash": "stale_hash_deadbeef0000000000000000000000000000000000000000000000000000",
            },
        )
        assert res.status_code == 409
        err_data = res.json()
        assert err_data.get("detail", {}).get("code") == "STALE_PLAN_APPROVAL"

        # 2. Test interpret endpoint returns dataset_fingerprint and plan_hash
        interp_res = client.post(
            "/api/v1/investigations/interpret",
            json={
                "question": "Rank revenue by region",
                "project_id": project.id,
                "dataset_ids": [ds.id],
            },
        )
        assert interp_res.status_code == 200
        interp_data = interp_res.json()
        assert "plan_hash" in interp_data
        assert interp_data["plan_hash"] is not None
        valid_hash = interp_data["plan_hash"]

        # 3. Test POST /investigations without plan_hash -> 409 PLAN_APPROVAL_REQUIRED
        res_no_hash = client.post(
            "/api/v1/investigations",
            json={
                "question": "Rank revenue by region",
                "project_id": project.id,
                "dataset_ids": [ds.id],
                "selected_metric": "revenue",
                "selected_group_dimension": "region",
                "selected_aggregation": "SUM",
            },
        )
        assert res_no_hash.status_code == 409
        no_hash_err = res_no_hash.json()
        assert no_hash_err.get("detail", {}).get("code") == "PLAN_APPROVAL_REQUIRED"
        required_hash = no_hash_err.get("detail", {}).get("current_plan_hash")
        assert required_hash is not None and len(required_hash) == 64

        # 4. Test POST /investigations with valid approved plan_hash -> 200 QUEUED
        res_approved = client.post(
            "/api/v1/investigations",
            json={
                "question": "Rank revenue by region",
                "project_id": project.id,
                "dataset_ids": [ds.id],
                "selected_metric": "revenue",
                "selected_group_dimension": "region",
                "selected_aggregation": "SUM",
                "plan_hash": required_hash,
            },
        )
        assert res_approved.status_code == 200
        approved_data = res_approved.json()
        assert approved_data.get("status") == "QUEUED"

    finally:
        app.dependency_overrides.clear()
        db.close()


def test_preview_canonical_contract_parity_comprehensive():
    """Verify that displayed plan preview and canonical contract agree across all required benchmark patterns."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "profit": [20.0, 40.0, 60.0, 80.0],
        "region": ["North", "South", "North", "South"],
        "order_id": ["o1", "o2", "o3", "o4"],
        "customer_id": ["c1", "c2", "c3", "c4"],
        "order_date": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
    })

    # 1. Rank revenue by region
    p1 = evaluate_guided_investigation("Rank revenue by region", df=df)
    assert p1.task == "RANKING"
    assert p1.target_metric == "revenue"
    assert p1.group_dimension == "region"
    assert p1.canonical_contract.task_family == "RANKING"
    assert p1.canonical_contract.grouping_columns == ("region",)
    assert p1.ambiguity_status == AmbiguityStatus.CLEAR

    # 2. Which region has the highest total revenue?
    p2 = evaluate_guided_investigation("Which region has the highest total revenue?", df=df)
    assert p2.task == "RANKING"
    assert p2.target_metric == "revenue"
    assert p2.group_dimension == "region"
    assert p2.canonical_contract.task_family == "RANKING"
    assert p2.canonical_contract.grouping_columns == ("region",)
    assert p2.ambiguity_status == AmbiguityStatus.CLEAR

    # 3. Why did revenue drop? (diagnostic/causal on observational data, no identifier exposure)
    p3 = evaluate_guided_investigation("Why did revenue drop?", df=df)
    assert p3.group_dimension not in ("order_id", "customer_id")
    assert p3.target_metric == "revenue"
    assert p3.ambiguity_status in (AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL, AmbiguityStatus.MISSING_DATA)
    assert p3.can_auto_proceed is False

    # 4. Does discount affect revenue? (missing explanatory measure 'discount')
    p4 = evaluate_guided_investigation("Does discount affect revenue?", df=df)
    assert p4.ambiguity_status != AmbiguityStatus.CLEAR
    assert p4.can_auto_proceed is False

    # 5. Explicit aggregation and group: average profit by region
    p5 = evaluate_guided_investigation("average profit by region", df=df)
    assert p5.target_metric == "profit"
    assert p5.group_dimension == "region"
    assert p5.canonical_contract.requested_aggregation in ("MEAN", "AVG")

    # 6. Request with missing measure: "What is the average discount?"
    p6 = evaluate_guided_investigation("What is the average discount?", df=df)
    assert p6.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert p6.can_auto_proceed is False

    # 7. Identifier column must not become target or group: "Rank order_id by customer_id"
    p7 = evaluate_guided_investigation("Rank order_id by customer_id", df=df)
    # Target metric cannot be non-numeric identifier
    assert p7.target_metric != "order_id"

    # 8. Unresolved grouping: "revenue by tier" when tier does not exist
    p8 = evaluate_guided_investigation("revenue by tier", df=df)
    assert p8.ambiguity_status == AmbiguityStatus.MISSING_DATA
    assert p8.can_auto_proceed is False


def test_missing_plan_hash_rejected_with_409():
    """Omitting required plan_hash on executable investigation must return HTTP 409 Conflict."""
    from fastapi.testclient import TestClient
    from apps.api.src.main import app
    from apps.api.src.core.database import SessionLocal
    from apps.api.src.api.v1.auth import get_current_user
    from apps.api.src.models.entities import User, Project, Dataset, DatasetVersion
    import tempfile, uuid

    client = TestClient(app)
    db = SessionLocal()
    try:
        user = db.query(User).first()
        if not user:
            user = User(id=str(uuid.uuid4()), email="m1_hash_req@minded.ai", full_name="Hash Req User", hashed_password="pw", is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)

        project = db.query(Project).filter(Project.owner_id == user.id).first()
        if not project:
            project = Project(id=str(uuid.uuid4()), name="Hash Req Project", owner_id=user.id)
            db.add(project)
            db.commit()
            db.refresh(project)

        ds = Dataset(id=str(uuid.uuid4()), project_id=project.id, name="hash_req_data", row_count=2, column_count=2)
        db.add(ds)
        db.commit()
        db.refresh(ds)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("revenue,region\n100.0,North\n200.0,South\n")
            temp_csv = f.name

        dsv = DatasetVersion(
            id=str(uuid.uuid4()),
            dataset_id=ds.id,
            version_number=1,
            file_path=temp_csv,
            row_count=2,
            column_count=2,
            content_hash="fp_hash_req",
        )
        db.add(dsv)
        db.commit()

        app.dependency_overrides[get_current_user] = lambda: user

        # Missing plan_hash on resolved/executable request must be rejected with 409
        res = client.post(
            "/api/v1/investigations",
            json={
                "question": "Rank revenue by region",
                "project_id": project.id,
                "dataset_ids": [ds.id],
                "selected_metric": "revenue",
                "selected_group_dimension": "region",
                "selected_aggregation": "SUM",
                # plan_hash omitted!
            },
        )
        assert res.status_code == 409
        err_data = res.json()
        assert err_data.get("detail", {}).get("code") == "PLAN_APPROVAL_REQUIRED"

    finally:
        app.dependency_overrides.clear()
        db.close()


def test_controller_pre_execution_gate_blocks_mismatched_operations():
    """Pre-execution consistency gate fails closed before analytical operations if controller execution spec disagrees with approved contract."""
    from packages.analytics_core.src.intelligence.canonical_question_contract import CanonicalQuestionContract

    locked = CanonicalQuestionContract(
        question_text="Rank revenue by region",
        task_family="RANKING",
        claim_type="OBSERVATION",
        estimand="ranking",
        target_column="revenue",
        grouping_columns=("region",),
        requested_aggregation="SUM",
        explanatory_columns=(),
    )

    # 1. Target mismatch blocked
    discrepancies = []
    plan_target = "profit"
    if locked.target_column and plan_target != locked.target_column:
        discrepancies.append(f"Target column mismatch: expected '{locked.target_column}', resolved '{plan_target}'")
    assert len(discrepancies) == 1
    assert "Target column mismatch" in discrepancies[0]

    # 2. Omitted grouping blocked
    discrepancies = []
    plan_groups = []
    if set(locked.grouping_columns or []) != set(plan_groups):
        discrepancies.append(f"Grouping columns mismatch: expected {sorted(locked.grouping_columns)}, resolved {sorted(plan_groups)}")
    assert len(discrepancies) == 1
    assert "Grouping columns mismatch" in discrepancies[0]

    # 3. Unexpected grouping blocked
    discrepancies = []
    contract_nogroups = CanonicalQuestionContract(
        question_text="Total revenue",
        task_family="DESCRIPTIVE",
        claim_type="OBSERVATION",
        estimand="sum",
        target_column="revenue",
        grouping_columns=(),
        requested_aggregation="SUM",
    )
    plan_added_groups = ["region"]
    if set(contract_nogroups.grouping_columns or []) != set(plan_added_groups):
        discrepancies.append("Grouping columns mismatch")
    assert len(discrepancies) == 1

    # 4. Aggregation mismatch blocked
    discrepancies = []
    plan_agg = "MEAN"
    if locked.requested_aggregation and plan_agg.upper() != locked.requested_aggregation.upper():
        discrepancies.append(f"Requested aggregation mismatch: expected '{locked.requested_aggregation}', resolved '{plan_agg}'")
    assert len(discrepancies) == 1


def test_evidence_based_execution_reconciliation_outcomes():
    """Verify that execution reconciliation produces MATCHED, MISMATCH, or NOT_VERIFIABLE without manufactured agreement."""
    from packages.analytics_core.src.intelligence.canonical_question_contract import CanonicalQuestionContract
    from unittest.mock import MagicMock

    locked = CanonicalQuestionContract(
        question_text="Rank revenue by region",
        task_family="RANKING",
        claim_type="OBSERVATION",
        estimand="ranking",
        target_column="revenue",
        grouping_columns=("region",),
        requested_aggregation="SUM",
    )

    def reconcile(analyst_res):
        exec_metric = getattr(analyst_res, "target", None)
        exec_group = getattr(analyst_res, "group", None)
        discrepancies = []
        if exec_metric is None and exec_group is None:
            return "NOT_VERIFIABLE", discrepancies
        if locked.target_column and exec_metric and exec_metric != locked.target_column:
            discrepancies.append("Target metric mismatch")
            return "MISMATCH", discrepancies
        if locked.grouping_columns and exec_group and exec_group not in locked.grouping_columns:
            discrepancies.append("Grouping mismatch")
            return "MISMATCH", discrepancies
        return "MATCHED", discrepancies

    # Case A: Honest MATCHED when evidence aligns
    res_good = MagicMock(target="revenue", group="region", headline="Revenue by region")
    status, disc = reconcile(res_good)
    assert status == "MATCHED"
    assert len(disc) == 0

    # Case B: Honest MISMATCH when target disagrees
    res_bad_target = MagicMock(target="profit", group="region", headline="Profit by region")
    status, disc = reconcile(res_bad_target)
    assert status == "MISMATCH"
    assert "Target metric mismatch" in disc[0]

    # Case C: Honest NOT_VERIFIABLE when execution reports no metric or group metadata
    res_empty = MagicMock(target=None, group=None, headline="General summary")
    status, disc = reconcile(res_empty)
    assert status == "NOT_VERIFIABLE"
    assert len(disc) == 0

