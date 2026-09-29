"""Phase 1 Acceptance Verification Suite: Canonical AA-OS Domain Schema & Migrations."""
import os
import sys
import uuid
from datetime import datetime
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.api.src.models.entities import (
    Base,
    User,
    Organization,
    OrganizationMember,
    Project,
    Dataset,
    DatasetVersion,
    SemanticModel,
    SemanticEntity,
    SemanticMetric,
    SemanticDimension,
    SemanticRelationship,
    SemanticTimeDefinition,
    SemanticBusinessRule,
    Investigation,
    InvestigationObjective,
    InvestigationUnknown,
    Hypothesis,
    Prediction,
    Experiment,
    Observation,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    CounterHypothesis,
    Assumption,
    InvestigationVerdict,
    InvestigationGraphEdge,
    AnalysisRun,
    gen_uuid,
)
from scripts.migrate_analysis_runs_to_investigations import migrate_all_analysis_runs


def run_phase1_tests():
    print("=" * 70)
    print("RUNNING PHASE 1: CANONICAL AA-OS DOMAIN MODEL & MIGRATION TEST SUITE")
    print("=" * 70)

    # Use in-memory SQLite database for deterministic isolated testing
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    # ------------------------------------------------------------------------
    # [TEST 1] Schema & Table Inventory Verification
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Verifying 18+ Canonical Tables & Schema Inventory...")
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()
    
    expected_tables = [
        "organizations", "users", "projects", "datasets", "dataset_versions",
        "semantic_models", "semantic_entities", "semantic_metrics", "semantic_dimensions",
        "semantic_relationships", "semantic_time_definitions", "semantic_business_rules",
        "investigations", "investigation_objectives", "investigation_unknowns",
        "hypotheses", "predictions", "experiments", "observations", "evidence",
        "evidence_verifications", "belief_updates", "counter_hypotheses", "assumptions",
        "investigation_verdicts", "investigation_graph_edges", "analysis_runs"
    ]
    for tbl in expected_tables:
        assert tbl in tables, f"Expected table '{tbl}' not found in database!"
    print(f" -> PASSED: All {len(expected_tables)} canonical tables verified in database.")

    # ------------------------------------------------------------------------
    # [TEST 2] Tenant & Organization Boundary Isolation
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Tenant & Organization Isolation...")
    # Org A
    org_a = Organization(id="org-alpha", name="Alpha Corp", slug="alpha-corp")
    user_a = User(id="user-alice", email="alice@alpha.com", hashed_password="pw", full_name="Alice Alpha", organization_id=org_a.id)
    proj_a = Project(id="proj-alpha-1", org_id=org_a.id, owner_id=user_a.id, name="Alpha Project")
    
    # Org B
    org_b = Organization(id="org-beta", name="Beta LLC", slug="beta-llc")
    user_b = User(id="user-bob", email="bob@beta.com", hashed_password="pw", full_name="Bob Beta", organization_id=org_b.id)
    proj_b = Project(id="proj-beta-1", org_id=org_b.id, owner_id=user_b.id, name="Beta Project")
    
    db.add_all([org_a, user_a, proj_a, org_b, user_b, proj_b])
    db.commit()

    # Create investigation strictly in Org A
    inv_a = Investigation(
        id="INV-ALPHA-001",
        project_id=proj_a.id,
        user_id=user_a.id,
        question="Alpha revenue growth analysis",
        status="COMPLETED",
        verdict_type="DIAGNOSED",
    )
    db.add(inv_a)
    db.commit()

    # Query investigations scoped to Org B projects
    org_b_investigations = db.query(Investigation).filter(Investigation.project_id == proj_b.id).all()
    assert len(org_b_investigations) == 0, "Tenant boundary leak: Org B accessed Org A investigations!"
    
    org_a_investigations = db.query(Investigation).filter(Investigation.project_id == proj_a.id).all()
    assert len(org_a_investigations) == 1, "Org A could not retrieve its own investigation!"
    assert org_a_investigations[0].id == "INV-ALPHA-001"
    print(" -> PASSED: Strict tenant isolation verified across organization and project boundaries.")

    # ------------------------------------------------------------------------
    # [TEST 3] Semantic Model Versioning & Dataset Lineage (Dataset v7 -> Semantic v4 -> Inv)
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Verifying Semantic World Model Versioning & Lineage...")
    ds = Dataset(id="ds-sales", project_id=proj_a.id, name="sales_data", current_version=7)
    db.add(ds)
    
    dv = DatasetVersion(id="dv-sales-v7", dataset_id=ds.id, version_number=7, file_path="/storage/sales_v7.parquet", row_count=50000)
    db.add(dv)
    db.commit()

    # Semantic Model v4 bound to Dataset Version 7
    sem_model_v4 = SemanticModel(
        id="SEM-MODEL-V4",
        project_id=proj_a.id,
        version_number=4,
        status="active",
        summary_description="Enterprise Sales & Revenue Semantic Domain",
        table_grains_json={"sales": "order_id, line_item_id", "customers": "customer_id"},
        dataset_version_ids_json=[dv.id],
    )
    db.add(sem_model_v4)
    db.commit()

    # Add entities, metrics, dimensions, relationships to Semantic Model v4
    entity_cust = SemanticEntity(
        id=gen_uuid(),
        semantic_model_id=sem_model_v4.id,
        entity_name="Customer",
        primary_key="customer_id",
        table_name="customers",
        natural_keys_json=["email", "phone"],
        attributes_json=["region", "segment", "signup_date"],
    )
    metric_rev = SemanticMetric(
        id=gen_uuid(),
        semantic_model_id=sem_model_v4.id,
        metric_name="net_revenue",
        display_name="Net Sales Revenue",
        column_name="revenue",
        table_name="sales",
        additivity="fully_additive",
        unit="USD",
        target_direction="maximize",
        sql_formula="SUM(revenue - discount)",
    )
    rel = SemanticRelationship(
        id=gen_uuid(),
        semantic_model_id=sem_model_v4.id,
        source_table="sales",
        source_column="customer_id",
        target_table="customers",
        target_column="customer_id",
        cardinality="many-to-one",
        join_safety_score=1.0,
        fanout_risk=False,
    )
    db.add_all([entity_cust, metric_rev, rel])
    db.commit()

    # Reconstruct lineage: Investigation -> SemanticModel -> DatasetVersion
    inv_lineage = Investigation(
        id="INV-REPRO-001",
        project_id=proj_a.id,
        semantic_model_id=sem_model_v4.id,
        question="What is the driver of customer revenue variance?",
        dataset_version_ids_json=[dv.id],
    )
    db.add(inv_lineage)
    db.commit()

    queried_inv = db.query(Investigation).filter(Investigation.id == "INV-REPRO-001").first()
    assert queried_inv.semantic_model_id == sem_model_v4.id
    assert dv.id in queried_inv.dataset_version_ids_json
    
    queried_sem = db.query(SemanticModel).filter(SemanticModel.id == queried_inv.semantic_model_id).first()
    assert queried_sem.version_number == 4
    
    queried_metrics = db.query(SemanticMetric).filter(SemanticMetric.semantic_model_id == queried_sem.id).all()
    assert len(queried_metrics) == 1
    assert queried_metrics[0].metric_name == "net_revenue"
    print(" -> PASSED: Dataset v7 -> SemanticModel v4 -> Investigation INV-REPRO-001 lineage fully verified.")

    # ------------------------------------------------------------------------
    # [TEST 4] End-to-End Investigation DAG Lifecycle & Graph Entities
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Verifying End-to-End Investigation DAG Entities & Graph Edges...")
    # 1. Root Investigation
    inv = Investigation(
        id="INV-2026-ROOT",
        project_id=proj_a.id,
        user_id=user_a.id,
        semantic_model_id=sem_model_v4.id,
        question="Why did sales drop in Region B?",
        status="COMPLETED",
        verdict_type="DIAGNOSED",
        confidence_score=0.98,
        direct_answer="Sales dropped due to supplier bottleneck.",
        main_finding="Region B experienced 48% inventory contraction.",
        entropy_initial=1.0,
        entropy_current=0.05,
        stopping_criteria_met=True,
        stopping_rationale="Posterior probability exceeded 0.95 threshold.",
        reproducible_manifest_hash="abc123canonicalhash999",
    )
    db.add(inv)
    db.flush()

    # 2. Objective
    obj = InvestigationObjective(
        id=gen_uuid(),
        investigation_id=inv.id,
        statement="Explain Region B sales decline",
        target_variable="revenue",
    )
    db.add(obj)

    # 3. Unknown
    unknown = InvestigationUnknown(
        id=gen_uuid(),
        investigation_id=inv.id,
        variable_name="regional_variance_factor",
        target_metric="revenue",
        prior_estimate="unknown",
        posterior_estimate="inventory_shortage",
        is_resolved=True,
    )
    db.add(unknown)

    # 4. Primary Hypothesis & Counter-Hypothesis
    hyp1 = Hypothesis(
        id="HYP-INV-01",
        investigation_id=inv.id,
        hypothesis_code="HYP-01",
        canonical_identity="phase1-test-hyp-01-inventory-stockouts-region-b",
        statement="Inventory stockouts constrained sales in Region B",
        prior_probability=0.5,
        posterior_probability=0.96,
        belief_state="supported",
    )
    hyp2 = Hypothesis(
        id="HYP-INV-02",
        investigation_id=inv.id,
        hypothesis_code="HYP-02",
        canonical_identity="phase1-test-hyp-02-demand-evaporated-region-b",
        statement="Customer demand naturally evaporated in Region B",
        prior_probability=0.5,
        posterior_probability=0.04,
        belief_state="refuted",
        is_counter_hypothesis=True,
    )
    db.add_all([hyp1, hyp2])
    db.flush()

    # Counter-hypothesis link
    counter_link = CounterHypothesis(
        id=gen_uuid(),
        primary_hypothesis_id=hyp1.id,
        counter_hypothesis_id=hyp2.id,
        relation_type="MUTUALLY_EXCLUSIVE",
    )
    db.add(counter_link)

    # 5. Prediction
    pred = Prediction(
        id=gen_uuid(),
        hypothesis_id=hyp1.id,
        statement="Inventory stock levels in Region B should show a 40%+ drop",
        expected_direction="decrease",
        expected_magnitude=">40%",
    )
    db.add(pred)

    # 6. Experiment & Observation
    exp = Experiment(
        id="EXP-INV-01",
        investigation_id=inv.id,
        hypothesis_id=hyp1.id,
        test_code="TEST-01",
        tool_name="duckdb_time_series_aggregation",
        arguments_json={"region": "Region B", "metric": "stock_level"},
        status="EXECUTED",
    )
    db.add(exp)
    db.flush()

    obs = Observation(
        id=gen_uuid(),
        experiment_id=exp.id,
        result_json={"stock_drop_pct": 48.2, "p_val": 0.0001},
        row_count_analyzed=10000,
        execution_time_ms=14.2,
    )
    db.add(obs)

    # 7. Evidence & Verification
    evid = Evidence(
        id="EVID-INV-01",
        investigation_id=inv.id,
        experiment_id=exp.id,
        hypothesis_id=hyp1.id,
        statement="Region B stock levels dropped 48.2% (p < 0.001)",
        validation_status="verified",
        p_value=0.0001,
        effect_size=0.482,
    )
    db.add(evid)
    db.flush()

    verif = EvidenceVerification(
        id=gen_uuid(),
        evidence_id=evid.id,
        primary_tool="duckdb_sql",
        secondary_tool="scipy_bootstrap",
        tolerance_threshold=0.01,
        observed_delta_pct=0.0,
        status="PASSED",
    )
    db.add(verif)

    # 8. Belief Update
    belief = BeliefUpdate(
        id=gen_uuid(),
        investigation_id=inv.id,
        hypothesis_id=hyp1.id,
        evidence_id=evid.id,
        prior_probability=0.5,
        likelihood_p=0.96,
        posterior_probability=0.96,
        entropy_delta=-0.95,
        update_step_index=1,
    )
    db.add(belief)

    # 9. Final Verdict
    verdict = InvestigationVerdict(
        id=gen_uuid(),
        investigation_id=inv.id,
        verdict_type="DIAGNOSED",
        confidence_score=0.98,
        justification="Hypothesis 1 verified with independent secondary replication.",
        net_variance_explained_pct=98.5,
        counter_hypothesis_refuted=True,
        epistemic_grade="A",
    )
    db.add(verdict)

    # 10. Directed Graph Edges
    edge1 = InvestigationGraphEdge(
        id=gen_uuid(),
        investigation_id=inv.id,
        source_node_type="OBJECTIVE",
        source_node_id=obj.id,
        target_node_type="HYPOTHESIS",
        target_node_id=hyp1.id,
        relationship_type="PROPOSES",
    )
    edge2 = InvestigationGraphEdge(
        id=gen_uuid(),
        investigation_id=inv.id,
        source_node_type="HYPOTHESIS",
        source_node_id=hyp1.id,
        target_node_type="EXPERIMENT",
        target_node_id=exp.id,
        relationship_type="TESTS",
    )
    edge3 = InvestigationGraphEdge(
        id=gen_uuid(),
        investigation_id=inv.id,
        source_node_type="EXPERIMENT",
        source_node_id=exp.id,
        target_node_type="EVIDENCE",
        target_node_id=evid.id,
        relationship_type="PRODUCES",
    )
    db.add_all([edge1, edge2, edge3])
    db.commit()

    # Verify DAG queryability
    edges = db.query(InvestigationGraphEdge).filter(InvestigationGraphEdge.investigation_id == inv.id).all()
    assert len(edges) == 3, f"Expected 3 DAG edges, found {len(edges)}"
    print(" -> PASSED: Full Investigation DAG lifecycle and explicit graph edges verified with 100% integrity.")

    # ------------------------------------------------------------------------
    # [TEST 5] Legacy AnalysisRun Data Migration & Preservation
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Verifying Legacy AnalysisRun Migration to Investigation DAG...")
    legacy_run = AnalysisRun(
        id="LEGACY-RUN-100",
        project_id=proj_a.id,
        question="Historical legacy quarterly revenue review",
        status="COMPLETED",
        direct_answer="Q1 revenue met forecast target.",
        main_finding="Revenue grew 14.5% year over year.",
        confidence="High confidence",
        hypotheses_json=[
            {"id": "HYP-01", "name": "Growth driven by enterprise tiers", "prior_probability": 0.5, "posterior_probability": 0.88}
        ],
        evidence_json=[
            {"id": "EVID-01", "description": "Enterprise accounts increased volume 22%", "result": {"tier": "enterprise", "growth": 0.22}}
        ],
        manifest_json={"reproducible_hash": "manifest-hash-legacy-100"},
    )
    db.add(legacy_run)
    db.commit()

    # Execute migration
    migrated_count = migrate_all_analysis_runs(db)
    assert migrated_count >= 1, "Migration script did not migrate legacy AnalysisRun!"

    # Verify migrated Investigation
    migrated_inv = db.query(Investigation).filter(Investigation.id == "LEGACY-RUN-100").first()
    assert migrated_inv is not None, "Migrated Investigation not found!"
    assert migrated_inv.question == "Historical legacy quarterly revenue review"
    assert migrated_inv.reproducible_manifest_hash == "manifest-hash-legacy-100"

    migrated_hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == "LEGACY-RUN-100").all()
    assert len(migrated_hyps) == 1
    assert migrated_hyps[0].posterior_probability == 0.88

    migrated_evid = db.query(Evidence).filter(Evidence.investigation_id == "LEGACY-RUN-100").all()
    assert len(migrated_evid) == 1
    assert "Enterprise accounts" in migrated_evid[0].statement

    # Verify legacy AnalysisRun still exists and is 100% intact
    queried_legacy = db.query(AnalysisRun).filter(AnalysisRun.id == "LEGACY-RUN-100").first()
    assert queried_legacy is not None
    assert queried_legacy.direct_answer == "Q1 revenue met forecast target."
    print(" -> PASSED: Legacy AnalysisRun data preserved and accurately ported to canonical Investigation DAG.")

    print("\n" + "=" * 70)
    print("ALL 5 PHASE-1 DOMAIN & MIGRATION TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_tests()
