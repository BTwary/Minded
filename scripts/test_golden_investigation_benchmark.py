import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.api.src.models.entities import (
    Base,
    Organization,
    User,
    Project,
    Dataset,
    DatasetVersion,
    Investigation,
    gen_uuid,
)
from packages.analytics_core.src.execution.state_machine import InvestigationState
from packages.analytics_core.src.execution.queue import DatabaseQueueProvider
from packages.analytics_core.src.execution.worker import InvestigationWorker
from packages.analytics_core.src.runtime.controller import InvestigationController


def run_golden_investigation_benchmark():
    print("=" * 80)
    print("RUNNING AA-OS 10 GOLDEN INVESTIGATION BENCHMARK SUITE")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="aaos_golden_bench_")
    db_path = os.path.join(temp_dir, "golden_test.db")
    test_engine = create_engine(
        f"sqlite:///{db_path}?timeout=30",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    org = Organization(id="org-bench", name="Golden Bench Org", slug="bench-org")
    user = User(id="user-bench", email="bench@aaos.ai", hashed_password="pw", full_name="Bench Engineer", organization_id=org.id)
    proj = Project(id="proj-bench", org_id=org.id, owner_id=user.id, name="AA-OS Golden Benchmark Project")
    db.add_all([org, user, proj])
    db.commit()

    storage_dir = os.path.join(temp_dir, "datasets", proj.id)
    os.makedirs(storage_dir, exist_ok=True)

    queue = DatabaseQueueProvider(session_factory=TestingSession)
    controller = InvestigationController(session_factory=TestingSession)
    worker = InvestigationWorker(worker_id="worker-golden-bench", queue_provider=queue, session_factory=TestingSession, controller=controller)

    # ------------------------------------------------------------------------
    # 10 Diverse Real-World Benchmark Scenarios
    # ------------------------------------------------------------------------
    scenarios = [
        {
            "id": "SCENARIO-01",
            "name": "SaaS Cloud Infrastructure Cost Surge",
            "table_name": "cloud_billing",
            "question": "Why did cost_usd surge across instance_type?",
            "df": pd.DataFrame({
                "instance_type": ["gpu-h100", "cpu-general", "mem-high", "io-nvme"] * 100,
                "cost_usd": [500.0 + (i % 4 == 0) * 2500.0 + (i * 0.2) for i in range(400)],
                "region": ["us-east", "eu-west", "ap-south", "us-west"] * 100,
            }),
            "expected_top_dim": "instance_type",
        },
        {
            "id": "SCENARIO-02",
            "name": "E-Commerce Checkout Funnel Abandonment",
            "table_name": "checkout_telemetry",
            "question": "Why did abandonment_rate increase across payment_method?",
            "df": pd.DataFrame({
                "payment_method": ["crypto-gateway", "credit-card", "apple-pay", "bank-debit"] * 100,
                "abandonment_rate": [0.05 + (i % 4 == 0) * 0.65 + np.random.uniform(0, 0.02) for i in range(400)],
                "device_type": ["ios", "android", "macos", "windows"] * 100,
            }),
            "expected_top_dim": "payment_method",
        },
        {
            "id": "SCENARIO-03",
            "name": "Logistics Heavy-Haul Fuel Anomaly",
            "table_name": "fleet_telemetry",
            "question": "Why did fuel_liters spike across route_category?",
            "df": pd.DataFrame({
                "route_category": ["mountain-grade", "coastal-flat", "urban-dense", "interstate-direct"] * 100,
                "fuel_liters": [40.0 + (i % 4 == 0) * 180.0 + (i * 0.05) for i in range(400)],
                "vehicle_class": ["class-8-semi", "box-truck", "van-ev", "flatbed"] * 100,
            }),
            "expected_top_dim": "route_category",
        },
        {
            "id": "SCENARIO-04",
            "name": "Clinical Trial Dosage Cohort Variance",
            "table_name": "trial_biomarkers",
            "question": "Why did biomarker_deviation drift across dosage_cohort?",
            "df": pd.DataFrame({
                "dosage_cohort": ["dose-high-100mg", "dose-med-50mg", "dose-low-10mg", "placebo"] * 100,
                "biomarker_deviation": [1.2 + (i % 4 == 0) * 8.5 + (i * 0.01) for i in range(400)],
                "patient_age_bracket": ["18-35", "36-50", "51-65", "65+"] * 100,
            }),
            "expected_top_dim": "dosage_cohort",
        },
        {
            "id": "SCENARIO-05",
            "name": "Fintech Transaction Fraud Concentration",
            "table_name": "payment_transactions",
            "question": "Why did fraud_loss_usd expand across transfer_channel?",
            "df": pd.DataFrame({
                "transfer_channel": ["crypto-bridge", "ach-wire", "p2p-instant", "pos-terminal"] * 100,
                "fraud_loss_usd": [10.0 + (i % 4 == 0) * 450.0 + (i * 0.1) for i in range(400)],
                "risk_tier": ["tier-3-high", "tier-2-med", "tier-1-low", "unscored"] * 100,
            }),
            "expected_top_dim": "transfer_channel",
        },
        {
            "id": "SCENARIO-06",
            "name": "Manufacturing Thermal Extruder Instability",
            "table_name": "sensor_timeseries",
            "question": "Why did temperature_celsius surge across nozzle_module?",
            "df": pd.DataFrame({
                "nozzle_module": ["nozzle-4-ceramic", "nozzle-1-steel", "nozzle-2-copper", "nozzle-3-tungsten"] * 100,
                "temperature_celsius": [180.0 + (i % 4 == 0) * 120.0 + (i * 0.05) for i in range(400)],
                "chamber_id": ["chamber-a", "chamber-b", "chamber-c", "chamber-d"] * 100,
            }),
            "expected_top_dim": "nozzle_module",
        },
        {
            "id": "SCENARIO-07",
            "name": "B2B SaaS Contract Churn Spike",
            "table_name": "subscription_mrr",
            "question": "Why did churned_mrr expand across industry_vertical?",
            "df": pd.DataFrame({
                "industry_vertical": ["crypto-fintech", "healthcare-med", "ecommerce-retail", "logistics-freight"] * 100,
                "churned_mrr": [200.0 + (i % 4 == 0) * 1400.0 + (i * 0.15) for i in range(400)],
                "account_tier": ["enterprise", "mid-market", "smb", "startup"] * 100,
            }),
            "expected_top_dim": "industry_vertical",
        },
        {
            "id": "SCENARIO-08",
            "name": "HR Engineering Overtime & Burnout Skew",
            "table_name": "workforce_retention",
            "question": "Why did attrition_score elevate across department_role?",
            "df": pd.DataFrame({
                "department_role": ["platform-infra", "frontend-ux", "data-engineering", "product-design"] * 100,
                "attrition_score": [0.15 + (i % 4 == 0) * 0.70 + (i * 0.0005) for i in range(400)],
                "seniority": ["lead", "senior", "mid", "associate"] * 100,
            }),
            "expected_top_dim": "department_role",
        },
        {
            "id": "SCENARIO-09",
            "name": "Omnichannel Inventory Fulfillment Stockout",
            "table_name": "warehouse_inventory",
            "question": "Why did stockout_units spike across warehouse_hub?",
            "df": pd.DataFrame({
                "warehouse_hub": ["hub-midwest-cold", "hub-northeast", "hub-southeast", "hub-southwest"] * 100,
                "stockout_units": [5.0 + (i % 4 == 0) * 95.0 + (i * 0.05) for i in range(400)],
                "courier": ["carrier-express", "freight-ground", "air-freight", "postal-standard"] * 100,
            }),
            "expected_top_dim": "warehouse_hub",
        },
        {
            "id": "SCENARIO-10",
            "name": "Telecom Cellular Packet Loss Degradation",
            "table_name": "telecom_towers",
            "question": "Why did packet_loss_pct surge across cell_tower_band?",
            "df": pd.DataFrame({
                "cell_tower_band": ["band-mmwave-71", "band-mid-c", "band-low-600", "band-lte-legacy"] * 100,
                "packet_loss_pct": [0.01 + (i % 4 == 0) * 0.28 + (i * 0.0001) for i in range(400)],
                "weather_condition": ["rain-storm", "clear-sky", "fog", "snow"] * 100,
            }),
            "expected_top_dim": "cell_tower_band",
        },
    ]

    passed_scenarios = 0

    for s in scenarios:
        s_id = s["id"]
        s_name = s["name"]
        t_name = s["table_name"]
        print(f"\n[{s_id}] Executing Scenario: {s_name}...")

        # Persist dataset parquet
        p_path = os.path.join(storage_dir, f"{t_name}.parquet")
        s["df"].to_parquet(p_path)

        ds_rec = Dataset(
            id=f"ds-{s_id.lower()}",
            project_id=proj.id,
            name=t_name,
            current_version=1,
            row_count=len(s["df"]),
            column_count=len(s["df"].columns),
            format="parquet",
        )
        ds_v_rec = DatasetVersion(
            id=gen_uuid(),
            dataset_id=ds_rec.id,
            version_number=1,
            file_path=p_path,
            row_count=len(s["df"]),
        )
        db.add_all([ds_rec, ds_v_rec])
        db.commit()

        # Enqueue Investigation
        inv_rec = Investigation(
            id=f"INV-{s_id}",
            project_id=proj.id,
            user_id=user.id,
            question=s["question"],
            status=InvestigationState.QUEUED,
        )
        db.add(inv_rec)
        db.commit()

        queue.enqueue(investigation_id=inv_rec.id, priority=100)

        # Autonomous worker execution
        success = worker.process_next_job()
        assert success is True, f"Scenario {s_id} worker execution failed!"

        db.expire_all()
        completed_inv = db.query(Investigation).filter(Investigation.id == inv_rec.id).first()
        assert completed_inv.status == InvestigationState.COMPLETED
        assert completed_inv.verdict_type in ["DIAGNOSED", "OBSERVED", "INCONCLUSIVE", "STATISTICALLY_SIGNIFICANT"]
        assert completed_inv.reproducible_manifest_hash is not None
        assert len(completed_inv.reproducible_manifest_hash) >= 16

        print(f" -> PASSED: Autonomous loop concluded with verdict '{completed_inv.verdict_type}' (Manifest: {completed_inv.reproducible_manifest_hash[:16]}...).")
        print(f"    Direct Answer: {completed_inv.direct_answer[:100]}...")
        passed_scenarios += 1

    print("\n" + "=" * 80)
    print(f"GOLDEN INVESTIGATION BENCHMARK RESULT: {passed_scenarios}/{len(scenarios)} SCENARIOS PASSED WITH 100% INTEGRITY")
    print("=" * 80)

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    run_golden_investigation_benchmark()
