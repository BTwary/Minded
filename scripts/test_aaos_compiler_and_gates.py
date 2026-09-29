import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import polars as pl
from packages.schemas.src.analysis import (
    AggregationType,
    CausalIdentifiabilityProof,
    CausalIdentifiabilityStatus,
    CausalIntent,
    FilterAST,
    FilterOperator,
    GrainPreservationStatus,
    GrainRef,
    MetricRef,
    ObjectiveType,
    TemporalScope,
    TypedAnalyticalIntent,
    VariableRef,
)
from packages.schemas.src.semantic_graph import (
    EntityNode,
    MetricAdditivity,
    MetricNode,
    SemanticWorldModelSchema,
)
from packages.analytics_core.src.validation.ir_validator import AnalyticalIRValidator
from packages.analytics_core.src.sql.relational_compiler import RelationalCompiler
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.causal.identifiability_gate import (
    CausalGraphSpec,
    CausalIdentifiabilityGate,
)


def run_tests():
    print("=" * 80)
    print("RUNNING AA-OS COMPILER, IR VALIDATOR, VERIFICATION & CAUSAL GATE SUITE")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # TEST 1: Semantic IR Validation
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Testing Deterministic Analytical IR Validator...")
    swm = SemanticWorldModelSchema(
        dataset_hash="swm_test_hash",
        entities=[
            EntityNode(entity_name="Orders", primary_key="order_id", table_name="orders")
        ],
        metrics=[
            MetricNode(metric_name="Profit Margin", table_name="orders", additivity=MetricAdditivity.NON_ADDITIVE)
        ]
    )

    validator = AnalyticalIRValidator()
    valid_intent = TypedAnalyticalIntent(
        intent_id="INT-VALID-01",
        objective=ObjectiveType.DESCRIBE,
        target_metric=MetricRef(name="Revenue", table="orders", column="amount", aggregation=AggregationType.SUM),
        unit_of_analysis=GrainRef(table="orders", keys=["order_id"]),
    )
    res_valid = validator.validate(valid_intent, swm)
    assert res_valid.is_valid is True, f"Expected valid, got: {res_valid.errors}"

    invalid_intent = TypedAnalyticalIntent(
        intent_id="INT-INVALID-01",
        objective=ObjectiveType.DESCRIBE,
        target_metric=MetricRef(name="Profit Margin", table="orders", column="margin", aggregation=AggregationType.SUM),
        unit_of_analysis=GrainRef(table="orders", keys=["order_id"]),
    )
    res_invalid = validator.validate(invalid_intent, swm)
    assert res_invalid.is_valid is False
    assert any(e.code == "ERR_INVALID_ADDITIVITY_AGGREGATION" for e in res_invalid.errors)
    print(" -> PASSED: Valid intents accepted, non-additive SUM aggregations strictly rejected.")

    # ------------------------------------------------------------------------
    # TEST 2: Relational Compilation (Logical Plan -> Physical SQL)
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Testing Relational Plan Generation & SQL Compiler...")
    compiler = RelationalCompiler()
    complex_intent = TypedAnalyticalIntent(
        intent_id="INT-COMPLEX-01",
        objective=ObjectiveType.ROOT_CAUSE,
        target_metric=MetricRef(name="Revenue", table="orders", column="amount", aggregation=AggregationType.SUM),
        unit_of_analysis=GrainRef(table="orders", keys=["order_id"]),
        dimensions=[VariableRef(table="orders", column="region")],
        population=[
            FilterAST(variable=VariableRef(table="orders", column="status"), operator=FilterOperator.EQ, value="completed"),
            FilterAST(variable=VariableRef(table="orders", column="tier"), operator=FilterOperator.IN, value=["Enterprise", "SMB"]),
        ],
        temporal_scope=TemporalScope(
            column=VariableRef(table="orders", column="order_date"),
            start="2026-01-01",
            end="2026-03-31",
        ),
    )
    plan = compiler.compile(complex_intent)
    assert "SELECT region, SUM(amount) AS Revenue FROM data_table" in plan.sql_query
    assert "status = 'completed'" in plan.sql_query
    assert "tier IN ('Enterprise', 'SMB')" in plan.sql_query
    assert "order_date >= '2026-01-01' AND order_date <= '2026-03-31'" in plan.sql_query
    assert "GROUP BY region" in plan.sql_query
    assert "ORDER BY Revenue DESC" in plan.sql_query
    assert len(plan.logical_nodes) == 5
    print(f" -> PASSED: Compiled 5 relational nodes into physical SQL: {plan.sql_query[:75]}...")

    # ------------------------------------------------------------------------
    # TEST 3: Grain Preservation & Join Fanout Gate
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Testing VerificationEngine Grain & Join Fanout Proofs...")
    df_orders = pd.DataFrame({
        "order_id": [101, 102, 103],
        "amount": [150.0, 250.0, 350.0],
    })
    df_safe = pd.DataFrame({
        "order_id": [101, 102, 103],
        "amount": [150.0, 250.0, 350.0],
        "customer": ["Alice", "Bob", "Charlie"],
    })
    safe_proof = VerificationEngine.verify_grain_preservation(df_orders, df_safe, ["order_id"])
    assert safe_proof.preservation_status == GrainPreservationStatus.PRESERVED
    assert safe_proof.fanout_factor == 1.0

    df_fanout = pd.DataFrame({
        "order_id": [101, 101, 102, 103],
        "amount": [150.0, 150.0, 250.0, 350.0],
        "item": ["A", "B", "C", "D"],
    })
    fanout_proof = VerificationEngine.verify_grain_preservation(df_orders, df_fanout, ["order_id"])
    assert fanout_proof.preservation_status == GrainPreservationStatus.FANOUT_DETECTED
    assert fanout_proof.fanout_factor > 1.0
    print(f" -> PASSED: Fanout detected on 1:N join (Fanout: {fanout_proof.fanout_factor:.2f}), safe joins preserved.")

    # ------------------------------------------------------------------------
    # TEST 4: Dual-Engine Cross-Validation (DuckDB vs Polars Rust)
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Testing Dual-Engine Secondary Cross-Computation...")
    ver_res = VerificationEngine.verify_secondary(
        primary_df=df_orders,
        target_metric_col="amount",
        aggregation_type="SUM",
        primary_metric=750.0,
        query_sql="SELECT SUM(amount) AS total FROM data_table",
        unit_of_analysis_keys=["order_id"],
    )
    assert ver_res.status == "VERIFIED"
    assert ver_res.observed_delta_pct <= 1e-4
    print(" -> PASSED: Mathematical equality verified between DuckDB and Polars SQLContext (delta = 0.000000).")

    # ------------------------------------------------------------------------
    # TEST 5: Causal Identifiability Gate (Pearl's Backdoor Criterion & E-value)
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Testing Causal Safety Gate & Backdoor Criterion...")
    gate = CausalIdentifiabilityGate()
    ci = CausalIntent(
        treatment=VariableRef(table="marketing", column="spend"),
        outcome=VariableRef(table="sales", column="revenue"),
        assumed_dag_id="dag_marketing_sales"
    )

    # Identifiable with observable confounder
    dag_valid = CausalGraphSpec(
        dag_id="dag_marketing_sales",
        nodes=["marketing.spend", "sales.revenue", "market.seasonality"],
        directed_edges=[
            ("market.seasonality", "marketing.spend"),
            ("market.seasonality", "sales.revenue"),
            ("marketing.spend", "sales.revenue"),
        ]
    )
    res_causal_valid = gate.evaluate_identifiability(ci, dag_valid, observed_effect_risk_ratio=2.0)
    assert res_causal_valid.is_identifiable is True
    assert res_causal_valid.status == CausalIdentifiabilityStatus.IDENTIFIED_BACKDOOR
    assert "market.seasonality" in res_causal_valid.backdoor_adjustment_set
    assert res_causal_valid.sensitivity_e_value == 3.41

    # Non-identifiable with unmeasured confounder
    dag_unmeasured = CausalGraphSpec(
        dag_id="dag_unmeasured",
        nodes=["marketing.spend", "sales.revenue", "U_hidden"],
        directed_edges=[
            ("U_hidden", "marketing.spend"),
            ("U_hidden", "sales.revenue"),
            ("marketing.spend", "sales.revenue"),
        ],
        unobserved_confounders=["U_hidden"]
    )
    res_causal_invalid = gate.evaluate_identifiability(ci, dag_unmeasured)
    assert res_causal_invalid.is_identifiable is False
    assert res_causal_invalid.status == CausalIdentifiabilityStatus.NOT_IDENTIFIABLE
    print(f" -> PASSED: Backdoor adjustment set resolved: {res_causal_valid.backdoor_adjustment_set}; unmeasured confounding blocked.")

    print("\n" + "=" * 80)
    print("ALL 5 AA-OS COMPILER, VALIDATOR, VERIFICATION & CAUSAL TESTS PASSED (100% INTEGRITY)")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
