"""Milestone Section 15/16/19 acceptance tests for the new standalone
relational execution foundation (join_safety.py + compiler.py +
evidence_identity.py).

IMPORTANT SCOPE NOTE: these tests exercise the new modules directly and
via DuckDB -- they do NOT run through InvestigationController's autonomous
NL-question path, because that full integration (intent -> semantic model
-> EIG-driven relational experiment selection) was not built this pass.
That gap is disclosed, not hidden -- see the milestone report.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.relational.join_safety import (
    Cardinality, SafetyStatus, assess_join_safety,
)
from packages.analytics_core.src.relational.compiler import (
    JoinHop, RelationalCompiler, RelationalPlan,
)
from packages.analytics_core.src.relational.expressions import (
    AggExpr, ColumnRef, Comparison, Literal,
)
from packages.analytics_core.src.graph.evidence_identity import (
    EvidenceRegistry, compute_evidence_identity,
)


def _make_fixture():
    rng = np.random.default_rng(123)

    customers = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(1, 21)],
        "segment": (["Enterprise"] * 5 + ["SMB"] * 8 + ["Consumer"] * 7),
    })

    products = pd.DataFrame({
        "product_id": [f"P{i}" for i in range(1, 6)],
        "category": ["Premium", "Premium", "Standard", "Standard", "Standard"],
    })

    order_rows = []
    oid = 1
    for _, cust in customers.iterrows():
        n_orders = rng.integers(3, 8)
        for _ in range(n_orders):
            prod = products.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            # Enterprise customers systematically spend more on Premium
            base = 500 if (cust["segment"] == "Enterprise" and prod["category"] == "Premium") else 80
            revenue = float(base + rng.normal(0, 15))
            order_rows.append({
                "order_id": f"O{oid}",
                "customer_id": cust["customer_id"],
                "product_id": prod["product_id"],
                "revenue": max(revenue, 0.0),
            })
            oid += 1
    orders = pd.DataFrame(order_rows)
    return customers, products, orders


class TestBlackBoxMultiTableQuestion(unittest.TestCase):
    """Section 15: 'Which customer segment generated the highest revenue
    from the premium product category?' -- requires customers + orders +
    products; cannot be answered from any single table."""

    def setUp(self):
        self.customers, self.products, self.orders = _make_fixture()

    def test_correct_segment_and_revenue_via_real_multi_table_join(self):
        # --- Independent ground truth, computed OUTSIDE the compiler ---
        merged = self.orders.merge(self.customers, on="customer_id").merge(
            self.products, on="product_id"
        )
        premium = merged[merged["category"] == "Premium"]
        ground_truth = premium.groupby("segment")["revenue"].sum().sort_values(ascending=False)
        expected_segment = ground_truth.index[0]
        expected_revenue = float(ground_truth.iloc[0])

        # --- Sanity: this genuinely requires all three tables. A
        # single-table computation cannot reproduce it (orders alone has
        # no segment or category). ---
        self.assertNotIn("segment", self.orders.columns)
        self.assertNotIn("category", self.orders.columns)

        # --- Now via the real compiler: genuine multi-table SQL, join
        # safety verified first, executed through DuckDB. ---
        compiler = RelationalCompiler({
            "customers": self.customers, "orders": self.orders, "products": self.products,
        })
        plan = RelationalPlan(
            base_table="orders",
            hops=[
                JoinHop("orders", "customers", "customer_id", "customer_id"),
                JoinHop("orders", "products", "product_id", "product_id"),
            ],
            select_columns=["customers.segment AS segment"],
            aggregations={
                "sum_orders_revenue": AggExpr("SUM", ColumnRef("revenue", "orders")),
            },
            filters=[
                Comparison(
                    ColumnRef("category", "products"), "=", Literal("Premium")
                )
            ],
            group_by=["customers.segment"],
            order_by="sum_orders_revenue DESC",
        )
        result = compiler.compile_and_execute(plan, base_natural_key="segment")

        self.assertTrue(result.safe, f"Plan should be safe: {result.blocked_reason}")
        self.assertIsNotNone(result.result_df)
        self.assertIn("FROM orders", result.sql)
        self.assertIn("JOIN customers", result.sql)
        self.assertIn("JOIN products", result.sql)
        # This is the literal check the brief requires: reject anything
        # that isn't genuine multi-table SQL.
        self.assertNotEqual(result.sql.strip().split("\n")[1].strip(), "FROM data_table")

        top_row = result.result_df.iloc[0]
        print(f"\n[multi-table test] ground truth: {expected_segment} = {expected_revenue:.2f}")
        print(f"[multi-table test] compiler result: {top_row['segment']} = {top_row['sum_orders_revenue']:.2f}")

        self.assertEqual(top_row["segment"], expected_segment)
        self.assertAlmostEqual(top_row["sum_orders_revenue"], expected_revenue, places=2)

        # --- Cardinality/grain proof is real, not a self-comparison ---
        gp = result.grain_proof
        self.assertEqual(gp.rows_before["orders"], len(self.orders))
        self.assertEqual(gp.rows_before["customers"], len(self.customers))
        self.assertEqual(gp.rows_before["products"], len(self.products))
        self.assertTrue(gp.aggregation_applied)
        # customers/products are the "one" side of many-to-one from orders
        cardinalities = {r.cardinality for r in result.safety_reports}
        self.assertIn(Cardinality.MANY_TO_ONE, cardinalities)


class TestJoinFailureCases(unittest.TestCase):
    """Section 16: required join-failure test matrix."""

    def test_safe_many_to_one(self):
        orders = pd.DataFrame({"order_id": ["O1", "O2", "O3"], "customer_id": ["C1", "C1", "C2"]})
        customers = pd.DataFrame({"customer_id": ["C1", "C2"]})
        report = assess_join_safety(orders, customers, "customer_id", "customer_id")
        self.assertEqual(report.status, SafetyStatus.SAFE)
        self.assertEqual(report.cardinality, Cardinality.MANY_TO_ONE)

    def test_safe_one_to_one(self):
        a = pd.DataFrame({"id": ["1", "2", "3"]})
        b = pd.DataFrame({"id": ["1", "2", "3"]})
        report = assess_join_safety(a, b, "id", "id")
        self.assertEqual(report.status, SafetyStatus.SAFE)
        self.assertEqual(report.cardinality, Cardinality.ONE_TO_ONE)

    def test_unsafe_many_to_many_blocked_by_default(self):
        a = pd.DataFrame({"tag": ["x", "x", "y", "y"]})
        b = pd.DataFrame({"tag": ["x", "x", "y", "y"]})
        report = assess_join_safety(a, b, "tag", "tag")
        self.assertEqual(report.cardinality, Cardinality.MANY_TO_MANY)
        self.assertEqual(report.status, SafetyStatus.UNSAFE)

    def test_many_to_many_explicitly_authorized_is_safe(self):
        a = pd.DataFrame({"tag": ["x", "x", "y", "y"]})
        b = pd.DataFrame({"tag": ["x", "x", "y", "y"]})
        report = assess_join_safety(a, b, "tag", "tag", authorize_many_to_many=True)
        self.assertEqual(report.status, SafetyStatus.SAFE)

    def test_unknown_cardinality_blocked_not_defaulted_safe(self):
        empty = pd.DataFrame({"id": pd.Series(dtype="object")})
        other = pd.DataFrame({"id": ["1", "2"]})
        report = assess_join_safety(empty, other, "id", "id")
        self.assertEqual(report.status, SafetyStatus.UNKNOWN)
        self.assertNotEqual(report.status, SafetyStatus.SAFE)

    def test_missing_key_is_unsafe_not_silently_safe(self):
        a = pd.DataFrame({"wrong_key": ["1", "2"]})
        b = pd.DataFrame({"id": ["1", "2"]})
        report = assess_join_safety(a, b, "id", "id")
        self.assertEqual(report.status, SafetyStatus.UNSAFE)
        self.assertFalse(report.keys_exist)

    def test_incompatible_key_types_unsafe(self):
        a = pd.DataFrame({"id": [1, 2, 3]})
        b = pd.DataFrame({"id": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
        report = assess_join_safety(a, b, "id", "id")
        self.assertEqual(report.status, SafetyStatus.UNSAFE)
        self.assertFalse(report.key_type_compatible)

    def test_grain_changing_aggregation_is_documented_not_silent(self):
        customers = pd.DataFrame({"customer_id": ["C1", "C2"]})
        orders = pd.DataFrame({
            "customer_id": ["C1", "C1", "C1", "C2"],
            "revenue": [10.0, 20.0, 30.0, 40.0],
        })
        compiler = RelationalCompiler({"customers": customers, "orders": orders})
        plan = RelationalPlan(
            base_table="orders",
            hops=[JoinHop("orders", "customers", "customer_id", "customer_id")],
            select_columns=["orders.customer_id AS customer_id"],
            aggregations={
                "sum_orders_revenue": AggExpr("SUM", ColumnRef("revenue", "orders")),
            },
            group_by=["orders.customer_id"],
        )
        result = compiler.compile_and_execute(plan, base_natural_key="customer_id")
        self.assertTrue(result.safe)
        gp = result.grain_proof
        # Grain genuinely changes: 4 order rows -> 2 customer rows via
        # aggregation. This must be recorded, not silently accepted.
        self.assertEqual(gp.rows_before["orders"], 4)
        self.assertEqual(gp.rows_after, 2)
        self.assertTrue(gp.aggregation_applied)
        self.assertIn(Cardinality.MANY_TO_ONE, [c for c in [
            Cardinality(v) for v in gp.expected_cardinality
        ]])


class TestEvidenceDeduplication(unittest.TestCase):
    """Section 19: identical evidence presented twice must not double-count;
    genuinely new evidence must be eligible for update."""

    def test_identical_computation_deduplicated(self):
        registry = EvidenceRegistry()
        identity1 = compute_evidence_identity("ds-hash-abc", "SELECT SUM(revenue) FROM orders", "duckdb")
        identity2 = compute_evidence_identity("ds-hash-abc", "select sum(revenue) from orders  ", "DUCKDB")

        is_dup1, canon1 = registry.register(identity1, "EVD-001")
        self.assertFalse(is_dup1)

        is_dup2, canon2 = registry.register(identity2, "EVD-002")
        self.assertTrue(is_dup2, "Cosmetically different but substantively identical query must dedupe.")
        self.assertEqual(canon2, "EVD-001")

    def test_genuinely_new_evidence_not_deduplicated(self):
        registry = EvidenceRegistry()
        identity1 = compute_evidence_identity("ds-hash-abc", "SELECT SUM(revenue) FROM orders", "duckdb")
        identity2 = compute_evidence_identity("ds-hash-abc", "SELECT AVG(revenue) FROM orders", "duckdb")
        _, _ = registry.register(identity1, "EVD-001")
        is_dup, canon = registry.register(identity2, "EVD-003")
        self.assertFalse(is_dup)
        self.assertEqual(canon, "EVD-003")

    def test_dual_engine_verification_not_collapsed_into_one_identity(self):
        # DuckDB and Polars independently computing the same query is
        # dual-engine verification -- these must remain TWO distinct
        # evidence identities that are expected to agree, not one.
        registry = EvidenceRegistry()
        id_duckdb = compute_evidence_identity("ds-hash-abc", "SELECT SUM(revenue) FROM orders", "duckdb")
        id_polars = compute_evidence_identity("ds-hash-abc", "SELECT SUM(revenue) FROM orders", "polars")
        self.assertNotEqual(id_duckdb, id_polars)
        is_dup1, _ = registry.register(id_duckdb, "EVD-DUCKDB")
        is_dup2, _ = registry.register(id_polars, "EVD-POLARS")
        self.assertFalse(is_dup1)
        self.assertFalse(is_dup2)


if __name__ == "__main__":
    unittest.main()
