"""Layer 1 focused tests for the typed RelationalPlan contract."""
from __future__ import annotations

import unittest

from packages.analytics_core.src.relational.compiler import JoinHop, RelationalPlan
from packages.analytics_core.src.relational.expressions import (
    AggExpr,
    And,
    ColumnRef,
    Comparison,
    DistinctCount,
    DivExpr,
    IsNotNull,
    IsNull,
    Literal,
    Or,
    is_row_level,
    to_polars,
    to_sql,
)


class TestExpressionSql(unittest.TestCase):
    def test_literals_and_escaping(self):
        self.assertEqual(to_sql(Literal(42)), "42")
        self.assertEqual(to_sql(Literal(3.5)), "3.5")
        self.assertEqual(to_sql(Literal(True)), "TRUE")
        self.assertEqual(to_sql(Literal(None)), "NULL")
        self.assertEqual(to_sql(Literal("O'Hara")), "'O''Hara'")

    def test_identifier_validation(self):
        with self.assertRaises(ValueError):
            ColumnRef("orders.revenue")
        with self.assertRaises(ValueError):
            ColumnRef("revenue; DROP TABLE x")
        with self.assertRaises(ValueError):
            ColumnRef("bad qualifier", "orders")
        self.assertEqual(ColumnRef("_revenue", "orders").column, "_revenue")

    def test_literal_whitelist(self):
        for bad in ([1], {"x": 1}, b"raw", object()):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    Literal(bad)

    def test_aggregates_and_distinct_count(self):
        self.assertEqual(to_sql(AggExpr("sum", ColumnRef("revenue"))), "SUM(revenue)")
        self.assertEqual(to_sql(DistinctCount(ColumnRef("customer_id"))), "COUNT(DISTINCT customer_id)")

    def test_division_zero_denominator(self):
        expr = DivExpr(
            AggExpr("SUM", ColumnRef("revenue")),
            DistinctCount(ColumnRef("customer_id")),
        )
        self.assertEqual(
            to_sql(expr),
            "(SUM(revenue) / NULLIF(COUNT(DISTINCT customer_id), 0))",
        )
        with self.assertRaises(TypeError):
            DivExpr(
                AggExpr("SUM", ColumnRef("revenue")),
                DistinctCount(ColumnRef("customer_id")),
                zero_denominator=object(),
            )


class TestPredicateValidation(unittest.TestCase):
    def test_and_or_require_predicates(self):
        a = Comparison(ColumnRef("a"), ">", Literal(1))
        b = IsNotNull(ColumnRef("b"))
        And([a, b])
        Or([a, b])
        with self.assertRaises(TypeError):
            And([ColumnRef("a")])
        with self.assertRaises(ValueError):
            Or([])

    def test_predicate_nodes_reject_predicate_operands(self):
        pred = Comparison(ColumnRef("a"), "=", Literal(1))
        with self.assertRaises(TypeError):
            Comparison(pred, "=", Literal(1))
        with self.assertRaises(TypeError):
            IsNull(pred)

    def test_is_row_level(self):
        self.assertTrue(is_row_level(ColumnRef("x")))
        self.assertTrue(is_row_level(Literal(1)))
        self.assertFalse(is_row_level(AggExpr("SUM", ColumnRef("x"))))
        self.assertFalse(is_row_level(DistinctCount(ColumnRef("x"))))
        self.assertTrue(is_row_level(DivExpr(ColumnRef("a"), ColumnRef("b"))))
        self.assertFalse(is_row_level(DivExpr(AggExpr("SUM", ColumnRef("a")), ColumnRef("b"))))
        self.assertTrue(is_row_level(Comparison(ColumnRef("a"), ">", Literal(1))))
        self.assertTrue(is_row_level(And([
            Comparison(ColumnRef("a"), ">", Literal(1)),
            IsNotNull(ColumnRef("b")),
        ])))
        self.assertFalse(is_row_level(And([
            Comparison(ColumnRef("a"), ">", Literal(1)),
            Comparison(AggExpr("SUM", ColumnRef("b")), ">", Literal(1)),
        ])))


class TestRelationalPlanContract(unittest.TestCase):
    def _plan(self) -> RelationalPlan:
        return RelationalPlan(
            base_table="orders",
            hops=[JoinHop("orders", "customers", "customer_id", "customer_id")],
            select_columns=["customers.segment AS segment"],
            aggregations={
                "sum_orders_revenue": AggExpr("SUM", ColumnRef("revenue", "orders")),
            },
            filters=[Comparison(ColumnRef("segment", "customers"), "=", Literal("Enterprise"))],
            group_by=["customers.segment"],
            order_by="sum_orders_revenue DESC",
        )

    def test_alias_is_declared_and_preserved(self):
        sql = self._plan().to_sql()
        self.assertIn("SUM(orders.revenue) AS sum_orders_revenue", sql)
        self.assertIn("(customers.segment = 'Enterprise')", sql)

    def test_invalid_alias_rejected(self):
        plan = RelationalPlan(
            base_table="orders",
            hops=[],
            select_columns=[],
            aggregations={"1bad_alias": AggExpr("SUM", ColumnRef("revenue"))},
        )
        with self.assertRaises(ValueError):
            plan.to_sql()

    def test_filter_with_aggregate_rejected(self):
        plan = RelationalPlan(
            base_table="orders",
            hops=[],
            select_columns=[],
            aggregations={"total": AggExpr("SUM", ColumnRef("revenue"))},
            filters=[Comparison(AggExpr("SUM", ColumnRef("revenue")), ">", Literal(100))],
        )
        with self.assertRaises(ValueError):
            plan.to_sql()

    def test_scalar_division_filter_is_allowed(self):
        plan = RelationalPlan(
            base_table="orders",
            hops=[],
            select_columns=["id"],
            filters=[
                Comparison(
                    DivExpr(ColumnRef("revenue"), ColumnRef("quantity")),
                    ">",
                    Literal(10),
                )
            ],
        )
        sql = plan.to_sql()
        self.assertIn("revenue", sql)
        self.assertIn("quantity", sql)

    def test_typed_polars_primitive_is_callable(self):
        # This test only checks that expressions compile to a Polars Expr.
        # Full plan execution belongs to Layer 2.
        expr = to_polars(AggExpr("SUM", ColumnRef("revenue")))
        self.assertTrue(hasattr(expr, "meta"))


if __name__ == "__main__":
    unittest.main()
