"""Layer 2 focused tests for native RelationalPlan.to_polars()."""

from __future__ import annotations

import os
import sys
import unittest

import duckdb
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.relational.compiler import (
    JoinHop,
    RelationalPlan,
)

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
    to_polars as compile_expr,
)


def duckdb_execute(plan, tables):
    con = duckdb.connect(":memory:")

    try:
        for name, frame in tables.items():
            con.register(name, frame)

        return con.execute(
            plan.to_sql()
        ).fetchdf()

    finally:
        con.close()


def polars_execute(plan, tables):
    return plan.to_polars(
        {
            name: (
                frame
                if isinstance(frame, pl.DataFrame)
                else pl.from_pandas(frame)
            )
            for name, frame in tables.items()
        }
    ).to_pandas()


def assert_equivalent(
    testcase,
    primary,
    secondary,
    sort_by=None,
    atol=1e-9,
):
    p = primary.copy()
    s = secondary.copy()

    if sort_by:
        p = p.sort_values(
            sort_by
        ).reset_index(drop=True)

        s = s.sort_values(
            sort_by
        ).reset_index(drop=True)

    else:
        p = p.reset_index(drop=True)
        s = s.reset_index(drop=True)

    testcase.assertEqual(
        set(p.columns),
        set(s.columns),
    )

    testcase.assertEqual(
        len(p),
        len(s),
    )

    for column in sorted(p.columns):
        if pd.api.types.is_numeric_dtype(
            p[column]
        ):
            np.testing.assert_allclose(
                p[column].astype(float).to_numpy(),
                s[column].astype(float).to_numpy(),
                rtol=0.0,
                atol=atol,
                equal_nan=True,
            )
        else:
            testcase.assertEqual(
                list(p[column]),
                list(s[column]),
            )


class TestSingleTable(unittest.TestCase):

    def test_aggregate(self):
        tables = {
            "sales": pd.DataFrame(
                {
                    "amount": [10, 20, 30, 40],
                    "customer": [1, 1, 2, 3],
                }
            )
        }

        plan = RelationalPlan(
            base_table="sales",
            aggregations={
                "total": AggExpr(
                    "SUM",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                ),
                "mean": AggExpr(
                    "AVG",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                ),
                "count": AggExpr(
                    "COUNT",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                ),
                "distinct_customers": DistinctCount(
                    ColumnRef(
                        "customer",
                        "sales",
                    )
                ),
            },
        )

        primary = duckdb_execute(
            plan,
            tables,
        )

        secondary = polars_execute(
            plan,
            tables,
        )

        assert_equivalent(
            self,
            primary,
            secondary,
        )

        self.assertEqual(
            int(primary["count"].iloc[0]),
            4,
        )

        self.assertEqual(
            int(
                primary[
                    "distinct_customers"
                ].iloc[0]
            ),
            3,
        )

    def test_filter(self):
        tables = {
            "sales": pd.DataFrame(
                {
                    "amount": [10, 20, 30],
                }
            )
        }

        plan = RelationalPlan(
            base_table="sales",
            aggregations={
                "total": AggExpr(
                    "SUM",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                )
            },
            filters=[
                Comparison(
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                    ">=",
                    Literal(20),
                )
            ],
        )

        assert_equivalent(
            self,
            duckdb_execute(plan, tables),
            polars_execute(plan, tables),
        )


class TestFilters(unittest.TestCase):

    def setUp(self):
        self.tables = {
            "data": pd.DataFrame(
                {
                    "n": [1, 2, 3, 4, 5],
                    "label": [
                        "a",
                        "b",
                        "a",
                        "b",
                        "a",
                    ],
                    "value": [
                        10.0,
                        None,
                        30.0,
                        None,
                        50.0,
                    ],
                }
            )
        }

    def run_filter(self, expression):
        plan = RelationalPlan(
            base_table="data",
            select_columns=[
                "data.n AS n",
                "data.label AS label",
                "data.value AS value",
            ],
            filters=[expression],
        )

        assert_equivalent(
            self,
            duckdb_execute(
                plan,
                self.tables,
            ),
            polars_execute(
                plan,
                self.tables,
            ),
            sort_by=["n"],
        )

    def test_comparison_operators(self):
        expressions = [
            Comparison(
                ColumnRef("n", "data"),
                "=",
                Literal(3),
            ),
            Comparison(
                ColumnRef("n", "data"),
                "!=",
                Literal(3),
            ),
            Comparison(
                ColumnRef("n", "data"),
                "<",
                Literal(3),
            ),
            Comparison(
                ColumnRef("n", "data"),
                "<=",
                Literal(3),
            ),
            Comparison(
                ColumnRef("n", "data"),
                ">",
                Literal(3),
            ),
            Comparison(
                ColumnRef("n", "data"),
                ">=",
                Literal(3),
            ),
        ]

        for expression in expressions:
            with self.subTest(expression=expression):
                self.run_filter(expression)

    def test_and_or(self):
        self.run_filter(
            And(
                [
                    Comparison(
                        ColumnRef(
                            "n",
                            "data",
                        ),
                        ">",
                        Literal(1),
                    ),
                    Comparison(
                        ColumnRef(
                            "label",
                            "data",
                        ),
                        "=",
                        Literal("a"),
                    ),
                ]
            )
        )

        self.run_filter(
            Or(
                [
                    Comparison(
                        ColumnRef(
                            "n",
                            "data",
                        ),
                        "=",
                        Literal(1),
                    ),
                    Comparison(
                        ColumnRef(
                            "n",
                            "data",
                        ),
                        "=",
                        Literal(5),
                    ),
                ]
            )
        )

    def test_null_checks(self):
        self.run_filter(
            IsNull(
                ColumnRef(
                    "value",
                    "data",
                )
            )
        )

        self.run_filter(
            IsNotNull(
                ColumnRef(
                    "value",
                    "data",
                )
            )
        )


class TestJoins(unittest.TestCase):

    def test_inner_join(self):
        tables = {
            "orders": pd.DataFrame(
                {
                    "customer_id": [
                        "C1",
                        "C2",
                        "C1",
                    ],
                    "revenue": [
                        100.0,
                        200.0,
                        150.0,
                    ],
                }
            ),
            "customers": pd.DataFrame(
                {
                    "customer_id": [
                        "C1",
                        "C2",
                    ],
                    "segment": [
                        "A",
                        "B",
                    ],
                }
            ),
        }

        plan = RelationalPlan(
            base_table="orders",
            hops=[
                JoinHop(
                    "orders",
                    "customers",
                    "customer_id",
                    "customer_id",
                )
            ],
            select_columns=[
                "customers.segment AS segment"
            ],
            aggregations={
                "total_revenue": AggExpr(
                    "SUM",
                    ColumnRef(
                        "revenue",
                        "orders",
                    ),
                )
            },
            group_by=[
                "customers.segment"
            ],
        )

        assert_equivalent(
            self,
            duckdb_execute(
                plan,
                tables,
            ),
            polars_execute(
                plan,
                tables,
            ),
            sort_by=["segment"],
        )

    def test_left_join(self):
        tables = {
            "orders": pd.DataFrame(
                {
                    "order_id": [
                        "O1",
                        "O2",
                    ],
                    "customer_id": [
                        "C1",
                        "C99",
                    ],
                }
            ),
            "customers": pd.DataFrame(
                {
                    "customer_id": [
                        "C1",
                        "C2",
                    ],
                    "segment": [
                        "A",
                        "B",
                    ],
                }
            ),
        }

        plan = RelationalPlan(
            base_table="orders",
            hops=[
                JoinHop(
                    "orders",
                    "customers",
                    "customer_id",
                    "customer_id",
                    join_type="LEFT",
                )
            ],
            select_columns=[
                "orders.order_id AS order_id",
                "customers.segment AS segment",
            ],
            order_by="order_id",
        )

        primary = duckdb_execute(
            plan,
            tables,
        )

        secondary = polars_execute(
            plan,
            tables,
        )

        assert_equivalent(
            self,
            primary,
            secondary,
            sort_by=["order_id"],
        )

        self.assertTrue(
            pd.isna(
                secondary.loc[
                    secondary["order_id"] == "O2",
                    "segment",
                ].iloc[0]
            )
        )


class TestGroupedAndDerived(unittest.TestCase):

    def test_grouped_aggregation(self):
        tables = {
            "sales": pd.DataFrame(
                {
                    "region": [
                        "N",
                        "N",
                        "S",
                        "S",
                        "S",
                    ],
                    "amount": [
                        10,
                        20,
                        30,
                        40,
                        50,
                    ],
                }
            )
        }

        plan = RelationalPlan(
            base_table="sales",
            select_columns=[
                "sales.region AS region"
            ],
            aggregations={
                "total": AggExpr(
                    "SUM",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                ),
                "count": AggExpr(
                    "COUNT",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                ),
            },
            group_by=[
                "sales.region"
            ],
            order_by="region",
        )

        assert_equivalent(
            self,
            duckdb_execute(
                plan,
                tables,
            ),
            polars_execute(
                plan,
                tables,
            ),
            sort_by=["region"],
        )

    def test_division(self):
        tables = {
            "orders": pd.DataFrame(
                {
                    "customer": [
                        "C1",
                        "C1",
                        "C2",
                    ],
                    "revenue": [
                        10.0,
                        20.0,
                        30.0,
                    ],
                }
            )
        }

        plan = RelationalPlan(
            base_table="orders",
            aggregations={
                "revenue_per_customer": DivExpr(
                    AggExpr(
                        "SUM",
                        ColumnRef(
                            "revenue",
                            "orders",
                        ),
                    ),
                    DistinctCount(
                        ColumnRef(
                            "customer",
                            "orders",
                        )
                    ),
                )
            },
        )

        primary = duckdb_execute(
            plan,
            tables,
        )

        secondary = polars_execute(
            plan,
            tables,
        )

        assert_equivalent(
            self,
            primary,
            secondary,
        )

    def test_zero_denominator(self):
        tables = {
            "orders": pd.DataFrame(
                {
                    "customer": [
                        None,
                        None,
                    ],
                    "revenue": [
                        10.0,
                        20.0,
                    ],
                }
            )
        }

        plan = RelationalPlan(
            base_table="orders",
            aggregations={
                "ratio": DivExpr(
                    AggExpr(
                        "SUM",
                        ColumnRef(
                            "revenue",
                            "orders",
                        ),
                    ),
                    DistinctCount(
                        ColumnRef(
                            "customer",
                            "orders",
                        )
                    ),
                )
            },
        )

        primary = duckdb_execute(
            plan,
            tables,
        )

        secondary = polars_execute(
            plan,
            tables,
        )

        self.assertTrue(
            pd.isna(primary["ratio"].iloc[0])
        )

        self.assertTrue(
            pd.isna(secondary["ratio"].iloc[0])
        )

        assert_equivalent(
            self,
            primary,
            secondary,
        )

    def test_order_and_limit(self):
        tables = {
            "sales": pd.DataFrame(
                {
                    "region": [
                        "N",
                        "N",
                        "S",
                        "S",
                        "E",
                    ],
                    "amount": [
                        10,
                        20,
                        30,
                        40,
                        50,
                    ],
                }
            )
        }

        plan = RelationalPlan(
            base_table="sales",
            select_columns=[
                "sales.region AS region"
            ],
            aggregations={
                "total": AggExpr(
                    "SUM",
                    ColumnRef(
                        "amount",
                        "sales",
                    ),
                )
            },
            group_by=[
                "sales.region"
            ],
            order_by="total DESC",
            limit=2,
        )

        primary = duckdb_execute(
            plan,
            tables,
        )

        secondary = polars_execute(
            plan,
            tables,
        )

        self.assertEqual(
            list(primary["region"]),
            list(secondary["region"]),
        )

        self.assertEqual(
            list(primary["region"]),
            ["S", "E"],
        )


        assert_equivalent(
            self,
            primary,
            secondary,
        )


class TestFailClosed(unittest.TestCase):

    def test_missing_table(self):
        plan = RelationalPlan(
            base_table="missing",
            select_columns=[
                "missing.x AS x"
            ],
        )

        with self.assertRaises(
            KeyError
        ):
            plan.to_polars({})

    def test_missing_join_table(self):
        plan = RelationalPlan(
            base_table="a",
            hops=[
                JoinHop(
                    "a",
                    "b",
                    "k",
                    "k",
                )
            ],
            select_columns=[
                "a.x AS x"
            ],
        )

        with self.assertRaises(
            KeyError
        ):
            plan.to_polars(
                {
                    "a": pl.DataFrame(
                        {
                            "k": [1],
                            "x": [1],
                        }
                    )
                }
            )

    def test_missing_select_column(self):
        plan = RelationalPlan(
            base_table="t",
            select_columns=[
                "t.missing AS missing"
            ],
        )

        with self.assertRaises(
            KeyError
        ):
            plan.to_polars(
                {
                    "t": pl.DataFrame(
                        {
                            "x": [1]
                        }
                    )
                }
            )

    def test_bad_join_key(self):
        plan = RelationalPlan(
            base_table="a",
            hops=[
                JoinHop(
                    "a",
                    "b",
                    "missing",
                    "k",
                )
            ],
            select_columns=[
                "a.x AS x"
            ],
        )

        with self.assertRaises(
            KeyError
        ):
            plan.to_polars(
                {
                    "a": pl.DataFrame(
                        {
                            "k": [1],
                            "x": [1],
                        }
                    ),
                    "b": pl.DataFrame(
                        {
                            "k": [1],
                            "y": [2],
                        }
                    ),
                }
            )

    def test_bad_join_type(self):
        plan = RelationalPlan(
            base_table="a",
            hops=[
                JoinHop(
                    "a",
                    "b",
                    "k",
                    "k",
                    join_type="FULL",
                )
            ],
            select_columns=[
                "a.x AS x"
            ],
        )

        with self.assertRaises(
            ValueError
        ):
            plan.to_polars(
                {
                    "a": pl.DataFrame(
                        {
                            "k": [1],
                            "x": [1],
                        }
                    ),
                    "b": pl.DataFrame(
                        {
                            "k": [1],
                            "y": [2],
                        }
                    ),
                }
            )

    def test_bad_select_spec(self):
        plan = RelationalPlan(
            base_table="t",
            select_columns=[
                "t.x; DROP TABLE t AS x"
            ],
        )

        with self.assertRaises(
            ValueError
        ):
            plan.to_polars(
                {
                    "t": pl.DataFrame(
                        {
                            "x": [1]
                        }
                    )
                }
            )

    def test_bad_order_spec(self):
        plan = RelationalPlan(
            base_table="t",
            select_columns=[
                "t.x AS x"
            ],
            order_by="x; DROP TABLE t",
        )

        with self.assertRaises(
            ValueError
        ):
            plan.to_polars(
                {
                    "t": pl.DataFrame(
                        {
                            "x": [1]
                        }
                    )
                }
            )

    def test_aggregate_filter_rejected(self):
        plan = RelationalPlan(
            base_table="t",
            aggregations={
                "total": AggExpr(
                    "SUM",
                    ColumnRef(
                        "x",
                        "t",
                    ),
                )
            },
            filters=[
                Comparison(
                    AggExpr(
                        "SUM",
                        ColumnRef(
                            "x",
                            "t",
                        ),
                    ),
                    ">",
                    Literal(10),
                )
            ],
        )

        with self.assertRaises(
            ValueError
        ):
            plan.to_polars(
                {
                    "t": pl.DataFrame(
                        {
                            "x": [1, 2, 3]
                        }
                    )
                }
            )

    def test_unknown_expression(self):
        class NotAPlanExpr:
            pass

        with self.assertRaises(
            TypeError
        ):
            compile_expr(
                NotAPlanExpr()
            )


if __name__ == "__main__":
    unittest.main()
