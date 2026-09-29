"""P0-5 Regression: relational_compiler uses safe SQL literals — no raw string interpolation.

Tests that the _sql_literal function and _map_operator produce properly
escaped SQL for values that would cause SQL injection or analytic-contract
violation if interpolated raw.
"""
import sys
import unittest
sys.path.insert(0, '.')

from packages.analytics_core.src.sql.relational_compiler import RelationalCompiler, _sql_literal
from packages.schemas.src.analysis import FilterOperator


class TestSqlLiteral(unittest.TestCase):
    def test_plain_string_quoted(self):
        self.assertEqual(_sql_literal('hello'), "'hello'")

    def test_single_quote_in_string_escaped(self):
        # A value like "O'Brien" must become 'O''Brien'
        result = _sql_literal("O'Brien")
        self.assertEqual(result, "'O''Brien'")

    def test_sql_injection_pattern_escaped(self):
        # Classic injection: ' OR '1'='1
        result = _sql_literal("' OR '1'='1")
        # Must be fully enclosed in outer quotes with inner quotes doubled
        self.assertEqual(result, "''' OR ''1''=''1'")

    def test_integer_bare(self):
        self.assertEqual(_sql_literal(42), '42')

    def test_none_is_null(self):
        self.assertEqual(_sql_literal(None), 'NULL')

    def test_bool_true(self):
        self.assertEqual(_sql_literal(True), 'TRUE')


class TestMapOperator(unittest.TestCase):
    def setUp(self):
        self.compiler = RelationalCompiler()

    def test_eq_string_uses_safe_literal(self):
        result = self.compiler._map_operator(FilterOperator.EQ, "us-east")
        self.assertEqual(result, "= 'us-east'")

    def test_eq_string_with_quote_escapes(self):
        result = self.compiler._map_operator(FilterOperator.EQ, "O'Brien")
        self.assertIn("''", result)  # single-quote doubled

    def test_ne_string(self):
        result = self.compiler._map_operator(FilterOperator.NE, "active")
        self.assertEqual(result, "!= 'active'")

    def test_in_list_strings_all_quoted(self):
        result = self.compiler._map_operator(FilterOperator.IN, ['a', 'b', "c'd"])
        self.assertIn("'a'", result)
        self.assertIn("'b'", result)
        self.assertIn("'c''d'", result)  # quote escaped

    def test_not_in_list(self):
        result = self.compiler._map_operator(FilterOperator.NOT_IN, ['x', 'y'])
        self.assertIn('NOT IN', result)
        self.assertIn("'x'", result)

    def test_numeric_operators_no_quotes(self):
        self.assertEqual(self.compiler._map_operator(FilterOperator.GT, 100), '> 100')
        self.assertEqual(self.compiler._map_operator(FilterOperator.LTE, 50), '<= 50')

    def test_injection_in_eq_value_neutralised(self):
        # If val contains SQL metacharacters, they must be inside the quoted literal
        result = self.compiler._map_operator(FilterOperator.EQ, "'; DROP TABLE users; --")
        # The entire value must be inside the outer quotes
        self.assertTrue(result.startswith("= '"))
        self.assertTrue(result.endswith("'"))
        # No semicolons should appear outside the string literal
        inner = result[3:-1]  # strip "= '" and trailing "'"
        # All single quotes inside should be doubled
        self.assertNotIn("'", inner.replace("''", ""))


if __name__ == '__main__':
    unittest.main()
