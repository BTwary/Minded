"""Validation for analyst-provided scope-filter expressions."""
from typing import Any, List
import re


def validate_scope_filter_expression(filter_sql: str, allowed_columns: List[str]) -> str:
    """Validate and normalize a user-entered SQL WHERE expression before DuckDB sees it.

    Accepted expressions are deliberately a predicate subset, not arbitrary SQL:
    comparisons, AND, IN/NOT IN, BETWEEN, IS [NOT] NULL, dataset columns and
    scalar literals. Functions, OR clauses, qualified identifiers, subqueries, DDL/DML,
    comments and statement separators are rejected. Column names must be present in
    the selected dataframe. Boolean literals are accepted only as comparison/list
    values (for example ``is_active = TRUE``), never as standalone tautologies.
    """
    clean = filter_sql.strip()
    if not clean:
        return ""

    import sqlglot
    from sqlglot import exp

    # Reject statement separators and comments by inspecting the parse, not the raw
    # text: a ';' or '--' inside a quoted literal (name = 'a;b') is just data.
    wrapped = f"SELECT * FROM _table WHERE {clean}"
    try:
        statements = sqlglot.parse(wrapped, read="duckdb")
    except Exception as parse_err:
        raise ValueError(f"Syntax error in scope filter expression: {parse_err}") from parse_err
    statements = [stmt for stmt in statements if stmt is not None]
    if len(statements) != 1:
        raise ValueError("Scope filter cannot contain semicolons, comment markers, or multiple statements.")
    parsed = statements[0]
    if any(getattr(node, "comments", None) for node in parsed.walk()):
        raise ValueError("Scope filter cannot contain semicolons, comment markers, or multiple statements.")

    where_node = parsed.find(exp.Where)
    if not where_node or where_node.this is None:
        raise ValueError("No valid WHERE expression found in scope filter.")

    allowed_set = {str(column).casefold() for column in allowed_columns}
    # Restrict execution to a small, predictable boolean-expression language. This prevents
    # otherwise-parseable function calls or unfamiliar AST nodes from reaching DuckDB.
    allowed_node_names = {
        "Where", "And", "Not", "Paren",
        "EQ", "NEQ", "GT", "GTE", "LT", "LTE", "Is", "In", "Tuple",
        "Between", "Column", "Identifier", "Literal", "Boolean", "Null", "Neg",
    }
    # Report injected constants precisely before the structural check, so "x = 1 OR TRUE" and
    # "x = 1 OR 1=1" are diagnosed as what they are rather than as a generic shape error.
    _cmp_names = {"EQ", "NEQ", "GT", "GTE", "LT", "LTE"}
    for node in where_node.walk():
        if isinstance(node, exp.Boolean):
            _parent = type(node.parent).__name__
            _grand = type(getattr(node.parent, "parent", None)).__name__
            if _parent not in _cmp_names and _parent != "In" and not (_parent == "Tuple" and _grand == "In"):
                raise ValueError("Boolean literals are permitted only as values in a comparison or IN list.")
        if type(node).__name__ in _cmp_names and not any(
            isinstance(child, exp.Column) for side in (node.this, node.expression) for child in side.walk()
        ):
            raise ValueError("Tautological comparisons between constants are not permitted in a scope filter.")
    predicate_node_names = {
        "And", "Not", "EQ", "NEQ", "GT", "GTE", "LT", "LTE", "Is", "In", "Between",
    }
    if type(where_node.this).__name__ not in predicate_node_names:
        raise ValueError("Filter must be a boolean predicate over dataset columns, not a standalone value.")
    for node in where_node.walk():
        node_name = type(node).__name__
        if node_name not in allowed_node_names:
            raise ValueError(
                f"Unsupported expression '{node_name}' in scope filter. Use column comparisons, "
                "AND, IN/NOT IN, BETWEEN, comparisons, or IS [NOT] NULL."
            )

        if isinstance(node, exp.Neg):
            inner = node.this
            if not (isinstance(inner, exp.Literal) and not inner.is_string):
                raise ValueError("Unary minus is only permitted on numeric literals (for example -100).")

        if isinstance(node, exp.Column):
            col_name = str(node.name)
            if getattr(node, "table", None) or getattr(node, "db", None) or getattr(node, "catalog", None):
                raise ValueError("Qualified column references are not permitted; use a column name from the selected dataset.")
            if col_name.casefold() not in allowed_set:
                raise ValueError(
                    f"Column '{col_name}' referenced in scope filter does not exist in dataset. "
                    f"Available columns: {sorted(map(str, allowed_columns))}"
                )

        if isinstance(node, exp.Boolean):
            parent = node.parent
            comparison_names = {"EQ", "NEQ", "GT", "GTE", "LT", "LTE"}
            allowed_boolean_value = type(parent).__name__ in comparison_names or type(parent).__name__ == "In"
            if type(parent).__name__ == "Tuple" and type(getattr(parent, "parent", None)).__name__ == "In":
                allowed_boolean_value = True
            if not allowed_boolean_value:
                raise ValueError("Boolean literals are permitted only as values in a comparison or IN list.")

        # NOT is only valid as SQL's NOT IN / IS NOT NULL representation.
        if type(node).__name__ == "Not" and type(getattr(node, "this", None)).__name__ not in {"In", "Is"}:
            raise ValueError("Use 'NOT IN' or 'IS NOT NULL'; arbitrary NOT expressions are not supported.")

        # Comparisons must be column-to-literal, never constant-only or column-to-column.
        if type(node).__name__ in {"EQ", "NEQ", "GT", "GTE", "LT", "LTE"}:
            left, right = node.this, node.expression
            left_has_column = any(isinstance(child, exp.Column) for child in left.walk())
            right_has_column = any(isinstance(child, exp.Column) for child in right.walk())
            if not left_has_column or right_has_column:
                raise ValueError("Use a dataset column on the left and a literal value on the right of each comparison.")
            if type(right).__name__ == "Null":
                raise ValueError("Use IS NULL or IS NOT NULL instead of '= NULL' or '!= NULL'.")

        if type(node).__name__ in {"In", "Between", "Is"}:
            lhs = getattr(node, "this", None)
            if lhs is None or type(lhs).__name__ != "Column":
                raise ValueError("Filter predicates must use a dataset column as the field.")
            rhs_nodes = []
            if type(node).__name__ == "In":
                rhs_nodes = list(getattr(node, "expressions", []) or [])
            elif type(node).__name__ == "Between":
                rhs_nodes = [node.args.get("low"), node.args.get("high")]
            elif type(node).__name__ == "Is":
                rhs_nodes = [getattr(node, "expression", None)]
            if any(
                rhs is not None and any(isinstance(child, exp.Column) for child in rhs.walk())
                for rhs in rhs_nodes
            ):
                raise ValueError("Filter values must be literals, not references to other columns.")
            if type(node).__name__ in {"In", "Between"} and any(
                rhs is not None and isinstance(rhs, exp.Null) for rhs in rhs_nodes
            ):
                raise ValueError("IN/NOT IN lists and BETWEEN bounds cannot contain NULL; use IS NULL or IS NOT NULL separately.")
    # Execute the serialized, validated AST rather than the original user string. The
    # whitelist above ensures the serialized predicate contains only the supported
    # operators, identifiers, and literals; using the normalized AST also prevents a
    # parser differential between the validation string and the SQL sent to DuckDB.
    normalized = where_node.this.sql(dialect="duckdb")
    if not normalized:
        raise ValueError("Scope filter did not produce a normalized predicate.")
    return normalized



def same_scope_filter(requested: Any, approved: Any) -> bool:
    """Match a proposed scope predicate to the user-approved preview.

    Whitespace and keyword casing are presentation details; predicate contents and
    string literals are not. This allows a pre-approved UI filter while refusing
    filters introduced only after the canonical contract has been locked.
    """
    def normalized(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    candidate = normalized(requested)
    return bool(candidate) and candidate == normalized(approved)
