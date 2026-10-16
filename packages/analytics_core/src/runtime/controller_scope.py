"""Validation for analyst-provided scope-filter expressions."""
from typing import List


def validate_scope_filter_expression(filter_sql: str, allowed_columns: List[str]) -> None:
    """Validate user-supplied scope filter SQL syntax, operators, and column references.

    Fail-closed against:
    - Multiple statements, semicolons, comments, subqueries, unions, DDL/DML.
    - Boolean tautology injection (e.g. 'OR TRUE', 'OR 1=1').
    - Non-existent column references.
    """
    clean = filter_sql.strip()
    if not clean:
        return

    # Reject comment markers or statement separators
    if any(tok in clean for tok in [";", "--", "/*", "*/"]):
        raise ValueError("Scope filter cannot contain semicolons, comment markers, or multiple statements.")

    import sqlglot
    from sqlglot import exp

    try:
        parsed = sqlglot.parse_one(f"SELECT * FROM _table WHERE {clean}")
    except Exception as parse_err:
        raise ValueError(f"Syntax error in scope filter expression: {parse_err}") from parse_err

    where_node = parsed.find(exp.Where)
    if not where_node:
        raise ValueError("No valid WHERE expression found in scope filter.")

    allowed_set = set(allowed_columns)
    disallowed_types = (
        exp.Subquery,
        exp.Select,
        exp.Command,
        exp.Drop,
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Union,
    )

    for node in where_node.walk():
        if isinstance(node, disallowed_types):
            raise ValueError(f"Disallowed expression in scope filter: {type(node).__name__}")
        if isinstance(node, exp.Column):
            col_name = node.name
            if col_name not in allowed_set:
                raise ValueError(
                    f"Column '{col_name}' referenced in scope filter does not exist in dataset. "
                    f"Available columns: {sorted(allowed_set)}"
                )
        if isinstance(node, exp.Boolean):
            raise ValueError("Boolean literals (TRUE/FALSE) are not permitted in scope filter expressions.")
        if isinstance(node, exp.EQ):
            if isinstance(node.this, exp.Literal) and isinstance(node.expression, exp.Literal):
                if node.this.this == node.expression.this:
                    raise ValueError("Tautological comparisons (e.g. 1=1) are not permitted in scope filter expressions.")
