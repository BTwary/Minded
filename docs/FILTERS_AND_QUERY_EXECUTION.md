# MindEd AA-OS: filter handling and analytical output

## How a filter travels through the application

MindEd has two sources of filter intent. They meet in the guided-investigation contract; neither source is allowed to silently replace the other.

1. **Natural-language question.** The canonical question contract extracts schema-grounded predicate records such as `{column: "order_status", operator: "!=", value: "Cancelled"}`. If the field or value cannot be resolved from the selected dataset, the question contract should leave that requirement unresolved rather than invent a column or value.
2. **Review panel.** The analyst sees the question-derived predicates and the generated calculation preview. An additional filter entered in the panel is labelled as an additional scope condition. The preview adds it to the question's predicates with `AND`.
3. **Guidance resolution.** `parse_relational_filters()` converts the additional filter expression to the same typed `{column, operator, value}` representation and resolves the field against the loaded schema. The guided resolver merges those records with the question's existing predicates.
4. **Approval and execution.** The approved plan/contract records which fields, operation, grouping and filters should be used. A user-supplied scope expression is parsed as a restricted SQL predicate and checked against the selected dataframe's columns. The accepted SQLGlot predicate AST is normalized and that normalized predicate—not the original free-form string—is passed to DuckDB. Invalid filters stop execution. Zero matching rows stop execution; MindEd must never quietly rerun the calculation on the unfiltered data.
5. **Result and audit.** The result screen should show the returned analytical answer first, then the actual run scope (dataset, rows in scope, result rows, aggregation, filters and verification state). The saved execution/evidence record—not the proposed SQL text—is the source of truth for what actually ran.

## Supported filter forms

The standard builder emits only simple predicates. The current supported forms are:

- `status = 'Delivered'`
- `status != 'Cancelled'`
- `revenue >= 100`
- `revenue BETWEEN 100 AND 500`
- `region IN ('East', 'West')`
- `region NOT IN ('East', 'West')`
- `cancelled_at IS NULL`
- `returned_at IS NOT NULL`
- Several conditions joined with `AND`

Use the UI's condition builder for common cases. It chooses a known column, operator and literal value, quotes identifiers, escapes apostrophes in text values, validates numbers, and displays the compiled filter. The advanced text field is still a restricted **WHERE expression**, not a full `SELECT` statement. Unsupported syntax, unknown fields and unsafe expressions are rejected. `OR`, arbitrary SQL functions, comments, statement separators, qualified column names and subqueries are outside this filter interface's current grammar.

For a list, enter comma-separated values in the builder. `BETWEEN` requires exactly two comma-separated bounds. `IN` / `NOT IN` lists must not contain `NULL`; SQL uses three-valued logic for null comparisons, so use a separate `IS NULL` / `IS NOT NULL` predicate instead. Do not write `= NULL` or `!= NULL`.

## Database semantics that matter

The `WHERE` predicate is evaluated before grouping and aggregation. Multiple `AND` predicates narrow the population: a row survives only when every condition is true. SQL comparisons against `NULL` evaluate to unknown and are excluded by `WHERE`; `IS NULL` is the explicit test for missing values. `IN` tests list membership; `BETWEEN low AND high` includes both boundaries. The generated SQL is a readable representation of the approved semantics, but its appearance alone is not proof that the query ran or verified.

The additional scope expression is deliberately limited to a small supported grammar and syntax/schema-validated before DuckDB execution. The natural-language contract path and advanced-expression path are not interchangeable: natural-language alternatives on one field can become `IN`/`NOT IN`, whereas the advanced filter editor accepts only simple predicates connected with `AND`. The validated AST is serialized in DuckDB dialect before execution so the SQL actually executed is the expression that passed the allowlist. Literals are currently embedded in that restricted normalized predicate, rather than individually bound as prepared-statement parameters; arbitrary SQL is not allowed. If any validation stage cannot safely resolve the filter, execution must fail closed rather than continue with a different population.

## Analyst-facing output

The result interface follows a progressive-disclosure layout:

1. **Answer:** one plain-language finding, qualified by the verdict and evidence status.
2. **Analysis at a glance:** the saved dataset name, rows in scope, output rows/groups, metric/aggregation, actual filters and verification state.
3. **Result view:** the exact result table and/or an appropriate chart; no fabricated visual values.
4. **Calculation details:** executed SQL/specification, approved contract, method and assumptions.
5. **Evidence and provenance:** verification records, hashes, evidence items and limitations.
6. **Human review/export:** record acceptance or correction separately from machine verification, and export the same saved result rather than recomputing it in the UI.

A visible `VERIFIED` label is reserved for the reconciled execution/evidence status; a completed plan, a returned number or a reviewer accepting the output is not sufficient by itself.

## Filter parsing: from a question to database execution

MindEd should treat filter parsing as a typed compilation pipeline, not as a string replacement exercise. The same question can pass through the following stages:

1. **Read the question and identify predicate intent.** Detect comparison language (`above`, `at least`, `below`), category inclusion/exclusion (`status is Delivered`, `excluding refunds`), date constraints, and explicit `WHERE` clauses. The parser must retain the original clause text for auditability.
2. **Resolve the field against the selected dataset schema.** `revenue` must map to a real column, and numeric comparators should bind to numeric columns. If two fields could fit, or the named field does not exist, the contract must be incomplete and ask for clarification. Never pick a column based on dataframe order.
3. **Resolve the value against its type and observed domain.** Numeric literals are parsed as numbers, booleans as booleans, and categorical values are checked against observed values. Case normalization is safe when the match is unique. Inflection recovery such as `refunds` → `refunded` should be accepted only when the candidate is unique and the intended dimension is clear. Fuzzy similarity alone is not sufficient evidence.
4. **Build a typed predicate.** For example, `revenue above 500` becomes `{column: "revenue", operator: ">", value: 500}`. `status is Returned or Cancelled` becomes an `IN` predicate; with exclusion language it becomes `NOT IN`. A date interval becomes two inclusive bounds when that is the requested meaning.
5. **Validate the predicate and the full contract.** Check allowed operator, field existence, compatible types, value resolution, and whether every detected material filter clause has a matching typed predicate. The ledger should match the literal and comparator for each clause, not merely count predicates. If a clause is unbound, execution is blocked.
6. **Compile the typed predicate to SQL.** Quote identifiers using the target dialect's rules and pass values as bound parameters where the database API supports it. Do not concatenate untrusted user text into SQL. For restricted advanced filters, parse to an AST, allow only supported nodes/operators, validate every column against the schema, then serialize the validated AST.
7. **Execute in the selected dataset scope.** `WHERE` filters rows before `GROUP BY` and aggregation. Conditions joined with `AND` require every condition to be true. `BETWEEN` includes both endpoints. `NULL` comparisons use SQL's three-valued logic; use `IS NULL` / `IS NOT NULL`, not `= NULL`.
8. **Reconcile the actual run against the approved contract.** Compare executed filters, metric, aggregation, grouping, time bounds, denominator, dataset identity/version, and row counts. A mismatch is `INCONCLUSIVE` or `FAILED`, never a verified answer. Zero rows is a valid outcome to report, not a reason to silently retry without filters.
9. **Show the same saved execution record in the UI and exports.** Lead with the answer, then show dataset, population rows, metric, grouping, readable filters, verification status, result table, and expandable SQL/method/evidence. Do not reconstruct a different result in the frontend.

### Example

Question: “How many orders had revenue above 500, excluding refunded orders?”

Assuming the dataset has a numeric `revenue` column and a categorical `status` column with the observed value `refunded`, the intended row predicate is conceptually:

```sql
WHERE revenue > ? AND status <> ?
```

with bound values `500` and `refunded`. If the question asks for a count of orders, the aggregation must count the correct order entity or rows according to the dataset's declared grain. The predicate alone does not establish whether `COUNT(*)` or `COUNT(DISTINCT order_id)` is correct.

### Database execution versus question parsing

SQL engines do not infer the user's intended meaning. They evaluate the predicate they are given. If the parser drops `revenue > 500`, SQL can successfully return a number for the wrong population. Therefore, syntax validity is only one gate: semantic coverage, schema/type validation, and post-execution contract reconciliation are separate required gates.
