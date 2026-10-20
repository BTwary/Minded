# Multi-question analysis sessions: release notes and verification

## Product model

An analysis session is the user-facing history object. It pins a workspace, the selected dataset IDs and an optional shared population filter. It owns an ordered trail of question-level `Investigation` records. Each question retains its own approved canonical contract, execution evidence, verification state and review decision.

Accepting a verified result adds it to the session's cumulative report. It does not close the session. A user can export a versioned draft PDF/DOCX while the session is open and continue asking questions. **Close Analysis** freezes session membership and title/scope; closed sessions reject new question creation and review updates. Exports include only question results whose latest review is `ACCEPTED_AND_VERIFIED` and whose saved execution is complete.

## Population/filter behavior

- The dataset scope is pinned when a session is created.
- An optional session filter is included in each question's pre-analysis contract.
- Question-specific filters are added to the session filters with AND semantics and remain part of that question's exact executed population.
- A correction after contract lock may repeat the already-approved filter, but a different/new post-lock filter is rejected until the plan is previewed and approved again.
- Zero-row and invalid filters fail closed; the controller must not retry against all records.

Example regression predicate:

```sql
customer_segment = 'Small Business' AND quarter = 'Q3' AND delivery_completed_flag = 1
```

The test fixture confirms that this combination is preserved in the resolved contract and produces one eligible row before grouping.

## Visual/diagnostic answer behavior

Visualizations use the saved result contract. Supported outputs include ranking/comparison bars, trend lines, distribution plots, scatter plots, and period-change waterfalls. Unsupported, inconsistent or unverified outputs do not receive confidently inferred charts.

`PERIOD_CHANGE` answers explain the observed prior/current values and saved bridge contributions. `ROOT_CAUSE` answers expose measured candidate factors and how adjustment changed the observed difference. Both paths explicitly say when observed decomposition/adjustment does not establish causality. The canonical parser also preserves compound intent: a grouped ranking question followed by “and why” becomes `ROOT_CAUSE`/`DIAGNOSTIC` while retaining the metric, group and aggregation, rather than silently dropping the explanation clause.

## History cleanup/visibility

Generated project names beginning `Stress Project` or `Debug Project` are hidden in ordinary project/history views so test artifacts do not dominate user-facing history. Records are **not deleted**. Set `AAOS_SHOW_DIAGNOSTIC_PROJECTS=1` to expose them for development/support. Clearing analysis history remains a separate explicit user action.

## Database migration

Apply Alembic migration `ab42d0e9c771_multi_question_analysis_sessions` after its parent migration. It creates the `analysis_sessions` table and adds nullable `investigations.analysis_session_id`, so older standalone investigations remain supported.

## Test status in this source environment

The focused regression set includes multi-condition filter ingestion, contract locking, the session/export model, diagnostic project visibility, cumulative PDF/DOCX sections, question-specific visualization selection, period-change explanations, and root-cause candidate-factor explanations.

Latest verification snapshot for this source package:

- Focused Python parser, planner, filter, controller, session, exporter, history and diagnostic suites: **105 passed**, 4 existing pandas `FutureWarning`s.
- Standalone Node visualization/report-selection suites: **26 passed** (17 analytical report-selection tests and 9 visualization renderer tests).
- TypeScript transpilation: all 5 changed frontend files transpile without syntax diagnostics. This is a syntax check, not a substitute for `tsc` with all React/Next dependencies installed.
- Python compilation: changed API, model, migration, runtime, and exporter modules compile successfully.
- Alembic: `ab42d0e9c771` is the single reported migration head.
- SQLAlchemy metadata smoke test: `analysis_sessions` and the nullable session relationship are created in an in-memory SQLite schema.

A complete web test run could not start because `react` is not installed in this source environment. A full desktop/web deployment check is still required in an environment with the project's locked dependencies installed. This editing environment also lacked runtime packages (`sqlglot`, `duckdb`, `python-jose`, and web React/Next packages), and network access was unavailable for installing them. The three-condition filter test validates ingestion, contract preservation, and pre-aggregation behavior in its deterministic fixture, but does not claim execution against DuckDB here. Therefore, missing-dependency/API integration and final Windows GUI-to-worker-to-export smoke tests remain release gates rather than being claimed as passed.
