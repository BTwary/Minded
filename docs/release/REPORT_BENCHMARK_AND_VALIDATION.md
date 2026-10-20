# MindEd Report Benchmark and Validation Contract

## What this release adds

MindEd now has a versioned, machine-checkable contract for the quality benchmark that is intended to govern report generation. The contract and validator are:

- `packages/schemas/src/report_benchmark.schema.json`: JSON Schema Draft 2020-12 for a normalized stakeholder-facing report.
- `packages/schemas/src/benchmark_run_metrics.schema.json`: schema for golden-set evaluation metrics.
- `packages/analytics_core/src/reporting/report_benchmark.py`: Pydantic contract, cross-reference/lineage checks, hard-fail rules, quality rubric, and Bronze/Silver/Gold decision logic.
- `scripts/validate_report_benchmark.py`: CLI for validating a report JSON and optionally a golden-suite metrics JSON.
- `tests/benchmark/test_report_benchmark_validator.py`: regression tests for completeness, numeric traceability, consistency, causation, reconciliation, exclusions, forecasts, and schema parity.

## Use the validator

First serialize a report into the `ReportBenchmarkContract` shape. The payload must come from saved execution/evidence records; do not create facts or evidence merely to satisfy the schema.

```bash
python scripts/validate_report_benchmark.py path/to/report_contract.json
python scripts/validate_report_benchmark.py path/to/report_contract.json --metrics path/to/golden_metrics.json
```

Exit codes: `0` means report-level publishability passed, `1` means the report failed the hard gate, and `2` means the input/runtime could not be read. JSON output includes a grade, completeness rate, numeric traceability rate, quality rubric, errors, and warnings.

## Hard-fail rules checked

- Every objective question has exactly one conclusion answer, using the same question ID set.
- Metric names used by findings have definitions in scope.
- Every referenced fact/evidence/finding/chart exists, and findings/claims/conclusions use verified evidence.
- Numeric claims register their literal values against computation-backed fact IDs. Summary key numbers, finding narratives, conclusions, tables, and chart points are also checked against facts.
- Text/table/chart values cannot contradict the saved fact value.
- Unhedged causal claims require a supported causal-design record, verified evidence and recorded assumption checks; observational-only/unsupported cause claims fail.
- Forecasts require lower and upper uncertainty bounds and an interval level.
- Row-changing cleaning steps must be disclosed, and failed reconciliation needs an explicit explanation.
- Data quality reports missingness, duplicates, outlier handling and at least one reconciliation check.
- Executive summary is at most six non-empty lines.

## Grade interpretation

- **FAIL**: one or more hard-fail conditions or schema errors.
- **BRONZE**: report-level contract passes; claims and displayed values are traceable under this contract.
- **SILVER**: Bronze plus golden-suite metrics proving exact rerun matching, full key-figure traceability, passed reconciliation/statistical validity, and quality scores of at least 1.3.
- **GOLD**: Silver plus planted-issue detection greater than 90%, zero false insights on no-signal data, quality score at least 1.7, at least 20 golden datasets, and multiple human reviewers with inter-rater agreement recorded.

The Bronze validator can check only what is represented in the structured contract. The golden-suite metrics are supplied separately because one report cannot establish rerun reproducibility, detection rate on planted defects, false-insight rate on no-signal data, or human-review agreement.

## Limits and integration status

This first implementation is an executable contract and release benchmark layer. It is not an automatic parser for arbitrary PDFs, nor does it yet synthesize all the metadata and data-quality diagnostics from every existing session export path. Production integration should construct the normalized contract from the authoritative session, saved query/execution record, verification evidence, and result payload, then call `validate_report_contract` before publication. Avoid silently bypassing the validator when the required evidence is missing.

Schema presence alone does not prove correctness. Statistical test appropriateness, reconciliation against external systems, absence of all unregistered claims in arbitrary free text, and human-quality ratings still require golden datasets, independent reference calculations, and review. Maintain a dataset-level regression suite with clean data, injected defects, no-signal cases, confounding/Simpson's paradox, and experimental validity defects before claiming Silver or Gold.

## Verification snapshot for this change

- `pytest -q tests/benchmark/test_report_benchmark_validator.py`: **15 passed**.
- `pytest -q tests/benchmark/test_report_benchmark_validator.py tests/independent_release/test_multi_question_analysis_session_contract.py tests/independent_release/test_session_exporter_cumulative_sections.py`: **23 passed**.
- Both JSON Schema files pass JSON Schema Draft 2020-12 meta-schema validation.
- The CLI validates `tests/fixtures/report_benchmark/valid_report.json` with exit code `0`, a 100% structural completeness score, and a 100% numerical traceability score.
- A broader command that also ran the existing `tests/benchmark` suite produced **27 passed and 1 failed**. The unrelated failure is `test_association_binary_outcome_is_admissible`: it expected `association_categorical_binary`, but the current method selector returned `generic_diagnostic_battery`. That should be investigated separately before release.

## Not yet certified

The schema, validator, and CLI are implemented and tested. The existing session PDF/DOCX exporters do not yet construct a complete `ReportBenchmarkContract` from every saved session, so this validator is not yet an automatic publication gate for every in-app export. The next integration step is to map the authoritative session, dataset-profile, execution, verification and review records into this normalized contract and fail publication when the contract fails. The full 20–50-dataset golden set and multi-reviewer quality exercise also remain to be populated and run; the validator deliberately does not fabricate those benchmark metrics.
