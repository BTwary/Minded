# Phase 2 P0: final verification-gate correction

## Why this revision exists

An independent audit reported that the analyst-result overlay could restore a conclusive hypothesis verdict after an earlier gate, and that the previous patch had not been proven on the production path. I independently inspected the Phase 2 source and found an additional evidence-lineage weakness: the controller accepted any historical `VERIFIED` record across both supporting and contradicting evidence IDs. That is not sufficient to substantiate the leading hypothesis.

## Corrective changes

1. The final verification gate now runs after the analyst overlay, execution/contract reconciliation, grouping-note logic, and data-quality qualification, immediately before the final provenance manifest is built. Downstream code after this gate may downgrade a verdict, but does not upgrade one.
2. A conclusive hypothesis verdict requires `all_verifications_passed is True` and independently verified supporting evidence. Truthiness is not accepted as proof (`"false"` is not `True`).
3. Evidence lineage now considers supporting evidence only. Contradictory evidence cannot satisfy the support requirement.
4. A supporting evidence item counts only when both the evidence row is currently `VERIFIED` and its latest independent verification record is `VERIFIED`; a historical pass followed by failure no longer qualifies.
5. The blocked outcome is `INCONCLUSIVE`, confidence `0.0`, capped posterior, refusal wording, and an audit event. The gate does not reclassify descriptive `OBSERVED` outputs, which remain subject to their separate contract path.
6. Added pure-policy tests plus a controller source-order regression check.

## Checks independently run in this workspace

- 10 focused verification-gate tests passed.
- `py_compile` passed for the controller, verification helper, and regression test.
- Static source-order check passed: final gate is after the analyst overlay and before the final manifest; no conclusive verdict upgrade occurs after the gate.
- Archive integrity and SHA-256 recorded in the accompanying report.

## Not established

- No full production-path injection run: this environment is Python 3.13 and lacks `duckdb` and `polars`; the existing full controller integration test therefore was not run.
- The original 120-case matrix and 126-case real-path matrix were not rerun.
- The audit's filter omission, wrong grouping, date-filter crash, scalar aggregate refusals, missing `python-docx` / `reportlab` / `pypdf` lock entries, and 26 full-suite failures have not been resolved by this P0 patch.
- Windows EXE, live HTTP startup, clean-install release gate, and end-to-end investigation behavior remain unverified.

This is a stronger targeted P0 correction, **not release certification**.
