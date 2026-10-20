# Adjudicated acceptance benchmark

## Scoring contract

The acceptance score separates four outcomes rather than collapsing them into one accuracy number:

- **Correct answers:** answerable case and structured result equals the adjudicated expected value.
- **Wrong answers:** incorrect/unsupported answer, or a refusal that does not meet the safe-refusal label.
- **Safe refusals:** unanswerable case explicitly refused safely.
- **Answerable refusals:** case was adjudicated answerable, but the system refused instead of answering.

The scorer fails closed if the case count differs from the configured count, IDs are missing/duplicated, or the results omit or add cases. Numeric values are compared at a fixed 9-decimal normalization; dictionaries are key-order independent. The case fixture is the source of truth for answerability and expected values.

## Required input files

The original adjudicated fixture must be supplied as JSON with exactly 152 records by default:

```json
[
  {"id": "CASE-001", "answerable": true, "expected": {"value": 42}},
  {"id": "CASE-002", "answerable": false, "expected": null}
]
```

The production run must emit one result for every case ID:

```json
[
  {"id": "CASE-001", "outcome": "ANSWER", "actual": {"value": 42}},
  {"id": "CASE-002", "outcome": "SAFE_REFUSAL", "reason": "The request is ambiguous."}
]
```

`outcome` must be `ANSWER`, `SAFE_REFUSAL`, or `REFUSAL`. The run should preserve the raw per-case execution receipt, contract hash, dataset version/fingerprint, SQL, verification status, and refusal reason alongside the normalized `actual` result. Do not convert a missing result into a refusal or silently drop a failed case.

## Score a captured run

```bash
python scripts/summarize_adjudicated_benchmark.py \
  --cases path/to/original_152_cases.json \
  --results path/to/updated_archive_results.json \
  --output release_evidence/adjudicated_acceptance_report.json
```

The report contains per-case classifications and separate totals for correct answers, wrong answers, safe refusals, and answerable refusals. It also reports answerable-case accuracy. Keep the case fixture, raw run results, scored report, dependency lock, and source revision together as the acceptance artifact.

## Current limitation

This checkout contains a 120-case independent audit matrix and a separate 32-case stress-results fixture. Those are useful existing assets, but their presence alone does not establish that they are the same original 152 adjudicated cases. The original combined fixture and a fresh production-run result file were not present in the supplied archive, so this release does **not** claim a 152-case benchmark pass. Run the scorer once the original fixture and captured run results are available.
