# AA-OS Clean-Room Acceptance Benchmark

This benchmark is an independent acceptance layer for the Autonomous Analytical Intelligence OS.

## Design rules

- Expected properties are computed independently with pandas/NumPy/scikit-learn where possible.
- No AA-OS result object is reused to manufacture its own expected answer.
- Test data is deterministic; uncontrolled randomness is forbidden.
- Row-order permutations, duplicate rows, renamed schemas, missingness, join fanout, Simpson's paradox, censoring, temporal leakage, and method admissibility are explicit acceptance cases.
- The benchmark never converts a missing execution dependency into a PASS.

## Commands

Engine-independent oracle suite:

```bash
python -m unittest -v tests.independent_release.test_clean_room_acceptance
```

Combined focused regression:

```bash
python -m unittest -v \
  tests.independent_release.test_clean_room_acceptance \
  tests.test_adopted_scientific_integrity_hardening \
  tests.test_method_registry_authority \
  tests.test_hard_sql_isolation \
  tests.test_event_sourced_recovery \
  tests.test_execution_recovery_hardening
```

Explicit clean-room runner:

```bash
python scripts/run_clean_room_acceptance.py
```

The runner reports `controller E2E: BLOCKED` unless DuckDB, Polars, and PyArrow are installed. A blocked E2E environment is never represented as a successful controller acceptance result.

## Calibration profile authority
Calibration profiles are self-identifying and integrity-hashed. Production use requires an explicit population scope and an explicit environment match for the calibrated model/scope. A mismatched profile is never applied.
