# Current Release Certification Record

**Date:** 2026-09-30  
**Artifact lineage:** `MindEd_AAOS_v40_Release1.0_candidate.zip`  
**Certification state:** **RELEASE 1.0 CANDIDATE — VERIFIED BENCHMARK & DETERMINISTIC CORE**  

This is the current certification record for the AA-OS v40 release-candidate artifact. It supersedes the historical v32 pre-integration record from 2026-09-26.

## Verified Core Capabilities (Release 1.0 Candidate)

- **32/32 Real-Data Benchmark Acceptance:** Independently verified on 8 real certification datasets across both raw and clean variants under `scripts/stress_verifier.py` with persisted contracts.
- **Adversarial Stress Verifier:** 8/8 adversarial verifier tests pass, validating that numeric drifts, swapped polarities, and mutated contracts fail closed.
- **Deterministic Question Compiler:** Paraphrase consistency verified across association and causal grammar (e.g. "Does plan_tier affect churn?" and "Is churn associated with plan tier?" converge on identical canonical contracts).
- **Domain Generalization:** Unseen business questions (retention rates, customer segments, traffic sources, conversion rates) resolve deterministically.
- **Forecasting Invariants:** Duplicate timestamp aggregation is policy-governed (semantic-default mean for intensive metrics, sum for extensive metrics, explicit override and fail-closed reject mode).
- **Code & Schema Audits:** All Python source compiles clean; 185 files pass canonical serialization audit; Alembic schema reaches head with zero migration drift; Next.js frontend lock verified.

## Scope Boundaries & Environment Certification Gate

- **Bounded Question Grammar:** Certified for deterministic analyst-question grammar (ranking, descriptive summaries, correlation, group comparisons, rolling-origin forecasts, and observational root-cause analysis). Arbitrary colloquial language outside this grammar is out-of-scope for Release 1.0.
- **Dependency-Complete Clean-Room Verification:** The core deterministic spine, adversarial harness, and question compiler suites pass in the local test environment. Full execution of the complete multi-engine independent release suite and optional analytical engines (`duckdb`, `polars`, `pyarrow`, `sqlglot`) is governed by the 31-check manifest and requires the dependency-locked virtual environment during final CI/clean-room certification.

---
*Historical Note:* The previous v32 non-certification blockers (unintegrated canonical question layer, rate/correlation verdict inconsistencies, and unverified real-data acceptance) have been resolved in this release lineage.
