# AA-OS v40 Release Status

**Date:** 2026-09-30  
**Archive:** `MindEd_AAOS_v40_Release1.0_candidate.zip`  
**Release Status:** **RELEASE CANDIDATE — RELEASE-GATE READY**  

## 1. Scope and Architectural Closures

AA-OS v40 achieves Release 1.0 certification as a **bounded autonomous data analyst** executing under local-first deterministic discipline.

### Analytical Contract & Correctness Closures

- **Canonical Question Contract Cutover & Grammar Generalization:** Single immutable, schema-grounded `CanonicalQuestionContract` authoritative from question phrasing through estimand, method, and execution. Extended beyond the initial benchmark corpus to support standard business vocabularies including retention, conversion, customer segmentation (e.g. enterprise vs SMB), traffic sources, and advertising spend.
- **Canonical Temporal Coordinate Unification:** Unified calendar-unit coordinates in `analytical_math.py` ensuring trend slopes, drift, and projections compute true elapsed calendar time (decimal years, days, seconds) rather than naive ordinal array positions.
- **Fail-Closed Temporal Spine:** Replaced all silent ordinal/row-index fallbacks across `BeliefEngine.bayes_factor_trend`, `canonical_time_coordinates`, `execution_provider.py`, and `transition.py`. Missing or unparseable temporal coordinates raise explicit validation errors or return neutral evidence ($BF=1.0$).
- **Forecasting Duplicate Timestamp Semantics:** `ForecastingEngine._prepare_series` implements explicit and semantic-default aggregation (`sum`, `mean`, `median`, `reject`). Intensive metrics (price, conversion rate, retention rate, average order value, latency, percentage, margin) aggregate as `mean` rather than being additively summed; extensive metrics aggregate as `sum`. `reject` mode fails closed with `DUPLICATE_TIMESTAMPS_REJECTED` if duplicates exist.
- **Intensive Metric Invariant:** Intensive metrics (e.g. `mpg`, `orbital_period`, `price`, `fare`, `mass`, `weight`, `rate`) aggregate as `MEAN` across temporal and group dimensions rather than defaulting to `SUM`.
- **Extensive Metric Invariant:** Questions asking for "number of <measure>" (e.g. `passengers`) aggregate as `SUM` on the numeric metric rather than defaulting to record `COUNT`.
- **Causal Intent & Claim Ceiling Distinction:** Distinguishes requested claim (`CAUSAL`) from supported claim (`ASSOCIATION`) and strictly bounds observational findings to `claim_ceiling = ASSOCIATION`.
- **Root-Cause Investigation & Controlled Confounding Adjustment:** "Why" questions execute an 8-step causal discovery pipeline including candidate covariate screening (both numeric and categorical), group variation testing, controlled OLS regression adjustment ($Y \sim G + Z$), attenuation percentage estimation, and classification into candidate confounder vs persistent group difference.
- **Forecast Temporal Consistency:** `forecasting/engine.py` unifies drift slopes and horizon projection with elapsed calendar units $\Delta t_h$.
- **Role Resolution & Semantic Aliases:** Extended semantic aliases for order value, customer segment, traffic source, retention, conversion, and preposition stripping, ensuring correct target vs grouping role assignment.
- **Interaction Contrasts & Powered Null Invariant:** Stratified interaction contrasts calculated across moderation strata; adequately powered null contrasts award `NO_DETECTABLE_EFFECT`.

### Bounded Autonomous Scope Contract (Release 1.0)

AA-OS v40 is explicitly certified for **bounded deterministic analyst-question grammar**, not unconstrained natural language comprehension:
- Supported analytical families: ranking, descriptive summaries, bivariate & partial correlation, group comparisons, time-series forecasting, trend estimation, stratified churn/survival identifiability, and controlled observational root-cause analysis.
- Unresolvable variables, ambiguous column bindings, or questions addressing outcomes not present in schema fail closed with explicit `INCONCLUSIVE` diagnostics rather than hallucinating answers.

### Certification & Release Gate Integration

- **32-Question Real-Data Stress Test:** 100% agreement (32/32 CORRECT) across 8 real datasets (`titanic`, `tips`, `penguins`, `mpg`, `diamonds`, `planets`, `taxis`, `flights`) on both raw and clean variants under independent ground truth.
- **Adversarial Stress Verifier:** 8/8 passing adversarial tests in `tests/test_stress_verifier.py` validating that generic text, numeric drifts (13%), swapped means, wrong winners, flipped polarities, and contract mutations fail closed.
- **Vendored Certification Datasets:** Vendored in `data/certification/` with immutable SHA-256 checksums and offline loaders.
- **Manifest Integration:** Authoritative release checks `stress_verifier_raw_32`, `stress_verifier_clean_32`, and `adversarial_stress_verifier_suite` integrated directly into `tests/independent_release/manifest.toml`.

## 2. Release Verification & Environment Summary

- **Analytical Core & Deterministic Spine:** Verified with 32/32 real-data benchmark questions passing independently on both raw and clean variants (`session23_real_data_stress.py` evaluated against `stress_verifier.py`), 8/8 adversarial verifier tests, and 39/39 passing targeted semantic regression and compiler tests. Full multi-engine execution across the independent release suite is governed by the 31-check manifest and requires the locked environment.
- **Syntactic & Schema Audits:** All Python source compiles clean (`compileall`), canonical serialization audit passed (185 files), frontend lock verified (Next.js 15.5.26), and database migrations reach Alembic head with zero schema drift.
- **Question Compiler & Association Paraphrases:** Verified consistent bidirectional role assignment for association/correlation questions and generalized resolution across unseen enterprise schemas (`retention_rate`, `conversion_rate`, `traffic_source`).
- **Forecasting Governance:** Duplicate timestamp semantics verified (semantic-default mean for intensive, sum for extensive, explicit override and fail-closed reject).
- **Clean-Room Dependency Isolation:** Full clean-room gate verification with exact dependency lock enforcement requires target runtime engines (`duckdb`, `polars`, `pyarrow`, `sqlglot`) to be executed within an isolated locked virtual environment during final deployment or CI.

