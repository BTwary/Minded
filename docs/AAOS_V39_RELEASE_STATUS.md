# AA-OS v39 Release Status

**Date:** 2026-09-29  
**Archive:** `minded.zip`  
**Release Status:** **RELEASE READY**  

## 1. Scope and Architectural Closures

AA-OS v39 achieves full Release 1.0 certification for autonomous data analysis under local-first deterministic execution.

### Analytical Contract & Correctness Closures

- **Canonical Question Contract Cutover:** Single immutable, schema-grounded `CanonicalQuestionContract` representation authoritative from natural language question through estimand, method, and execution.
- **Canonical Temporal Coordinate Unification:** Unified temporal coordinate calculation in `analytical_math.py` ensuring trend slopes, drift, and time steps compute true elapsed calendar units (decimal years, days, seconds) rather than naive ordinal array positions.
- **Fail-Closed Temporal Spine:** Replaced all silent ordinal/row-index fallbacks across `BeliefEngine.bayes_factor_trend`, `canonical_time_coordinates`, `execution_provider.py`, and `transition.py`. Missing or unparseable temporal coordinates raise explicit validation errors or return neutral evidence ($BF=1.0$).
- **Intensive Metric Invariant:** Intensive metrics (e.g. `mpg`, `orbital_period`, `price`, `fare`, `mass`, `weight`) aggregate as `MEAN` across temporal and group dimensions rather than defaulting to `SUM`.
- **Extensive Metric Invariant:** Questions asking for "number of <measure>" (e.g. `passengers`) aggregate as `SUM` on the numeric metric rather than defaulting to record `COUNT`.
- **Causal Intent & Claim Ceiling Distinction:** Distinguishes requested claim (`CAUSAL`) from supported claim (`ASSOCIATION`) and strictly bounds observational findings to `claim_ceiling = ASSOCIATION`.
- **Root-Cause Investigation & Controlled Confounding Adjustment:** "Why" questions execute an 8-step causal discovery pipeline including candidate covariate screening, group variation testing, controlled OLS regression adjustment ($Y \sim G + Z$), attenuation percentage estimation, and classification into candidate confounder vs persistent group difference. All statements use nuanced, evidence-grounded phrasing without dogmatic assertions.
- **Forecast Temporal Consistency:** `forecasting/engine.py` unifies drift slopes and horizon projection with elapsed calendar units $\Delta t_h$.
- **Role Resolution & Semantic Aliases:** Extended semantic aliases for order value, customer segment, and preposition stripping, ensuring correct target vs grouping role assignment.
- **Interaction Contrasts:** Stratified interaction contrasts calculated and verified across moderation strata.
- **Powered Null Invariant:** Adequately powered null contrasts award `NO_DETECTABLE_EFFECT`.

### Certification & Release Gate Integration

- **32-Question Real-Data Stress Test:** 100% agreement (32/32 CORRECT) across 8 real datasets (`titanic`, `tips`, `penguins`, `mpg`, `diamonds`, `planets`, `taxis`, `flights`) on both raw and clean variants under independent ground truth.
- **Adversarial Stress Verifier:** 8/8 passing adversarial tests in `tests/test_stress_verifier.py` validating that generic text, numeric drifts (13%), swapped means, wrong winners, flipped polarities, and contract mutations fail closed.
- **Vendored Certification Datasets:** Vendored in `data/certification/` with immutable SHA-256 checksums and offline loaders.
- **Manifest Integration:** Authoritative release checks `stress_verifier_raw_32`, `stress_verifier_clean_32`, and `adversarial_stress_verifier_suite` integrated directly into `tests/independent_release/manifest.toml`.

## 2. Release Gate Verification Summary

- All 31 manifest gate checks active and verified.
- Zero open blocker items.
- Migration drift: clean schema verified to Alembic head.
- Clean-room and offline-first autonomy verified.
