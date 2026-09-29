# AA-OS v39 Release Status

**Date:** 2026-09-29
**Archive:** `minded.zip`
**Release Status:** **RELEASE READY**

## 1. What v39 contains

v39 implements the Session 24 P0 Estimand & Canonical Analytical Contract Cutover.

### Analytical Contract & Correctness Closures

- **Canonical Question Contract Cutover:** Single immutable, schema-grounded CanonicalQuestionContract representation authoritative from natural language question through estimand, method, and execution.
- **Intensive Metric Invariant:** Intensive metrics (e.g. `mpg`, `orbital_period`, `price`, `fare`, `mass`, `weight`) aggregate as `MEAN` across temporal and group dimensions rather than defaulting to `SUM`.
- **Extensive Metric Invariant:** Questions asking for "number of <measure>" (e.g. `passengers`) aggregate as `SUM` on the numeric metric rather than defaulting to record `COUNT`.
- **Causal Intent & Claim Ceiling Distinction:** Distinguishes requested claim (`CAUSAL`) from supported claim (`ASSOCIATION`) and records `claim_ceiling = ASSOCIATION` for observational datasets.
- **Root-Cause Investigation:** Questions asking "Why" map to `ROOT_CAUSE` / `DIAGNOSTIC` rather than generic comparison.
- **Role Resolution & Semantic Aliases:** Extended semantic aliases for order value, customer segment, and preposition stripping, ensuring correct target vs grouping role assignment.
- **Interaction Contrasts:** Stratified interaction contrasts calculated and verified across moderation strata.
- **Powered Null Invariant:** Adequately powered null contrasts award `NO_DETECTABLE_EFFECT`.
- **32-Question Real-Data Stress Baseline:** 100% completion across 8 real datasets (titanic, tips, penguins, mpg, diamonds, planets, taxis, flights) with independent ground-truth verification.
