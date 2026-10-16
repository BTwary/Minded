# AA-OS v39 — Stress-Verifier Certification & Release Verification

**Status:** **CERTIFIED RELEASE READY (32/32 Raw, 32/32 Clean)**  
**Authority:** `scripts/stress_verifier.py` independent ground truth engine  
**Release Gate Checks:** `stress_verifier_raw_32`, `stress_verifier_clean_32`, `adversarial_stress_verifier_suite`

---

## 1. Executive Summary & Certified Results

All 32 questions across 8 real datasets have been verified under the independent ground-truth verifier on both the **raw** dataset variant and the **clean** (category-coerced, non-redundant) dataset variant:

| Variant | Total Questions | CORRECT | Failing / Defect | Success Rate |
|---|---|---|---|---|
| **Raw Certification Datasets** | 32 | 32 | 0 | **100%** |
| **Clean Certification Datasets** | 32 | 32 | 0 | **100%** |
| **Adversarial Verifier Tests** | 8 | 8 passed | 0 | **100%** |

Independent ground-truth is evaluated strictly by `scripts/stress_verifier.py` using `pandas`, `scipy`, and `statsmodels` directly against the raw tabular data. Engine self-reported claims, ungrounded outputs, or missing contracts fail closed.

---

## 2. Engineering Closures & Architectural Hardening

### Fix 1 — Canonical Trend Coordinate Unification
- **Defect Identified (Historical Q26):** Trend slopes were previously regressed against the ordinal array index `0..n-1` instead of actual calendar time coordinates. Missing years (e.g. 1990, 1991, 1993 in `planets`) led to erroneous slope estimates (+144.6/yr instead of the true OLS slope +139.3/yr).
- **Resolution:** Introduced canonical temporal coordinate resolution in `analytical_math.py` and `analyst_answer.py`. All trend slopes, calendar intervals, and rate calculations regress against the true elapsed coordinate units (decimal years, days, or seconds).

### Fix 2 — Fail-Closed Temporal Spine
- **Defect Identified:** Fallback heuristics silently defaulted to row indices `np.arange(len(df))` when temporal metadata was missing, unparseable, or invalid.
- **Resolution:**
  - `BeliefEngine.bayes_factor_trend`: Explicitly returns `UNRESOLVED_TEMPORAL_COORDINATES` with neutral Bayes Factor ($BF=1.0$) when $x$ is missing; rejects invalid dimensions with `INVALID_TEMPORAL_COORDINATES`.
  - `canonical_time_coordinates`: Rejects missing coordinates, non-temporal numeric columns (grain `"none"`), unparseable series, and length mismatches with explicit errors. Zero silent ordinal degradation.
  - `execution_provider.py` & `transition.py`: Fail closed when time coordinates cannot be resolved chronologically.

### ROOT_CAUSE Statistical Rigor & Confounder Adjustment
- **Defect Identified:** Questions asking "Why" previously produced basic bivariate group differences without screening potential confounding factors.
- **Resolution:** Implemented an 8-step causal discovery and attenuation pipeline in `analyst_answer.py::_root_cause`:
  1. Discovery of candidate covariates (excluding primary grouping, target, and surrogate identifiers).
  2. Screening outcome association ($|r| \ge 0.10, p < 0.05$).
  3. Screening group variation ($F$-statistic / ANOVA across group levels).
  4. Controlled OLS regression adjustment: $Y \sim G + Z$.
  5. Attenuation computation: $\Delta \beta = \frac{|\beta_{\text{raw}}| - |\beta_{\text{adj}}|}{|\beta_{\text{raw}}|} \times 100\%$.
  6. Rigorous evidence classification: `CONFOUNDER_EFFECT_REVERSAL`, `CANDIDATE_CONFOUNDER_SUBSTANTIAL_ATTENUATION` ($\ge 30\%$), `CANDIDATE_CONFOUNDER_PARTIAL_ATTENUATION` ($10\%\text{--}30\%$), or `PERSISTENT_GROUP_DIFFERENCE`.
  7. Strict observational claim ceiling: `claim_ceiling = ASSOCIATION`.
  8. Nuanced, evidence-grounded analyst language: e.g., *"The observed data are consistent with confounding by X rather than an effect of G in isolation"*. Dogmatic assertions are prohibited.

### Fix 3 — Forecast Temporal Consistency
- **Defect Identified:** Drift calculations in `forecasting/engine.py` evaluated naive row spacing `(y[-1] - y[0]) / (n - 1)`.
- **Resolution:** Unified with canonical temporal coordinates $t$. Drift slope is computed as $(y[-1] - y[0]) / (t[-1] - t[0])$ and multi-step forecast horizons advance by true future elapsed calendar coordinates $\Delta t_h$.

---

## 3. Vendored Offline Certification Datasets

To ensure deterministic offline execution with zero network dependency, all 8 certification datasets are vendored directly in `data/certification/` and verified against immutable SHA-256 hashes:

| Dataset | Filename | SHA-256 Checksum |
|---|---|---|
| Diamonds | `diamonds.csv` | `9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4` |
| Flights | `flights.csv` | `237d834127d9c6355630d8f443a7a2377b5925923010009b59809ba0b67f4fac` |
| MPG | `mpg.csv` | `c14b8b855ea7ee86cb9736bf8caaf281c4685ca08826f3eb2acaccaaf40f0d5a` |
| Penguins | `penguins.csv` | `e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1` |
| Planets | `planets.csv` | `a6d10044887e17396974525a366f5fa2e4b34df70f491e64eb9943de0e3d3825` |
| Taxis | `taxis.csv` | `08d6d71784dbaa2651fee37fc03389754194c05d72d2d19cbc2c799dea6ac09d` |
| Tips | `tips.csv` | `e54cc4d2ce1bff65d32ca60b3e4b802e06bde1d7e7caf6f796f6bf7370e863b0` |
| Titanic | `titanic.csv` | `81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2` |

Loaded via `packages.analytics_core.src.data.certification_data.load_all_certification_datasets()`.

---

## 4. Historical Pre-Fix Baseline (Reference Only)

Prior to Fix 1, Fix 2, ROOT_CAUSE hardening, and Fix 3, the independent verifier diagnosed 9 defects across the 32 questions (23/32 passing):

| # | Question | Historical Class | Root Cause Diagnosis | Status in Release 1.0 |
|---|---|---|---|---|
| 6 | tip ~ total_bill | AGGREGATION_WRONG | Correlation contract requested SUM | Fixed: Intensive metric preservation |
| 15 | mpg ~ weight | CONTRACT_WRONG | Spurious grouping `origin` | Fixed: Semantic role resolution |
| 17 | fuel efficiency over years | CONTRACT_WRONG | Spurious grouping `origin` | Fixed: Temporal contract normalization |
| 19 | horsepower vs weight | CONTRACT_WRONG | Spurious grouping `origin` | Fixed: Multivariate role filtering |
| 22 | avg price by clarity | INCOMPLETE_OUTPUT | Truncation of low-frequency clarity bins | Fixed: Complete grouped output rule |
| 23 | Why Fair > Ideal | CONTRACT_WRONG | Routed `group_difference` instead of root cause | Fixed: Why grammar & confounder analysis |
| 26 | orbital period over years | CONTRACT_WRONG + NUMERICALLY_WRONG | Ordinal index regression (+144.6/yr vs true +139.3/yr) | Fixed: Fix 1 Canonical trend coordinate |
| 28 | borough with highest fare | CONTRACT_WRONG | Spurious time field `pickup` | Fixed: Contract pruning |
| 30 | passengers over time | CONTRACT_WRONG | Spurious grouping `month` | Fixed: Extensive metric SUM invariant |

All 9 historical defects are permanently resolved and covered by regression tests.

---

## 5. Release Gate Manifest Integration

The 32-question stress verification and adversarial verifier tests are permanently integrated into `tests/independent_release/manifest.toml`:
1. `stress_verifier_raw_32`: Mandates 32/32 exit code 0 on raw datasets.
2. `stress_verifier_clean_32`: Mandates 32/32 exit code 0 on clean datasets.
3. `adversarial_stress_verifier_suite`: Mandates 8/8 passing adversarial tests in `tests/test_stress_verifier.py`.
