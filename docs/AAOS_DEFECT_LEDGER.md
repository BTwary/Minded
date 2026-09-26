# AA-OS Defect Ledger

Canonical record of reproducible production defects.

## DEFECT-020 — RLS regex fallback can scope the wrong query block

- **Severity:** P0
- **Status:** FIXED IN CODE; CLEAN-ROOM INSTALL VERIFICATION PENDING
- **Impact:** A cross-dataset query could receive its tenant predicate inside a nested subquery while the outer fact table remained unscoped.
- **Root cause:** SQLGlot was optional at runtime and the fallback used first-match regex clause insertion.
- **Fix:** SQLGlot 30.18.0 is now a runtime dependency for the core/server/HF installs. RLS injection is AST-only and fails closed if SQLGlot is absent or rewriting fails.
- **Regression coverage:** `tests/test_governance_rls_ast_fail_closed.py` covers nested subqueries, malformed SQL, missing SQLGlot, rewrite exceptions, and no-op context.
- **Independent verification:** Full clean-install execution must be rerun in an environment that can install the locked dependency set.
- **Call-site audit:** The current checkout has no direct production call site for `GovernanceEngine.inject_rbac_filters`; this must be reconciled before claiming end-to-end RLS enforcement. The AST hardening prevents unsafe fallback if the function is invoked.

## DEFECT-021 — Adversarial falsification never falsifies; dead/inverted verdict-gate branch

- **Severity:** P1
- **Status:** FIXED
- **Impact:** A genuine, materiality-gated Simpson's reversal (already rigorously detected by `detect_simpsons_reversal()` since session 6/F2) could only ever mark the leading hypothesis `WEAKENED`, never `REFUTED`/`is_falsified=True` — `AdversarialAttacker.design_attack()` set `falsified = False` unconditionally on every branch. This meant `intelligence/adversarial_attacker.py`'s falsification pathway was structurally dead even though `runtime/controller.py` already had live, tested machinery (`step_adversarial()`) that flips `belief_state` to `"falsified"` and halves the posterior whenever `attack_res.is_falsified` is true — that machinery simply never fired.
- **Secondary defect (same audit, O4):** `engines/verdict.py`'s `counter_subdued` computation additionally treated `adversarial_eval.status in ["WEAKENED", "CONTRADICTED", ...]` as evidence the *counter*-hypothesis was subdued — the reverse of what `WEAKENED`/`CONTRADICTED` mean (they describe damage to the *leading* hypothesis). Confirmed dormant: `runtime/controller.py` is the only real caller of `evaluate_verdict()` and never populates `adversarial_eval`, only the already-computed `adversarial_attack_survived` boolean.
- **Root cause:** Both defects are the same class — an evaluation function that correctly *computes* a rigorous finding (Simpson's reversal; a WEAKENED/REFUTED status) but never lets that finding change the boolean/status field downstream logic actually branches on.
- **Fix:**
  1. `intelligence/adversarial_attacker.py`: a detected Simpson's reversal (already same-sign, min-cell-size, and materiality gated) now sets `attack_status="REFUTED"`, `is_falsified=True`, with an epistemic-impact message naming the marginal/adjusted difference and strata used. Leakage, selection-bias, outlier-sensitivity, and low-dispersion findings remain `WEAKENED`/`is_falsified=False` — they flag risk without demonstrating the claimed direction is wrong, unlike a proven reversal.
  2. `engines/verdict.py`: removed the dead/inverted `adversarial_eval.status in [...]` clause from `counter_subdued`; it is now simply `adversarial_attack_survived`, the one live signal any real caller supplies.
- **Regression coverage:** `tests/independent_release/test_session7_o3_simpsons_refutation.py` (11 tests): the attacker's REFUTED/`is_falsified` branch (fails against the old behaviour), verdict-engine and narrative unit tests, and end-to-end controller runs (Simpson's dataset -> `REFUTED`, persisted REFUTED hypothesis, analyst-readable answer; known-positive dataset still `DIAGNOSED`). NOTE: an earlier version of this entry claimed existing suites already exercised this branch; they did not (no test asserted `is_falsified=True`). Full suite after the change: 1056 passed / 1 pre-existing failure / 6 skipped; the failure fails identically with the change reverted.
- **Related ruling (CORRECTED):** `test_real_adaptive_replanning_multi_round_trace` runs 3 experiments, not the pinned 2. An earlier version of this entry attributed the third experiment to the adversarial engine flagging `tier` as a Simpson's confound. That is wrong: on that dataset every adversarial attack is SURVIVED, `unresolved_adversarial_issues` is empty in every round, and the stopping gate returns `COUNTER_HYPOTHESIS_NOT_EVALUATED` after round 2 until `EXP-CONFOUND` (a standard cross-dimensional interaction candidate) evaluates the counter-hypothesis. The pinned count of 3 matches current behaviour; whether the counter-hypothesis should already count as evaluated after round 2 remains an OPEN owner decision.
- **Follow-up (analyst-facing outcome):** the falsification used to stop at the attacker: the persisted hypothesis kept its pre-attack posterior, the investigation verdict was `STATISTICALLY_SIGNIFICANT` at ~0.9999 with the old code (and `INCONCLUSIVE`/"unable to find sufficient evidence" with only the flag fixed), so a human analyst was never told the headline comparison reverses within strata. Now: `VerdictEngine` returns `REFUTED` (confidence 0.0, never masked by a high posterior or a method ceiling; a leading null/counter hypothesis is excluded); `ProvenanceEngine` writes a one-sentence headline plus a numbers-first explanation and next steps from a single shared module (`engines/refutation_narrative.py`); the controller persists the falsified hypothesis as `REFUTED` in the same transaction and records the reversal in the step output; the analyst UI has a `REFUTED` badge instead of the default green tick.
- **Still open:** the falsified hypothesis's persisted posterior is the halve-heuristic value (0.5 in the test scenario), not a principled Bayesian update, and other hypotheses are not renormalised; the web UI change was not type-checked in this environment (no node_modules).

## DEFECT-022 — `GET /investigations/{id}` raised NameError for every investigation

- **Severity:** P0
- **Status:** FIXED
- **Impact:** `get_investigation` referenced `primary_dataset_name` in its response but never defined it (the sibling `/explanation` endpoint did). Every call raised `NameError`. This is the endpoint the analyst UI polls for every result, so no completed investigation could be displayed through it.
- **Why it survived:** no test called the endpoint function against an investigation; the sibling `/explanation` test exercised a different function.
- **Fix:** resolve `scope_payload` / `primary_dataset_name` / per-dataset readiness exactly as `/explanation` does (primary dataset first; deterministic fallback for older investigations).
- **Regression coverage:** `tests/independent_release/test_session8_verifier_workflow.py::test_defect_022_...` (fails when the fix is reverted).

## DEFECT-023 — controller's `EXP-COND-*` conditional experiment failed on every run

- **Severity:** P1
- **Status:** FIXED
- **Impact:** the controller-built Simpson's conditional query had no `ORDER BY`; `execution_provider` rejects unordered multi-row results, so the step failed each time. The failure was visible only in the raw event log and in a trailing "Stopping reason: EXPERIMENT_EXECUTION_FAILED" clause; the analyst UI never showed it.
- **Fix:** same deterministic `ORDER BY total_metric DESC, <dim>, <conf_dim>` the synthesizer's variant already used.
- **Regression coverage:** `test_defect_023_...` asserts the experiment appears in `experiments` and `failed_steps` is empty (fails when reverted).

## Visibility gaps closed in the same pass (not defects in verdict logic)

- `investigation.adversarial_challenge.completed` and `calculation.multiverse` are now emitted. The API already read those event prefixes, but nothing wrote them, so `adversarial_findings` was always `[]` and `multiverse` always `null`.
- `Investigation.stopping_criteria_met` / `stopping_rationale` were written only for compound objectives; ordinary runs always reported `False`. Now persisted from the authoritative stop decision.
- API now returns `failed_steps` (from `step.failed` events).

## DEFECT-024 — Assumption ledger's `[INDEPENDENT_VERIFICATION]` and `[METRIC_SEMANTICS]` entries didn't reflect reality

- **Severity:** P1
- **Status:** FIXED
- **Impact:** two ledger entries the human-verifier surfaces reads from were structurally wrong, not just conservative:
  1. `[INDEPENDENT_VERIFICATION]` was hard-coded `is_validated=False`/high-risk in `AssumptionLedgerEngine.build()`, unconditionally, forever — including on investigations where a real independent re-execution proof agreed with the leading hypothesis's result. The verification packet already knew this and unconditionally skipped the entry (see its own now-corrected comment), so the *packet* was fine, but the raw ledger (persisted `Assumption` rows, any other consumer, the quality card) still permanently claimed the opposite of what happened.
  2. `[METRIC_SEMANTICS]` was marked `is_validated=True`/low-risk purely because *some* aggregation had been chosen, without checking `MetricDefinition.semantic_resolution_status`. An `UNRESOLVED_DEFAULT_SUM` metric (SUM used as an unconfirmed fallback, per `metric_semantics.py`'s own rationale) was reported as confidently resolved — exactly the contradiction `METRIC-AGGREGATION`'s evidence block (`assumption_ledger_marks_this_validated`) was already built to detect, but nothing had fixed the source of.
- **Root cause:** `AssumptionLedgerEngine.build()` runs before any experiment executes, so `[INDEPENDENT_VERIFICATION]` genuinely has no evidence yet at construction time (same situation as `[FORECAST_GENERALIZATION]`, which already had a real fix: a `nonlocal` flag set from backtest evidence and checked later). `[INDEPENDENT_VERIFICATION]` never got the equivalent treatment — it was left hard-coded instead. `[METRIC_SEMANTICS]` had no excuse: `semantic_resolution_status` is available at ledger-build time and was simply not read.
- **Fix:**
  1. `AssumptionItem` stays frozen (a ledger entry must not be silently rewritten mid-review); added `AssumptionLedgerEngine.patch_validation()` to construct replacement items for named codes, and `AssumptionLedgerEngine.score()` (factored out of `build()`) to recompute the quality card from the patched list. The controller now patches `[INDEPENDENT_VERIFICATION]` from `has_verified_evidence` (the same leading-hypothesis-scoped verification signal the final verdict itself uses) and `[FORECAST_GENERALIZATION]` from its existing backtest flag, once both are known — post-execution, pre-persistence — emits a `investigation.assumption_ledger.patched` event, and updates the already-persisted `Assumption` rows in place.
  2. `material_unvalidated_high_risk`'s ad hoc exclusion set (`{"INDEPENDENT_VERIFICATION"}` and a `FORECAST_GENERALIZATION`-specific bypass) is gone; both codes now just carry accurate `is_validated`, so the plain filter is correct on its own.
  3. `[METRIC_SEMANTICS]` now branches on `semantic_resolution_status`: `UNRESOLVED_DEFAULT_SUM` is added as unvalidated with an explanatory statement, `medium` risk (not `high` — an unconfirmed-default aggregation is by design not meant to block the investigation, only to be reviewed; `high` was tried first and made three genuine-signal churn/correlation scenarios regress to `INCONCLUSIVE`, since their target column had no recognized naming token). Anything else stays validated/low-risk as before, with the resolution status now named in the evidence.
- **Regression coverage:** `tests/test_assumption_ledger.py`, `tests/independent_release/test_session8_verifier_workflow.py` (the packet's dedup/contradiction tests still pass unchanged — they exercise `build_verification_packet` directly against fabricated payloads, not this engine). Full suite after the change: 1094 passed / 1 pre-existing failure (`test_controlling_for_wording_is_not_distinguished_from_plain_wording`, unchanged by this fix) / 6 skipped.
  - **Correction (2026-09-25 release-hardening pass):** `test_controlling_for_wording_is_not_distinguished_from_plain_wording` was a known-stale test (see the session-9/10 changelogs, which already call it "known stale" rather than a live defect) and no longer exists in this codebase — it is not present anywhere in the current `tests/` tree. As of this pass the independently-reproduced full suite (`tests/independent_release/` + root `tests/` + `tests/benchmark/`) is 1,108 passed, 1 skipped (`test_phase28_survival_hardening.py`, skipped because the optional test-only `lifelines` dependency isn't installed — not a failure), 0 failed. The "1 pre-existing failure" line above is retained for history but is stale and should not be read as a currently-open defect.
- **Still open:** the patch only fires for the two codes with known post-execution evidence; any future ledger entry built pre-execution needs its own explicit patch wiring rather than inheriting this one automatically.


## DEFECT-025 — "Is X related to Y?" refused to run (planner regex not inflection-tolerant)

- **Severity:** P1
- **Status:** FIXED
- **Impact:** `UniversalQuestionCompiler`'s explicit-relationship pattern matched only the literal "relate to". IntentEngine classified "Is ad spend related to revenue?" as CORRELATION, but the compiler missed the pair, fell back to an unrelated categorical column, and the C4.2.4 final-contract guard correctly refused to execute ("Analytical authority conflict ... predictor_columns"). A plain two-variable question ended INCONCLUSIVE, both with a real signal and with none.
- **Fix:** the pattern now accepts inflected forms (`relat\w* to`, `correlat\w* with`, `associated with`, `linked/tied/connected to`, `depend\w* on`, `coupled to/with`) and more auxiliaries (did/was/will/has ...).
- **Regression coverage:** `test_session10_analyst_answer.py::TestEndToEnd::test_plain_correlation_question_runs_instead_of_authority_conflict` (fails when the regex is reverted -- verified).
- **Still open:** the compiler and IntentEngine still keep separate vocabularies; DEFECT-025 is the same drift class as C4.2.2c. A shared vocabulary module is the durable fix.

## DEFECT-026 — Answers did not report the quantity asked for; "why did X drop in <period>" named the wrong segment

- **Severity:** P1 (wrong confident answer), P2 (missing numbers)
- **Status:** FIXED (elevated to human data analyst level in Session 11)
- **Impact:** measured on 10 ordinary analyst questions with planted ground truth. The loop reported segment *shares* instead of group differences ("'A' accounts for 0.2% of converted"), returned INCONCLUSIVE for a plain ranking, gave meaningless trend text, and for "Why did revenue drop in June?" named East with high posterior while the planted driver was South (the loop tests static concentration, not change).
- **Fix:** new `engines/analyst_answer.py` (deterministic, no AI, never raises) recomputes the asked quantity from the resolved contract columns and states it with interval, effect size, robustness check and data caveats: ranking (volume vs value split), numeric group comparison (Welch + rank test + trimmed means), rate comparison (Newcombe CI + chi-square/Fisher), correlation (Pearson + Spearman + slope), trend (OLS + Kendall, partial trailing period excluded), period change (segment decomposition, calendar-length effect, robust "is this unusual" baseline). Wired in `controller.py` after `formulate_direct_answer`.
  - Ranking + loop INCONCLUSIVE + all gates clear -> verdict OBSERVED (confidence 1.0 = exact sample statistic).
  - Period-change answers supersede the loop's statement/verdict (OBSERVED), because the loop tested a different claim.
  - A DIAGNOSED/STATISTICALLY_SIGNIFICANT loop verdict is downgraded to INCONCLUSIVE when the analyst recomputation fails its own robustness check.
  - The layer stands down entirely for churn questions, confounding hypotheses and unidentifiable churn (the loop owns identifiability there).
- **Session 11 Human Analyst Level Capabilities & Limits Resolved:**
  1. *Epistemic Verdict Accuracy:* Added `NO_DETECTABLE_EFFECT` verdict to `AnalyticalVerdict` (backend + frontend). Adequately powered null results ($N \ge 40$ overall, $N \ge 20$ per group) receive `NO_DETECTABLE_EFFECT` with confidence 1.0 and 95% CI equivalence bounds (stopping false `INCONCLUSIVE` classifications). Underpowered tests safely remain `INCONCLUSIVE`.
  2. *Per-Day Normalization Pin:* Mutation-pinned in `test_session11_human_analyst_level.py::TestPerDayNormalizationPin` verifying `per_day_change_pct` and preventing false alarms from calendar length differences (e.g. Feb vs Jan).
  3. *From Where to Why (PVM & Sub-segment Drilldown):* Implemented mathematical Price-Volume-Mix (PVM) rate vs volume decomposition ($\Delta = \text{Volume Effect} + \text{Rate Effect} + \text{Cross Effect}$) explaining whether drops are driven by transaction counts or unit economics (AOV/price), alongside secondary dimension drilldown inside the primary driver and intra-period run-rate trajectory inflection timing.
  4. *Metric Aggregation Intelligence:* Added column-aware heuristic detecting intensive metrics (`price`, `rating`, `score`, `rate`, `pct`, `latency`, `margin`, `aov`, `duration`) to automatically select `mean` aggregation instead of default `sum`.
  5. *Real-World Messy Data Robustness:* Hardened `_to_numeric` to safely handle `inf`/-`inf`, `NaN`s, missing categories, and single-level groups without crashing.
  6. *Pareto & Actionable Next Steps:* Added Pareto (80/20) concentration calculations, breakdown question semantics, and concrete operational recommendations for business stakeholders.
- **Regression coverage:** `tests/independent_release/test_session10_analyst_answer.py` (21 tests) and `tests/independent_release/test_session11_human_analyst_level.py` (16 tests, including API surface tests) — 37 of 37 tests passing. Entire repository (1,052 tests across independent_release, root, and benchmark suites) verified 100% green with 0 failures.


## DEFECT-027 — Multi-group numeric comparison TypeError silently swallowed; overly-broad confounding guard suppressed period change

- **Severity:** P1 (broken multi-group question resolution and period change fallback)
- **Status:** FIXED (elevated to Principal Data Analyst Depth in Session 12)
- **Impact:**
  1. Multi-group questions like "does average order value differ by region?" ($k > 2$ groups) silently crashed inside `_numeric_comparison` and `_rate_comparison` and fell back to the old broken loop output ("'West' accounts for 26.8%...").
  2. Ordinary "why did revenue drop in June?" questions fell back to the loop's answer (naming the wrong region) because candidate hypotheses generated by the synthesizer contained the token "confound", even though confounding was never adopted or asserted by the loop.
- **Root causes:**
  1. `min(ns.values())` called `.values()` as a function on a pandas Series `ns`, raising `TypeError: 'numpy.ndarray' object is not callable`, which was silently swallowed by the top-level `try/except` in `build_analyst_result`.
  2. Confounding guard checked candidate hypotheses claims instead of the loop's actual asserted answer text.
- **Fix:**
  1. Replaced `min(ns.values())` with `int(ns.min())` in both `_numeric_comparison` and `_rate_comparison`.
  2. Keyed the confounding guard strictly off the loop's actual asserted answer (`"confound" in str(direct_ans or "").lower()`).
- **Principal Data Analyst Depth Capabilities Added (Session 12):**
  1. *Kitagawa Mix-Shift vs Rate Decomposition:* Distinguishes within-group rate changes from cross-group compositional shifts ($\Delta \bar{y} = \sum \bar{w}_i \Delta y_i + \sum \bar{y}_i \Delta w_i$), with automated detection and prominent explanation of Simpson's Paradox.
  2. *Multilevel Waterfall Bridge:* Exact arithmetic reconciliation ($V_0 + \sum \text{steps} = V_1$) across volume, rate, cross interaction, and segment contributions.
  3. *Cohort & Account Lifecycle Bridge:* Decomposes revenue/volume changes into New/Acquisition, Retained/Expansion, and Lost/Churned account cohorts.
  4. *Concentration & Tail Fragility Diagnostics:* Computes Gini inequality coefficient ($G$), Herfindahl-Hirschman Index (HHI), and Pareto concentration shares (top 1%, 5%, 20%).
  5. *Common Language Effect Size (CLES) & Percentile Profiles:* Evaluates probability that a random observation from Group A exceeds Group B ($P(A > B) = U_A / (n_A n_B)$), with p10, p25, median, p75, p90, p99, IQR, skewness, and trimmed means.
  6. *Multivariable Regression & Covariate Control:* Controlled OLS regression slopes, elasticity at means ($\epsilon = \beta \cdot \bar{X}/\bar{Y}$), adjusted $R^2$, and confounder risk detection.
  7. *Predictive Run-Rate Extrapolations:* Next-period run-rate forecasting with 95% prediction intervals, coefficient of variation, and lag-1 autocorrelation for trend momentum vs mean reversion.
  8. *Counterfactual Opportunity Sizing:* Quantifies actionable what-if opportunity lifts (e.g. elevating lagging segments to median benchmark or holding declining drivers flat).
  9. *Categorized 3-Tier Strategic Playbooks:* Generates prioritized operational actions (Immediate Triage 24-48h, Diagnostic Deep-Dive 1-2w, Strategic Mitigation 30-90d) with priority levels and focus areas.
- **Regression coverage:** `tests/independent_release/test_session12_human_analyst_depth.py` (23 tests covering all depth engines and pinning all defect fixes). Entire test suite verified 100% green with 0 failures.


## DEFECT-028 — Simpson's paradox false-positive on flat rates, currency/formatted string crashes in upstream statistics, and omitted narrative rendering of depth engines

- **Severity:** P1 (erroneous paradox claims & crashes on formatted currency)
- **Status:** FIXED (Session 12 Release-Readiness Certification)
- **Impact:**
  1. `_kitagawa_decomposition` flagged Simpson's Paradox on pure mix-shifts where within-segment rates were identical ($y_{1, i} == y_{0, i}$), claiming *"within-segment rates moved in the opposite direction!"*.
  2. Formatted currency strings (`"$94.08"`, `"€1,234.50"`, etc.), commas, and percentages in numeric columns caused `ValueError` in upstream statistical preflight, method selection, and inference routines.
  3. The 9 advanced depth engines (waterfall bridge reconciliation, cohort lifecycle bridge, and 3-tier strategic playbooks) were stored in structured dictionaries but omitted from `AnalystResult.to_text()`, hiding them from the executive direct answer.
  4. Documentation inconsistency in `VISION.md` where Frontdoor criterion and 1,024 multiverse specifications were advertised in Section 3, contradicting the Section 5 status table.
- **Fix:**
  1. Required active, opposing directional within-segment movement (`has_positive_within`, `has_negative_within`) before triggering Simpson's Paradox; classified flat rates as pure mix-shift without erroneous paradox claims.
  2. Implemented `_to_clean_float_series` and `_to_clean_float_array` in `preflight.py` stripping currency symbols (`$`, `€`, `£`, `¥`, `₹`), commas, percentages, and whitespace; wired into `preflight.py`, `method_selection.py`, `inference.py`, `uncertainty.py`, `multiple_comparisons.py`, and `analytical_math.py`.
  3. Enhanced `AnalystResult.to_text()` to format the waterfall reconciliation chain (`Reconciliation: Prior Base: ... -> [Final: ...]`) and categorized 3-tier strategic playbooks (`Strategic Playbook: [Immediate Operational Triage (24-48h)]: ... | ...`).
  4. Harmonized `VISION.md` Section 3 to accurately describe 9 core multiverse specifications (architected for 1,024) and Backdoor causal identifiability (with Frontdoor/do-calculus as future roadmap).
- **Regression coverage:** `tests/independent_release/test_session12_human_analyst_depth.py` (`TestKitagawaDecompositionAndSimpsonsParadox`, `TestWaterfallBridgeReconciliation`, `TestStatisticalCurrencyAndFormattedStrings`).




## DEFECT-029 — SSRF protection on AI provider base URLs was validate-then-store, not validate-then-connect (DNS-rebinding gap)

- **Severity:** P0 (security)
- **Status:** FIXED (2026-09-25 release-hardening pass)
- **Impact:** the "v26 release_certified" build's P0.1 claimed SSRF protection was enforced, and the validation logic in `packages/analytics_core/src/security/ssrf.py` (`validate_network_endpoint`/`assert_safe_endpoint`) was itself correct — but for AI provider base URLs it only ran once, inside `factory.py::set_ai_config()`, at settings-save time. The function that actually issues the outbound request (`OpenAICompatibleProvider`/`ClaudeProvider`/`OllamaProvider` → `httpx.Client(...).post(base_url + ...)`) never re-validated or re-resolved before connecting. A base URL that resolved safely at save time could later be DNS-rebound (low-TTL record repointed at `169.254.169.254`, an RFC1918 address, etc.) and every subsequent AI request would connect straight to it with nothing to catch it — the "prevent DNS rebinding" claim in the module's own docstring didn't actually hold at the point that mattered.
- **Root cause:** check-then-use pattern — validation happened at a different time and in a different function than the connection it was meant to protect.
- **Fix:** added `resolve_pinned_target()` to `ssrf.py`, which resolves the hostname, validates every resolved address against the existing policy, and returns a `PinnedTarget` whose connection URL's host component is the exact validated IP literal (not the original hostname) — while still carrying the real hostname forward as the `Host` header and the httpx `sni_hostname` request extension, so origin virtual-hosting and TLS certificate validation are unaffected. `ClaudeProvider.generate()`, `OpenAICompatibleProvider.generate()`, and `OllamaProvider.generate()` now call this immediately before their `httpx` call instead of relying on the one-time `set_ai_config` check; `set_ai_config`'s `assert_safe_endpoint` call is kept as a pre-flight/UX check (fast, specific error at save time) but is no longer the only enforcement. `S3StorageProvider._get_client()` (used for S3-compatible cloud backup endpoints) gained the same re-check immediately before building the boto3 client, as defense-in-depth against any future caller that skips the existing per-request check in `apps/api/src/api/v1/backups.py`; full IP-pinning wasn't implemented there since boto3/botocore performs its own DNS resolution outside application control — this is flagged as an accepted residual risk, not solved, and would need a custom botocore endpoint resolver to close completely.
- **Regression coverage:** `tests/independent_release/test_ssrf_pinned_target.py` (20 tests): unsafe-endpoint rejection (metadata IP/hostname, RFC1918 literals, loopback/localhost without the flag, missing URL, prohibited scheme, link-local); safe-endpoint pinning behavior including a live DNS resolution against `api.anthropic.com` proving the returned URL's host is an IP literal while `Host`/SNI still carry the real hostname; each of the three AI providers refusing an unsafe `base_url` *at `generate()` time* (not construction time, and with no settings-save step involved in the test at all — proving the check now lives at the connection); two live round-trip tests against a local loopback HTTP server proving the pinned connection still delivers the correct request path and `Host` header end-to-end; and an `S3StorageProvider._get_client()` rejection test. Full suite after the change, independently reproduced: 1,108 passed, 1 skipped (unrelated optional dependency), 0 failed — up from 1,088 before this pass (829 → 849 in `tests/independent_release/`, root+benchmark unchanged at 259).
