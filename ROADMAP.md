# ROADMAP: AA-OS Development Milestones

This roadmap documents the architectural progression of **AA-OS (Autonomous Analytical Intelligence Operating System)** from foundational relational compiler to an autonomous scientific intelligence mesh.

---

## 🗺️ Milestone Matrix

| Phase | Milestone Name | Focus & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Semantic Resolution & Relational IR** | Typed Analytical Intent AST, Grain resolution, Schema mapping, DuckDB SQL compiler | **COMPLETED &check;** |
| **Phase 2** | **Durable Execution & State Machine** | Asynchronous job queues, checkpointer, atomic lease worker, SSE event streaming | **COMPLETED &check;** |
| **Phase 3** | **Decomposed Autonomous Loop** | Hypothesis synthesizer, prediction deduction, EIG experiment planner, Bayesian updater | **COMPLETED &check;** |
| **Phase 4** | **Adversarial & Causal Gates** | Simpson's paradox detector, Multiverse specification curve, Pearl Causal DAG / Backdoor criterion | **COMPLETED &check;** |
| **Phase 5** | **Internally Coherent Scientific Kernel** | Prediction linkage, pattern-specific emergence, fail-closed state, single authority, dual operating modes | **COMPLETED &check;** |
| **Phase 6** | **Multi-Domain Generalization & Autonomy** | Schema-agnostic semantic discovery, unseen schemas, adversarial Simpson's checks, emergent experiments | **COMPLETED &check;** |
| **Phase 7** | **Distributed Worker Mesh & Federation** | Distributed worker orchestration, multi-dataset join synthesis, automated anomaly monitors | **IN PROGRESS ⏳** |
| **Phase 8** | **Interactive Human-in-the-Loop Studio** | Real-time DAG node manipulation, custom prior injection, hypothesis grafting | **PLANNED 📋** |
| **Phase 9** | **Enterprise Federation & Lakehouse Pushdown** | Native pushdown to Snowflake, BigQuery, Databricks, and ClickHouse | **PLANNED 📋** |
| **Phase 10** | **Real-World Adversarial Safety Validation** | 14-scenario hostile-data benchmark against the live autonomous controller; 2 unsafe-confidence findings fixed at the data-quality-gate level, 4 more documented as architectural gaps requiring new subsystems | **PARTIALLY COMPLETE ⚠️** |

---

## 🔍 Detailed Progress by Phase

### Phase 1: Semantic Resolution & Relational IR [COMPLETED &check;]
- [x] Defined canonical `TypedAnalyticalIntent` IR schema.
- [x] Semantic World Model for grain, entities, dimensions, and metric definitions.
- [x] Deterministic SQL AST compilation with automatic join path discovery.
- [x] Zero-hallucination validation gates before execution.

### Phase 2: Durable Asynchronous Execution [COMPLETED &check;]
- [x] State machine with 11 discrete lifecycle states (`PLANNED`, `QUEUED`, `CLAIMED`, `RUNNING`, `WAITING_FOR_USER`, etc.).
- [x] In-process and database-backed distributed queue providers with atomic lease claiming.
- [x] Heartbeat renewal loop and automatic reclamation of stale/abandoned worker leases.
- [x] Checkpointer persisting all intermediate state snapshots.

### Phase 3: Recursive Autonomous Loop [COMPLETED &check;]
- [x] Pattern detector discovering concentration, segment variance, temporal shifts, and anomalies.
- [x] Bayesian updating engine calculating normalized posterior probabilities across $N$ competing hypotheses.
- [x] Shannon entropy computation & convergence stopping triggers.
- [x] SciPy dual-engine verification with ANOVA $F$-statistics and $\eta^2$ effect sizes.

### Phase 4: Adversarial Analysis & Causal Gates [COMPLETED &check;]
- [x] Automated adversarial challenge testing for Simpson's paradox and latent confounders.
- [x] Dynamic counter-experiment synthesis to challenge leading hypotheses.
- [x] Multiverse specification curve analysis (9 specifications). [ ] Expand toward the 1,024-pipeline grid (date truncation, min-group-size, winsorization, missing-data handling).
- [x] Pearl causal identifiability validation (Backdoor criterion). [ ] Frontdoor criterion and do-calculus rules are not yet implemented.

### Phase 5: Internally Coherent Scientific Kernel & Dual-Operating Modes [COMPLETED &check;]
- [x] Explicit prediction $\rightarrow$ experiment linkage via `prediction_ids` and deductive status matching.
- [x] Pattern-specific emergent hypothesis formulation without collapsing distinct evidence types.
- [x] Fail-closed state consistency validator (`validate_state()`) terminating on corruption.
- [x] Single authoritative state ownership in `InvestigationStateManager`.
- [x] Unified candidate pool scoring integrating $EIG$, adversarial falsification, robustness, and causal value.
- [x] **Dual Operating Modes**: Mode 1 (100% Free / Zero-API Deterministic local runtime) & Mode 2 (Optional Bring-Your-Own-Key AI Augmentation).
- [x] Universal AI Provider Factory (Gemini, Claude, OpenAI, Ollama, Groq, Custom) with graceful fail-safe fallbacks.

### Phase 8: Business Context, Prescriptive Action & State Forking [COMPLETED &check;]
- [x] dbt semantic manifest ingestion (`manifest.json`) and metric filter injection.
- [x] Pre-flight A/B statistical power & sample size calculator (`ExperimentDesignAgent`).
- [x] Prescriptive linear programming budget optimization (`PrescriptiveOptimizer` with HiGHS).
- [x] T-Learner CATE uplift modeling and 4-quadrant cohort segmentation (`UpliftEngine`).
- [x] Continuous background drift daemon with K-S, PSI, and CUSUM tests (`DriftMonitorEngine`).
- [x] Investigation state DAG forking ("Git for Hypotheses") with parent/child lineage.
- [x] Enterprise Lakehouse SQL transpilation and pushdown compiler (`LakehousePushdownCompiler`).

### Phase 9: Automated Causal DAG Discovery & Inter-Agent Protocol [COMPLETED &check;]
- [x] Order-independent Peter-Clark (PC) algorithm for continuous observational causal discovery (`PCAlgorithmEngine`).
- [x] Fisher's Z-transform on Frisch-Waugh-Lovell OLS regression residuals for partial correlation conditional independence.
- [x] Unshielded v-structure / collider orientation and Meek's orientation rules 1–4.
- [x] Pearl Backdoor Criterion adjustment set solver (`CausalIdentifiabilityGate.discover_and_evaluate`).
- [x] MindEd Inter-Agent Protocol (MIAP) with machine-readable ````aa-os-fingerprint-v1` generation (`UniversalFingerprintEngine`).
- **Caveat added in Phase 10:** the above algorithms are real and pass their
  own dedicated test suite (`test_phase9_causal_discovery.py`), but are not
  currently exercised by `InvestigationController.execute_investigation()`
  for an ordinary question -- `causal_intent` stays `None` unless the
  question uses explicit cause/effect phrasing, and even then no causal DAG
  is ever actually constructed and passed to the identifiability gate. The
  practical effect today is safe (the controller never overclaims causally)
  but this is safety by omission, not by a working, wired discrimination
  mechanism. See `PHASE10_REAL_WORLD_SAFETY_REPORT.md` Scenario 8 and
  `AUDIT_PHASE10_2026-08-26.md` Section 2.3.

### Phase 10: Real-World Adversarial Safety Validation [PARTIALLY COMPLETE ⚠️]
A 14-scenario adversarial benchmark (`scripts/test_phase10_adversarial_real_world.py`)
was run against the real, unmodified autonomous controller -- not against
individual engines called manually. Full results, safety metrics, and
per-scenario detail: `PHASE10_REAL_WORLD_SAFETY_REPORT.md`. Pre-work audit
and list of production defects found/fixed: `AUDIT_PHASE10_2026-08-26.md`.

- [x] Duplicate primary/business key detection (was previously undetected for
      non-identical duplicate rows) -- now blocks with quantified impact.
- [x] Silent unit/scale-change detection (e.g. dollars vs. cents between
      time periods) -- now blocks with an explicit discontinuity warning.
- [x] Fixed a silent field-name bug in `checkpointer.py` that discarded every
      failure's structured classification (`error_code` was always `NULL`).
- [x] Fixed an uncaught `ValueError` in IR validation that left investigations
      stuck in `RUNNING` forever instead of reaching a safe `FAILED` state.
- [x] Verified schema-agnostic semantic resolution under column renaming and
      under a 119-column, mostly-irrelevant, unfamiliar-names stress test.
- [x] Verified Simpson's-paradox/confounding detection spawns a real targeted
      conditional experiment and correctly declines a decisive verdict.
- [ ] **NOT YET SAFE:** MNAR/structured missingness can silently bias a
      confident (95%) conclusion with zero disclosure (Scenario 6).
- [ ] **NOT YET SAFE:** a post-outcome/leaked categorical field can be
      confidently (93.7%) treated as a legitimate explanatory driver
      (Scenario 7).
- [ ] **MISSING:** cross-table join execution does not exist in the
      autonomous controller path at all (Scenario 3) -- the world model
      computes join/fanout metadata that is never consulted.
- [ ] **MISSING:** timezone-aware business-day/period semantics do not exist
      anywhere in the codebase (Scenario 5).
- [ ] **MISSING:** selection/survivorship-bias detection (Scenario 11) and
      future-information/temporal-leakage detection (Scenario 12).
- [ ] **MISSING:** dynamic aggregation-type selection -- any binary or
      rate/percentage target metric currently fails immediately because
      aggregation is hardcoded to `SUM`.

This phase is not marked complete. Two `UNSAFE_FALSE_CONFIDENCE` findings
remain unresolved, and the single largest architectural weakness -- the
controller's paradigm is built entirely around single-table, additive
(SUM-based), single-categorical-dimension analysis -- is documented in
`PHASE10_REAL_WORLD_SAFETY_REPORT.md` Section 5 rather than claimed fixed.
Do not proceed to a "local-first packaging" phase until at least the two
unsafe-confidence findings are addressed and this suite is re-run clean.

---

## 🧪 Golden Verification Benchmarks

Every release is verified by 15 deterministic regression test suites (90 tests total):
- `scripts/test_zero_ai_mode.py` — **7/7 Zero-AI Mode Tests Passed**
- `scripts/test_ai_fallback.py` — **5/5 AI Fallback Tests Passed**
- `scripts/test_native_analytical_mathematics.py` — **10/10 Native Mathematics Tests Passed**
- `scripts/test_data_quality_and_semantic_gate.py` — **7/7 Data Quality Gate Tests Passed**
- `scripts/test_golden_deterministic_investigation.py` — **1/1 Golden Deterministic Investigation Passed**
- `scripts/test_unified_canonical_loop.py` — **1/1 Unified Canonical Loop Test Passed**
- `scripts/test_aaos_generalization.py` — **6/6 Component Generalization Tests Passed**
- `scripts/test_aaos_generalization_e2e.py` — **8/8 Canonical Controller E2E Generalization Tests Passed**
- `scripts/test_phase7_enterprise_guardrails.py` — **4/4 Phase 7 Enterprise Guardrails Tests Passed**
- `scripts/test_phase8_business_context.py` — **4/4 Phase 8 Business Context Tests Passed**
- `scripts/test_phase8_proactive_forking.py` — **8/8 Phase 8 Proactive Drift & Forking Tests Passed**
- `scripts/test_phase9_causal_discovery.py` — **5/5 Phase 9 Causal Discovery & MIAP Tests Passed**
- `scripts/test_unified_scientific_loop.py` — **7/7 Golden Coherence Tests Passed**
- `scripts/test_phase2_autonomous_kernel.py` — **8/8 Critical Acceptance Tests Passed**
- `scripts/test_golden_adaptive_investigation.py` — **9/9 Adaptive Loop Tests Passed**
- `scripts/test_phase10_adversarial_real_world.py` — **14/14 scenarios executed and classified** (4 CORRECT, 1 CORRECT_WITH_WARNING, 1 CORRECTLY_INCONCLUSIVE, 2 DATA_QUALITY_BLOCKED, 2 UNSAFE_FALSE_CONFIDENCE, 4 ARCHITECTURAL_GAP — full detail in `PHASE10_REAL_WORLD_SAFETY_REPORT.md`; **not all-pass, by design and intent**)
