# VISION: AA-OS — Autonomous Analytical Intelligence Operating System

> *"The future of analytical intelligence is not generating text-to-SQL or charting queries. It is autonomous scientific investigation."*

---

## 1. The Core Analytical Dilemma

The prevailing paradigm of "AI Data Analysts" suffers from a fundamental architectural flaw: **single-shot answer generation**.

When a human business leader asks:
> *"Why did enterprise customer retention drop 18% in Q3?"*

Traditional LLM assistants perform a single-step translation:
1. Parse natural language into SQL.
2. Execute the query against a database.
3. Summarize the returned rows in prose.

### Why This Fails in the Real World:
- **No Competing Hypotheses**: Real data phenomena are multi-causal. A single query cannot distinguish between a volume drop, price sensitivity, mix shift, or product outage.
- **Vulnerability to Confounding & Simpson's Paradox**: An aggregate trend can completely reverse when conditioned on an unobserved confounder (e.g., regional cohort shifts).
- **Zero Hallucination Control**: When LLMs synthesize conclusions from ungrounded statistical summaries, they invent causal narratives without empirical proof.
- **Absence of Epistemic Uncertainty**: Traditional tools return a static answer with artificial confidence, unable to quantify remaining Shannon entropy or statistical power.

---

## 2. The AA-OS Paradigm: Recursive Scientific Investigation

**AA-OS (Autonomous Analytical Intelligence Operating System)** shifts analytical systems from **passive query compilers** to **active scientific investigators**.

AA-OS treats data analysis as an iterative scientific loop:

```
                            OBSERVATION
                                 │
                                 ▼
                          EVIDENCE PATTERNS
             (Concentration, Segment Diff, Temporal, Anomaly)
                                 │
                                 ▼
                     PATTERN-SPECIFIC HYPOTHESES
                                 │
                                 ▼
                            PREDICTIONS
                    (Deductive Falsification Rules)
                                 │
                                 ▼
                   PREDICTION-LINKED EXPERIMENTS
                                 │
                                 ▼
                       UNIFIED CANDIDATE POOL
                [Composite Multi-Objective Utility]
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
             EIG            Adversarial         Multiverse
          (Shannon)       (Falsification)      (Robustness)
              └──────────────────┬──────────────────┘
                                 ▼
                       DETERMINISTIC OLAP
                    (In-Process DuckDB Engine)
                                 │
                                 ▼
                       DUAL-ENGINE VERIFICATION
                     (DuckDB SQL ↔ Polars Recompute)
                                 │
                                 ▼
                    BAYESIAN BELIEF CONVERGENCE
                     [P(H_i | E) Posterior Update]
                                 │
                                 ▼
                    RECURSIVE UNCERTAINTY GATE
                   Is Shannon Entropy Delta <= 0.05
                   or Max Posterior >= 0.85?
                      ├── YES ──► Epistemic Verdict & Decision
                      └── NO  ──► Replan Next Investigation Turn
```

---

## 3. Core Architectural Pillars

### I. Strict Decoupling of Reasoning, Execution, and Dual Operating Modes
- **Mode 1 (Free / Zero AI API)**: The entire core analytical pipeline (profiling, schema ontology, deterministic hypothesis synthesis, EIG experiment ranking, DuckDB SQL execution, ANOVA variance decomposition, Simpson's paradox detection, Bayesian updates, and provenance manifest generation) executes 100% locally and deterministically with zero external API dependencies.
- **Mode 2 (Optional AI Augmentation)**: External LLMs (Gemini, Claude, OpenAI, Ollama, Groq) optionally assist with natural-language parsing, semantic metadata expansion, and narrative report synthesis.
- **Strict Epistemic Invariant**: AI models are NEVER authoritative for numerical values, $p$-values, effect sizes, Bayesian posteriors, dataset facts, or verification proofs. All empirical facts are computed deterministically by DuckDB and SciPy. If AI is unavailable or fails, AA-OS logs a warning and completes the investigation deterministically.

### II. Expected Information Gain (EIG) Optimization
Instead of brute-forcing all possible table slices, the **Experiment Planner** evaluates candidate experiments through mutual information / Shannon entropy reduction:
$$\mathbb{E}[IG(e)] = H(H) - \sum_{y \in \mathcal{Y}} P(y|e) H(H | y, e)$$
Experiments that maximally separate competing hypotheses are prioritized over redundant queries.

### III. In-Loop Adversarial Falsification
AA-OS actively attempts to **break its own leading hypotheses**:
- Simulates confounding variables to test for **Simpson's Paradox**.
- Injects covariate conditioning to evaluate whether an observed effect vanishes under subgroup stratification.
- Synthesizes counter-experiments specifically designed to falsify the current frontrunner before presenting conclusions to human decision-makers.

### IV. Multiverse Specification Curve Robustness
To eliminate researcher degrees of freedom (outlier filtering, date truncation, metric definitions), AA-OS runs **multiverse specification curve analysis** across candidate pipelines (currently 9 core specifications, architected for expansion up to 1,024):
$$\text{Robustness Score} = \frac{\sum_{m \in \mathcal{M}} \mathbb{I}(\text{Effect Direction is Invariant})}{\lvert \mathcal{M} \rvert}$$

### V. Pearl's Causal Identifiability (Backdoor Criterion)
Distinguishes between observational correlation $P(Y \mid X)$ and true interventional effect $P(Y \mid \text{do}(X))$:
- Analyzes assumed Directed Acyclic Graphs (DAGs).
- Validates the Backdoor adjustment criterion (with future expansion planned for Frontdoor and full do-calculus).
- Automatically demotes unidentifiable claims to purely observational scope when unmeasured confounders exist.

---

## 4. The End Goal

The ultimate vision of AA-OS is to serve as an **autonomous analytical co-pilot for scientific, enterprise, and institutional intelligence**:
A system that can ingest petabyte-scale data lakes, formulate thousands of competing causal theories, design and run optimal experimental queries, falsify invalid hypotheses, and deliver calibrated, provable answers with complete provenance.


---

## 5. Implementation Status vs. This Vision (audited 2026-09-24)

This document states the target. See `docs/AAOS_VISION_AUDIT_2026-09-24.md` for the evidence.

| Pillar | Status |
|---|---|
| I. Zero-AI Mode 1, AI never authoritative | Implemented. AI-enabled state now has a single resolver; an explicit `AI_ENABLED=false` always wins |
| II. EIG experiment selection | Implemented |
| III. In-loop adversarial falsification | Partial. Simpson's reversal detector rewritten; `is_falsified` is never set True by the active attacker, so no posterior penalty is applied from attacks |
| IV. Multiverse robustness | Partial. 9 specifications, not up to 1,024 |
| V. Causal identifiability | Partial. Backdoor only; no frontdoor, no do-calculus |
| Recursive uncertainty gate | Implemented (stagnation now judged on recent rounds, not cumulative change) |
| "Dual-engine verification" | DuckDB-vs-Polars arithmetic agreement; SciPy inference is a separate layer, not a second engine |

---

## 6. The Human Verifier Contract

AA-OS is a worker, not an oracle. The human data analyst is the **verifier**, and the product is built so that verifying is
fast and cannot be done carelessly:

- AA-OS does the analysis and checks what a machine *can* check (arithmetic agreement, data fitness, adversarial challenge,
  robustness, claim strength). Those results are offered as optional spot checks, with the SQL to reproduce them.
- The verifier's attention is spent on what a machine *cannot* check: whether the question was understood, whether a metric was
  aggregated meaningfully, whether a stratifying variable is a confounder or a mediator, whether high-risk assumptions hold.
- Anything not tested is shown as **not tested**, never as passed.
- Sign-off is explicit, needs a comment for blockers and objections, is bound to the exact result reviewed, and is an
  append-only audit trail. A verifier can always reject; `VERIFIED` cannot be reached by skipping items.

Status: backend, API and UI implemented in session 8 (see `BUGFIX_2026-09-24_session8_human_verifier_workflow.md`). The panel
has been type-checked and built but not exercised in a browser. Reports and exports do not yet carry the sign-off state.

