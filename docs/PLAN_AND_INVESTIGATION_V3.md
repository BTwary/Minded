# AA-OS Plan & Investigation Layer v3

## Objective

This revision tightens the boundary between natural-language interpretation, investigation planning, dependency resolution, and runtime execution authority. It adds auditable structure while preserving the rule that semantic proposals never override the canonical question contract.

## 1. Natural-language intent cues

The parser records candidate task family, metric phrases, grouping and entity phrases, time/filter/comparison cues, aggregation, ranking direction/limit, output preferences, and references to previous findings. `task_confidence` is explicitly labelled `UNCALIBRATED_HEURISTIC_SIGNAL`; it is not a calibrated probability and must not be used alone to authorize an analytical method.

Sentence boundaries can now create separate objectives when the next sentence begins with a recognized analytical task lead. Bivariate and multi-predictor questions continue to be protected against false decomposition.

## 2. Compound objective DAG

Each objective has a stable ordinal and structured fields for task hint, intent signals, expected output, success criteria, priority, connector metadata, dependency kind, reference strength, and clarification requirements.

Dependency resolution is no longer synonymous with "the immediately preceding clause". A typed entity reference such as "those customers" prefers the closest earlier objective that mentions a compatible entity family. Explicit references to the first/original/second result can bind to that objective. Unresolved pronouns are represented as clarification gates; they are not silently grounded.

Connector metadata (e.g. `THEN`, `AND`, `ADDITIVE`) is descriptive only. Sequence wording alone does not manufacture a semantic dependency. Plan validation checks that prerequisite ordinals exist and point backward.

## 3. Typed entity grounding

Entity grounding is isolated in `intelligence/followup_grounding.py`. It only extracts a short, quoted label from a small number of recognized finding templates. Unquoted spans and instruction-like labels are rejected. Substitution happens only for typed entity anaphora, such as "those customers"; vague phrases such as "for those" are not rewritten.

This deliberately favors clarification over guessing. The pattern recognizer is not a general-purpose named-entity recognizer.

## 4. Experiment and stage plan validation

`intelligence/investigation_plan_contract.py` validates:

- unique experiment and stage identifiers;
- existence of prerequisite identifiers;
- self-dependencies and cyclic dependencies;
- topological ordering for experiment and stage graphs;
- presence of evidence requirements;
- implementation-module evidence for each proposed method.

The method catalog distinguishes `MODULE_PRESENT`, `ORCHESTRATION_REQUIRED`, and `PLAN_ONLY`. A module path is evidence that related implementation code exists, not proof that a callable executor is wired. **The audit helper never grants execution authority.** The canonical runtime method registry and contract checks remain mandatory.

Every serialized plan includes `plan_validation`, with graph edges, topological order, issues/warnings, capability review, and the explicit values `execution_authorized: false` and `runtime_authorization_required: true`.

## 5. Staged investigation lifecycle

The plan describes scope resolution, data/method readiness, task-specific gates (causal identification, prediction leakage, forecasting design, or decision constraints), experiments, independent verification, and claim/stopping gates. When real unresolved questions remain, downstream stages are marked blocked upstream rather than appearing executable.

The internal requirement to preserve clauses in a compound question is recorded as a limitation/routing requirement, not misreported as a question that the user must clarify.

## 6. Runtime hardening

The controller rejects a compound plan that fails structural validation before creating child investigations. It blocks explicitly ambiguous references, checks the declared prerequisite records, grounds entity references only with the typed grounding helper, and records failures in the durable objective status/event stream.

## 7. Non-goals and limitations

- The natural-language layer remains deterministic and bounded; it is not general-purpose coreference or unrestricted language understanding.
- Module existence is not proof of executor registration. The catalog is review metadata, not an alternative method registry.
- Statistical validity still depends on the data, assumptions, canonical semantic contract, and executed method.
- Full API/database integration tests require the repository's complete dependency environment, including DuckDB. This environment did not have DuckDB, so selected integration tests could not be collected here.
