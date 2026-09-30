# Metric Semantics — Canonical Contract

**Version: 1.1**
**Status: FROZEN FOR PHASE 12**

This document is the authoritative interface specification for metric
semantics in AA-OS, produced at the close of the Phase 11 finalization pass
(see `AUDIT_PHASE11_METRIC_SEMANTICS.md` for the verification work behind
it). It documents what actually exists in the codebase as of this freeze —
not an aspirational schema. Future phases must consume this contract rather
than silently redefining metric semantics; a fundamental deficiency must be
addressed with an explicit, versioned change to this document, not a quiet
behavior change in the resolver.

---

## 1. Meaning

A `MetricDefinition` (`packages/analytics_core/src/semantic/metric_semantics.py`)
represents everything the autonomous loop needs to compute, verify, and
narrate a metric correctly — not merely the name of a numeric column. A
metric is treated as: **meaning + aggregation + grain + population +
numerator/denominator (where applicable) + weight (where applicable) +
validity + provenance**, carried from the moment a column is resolved as the
analytical target all the way through to the final verdict.

## 2. Lifecycle

1. `SemanticEngine.resolve_schema` (`engines/semantic.py:196`) calls
   `MetricSemanticsResolver.resolve(metric_col, df, table_name,
   group_dimension_col, grain, question_tokens)` for the resolved target
   metric column, unconditionally, on every investigation.
2. The returned `MetricDefinition` is attached to `SemanticResolution.metric_definition`
   and flows into `TypedAnalyticalIntent.target_metric.aggregation`
   (`runtime/controller.py:256-270`), `ExperimentSynthesizer` (which uses it
   to build the real query SQL), `verification.py` (dual-engine check), and
   the evidence/provenance/narrative layers.
3. A `MetricDefinition` is immutable once produced for a given investigation
   run — it is not mutated in place; a re-resolution produces a new object.

## 3. Sources of evidence the resolver considers

Purely deterministic, schema-agnostic, no AI/network (`metric_semantics.py:1-38`):

- **Column name tokens** — underscore/hyphen-split, checked against
  fixed token sets: `_RATE_NAME_TOKENS` (rate/pct/ratio/margin/utilization/...),
  `_MEAN_NAME_TOKENS` (avg/mean/aov/aur), `_ID_NAME_TOKENS`
  (id/key/pk/fk/code/zip/uuid), `_WEIGHT_NAME_TOKENS` (count/volume/quantity/
  units/impressions/clicks/visits/sessions/orders/transactions/population/
  eligible/n/weight/users/customers/requests), and `_ADDITIVE_NAME_TOKENS`
  (total/amount/revenue/sales/cost/spend/events/errors/... — a superset of
  `_WEIGHT_NAME_TOKENS`, added at the Phase 11 finalization pass, Section 6
  of the audit).
- **Observed value distribution** — min/max range of the non-null series;
  a bounded `[0, 1]` or `[0, 100]` range is a signal (not a guarantee) of a
  proportion/percentage.
- **Companion columns in the same table** — used to recover a true
  numerator/denominator pair for a precomputed rate column, or a plausible
  weight/volume column for a weighted mean, via name-token matching against
  sibling columns (`_find_numerator_denominator_pair`, `_find_denominator_pair`,
  `_find_weight_column`).
- **Question tokens** — an optional `question_tokens` set (e.g.
  `{"how", "many", "number", "count"}`) that can steer a numeric column
  toward `COUNT` instead of the row-default.

No hardcoded benchmark-specific answers: the resolver has no lookup table
mapping specific column names (e.g. `"revenue"`) to a hardcoded aggregation.
Every decision is derived from the general token/range/companion-column
rules above, which is why the resolver generalizes to unfamiliar schema
names (`gross_amt`, `case_volume`, `resolved_cases`, `active_base`, ...)
without modification.

## 4. Valid states — how ambiguity is represented

Every `MetricDefinition` carries `semantic_resolution_status`, one of:

| Status | Meaning | Set by |
|---|---|---|
| `"RESOLVED"` (default) | The resolver reached this aggregation via a positive signal — an explicit rate/mean/id token, a bounded-proportion range, a recognized additive-magnitude token, or a `question_tokens`-driven COUNT/COUNT_DISTINCT. | `MetricSemanticsResolver.resolve`, all branches except the ambiguous default |
| `"UNRESOLVED_DEFAULT_SUM"` | The column had **no** rate/mean/id/additive-magnitude naming signal and no bounded-proportion range. SUM is used so investigations over unfamiliar schemas don't stall, but this is an *unconfirmed default*, not a confirmed inference — narrative and provenance both say so explicitly. | `MetricSemanticsResolver.resolve`, rule 5, when `_ADDITIVE_NAME_TOKENS` finds no match |
| `"LEGACY_FALLBACK"` | No `MetricDefinition` was available at all (a hand-built `SemanticResolution` with the field unset — not reachable from any current production caller, since `engines/semantic.py:196` always populates it). A defensive SUM stand-in was substituted. | `controller.py:604-609`, `experiment_synthesizer.py:141-146` |

This is the mechanism that satisfies "no silent arbitrary aggregation": an
ambiguous or defensive SUM is always distinguishable, by a machine-readable
field (not just rationale prose), from a confidently-inferred one.

Acceptable alternative behaviors the resolver does **not** currently
implement, but that remain compatible with this contract for a future
version: hard-blocking with a distinct `INSUFFICIENT_SEMANTIC_EVIDENCE`
status, or an interactive clarification round-trip. `UNRESOLVED_DEFAULT_SUM`
was chosen instead — a schema-agnostic system running against a brand-new
dataset would otherwise stall on nearly every first-contact column.

### 4.1 "How many X" disambiguation (added in v1.1)

When a question's tokens include `how`/`many`/`number`/`count` and the
target column X is numeric but not identifier-like, rate-like, or
mean-like, the resolver distinguishes two cases by inspecting X's own
observed values (not its name):

- **`max(X) > 1.0`** — some row already records more than one occurrence
  (e.g. `error_events` with a row valued at `4`). A row/occurrence `COUNT`
  would silently undercount every such row. Resolves to `SUM`
  (`semantic_type = "event_volume_sum"`, `is_additive = True`,
  `semantic_resolution_status = "RESOLVED"`) — the total volume across
  rows, which is what "how many X occurred in total" actually means once a
  single row can hold more than one occurrence.
- **`max(X) <= 1.0`** — every observed value is consistent with at most one
  occurrence per row (a dense or sparse 0/1 flag). Resolves to `COUNT`
  (non-null row count, `semantic_resolution_status = "RESOLVED"`), as
  before.

This is a values-based check, not a name-based or vocabulary-based rule —
consistent with the rest of the resolver's schema-agnostic design (Section
3). See Section 14 for the one residual case this does not resolve.

## 5. Aggregations

`AggregationType` (`packages/schemas/src/analysis.py:31-51`):
`SUM`, `MEAN`, `COUNT`, `COUNT_DISTINCT`, `MEDIAN`, `MIN`, `MAX`, `RATE`,
`RATIO`, `PROPORTION`, `WEIGHTED_MEAN`.

`MetricSemanticsResolver.sql_aggregation_expression(metric_def, alias)`
(`metric_semantics.py:433-462`) is the single place that turns a
`MetricDefinition` into a SQL fragment:

| Aggregation | SQL |
|---|---|
| `SUM` | `SUM(col)` |
| `MEAN` | `AVG(col)` |
| `COUNT` | `COUNT(col)` (non-null count) |
| `COUNT_DISTINCT` | `COUNT(DISTINCT col)` |
| `RATE` / `RATIO` / `PROPORTION`, with numerator+denominator | `SUM(numerator) / NULLIF(SUM(denominator), 0)` |
| `RATE` / `RATIO` / `PROPORTION`, with weight only | `SUM(col * weight) / NULLIF(SUM(weight), 0)` |
| `RATE` / `RATIO` / `PROPORTION`, no numerator/denominator/weight | `AVG(col)` |
| `WEIGHTED_MEAN`, with weight | `SUM(col * weight) / NULLIF(SUM(weight), 0)` |
| `WEIGHTED_MEAN`, no weight | `AVG(col)` |

Row count (`COUNT(*)`) is distinct from non-null count (`COUNT(col)`) is
distinct from distinct-entity count (`COUNT(DISTINCT col)`) is distinct from
`SUM` of an already-per-row event-volume measure — see Section 4 above
(`"how many"` disambiguation, added in v1.1) and Section 14 below for the
one narrower residual gap that remains in this distinction.

## 6. Grain

`MetricDefinition.grain: str` (default `"row"`) records the level at which
the metric was resolved (e.g. `"row"`, or a grouping key passed in by the
caller). `requires_grouping: bool` flags metrics that only make sense
alongside an explicit `GROUP BY`. Grain is threaded into
`narrative_explanation()` (e.g. *"aggregated as SUM at the row grain"*) so
the analyst-facing explanation names the population the number describes,
not just the number.

## 7. Denominators

For `RATE`/`RATIO`/`PROPORTION` metrics, `numerator_column` and
`denominator_column` hold the raw columns used to recompose the ratio as
`SUM(numerator)/SUM(denominator)` — the mathematically correct way to
re-aggregate a ratio across rows or groups, as opposed to summing or
averaging a precomputed rate column directly (`SUM(conversion_rate)` is
**not** `SUM(conversions)/SUM(sessions)`). When no raw numerator/denominator
pair is observable in the table, the resolver degrades honestly to
`WEIGHTED_MEAN`/`MEAN` (Section 9) rather than fabricating a denominator.

## 8. Weights

`weight_column` holds a plausible volume/exposure column (matched via
`_WEIGHT_NAME_TOKENS`) used to weight a `WEIGHTED_MEAN` or a ratio's
fallback mean, so that larger groups contribute proportionally more than an
unweighted row average would give them. `is_compositional: bool` is `True`
whenever a metric is built from a numerator/denominator pair or a weight
column, signaling to any consumer that this metric's value is not a raw
column read but a computed composition.

## 9. Verification

`packages/analytics_core/src/engines/verification.py:140-220` runs a
**dual-engine** check: the primary execution path (DuckDB SQL, generated
from `sql_aggregation_expression`) is cross-checked against an independent
Polars computation that branches on the same `aggregation_type` string:

- `SUM` → `pl.col(target).sum()`
- `MEAN`/`AVG` → `pl.col(target).mean()`
- `COUNT` → `pl.col(target).count()`
- `COUNT_DISTINCT` → `pl.col(target).n_unique()`
- `RATE`/`RATIO`/`PROPORTION`/`WEIGHTED_MEAN` → `_ratio_agg`, which itself
  branches on whether a true numerator/denominator pair or a weight column
  is present, falling back to a plain mean only if neither exists.

Ratio-type metrics use a wider default numeric tolerance (`1e-3` vs the
baseline) since they involve an extra floating-point division. This ensures
`SUM(metric)` and `AVG(metric)` (or any other pair of semantically distinct
aggregations) are never accidentally verified through a shared
representation — each aggregation type has its own independent computation
path in both engines.

## 10. Evidence

Every evidence record that cites a metric value carries the metric's
`to_provenance_dict()` output (Section 12), so an evidence statement is
reconstructible as: what metric, how aggregated, at what grain, with what
denominator/weight — not just a bare number. `narrative_explanation()`
(Section 13) is the human-readable form of the same information, appended to
`main_finding` in `controller.py:1065-1066`.

## 11. Semantic evidence — narrative form

`MetricDefinition.narrative_explanation()` (`metric_semantics.py:117-146`)
produces an analyst-readable sentence per aggregation type, e.g.:

- SUM (confident): *"'total_revenue' was aggregated as SUM because it is
  additive at the row grain."*
- SUM (unresolved default): *"'value' was aggregated as SUM at the row
  grain as an unconfirmed default — no rate/ratio/average signal was found,
  but additivity was not positively confirmed either."*
- RATE with numerator/denominator: *"'conversion_rate' was calculated as
  SUM(conversions) / SUM(sessions) rather than averaged or summed
  directly, because it is a ratio metric whose numerator and denominator
  must both be re-aggregated at the requested grain before dividing."*
- WEIGHTED_MEAN: *"'aov' was calculated as a mean weighted by 'orders'
  rather than an unweighted row average, because larger groups should
  contribute proportionally more."*

## 12. Semantic provenance

`MetricDefinition.to_provenance_dict()` (`metric_semantics.py:98-115`)
returns: `metric_name`, `table_name`, `source_columns`, `semantic_type`,
`aggregation_type`, `numerator_column`, `denominator_column`,
`weight_column`, `unit`, `is_additive`, `is_compositional`,
`valid_aggregations`, `null_handling`, `rationale`, and
`semantic_resolution_status`. This is the full audit trail wired into the
reproducibility manifest (`controller.py:669`) — enough to reconstruct the
metric definition from the manifest alone, not merely a hash of the final
number.

## 13. Backward compatibility

- A `MetricDefinition` constructed by old/manual code without
  `semantic_resolution_status` set defaults to `"RESOLVED"` — no existing
  caller breaks.
- The `AggregationType.SUM` default on the `MetricRef`/`ExperimentIR`
  Pydantic schema (`packages/schemas/src/analysis.py:131,241`) is a wire
  default only; it is overridden by the controller with the real resolved
  aggregation on every production request (Section 3 of the audit).
- Two backward-compatibility SUM stand-ins remain, for callers that hand-build
  a `SemanticResolution` without a `metric_definition` (currently
  unreachable from any production caller). Both now set
  `semantic_resolution_status="LEGACY_FALLBACK"` so they cannot be mistaken
  for a confidently-resolved metric downstream.

## 14. Known gap carried into Phase 12 (not silently hidden)

**Resolved in v1.1** for the general case: `COUNT` is no longer selected for
every "how many X" question regardless of X's own values — see Section 4.1.
The specific failure mode named in the original Phase 11 finalization spec
(`error_events`-style columns where a row can hold more than one occurrence)
is fixed and regression-tested
(`test_sum_metric_from_how_many_question_over_event_volume_column`).

**One narrower residual remains, not resolved by v1.1:** a *dense* 0/1
occurrence-flag column (every row has a non-null value, 0 meaning "did not
occur" and 1 meaning "occurred") still resolves `COUNT(non-null)` for "how
many X", which counts every row (0s and 1s together) rather than
`SUM(flag)` (count of 1s only, the true occurrence count). This is
distinguishable from a *sparse* encoding (null means "did not occur", any
non-null value means "occurred") only with a declared metric catalog or
explicit user confirmation — not from the values alone, since both
encodings can produce an identical 0/1-or-null value distribution. This
contract does not claim this narrower case is resolved; any future phase
relying on `COUNT` selection for a dense 0/1 flag's "how many" framing
should treat it as heuristic, not guaranteed-correct, until a metric
catalog or declared-encoding mechanism exists to disambiguate it.

## 14a. Version history

- **v1.0** (Phase 11 finalization, initial pass) — original freeze. Known
  gap: COUNT was selected for every "how many X" question over a numeric
  column regardless of the column's own value distribution.
- **v1.1** (Phase 11.1) — added the value-distribution check in Section
  4.1: `max(X) > 1.0` under "how many" framing now resolves to `SUM`
  (event-volume total) instead of `COUNT` (row count). Narrowed, did not
  eliminate, the residual gap (Section 14) to dense 0/1-flag columns
  specifically. No other section of this contract changed; `MetricDefinition`
  gained no new fields in this version (the v1.0 `semantic_resolution_status`
  field already covered marking the new `SUM` branch as `"RESOLVED"`).

## 15. What's explicitly out of scope for this contract (v1.0)

Per the finalization spec, the following are **not** part of this frozen
contract and must be proposed as an explicit, versioned change if a future
phase needs them: MNAR sensitivity, selection-bias correction, target
leakage attribution, temporal semantics, timezone handling, multi-table join
planning, causal controller integration. The AI-augmentation layer's SQL
templates (`apps/api/src/ai/...`) do not yet consume this contract at all
(Section 9.2 of the audit) — bringing them into alignment is future-phase
work, not a v1.0 requirement, since the deterministic core does not depend
on them.
