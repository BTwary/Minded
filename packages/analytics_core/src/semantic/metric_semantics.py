"""Phase 11: Metric Semantics Engine.

Removes the implicit assumption -- present throughout the pre-Phase-11 codebase
(experiment_synthesizer.py, runtime/controller.py) -- that every analytical
metric is SUM-based. Introduces a principled ``MetricDefinition`` abstraction
and a deterministic, rule-based resolver that infers the correct aggregation
semantics for a metric column from its name, its observed value distribution,
and the companion columns available in the same table.

This module performs NO network calls and requires NO AI/LLM. It is pure,
schema-agnostic, deterministic Python -- consistent with the AA-OS north star
("remain useful without AI; use AI only as an optional efficiency layer").

Design note (Phase 11 audit, Part 5 / Part 17):
    The pre-existing ``packages.schemas.src.semantic_graph.MetricNode`` already
    carried an ``additivity`` classification (FULLY_ADDITIVE / SEMI_ADDITIVE /
    NON_ADDITIVE) computed by ``SemanticWorldModelBuilder._extract_metrics``,
    but it was decorative: nothing in the production investigation loop ever
    read it. Rather than introduce a second, parallel metric-typing scheme,
    this module is the integration point that (a) reuses that additivity
    signal as one input, and (b) produces the richer ``MetricDefinition`` that
    the controller, experiment synthesizer, verification engine, and
    provenance layer now actually consume (see AUDIT_PHASE11_METRIC_SEMANTICS.md
    Part 5 "Production call graph").

Honesty note (Part 17 scope control):
    Numerator/denominator recovery is heuristic name-matching over whatever
    columns exist in the *same table* the metric was resolved against. A
    dataset that stores a precomputed rate column with no raw numerator or
    denominator anywhere in scope cannot be perfectly decomposed by any
    schema-agnostic system without a declared metric catalog -- AA-OS does not
    fabricate a denominator it cannot observe. In that case this resolver
    degrades honestly to WEIGHTED_MEAN (if a plausible volume/weight column is
    present) or MEAN (if not), and records exactly that reasoning in
    ``MetricDefinition.rationale`` rather than silently guessing. A declarative
    metric catalog (explicit numerator/denominator/weight declarations) is the
    recommended Phase 12 dependency for cases this cannot resolve.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import pandas as pd

from packages.schemas.src.analysis import AggregationType

# Column-name tokens that mark a metric as non-additive (a ratio/rate/percentage
# that must never be summed across rows or groups).
_RATE_NAME_TOKENS = {
    "rate", "pct", "percent", "percentage", "ratio", "proportion",
    "conversion", "margin_rate", "margin_pct", "utilization", "share",
    "fraction", "yield", "efficiency",
}
# Tokens that mark an already-averaged quantity (not itself decomposable into
# a numerator/denominator pair, but still non-additive across raw rows).
_MEAN_NAME_TOKENS = {"avg", "average", "mean", "aov", "aur"}
# Tokens that mark an identifier / key column, which should never be
# aggregated with SUM/MEAN if it is ever (mis)selected as a "metric".
_ID_NAME_TOKENS = {"_id", "id", "key", "pk", "fk", "code", "zip", "uuid"}
# Tokens that suggest a plausible volume/weight/count companion column,
# usable as the weight in a WEIGHTED_MEAN or as an inferred denominator.
_WEIGHT_NAME_TOKENS = {
    "count", "volume", "quantity", "units", "impressions", "clicks",
    "visits", "sessions", "orders", "transactions", "population",
    "eligible", "n", "weight", "users", "customers", "requests",
}
# Tokens that positively confirm a column is an additive magnitude (a total
# that should be summed), as opposed to an ambiguous numeric column that
# merely defaults to SUM for lack of any other signal. Superset of
# _WEIGHT_NAME_TOKENS (Phase 11 finalization pass, audit Section 6).
_ADDITIVE_NAME_TOKENS = _WEIGHT_NAME_TOKENS | {
    "total", "amount", "revenue", "sales", "cost", "spend", "spending",
    "events", "errors", "failures", "incidents", "bytes", "duration",
    "expense", "expenses", "profit", "loss", "payout", "payouts",
    "charges", "refunds", "inventory",
}


@dataclass
class MetricDefinition:
    """Principled, schema-agnostic metric contract (Phase 11 Part 5).

    Represents everything the autonomous loop needs to know to compute,
    verify, and narrate a metric correctly -- not merely its column name.
    """
    name: str
    table_name: str
    source_columns: List[str]                      # metric identity + all columns it depends on
    semantic_type: str                              # human label, e.g. "rate", "sum_measure", "count"
    aggregation_type: AggregationType
    numerator_column: Optional[str] = None
    denominator_column: Optional[str] = None
    weight_column: Optional[str] = None
    unit: str = "count"
    direction: str = "neutral"                      # increasing / decreasing / neutral
    grain: str = "row"
    is_additive: bool = True
    is_compositional: bool = False                  # True if built from numerator/denominator or weight
    valid_aggregations: List[AggregationType] = field(default_factory=list)
    null_handling: str = "exclude_nulls"
    requires_grouping: bool = False
    rationale: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Phase 11 finalization: machine-readable disambiguation between a
    # confidently-inferred aggregation and an unconfirmed default or a
    # defensive stand-in, so downstream consumers (Phase 12 sensitivity
    # analysis in particular) never mistake one for the other by reading
    # free-text rationale. One of RESOLVED / UNRESOLVED_DEFAULT_SUM /
    # LEGACY_FALLBACK. Default "RESOLVED" is backward-compatible with every
    # pre-existing construction site.
    semantic_resolution_status: str = "RESOLVED"

    @property
    def column(self) -> str:
        return self.source_columns[0] if self.source_columns else self.name

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Full audit trail of the metric-semantics decision (Phase 11 Part 14)."""
        return {
            "metric_name": self.name,
            "table_name": self.table_name,
            "source_columns": self.source_columns,
            "semantic_type": self.semantic_type,
            "aggregation_type": self.aggregation_type.value,
            "numerator_column": self.numerator_column,
            "denominator_column": self.denominator_column,
            "weight_column": self.weight_column,
            "unit": self.unit,
            "is_additive": self.is_additive,
            "is_compositional": self.is_compositional,
            "valid_aggregations": [a.value for a in self.valid_aggregations],
            "null_handling": self.null_handling,
            "rationale": self.rationale,
            "semantic_resolution_status": self.semantic_resolution_status,
        }

    def narrative_explanation(self) -> str:
        """Analyst-readable sentence explaining the aggregation choice (Phase 11 Part 15)."""
        agg = self.aggregation_type
        if agg == AggregationType.SUM:
            if self.semantic_resolution_status == "UNRESOLVED_DEFAULT_SUM":
                return (
                    f"'{self.name}' was aggregated as SUM at the {self.grain} grain as an unconfirmed "
                    f"default -- no rate/ratio/average signal was found, but additivity was not "
                    f"positively confirmed either."
                )
            return f"'{self.name}' was aggregated as SUM because it is additive at the {self.grain} grain."
        if agg == AggregationType.COUNT_DISTINCT:
            return f"'{self.name}' was aggregated as COUNT(DISTINCT) because it identifies discrete entities, not a summable quantity."
        if agg == AggregationType.COUNT:
            return f"'{self.name}' was aggregated as COUNT because it represents an occurrence count, not a summable magnitude."
        if agg in (AggregationType.RATE, AggregationType.RATIO, AggregationType.PROPORTION):
            if self.numerator_column and self.denominator_column:
                return (
                    f"'{self.name}' was calculated as SUM({self.numerator_column}) / SUM({self.denominator_column}) "
                    f"rather than averaged or summed directly, because it is a ratio metric whose numerator and "
                    f"denominator must both be re-aggregated at the requested grain before dividing."
                )
            return (
                f"'{self.name}' is a non-additive {agg.value} metric with no observable numerator/denominator pair "
                f"in this dataset; it was aggregated as a {'weighted' if self.weight_column else 'plain'} mean "
                f"rather than summed, to avoid producing a meaningless total-of-percentages."
            )
        if agg == AggregationType.WEIGHTED_MEAN:
            return (
                f"'{self.name}' was calculated as a mean weighted by '{self.weight_column}' rather than an "
                f"unweighted row average, because larger groups should contribute proportionally more."
            )
        if agg == AggregationType.MEAN:
            return f"'{self.name}' was averaged (MEAN) rather than summed, because it is not additive across rows."
        return f"'{self.name}' was aggregated as {agg.value}."


class MetricSemanticsResolver:
    """Deterministic, rule-based resolver: column + table -> MetricDefinition.

    No AI, no network calls, no fabrication. Every decision is derived from
    (a) the column's name, (b) its observed value distribution, and (c) the
    presence of plausible companion columns in the same table. Every decision
    records its rationale so a reviewer can see exactly why a given
    aggregation was chosen (Phase 11 Part 14/15).
    """

    @staticmethod
    def _tokens(col: str) -> List[str]:
        return [t for t in re.split(r"[_\-\s]+", col.lower()) if t]

    @staticmethod
    def _looks_like_id(col: str) -> bool:
        low = col.lower()
        return low == "id" or any(low.endswith(f"_{t}") or low == t for t in ("id", "key", "pk", "fk", "uuid")) or "zip" in low

    @staticmethod
    def _has_weight_token(low_col_name: str) -> bool:
        """Token-level (not substring) match against _WEIGHT_NAME_TOKENS.

        Deliberately NOT a raw substring check: several tokens (notably the
        bare "n") are short enough that a naive `tok in low` would false-
        positive on almost any column name that happens to contain that
        letter sequence (e.g. "revenue" contains "n"). Splitting the column
        name into underscore/hyphen-delimited parts and requiring exact
        token membership avoids that class of bug.
        """
        parts = set(re.split(r"[_\-\s]+", low_col_name))
        return bool(parts & _WEIGHT_NAME_TOKENS)

    @staticmethod
    def _find_numerator_denominator_pair(metric_col: str, df: pd.DataFrame):
        """For a PRECOMPUTED rate/ratio/proportion column (e.g. 'conversion_rate'),
        look for both a raw numerator count column (e.g. 'conversions', sharing
        the metric's base name) AND a separate raw denominator/exposure column
        (e.g. 'sessions', 'eligible_users') in the same table, so the ratio can
        be correctly recomposed as SUM(numerator) / SUM(denominator) rather than
        summing the already-divided rate column itself (which would be wrong --
        SUM(conversion_rate) is not SUM(conversions)/SUM(sessions)).

        Returns (numerator_column, denominator_column), either of which may be
        None if a confident pair cannot be identified.
        """
        base = re.sub(r"(_rate|_pct|_percent|_percentage|_ratio|_proportion)$", "", metric_col.lower())
        if not base or base == metric_col.lower():
            return None, None
        numer_candidates, denom_candidates = [], []
        for col in df.columns:
            if col == metric_col or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            low = col.lower()
            if MetricSemanticsResolver._looks_like_id(col):
                continue
            base_singular = base.rstrip("s")
            if base in low or low.rstrip("s") == base_singular or low.rstrip("s").startswith(base_singular):
                numer_candidates.append(col)
            elif MetricSemanticsResolver._has_weight_token(low):
                denom_candidates.append(col)
        if len(numer_candidates) == 1 and len(denom_candidates) == 1:
            return numer_candidates[0], denom_candidates[0]
        return None, None

    @staticmethod
    def _find_denominator_pair(metric_col: str, df: pd.DataFrame) -> Optional[str]:
        """Look for a plausible volume/exposure column to use as a WEIGHT when
        no true raw numerator/denominator pair exists (see
        ``_find_numerator_denominator_pair`` for the fully-recomposable case).
        """
        candidates = []
        for col in df.columns:
            if col == metric_col or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            low = col.lower()
            if MetricSemanticsResolver._looks_like_id(col):
                continue
            if MetricSemanticsResolver._has_weight_token(low):
                candidates.append(col)
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _find_weight_column(metric_col: str, df: pd.DataFrame, group_col: Optional[str]) -> Optional[str]:
        """Find a plausible volume/weight column to weight a non-additive metric's mean by."""
        best = None
        for col in df.columns:
            if col == metric_col or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            low = col.lower()
            if MetricSemanticsResolver._looks_like_id(col):
                continue
            if MetricSemanticsResolver._has_weight_token(low):
                if best is not None:
                    return None
                best = col
        return best

    @staticmethod
    def resolve(
        metric_col: str,
        df: pd.DataFrame,
        table_name: str = "dataset",
        group_dimension_col: Optional[str] = None,
        grain: str = "row",
        question_tokens: Optional[set] = None,
    ) -> MetricDefinition:
        """Infer the MetricDefinition for a resolved target metric column."""
        question_tokens = question_tokens or set()

        # No metric column was semantically resolved upstream (e.g. a pure
        # dimension table with no numeric measure at all -- see the
        # DEFECT-005 comment in engines/semantic.py: target_col is
        # deliberately left None rather than fabricated, so IR validation
        # can fail closed). `_tokens()` and the distribution probing below
        # both assume a real column name/Series and would otherwise crash
        # with AttributeError on `None.lower()` before validation ever runs.
        # Return an explicitly unresolved definition instead of guessing.
        if not metric_col or metric_col not in df.columns:
            return MetricDefinition(
                name=metric_col or "",
                table_name=table_name,
                source_columns=[metric_col] if metric_col else [],
                semantic_type="unresolved",
                aggregation_type=AggregationType.COUNT,
                unit="count",
                grain=grain,
                is_additive=False,
                valid_aggregations=[AggregationType.COUNT],
                semantic_resolution_status="UNRESOLVED_NO_METRIC",
                rationale=(
                    "No numeric or otherwise resolvable metric column was identified "
                    "for this table; no aggregation semantics could be inferred."
                ),
            )

        tokens = set(MetricSemanticsResolver._tokens(metric_col))
        series = df[metric_col].dropna() if metric_col in df.columns else pd.Series(dtype=float)
        is_numeric = pd.api.types.is_numeric_dtype(df[metric_col]) if metric_col in df.columns else False

        # --- 1. Explicit identifier requested as a metric -> COUNT_DISTINCT ---
        wants_count = bool({"how", "many", "number", "count"} & question_tokens)
        if MetricSemanticsResolver._looks_like_id(metric_col):
            return MetricDefinition(
                name=metric_col,
                table_name=table_name,
                source_columns=[metric_col],
                semantic_type="distinct_count",
                aggregation_type=AggregationType.COUNT_DISTINCT,
                unit="entities",
                grain=grain,
                is_additive=False,
                valid_aggregations=[AggregationType.COUNT_DISTINCT, AggregationType.COUNT],
                null_handling="exclude_nulls",
                rationale=(
                    f"'{metric_col}' is named like a unique business identifier; it identifies "
                    f"entities and must be counted (COUNT DISTINCT), never summed or averaged."
                ),
            )

        if not is_numeric or len(series) == 0:
            # Non-numeric column resolved as a "metric" -> only a count makes sense.
            return MetricDefinition(
                name=metric_col,
                table_name=table_name,
                source_columns=[metric_col],
                semantic_type="count",
                aggregation_type=AggregationType.COUNT,
                unit="occurrences",
                grain=grain,
                is_additive=False,
                valid_aggregations=[AggregationType.COUNT, AggregationType.COUNT_DISTINCT],
                rationale=f"'{metric_col}' is non-numeric; only occurrence counting is well-defined.",
            )

        min_v, max_v = float(series.min()), float(series.max())
        bounded_unit_interval = (-1e-9 <= min_v) and (max_v <= 1.0 + 1e-9)
        bounded_percent = (-1e-9 <= min_v) and (max_v <= 100.0 + 1e-9) and max_v > 1.0 + 1e-9

        rate_like_name = bool(tokens & _RATE_NAME_TOKENS)
        mean_like_name = bool(tokens & _MEAN_NAME_TOKENS)

        # --- 2. Explicit "how many" question over a numeric column ---
        # v1.1 (Phase 11.1): distinguish, from the column's OWN observed
        # values (not its name), whether a single row can already record
        # more than one occurrence. If so, a row/occurrence COUNT would
        # silently undercount every such row -- the correct answer is the
        # total volume (SUM), not the row count. See docs/METRIC_SEMANTICS.md
        # Section 4.1 and CHANGELOG_PHASE11_1.md.
        if wants_count and not rate_like_name and not mean_like_name:
            max_v_for_count = float(series.max()) if len(series) else 0.0
            if max_v_for_count > 1.0 + 1e-9:
                return MetricDefinition(
                    name=metric_col,
                    table_name=table_name,
                    source_columns=[metric_col],
                    semantic_type="event_volume_sum",
                    aggregation_type=AggregationType.SUM,
                    unit="occurrences",
                    grain=grain,
                    is_additive=True,
                    valid_aggregations=[AggregationType.SUM],
                    semantic_resolution_status="RESOLVED",
                    rationale=(
                        f"Question asks 'how many' over '{metric_col}', but its own observed values "
                        f"show at least one row records more than one occurrence (max={max_v_for_count:g} "
                        f"> 1); a row/occurrence COUNT would silently undercount those rows, so this "
                        f"resolves to SUM (total volume across rows) instead."
                    ),
                )
            return MetricDefinition(
                name=metric_col,
                table_name=table_name,
                source_columns=[metric_col],
                semantic_type="count",
                aggregation_type=AggregationType.COUNT,
                unit="occurrences",
                grain=grain,
                is_additive=False,
                valid_aggregations=[AggregationType.COUNT],
                semantic_resolution_status="RESOLVED",
                rationale=(
                    f"Question asks 'how many' over '{metric_col}'; every observed value is consistent "
                    f"with at most one occurrence per row (max<=1), so resolved to a row/occurrence COUNT."
                ),
            )

        # --- 3. Rate / Ratio / Proportion (non-additive, denominator-dependent) ---
        if rate_like_name or (bounded_unit_interval and not mean_like_name and len(series.unique()) > 2):
            # First choice: a TRUE raw numerator + denominator pair, which lets
            # us recompose the exact ratio as SUM(numerator)/SUM(denominator)
            # rather than approximate it (Phase 11 Part 8).
            num_col, denom_col = MetricSemanticsResolver._find_numerator_denominator_pair(metric_col, df)
            # Fallback: no raw numerator exists in this table (only the
            # precomputed rate column itself), so at best we can find a
            # plausible volume/exposure column to weight the mean by.
            weight_col = None if (num_col and denom_col) else MetricSemanticsResolver._find_denominator_pair(metric_col, df)

            if "ratio" in tokens and not bounded_unit_interval:
                agg_type = AggregationType.RATIO
                sem_type = "ratio"
            elif "conversion" in tokens or "rate" in tokens:
                agg_type = AggregationType.RATE
                sem_type = "rate"
            else:
                agg_type = AggregationType.PROPORTION
                sem_type = "proportion"

            if num_col and denom_col:
                return MetricDefinition(
                    name=metric_col,
                    table_name=table_name,
                    source_columns=[metric_col, num_col, denom_col],
                    semantic_type=sem_type,
                    aggregation_type=agg_type,
                    numerator_column=num_col,
                    denominator_column=denom_col,
                    unit="%" if (bounded_unit_interval or bounded_percent) else "ratio",
                    grain=grain,
                    is_additive=False,
                    is_compositional=True,
                    requires_grouping=False,
                    valid_aggregations=[agg_type],
                    null_handling="exclude_nulls",
                    rationale=(
                        f"'{metric_col}' is name- and/or range-consistent with a {sem_type} "
                        f"(range [{min_v:.3f}, {max_v:.3f}]); a raw numerator column '{num_col}' and "
                        f"denominator column '{denom_col}' were both found in the same table, so the metric "
                        f"is recomposed as SUM({num_col}) / SUM({denom_col}) at the requested grain -- the "
                        f"exact ratio -- rather than summing or naively averaging the precomputed "
                        f"'{metric_col}' column itself."
                    ),
                )

            return MetricDefinition(
                name=metric_col,
                table_name=table_name,
                source_columns=[metric_col] + ([weight_col] if weight_col else []),
                semantic_type=sem_type,
                aggregation_type=AggregationType.WEIGHTED_MEAN if weight_col else AggregationType.MEAN,
                weight_column=weight_col,
                unit="%" if (bounded_unit_interval or bounded_percent) else "ratio",
                grain=grain,
                is_additive=False,
                is_compositional=bool(weight_col),
                valid_aggregations=[AggregationType.WEIGHTED_MEAN if weight_col else AggregationType.MEAN],
                null_handling="exclude_nulls",
                rationale=(
                    f"'{metric_col}' is name- and/or range-consistent with a {sem_type} "
                    f"(range [{min_v:.3f}, {max_v:.3f}]), but no raw numerator/denominator pair is "
                    f"observable in this table. " +
                    (
                        f"'{weight_col}' was identified as a plausible volume/weight column, so the metric "
                        f"is aggregated as a mean weighted by '{weight_col}' rather than an unweighted "
                        f"average or a SUM, which would silently sum a total-of-percentages."
                        if weight_col else
                        "No plausible weight/volume column was found either, so the metric is aggregated "
                        "as a plain (unweighted) mean. SUM is explicitly disallowed for this metric because "
                        "summing a rate/ratio across rows or groups produces a meaningless total."
                    )
                ),
            )

        # --- 4. Already-averaged quantity by name (e.g. average_order_value) ---
        if mean_like_name:
            weight_col = MetricSemanticsResolver._find_weight_column(metric_col, df, group_dimension_col)
            return MetricDefinition(
                name=metric_col,
                table_name=table_name,
                source_columns=[metric_col] + ([weight_col] if weight_col else []),
                semantic_type="mean_measure",
                aggregation_type=AggregationType.WEIGHTED_MEAN if weight_col else AggregationType.MEAN,
                weight_column=weight_col,
                unit="value",
                grain=grain,
                is_additive=False,
                is_compositional=bool(weight_col),
                valid_aggregations=[AggregationType.WEIGHTED_MEAN if weight_col else AggregationType.MEAN],
                rationale=(
                    f"'{metric_col}' is named as an average/mean quantity, so it is aggregated as a "
                    + (f"mean weighted by '{weight_col}'" if weight_col else "plain mean")
                    + " rather than summed -- summing an average across rows or groups is not meaningful."
                ),
            )

        # --- 5. Default: no rate/mean/id signal and no bounded-proportion range ---
        # Phase 11 finalization (audit Section 6): a column name that positively
        # confirms additivity (_ADDITIVE_NAME_TOKENS) is RESOLVED as SUM with
        # the same confidence as before. A column with NO naming signal at all
        # still defaults to SUM (so a schema-agnostic system doesn't stall on
        # every first-contact column), but is marked UNRESOLVED_DEFAULT_SUM so
        # it is never mistaken, by any consumer, for a confidently-inferred SUM.
        has_additive_token = bool(tokens & _ADDITIVE_NAME_TOKENS)
        if has_additive_token:
            return MetricDefinition(
                name=metric_col,
                table_name=table_name,
                source_columns=[metric_col],
                semantic_type="sum_measure",
                aggregation_type=AggregationType.SUM,
                unit="value",
                grain=grain,
                is_additive=True,
                valid_aggregations=[AggregationType.SUM, AggregationType.MEAN, AggregationType.COUNT],
                semantic_resolution_status="RESOLVED",
                rationale=(
                    f"'{metric_col}' matches a recognized additive-magnitude naming token "
                    f"(observed range [{min_v:,.2f}, {max_v:,.2f}]); aggregated as SUM at the "
                    f"{grain} grain."
                ),
            )
        return MetricDefinition(
            name=metric_col,
            table_name=table_name,
            source_columns=[metric_col],
            semantic_type="sum_measure",
            aggregation_type=AggregationType.SUM,
            unit="value",
            grain=grain,
            is_additive=True,
            valid_aggregations=[AggregationType.SUM, AggregationType.MEAN, AggregationType.COUNT],
            semantic_resolution_status="UNRESOLVED_DEFAULT_SUM",
            rationale=(
                f"'{metric_col}' shows no rate/ratio/average/additive-magnitude naming signal and no "
                f"bounded-proportion range (observed range [{min_v:,.2f}, {max_v:,.2f}]); SUM is used "
                f"as an unconfirmed default so the investigation is not blocked, but additivity was "
                f"NOT positively confirmed -- this is an UNRESOLVED_DEFAULT_SUM, not a confident "
                f"inference."
            ),
        )

    @staticmethod
    def sql_aggregation_expression(metric_def: MetricDefinition, alias: str = "total_metric") -> str:
        """Build the correct SQL aggregation expression for this metric's semantics."""
        agg = metric_def.aggregation_type
        col = metric_def.column
        if agg == AggregationType.SUM:
            return f"SUM({col}) AS {alias}"
        if agg == AggregationType.MEAN:
            return f"AVG({col}) AS {alias}"
        if agg == AggregationType.COUNT:
            return f"COUNT({col}) AS {alias}"
        if agg == AggregationType.COUNT_DISTINCT:
            return f"COUNT(DISTINCT {col}) AS {alias}"
        if agg in (AggregationType.RATE, AggregationType.RATIO, AggregationType.PROPORTION):
            if metric_def.numerator_column and metric_def.denominator_column:
                return (
                    f"SUM({metric_def.numerator_column}) / NULLIF(SUM({metric_def.denominator_column}), 0) AS {alias}"
                )
            if metric_def.weight_column:
                return (
                    f"SUM({col} * {metric_def.weight_column}) / NULLIF(SUM({metric_def.weight_column}), 0) AS {alias}"
                )
            return f"AVG({col}) AS {alias}"
        if agg == AggregationType.WEIGHTED_MEAN:
            if metric_def.weight_column:
                return (
                    f"SUM({col} * {metric_def.weight_column}) / NULLIF(SUM({metric_def.weight_column}), 0) AS {alias}"
                )
            return f"AVG({col}) AS {alias}"
        return f"SUM({col}) AS {alias}"
