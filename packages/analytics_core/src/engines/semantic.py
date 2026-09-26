"""SemanticEngine: Resolves business intent against rich Semantic World Models."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import os
import re
import numpy as np
import pandas as pd
from packages.analytics_core.src.engines.intent import InvestigationIntent
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition, MetricSemanticsResolver
from packages.schemas.src.semantic_graph import SemanticWorldModelSchema
from packages.analytics_core.src.relational.semantic_planner import find_safe_two_table_plan, find_safe_n_table_plan
from packages.analytics_core.src.relational.join_safety import assess_join_safety, assess_metric_aggregation_safety
from packages.shared.src.constants import MAX_AUTONOMOUS_RELATIONAL_HOPS


# Conceptual synonym clusters for schema-agnostic matching
SEMANTIC_SYNONYMS = {
    "revenue": {"revenue", "sales", "net_sales", "gross_sales", "turnover", "income", "gross_value", "arr_value", "arr", "mrr", "amount", "total_amount", "subtotal", "order_value", "ticket_size", "spend", "gmv", "realized_value", "realized", "economic_value", "economic", "contract_value"},
    "cost": {"cost", "infrastructure_cost", "expense", "spend", "ad_spend", "operating_expense", "operating_cost", "cogs", "loss", "budget", "outflow_value", "outflow", "campaign_cost", "expense_bucket"},
    "profit": {"profit", "margin", "gross_margin", "net_margin", "profitability", "ebitda", "gain"},
    "churn": {"churn", "cancellation", "is_churn", "churn_flag", "is_cancelled", "cancelled", "canceled", "attrition", "lost", "status", "attrition_event", "churn_event", "dropoff", "terminated", "churned", "is_churned"},
    "conversion": {"conversion", "conversions", "converted", "is_converted", "conversion_rate", "cv_rate", "success", "success_ratio", "ratio", "deterioration"},
    "delay": {"delay", "delays", "transit_delay_hours", "latency", "delay_minutes", "late_flag", "delivery_time", "wait_time", "elapsed_hours", "elapsed", "delivery", "times"},
    "discount": {"discount", "discount_rate", "discount_amount", "markdown", "rebate", "voucher_value"},
    "volume": {"volume", "quantity", "units", "items", "count", "impressions", "clicks", "orders", "transactions", "server_count", "support_tickets"},
    "segment": {"segment", "segment_tier", "tier", "billing_tier", "customer_segment", "cohort", "plan", "class", "group", "category", "plan_class", "offering", "offering_code", "expense_bucket", "bucket"},
    "region": {"region", "region_code", "territory", "market", "market_area", "location", "depot", "depot_location", "zone", "country", "state", "city", "warehouse", "market_zone", "facility_zone", "facility"},
    "channel": {"channel", "acquisition_channel", "carrier", "carrier_partner", "source", "medium", "campaign", "partner", "device_category", "cost_center", "source_route", "carrier_code", "route", "acquisition_source"},
    "customer": {"customer", "customers", "customer_id", "client", "clients", "client_key", "account", "accounts", "account_id", "account_identifier", "lead_id", "user_id", "member_id", "acct_ref", "subscriber_key", "lead_ref", "shipment_ref", "ledger_ref", "member_ref"},
    "product": {"product", "products", "product_id", "product_name", "sku", "item", "items", "listing", "offering"},
}

# BUGFIX (DEFECT-005): question tokens made of ordinary short English function
# words (articles, prepositions, auxiliary verbs, pronouns, "wh-" question
# words) must NEVER participate in substring matching against column names.
# Words like "are", "to", "is", "in", "at" are near-certain to appear as a
# coincidental substring inside unrelated column names (e.g. "are" inside
# "warehouse_region", "to" inside "stock", "is" inside "discount"), which
# previously let a single accidental letter-sequence match outscore every
# genuinely relevant column and steer primary-table/target-metric selection
# to a completely wrong table. Filtering these out of the *tokens used for
# substring/exact matching* (not out of the raw question, which is still
# used for other heuristics such as temporal parsing) fixes that class of
# false positive without weakening real matches, since none of these words
# is ever itself the name of a business concept.
SEMANTIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "and", "or", "but", "if", "so", "as", "it", "its", "this", "that",
    "which", "who", "whom", "what", "when", "where", "why", "how",
    "do", "does", "did", "will", "would", "can", "could", "should",
    "may", "might", "must", "shall", "likely", "please", "show", "me",
    "us", "we", "you", "i", "my", "our", "their", "there", "here",
}


@dataclass
class SemanticResolution:
    """Mapped physical schema bindings for the investigation."""
    primary_dataset_name: str
    target_metric_col: str
    group_dimension_col: Optional[str]
    time_col: Optional[str]
    table_grain: str
    available_numeric_cols: List[str]
    available_categorical_cols: List[str]
    world_model: Optional[SemanticWorldModelSchema] = None
    # Phase 11 (Metric Semantics Engine): the principled aggregation contract
    # for target_metric_col. Optional/defaulted so any pre-Phase-11 caller
    # constructing a SemanticResolution positionally/by-kwarg without this
    # field still works; _do_resolve always populates it.
    metric_definition: Optional[MetricDefinition] = None
    # Carried from InvestigationIntent.direction_hint -- the single
    # authoritative direction signal for every hypothesis in this
    # investigation (see hypothesis_identity.py for why this must not be
    # re-derived per-hypothesis from claim text).
    direction_hint: str = "unspecified"
    # DEFECT-005: a CORRELATION question is fundamentally bivariate (it asks
    # about the relationship BETWEEN two variables), unlike every other
    # intent type this engine was originally built for, which only ever
    # needed one target metric. Populated only when intent.intent_type ==
    # "CORRELATION" and a second, distinct numeric column can be identified;
    # None otherwise (including for every non-correlation intent).
    secondary_metric_col: Optional[str] = None
    # DEFECT-015: Churn identifiability bindings
    churn_event_col: Optional[str] = None
    churn_exposure_col: Optional[str] = None
    churn_censored_col: Optional[str] = None
    churn_confounder_cols: List[str] = field(default_factory=list)
    churn_outcome_available: bool = True
    # P0 (defect019 audit, section 8): explicit resolution status for the
    # churn event column. "RESOLVED" = exactly one candidate column was
    # found. "UNRESOLVED" = zero candidates. "AMBIGUOUS" = two or more
    # equally-plausible candidates existed and none was picked by column
    # order or any other implicit rule. churn_event_col is None whenever
    # this is not "RESOLVED".
    churn_event_resolution_status: str = "RESOLVED"
    churn_event_ambiguity: List[str] = field(default_factory=list)
    # Temporal semantics are explicit. Naive timestamps require a declared
    # business timezone before time-dependent claims are admissible.
    time_zone_policy: str = "UNRESOLVED"
    relational_access: Optional["RelationalAccess"] = None
    # Dataset names whose physical columns are explicitly referenced by the
    # question. Used only as a fail-closed guard: when a human-selected
    # multi-dataset question explicitly names fields from multiple tables, a
    # missing relational plan must never collapse silently to the primary table.
    question_referenced_datasets: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelationalAccess:
    base_table: str
    joined_table: str
    metric_column: str
    dim_column: str
    join_hops: List[Dict[str, str]]
    rationale: str
    # Autonomous discovery is bounded to a deterministic maximum hop count;
    # explicit RelationalPlan objects may carry more hops when independently
    # proven safe.
    joined_tables: List[str] = field(default_factory=list)
    # Deterministically resolved row-level predicates required by the question.
    # Each binding is {table, column, value}; values are accepted only when
    # they are observed in the actual dataset and unambiguously referenced by
    # the question.
    filters: List[Dict[str, Any]] = field(default_factory=list)


class SemanticEngine:
    """Binds high-level analytical intent to physical schemas using SemanticWorldModelBuilder."""

    def __init__(self, world_model_builder: Optional[SemanticWorldModelBuilder] = None):
        self.builder = world_model_builder or SemanticWorldModelBuilder()

    def resolve_schema(
        self,
        intent: InvestigationIntent,
        datasets_map: Dict[str, pd.DataFrame],
    ) -> SemanticResolution:
        builder = getattr(self, "builder", None) or SemanticWorldModelBuilder()
        return self._do_resolve(builder, intent, datasets_map)

    @staticmethod
    def _detect_relational_access(datasets_map, question_tokens):
        """Discover a proven-safe relational subgraph required by the question.

        The previous implementation resolved only one metric + one grouping
        column. That was unsafe for compound questions such as ``revenue by
        customer segment from the Premium product category``: the execution
        plan could contain customers + orders but silently omit the product
        restriction. This resolver now binds observed categorical values as
        row-level filters and extends the safe join subgraph to every table
        containing a required filter. Ambiguous/unproven bindings fail closed.
        """
        metrics = []
        dimensions = []
        meaningful = {str(t).lower() for t in question_tokens if str(t).lower() not in SEMANTIC_STOPWORDS}

        for table, frame in datasets_map.items():
            for col in frame.select_dtypes(include=[np.number]).columns:
                score = SemanticEngine._match_column_score(str(col), question_tokens)
                if score > 0 and not any(k in str(col).lower() for k in ("_id", "key", "code")):
                    metrics.append((score, table, str(col)))
            for col in frame.select_dtypes(include=["object", "category", "string"]).columns:
                score = SemanticEngine._match_column_score(str(col), question_tokens)
                cardinality = frame[col].nunique(dropna=True)
                n_obs = int(frame[col].notna().sum())
                uniqueness_ratio = cardinality / max(n_obs, 1)
                if (score > 0 and 2 <= cardinality <= 50 and uniqueness_ratio <= 0.80
                        and not any(k in str(col).lower() for k in ("_id", "key", "uuid"))):
                    dimensions.append((score, table, str(col)))

        def observed_filter_candidates():
            found = []
            for table, frame in datasets_map.items():
                for col in frame.select_dtypes(include=["object", "category", "string"]).columns:
                    col_score = SemanticEngine._match_column_score(str(col), question_tokens)
                    if col_score <= 0:
                        continue
                    for raw in frame[col].dropna().unique().tolist():
                        value = str(raw).strip()
                        if not value or len(value) > 80:
                            continue
                        value_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", value.lower()))
                        if not value_tokens or not value_tokens.issubset(meaningful):
                            continue
                        # Require an exact token/phrase reference to an observed
                        # value, not substring guessing (e.g. ``pre`` -> Premium).
                        phrase = " ".join(value_tokens)
                        value_score = 12.0 if value.lower() in meaningful else (10.0 if phrase and phrase in " ".join(sorted(meaningful)) else 0.0)
                        if value_score > 0:
                            found.append((value_score + col_score, table, str(col), raw))
            found.sort(key=lambda x: (-x[0], x[1], x[2], str(x[3])))
            return found

        filter_candidates = observed_filter_candidates()
        # Only accept filters that have a unique best semantic binding. This
        # avoids silently applying the same value to the wrong table/column.
        resolved_filters = []
        by_value = {}
        for item in filter_candidates:
            by_value.setdefault(str(item[3]).lower(), []).append(item)
        for value_key, items in by_value.items():
            if len(items) == 1 or items[0][0] > items[1][0]:
                resolved_filters.append(items[0])
            elif items[0][0] == items[1][0]:
                # Equal-score value bindings are ambiguous and are not guessed.
                return None

        # A column already bound to a question predicate is a constraint, not
        # the requested grouping dimension. Otherwise a phrase such as
        # ``by customer segment from the Premium product category`` creates
        # two equally scored interpretations (group by segment vs. category)
        # and the resolver cannot distinguish the intended analytical role.
        filter_columns = {(item[1], item[2]) for item in resolved_filters}
        dimensions = [d for d in dimensions if (d[1], d[2]) not in filter_columns]

        candidates = []
        for metric_score, metric_table, metric_col in metrics:
            for dim_score, dim_table, dim_col in dimensions:
                if metric_table == dim_table:
                    continue
                plan = find_safe_n_table_plan(
                    datasets_map, target_column=metric_col, grouping_column=dim_col,
                    max_hops=min(MAX_AUTONOMOUS_RELATIONAL_HOPS, max(1, len(datasets_map) - 1)),
                )
                if plan is None or plan.left_dataset != metric_table or plan.right_dataset != dim_table:
                    continue

                hops = [dict(h) for h in plan.join_hops]
                required_tables = {metric_table, dim_table}
                selected_filters = []
                valid = True
                for f_score, f_table, f_col, f_value in resolved_filters:
                    # Ignore observed values on unrelated tables unless the
                    # question explicitly binds them by their column semantics.
                    if f_table in required_tables:
                        selected_filters.append({"table": f_table, "column": f_col, "value": f_value})
                        continue
                    filter_plan = find_safe_n_table_plan(
                        datasets_map, target_column=metric_col, grouping_column=f_col,
                        max_hops=min(MAX_AUTONOMOUS_RELATIONAL_HOPS, max(1, len(datasets_map) - 1)),
                    )
                    if filter_plan is None or filter_plan.left_dataset != metric_table:
                        valid = False
                        break
                    for hop in filter_plan.join_hops:
                        if hop not in hops:
                            hops.append(dict(hop))
                    required_tables.add(f_table)
                    selected_filters.append({"table": f_table, "column": f_col, "value": f_value})
                if not valid:
                    continue

                # Re-prove every hop in the combined subgraph.
                reports = []
                for hop in hops:
                    reports.append(assess_join_safety(
                        datasets_map[hop["left_table"]], datasets_map[hop["right_table"]],
                        hop["left_key"], hop["right_key"],
                        left_table=hop["left_table"], right_table=hop["right_table"],
                    ))
                aggregate_safety = assess_metric_aggregation_safety(metric_table, reports)
                if not aggregate_safety.safe:
                    continue
                score = metric_score + dim_score + sum(f[0] for f in resolved_filters)
                candidates.append((score, metric_table, dim_table, metric_col, dim_col, hops, selected_filters, required_tables))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4], repr(item[5])))
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        _, metric_table, dim_table, metric_col, dim_col, hops, filters, required_tables = candidates[0]
        return RelationalAccess(
            metric_table, dim_table, metric_col, dim_col, hops,
            f"Safe relational subgraph covering {sorted(required_tables)}; all required observed filters were bound explicitly.",
            joined_tables=sorted(required_tables - {metric_table}),
            filters=filters,
        )

    @classmethod
    def resolve_schema_static(
        cls,
        intent: InvestigationIntent,
        datasets_map: Dict[str, pd.DataFrame],
    ) -> SemanticResolution:
        return cls().resolve_schema(intent, datasets_map)

    @staticmethod
    def _match_column_score(col_name: str, question_tokens: Set[str]) -> float:
        """Computes semantic affinity between a physical column name and question tokens."""
        col_clean = col_name.lower().replace("-", "_")
        col_parts = set(col_clean.split("_"))

        # BUGFIX (DEFECT-005): strip ordinary function words out of the token
        # set used for matching. Without this, a token like "are" would
        # substring-match inside "warehouse_region" and outscore every
        # actually-relevant column, silently steering primary-table and
        # target-metric selection to the wrong table (see SEMANTIC_STOPWORDS
        # docstring above). The raw question_tokens passed in still contains
        # these words for callers that need them; only this scoring pass
        # excludes them.
        meaningful_tokens = {t for t in question_tokens if t not in SEMANTIC_STOPWORDS}
        if not meaningful_tokens:
            return 0.0

        # 1. Exact match
        if col_clean in meaningful_tokens or col_name.lower() in meaningful_tokens:
            return 10.0

        # 1.5 Simple plural/stem match: a question token that is the column
        # name (or a column name part) plus a trailing "s"/"es", or vice
        # versa (e.g. "customers" <-> "customer", "products" <-> "product").
        # This closes a real gap where "Which customers..." never matched
        # the customers table's "customer_id"/"customer_name" columns at all
        # because the token is pluralized and the columns are not.
        #
        # BUGFIX (generalization): this used to `return 9.0` on the FIRST
        # matching token, so a column matching only one generic token (e.g.
        # "affiliate_spend" matching just "spend") scored identically to a
        # column matching multiple tokens from the question (e.g.
        # "search_spend" matching both "search" AND "spend"). On any
        # dataset with several similarly-suffixed numeric columns (common
        # in real data: several "*_spend", "*_count", "*_amount" columns),
        # every sibling column tied at the same score and the tie was then
        # broken alphabetically -- silently picking the wrong column (e.g.
        # "affiliate_spend" instead of the one actually named in the
        # question) for CORRELATION/target-metric resolution. Now every
        # matching token is counted and a small per-extra-match bonus
        # rewards the more specific, multi-token match, while staying
        # safely below the next tier (10.0 exact match).
        stem_match_count = 0
        for tok in meaningful_tokens:
            tok_stem = tok[:-2] if tok.endswith("es") else (tok[:-1] if tok.endswith("s") else tok)
            if len(tok_stem) > 2 and (tok_stem in col_parts or any(part.startswith(tok_stem) for part in col_parts)):
                stem_match_count += 1
        if stem_match_count > 0:
            return min(9.9, 9.0 + 0.2 * (stem_match_count - 1))

        # 2. Substring match -- restricted to whole underscore-delimited
        # column parts (col_parts) with a minimum token length of 4. The
        # stopword filter above is what actually prevents false positives
        # like "are" matching inside "warehouse"; this length/part
        # restriction is a second, independent guard against short
        # meaningful-but-generic tokens producing noisy partial matches.
        # Same multi-token specificity bonus as the stem-match tier above.
        substring_match_count = 0
        for tok in meaningful_tokens:
            if len(tok) > 3:
                for part in col_parts:
                    if tok in part or part in tok:
                        substring_match_count += 1
                        break
        if substring_match_count > 0:
            return min(7.9, 7.0 + 0.2 * (substring_match_count - 1))

        # 3. Synonym match
        score = 0.0
        for concept, syns in SEMANTIC_SYNONYMS.items():
            # If a question token relates to this concept
            if any(q_tok in syns or q_tok == concept for q_tok in meaningful_tokens):
                # And the column name has synonyms in this concept
                if any(cp in syns or cp == concept for cp in col_parts) or any(s in col_clean for s in syns):
                    score = max(score, 8.5)

        return score

    @staticmethod
    def _do_resolve(
        builder: SemanticWorldModelBuilder,
        intent: InvestigationIntent,
        datasets_map: Dict[str, pd.DataFrame],
    ) -> SemanticResolution:
        if not datasets_map:
            raise ValueError("No datasets available for semantic resolution.")

        # 1. Build comprehensive multi-table Semantic World Model
        world_model = builder.build_world_model(datasets_map)

        q_lower = intent.raw_question.lower()
        import re
        import os
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower))

        # Select the primary table that best matches question tokens
        primary_table = next(iter(datasets_map.keys()))
        max_matches = -1
        for tbl_name, tbl_df in datasets_map.items():
            matches = sum(SemanticEngine._match_column_score(col, q_tokens) for col in tbl_df.columns)
            if matches > max_matches:
                max_matches = matches
                primary_table = tbl_name

        primary_df = datasets_map[primary_table]

        numeric_cols = primary_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = primary_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        date_cols = primary_df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

        # 2. Match Target Metric by highest semantic affinity
        scored_metrics = []
        for col in numeric_cols:
            score = SemanticEngine._match_column_score(col, q_tokens)
            # Penalize pure integer ID columns
            col_low = col.lower()
            if any(id_k in col_low for id_k in ["_id", "id", "key", "pk", "fk", "code", "zip", "num"]):
                score -= 4.0
            scored_metrics.append((score, col))

        scored_metrics.sort(key=lambda x: (-x[0], x[1]))

        # DEFECT-005 (correlation bivariate fix): a CORRELATION question is
        # inherently about TWO named variables (e.g. "correlation between
        # unit_price and discount_rate"), so both will legitimately score
        # equally highly -- that is the expected, unambiguous case, not the
        # arbitrary tie the strict-single-top-scorer rule below exists to
        # guard against. Without this, two explicitly-named columns tying at
        # the top score caused the tie-break to reject BOTH and fall through
        # to an unrelated global/world-model default metric, silently
        # analyzing a different variable pair than the one the question
        # actually named. Take the top two distinct, positively-matched
        # columns directly for CORRELATION before the generic single-metric
        # logic runs.
        correlation_named_pair: Optional[List[str]] = None
        relation_target = None
        if getattr(intent, "intent_type", None) == "CORRELATION":
            hinted = getattr(intent, "target_metric_hint", None)
            if hinted in numeric_cols:
                relation_target = hinted
            named = [(s, c) for s, c in scored_metrics if s > 0]
            named.sort(key=lambda x: (-x[0], x[1]))
            if relation_target:
                others = [c for _score, c in named if c != relation_target]
                if others:
                    correlation_named_pair = [relation_target, others[0]]
            elif len(named) >= 2:
                correlation_named_pair = [named[0][1], named[1][1]]

        if relation_target:
            target_col = relation_target
        elif correlation_named_pair:
            target_col = correlation_named_pair[0]
        elif scored_metrics and scored_metrics[0][0] > 0 and (len(scored_metrics) == 1 or scored_metrics[0][0] > scored_metrics[1][0]):
            target_col = scored_metrics[0][1]
        elif [m for m in world_model.metrics if m.table_name == primary_table]:
            # BUGFIX (DEFECT-005): this previously read world_model.metrics[0]
            # unconditionally -- a *global*, table-unscoped list. When no
            # column in the primary table scored above 0 (e.g. a table whose
            # columns just don't relate to the question), it would silently
            # hand back a metric column belonging to a DIFFERENT table
            # entirely (observed: primary_table="inventory" but
            # target_col="quantity", a sales-table-only column), which then
            # fails IR validation with "column does not exist in table" and
            # the investigation dies before hypothesis generation ever runs.
            # Scoping this fallback to metrics whose table_name matches the
            # already-chosen primary_table keeps target_col guaranteed valid
            # for that table.
            target_col = next(m for m in world_model.metrics if m.table_name == primary_table).column_name
        else:
            # BUGFIX (DEFECT-005): this previously fell back to the literal
            # string "value" -- not a real column in ANY table -- whenever
            # the primary table had zero numeric columns and no table-scoped
            # world-model metric (e.g. a pure dimension table like
            # customers.csv: customer_id/name/email/segment/region/date,
            # with no numeric measure at all). That guaranteed an immediate
            # "column does not exist" IR validation failure before hypothesis
            # generation ever ran. MetricSemanticsResolver.resolve() already
            # knows how to turn a non-numeric or ID-like column into a
            # well-defined COUNT/COUNT_DISTINCT metric (see resolve() above),
            # so fall back to the first REAL column of the primary table
            # instead of a fabricated name -- this keeps the investigation
            # running (e.g. "count of customers by segment/region") instead
            # of crashing, and lets the loop honestly reach
            # INSUFFICIENT_EVIDENCE if that count-based signal turns out not
            # to discriminate the hypotheses, rather than never starting.
            # No metric was semantically resolved. Do not invent one from
            # dataframe order. Count-style questions can be represented later
            # by an explicit count estimand; otherwise leave the metric
            # unresolved so IR validation fails closed.
            target_col = None

        # 3. Match Group Dimension by highest semantic affinity
        entity_nouns = {"customer", "customers", "user", "users", "account", "accounts", "client", "clients", "item", "items", "record", "records"}
        dim_q_tokens = {t for t in q_tokens if t.lower() not in entity_nouns and (t.lower()[:-1] if t.lower().endswith("s") else t.lower()) not in entity_nouns}
        scored_dims = []
        for col in categorical_cols:
            score = SemanticEngine._match_column_score(col, dim_q_tokens if dim_q_tokens else q_tokens)
            if not dim_q_tokens:
                score = 0.0
            col_low = col.lower()
            # Prioritize categorical classification columns
            if any(dim_k in col_low for dim_k in ["tier", "segment", "type", "category", "region", "channel", "plan", "status", "group", "class", "band", "level", "carrier", "warehouse"]):
                score += 2.0 if score > 0 else 0.0
            # Disqualify explicit identifier columns unless specifically named
            if any(id_k in col_low for id_k in ["_id", "id", "key", "pk", "fk", "uuid"]):
                if not any(token == col_low for token in q_tokens):
                    score = -1e9
            # Penalize high-cardinality unique user/row IDs. Previously
            # gated on n_tot >= 10, which let a per-row identifier-like
            # column (e.g. "product_name" in a small products table) tie
            # with the real categorical dimension (e.g. "category") on
            # small dimension tables -- exactly where an ID-like column is
            # *most* likely to be unique per row. The ratio itself already
            # requires >80% uniqueness, so this floor was only ever
            # suppressing the one case it needed to catch; lower it so it
            # still applies to small reference/dimension tables.
            n_uniq = primary_df[col].nunique()
            n_tot = len(primary_df)
            if n_tot >= 3 and (n_uniq / n_tot) > 0.8:
                score = -1e9
            scored_dims.append((score, col))

        scored_dims.sort(key=lambda x: (-x[0], x[1]))
        if scored_dims and scored_dims[0][0] > 0 and (len(scored_dims) == 1 or scored_dims[0][0] > scored_dims[1][0]):
            group_col = scored_dims[0][1]
        elif categorical_cols:
            # Fallback grouping must reject identifier-like dimensions by BOTH
            # name and observed uniqueness.  A human-readable entity label such
            # as `product_name` is still an ID in an 8-row product catalog when
            # every value is unique; allowing it here creates an ambiguity with
            # the real business dimension (`category`) and downstream consumers
            # must then never fall back to raw column order.
            non_id_cats = [
                c for c in categorical_cols
                if not any(id_k in c.lower() for id_k in ["_id", "id", "key", "pk", "fk", "uuid"])
            ]
            candidates = []
            for c in non_id_cats:
                n_obs = int(primary_df[c].notna().sum())
                n_unique = int(primary_df[c].nunique(dropna=True))
                uniqueness_ratio = n_unique / max(n_obs, 1)
                if 2 <= n_unique <= 50 and uniqueness_ratio <= 0.80:
                    candidates.append(c)
            if len(candidates) == 1:
                group_col = candidates[0]
            else:
                group_col = None
        else:
            group_col = None

        # 4. Match Time Dimension. Never choose the first physical datetime
        # column when several plausible time dimensions exist.
        time_col = None
        explicit_times = [c for c in date_cols if str(c).lower() in q_tokens]
        if len(explicit_times) == 1:
            time_col = explicit_times[0]
        elif len(explicit_times) > 1:
            time_col = None
        else:
            wm_times = [t.column_name for t in world_model.time_dimensions
                        if t.table_name == primary_table and t.column_name in primary_df.columns]
            if len(wm_times) == 1:
                time_col = wm_times[0]
            elif len(wm_times) > 1:
                time_col = None
            else:
                named_times = [c for c in date_cols if any(t in str(c).lower() for t in ["date", "time", "period"])]
                if len(named_times) == 1:
                    time_col = named_times[0]
                else:
                    time_col = None

        # Explicit temporal timezone policy. A timezone-aware physical column is
        # already self-describing; a naive timestamp requires a declared business
        # timezone from dataframe metadata or AAOS_BUSINESS_TIMEZONE. We do not
        # guess from the machine locale.
        time_zone_policy = "NOT_APPLICABLE"
        if time_col and time_col in primary_df.columns:
            series = primary_df[time_col]
            declared_tz = (getattr(primary_df, "attrs", {}) or {}).get("business_timezone") or os.getenv("AAOS_BUSINESS_TIMEZONE")
            if pd.api.types.is_datetime64_any_dtype(series) and getattr(series.dt, "tz", None) is not None:
                time_zone_policy = "DATASET_TIMEZONE_AWARE"
            elif declared_tz:
                time_zone_policy = f"DECLARED_BUSINESS_TIMEZONE:{declared_tz}"
            else:
                time_zone_policy = "UNRESOLVED_NAIVE_TIMEZONE"

        table_grain = world_model.table_grains.get(primary_table, "record_level")

        # Phase 11: resolve the principled MetricDefinition for the chosen
        # target metric column. This is what makes metric semantics actually
        # reach the production loop, rather than the world model's decorative
        # additivity classification that nothing downstream previously read.
        metric_definition = MetricSemanticsResolver.resolve(
            metric_col=target_col,
            df=primary_df,
            table_name=primary_table,
            group_dimension_col=group_col,
            grain=table_grain,
            question_tokens=q_tokens,
        )

        # DEFECT-005: identify a secondary numeric variable for CORRELATION
        # questions. Pick the best-scoring OTHER numeric column (excluding
        # the already-chosen target and any ID-like columns), falling back
        # to the numeric column with the highest variance (a reasonable,
        # non-arbitrary default when the question doesn't name a second
        # variable explicitly, e.g. "what correlates with revenue?").
        secondary_metric_col: Optional[str] = None
        if getattr(intent, "intent_type", None) == "CORRELATION":
            if correlation_named_pair:
                # Both variables were explicitly identified together above;
                # use the second one directly rather than re-running the
                # single-scorer tie-break against target_col a second time
                # (which would hit the exact same tie and discard it again).
                secondary_metric_col = correlation_named_pair[1]
            else:
                other_numeric = [c for c in numeric_cols if c != target_col]
                candidates = []
                for col in other_numeric:
                    col_low = col.lower()
                    if any(id_k in col_low for id_k in ["_id", "id", "key", "pk", "fk", "code", "zip", "num"]):
                        continue
                    candidates.append((SemanticEngine._match_column_score(col, q_tokens), col))
                candidates.sort(key=lambda x: (-x[0], x[1]))
                if candidates and candidates[0][0] > 0 and (len(candidates) == 1 or candidates[0][0] > candidates[1][0]):
                    secondary_metric_col = candidates[0][1]
                elif other_numeric:
                    # No explicit second-variable mention in the question --
                    # fall back to the numeric column with the highest variance
                    # (most likely to be analytically interesting), skipping any
                    # remaining ID-like columns entirely rather than picking one.
                    non_id_numeric = [
                        c for c in other_numeric
                        if not any(id_k in c.lower() for id_k in ["_id", "id", "key", "pk", "fk", "code", "zip", "num"])
                    ]
                    pool = non_id_numeric or other_numeric
                    secondary_metric_col = max(pool, key=lambda c: primary_df[c].var() if primary_df[c].notna().any() else -1.0)

        # DEFECT-015: identify churn outcome, exposure, censoring, and confounders
        churn_event_col: Optional[str] = None
        churn_exposure_col: Optional[str] = None
        churn_censored_col: Optional[str] = None
        churn_confounder_cols: List[str] = []
        churn_outcome_available: bool = True
        churn_event_resolution_status: str = "RESOLVED"
        churn_event_ambiguity: List[str] = []

        intent_type = getattr(intent, "intent_type", None)
        # DEFECT-018: this token list previously only matched a subset of
        # "churn"'s common verb inflections ("churn", "churned") and missed
        # "churning"/"churns" -- unlike intent.py's ROOT_CAUSE/CHURN keyword
        # lists (BUGFIX_2026-09-13_v12), which already cover the equivalent
        # gap for "cause"/"causing". A ROOT_CAUSE-classified question like
        # "Why are customers churning?" (intent_type != "CHURN", and the
        # bare token "churning" matched none of the enumerated forms) fell
        # through this check entirely, so churn_event_col was never
        # resolved even though the dataset had an obvious, unambiguous
        # churn column -- downstream this produced a false "no identifiable
        # churn outcome" answer instead of running the actual investigation.
        is_churn_question = intent_type == "CHURN" or any(w in q_tokens for w in ["churn", "churned", "churning", "churns", "cancellation", "cancelling", "canceled", "cancelled", "attrition", "dropoff", "retention"])

        if is_churn_question:
            # 1. Search for churn event column
            churn_syns = SEMANTIC_SYNONYMS.get("churn", set()) | {
                "churn_event", "churn", "is_churn", "churn_flag", "cancelled",
                "canceled", "cancellation", "cancellation_event", "attrition",
                "attrition_flag", "dropoff", "dropout", "customer_dropout",
                "terminated", "churned", "is_churned", "has_churned"
            }
            negative_churn_keywords = [
                "_reason", "reason", "_note", "note", "_score", "score",
                "_prob", "_probability", "prob", "_offer", "offer", "_desc",
                "_text", "_comment", "_feedback"
            ]
            # P0 (defect019 audit, section 8): collect EVERY column that
            # matches the churn-outcome heuristics instead of taking the
            # first match by column order. 0 candidates -> UNRESOLVED,
            # 1 candidate -> RESOLVED, >1 candidates -> AMBIGUOUS. The
            # engine must never silently choose whichever happens to occur
            # first in the schema.
            churn_event_candidates: List[str] = []
            for col in primary_df.columns:
                c_clean = col.lower().replace("-", "_")
                if any(neg in c_clean for neg in negative_churn_keywords):
                    continue
                if c_clean in churn_syns or any(k in c_clean for k in ["churn_event", "is_churn", "churn_flag", "churn", "attrition", "cancelled", "canceled", "cancellation", "dropoff", "dropout", "has_churned"]):
                    # Verify column contains non-null values and is binary / discrete outcome
                    if primary_df[col].notna().any():
                        u_vals = set(primary_df[col].dropna().unique())
                        if len(u_vals) <= 2:
                            churn_event_candidates.append(col)

            churn_event_ambiguity: List[str] = []
            if len(churn_event_candidates) == 1:
                churn_event_col = churn_event_candidates[0]
                churn_event_resolution_status = "RESOLVED"
            elif len(churn_event_candidates) == 0:
                churn_event_col = None
                churn_event_resolution_status = "UNRESOLVED"
            else:
                churn_event_col = None
                churn_event_resolution_status = "AMBIGUOUS"
                churn_event_ambiguity = churn_event_candidates

            if churn_event_col is not None:
                churn_outcome_available = True
                # BUGFIX: this reassigns target_col to the resolved churn
                # outcome column whenever the question is churn-flavored --
                # correct on its own (the churn event should be the target
                # for e.g. "why are customers churning?"), but it used to
                # do so unconditionally, even when target_col had *already*
                # been correctly set to something else entirely (e.g. by
                # the CORRELATION named-pair logic above, for a question
                # like "is there a correlation between support tickets and
                # churn?", where target_col was support_tickets and
                # secondary_metric_col was already correctly bound to this
                # same churn_event_col as the *other* half of the named
                # pair). Overwriting target_col here without checking
                # secondary_metric_col left both fields pointing at the
                # same physical column -- a self-correlation
                # ("churned and churned are materially associated") that
                # silently discarded the actually-named explanatory
                # variable. If the old target_col is now orphaned by this
                # reassignment and secondary_metric_col was about to
                # collide with the new target_col, recover the orphaned
                # value into secondary_metric_col instead of losing it.
                if (
                    target_col is not None
                    and target_col != churn_event_col
                    and secondary_metric_col == churn_event_col
                ):
                    secondary_metric_col = target_col
                target_col = churn_event_col
            else:
                churn_outcome_available = False

            # 2. Search for exposure / person-time column
            exposure_candidates = ["observation_days", "exposure_days", "tenure_days", "tenure_months", "days_active", "exposure", "tenure"]
            exposure_matches = []
            for col in primary_df.columns:
                c_clean = str(col).lower().replace("-", "_")
                if c_clean in exposure_candidates or any(k in c_clean for k in ["observation_days", "exposure_days", "person_time"]):
                    if primary_df[col].notna().any() and pd.api.types.is_numeric_dtype(primary_df[col]):
                        exposure_matches.append(col)
            if len(exposure_matches) == 1:
                churn_exposure_col = exposure_matches[0]
            elif len(exposure_matches) > 1:
                # Multiple candidate exposure columns exist (e.g. a dataset
                # carries both a generic "observation_days" administrative
                # window and a "tenure_days" relationship length). Which one
                # is authoritative is a real, unresolved ambiguity when they
                # carry genuinely different values -- leave churn_exposure_col
                # unresolved in that case, same as before. But when every
                # candidate is numerically identical across all rows, there
                # is no statistical ambiguity in which to use for exposure/
                # person-time math, only a naming one -- break that tie
                # deterministically by preferring the more specific
                # "time-at-risk-since-signup" tenure naming over a generic
                # administrative-window name, rather than discarding a column
                # we have already unambiguously identified.
                first_vals = primary_df[exposure_matches[0]]
                all_identical = all(primary_df[c].equals(first_vals) for c in exposure_matches[1:])
                if all_identical:
                    tenure_named = [c for c in exposure_matches if "tenure" in str(c).lower()]
                    churn_exposure_col = tenure_named[0] if tenure_named else exposure_matches[0]

            # 3. Search for censoring column; ambiguity is unresolved.
            censor_candidates = ["censored", "is_censored", "right_censored", "censoring"]
            censor_matches = []
            for col in primary_df.columns:
                c_clean = str(col).lower().replace("-", "_")
                if c_clean in censor_candidates or any(k in c_clean for k in ["censored", "is_censored"]):
                    censor_matches.append(col)
            if len(censor_matches) == 1:
                churn_censored_col = censor_matches[0]

            # 4. Search for potential confounders
            excluded = {group_col, target_col, churn_event_col, churn_exposure_col, churn_censored_col}
            for col in primary_df.columns:
                if col in excluded or col is None:
                    continue
                col_low = col.lower()
                if any(id_k in col_low for id_k in ["_id", "id", "key", "pk", "fk", "code", "zip", "num", "email", "name"]):
                    continue
                # Keep categoricals or low-cardinality discrete columns as candidate confounders
                n_uniq = primary_df[col].nunique()
                if 2 <= n_uniq <= 50:
                    churn_confounder_cols.append(col)

            # Sort confounders: prioritize cohort, tenure, plan_tier
            def _confounder_priority(c_name: str) -> int:
                c_low = c_name.lower()
                if "cohort" in c_low:
                    return 0
                if "tenure" in c_low or "tier" in c_low:
                    return 1
                if "segment" in c_low or "industry" in c_low or "region" in c_low:
                    return 2
                return 3
            churn_confounder_cols.sort(key=lambda c: (_confounder_priority(c), str(c)))

        # Record only strong, explicit physical-column mentions. This is not a
        # semantic answer and is intentionally conservative: generic synonym
        # matches such as ``sales``/``amount`` do not make a dataset mandatory.
        normalized_question = re.sub(r"\s+", " ", q_lower.replace("_", " ")).strip()
        explicit_referenced_datasets: List[str] = []
        for table_name, frame in datasets_map.items():
            for col in frame.columns:
                col_text = str(col).lower()
                phrase = re.sub(r"\s+", " ", col_text.replace("_", " ")).strip()
                if col_text in q_tokens or (phrase and len(phrase) >= 4 and re.search(rf"\b{re.escape(phrase)}\b", normalized_question)):
                    explicit_referenced_datasets.append(table_name)
                    break
        explicit_referenced_datasets = sorted(set(explicit_referenced_datasets))

        relational_access = SemanticEngine._detect_relational_access(datasets_map, q_tokens)
        if relational_access is not None:
            primary_table = relational_access.base_table
            primary_df = datasets_map[primary_table]
            target_col = relational_access.metric_column
            group_col = relational_access.dim_column
            numeric_cols = primary_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = primary_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            table_grain = world_model.table_grains.get(primary_table, "record_level")
            metric_definition = MetricSemanticsResolver.resolve(metric_col=target_col, df=primary_df, table_name=primary_table, group_dimension_col=None, grain=table_grain, question_tokens=q_tokens)

        return SemanticResolution(
            primary_dataset_name=primary_table,
            target_metric_col=target_col,
            group_dimension_col=group_col,
            time_col=time_col,
            table_grain=table_grain,
            available_numeric_cols=numeric_cols,
            available_categorical_cols=categorical_cols,
            world_model=world_model,
            metric_definition=metric_definition,
            direction_hint=getattr(intent, "direction_hint", "unspecified"),
            secondary_metric_col=secondary_metric_col,
            churn_event_col=churn_event_col,
            churn_exposure_col=churn_exposure_col,
            churn_censored_col=churn_censored_col,
            churn_confounder_cols=churn_confounder_cols,
            churn_outcome_available=churn_outcome_available,
            churn_event_resolution_status=churn_event_resolution_status,
            churn_event_ambiguity=churn_event_ambiguity,
            time_zone_policy=time_zone_policy,
            relational_access=relational_access,
            question_referenced_datasets=explicit_referenced_datasets,
        )
