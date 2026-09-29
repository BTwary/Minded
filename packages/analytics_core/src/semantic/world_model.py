"""Schema-Independent Semantic World Model Builder for Multi-Table Analytical Intelligence."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from packages.analytics_core.src.relational.semantic_planner import discover_join_candidates
from packages.schemas.src.semantic_graph import (
    BusinessRuleConstraint,
    EntityNode,
    EpistemicSource,
    FunctionalDependency,
    MetricAdditivity,
    MetricNode,
    MetricTargetDirection,
    RelationshipEdge,
    SemanticWorldModelSchema,
    TimeDimensionNode,
)
from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver


class SemanticWorldModelBuilder:
    """Discovers, normalizes, and builds a comprehensive semantic world model across arbitrary enterprise datasets."""

    def build_world_model(self, datasets: Dict[str, pd.DataFrame]) -> SemanticWorldModelSchema:
        """Construct complete semantic world model using distributional and statistical data profiling."""
        model = SemanticWorldModelSchema()

        for table_name, df in datasets.items():
            # 1. Entity Extraction (Distribution & Uniqueness Driven)
            self._extract_entities(table_name, df, model)

            # 2. Metric Extraction (Distribution & Range Driven)
            self._extract_metrics(table_name, df, model)

            # 3. Time Dimensions (Temporal Continuity & Epoch Parsing)
            self._extract_time_dimensions(table_name, df, model)

            # 4. Business Rule Constraints (Empirical Bounds)
            self._extract_business_rules(table_name, df, model)

            # 5. Table Grain & Verified Candidate Keys
            grain_col = self._determine_table_grain(table_name, df, model)
            model.table_grains[table_name] = grain_col
            # `verified_grains` is consumed downstream as an evidentiary claim,
            # not as a best-effort display field.  UNKNOWN/record-level grains
            # must therefore not be populated with the first physical column,
            # because that would turn an arbitrary column into a false grain/key
            # proof.  Only explicitly/provenably keyed grains are recorded here.
            if grain_col.startswith(("entity_level (", "candidate_key (", "declared_key (")):
                key = grain_col[grain_col.find("(") + 1:-1]
                model.verified_grains[table_name] = [key] if key else []
            else:
                model.verified_grains[table_name] = []

            # 6. Functional Dependency Lattice Mining (X -> Y)
            self._extract_functional_dependencies(table_name, df, model)

        # 7. Cross-Table Relationship Graph (Fuzzy Jaccard & Containment Overlap)
        if len(datasets) > 1:
            self._extract_relationships(datasets, model)

        model.summary_description = (
            f"Semantic World Model: {len(datasets)} tables, {len(model.entities)} entities, "
            f"{len(model.metrics)} metrics, {len(model.relationships)} relationships, "
            f"{len(model.functional_dependencies)} functional dependencies, "
            f"{len(model.business_rules)} constraints."
        )
        return model

    def _extract_entities(self, table_name: str, df: pd.DataFrame, model: SemanticWorldModelSchema) -> None:
        """Detect business entities, primary keys, and correlated attributes using uniqueness and cardinality."""
        n_rows = len(df)
        if n_rows == 0:
            return

        for col in df.columns:
            series = df[col].dropna()
            n_unique = series.nunique()
            cardinality_ratio = n_unique / n_rows if n_rows > 0 else 0.0

            # Primary Key Candidate: 100% unique or high-cardinality discrete identifier.
            # A continuous float measure (e.g. a metric) is also virtually always
            # 100% unique by chance -- uniqueness alone is not identifier evidence
            # for a float column, so the dtype guard applies to both thresholds,
            # not just the >0.4 branch.
            is_pk_candidate = (
                cardinality_ratio >= 0.4
                and (pd.api.types.is_string_dtype(series) or pd.api.types.is_integer_dtype(series))
            )

            # Check semantic indicators or identifier shapes
            col_lower = col.lower()
            is_named_key = any(k in col_lower for k in ["_id", "id", "key", "code", "no", "num", "pk", "fk", "cust", "prod", "order", "acct", "user", "item"])

            if is_pk_candidate or is_named_key:
                attrs = [c for c in df.columns if c != col]
                proven = (cardinality_ratio == 1.0 and series.notna().all())
                epistemic_src = (
                    EpistemicSource.PHYSICAL_FACT if proven
                    else EpistemicSource.INFERRED_SEMANTICS
                )
                ev = (
                    f"Physical 100% uniqueness verified: {n_unique:,} distinct non-null rows."
                    if proven else
                    f"Inferred entity candidate '{col}' (distinct={n_unique:,}, ratio={cardinality_ratio:.2f})."
                )
                model.entities.append(
                    EntityNode(
                        entity_name=col.replace("_id", "").replace("_", " ").title(),
                        primary_key=col,
                        table_name=table_name,
                        natural_keys=[col],
                        attributes=attrs,
                        epistemic_source=epistemic_src,
                        evidence_statement=ev,
                        grain_proven=proven,
                        description=f"Entity identified by primary key '{col}' ({n_unique:,} distinct values, cardinality ratio: {cardinality_ratio:.2f}).",
                    )
                )

        if not any(e.table_name == table_name for e in model.entities):
            # Never manufacture an entity key from physical column order. A
            # table-level entity may be represented without a key, but it is
            # explicitly non-keyed and cannot be used for grain-sensitive
            # operations until a key is proven or declared.
            model.entities.append(
                EntityNode(
                    entity_name=table_name.replace("_", " ").title(),
                    primary_key="",
                    table_name=table_name,
                    natural_keys=[],
                    attributes=[c for c in df.columns],
                    epistemic_source=EpistemicSource.MODEL_HYPOTHESIS,
                    evidence_statement="Table-level entity hypothesis; no unique key was proven or declared.",
                    grain_proven=False,
                    description=f"Table entity '{table_name}' with {len(df.columns)} attributes; key unresolved.",
                )
            )

    def _extract_metrics(self, table_name: str, df: pd.DataFrame, model: SemanticWorldModelSchema) -> None:
        """Extract measures conservatively; never invent units or additivity.

        Semantic meaning is accepted only from explicit, high-signal schema conventions
        or trusted dataframe metadata. Distribution alone is insufficient to infer USD,
        percent, counts, or business additivity.
        """
        explicit_units = getattr(df, "attrs", {}).get("units", {}) or {}
        explicit_additivity = getattr(df, "attrs", {}).get("additivity", {}) or {}
        unit_suffixes = {
            "_usd": "USD", "_eur": "EUR", "_gbp": "GBP", "_jpy": "JPY",
            "_pct": "%", "_percent": "%", "_ratio": "ratio", "_rate": "rate",
            "_days": "days", "_day": "days", "_hours": "hours", "_minutes": "minutes",
            "_ms": "milliseconds", "_kg": "kg", "_km": "km", "_m": "meters",
            "_count": "count", "_quantity": "units",
        }
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            lower = str(col).lower()
            if any(k in lower for k in ["_id", "_key", "_code", "zip", "year"] ) or lower in {"id", "key", "code"}:
                continue

            unit = explicit_units.get(col) or next((u for suffix, u in unit_suffixes.items() if lower.endswith(suffix)), "UNKNOWN")
            add_raw = explicit_additivity.get(col)
            if add_raw:
                try:
                    additivity = MetricAdditivity(str(add_raw).upper())
                except ValueError:
                    additivity = MetricAdditivity.UNKNOWN
            else:
                additivity = MetricAdditivity.UNKNOWN

            direction = MetricTargetDirection.NEUTRAL
            if any(k in lower for k in ["cost", "churn", "loss", "error", "discount", "defect", "cancellation", "latency"]):
                direction = MetricTargetDirection.MINIMIZE
            elif any(k in lower for k in ["revenue", "sales", "profit", "margin", "conversion", "retention", "quantity", "orders"]):
                direction = MetricTargetDirection.MAXIMIZE

            # Use authoritative MetricSemanticsResolver for deep decomposition
            try:
                resolved_def = MetricSemanticsResolver.resolve(str(col), df, table_name)
                agg_type = resolved_def.aggregation_type.value
                num_col = resolved_def.numerator_column
                den_col = resolved_def.denominator_column
                wt_col = resolved_def.weight_column
                if unit == "UNKNOWN" and resolved_def.unit not in ("count", "value", "UNKNOWN"):
                    unit = resolved_def.unit
                epistemic_src = (
                    EpistemicSource.EXTERNAL_DECLARATION if (col in explicit_units or col in explicit_additivity)
                    else EpistemicSource.INFERRED_SEMANTICS
                )
                ev_statement = f"Resolved {agg_type} aggregation based on {resolved_def.rationale}"
            except Exception:
                # Semantic resolution failure is not evidence for SUM. Keep the
                # metric unresolved so downstream planning can fail closed.
                agg_type = "UNKNOWN"
                num_col, den_col, wt_col = None, None, None
                epistemic_src = EpistemicSource.INFERRED_SEMANTICS
                ev_statement = "Defaulted to SUM aggregation."

            # An explicit df.attrs declaration always wins (checked above). Absent
            # that, additivity is not invented from distribution alone -- but the
            # aggregation type the resolver just derived (from column-name tokens
            # and/or a structurally-confirmed numerator/denominator pair, never
            # from distribution shape) is itself principled evidence of whether
            # summing across rows is meaningful, so it is reused here rather than
            # leaving a second, disconnected classifier's evidence on the floor.
            # UNRESOLVED_DEFAULT_SUM means additivity was not positively confirmed.
            if additivity == MetricAdditivity.UNKNOWN and agg_type != "UNKNOWN":
                if getattr(resolved_def, "semantic_resolution_status", None) != "UNRESOLVED_DEFAULT_SUM":
                    if agg_type in ("sum", "count", "count_distinct"):
                        additivity = MetricAdditivity.FULLY_ADDITIVE
                    elif agg_type in ("mean", "median", "min", "max", "rate", "ratio", "proportion", "weighted_mean"):
                        additivity = MetricAdditivity.NON_ADDITIVE

            model.metrics.append(MetricNode(
                metric_name=str(col).replace("_", " ").title(),
                column_name=str(col),
                table_name=table_name,
                additivity=additivity,
                unit=unit,
                target_direction=direction,
                epistemic_source=epistemic_src,
                evidence_statement=ev_statement,
                aggregation_type=agg_type,
                numerator_column=num_col,
                denominator_column=den_col,
                weight_column=wt_col,
                description=(f"Metric in '{table_name}'. Semantic unit/additivity are "
                             f"{'explicitly declared' if unit != 'UNKNOWN' or additivity != MetricAdditivity.UNKNOWN else 'not established from the physical data alone'}."),
            ))

    def _extract_time_dimensions(self, table_name: str, df: pd.DataFrame, model: SemanticWorldModelSchema) -> None:
        """Identify time-series timestamps, periodicity, and date boundaries."""
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            lower = col.lower()
            is_time_name = any(k in lower for k in ["date", "time", "timestamp", "created_at", "updated_at", "period", "day", "month", "year"])
            is_datetime_type = pd.api.types.is_datetime64_any_dtype(series)

            if is_datetime_type or (is_time_name and pd.api.types.is_string_dtype(series)):
                try:
                    parsed = pd.to_datetime(series.head(50), errors="coerce")
                    if parsed.notna().sum() > 25:
                        min_d = str(parsed.min().date()) if parsed.min() is not pd.NaT else None
                        max_d = str(parsed.max().date()) if parsed.max() is not pd.NaT else None
                        model.time_dimensions.append(
                            TimeDimensionNode(
                                table_name=table_name,
                                column_name=col,
                                lowest_grain="day" if "time" not in lower else "second",
                                min_date=min_d,
                                max_date=max_d,
                            )
                        )
                except Exception:
                    pass

    def _extract_business_rules(self, table_name: str, df: pd.DataFrame, model: SemanticWorldModelSchema) -> None:
        """Extract domain and integrity constraints."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                min_v = float(series.min())
                if min_v >= 0:
                    model.business_rules.append(
                        BusinessRuleConstraint(
                            table_name=table_name,
                            column_name=col,
                            rule_type="non_negativity",
                            expression=f"{col} >= 0",
                            severity="error",
                            description=f"Empirical constraint: '{col}' is strictly non-negative (min: {min_v:,.2f}).",
                        )
                    )
                if 0.0 <= min_v and float(series.max()) <= 1.0:
                    model.business_rules.append(
                        BusinessRuleConstraint(
                            table_name=table_name,
                            column_name=col,
                            rule_type="range",
                            expression=f"0 <= {col} <= 1.0",
                            severity="warning",
                            description=f"Empirical constraint: '{col}' is bounded in [0, 1.0].",
                        )
                    )

    def _extract_functional_dependencies(self, table_name: str, df: pd.DataFrame, model: SemanticWorldModelSchema) -> None:
        """Mine exact and approximate functional dependencies X -> Y via deterministic lattice checking."""
        n_rows = len(df)
        if n_rows < 5:
            return

        cols = list(df.columns)
        for i, det_col in enumerate(cols):
            det_series = df[det_col].dropna()
            if len(det_series) < 5:
                continue
            det_n_unique = det_series.nunique()
            if det_n_unique == 1:
                continue

            for dep_col in cols:
                if det_col == dep_col:
                    continue
                
                # Check uniqueness of dep_col given det_col
                grouped = df.groupby(det_col)[dep_col].nunique()
                single_val_groups = (grouped == 1).sum()
                confidence = float(single_val_groups / max(len(grouped), 1))

                if confidence >= 0.98:
                    model.functional_dependencies.append(
                        FunctionalDependency(
                            determinant=[det_col],
                            dependent=[dep_col],
                            confidence=round(confidence, 3),
                        )
                    )

    def _extract_relationships(self, datasets: Dict[str, pd.DataFrame], model: SemanticWorldModelSchema) -> None:
        """Build only data-backed relationships that the canonical planner can prove.

        Column-name similarity generates candidates, but observed key overlap and
        the canonical join-safety authority determine the recorded relationship.
        Unsafe/unknown candidates are intentionally not promoted into the semantic
        world model as executable relationships.
        """
        existing = set()
        for plan in discover_join_candidates(datasets):
            if plan.status.upper() != "SAFE":
                continue
            key = (plan.left_dataset, plan.right_dataset, plan.left_key, plan.right_key)
            if key in existing:
                continue
            existing.add(key)

            # Re-evaluate from the same data here so the world model records the
            # independently established cardinality rather than a heuristic label.
            from packages.analytics_core.src.relational.join_safety import assess_join_safety
            safety = assess_join_safety(
                datasets[plan.left_dataset],
                datasets[plan.right_dataset],
                plan.left_key,
                plan.right_key,
                left_table=plan.left_dataset,
                right_table=plan.right_dataset,
            )
            # JoinSafetyReport intentionally does not carry a precomputed overlap count.
            # Measure it from the actual join keys rather than defaulting to a fabricated 0.
            left_keys = datasets[plan.left_dataset][plan.left_key].dropna()
            right_key_values = set(datasets[plan.right_dataset][plan.right_key].dropna().tolist())
            ov_count = int(left_keys.isin(right_key_values).sum())
            ov_score = float(max(0.0, min(1.0, plan.overlap_score)))
            is_fact = (ov_score >= 0.99 and not safety.fanout_risk)
            ep_src = EpistemicSource.PHYSICAL_FACT if is_fact else EpistemicSource.INFERRED_SEMANTICS
            ev_text = (
                f"Observed key overlap ratio {ov_score:.1%} between {plan.left_dataset}.{plan.left_key} "
                f"and {plan.right_dataset}.{plan.right_key} with cardinality {safety.cardinality.value}."
            )
            model.relationships.append(
                RelationshipEdge(
                    source_table=plan.left_dataset,
                    target_table=plan.right_dataset,
                    source_column=plan.left_key,
                    target_column=plan.right_key,
                    cardinality=safety.cardinality.value,
                    # Record the measured overlap as the relationship-strength
                    # score rather than an unconditional perfect score. The
                    # relationship is only admitted from a SAFE join plan.
                    join_safety_score=round(ov_score, 3),
                    fanout_risk=safety.fanout_risk,
                    confidence=round(ov_score, 3),
                    epistemic_source=ep_src,
                    evidence_statement=ev_text,
                    overlap_count=ov_count,
                    overlap_ratio=round(ov_score, 3),
                )
            )

    def _determine_table_grain(self, table_name: str, df: pd.DataFrame, model: SemanticWorldModelSchema) -> str:
        """Determine grain only when a reliable key is established.

        A unique arbitrary column is not automatically a business/entity grain.
        Prefer explicit metadata; otherwise fall back to a column already proven
        by _extract_entities to be a 100%-unique, fully-populated key for this
        table (grain_proven=True) -- that is a physical fact already established
        from the data itself, not a name-suffix guess, so it is reused here
        rather than re-derived by a second, narrower, name-only heuristic that
        would miss arbitrary/obfuscated key column names. A named "_id"-style
        column is still checked as a fallback for tables where uniqueness alone
        was not enough to register a proven entity. Otherwise return UNKNOWN and
        require the analytical contract to resolve it.
        """
        attrs = getattr(df, "attrs", {}) or {}
        declared = attrs.get("grain")
        if declared:
            return str(declared)
        candidates = attrs.get("primary_keys") or attrs.get("candidate_keys") or []
        if len(candidates) == 1:
            only = candidates[0]
            if isinstance(only, (list, tuple)):
                return f"declared_key ({','.join(map(str, only))})"
            return f"declared_key ({only})"
        if len(candidates) > 1:
            return "ambiguous_declared_key"
        proven = [
            e.primary_key for e in model.entities
            if e.table_name == table_name and e.grain_proven and e.primary_key in df.columns
        ]
        if len(proven) == 1:
            return f"entity_level ({proven[0]})"
        named = []
        for col in df.columns:
            lower = str(col).lower()
            if lower in {"id", "entity_id", "customer_id", "account_id", "order_id", "product_id", "user_id"} or lower.endswith("_id"):
                if df[col].notna().all() and df[col].is_unique:
                    named.append(str(col))
        if len(named) == 1:
            return f"candidate_key ({named[0]})"
        return "UNKNOWN"

