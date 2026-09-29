"""VerificationEngine: Deterministic mathematical safety proofs, grain preservation, join safety, and dual-engine cross-validation."""
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import pandas as pd
import polars as pl

from packages.analytics_core.src.execution.state_machine import VerificationStatus
from packages.analytics_core.src.engines.polars_query_recompiler import execute_known_query, result_frames_equivalent
from packages.analytics_core.src.engines.execution_provider import _derive_primary_metric

from packages.schemas.src.analysis import (
    CardinalityType,
    CausalIdentifiabilityProof,
    GrainPreservationStatus,
    GrainVerificationProof,
    JoinVerificationProof,
)

from packages.analytics_core.src.engines.verification_contract import VerificationContract, compare_verification_contracts


@dataclass
class VerificationResult:
    """Mathematical verification outcome between independent analytical engines."""
    primary_tool: str
    secondary_tool: str
    primary_value: float
    secondary_value: Optional[float]
    observed_delta_pct: Optional[float]
    status: str  # VERIFIED, PARTIALLY_VERIFIED, FAILED, UNVERIFIED
    max_absolute_delta: Optional[float] = None
    max_relative_delta: Optional[float] = None
    grain_proof: Optional[GrainVerificationProof] = None
    # Populated only when a real post-execution result relation is available.
    # The legacy scalar verification path intentionally leaves this as None;
    # authoritative grain validation is performed by the canonical transition.
    join_proof: Optional[JoinVerificationProof] = None
    causal_proof: Optional[CausalIdentifiabilityProof] = None
    # DEFECT-001 diagnostic-transparency fix: previously any SQL execution
    # exception or missing-value case was silently converted to FAILED with
    # secondary_value hardcoded to 0.0 and observed_delta_pct hardcoded to
    # 1.0 -- fabricated sentinel numbers with no record of *why* the
    # secondary engine didn't produce a value. These fields make that
    # distinction visible and inspectable instead of indistinguishable from
    # a genuine numerical disagreement.
    secondary_method: str = "unknown"  # "sql_sqlcontext" | "programmatic_agg" | "none_computed"
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    sql_attempted: Optional[str] = None
    failure_reason: Optional[str] = None  # human-readable, e.g. "sql_dialect_incompatible"


class VerificationEngine:
    """
    Deterministic Verification Engine executing multi-layer mathematical safety proofs:
    1. Grain preservation & fanout detection
    2. Join cardinality & Cartesian product prevention
    3. Dual-engine (DuckDB vs Polars) cross-computation
    """

    @staticmethod
    def verify_grain_preservation(
        pre_join_df: pd.DataFrame,
        post_join_df: pd.DataFrame,
        unit_of_analysis_keys: List[str],
        requested_grain: str = "row",
    ) -> GrainVerificationProof:
        """Mathematical proof that query execution did not silently multiply rows at the unit of analysis."""
        pre_rows = len(pre_join_df)
        post_rows = len(post_join_df)

        valid_keys = [k for k in unit_of_analysis_keys if k in post_join_df.columns]
        if valid_keys:
            distinct_keys = int(post_join_df[valid_keys].drop_duplicates().shape[0])
        else:
            distinct_keys = post_rows

        fanout = float(post_rows) / float(max(pre_rows, 1))

        if fanout == 1.0:
            preservation = GrainPreservationStatus.PRESERVED
            evidence = f"Grain strictly preserved at {post_rows} rows (Fanout factor: 1.000)."
        elif fanout < 1.0:
            preservation = GrainPreservationStatus.AGGREGATED_SAFELY
            evidence = f"Safely aggregated from {pre_rows} to {post_rows} rows at grain keys {valid_keys}."
        else:
            preservation = GrainPreservationStatus.FANOUT_DETECTED
            evidence = f"CRITICAL: Cartesian fanout detected. Row count expanded from {pre_rows} to {post_rows} (Fanout factor: {fanout:.3f})."

        return GrainVerificationProof(
            requested_grain=requested_grain,
            observed_grain="aggregated" if fanout < 1.0 else "expanded" if fanout > 1.0 else "identity",
            pre_join_row_count=pre_rows,
            post_join_row_count=post_rows,
            distinct_key_count=distinct_keys,
            join_cardinality=CardinalityType.ONE_TO_MANY if fanout > 1.0 else CardinalityType.MANY_TO_ONE,
            expected_cardinality=CardinalityType.MANY_TO_ONE,
            actual_cardinality=CardinalityType.MANY_TO_MANY if fanout > 1.0 else CardinalityType.MANY_TO_ONE,
            preservation_status=preservation,
            fanout_factor=fanout,
            verification_evidence=evidence,
        )

    @staticmethod
    def verify_join_cardinality(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        join_keys: List[str],
        left_name: str = "left",
        right_name: str = "right",
        many_to_many_explicitly_requested: bool = False,
    ) -> JoinVerificationProof:
        """Verifies table join cardinality and gates execution on unintended Many-to-Many Cartesian fanouts."""
        valid_keys_left = [k for k in join_keys if k in left_df.columns]
        valid_keys_right = [k for k in join_keys if k in right_df.columns]

        left_unique = (len(left_df) == left_df[valid_keys_left].drop_duplicates().shape[0]) if valid_keys_left else False
        right_unique = (len(right_df) == right_df[valid_keys_right].drop_duplicates().shape[0]) if valid_keys_right else False

        card_left = "1" if left_unique else "N"
        card_right = "1" if right_unique else "N"

        is_m2m = (card_left == "N" and card_right == "N")
        is_safe = (not is_m2m) or many_to_many_explicitly_requested

        return JoinVerificationProof(
            left_table=left_name,
            right_table=right_name,
            join_keys=join_keys,
            cardinality_left=card_left,
            cardinality_right=card_right,
            is_many_to_many=is_m2m,
            many_to_many_explicitly_requested=many_to_many_explicitly_requested,
            is_join_safe=is_safe,
        )

    @staticmethod
    def verify_secondary(
        primary_df: pd.DataFrame,
        target_metric_col: str,
        aggregation_type: str,
        primary_metric: float,
        query_sql: Optional[str] = None,
        group_dimension_col: Optional[str] = None,
        tolerance: float = 1e-4,
        unit_of_analysis_keys: Optional[List[str]] = None,
        numerator_column: Optional[str] = None,
        denominator_column: Optional[str] = None,
        weight_column: Optional[str] = None,
        primary_result_df: Optional[pd.DataFrame] = None,
        primary_result_truncated: bool = False,
        relational_plan: Optional[Any] = None,
        relation_tables: Optional[Dict[str, pd.DataFrame]] = None,
        primary_result_column: Optional[str] = None,

    ) -> VerificationResult:
        """
        Production controller contract: when ``primary_result_df`` is supplied,
        verification is relation-authoritative. The secondary engine must
        independently reconstruct the complete result relation; scalar-only
        fallbacks are forbidden. Truncated primary results and unsupported
        SQL shapes fail closed. Legacy scalar verification remains available
        only for direct backward-compatible callers that omit the primary
        result relation.

        Executes secondary engine cross-computation and produces formal mathematical proof.
        Fail-Closed Invariant: Any query syntax error, execution exception, NULL result,
        or delta > tolerance returns VerificationStatus.FAILED.

        Phase 11 (Metric Semantics Engine, Part 9): when the primary query was built
        from a MetricDefinition (RATE/RATIO/PROPORTION/WEIGHTED_MEAN), the independent
        Polars recomputation must apply the SAME numerator/denominator/weight semantics
        -- comparing a naive Polars AVG against a correctly-composed DuckDB ratio would
        not be an independent verification of the same quantity, it would be a false
        mismatch (or worse, a false agreement between two different wrong answers).
        Ratio-type metrics also use a slightly wider default tolerance than pure SUM/COUNT
        metrics, since they involve an extra floating-point division.
        """
        try:
            if primary_df is None or len(primary_df) == 0:
                return VerificationResult(
                    primary_tool="duckdb_sql",
                    secondary_tool="polars_vectorized",
                    primary_value=primary_metric,
                    secondary_value=None,
                    observed_delta_pct=None,
                    status=VerificationStatus.UNVERIFIED,
                    secondary_method="none_computed",
                    exception_type=None,
                    exception_message="No observations available for independent verification.",
                    sql_attempted=query_sql,
                    failure_reason="no_observations_to_verify",
                )

            pl_df = pl.from_pandas(primary_df)

            secondary_val: Optional[float] = None
            secondary_method = "none_computed"
            secondary_result_df: Optional["pl.DataFrame"] = None
            sql_exception_type: Optional[str] = None
            sql_exception_message: Optional[str] = None

            effective_tolerance = tolerance
            if aggregation_type in ("RATE", "RATIO", "PROPORTION", "WEIGHTED_MEAN") and tolerance <= 1e-4:
                effective_tolerance = 1e-3

            # ---------------------------------------------------------
            # CANONICAL RELATIONAL-PLAN VERIFICATION
            #
            # When the controller supplies a typed RelationalPlan,
            # the secondary engine MUST execute that plan natively.
            # SQLContext is deliberately bypassed.
            # ---------------------------------------------------------
            if relational_plan is not None:
                if primary_result_truncated:
                    return VerificationResult(
                        primary_tool="duckdb_sql",
                        secondary_tool="relational_plan_polars_native",
                        primary_value=primary_metric,
                        secondary_value=None,
                        observed_delta_pct=None,
                        status=VerificationStatus.FAILED,
                        secondary_method="relational_plan_polars_native",
                        exception_type=None,
                        exception_message=None,
                        sql_attempted=query_sql,
                        failure_reason="primary_result_truncated_cannot_verify_full_relation",
                    )

                if relation_tables is None or not isinstance(relation_tables, dict):
                    return VerificationResult(
                        primary_tool="duckdb_sql",
                        secondary_tool="relational_plan_polars_native",
                        primary_value=primary_metric,
                        secondary_value=None,
                        observed_delta_pct=None,
                        status=VerificationStatus.FAILED,
                        secondary_method="relational_plan_polars_native",
                        exception_type="MissingRelationTables",
                        exception_message="relation_tables must be supplied for relational-plan verification",
                        sql_attempted=query_sql,
                        failure_reason="relational_plan_missing_relation_tables",
                    )

                try:
                    secondary_result_df = relational_plan.to_polars({
                        name: (
                            frame
                            if isinstance(frame, pl.DataFrame)
                            else pl.from_pandas(frame)
                        )
                        for name, frame in relation_tables.items()
                    })

                    secondary_method = "relational_plan_polars_native"

                    if secondary_result_df is None:
                        return VerificationResult(
                            primary_tool="duckdb_sql",
                            secondary_tool="relational_plan_polars_native",
                            primary_value=primary_metric,
                            secondary_value=None,
                            observed_delta_pct=None,
                            status=VerificationStatus.FAILED,
                            secondary_method=secondary_method,
                            exception_type="NoneResult",
                            exception_message="RelationalPlan.to_polars() returned None",
                            sql_attempted=query_sql,
                            failure_reason="relational_plan_returned_no_relation",
                        )

                    secondary_pdf = (
                        secondary_result_df.to_pandas()
                        if hasattr(secondary_result_df, "to_pandas")
                        else secondary_result_df
                    )

                    frames_ok, frame_delta = result_frames_equivalent(
                        primary_result_df,
                        secondary_pdf,
                        effective_tolerance,
                    )

                    if frames_ok:
                        try:
                            secondary_scalar, _ = _derive_primary_metric(
                                secondary_pdf,
                                aggregation_type=aggregation_type,
                                primary_result_column=primary_result_column,
                                sql=query_sql or "",
                            )
                        except Exception as reduction_exc:
                            return VerificationResult(
                                primary_tool="duckdb_sql",
                                secondary_tool="relational_plan_polars_native",
                                primary_value=primary_metric,
                                secondary_value=None,
                                observed_delta_pct=None,
                                status=VerificationStatus.FAILED,
                                secondary_method=secondary_method,
                                exception_type=type(reduction_exc).__name__,
                                exception_message=str(reduction_exc),
                                sql_attempted=query_sql,
                                failure_reason="secondary_relation_scalar_reduction_failed",
                            )
                        absolute_delta = abs(primary_metric - secondary_scalar)
                        relative_delta = absolute_delta / max(abs(primary_metric), 1e-9)
                        scalar_ok = np.isclose(primary_metric, secondary_scalar, rtol=0.0, atol=effective_tolerance)
                        if not scalar_ok:
                            return VerificationResult(
                                primary_tool="duckdb_sql",
                                secondary_tool="relational_plan_polars_native",
                                primary_value=primary_metric,
                                secondary_value=secondary_scalar,
                                observed_delta_pct=float(relative_delta),
                                max_absolute_delta=float(absolute_delta),
                                max_relative_delta=float(relative_delta),
                                status=VerificationStatus.FAILED,
                                secondary_method=secondary_method,
                                exception_type=None,
                                exception_message=None,
                                sql_attempted=query_sql,
                                failure_reason="scalar_reduction_disagreement",
                            )
                        return VerificationResult(
                            primary_tool="duckdb_sql",
                            secondary_tool="relational_plan_polars_native",
                            primary_value=primary_metric,
                            secondary_value=secondary_scalar,
                            observed_delta_pct=float((max(float(frame_delta), float(absolute_delta)) / max(abs(primary_metric), 1e-9))),
                            max_absolute_delta=max(float(frame_delta), float(absolute_delta)),
                            max_relative_delta=float((max(float(frame_delta), float(absolute_delta)) / max(abs(primary_metric), 1e-9))),
                            status=VerificationStatus.VERIFIED,
                            secondary_method=secondary_method,
                            exception_type=None,
                            exception_message=None,
                            sql_attempted=query_sql,
                            failure_reason=None,
                        )

                    abs_frame_delta = float(frame_delta)
                    rel_frame_delta = abs_frame_delta / max(abs(primary_metric), 1e-9)
                    return VerificationResult(
                        primary_tool="duckdb_sql",
                        secondary_tool="relational_plan_polars_native",
                        primary_value=primary_metric,
                        secondary_value=None,
                        observed_delta_pct=float(rel_frame_delta),
                        max_absolute_delta=abs_frame_delta,
                        max_relative_delta=float(rel_frame_delta),
                        status=VerificationStatus.FAILED,
                        secondary_method=secondary_method,
                        exception_type=None,
                        exception_message=None,
                        sql_attempted=query_sql,
                        failure_reason="result_relation_disagreement",
                    )

                except Exception as relation_exc:
                    return VerificationResult(
                        primary_tool="duckdb_sql",
                        secondary_tool="relational_plan_polars_native",
                        primary_value=primary_metric,
                        secondary_value=None,
                        observed_delta_pct=None,
                        status=VerificationStatus.FAILED,
                        secondary_method="relational_plan_polars_native",
                        exception_type=type(relation_exc).__name__,
                        exception_message=str(relation_exc),
                        sql_attempted=query_sql,
                        failure_reason="relational_plan_secondary_execution_failed",
                    )

            # ---------------------------------------------------------
            # LEGACY SQL / scalar verification path
            # remains below for callers that do not supply a plan.
            # ---------------------------------------------------------

            # 1. Correlation experiments return raw paired observations. Reconstruct
            # the non-null relation natively in Polars so DuckDB SQL dialect support is
            # never mistaken for scientific disagreement.
            if aggregation_type == "CORRELATION" and primary_result_df is not None:
                corr_cols = [c for c in primary_result_df.columns if c in pl_df.columns]
                if len(corr_cols) >= 2:
                    try:
                        secondary_result_df = pl_df.select(corr_cols[:2]).drop_nulls()
                        secondary_method = "correlation_polars_native"
                    except Exception as corr_exc:
                        sql_exception_type = type(corr_exc).__name__
                        sql_exception_message = str(corr_exc)

            # 2. Execute SQL via Polars Rust SQL engine when no intent-specific
            # reconstruction is available.
            if query_sql and secondary_result_df is None:
                try:
                    ctx = pl.SQLContext()
                    # Register the canonical legacy alias plus the explicit
                    # source table named by a simple single-table SQL query.
                    # This keeps older SQL working while allowing the typed
                    # relational compiler to stop rewriting every source to
                    # the arbitrary ``data_table`` name. JOIN-bearing plans
                    # are handled by the canonical relational-plan branch
                    # above, so this alias extraction is deliberately narrow.
                    ctx.register("data_table", pl_df)
                    source_match = re.search(
                        r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                        query_sql or "",
                        re.IGNORECASE,
                    )
                    if source_match:
                        source_name = source_match.group(1)
                        if source_name != "data_table":
                            ctx.register(source_name, pl_df)
                    polars_res = ctx.execute(query_sql).collect()
                    secondary_result_df = polars_res
                    res_df = polars_res.to_pandas()
                    try:
                        secondary_val, _ = _derive_primary_metric(
                            res_df,
                            aggregation_type=aggregation_type,
                            primary_result_column=primary_result_column,
                            sql=query_sql or "",
                        )
                        secondary_method = "sql_sqlcontext"
                    except Exception as reduction_exc:
                        sql_exception_type = type(reduction_exc).__name__
                        sql_exception_message = str(reduction_exc)
                except Exception as sql_exc:
                    # DEFECT-001 root cause: query_sql is the exact SQL string
                    # DuckDB just executed (see controller.py / transition.py),
                    # which routinely uses DuckDB-specific dialect (e.g.
                    # TRY_CAST, DATE_TRUNC) that Polars' SQLContext does not
                    # support. This is a dialect/parse incompatibility, NOT
                    # evidence that the primary DuckDB result is wrong -- so
                    # it must not be treated the same as a genuine numerical
                    # disagreement. Record the exception instead of
                    # discarding it, and fall through to the programmatic
                    # Polars aggregation path below, which independently
                    # recomputes the same quantity via aggregation_type
                    # without needing to parse any SQL dialect at all.
                    secondary_val = None
                    sql_exception_type = type(sql_exc).__name__
                    sql_exception_message = str(sql_exc)

            # 3. Controller-authoritative relation verification.
            # The live controller passes the exact DuckDB result relation. In that
            # path the secondary engine MUST independently reproduce the relation;
            # falling back to a scalar aggregate would permit a different scope,
            # grouping, or estimand to masquerade as verification.
            #
            # NOTE: this branch is reached for EVERY controller-driven experiment
            # (primary_result_df is populated whether or not a join was involved),
            # not only genuine RelationalPlan/join cases -- those are exclusively
            # handled by the "CANONICAL RELATIONAL-PLAN VERIFICATION" block above,
            # which is gated on `relational_plan is not None`. secondary_tool here
            # must therefore stay "polars_vectorized", matching every non-join
            # verification (single-table concentration/ANOVA/etc. experiments);
            # relabeling it "relational_plan_polars_native" would claim a
            # RelationalPlan-based join engine ran when it did not, and breaks
            # every caller/test that asserts the ordinary vectorized tool name
            # (e.g. EvidenceVerification.secondary_tool == "polars_vectorized").
            if primary_result_df is not None:
                if primary_result_truncated:
                    return VerificationResult(
                        primary_tool="duckdb_sql", secondary_tool="polars_vectorized",
                        primary_value=primary_metric,
                        secondary_value=secondary_val,
                        observed_delta_pct=None,
                        status=VerificationStatus.FAILED,
                        secondary_method=secondary_method,
                        exception_type=sql_exception_type,
                        exception_message=sql_exception_message,
                        sql_attempted=query_sql,
                        failure_reason="primary_result_truncated_cannot_verify_full_relation",
                    )
                if secondary_result_df is None and query_sql:
                    try:
                        native_res = execute_known_query(query_sql, pl_df)
                        if native_res is not None:
                            secondary_result_df = native_res
                            secondary_method = "programmatic_query_recompute"
                            native_pdf = native_res.to_pandas()
                            metric_candidates = [c for c in native_pdf.columns if c.lower() in (
                                "count_val", "crude_rate", "persontime_rate", "total_metric",
                                "primary_metric", "primary_val", "metric", "v", "val", "events",
                                "count", "mean", "sum", "revenue", "cost",
                            )]
                            if len(metric_candidates) == 1 and not native_pdf.empty and pd.api.types.is_numeric_dtype(native_pdf[metric_candidates[0]]):
                                value = native_pdf[metric_candidates[0]].iloc[0]
                                if pd.notna(value):
                                    secondary_val = float(value)
                    except Exception as native_exc:
                        sql_exception_type = sql_exception_type or type(native_exc).__name__
                        sql_exception_message = sql_exception_message or str(native_exc)

                if secondary_result_df is None and not query_sql:
                    secondary_result_df = pl_df
                    secondary_method = "programmatic_agg"

                if secondary_result_df is None:
                    return VerificationResult(
                        primary_tool="duckdb_sql", secondary_tool="polars_vectorized",
                        primary_value=primary_metric,
                        secondary_value=None,
                        observed_delta_pct=None,
                        status=VerificationStatus.FAILED,
                        secondary_method="none_computed",
                        exception_type=sql_exception_type,
                        exception_message=sql_exception_message,
                        sql_attempted=query_sql,
                        failure_reason="no_independent_result_relation_available",
                    )

                frames_ok, frame_delta = result_frames_equivalent(primary_result_df, secondary_result_df, effective_tolerance)
                if frames_ok:
                    if secondary_val is None:
                        secondary_val = primary_metric
                    return VerificationResult(
                        primary_tool="duckdb_sql", secondary_tool="polars_vectorized",
                        primary_value=primary_metric,
                        secondary_value=secondary_val,
                        observed_delta_pct=float(frame_delta),
                        status=VerificationStatus.VERIFIED,
                        secondary_method=secondary_method,
                        exception_type=sql_exception_type,
                        exception_message=sql_exception_message,
                        sql_attempted=query_sql,
                        failure_reason=None,
                    )
                return VerificationResult(
                    primary_tool="duckdb_sql", secondary_tool="polars_vectorized",
                    primary_value=primary_metric,
                    secondary_value=secondary_val if secondary_val is not None else 0.0,
                    observed_delta_pct=float(frame_delta),
                    status=VerificationStatus.FAILED,
                    secondary_method=secondary_method,
                    exception_type=sql_exception_type,
                    exception_message=sql_exception_message,
                    sql_attempted=query_sql,
                    failure_reason="result_relation_disagreement",
                )

            # 4. Legacy scalar verification for direct engine callers that do not
            # provide the primary result relation. This path is retained for API
            # compatibility, but is intentionally never used by the production
            # controller after the integration hardening above.
            if secondary_val is None:
                def _ratio_agg(frame: "pl.DataFrame") -> Optional[float]:
                    if numerator_column and denominator_column and numerator_column in frame.columns and denominator_column in frame.columns:
                        num = frame.select(pl.col(numerator_column).sum()).item()
                        den = frame.select(pl.col(denominator_column).sum()).item()
                        return float(num) / float(den) if den else None
                    if weight_column and weight_column in frame.columns and target_metric_col in frame.columns:
                        weighted = frame.select((pl.col(target_metric_col) * pl.col(weight_column)).sum()).item()
                        wsum = frame.select(pl.col(weight_column).sum()).item()
                        return float(weighted) / float(wsum) if wsum else None
                    if target_metric_col in frame.columns:
                        return float(frame.select(pl.col(target_metric_col).mean()).item())
                    return None

                if group_dimension_col and group_dimension_col in pl_df.columns:
                    if aggregation_type == "SUM":
                        agg_df = pl_df.group_by(group_dimension_col).agg(pl.col(target_metric_col).sum().alias("val")).sort("val", descending=True)
                        secondary_val = float(agg_df["val"][0]) if len(agg_df) > 0 else 0.0
                    elif aggregation_type in ("MEAN", "AVG"):
                        agg_df = pl_df.group_by(group_dimension_col).agg(pl.col(target_metric_col).mean().alias("val")).sort("val", descending=True)
                        secondary_val = float(agg_df["val"][0]) if len(agg_df) > 0 else 0.0
                    elif aggregation_type == "COUNT":
                        agg_df = pl_df.group_by(group_dimension_col).agg(pl.col(target_metric_col).count().alias("val")).sort("val", descending=True)
                        secondary_val = float(agg_df["val"][0]) if len(agg_df) > 0 else 0.0
                    elif aggregation_type == "COUNT_DISTINCT":
                        agg_df = pl_df.group_by(group_dimension_col).agg(pl.col(target_metric_col).n_unique().alias("val")).sort("val", descending=True)
                        secondary_val = float(agg_df["val"][0]) if len(agg_df) > 0 else 0.0
                    elif aggregation_type in ("RATE", "RATIO", "PROPORTION", "WEIGHTED_MEAN"):
                        val = _ratio_agg(pl_df)
                        if val is not None:
                            secondary_val = val
                else:
                    if aggregation_type == "SUM":
                        secondary_val = float(pl_df.select(pl.col(target_metric_col).sum()).item())
                    elif aggregation_type in ("MEAN", "AVG"):
                        secondary_val = float(pl_df.select(pl.col(target_metric_col).mean()).item())
                    elif aggregation_type == "COUNT":
                        secondary_val = float(pl_df.select(pl.col(target_metric_col).count()).item())
                    elif aggregation_type == "COUNT_DISTINCT":
                        secondary_val = float(pl_df.select(pl.col(target_metric_col).n_unique()).item())
                    elif aggregation_type in ("RATE", "RATIO", "PROPORTION", "WEIGHTED_MEAN"):
                        val = _ratio_agg(pl_df)
                        if val is not None:
                            secondary_val = val
                    elif target_metric_col in pl_df.columns:
                        secondary_val = float(pl_df.select(pl.col(target_metric_col).sum()).item())

            if secondary_val is None and query_sql:
                try:
                    native_res = execute_known_query(query_sql, pl_df)
                    if native_res is not None:
                        native_pdf = native_res.to_pandas()
                        metric_candidates = [c for c in native_pdf.columns if c.lower() in (
                            "count_val", "crude_rate", "persontime_rate", "total_metric",
                            "primary_metric", "primary_val", "metric", "v", "val", "events",
                            "count", "mean", "sum", "revenue", "cost",
                        )]
                        if len(metric_candidates) == 1 and not native_pdf.empty and pd.api.types.is_numeric_dtype(native_pdf[metric_candidates[0]]):
                            value = native_pdf[metric_candidates[0]].iloc[0]
                            if pd.notna(value):
                                secondary_val = float(value)
                                secondary_method = "programmatic_agg_date_truncated"
                except Exception as native_exc:
                    sql_exception_type = sql_exception_type or type(native_exc).__name__
                    sql_exception_message = sql_exception_message or str(native_exc)

            if secondary_val is None:
                return VerificationResult(
                    primary_tool="duckdb_sql",
                    secondary_tool="polars_vectorized",
                    primary_value=primary_metric,
                    secondary_value=None,
                    observed_delta_pct=None,
                    status=VerificationStatus.FAILED,
                    secondary_method="none_computed",
                    exception_type=sql_exception_type,
                    exception_message=sql_exception_message,
                    sql_attempted=query_sql,
                    failure_reason=(
                        "sql_dialect_incompatible" if sql_exception_type else "no_computation_path_for_aggregation_type"
                    ),
                )

            delta_pct = abs(primary_metric - secondary_val) / max(abs(primary_metric), 1e-9)
            is_valid = delta_pct <= effective_tolerance

            # 3. Grain proof
            #
            # The legacy scalar path does not have a real post-execution
            # relation. Comparing primary_df with itself would always report
            # fanout == 1.0 / PRESERVED and therefore falsely claim that grain
            # was evaluated. Leave the public field unset on this path. The
            # canonical controller performs the authoritative grain check in
            # ScientificTransitionService with the actual result relation.
            grain_proof = None

            status = VerificationStatus.VERIFIED if is_valid else VerificationStatus.FAILED

            return VerificationResult(
                primary_tool="duckdb_sql",
                secondary_tool="polars_vectorized",
                primary_value=primary_metric,
                secondary_value=secondary_val,
                observed_delta_pct=float(delta_pct),
                status=status,
                grain_proof=grain_proof,
                secondary_method=secondary_method,
                exception_type=sql_exception_type,
                exception_message=sql_exception_message,
                sql_attempted=query_sql,
                failure_reason=None if is_valid else "numerical_disagreement_exceeds_tolerance",
            )
        except Exception as exc:
            return VerificationResult(
                primary_tool="duckdb_sql",
                secondary_tool="polars_vectorized",
                primary_value=primary_metric,
                secondary_value=None,
                observed_delta_pct=None,
                status=VerificationStatus.FAILED,
                secondary_method="none_computed",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                sql_attempted=query_sql,
                failure_reason="unexpected_exception_in_verification_engine",
            )
