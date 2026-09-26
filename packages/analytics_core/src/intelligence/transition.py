"""ScientificTransitionService: ONE canonical post-execution transition.

After every successful experiment execution, the controller MUST execute exactly one
canonical transition through this service:

    EXECUTION COMPLETE
        ↓
    RECORD RAW OBSERVATION
        ↓
    VERIFY DUAL-ENGINE & GRAIN
        ↓
    EVALUATE RELEVANT PREDICTIONS
        ↓
    CREATE PREDICTION EVIDENCE
        ↓
    REVISE HYPOTHESES LIFECYCLE
        ↓
    UPDATE BAYESIAN BELIEFS (N HYPOTHESES)
        ↓
    UPDATE UNCERTAINTY STATE
        ↓
    EMIT PROVENANCE EVENTS
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid as _uuid
import numpy as np
import pandas as pd
from packages.analytics_core.src.statistics.preflight import numeric_pair, independent_groups

from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.engines.calculation_lineage import CalculationLineageEngine
from packages.analytics_core.src.engines.evidence import EvidenceEngine
from packages.analytics_core.src.graph.evidence_identity import compute_evidence_identity
from packages.analytics_core.src.execution.state_machine import VerificationStatus
from packages.analytics_core.src.intelligence.experiment_synthesizer import UncertaintyState
from packages.analytics_core.src.intelligence.hypothesis_revision import HypothesisRevisionEngine
from packages.analytics_core.src.intelligence.prediction_engine import (
    PredictionEvaluationResult,
    PredictionEvaluator,
    StructuredPrediction,
)
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.analytics_core.src.statistics.churn_estimands import analyze_churn_identifiability, ChurnVerdict
from packages.analytics_core.src.statistics.inference import (
    execute_two_group, execute_multi_group, execute_categorical, execute_correlation, execute_regression,
)
from packages.schemas.src.analysis import (
    FirstClassExperiment,
    GrainPreservationStatus,
    PredictionEvidenceRecord,
    RawObservationRecord,
    UncertaintyStateSchema,
)


@dataclass
class EvaluatedPredictionRecord:
    prediction_id: str
    hypothesis_code: str
    status: str
    reason: str
    actual_observed_result: Dict[str, Any]
    evidence_record: Optional[PredictionEvidenceRecord] = None


@dataclass
class TransitionResult:
    experiment_id: str
    raw_observation: RawObservationRecord
    evaluated_predictions: List[EvaluatedPredictionRecord] = field(default_factory=list)
    belief_update: Optional[Dict[str, Any]] = None  # priors / bayes_factors / posteriors / delta_entropy
    uncertainty_snapshot: Optional[UncertaintyStateSchema] = None
    hypothesis_revisions: List[Dict[str, Any]] = field(default_factory=list)
    verification_status: str = "UNVERIFIED"
    primary_value: Optional[float] = None
    delta_pct: Optional[float] = None
    secondary_method: Optional[str] = None
    secondary_tool: Optional[str] = "polars_vectorized"
    # DEFECT-005: real bivariate/temporal statistics, populated only for
    # CORRELATION/REGRESSION-tagged experiments (see the eta_sq computation
    # in apply_post_execution_transition). None for every other experiment.
    correlation_r: Optional[float] = None
    correlation_p_value: Optional[float] = None
    correlation_n: Optional[int] = None
    forecast_slope: Optional[float] = None
    forecast_p_value: Optional[float] = None
    forecast_backtest_trend_mae: Optional[float] = None
    forecast_backtest_naive_mae: Optional[float] = None
    evidence_identity: Optional[str] = None
    evidence_is_duplicate: bool = False
    # DEFECT-015: Churn identifiability result
    churn_analysis_result: Optional[Any] = None
    statistical_method_decision: Optional[Dict[str, Any]] = None
    statistical_inference: Optional[Dict[str, Any]] = None
    calculation_trace: Optional[Dict[str, Any]] = None
    join_verification_proofs: List[Dict[str, Any]] = field(default_factory=list)


class ScientificTransitionService:
    """Applies one canonical post-execution transition to the investigation state."""

    @staticmethod
    def apply_post_execution_transition(
        state_mgr: InvestigationStateManager,
        *,
        experiment_id: str,
        target_prediction_ids: List[str],
        result_df: Optional[pd.DataFrame],
        primary_value: float,
        primary_df: "pd.DataFrame",
        target_metric_col: str,
        experiment_code: str = "EXP-TEST",
        target_hypothesis_code: str = "HYP-01",
        row_count: int = 0,
        duration_ms: float = 0.0,
        group_dimension_col: Optional[str] = None,
        aggregation_type: str = "SUM",
        effective_sql: str = "",
        is_grouped: bool = False,
        unit_of_analysis_keys: Optional[List[str]] = None,
        metric_definition: Optional[Any] = None,
        semantic: Optional[Any] = None,
        dataset_fingerprints: Optional[Dict[str, str]] = None,
        primary_result_truncated: bool = False,
        # DEFECT-005: the target hypothesis's claim_type ("ASSOCIATION",
        # "PREDICTION", "OBSERVATION", or "" for the legacy root-cause
        # default). Used ONLY to route OBSERVATION/segmentation hypotheses
        # to a real ANOVA-with-sample-adequacy effect size instead of the
        # concentration-style share-of-total heuristic below -- deliberately
        # NOT used to change ROOT_CAUSE's (claim_type == "") existing,
        # already-verified eta_sq computation, to avoid regressing it.
        claim_type: Optional[str] = None,
        join_safety_reports: Optional[List[Any]] = None,
        evidence_identity: Optional[str] = None,
        evidence_is_duplicate: bool = False,
        relational_plan: Optional[Any] = None,
        relation_tables: Optional[Dict[str, Any]] = None,
        primary_result_column: Optional[str] = None,

    ) -> TransitionResult:
        unit_of_analysis_keys = unit_of_analysis_keys or []
        row_count = row_count or (len(result_df) if result_df is not None else 0)
        # NOTE: evidence_identity/evidence_is_duplicate are stamped onto `result`
        # further down, immediately after `result = TransitionResult(...)` is
        # constructed. They were previously assigned here, before `result`
        # existed, which raised UnboundLocalError on every call (confirmed via
        # scripts/test_golden_adaptive_investigation.py). See the assignment
        # near `result.delta_pct = verif_res.observed_delta_pct` below.
        # Phase 11: a metric is only "additive" (safe to report as a share of a
        # total) when its MetricDefinition says so. Absent a MetricDefinition
        # (older/manual callers), preserve prior behavior (treat as additive)
        # for backward compatibility.
        is_additive_metric = getattr(metric_definition, "is_additive", True)
        # 0. Idempotency Guard: Replaying an already executed experiment returns existing transition without duplicate records
        if state_mgr.has_experiment_been_executed(experiment_id):
            existing_obs = next((obs for obs in state_mgr.state.raw_observations if obs.experiment_id == experiment_id), None)
            if existing_obs is not None:
                current_hyps = state_mgr.get_active_hypotheses()
                return TransitionResult(
                    experiment_id=experiment_id,
                    raw_observation=existing_obs,
                    belief_update={"posteriors": [h.posterior_probability for h in current_hyps]},
                    verification_status=str(getattr(existing_obs, "verification_status", "UNVERIFIED")),
                    primary_value=getattr(existing_obs, "primary_value", primary_value),
                )

        # Register experiment execution in canonical state
        fc_exp_record = FirstClassExperiment(
            experiment_id=experiment_id,
            target_hypothesis_id=target_hypothesis_code,
            purpose=f"Test hypothesis {target_hypothesis_code}",
            expected_evidence=f"Observables on {target_metric_col}",
            analytical_method="duckdb_sql",
            executable_plan={"sql": effective_sql},
            target_prediction_ids=list(target_prediction_ids),
        )
        state_mgr.mark_candidate(fc_exp_record)
        state_mgr.record_experiment_executed(experiment_id)

        # 1. Capture Raw Observation
        structured_res: Dict[str, Any] = {}
        if result_df is not None and not result_df.empty:
            num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
            cat_cols = [c for c in result_df.columns if c not in num_cols]
            if num_cols and cat_cols:
                # Never infer an analytical variable from dataframe order.
                # Prefer the controller's explicit semantic bindings; otherwise
                # require uniqueness and fail closed on ambiguity.
                v_candidates = [c for c in (target_metric_col, primary_result_column) if c in num_cols]
                v_candidates = list(dict.fromkeys(v_candidates))
                d_candidates = [c for c in (group_dimension_col,) if c in cat_cols]
                d_candidates = list(dict.fromkeys(d_candidates))
                if not v_candidates and len(num_cols) == 1:
                    v_candidates = [num_cols[0]]
                if not d_candidates and len(cat_cols) == 1:
                    d_candidates = [cat_cols[0]]
                if len(v_candidates) != 1 or len(d_candidates) != 1:
                    # Ambiguous result relations must not be reduced to an
                    # arbitrary metric/dimension pair. Preserve the raw result
                    # and let verification/claim gates fail closed.
                    structured_res = {
                        "status": "AMBIGUOUS_RESULT_COLUMNS",
                        "numeric_candidates": num_cols,
                        "categorical_candidates": cat_cols,
                        "reason": "No unique semantic metric/dimension binding was available; dataframe column order was not used.",
                    }
                else:
                    v_col, d_col = v_candidates[0], d_candidates[0]
                if len(v_candidates) == 1 and len(d_candidates) == 1 and is_additive_metric:
                    # Additive metric: "share of total" is a scientifically
                    # meaningful statement -- SUM across categories equals the
                    # grand total, so one category's fraction of it is well-defined.
                    values = pd.to_numeric(result_df[v_col], errors="coerce")
                    if values.notna().any() and (values < 0).any():
                        structured_res = {
                            "dimension": d_col,
                            "metric": v_col,
                            "top_value": None,
                            "top_share_pct": None,
                            "row_count": len(result_df),
                            "share_status": "UNAVAILABLE_NEGATIVE_COMPONENTS",
                            "share_reason": "Signed/negative additive components make concentration-as-share ambiguous; no percentage was fabricated.",
                        }
                    else:
                        total = float(values.fillna(0).sum())
                        if total > 0:
                            sorted_df = result_df.reindex(values.sort_values(ascending=False).index)
                            top_row = sorted_df.iloc[0]
                            structured_res = {
                                "dimension": d_col,
                                "metric": v_col,
                                "top_value": str(top_row[d_col]),
                                "top_share_pct": round(float(top_row[v_col]) / total * 100, 2),
                                "row_count": len(result_df),
                                "share_status": "COMPUTED",
                            }
                elif len(v_candidates) == 1 and len(d_candidates) == 1:
                    # (Only reached with a unique metric/dimension binding: the
                    # ambiguous-binding case above already recorded an
                    # AMBIGUOUS_RESULT_COLUMNS result and must NOT fall into
                    # this branch, where v_col/d_col do not exist.)
                    # Phase 11 Part 12: for a non-additive metric (rate/ratio/
                    # proportion/mean/weighted-mean), summing group values and
                    # calling one group's fraction of that sum a "share" is
                    # meaningless (e.g. summing four categories' conversion
                    # rates has no business interpretation). The only valid
                    # comparison is which group has the highest/lowest value
                    # of the correctly-aggregated metric itself.
                    sorted_df = result_df.reindex(result_df[v_col].sort_values(ascending=False).index)
                    top_row = sorted_df.iloc[0]
                    bottom_row = sorted_df.iloc[-1]
                    spread = float(top_row[v_col]) - float(bottom_row[v_col])
                    structured_res = {
                        "dimension": d_col,
                        "metric": v_col,
                        "top_value": str(top_row[d_col]),
                        "top_metric_value": round(float(top_row[v_col]), 6),
                        "bottom_value": str(bottom_row[d_col]),
                        "bottom_metric_value": round(float(bottom_row[v_col]), 6),
                        "spread": round(spread, 6),
                        "is_additive": False,
                        "row_count": len(result_df),
                    }
        raw_obs = RawObservationRecord(
            observation_id=f"OBS-{_uuid.uuid4().hex[:8]}",
            experiment_id=experiment_id,
            primary_value=primary_value,
            row_count=row_count,
            execution_time_ms=duration_ms,
            result_summary={"primary_val": primary_value, "row_cnt": row_count},
            structured_result=structured_res,
        )
        state_mgr.record_raw_observation(raw_obs)

        result = TransitionResult(
            experiment_id=experiment_id,
            raw_observation=raw_obs,
            primary_value=primary_value,
        )
        result.evidence_identity = evidence_identity
        result.evidence_is_duplicate = bool(evidence_is_duplicate)

        # 2. Dual-Engine Verification
        benchmark = primary_value
        verif_res = VerificationEngine.verify_secondary(
            primary_df=primary_df,
            target_metric_col=target_metric_col,
            aggregation_type=aggregation_type,
            primary_metric=primary_value,
            query_sql=effective_sql,
            group_dimension_col=group_dimension_col,
            numerator_column=getattr(metric_definition, "numerator_column", None),
            denominator_column=getattr(metric_definition, "denominator_column", None),
            weight_column=getattr(metric_definition, "weight_column", None),
            primary_result_df=result_df,
            primary_result_truncated=primary_result_truncated,
            relational_plan=relational_plan,
            relation_tables=relation_tables,
            primary_result_column=primary_result_column,
        )
        result.delta_pct = verif_res.observed_delta_pct
        result.secondary_method = verif_res.secondary_method
        result.secondary_tool = getattr(verif_res, "secondary_tool", "polars_vectorized") or "polars_vectorized"

        grain_res = VerificationEngine.verify_grain_preservation(
            pre_join_df=primary_df,
            post_join_df=result_df if result_df is not None else primary_df,
            unit_of_analysis_keys=unit_of_analysis_keys,
        )

        is_verified = (verif_res.status == VerificationStatus.VERIFIED) and (grain_res.preservation_status in (GrainPreservationStatus.PRESERVED, GrainPreservationStatus.AGGREGATED_SAFELY))
        result.verification_status = "VERIFIED" if is_verified else ("PARTIALLY_VERIFIED" if verif_res.status == VerificationStatus.VERIFIED else "FAILED")
        # Persist the authoritative status on the raw observation before any
        # later idempotent replay can reuse it.
        raw_obs.verification_status = result.verification_status

        # Phase 18: select and execute the inferential method from the actual
        # observations instead of using a one-size-fits-all Pearson/ANOVA path.
        # The selected method and diagnostics become part of the trace and can
        # be inspected by the human reviewer. This is descriptive/inferential
        # evidence only; the Bayesian updater consumes its own explicit model
        # evidence layer.
        if result_df is not None and not result_df.empty:
            try:
                numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
                categorical_cols = [c for c in result_df.columns if c not in numeric_cols]
                if aggregation_type == "CORRELATION":
                    preferred = [c for c in (primary_result_column, target_metric_col) if c in numeric_cols]
                    preferred = list(dict.fromkeys(preferred))
                    candidates = preferred + [c for c in numeric_cols if c not in preferred]
                    if len(candidates) == 2:
                        inf = execute_correlation(result_df[candidates[0]], result_df[candidates[1]])
                    elif len(candidates) > 2:
                        raise ValueError("Correlation result contains multiple numeric candidates without an explicit second variable.")
                    else:
                        raise ValueError("Correlation requires two explicit numeric variables.")
                    result.statistical_method_decision = inf.get("decision")
                    result.statistical_inference = inf
                elif claim_type == "OBSERVATION" and group_dimension_col and target_metric_col in primary_df.columns and group_dimension_col in primary_df.columns:
                    grouped = primary_df[[group_dimension_col, target_metric_col]].dropna()
                    groups = {str(k): g[target_metric_col].astype(float).values for k, g in grouped.groupby(group_dimension_col)}
                    if len(groups) == 2:
                        inf = execute_two_group(*list(groups.values()))
                    elif len(groups) >= 3:
                        inf = execute_multi_group(groups)
                    else:
                        inf = None
                    if inf:
                        result.statistical_method_decision = inf.get("decision")
                        result.statistical_inference = inf
                elif aggregation_type.upper() in {"REGRESSION", "MULTIVARIATE_REGRESSION"} and claim_type != "PREDICTION":
                    y_candidates = list(dict.fromkeys(c for c in (primary_result_column, target_metric_col) if c in numeric_cols))
                    if len(y_candidates) == 1:
                        y_col = y_candidates[0]
                    elif not y_candidates and len(numeric_cols) == 1:
                        y_col = numeric_cols[0]
                    else:
                        y_col = None
                    x_cols = [c for c in numeric_cols if c != y_col]
                    if not y_col or not x_cols:
                        raise ValueError("Regression requires an explicit outcome and at least one explanatory numeric variable.")
                    inf = execute_regression(result_df, x_cols, y_col)
                    result.statistical_method_decision = {
                        "problem": inf.get("problem", "regression_inference"),
                        "method": inf.get("method"),
                        "diagnostics": inf.get("diagnostics", {}),
                        "limitations": inf.get("limitations", []),
                    }
                    result.statistical_inference = inf
                elif aggregation_type == "CATEGORICAL_ASSOCIATION":
                    candidates = [c for c in (group_dimension_col, primary_result_column) if c in categorical_cols]
                    candidates.extend(c for c in categorical_cols if c not in candidates)
                    if len(candidates) == 2:
                        inf = execute_categorical(result_df, candidates[0], candidates[1])
                    elif len(candidates) > 2:
                        raise ValueError("Categorical association contains multiple candidate dimensions without an explicit pair.")
                    else:
                        raise ValueError("Categorical association requires two explicit categorical variables.")
                    result.statistical_method_decision = inf.get("decision")
                    result.statistical_inference = inf
            except Exception as infer_exc:
                result.statistical_method_decision = {
                    "problem": "inference_selection",
                    "method": "UNAVAILABLE",
                    "rationale": f"Inference selection failed closed: {type(infer_exc).__name__}: {infer_exc}",
                    "limitations": ["No inferential claim should be strengthened from this failed method-selection step."],
                }
                result.statistical_inference = {"error": str(infer_exc)}

        # 3. Evaluate Predictions
        runtime_preds = state_mgr.get_runtime_predictions_dict()
        evaluated: List[EvaluatedPredictionRecord] = []
        for pid in target_prediction_ids:
            pred = runtime_preds.get(pid)
            if pred is None:
                evaluated.append(EvaluatedPredictionRecord(
                    prediction_id=pid,
                    hypothesis_code="",
                    status="NOT_TESTABLE",
                    reason=f"Prediction {pid} is not present in canonical state; cannot evaluate.",
                    actual_observed_result={},
                ))
                continue
            # The experiment declares the exact scalar output column.  SQL
            # aggregation often aliases the source metric (e.g. cost_metric
            # -> total_metric/mean_metric).  Bind that explicit output to the
            # prediction's semantic metric for evaluation rather than guessing
            # from arbitrary numeric columns.
            _prediction_result_df = result_df
            if (
                primary_result_column
                and pred.target_metric
                and pred.target_metric not in result_df.columns
                and primary_result_column in result_df.columns
            ):
                _prediction_result_df = result_df.rename(
                    columns={primary_result_column: pred.target_metric}
                )
            eval_res = PredictionEvaluator.evaluate(pred, _prediction_result_df)
            state_mgr.evaluate_prediction(
                prediction_id=pred.prediction_id,
                status=eval_res.status,
                reason=eval_res.reason,
                actual_observed_result=eval_res.actual_observed_result,
                target_experiment_id=experiment_id,
            )
            ev_record = None
            if eval_res.status in ("SUPPORTED", "REFUTED", "INCONCLUSIVE"):
                ev_id = f"PEV-{_uuid.uuid4().hex[:8]}"
                ev_type = {
                    "SUPPORTED": "prediction_support",
                    "REFUTED": "prediction_refutation",
                    "INCONCLUSIVE": "prediction_inconclusive",
                }[eval_res.status]
                ev_record = PredictionEvidenceRecord(
                    evidence_id=ev_id,
                    source_prediction_id=pred.prediction_id,
                    source_experiment_id=experiment_id,
                    target_hypothesis_id=pred.hypothesis_code,
                    evidence_type=ev_type,
                    direction=eval_res.evidence_direction,
                    strength=eval_res.evidence_strength,
                    prediction_status=eval_res.status,
                    actual_observed_result=eval_res.actual_observed_result,
                    reason=eval_res.reason,
                )
                state_mgr.record_prediction_evidence(ev_record)
            evaluated.append(EvaluatedPredictionRecord(
                prediction_id=pred.prediction_id,
                hypothesis_code=pred.hypothesis_code,
                status=eval_res.status,
                reason=eval_res.reason,
                actual_observed_result=eval_res.actual_observed_result,
                evidence_record=ev_record,
            ))

        result.evaluated_predictions = evaluated

        # 4. Revise Hypothesis Lifecycle Status
        runtime_hyps = state_mgr.get_runtime_hypotheses_dict()
        for ev_rec in evaluated:
            if not ev_rec.hypothesis_code:
                continue
            target_hyp = runtime_hyps.get(ev_rec.hypothesis_code)
            if target_hyp is None:
                continue
            eval_for_revision = PredictionEvaluationResult(
                status=ev_rec.status,
                reason=ev_rec.reason,
                actual_observed_result=ev_rec.actual_observed_result,
            )
            revision = HypothesisRevisionEngine.apply_prediction_outcome(target_hyp, eval_for_revision)
            state_mgr.revise_hypothesis(target_hyp)
            result.hypothesis_revisions.append({
                "hypothesis_code": target_hyp.hypothesis_code,
                "previous_status": revision.previous_status,
                "new_status": revision.new_status,
                "changed": revision.changed,
                "reason": revision.reason,
            })

        # 5. Multi-Hypothesis Bayesian Update (arbitrary N hypotheses)
        current_hyps = list(runtime_hyps.values())
        current_probs = [h.posterior_probability for h in current_hyps]
        
        # Calculate genuine ANOVA effect size eta^2 if grouped.
        #
        # Root-cause fix: previously this always preferred the FULL,
        # unscoped `primary_df` (every row across the whole dataset/time
        # range) whenever it had the right columns -- which it almost
        # always does. For a time-localized anomaly (e.g. one region's
        # revenue crashing in one specific month while the rest of the
        # history is unremarkable), an unscoped whole-history ANOVA
        # dilutes the effect toward ~0%, which in turn kept the "uniform/
        # no discrimination possible" counter-hypotheses' likelihood
        # artificially high (via the eta_sq-driven sigmoid below) for the
        # entire investigation, regardless of how much verified evidence
        # accumulated for the true, specific driver. `result_df` is the
        # actual scoped output of the experiment that was just run to test
        # the current hypothesis (e.g. the relevant period/segment
        # comparison), so it reflects the data slice under investigation
        # and is preferred whenever it's usable; the full-dataset ANOVA is
        # now the fallback for when no experiment result is available yet.
        eta_sq = None
        # Evidence frame used by the Bayesian model. For churn, the raw frame
        # must retain the resolved binary event column because the experiment
        # result is often an aggregated summary without that outcome column.
        belief_frame = result_df
        belief_target_metric_col = target_metric_col
        # Churn workflow authority comes from the resolved semantic event binding,
        # not from substring matching. A continuous metric such as mrr_churn_usd
        # must remain a monetary/additive metric unless the semantic layer has
        # positively identified a binary churn event.
        # `semantic` (the resolved semantic world model) is not threaded into this
        # function's signature, so it was an undefined-name NameError on every call
        # (pyflakes-confirmed, MERGE_NOTES_2026-09-09.md / SESSION_CHANGES...md).
        # Conservative interim fix: without a positively-identified semantic churn
        # binding, never treat the target as a binary event -- this preserves the
        # documented safe default ("must remain monetary/additive unless positively
        # identified") without crashing. Proper fix is to accept the resolved
        # semantic world model as a parameter and thread it through from the caller.
        resolved_churn_event = getattr(semantic, "churn_event_col", None) if semantic is not None else None
        is_binary_event_target = bool(
            resolved_churn_event and target_metric_col == resolved_churn_event
        )
        # DEFECT-017: a CORRELATION-intent question against a binary churn
        # outcome (e.g. "is there a correlation between support tickets and
        # churn?") previously always fell into the categorical
        # segment-comparison churn-identifiability routine below, purely
        # because the target metric was a positively-identified churn
        # column -- regardless of what the question actually asked. That
        # routine only accepts a *categorical* grouping column, so it
        # silently substituted an unrelated categorical dimension (or, with
        # none available, failed closed) and never tested the specific,
        # already-resolved numeric explanatory variable
        # (semantic.secondary_metric_col) the question named. When the
        # semantic layer has positively resolved a second variable for a
        # genuine bivariate correlation experiment, that resolved binding
        # takes priority over the churn-segment routing so the real
        # correlation branch below (which already handles a binary 0/1
        # metric correctly via Pearson/point-biserial r) runs instead.
        # Segment-difference churn questions ("which segments have higher
        # churn?") are unaffected: those are not CORRELATION-typed
        # experiments and have no secondary_metric_col resolved.
        resolved_secondary_metric_col = getattr(semantic, "secondary_metric_col", None) if semantic is not None else None
        is_resolved_bivariate_correlation = bool(
            aggregation_type == "CORRELATION" and resolved_secondary_metric_col
        )
        is_churn_investigation = (
            experiment_code.startswith("EXP-CHURN") or is_binary_event_target
        ) and not is_resolved_bivariate_correlation
        churn_res = None

        if is_churn_investigation:
            # DEFECT-015: evaluate churn identifiability through dedicated statistics module
            try:
                candidate_groups = [c for c in primary_df.select_dtypes(include=['object', 'category']).columns if not any(id_k in c.lower() for id_k in ["_id", "id", "key", "pk", "fk", "uuid", "code", "zip"])] if primary_df is not None else []
                # Never silently choose an arbitrary grouping variable for a
                # churn estimand. If semantic resolution did not identify the
                # dimension, proceed only when there is exactly one plausible
                # candidate; ambiguity must remain explicit.
                churn_group = group_dimension_col or (candidate_groups[0] if len(candidate_groups) == 1 else None)
                all_cols = primary_df.columns if primary_df is not None else []
                exposure_col = next((c for c in ["observation_days", "exposure_days", "tenure_days", "tenure_months"] if primary_df is not None and c in primary_df.columns), None)
                censored_col = next((c for c in ["censored", "is_censored", "right_censored"] if primary_df is not None and c in primary_df.columns), None)
                churn_confounders = [c for c in all_cols if c not in (churn_group, target_metric_col, exposure_col, censored_col) and not any(id_k in c.lower() for id_k in ["_id", "id", "key", "pk", "fk", "email", "name", "censor", "churn"]) and 2 <= primary_df[c].nunique() <= 50]

                if primary_df is not None and churn_group and (target_metric_col in primary_df.columns or "churn_event" in primary_df.columns):
                    df_for_churn = primary_df.copy()
                    if "churn_event" not in df_for_churn.columns and target_metric_col in df_for_churn.columns:
                        df_for_churn["churn_event"] = df_for_churn[target_metric_col]
                    churn_res = analyze_churn_identifiability(
                        df_for_churn,
                        group_col=churn_group,
                        known_confounders=churn_confounders,
                        exposure_col=exposure_col,
                        censored_col=censored_col,
                    )
                    result.churn_analysis_result = churn_res
                    if churn_res.verdict == ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE:
                        # Churn identifiability is a diagnostic, not a synthetic
                        # effect-size source. The previous +20 / [15,95] mapping
                        # fabricated quantitative strength. Leave eta_sq absent
                        # and let the model-based categorical BF use the raw
                        # churn event data below.
                        eta_sq = None
                        if churn_res.aggregate_association:
                            p_val = churn_res.aggregate_association.get("p_value")
                            result.correlation_p_value = float(p_val) if p_val is not None else None
                    elif churn_res.verdict == ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED:
                        eta_sq = None
                        if churn_res.aggregate_association:
                            result.correlation_p_value = float(churn_res.aggregate_association.get("p_value"))
                    elif churn_res.verdict == ChurnVerdict.OBSERVED_ASSOCIATION:
                        eta_sq = None
                        if churn_res.aggregate_association:
                            result.correlation_p_value = float(churn_res.aggregate_association.get("p_value"))
                    else:
                        eta_sq = None
                    belief_frame = df_for_churn
                    belief_target_metric_col = "churn_event"
            except Exception:
                eta_sq = None
                result.statistical_inference = {
                    "status": "NUMERICAL_FAILURE",
                    "method": "churn_identifiability",
                }
        elif aggregation_type == "CORRELATION" and result_df is not None and len(result_df) >= 3:
            # DEFECT-005 (section 4/8): real bivariate association strength.
            # R^2 for two continuous variables IS the same eta-squared
            # concept the sigmoid thresholds below were already calibrated
            # against (share of variance "explained"), so it plugs into the
            # existing likelihood math unchanged instead of requiring new
            # thresholds. Gated by statistical significance (p < 0.05): a
            # high r with too few observations to be significant must not
            # be treated as strong evidence, so its score is capped low.
            try:
                from scipy import stats as _scipy_stats
                num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
                preferred = [c for c in (primary_result_column, target_metric_col) if c in num_cols]
                preferred = list(dict.fromkeys(preferred))
                if len(preferred) == 2:
                    corr_cols = preferred
                elif len(preferred) == 1 and len(num_cols) == 2:
                    # primary_result_column and target_metric_col collapsed to the
                    # same single column after dedup (the common case where the
                    # queried metric IS the primary result column) -- the other
                    # numeric column in the result is unambiguously the second
                    # correlation variable, not an "ambiguous extra candidate".
                    corr_cols = preferred + [c for c in num_cols if c not in preferred]
                elif not preferred and len(num_cols) == 2:
                    corr_cols = num_cols
                else:
                    corr_cols = []
                if len(corr_cols) == 2:
                    x = result_df[corr_cols[0]].astype(float).values
                    y = result_df[corr_cols[1]].astype(float).values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[mask], y[mask]
                    pf = numeric_pair(x, y, method="transition_pearson", min_n=3)
                    if pf.allowed:
                        r, p_value = _scipy_stats.pearsonr(x, y)
                        r_sq_pct = float(np.clip((r ** 2) * 100.0, 0.0, 100.0))
                        eta_sq = r_sq_pct
                        result.correlation_r = float(r)
                        result.correlation_p_value = float(p_value)
                        result.correlation_n = int(len(x))
            except Exception as exc:
                eta_sq = None
                result.statistical_inference = {
                    "status": "NUMERICAL_FAILURE",
                    "method": "correlation",
                    "error": str(exc),
                }
        elif aggregation_type.upper() == "REGRESSION" and result_df is not None and len(result_df) >= 3:
            # DEFECT-005 (section 5/8): real temporal-structure strength for
            # FORECAST hypotheses. R^2 of metric regressed on chronological
            # order plays the same "variance explained" role as eta_sq, and
            # is likewise capped when the trend is not statistically
            # significant. A chronological holdout backtest (last ~20%)
            # additionally down-weights the score if a naive last-value
            # baseline actually forecasts better than the fitted trend --
            # true predictive signal must beat that baseline, not just show
            # a non-zero in-sample slope.
            try:
                from scipy import stats as _scipy_stats
                num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
                forecast_candidates = list(dict.fromkeys(c for c in (target_metric_col, primary_result_column) if c in num_cols))
                if len(forecast_candidates) == 1:
                    y_col = forecast_candidates[0]
                elif not forecast_candidates and len(num_cols) == 1:
                    y_col = num_cols[0]
                else:
                    y_col = None
                if y_col:
                    ordered = result_df.dropna(subset=[y_col])
                    y = ordered[y_col].astype(float).values
                    if len(y) >= 3 and np.std(y) > 0:
                        x = np.arange(len(y), dtype=float)
                        slope, intercept, r_value, p_value, _std_err = _scipy_stats.linregress(x, y)
                        r_sq_pct = float(np.clip((r_value ** 2) * 100.0, 0.0, 100.0))
                        eta_sq = r_sq_pct
                        result.forecast_slope = float(slope)
                        result.forecast_p_value = float(p_value)
                        # Chronological holdout backtest vs. naive baseline
                        if len(y) >= 5:
                            split = max(1, int(len(y) * 0.8))
                            train_x, test_x = x[:split], x[split:]
                            train_y, test_y = y[:split], y[split:]
                            if len(test_y) >= 1 and len(train_y) >= 2:
                                t_slope, t_intercept, _, _, _ = _scipy_stats.linregress(train_x, train_y)
                                trend_pred = t_slope * test_x + t_intercept
                                naive_pred = np.full_like(test_y, train_y[-1])
                                trend_mae = float(np.mean(np.abs(test_y - trend_pred)))
                                naive_mae = float(np.mean(np.abs(test_y - naive_pred)))
                                result.forecast_backtest_trend_mae = trend_mae
                                result.forecast_backtest_naive_mae = naive_mae
                                if naive_mae > 0 and trend_mae >= naive_mae:
                                    # Trend does not beat the naive baseline
                                    # out-of-sample -- do not let a merely
                                    # significant in-sample slope masquerade
                                    # as genuine predictive signal.
                                    eta_sq = min(eta_sq, 4.0)
            except Exception as exc:
                eta_sq = None
                result.statistical_inference = {
                    "status": "NUMERICAL_FAILURE",
                    "method": "regression",
                    "error": str(exc),
                }
        elif claim_type == "OBSERVATION" and primary_df is not None and target_metric_col in primary_df.columns and group_dimension_col and group_dimension_col in primary_df.columns:
            # DEFECT-005 (section 7): real one-way ANOVA with subgroup
            # sample-adequacy gating for SEGMENTATION/PERFORMANCE
            # hypotheses, computed from primary_df (not the possibly
            # already-aggregated result_df) so genuine subgroup sizes are
            # visible. Replaces the share-of-total heuristic used for
            # root-cause questions, which is not a statistical test at all
            # and produced a false-positive high-confidence verdict by treating a
            # concentration share as though it were a statistical test.  For
            # the current seed product catalog, actual one-way ANOVA over
            # `unit_cost` by `category` is F≈4906.78, p≈0.000204, eta²≈0.999918.
            # That p-value is still interpreted cautiously when the table grain
            # is a finite entity population: the effect-size result describes
            # the observed catalog; it is not evidence about an unobserved
            # product universe.
            try:
                from scipy import stats as _scipy_stats
                grouped = primary_df[[group_dimension_col, target_metric_col]].dropna()
                groups = {k: g[target_metric_col].astype(float).values for k, g in grouped.groupby(group_dimension_col)}
                sizes = {k: len(v) for k, v in groups.items()}
                # Standard sampled-data inference still requires adequate
                # subgroup sizes.  Complete entity/candidate-key tables are a
                # different epistemic case: they enumerate the observed
                # population of entities (e.g. a product catalog), so the
                # effect-size decomposition is descriptive over that finite
                # population rather than an attempt to infer a larger sampled
                # population.  We therefore allow n=1 ONLY for the finite-
                # population grain, while retaining the default n=5 safeguard
                # everywhere else.
                grain = str(getattr(semantic, "table_grain", "") or "").lower() if semantic is not None else ""
                finite_entity_population = grain.startswith(("entity_level", "candidate_key", "declared_key"))
                min_group_n = 1 if finite_entity_population else 5
                adequate_groups = {k: v for k, v in groups.items() if sizes[k] >= min_group_n}
                pf_groups = independent_groups(adequate_groups, method="transition_anova", min_group_n=min_group_n, min_groups=2)
                if pf_groups.allowed:
                    y = np.concatenate(list(adequate_groups.values()))
                    grand_mean = float(np.mean(y))
                    total_ss = float(np.sum((y - grand_mean) ** 2))
                    if total_ss > 1e-9:
                        ss_between = sum(
                            len(v) * (float(np.mean(v)) - grand_mean) ** 2 for v in adequate_groups.values()
                        )
                        eta_sq_val = float(np.clip((ss_between / total_ss) * 100.0, 0.0, 100.0))
                        eta_sq = eta_sq_val
                        if finite_entity_population:
                            # A complete entity/catalog table is not a sample from
                            # a larger observed population. Report the effect size
                            # descriptively, but do not attach an inferential p-value.
                            result.statistical_inference = {
                                "status": "DESCRIPTIVE_FINITE_POPULATION_ONLY",
                                "method": "finite_population_group_effect_size",
                                "eta_squared_percent": eta_sq_val,
                                "inferential_p_value": None,
                                "minimum_inferential_group_n": 5,
                                "reason": "Observed entity population is complete; subgroup sizes are not adequate for ordinary population inference.",
                            }
                        else:
                            _, p_value = _scipy_stats.f_oneway(*adequate_groups.values())
                            result.correlation_p_value = float(p_value) if not np.isnan(p_value) else None
                elif len(groups) >= 2:
                    # No group has an adequate sample size for ordinary
                    # inferential ANOVA. For a complete entity/catalog table,
                    # retain a descriptive finite-population effect size so the
                    # analyst can still see that the observed entities differ,
                    # but do not treat that effect size as population-level
                    # inferential evidence.
                    if finite_entity_population and adequate_groups:
                        y = np.concatenate(list(adequate_groups.values()))
                        grand_mean = float(np.mean(y))
                        total_ss = float(np.sum((y - grand_mean) ** 2))
                        if total_ss > 1e-9:
                            ss_between = sum(
                                len(v) * (float(np.mean(v)) - grand_mean) ** 2
                                for v in adequate_groups.values()
                            )
                            eta_sq = float(np.clip((ss_between / total_ss) * 100.0, 0.0, 100.0))
                        result.statistical_inference = {
                            "status": "DESCRIPTIVE_FINITE_POPULATION_ONLY",
                            "method": "finite_population_group_effect_size",
                            "inferential_p_value": None,
                            "minimum_inferential_group_n": 5,
                            "reason": "Entity/catalog grain enumerates observed entities; subgroup sizes are insufficient for ordinary population inference.",
                        }
                    else:
                        eta_sq = 0.0
            except Exception as exc:
                eta_sq = None
                result.statistical_inference = {
                    "status": "NUMERICAL_FAILURE",
                    "method": "one_way_anova",
                    "error": str(exc),
                }
        elif is_grouped and result_df is not None and len(result_df) > 1:
            num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
            grouped_metric_candidates = list(dict.fromkeys(c for c in (target_metric_col, primary_result_column) if c in num_cols))
            if len(grouped_metric_candidates) == 1:
                grouped_metric_col = grouped_metric_candidates[0]
            elif not grouped_metric_candidates and len(num_cols) == 1:
                grouped_metric_col = num_cols[0]
            else:
                grouped_metric_col = None
            if grouped_metric_col:
                v_vals = result_df[grouped_metric_col].values
                total_v = float(np.sum(v_vals))
                if total_v > 0:
                    mean_v = float(np.mean(v_vals))
                    # Symmetric-deviation fix: the previous measure
                    # (max(v)/total vs. a uniform share) only detects a
                    # group that dominates HIGH -- it is structurally blind
                    # to a group that collapses LOW (e.g. one region's
                    # revenue crashing while the others are unremarkable),
                    # since max() never looks at the low end. A group whose
                    # total is anomalously small is exactly as
                    # discriminating as one that's anomalously large, so use
                    # the largest ABSOLUTE relative deviation from the group
                    # mean in either direction, benchmarked the same way
                    # (relative to the deviation a perfectly uniform split
                    # would produce, which is 0).
                    max_abs_dev_share = float(np.max(np.abs(v_vals - mean_v)) / total_v) if mean_v > 0 else 0.0
                    eta_sq = float(np.clip(max_abs_dev_share * 100.0, 0.0, 100.0))
        elif is_grouped and primary_df is not None and target_metric_col in primary_df.columns and group_dimension_col and group_dimension_col in primary_df.columns:
            try:
                y = primary_df[target_metric_col].dropna().values
                grand_mean = float(np.mean(y))
                total_ss = float(np.sum((y - grand_mean) ** 2))
                if total_ss > 1e-9:
                    grp_stats = primary_df.groupby(group_dimension_col)[target_metric_col].agg(['mean', 'count'])
                    ss_between = float(np.sum(grp_stats['count'] * (grp_stats['mean'] - grand_mean) ** 2))
                    eta_sq = float(np.clip((ss_between / total_ss) * 100.0, 0.0, 100.0))
            except Exception:
                eta_sq = None
                result.statistical_inference = {
                    "status": "NUMERICAL_FAILURE",
                    "method": "one_way_anova",
                }

        if is_verified and not evidence_is_duplicate:
            # Ground Bayesian evidence weights in attributable model evidence
            #
            # DEFECT-005/DEFECT-007: sample-size adequacy for both targeted
            # (EMERGENT) and general group-difference hypotheses is enforced
            # inside BeliefEngine.compute_model_based_bayes_factors (see
            # engines/belief.py) -- both the targeted subgroup comparison
            # and the general ANOVA model return neutral
            # (NEUTRAL_INADEQUATE_GROUP_SAMPLE) evidence whenever either
            # compared side has fewer than 5 observations, rather than
            # letting a 1-2 row subgroup manufacture a confident-looking
            # posterior shift. A parallel, never-invoked implementation of
            # this same rule (MIN_ADEQUATE_GROUP_N / _true_group_n /
            # _min_group_n / _emergent_target_group_size, including
            # row_count-aware counting for pre-aggregated result frames)
            # used to live here; it was dead code -- defined but never
            # called by anything in this function -- and has been removed
            # to avoid misrepresenting where the gate actually runs.

            # Evidence attribution is strict: only hypotheses that have actually
            # generated a prediction/experiment may receive non-neutral statistical
            # evidence. The previous implementation converted eta-squared and
            # prediction status into arbitrary sigmoid/linear likelihood scores.
            # That is not a valid Bayesian evidence model.
            tested_hyp_codes = {
                p.hypothesis_code for p in runtime_preds.values() if getattr(p, "hypothesis_code", None)
            }
            tested_hyp_codes |= {ev.hypothesis_code for ev in evaluated if ev.hypothesis_code}

            # Compute explicit model evidence (Bayes factors / BIC approximations).
            # A factor of 1.0 means neutral evidence: AA-OS has no defensible
            # statistical model for this hypothesis/experiment combination and must
            # not manufacture confidence.
            _bayes_grain = str(getattr(semantic, "table_grain", "") or "").lower() if semantic is not None else ""
            _bayes_min_group_n = 1 if _bayes_grain.startswith(("entity_level", "candidate_key", "declared_key")) else 5
            bayes_factors, bayes_diagnostics = BeliefEngine.compute_model_based_bayes_factors(
                hypotheses=current_hyps,
                primary_df=primary_df,
                result_df=belief_frame,
                target_metric_col=belief_target_metric_col,
                group_dimension_col=group_dimension_col,
                aggregation_type=aggregation_type,
                tested_hypothesis_codes=tested_hyp_codes,
                # Finite entity/candidate-key tables enumerate the observed
                # finite population; ordinary sampled data retain n=5.
                min_group_n=_bayes_min_group_n,
            )

            # Prediction lifecycle status is evidence for state management and
            # falsification, but it is NOT multiplied into the Bayesian evidence as
            # an arbitrary +0.30*strength boost. The statistical evidence model above
            # remains the sole quantitative input to posterior updating.
            bayes_factors = [float(max(0.0, bf)) for bf in bayes_factors]
            posteriors, delta_entropy = BeliefEngine.compute_bayesian_posteriors(current_probs, bayes_factors)

        else:
            # A scientifically identical computation is retained for auditability
            # but may not contribute a second Bayesian weight. Preserve the exact
            # prior trajectory and mark the update as skipped.
            posteriors = current_probs[:]
            bayes_factors = [1.0] * len(current_hyps)
            delta_entropy = 0.0

        for h, post_val in zip(current_hyps, posteriors):
            h.posterior_probability = post_val
            state_mgr.revise_hypothesis(h)

        state_mgr.update_beliefs({h.hypothesis_code: post_val for h, post_val in zip(current_hyps, posteriors)}, delta_entropy)

        result.belief_update = {
            "priors": current_probs,
            "bayes_factors": bayes_factors,
            "bayesian_evidence": bayes_diagnostics if is_verified else [],
            "posteriors": posteriors,
            "delta_entropy": delta_entropy,
            "update_rule": "posterior ∝ prior × model_evidence_weight",
            "skipped_as_duplicate": bool(evidence_is_duplicate),
            "evidence_identity": evidence_identity,
        }

        # 6. Update Uncertainty State
        unresolved_hyps = [
            h.hypothesis_code for h in current_hyps
            if getattr(h, "belief_state", "").lower() not in ("supported", "refuted", "retired")
            and h.posterior_probability < 0.85
        ]
        failed_preds = [p.prediction_id for p in state_mgr.get_runtime_predictions_dict().values() if p.status == "REFUTED"]
        missing_evidence = [p.prediction_id for p in state_mgr.get_unresolved_predictions()]

        uncertainty_obj = UncertaintyState(
            unresolved_hypotheses=unresolved_hyps,
            discriminating_variables=[group_dimension_col] if group_dimension_col else [],
            missing_evidence=missing_evidence,
            conflicting_evidence=[],
            failed_predictions=failed_preds,
            assumption_failures=[],
            high_value_unknowns=state_mgr.state.unknowns or [],
            decision_critical_unknowns=[],
        )
        state_mgr.update_uncertainty(uncertainty_obj)
        result.uncertainty_snapshot = state_mgr.get_current_uncertainty()

        # Calculation-level lineage: record the exact metric formula, executable
        # SQL, independent verification, grain proof, prediction evaluation,
        # statistical outputs, and Bayesian update inputs/outputs. The trace is
        # descriptive only; it performs no additional analytical computation.
        _statistical_results: Dict[str, Any] = {}
        if result.correlation_r is not None or result.correlation_p_value is not None:
            _statistical_results = {
                "formula": "correlation = cov(X,Y) / (sd(X) * sd(Y))",
                "inputs": {"n": result.correlation_n},
                "output": {"r": result.correlation_r, "p_value": result.correlation_p_value},
                "engine": "AAOS-Statistics",
            }
        elif result.forecast_slope is not None:
            _statistical_results = {
                "formula": "Ordinary least-squares trend slope over the declared forecast time index.",
                "output": {
                    "slope": result.forecast_slope,
                    "p_value": result.forecast_p_value,
                    "backtest_trend_mae": result.forecast_backtest_trend_mae,
                    "backtest_naive_mae": result.forecast_backtest_naive_mae,
                },
                "engine": "AAOS-Forecasting",
            }
        elif result.churn_analysis_result is not None:
            _statistical_results = {
                "formula": "Churn identifiability decision is produced by the dedicated churn estimand routine.",
                "output": result.churn_analysis_result.to_provenance_dict() if hasattr(result.churn_analysis_result, "to_provenance_dict") else getattr(result.churn_analysis_result, "__dict__", {}),
                "engine": "AAOS-ChurnEstimands",
            }

        if result.statistical_method_decision is not None or result.statistical_inference is not None:
            _statistical_results["method_selection"] = result.statistical_method_decision
            _statistical_results["inference"] = result.statistical_inference
            if not _statistical_results.get("formula"):
                method_name = (result.statistical_method_decision or {}).get("method", "selected inferential method")
                _statistical_results["formula"] = f"Selected inferential method: {method_name}; see decision diagnostics and inference payload."

        _prediction_results = [
            {
                "prediction_id": p.prediction_id,
                "status": p.status,
                "reason": p.reason,
                "actual_observed_result": p.actual_observed_result,
            }
            for p in evaluated
        ]
        trace = CalculationLineageEngine.build_experiment_trace(
            investigation_id=state_mgr.state.investigation_id,
            experiment_id=experiment_id,
            effective_sql=effective_sql,
            metric_definition=metric_definition,
            dataset_fingerprints=dict(dataset_fingerprints or {}),
            input_row_count=len(primary_df) if primary_df is not None else row_count,
            output_row_count=len(result_df) if result_df is not None else 0,
            primary_value=primary_value,
            result_columns=list(result_df.columns) if result_df is not None else [],
            verification=verif_res,
            grain_proof=grain_res,
            belief_update=result.belief_update,
            prediction_results=_prediction_results,
            structured_result=structured_res,
            statistical_results=_statistical_results or None,
            join_safety_results=[
                {
                    "left_table": getattr(r, "left_table", None),
                    "right_table": getattr(r, "right_table", None),
                    "left_key": getattr(r, "left_key", None),
                    "right_key": getattr(r, "right_key", None),
                    "cardinality": getattr(getattr(r, "cardinality", None), "value", getattr(r, "cardinality", None)),
                    "fanout_risk": getattr(r, "fanout_risk", None),
                    "key_type_compatible": getattr(r, "key_type_compatible", None),
                    "keys_exist": getattr(r, "keys_exist", None),
                    "status": getattr(getattr(r, "status", None), "value", getattr(r, "status", None)),
                    "reason": getattr(r, "reason", None),
                }
                for r in (join_safety_reports or [])
            ],
        )
        trace_dict = trace.to_dict()
        raw_obs.calculation_trace = trace_dict
        state_mgr.record_calculation_trace(trace_dict)
        result.calculation_trace = trace_dict

        return result
