"""Calculation lineage for AA-OS.

Every material analytical output gets a deterministic, machine-readable trace:
source snapshot -> scope -> formula -> execution -> verification -> derived
statistics/belief updates.  This layer records *how* a value was obtained; it
does not perform analytical work itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class CalculationTraceStep:
    step_id: str
    kind: str
    title: str
    formula: Optional[str] = None
    executable_expression: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    source_columns: List[str] = field(default_factory=list)
    source_row_scope: Optional[str] = None
    execution_engine: Optional[str] = None
    parent_step_ids: List[str] = field(default_factory=list)
    verification_status: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _safe({
            "step_id": self.step_id,
            "kind": self.kind,
            "title": self.title,
            "formula": self.formula,
            "executable_expression": self.executable_expression,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "output": self.output,
            "source_columns": self.source_columns,
            "source_row_scope": self.source_row_scope,
            "execution_engine": self.execution_engine,
            "parent_step_ids": self.parent_step_ids,
            "verification_status": self.verification_status,
            "notes": self.notes,
        })


@dataclass(frozen=True)
class CalculationTrace:
    trace_id: str
    investigation_id: str
    experiment_id: str
    dataset_fingerprints: Dict[str, str]
    input_row_count: int
    output_row_count: int
    steps: List[CalculationTraceStep]
    final_output: Dict[str, Any]
    trace_version: str = "1.0"
    canonical_hash: str = ""
    reproducibility_statement: str = "Reproduce by executing the recorded executable expressions against the recorded dataset snapshot fingerprints."

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "trace_version": self.trace_version,
            "investigation_id": self.investigation_id,
            "experiment_id": self.experiment_id,
            "dataset_fingerprints": self.dataset_fingerprints,
            "input_row_count": self.input_row_count,
            "output_row_count": self.output_row_count,
            "steps": [s.to_dict() for s in self.steps],
            "final_output": _safe(self.final_output),
            "reproducibility_statement": self.reproducibility_statement,
        }
        canonical_hash = self.canonical_hash or hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        payload["canonical_hash"] = canonical_hash
        return payload


class CalculationLineageEngine:
    """Build immutable calculation traces from canonical execution facts."""

    @staticmethod
    def build_experiment_trace(
        *,
        investigation_id: str,
        experiment_id: str,
        effective_sql: str,
        metric_definition: Optional[Any],
        dataset_fingerprints: Dict[str, str],
        input_row_count: int,
        output_row_count: int,
        primary_value: Optional[float],
        result_columns: List[str],
        verification: Optional[Any] = None,
        grain_proof: Optional[Any] = None,
        belief_update: Optional[Dict[str, Any]] = None,
        prediction_results: Optional[List[Dict[str, Any]]] = None,
        structured_result: Optional[Dict[str, Any]] = None,
        statistical_results: Optional[Dict[str, Any]] = None,
        join_safety_results: Optional[List[Dict[str, Any]]] = None,
    ) -> CalculationTrace:
        trace_id = f"TRACE-{experiment_id}"
        steps: List[CalculationTraceStep] = []
        prev: List[str] = []

        metric_columns = list(getattr(metric_definition, "source_columns", []) or [])
        metric_formula = None
        if metric_definition is not None:
            try:
                from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver
                metric_formula = MetricSemanticsResolver.sql_aggregation_expression(metric_definition, "trace_metric")
            except Exception:
                metric_formula = None

        steps.append(CalculationTraceStep(
            step_id=f"{trace_id}:01",
            kind="SOURCE_SCOPE",
            title="Immutable source snapshot and analytical scope",
            formula="Source rows are the dataset snapshot addressed by the recorded executable SQL.",
            executable_expression=effective_sql,
            inputs={"dataset_fingerprints": dataset_fingerprints, "input_row_count": input_row_count},
            output={"result_columns": result_columns},
            source_columns=metric_columns,
            source_row_scope=effective_sql,
            execution_engine="AAOS-DatasetSnapshot",
            notes="The SQL and dataset fingerprint together identify the exact analytical input scope; raw source files are not copied into the trace.",
        ))
        prev = [steps[-1].step_id]

        steps.append(CalculationTraceStep(
            step_id=f"{trace_id}:02",
            kind="METRIC_FORMULA",
            title="Metric/estimand calculation",
            formula=(getattr(metric_definition, "sql_formula", None) or metric_formula or "Derived directly from the executable analytical expression."),
            executable_expression=metric_formula or effective_sql,
            inputs={"metric": getattr(metric_definition, "name", None), "aggregation": getattr(getattr(metric_definition, "aggregation_type", None), "value", None)},
            parameters={
                "numerator_column": getattr(metric_definition, "numerator_column", None),
                "denominator_column": getattr(metric_definition, "denominator_column", None),
                "weight_column": getattr(metric_definition, "weight_column", None),
                "semantic_resolution_status": getattr(metric_definition, "semantic_resolution_status", None),
            },
            output={"metric_value": primary_value},
            source_columns=metric_columns,
            source_row_scope=effective_sql,
            execution_engine="DuckDB",
            parent_step_ids=prev,
            notes=(getattr(metric_definition, "rationale", None) or "Metric semantics were resolved by the deterministic metric-semantics engine."),
        ))
        prev = [steps[-1].step_id]

        if join_safety_results:
            steps.append(CalculationTraceStep(
                step_id=f"{trace_id}:03J",
                kind="JOIN_SAFETY",
                title="Pre-execution relational join safety proof",
                formula="Join is executable only when each declared hop has compatible keys and positively established cardinality; many-to-many is blocked unless explicitly authorized.",
                executable_expression="assess_declared_join_hops(datasets, join_hops)",
                inputs={"join_hops": join_safety_results},
                output={
                    "all_safe": all(r.get("status") == "SAFE" for r in join_safety_results),
                    "hop_count": len(join_safety_results),
                },
                execution_engine="AAOS-RelationalJoinSafety",
                parent_step_ids=prev,
                verification_status=("VERIFIED" if all(r.get("status") == "SAFE" for r in join_safety_results) else "FAILED"),
                notes="This proof is evaluated against the actual local dataset snapshots before JOIN SQL execution.",
            ))
            prev = [steps[-1].step_id]

        if verification is not None:
            primary = getattr(verification, "primary_metric_val", primary_value)
            secondary = getattr(verification, "secondary_metric_val", None)
            delta = getattr(verification, "observed_delta_pct", None)
            tol = getattr(verification, "tolerance_threshold", None)
            formula = "absolute_delta = abs(primary - secondary) / max(abs(primary), epsilon)"
            steps.append(CalculationTraceStep(
                step_id=f"{trace_id}:03",
                kind="INDEPENDENT_VERIFICATION",
                title="Independent recomputation and agreement check",
                formula=formula,
                executable_expression="secondary_engine_recomputation",
                inputs={"primary_value": primary, "secondary_value": secondary},
                parameters={"tolerance_threshold": tol},
                output={"secondary_value": secondary, "observed_delta_pct": delta, "passed": getattr(verification, "is_mathematically_identical", None)},
                source_columns=metric_columns,
                execution_engine="DuckDB + Polars",
                parent_step_ids=prev,
                verification_status=getattr(getattr(verification, "status", None), "value", getattr(verification, "status", None)),
                notes="The secondary result is independently recomputed; it is not copied from the primary result.",
            ))
            prev = [steps[-1].step_id]

        if grain_proof is not None:
            steps.append(CalculationTraceStep(
                step_id=f"{trace_id}:04",
                kind="GRAIN_VERIFICATION",
                title="Grain and fanout verification",
                formula="fanout_factor = post_join_row_count / max(pre_join_row_count, 1)",
                inputs={
                    "pre_join_row_count": getattr(grain_proof, "pre_join_row_count", None),
                    "post_join_row_count": getattr(grain_proof, "post_join_row_count", None),
                    "distinct_key_count": getattr(grain_proof, "distinct_key_count", None),
                },
                output={
                    "fanout_factor": getattr(grain_proof, "fanout_factor", None),
                    "preservation_status": getattr(getattr(grain_proof, "preservation_status", None), "value", getattr(grain_proof, "preservation_status", None)),
                },
                source_columns=list(getattr(grain_proof, "requested_grain", "").split(",")) if getattr(grain_proof, "requested_grain", None) else [],
                execution_engine="AAOS-GrainVerifier",
                parent_step_ids=prev,
                verification_status=getattr(getattr(grain_proof, "preservation_status", None), "value", getattr(grain_proof, "preservation_status", None)),
            ))
            prev = [steps[-1].step_id]

        if statistical_results:
            steps.append(CalculationTraceStep(
                step_id=f"{trace_id}:05",
                kind="STATISTICAL_RESULT",
                title="Statistical/inferential calculation",
                formula=statistical_results.get("formula") or "Deterministic statistical routine recorded in the result payload.",
                executable_expression=statistical_results.get("executable_expression"),
                inputs=statistical_results.get("inputs", {}),
                parameters=statistical_results.get("parameters", {}),
                output=statistical_results.get("output", {}),
                execution_engine=statistical_results.get("engine", "AAOS-Statistics"),
                parent_step_ids=prev,
                notes=statistical_results.get("notes"),
            ))
            prev = [steps[-1].step_id]

        if prediction_results:
            steps.append(CalculationTraceStep(
                step_id=f"{trace_id}:06",
                kind="PREDICTION_EVALUATION",
                title="Falsifiable prediction evaluation",
                formula="Observed result is compared with the prediction's declared expected direction/range/threshold.",
                inputs={"predictions": prediction_results},
                output={"prediction_count": len(prediction_results)},
                execution_engine="AAOS-PredictionEvaluator",
                parent_step_ids=prev,
            ))
            prev = [steps[-1].step_id]

        if belief_update:
            evidence_weights = belief_update.get("bayes_factors") or belief_update.get("likelihoods", [])
            steps.append(CalculationTraceStep(
                step_id=f"{trace_id}:07",
                kind="BELIEF_UPDATE",
                title="Bayesian hypothesis belief update",
                formula="posterior_i = prior_i × model_evidence_weight_i / Σ_j(prior_j × model_evidence_weight_j)",
                executable_expression="BeliefEngine.compute_bayesian_posteriors(priors, bayes_factors)",
                inputs={
                    "priors": belief_update.get("priors", []),
                    "bayes_factors": evidence_weights,
                    "bayesian_evidence": belief_update.get("bayesian_evidence", []),
                },
                output={
                    "posteriors": belief_update.get("posteriors", []),
                    "delta_entropy": belief_update.get("delta_entropy"),
                },
                execution_engine="AAOS-BeliefEngine",
                parent_step_ids=prev,
                notes=(
                    "Quantitative evidence weights are Bayes factors/model-evidence weights. "
                    "A factor of 1.0 is neutral and means no defensible statistical evidence "
                    "model was available for that hypothesis in this experiment."
                ),
            ))
            prev = [steps[-1].step_id]

        final_output = {
            "primary_value": primary_value,
            "output_row_count": output_row_count,
            "structured_result": structured_result or {},
            "statistical_results": statistical_results or {},
        }
        trace = CalculationTrace(
            trace_id=trace_id,
            investigation_id=investigation_id,
            experiment_id=experiment_id,
            dataset_fingerprints=dict(dataset_fingerprints or {}),
            input_row_count=int(input_row_count or 0),
            output_row_count=int(output_row_count or 0),
            steps=steps,
            final_output=final_output,
        )
        # force computation of the canonical hash into the serialized payload
        trace.to_dict()
        return trace
