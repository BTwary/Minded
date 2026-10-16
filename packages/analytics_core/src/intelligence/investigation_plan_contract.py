"""Structural validation and capability audit for AA-OS investigation plans.

This module is deliberately conservative. An experiment method label is a plan
proposal, not proof that the runtime has a callable executor wired to it. The
validator checks plan integrity and records the level of implementation evidence
available in the repository; the runtime's canonical method registry remains the
execution authority.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTRACT_VERSION = "1.0"

# method_id -> (support level, likely implementation module(s), note)
# MODULE_PRESENT is evidence of related code in the repository only. It is not a
# runtime binding and must never grant execution authority by itself.
_METHOD_CATALOG: dict[str, tuple[str, tuple[str, ...], str]] = {
    "churn_rate_association": ("MODULE_PRESENT", ("packages/analytics_core/src/statistics/churn_estimands.py",), "Verify callable binding in the canonical method registry."),
    "targeted_stratified_churn_check": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/statistics/churn_estimands.py", "packages/analytics_core/src/statistics/engine.py"), "Requires a declared adjustment set and orchestration."),
    "correlation_or_regression": ("MODULE_PRESENT", ("packages/analytics_core/src/statistics/engine.py",), "Select method using data types and assumptions."),
    "targeted_association_sensitivity": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/semantic/missingness_sensitivity.py", "packages/analytics_core/src/statistics/engine.py"), "Sensitivity scope must be evidence-driven."),
    "stratified_or_partial_association": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/statistics/engine.py",), "Requires supported covariates and a valid adjustment strategy."),
    "causal_identification_gate": ("MODULE_PRESENT", ("packages/analytics_core/src/causal/identifiability_gate.py",), "Passing the gate is required before effect estimation."),
    "aipw_or_g_computation": ("MODULE_PRESENT", ("packages/analytics_core/src/causal/effect_estimation.py",), "Estimator choice depends on the approved estimand and design."),
    "time_series_readiness_check": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/forecasting/engine.py",), "Validate temporal index and history before fitting."),
    "forecast_backtest": ("MODULE_PRESENT", ("packages/analytics_core/src/forecasting/engine.py",), "Require time-ordered folds and a baseline."),
    "prediction_readiness_and_leakage_audit": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/intelligence/prediction_engine.py", "packages/analytics_core/src/statistics/preflight.py"), "Prediction timestamp and target horizon must be explicit."),
    "heldout_baseline_model_comparison": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/ml/engine.py",), "Runtime must bind split policy, model, and metric."),
    "probability_calibration_and_sliced_evaluation": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/statistics/probability_calibration.py", "packages/analytics_core/src/statistics/calibration.py"), "Calibration and subgroup support must be evaluated."),
    "type_aware_group_comparison": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/statistics/engine.py",), "Method depends on outcome type and sampling design."),
    "effect_size_and_uncertainty": ("MODULE_PRESENT", ("packages/analytics_core/src/statistics/uncertainty.py",), "Select interval/test based on data and design."),
    "comparison_sensitivity_check": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/statistics/multiple_comparisons.py", "packages/analytics_core/src/engines/multiverse_engine.py"), "Only test materially relevant sensitivity dimensions."),
    "change_baseline_and_temporal_profile": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/engines/temporal.py", "packages/analytics_core/src/statistics/engine.py"), "Windows, denominator, and seasonality must be explicit."),
    "contribution_decomposition": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/intelligence/universal_analyst.py", "packages/analytics_core/src/sql/engine.py"), "Contributions must reconcile to the total change."),
    "alternative_explanation_and_sensitivity_checks": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/intelligence/multiverse_engine.py", "packages/analytics_core/src/causal/sensitivity.py"), "Alternatives must be tied to measurable evidence."),
    "automatic_clustering": ("MODULE_PRESENT", ("packages/analytics_core/src/intelligence/segmentation_engine.py",), "Feature scaling, stability and interpretation are required."),
    "mathematical_reconciliation": ("MODULE_PRESENT", ("packages/analytics_core/src/intelligence/reconciliation_engine.py",), "Formula and tolerance must be explicit."),
    "data_quality_profile": ("MODULE_PRESENT", ("packages/analytics_core/src/profiling/profiler.py", "packages/analytics_core/src/profiling/data_quality_gate.py"), "Profile findings must gate downstream methods when material."),
    "governance_audit": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/security/governance.py", "packages/analytics_core/src/governance/claim_gate.py"), "A data scan cannot certify legal compliance."),
    "decision_context_gate": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/engines/decision_utility.py",), "Objective, constraints, costs and risk tolerance must be supplied."),
    "decision_impact_analysis": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/intelligence/prescriptive_optimizer.py", "packages/analytics_core/src/scenarios/engine.py"), "Predicted impacts must not be presented as measured causal effects."),
    "descriptive_query": ("MODULE_PRESENT", ("packages/analytics_core/src/sql/engine.py", "packages/analytics_core/src/relational/compiler.py"), "The canonical SQL/relational contract must authorize execution."),
    "aggregate_invariant_and_scope_check": ("ORCHESTRATION_REQUIRED", ("packages/analytics_core/src/engines/verification.py", "packages/analytics_core/src/engines/calculation_lineage.py"), "Independent calculation must use equivalent scope and grain."),
    "schema_and_data_profile": ("MODULE_PRESENT", ("packages/analytics_core/src/profiling/profiler.py", "packages/analytics_core/src/semantic/world_model.py"), "Inferred keys and joins remain hypotheses until checked."),
    "evidence_grounded_question_discovery": ("MODULE_PRESENT", ("packages/analytics_core/src/intelligence/dataset_question_discovery.py",), "Candidates must be grounded in available fields."),
}


def _repo_root() -> Path:
    # .../packages/analytics_core/src/intelligence/investigation_plan_contract.py
    return Path(__file__).resolve().parents[4]


def audit_method_capability(method_id: str, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Return transparent implementation evidence for a planned method.

    ``execution_authorized`` is always false here: only runtime policy/method
    registration may authorize executing an operation.
    """
    item = _METHOD_CATALOG.get(str(method_id or ""))
    if item is None:
        return {
            "method_id": method_id,
            "support_level": "PLAN_ONLY",
            "implementation_modules": [],
            "modules_present": [],
            "modules_missing": [],
            "runtime_binding_required": True,
            "execution_authorized": False,
            "note": "No explicit implementation mapping was found in the plan capability catalog.",
        }
    level, relative_paths, note = item
    root = repo_root or _repo_root()
    present = [p for p in relative_paths if (root / p).is_file()]
    missing = [p for p in relative_paths if not (root / p).is_file()]
    if not present:
        level = "PLAN_ONLY"
    elif missing and level == "MODULE_PRESENT":
        level = "PARTIAL_MODULE_EVIDENCE"
    return {
        "method_id": method_id,
        "support_level": level,
        "implementation_modules": list(relative_paths),
        "modules_present": present,
        "modules_missing": missing,
        "runtime_binding_required": True,
        "execution_authorized": False,
        "note": note,
    }


def _topological_order(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> tuple[list[str], list[str]]:
    node_set = set(nodes)
    incoming: dict[str, int] = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    issues: list[str] = []
    for source, target in edges:
        if source not in node_set:
            issues.append(f"dependency references unknown prerequisite {source!r}")
            continue
        if target not in node_set:
            issues.append(f"dependency references unknown dependent {target!r}")
            continue
        if source == target:
            issues.append(f"self-dependency detected for {source!r}")
            continue
        outgoing[source].append(target)
        incoming[target] += 1
    queue = deque(node for node in nodes if incoming[node] == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in outgoing[node]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if len(order) != len(nodes):
        issues.append("dependency graph contains a cycle")
    return order, issues


def audit_investigation_plan(
    *,
    task: str,
    experiments: Sequence[Any],
    stages: Sequence[Any],
    unresolved_questions: Sequence[str] = (),
    required_evidence: Sequence[Any] = (),
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Audit experiment/stage DAGs and compute conservative plan readiness."""
    issues: list[str] = []
    warnings: list[str] = []
    experiment_ids = [str(getattr(e, "code", "")) for e in experiments]
    if any(not item for item in experiment_ids):
        issues.append("experiment code must be non-empty")
    if len(experiment_ids) != len(set(experiment_ids)):
        issues.append("experiment codes must be unique")

    experiment_edges: list[tuple[str, str]] = []
    for exp in experiments:
        code = str(getattr(exp, "code", ""))
        prereqs = list(getattr(exp, "prerequisites", []) or [])
        if len([str(item) for item in prereqs]) != len(set(str(item) for item in prereqs)):
            issues.append(f"experiment {code!r} repeats a prerequisite")
        for prereq in prereqs:
            experiment_edges.append((str(prereq), code))
            if str(prereq) not in set(experiment_ids):
                issues.append(f"experiment {code!r} requires missing prerequisite {prereq!r}")
            elif str(prereq) == code:
                issues.append(f"experiment {code!r} depends on itself")
        # Prerequisites should be earlier in the declared plan. Topological order
        # is also checked, but this catches confusing serialized display order.
        order_by_code = {key: index for index, key in enumerate(experiment_ids)}
        for prereq in prereqs:
            if str(prereq) in order_by_code and order_by_code[str(prereq)] >= order_by_code.get(code, -1):
                warnings.append(f"experiment {code!r} depends on {prereq!r}, which appears later in the declared list")

    experiment_order, experiment_graph_issues = _topological_order(experiment_ids, experiment_edges)
    issues.extend(experiment_graph_issues)

    stage_ids = [str(getattr(stage, "code", "")) for stage in stages]
    if any(not item for item in stage_ids):
        issues.append("stage code must be non-empty")
    if len(stage_ids) != len(set(stage_ids)):
        issues.append("stage codes must be unique")
    stage_edges: list[tuple[str, str]] = []
    for stage in stages:
        code = str(getattr(stage, "code", ""))
        for prereq in list(getattr(stage, "prerequisites", []) or []):
            stage_edges.append((str(prereq), code))
            if str(prereq) not in set(stage_ids):
                issues.append(f"stage {code!r} requires missing prerequisite {prereq!r}")
    stage_order, stage_graph_issues = _topological_order(stage_ids, stage_edges)
    issues.extend(stage_graph_issues)

    capabilities = [audit_method_capability(str(getattr(e, "method", "")), repo_root=repo_root) for e in experiments]
    if not experiments:
        warnings.append("plan contains no experiment steps; verify that this task is intentionally non-experimental")
    if not required_evidence:
        warnings.append("plan contains no explicit evidence requirements")

    # The planner never grants execution authorization. Readiness refers only to
    # structural integrity and missing user-visible semantic requirements.
    if issues:
        readiness = "INVALID_PLAN"
    elif unresolved_questions:
        readiness = "NEEDS_CLARIFICATION"
    elif any(cap.get("support_level") == "PLAN_ONLY" for cap in capabilities):
        readiness = "STRUCTURALLY_VALID_WITH_UNMAPPED_METHODS"
    else:
        readiness = "STRUCTURALLY_VALID_REQUIRES_RUNTIME_AUTHORIZATION"

    return {
        "contract_version": CONTRACT_VERSION,
        "task": task,
        "readiness": readiness,
        "execution_authorized": False,
        "runtime_authorization_required": True,
        "issues": list(dict.fromkeys(issues)),
        "warnings": list(dict.fromkeys(warnings)),
        "experiment_graph": {
            "nodes": experiment_ids,
            "edges": [{"from": source, "to": target} for source, target in experiment_edges],
            "topological_order": experiment_order,
        },
        "stage_graph": {
            "nodes": stage_ids,
            "edges": [{"from": source, "to": target} for source, target in stage_edges],
            "topological_order": stage_order,
        },
        "capability_review": capabilities,
        "required_evidence_count": len(required_evidence),
    }
