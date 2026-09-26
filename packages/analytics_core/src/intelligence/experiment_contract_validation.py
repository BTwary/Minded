"""validate_experiment_against_contract() -- the v20-A hard execution gate.

No CandidateExperiment may reach an executor (EIGOptimizer selection ->
DuckDBExecutionProvider / ScientificTransitionService) without first passing
this validator against the canonical analytical contract for the current
investigation round. See MindEd_AAOS_v20-A sections 1, 9, 12-15.

This module intentionally does not execute anything itself; it is a pure
compatibility check that returns a structured, diagnosable result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from packages.analytics_core.src.intelligence.analytical_identity import (
    FinalAnalyticalContract, experiment_outcome, experiment_predictors,
)
from packages.schemas.src.semantic_binding import SemanticBindingSet
from packages.schemas.src.semantic_role import ExperimentRole, SemanticRole
from packages.analytics_core.src.engines.method_selection import (
    METHOD_REGISTRY, MethodCapability, MethodFamily,
)


# Role -> which OTHER experiment roles are compatible with a DIFFERENT
# method_code than the contract's selected_method. PRIMARY must match
# exactly; the other three may use a distinct, but still contract-relevant,
# method.
_ROLES_ALLOWING_METHOD_DIVERGENCE = frozenset({
    ExperimentRole.SUPPORTING, ExperimentRole.ADVERSARIAL, ExperimentRole.VERIFICATION,
})

# Coarse, explicit, method-family-level estimand compatibility for
# SUPPORTING/ADVERSARIAL/VERIFICATION experiments that legitimately use a
# different method family than the contract's PRIMARY method. This is a
# deliberately conservative allow-list, not a permissive default -- an
# unlisted (contract_family, experiment_family) pair is INCOMPATIBLE.
_SUPPORTING_FAMILY_COMPATIBILITY: Dict[MethodFamily, Set[MethodFamily]] = {
    MethodFamily.DIAGNOSTIC_BATTERY: {MethodFamily.DIAGNOSTIC_BATTERY, MethodFamily.CORRELATION},
    MethodFamily.CORRELATION: {MethodFamily.CORRELATION, MethodFamily.DIAGNOSTIC_BATTERY},
    MethodFamily.FORECAST: {MethodFamily.FORECAST, MethodFamily.DIAGNOSTIC_BATTERY},
    MethodFamily.CHURN: {MethodFamily.CHURN, MethodFamily.DIAGNOSTIC_BATTERY},
}

# VERIFICATION experiments are independent recomputation and are always
# expected to use the SAME method family as the contract (a different
# estimator would not verify the same claim).
_VERIFICATION_REQUIRES_SAME_FAMILY = True

# Problem classes for which a canonical OUTCOME/TARGET binding is a
# structural requirement -- descriptive/structural problem classes
# (SEGMENTATION, RECONCILIATION, DESCRIPTIVE, GENERAL_EXPLORATION) are
# deliberately excluded. These are the compiler task strings actually
# persisted as InvestigationContract.problem_class (see
# CANONICAL_TASK_AUTHORITY in engines/method_selection.py) -- "SURVIVAL_CHURN"
# is a decision-time ProblemClass specialization, not a compiler task string,
# so it is intentionally not listed here; churn questions compile to
# PREDICTION/DIAGNOSTIC/COMPARISON and are covered via those.
_PROBLEM_CLASSES_REQUIRING_OUTCOME = frozenset({
    "ASSOCIATION", "COMPARISON", "FORECAST", "PREDICTION", "DIAGNOSTIC", "CAUSAL",
})


@dataclass(frozen=True)
class AnalyticalContractSnapshot:
    """The minimal, explicit view of the canonical contract that an
    executable experiment must be checked against (v20-A section 1/9).
    """
    contract_id: str
    contract_version: int
    problem_class: str
    method_family: Optional[MethodFamily]
    selected_method: Optional[str]
    estimand: str
    semantic_bindings: SemanticBindingSet
    population_scope: str = "entire_dataset"
    grain: Optional[str] = None
    time_scope: Optional[Dict[str, Any]] = None
    # v20-C4.2.3: the FINAL analytical contract.  When present it is authoritative:
    # the snapshot's own problem_class / selected_method must agree with it, and
    # every experiment is additionally checked against its identity.  Optional so
    # pre-C4.2.3 callers (and hand-built test snapshots) keep working unchanged.
    final_contract: Optional[FinalAnalyticalContract] = None

    def outcome_columns(self) -> List[str]:
        cols = [b.column for b in self.semantic_bindings.get_by_role(SemanticRole.OUTCOME)]
        cols += [b.column for b in self.semantic_bindings.get_by_role(SemanticRole.TARGET)]
        return cols

    def explanatory_columns(self) -> List[str]:
        return [b.column for b in self.semantic_bindings.get_by_role(SemanticRole.EXPLANATORY_VARIABLE)]

    def predictor_compatible_columns(self) -> List[str]:
        """Columns whose bound role legitimately makes them usable as a
        predictor -- not just EXPLANATORY_VARIABLE. Per the role vocabulary's
        own LEGITIMATE_MULTIROLE stance (semantic_role.py: "a comparison's
        grouping dimension IS its explanatory variable"), a
        GROUPING_DIMENSION or TREATMENT binding is just as legitimate a
        predictor as an EXPLANATORY_VARIABLE one -- e.g. EXP-CHURN-CRUDE's
        stratifying "segment" column is bound GROUPING_DIMENSION, not
        EXPLANATORY_VARIABLE, and must not be flagged as an undeclared
        predictor. TIME_VARIABLE is included for the same reason: a
        forecast experiment's predictor is legitimately its bound time
        column (e.g. EXP-FORECAST-TREND regressing on order_date).
        """
        cols: List[str] = []
        for role in (
            SemanticRole.EXPLANATORY_VARIABLE, SemanticRole.GROUPING_DIMENSION,
            SemanticRole.TREATMENT, SemanticRole.TIME_VARIABLE,
        ):
            cols += [b.column for b in self.semantic_bindings.get_by_role(role)]
        # de-duplicate, preserve order
        seen: Set[str] = set()
        out = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out


@dataclass
class ExperimentContractValidation:
    compatible: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checked_fields": dict(self.checked_fields),
        }


def _experiment_role(experiment: Any) -> ExperimentRole:
    raw = getattr(experiment, "experiment_role", None) or ExperimentRole.UNASSIGNED.value
    try:
        return ExperimentRole(raw)
    except ValueError:
        return ExperimentRole.UNASSIGNED


def _experiment_outcome(experiment: Any) -> Optional[str]:
    return experiment_outcome(experiment)


def _experiment_predictors(experiment: Any) -> List[str]:
    return experiment_predictors(experiment)


def validate_experiment_against_contract(
    experiment: Any,
    contract: AnalyticalContractSnapshot,
) -> ExperimentContractValidation:
    """Validate one CandidateExperiment against the canonical analytical
    contract for this investigation round. Never returns a bare False --
    every rejection carries specific, structured diagnostics.
    """
    errors: List[str] = []
    warnings: List[str] = []
    checked: Dict[str, Any] = {}

    role = _experiment_role(experiment)
    checked["experiment_role"] = role.value

    if role == ExperimentRole.UNASSIGNED:
        errors.append("EXPERIMENT_ROLE_UNASSIGNED: experiment must be tagged PRIMARY, "
                      "SUPPORTING, ADVERSARIAL, or VERIFICATION before execution")

    # -- Semantic-binding sanity: reject if the contract's OWN bindings are
    # internally conflicting for any column this experiment touches. -------
    conflicts = contract.semantic_bindings.validate()
    if conflicts:
        touched = set(getattr(experiment, "metrics", None) or []) | set(getattr(experiment, "dimensions", None) or [])
        outcome = _experiment_outcome(experiment)
        if outcome:
            touched.add(outcome)
        touched |= set(_experiment_predictors(experiment))
        relevant = [c for c in conflicts if c.column in touched]
        if relevant:
            for c in relevant:
                errors.append(f"SEMANTIC_BINDING_CONFLICT: {c.reason}")
        checked["semantic_binding_conflicts"] = [c.to_dict() for c in conflicts]

    # -- Problem class ------------------------------------------------------
    exp_problem_class = getattr(experiment, "problem_class", None)
    checked["problem_class"] = {"contract": contract.problem_class, "experiment": exp_problem_class}
    if exp_problem_class and exp_problem_class != contract.problem_class and role == ExperimentRole.PRIMARY:
        errors.append(
            f"PROBLEM_CLASS_MISMATCH: contract={contract.problem_class!r} "
            f"experiment={exp_problem_class!r}"
        )

    # -- Method family / method code -----------------------------------------
    exp_method_code = getattr(experiment, "method_code", None)
    checked["method_code"] = {"contract": contract.selected_method, "experiment": exp_method_code}
    exp_capability: Optional[MethodCapability] = METHOD_REGISTRY.get(exp_method_code) if exp_method_code else None

    if exp_method_code and exp_capability is None:
        errors.append(f"UNKNOWN_METHOD_CODE: {exp_method_code!r} is not in the method registry")
    elif contract.selected_method and exp_method_code:
        if role == ExperimentRole.PRIMARY or role == ExperimentRole.UNASSIGNED:
            if exp_method_code != contract.selected_method:
                errors.append(
                    "METHOD_CODE_MISMATCH: PRIMARY experiment must use the contract's "
                    f"selected method ({contract.selected_method!r}), got {exp_method_code!r}"
                )
        elif role in _ROLES_ALLOWING_METHOD_DIVERGENCE and exp_capability is not None:
            exp_family = exp_capability.executor_id and _family_for_capability(exp_capability)
            contract_family = contract.method_family
            if role == ExperimentRole.VERIFICATION and _VERIFICATION_REQUIRES_SAME_FAMILY:
                if contract_family is not None and exp_family != contract_family:
                    errors.append(
                        "ESTIMAND_MISMATCH: VERIFICATION experiment must recompute with the "
                        f"same method family as the contract ({contract_family}), got {exp_family}"
                    )
            elif contract_family is not None and exp_family is not None:
                allowed = _SUPPORTING_FAMILY_COMPATIBILITY.get(contract_family, {contract_family})
                if exp_family not in allowed:
                    errors.append(
                        f"ESTIMAND_MISMATCH: {role.value} experiment's method family {exp_family} "
                        f"is not an approved complement to the contract's {contract_family}"
                    )
            if contract.problem_class not in exp_capability.supported_problem_classes and role != ExperimentRole.ADVERSARIAL:
                warnings.append(
                    f"{role.value}_METHOD_PROBLEM_CLASS_LOOSE: {exp_method_code!r} does not "
                    f"declare support for {contract.problem_class!r}"
                )

    # -- Outcome / predictor bindings -----------------------------------------
    contract_outcomes = contract.outcome_columns()
    exp_outcome = _experiment_outcome(experiment)
    checked["outcome"] = {"contract": contract_outcomes, "experiment": exp_outcome}
    if not contract_outcomes and not exp_outcome and contract.problem_class in _PROBLEM_CLASSES_REQUIRING_OUTCOME:
        errors.append(
            f"OUTCOME_MISSING: no OUTCOME/TARGET binding declared for problem class "
            f"{contract.problem_class!r}, and the experiment declares no outcome either"
        )
    elif contract_outcomes and exp_outcome and exp_outcome not in contract_outcomes:
        errors.append(
            f"OUTCOME_MISMATCH: expected one of {contract_outcomes}, received {exp_outcome!r}"
        )

    exp_predictors = _experiment_predictors(experiment)
    checked["predictors"] = exp_predictors

    # -- Self-correlation / outcome-predictor collision (section 15) --------
    agg_type = str(getattr(experiment, "aggregation_type", "") or "").upper()
    exp_metrics = [m for m in (getattr(experiment, "metrics", None) or []) if m]
    if agg_type == "CORRELATION":
        distinct_metrics = {m for m in exp_metrics}
        if len(exp_metrics) >= 2 and len(distinct_metrics) < 2:
            errors.append(
                "OUTCOME_PREDICTOR_COLLISION: correlation experiment references the same "
                f"column ({exp_metrics[0]!r}) as both sides of the pairing"
            )
        elif exp_outcome and all(m == exp_outcome for m in exp_metrics if m):
            errors.append(
                f"OUTCOME_PREDICTOR_COLLISION: no distinct predictor found alongside outcome {exp_outcome!r}"
            )

    contract_predictor_compatible = contract.predictor_compatible_columns()
    if contract_predictor_compatible and exp_predictors:
        unknown_predictors = [p for p in exp_predictors if p not in contract_predictor_compatible]
        # Only flag predictors that are not merely unresolved compatibility
        # noise (e.g. a grouping dimension double-counted as a "metric").
        if unknown_predictors and role in (ExperimentRole.PRIMARY, ExperimentRole.UNASSIGNED):
            # v20-B1: a PRIMARY (or not-yet-classified) experiment is the
            # contract's main claim -- an analytical variable it uses that
            # the contract never declared as EXPLANATORY_VARIABLE/
            # GROUPING_DIMENSION/TREATMENT is a semantic mismatch, not
            # merely noteworthy. A SUPPORTING/ADVERSARIAL/VERIFICATION
            # experiment may still legitimately probe a different variable
            # (see the unbound-column and role-compatibility checks below,
            # which apply to those too).
            errors.append(
                f"PREDICTOR_NOT_IN_CONTRACT: {unknown_predictors} not among contract "
                f"predictor-compatible bindings {contract_predictor_compatible}"
            )
    if contract.problem_class in {"ASSOCIATION", "COMPARISON"} and not exp_predictors and role in (
        ExperimentRole.PRIMARY, ExperimentRole.UNASSIGNED,
    ) and agg_type != "":
        # Missing predictor for an association/comparison-style experiment.
        if not exp_metrics or len(set(exp_metrics)) < 2:
            errors.append("PREDICTOR_MISSING: no explanatory variable distinguishable from the outcome")

    # -- Population / grain / time scope (best-effort; experiment-level
    # values are optional and, when absent, are treated as inheriting the
    # contract's scope rather than blocked). --------------------------------
    exp_population = getattr(experiment, "population", None)
    checked["population"] = {"contract": contract.population_scope, "experiment": exp_population}
    if exp_population and contract.population_scope and exp_population != contract.population_scope:
        errors.append(
            f"POPULATION_MISMATCH: contract scope={contract.population_scope!r}, "
            f"experiment scope={exp_population!r}"
        )

    exp_grain = getattr(experiment, "grain", None)
    checked["grain"] = {"contract": contract.grain, "experiment": exp_grain}
    if exp_grain and contract.grain and exp_grain != contract.grain:
        errors.append(f"GRAIN_MISMATCH: contract grain={contract.grain!r}, experiment grain={exp_grain!r}")

    exp_time_scope = getattr(experiment, "time_scope", None)
    checked["time_scope"] = {"contract": contract.time_scope, "experiment": exp_time_scope}
    if exp_time_scope and contract.time_scope and exp_time_scope != contract.time_scope:
        errors.append(
            f"TIME_SCOPE_MISMATCH: contract time_scope={contract.time_scope!r}, "
            f"experiment time_scope={exp_time_scope!r}"
        )

    # -- Every executable field must reference a resolved/executable binding,
    # AND (v20-B1) that binding must actually exist, and must play a role
    # compatible with how this experiment uses the column. Previously an
    # entirely unbound column (no SemanticBinding at all) produced
    # `matches == []`, which was silently treated as "nothing to check" --
    # letting a numerically valid query reference an analytically unvetted
    # column. That gap is now closed for PRIMARY/UNASSIGNED experiments
    # (the contract's main claim); SUPPORTING/ADVERSARIAL/VERIFICATION
    # experiments may still legitimately probe a column the canonical
    # binding set hasn't registered (e.g. a confounding dimension surfaced
    # by adversarial challenge) and are only warned, not blocked.
    all_touched_cols = set(exp_metrics) | set(getattr(experiment, "dimensions", None) or [])
    if exp_outcome:
        all_touched_cols.add(exp_outcome)
    _outcome_compatible_roles = {SemanticRole.OUTCOME, SemanticRole.TARGET}
    unexecutable = []
    unbound = []
    role_incompatible = []
    for col in all_touched_cols:
        matches = contract.semantic_bindings.get_by_column(col)
        if not matches:
            unbound.append(col)
            continue
        if not any(b.is_executable() for b in matches):
            unexecutable.append(col)
            continue
        col_roles = {b.role for b in matches}
        if col == exp_outcome:
            if not (col_roles & _outcome_compatible_roles):
                role_incompatible.append(
                    f"{col} (outcome use, bound as {sorted(r.value for r in col_roles)})"
                )
        elif col in exp_predictors and col_roles <= _outcome_compatible_roles:
            role_incompatible.append(
                f"{col} (predictor use, bound only as {sorted(r.value for r in col_roles)})"
            )
    if unexecutable:
        errors.append(
            f"UNRESOLVED_BINDING: column(s) {unexecutable} do not have an executable "
            "(RESOLVED/USER_SPECIFIED/sufficiently-confident INFERRED) semantic binding"
        )
    if unbound:
        message = (
            f"UNBOUND_COLUMN: column(s) {unbound} are referenced by this experiment but have "
            "no semantic binding at all in the canonical SemanticBindingSet"
        )
        if role in (ExperimentRole.PRIMARY, ExperimentRole.UNASSIGNED):
            errors.append(message)
        else:
            warnings.append(message)
    if role_incompatible:
        message = f"ROLE_INCOMPATIBLE: {role_incompatible}"
        if role in (ExperimentRole.PRIMARY, ExperimentRole.UNASSIGNED):
            errors.append(message)
        else:
            warnings.append(message)

    if contract.final_contract is not None:
        _validate_against_final_contract(experiment, role, contract, errors, warnings, checked)

    compatible = len(errors) == 0
    return ExperimentContractValidation(
        compatible=compatible, errors=errors, warnings=warnings, checked_fields=checked,
    )


_VALID_ROLES = frozenset({
    ExperimentRole.PRIMARY, ExperimentRole.SUPPORTING,
    ExperimentRole.ADVERSARIAL, ExperimentRole.VERIFICATION,
})


def _validate_against_final_contract(
    experiment: Any,
    role: ExperimentRole,
    contract: AnalyticalContractSnapshot,
    errors: List[str],
    warnings: List[str],
    checked: Dict[str, Any],
) -> None:
    """v20-C4.2.3: identity-level validation against the FINAL analytical contract.

    Fails closed: a PRIMARY experiment whose identity cannot be established is
    rejected, never silently repaired or re-attributed to another predictor.
    """
    final = contract.final_contract
    assert final is not None
    checked["final_contract"] = {"analytical_identity": final.analytical_identity}

    # The snapshot is a per-round view; it must not disagree with the final decision.
    if contract.problem_class != final.canonical_task:
        errors.append(
            f"FINAL_CONTRACT_PROBLEM_CLASS_DIVERGENCE: snapshot={contract.problem_class!r} "
            f"final={final.canonical_task!r}"
        )
    if contract.selected_method != final.selected_method_code:
        errors.append(
            f"FINAL_CONTRACT_METHOD_DIVERGENCE: snapshot={contract.selected_method!r} "
            f"final={final.selected_method_code!r}"
        )

    if role not in _VALID_ROLES:
        errors.append(f"EXPERIMENT_ROLE_INVALID: {role.value!r} is not a valid executable role")

    exp_identity = getattr(experiment, "analytical_identity", "") or ""
    hyp_identity = getattr(experiment, "hypothesis_analytical_identity", "") or ""
    known = set(final.pair_identities().values()) | {final.analytical_identity}
    checked["analytical_identity"] = {
        "experiment": exp_identity, "hypothesis": hyp_identity,
        "contract": final.analytical_identity,
    }

    # A non-PRIMARY experiment may omit identity, but may never claim a foreign one.
    if exp_identity and exp_identity not in known:
        errors.append(
            "ANALYTICAL_IDENTITY_NOT_IN_CONTRACT: experiment identity matches no analytical "
            "pair of the final contract"
        )

    if role != ExperimentRole.PRIMARY:
        return

    # ---- PRIMARY: the experiment IS the contract's analytical claim -----------
    exp_outcome = _experiment_outcome(experiment)
    if final.target_column and exp_outcome and exp_outcome != final.target_column:
        errors.append(
            f"TARGET_MISMATCH: contract target={final.target_column!r} experiment target={exp_outcome!r}"
        )

    exp_method_code = getattr(experiment, "method_code", None)
    if final.selected_method_code and exp_method_code and exp_method_code != final.selected_method_code:
        errors.append(
            f"METHOD_MISMATCH: contract={final.selected_method_code!r} experiment={exp_method_code!r}"
        )
    if exp_method_code and exp_method_code == final.selected_method_code:
        cap = METHOD_REGISTRY.get(exp_method_code)
        if cap is not None and cap.verification_method != final.verification_regime:
            errors.append(
                "VERIFICATION_REGIME_DRIFT: registry verification method "
                f"{cap.verification_method!r} != contract regime {final.verification_regime!r}"
            )

    if not exp_identity:
        errors.append(
            "ANALYTICAL_IDENTITY_MISSING: a PRIMARY experiment must carry the deterministic "
            "analytical identity of the contract claim it tests"
        )
        return

    if final.predictor_columns:
        preds = [p for p in _experiment_predictors(experiment) if p != exp_outcome]
        contract_preds = set(final.predictor_columns)
        agg_type = str(getattr(experiment, "aggregation_type", "") or "").upper()
        if agg_type == "MULTIVARIATE_REGRESSION" and len(preds) > 1:
            if set(preds) != contract_preds:
                errors.append(
                    f"PREDICTOR_MISMATCH: joint experiment predictors {preds!r} do not exactly match "
                    f"contract predictors {list(final.predictor_columns)!r}"
                )
            elif exp_identity != final.analytical_identity:
                errors.append(
                    "ANALYTICAL_IDENTITY_MISMATCH: joint multivariate experiment must use "
                    "the canonical contract analytical identity"
                )
        else:
            if len(preds) != 1:
                errors.append(
                    f"PREDICTOR_IDENTITY_AMBIGUOUS: a pairwise PRIMARY experiment must test exactly one "
                    f"contract predictor, got {preds!r}"
                )
            else:
                (pred,) = preds
                if pred not in contract_preds:
                    errors.append(
                        f"PREDICTOR_MISMATCH: experiment predictor {pred!r} is not a contract predictor "
                        f"{list(final.predictor_columns)!r}"
                    )
                elif exp_identity != final.pair_identity(pred):
                    errors.append(
                        f"ANALYTICAL_IDENTITY_MISMATCH: identity does not belong to declared predictor {pred!r}"
                    )
    elif exp_identity != final.analytical_identity:
        errors.append("ANALYTICAL_IDENTITY_MISMATCH: experiment identity != contract analytical identity")

    if not hyp_identity:
        errors.append(
            "HYPOTHESIS_IDENTITY_MISSING: a PRIMARY experiment must reference the identity of the "
            "hypothesis it tests"
        )
    elif hyp_identity != exp_identity:
        errors.append(
            "HYPOTHESIS_IDENTITY_MISMATCH: the experiment tests a different analytical pair than "
            "its target hypothesis"
        )


def assert_single_primary_per_identity(experiments: List[Any]) -> None:
    """One analytical identity -> at most one PRIMARY experiment.

    Raises ValueError on a duplicate so a verification/supporting run can never
    silently double up as (or replace) the primary test of a claim.
    """
    seen: Dict[str, str] = {}
    for exp in experiments:
        if _experiment_role(exp) != ExperimentRole.PRIMARY:
            continue
        ident = getattr(exp, "analytical_identity", "") or ""
        if not ident:
            continue
        if ident in seen:
            raise ValueError(
                f"duplicate PRIMARY experiments for analytical identity {ident[:12]}: "
                f"{seen[ident]!r} and {getattr(exp, 'code', '?')!r}"
            )
        seen[ident] = getattr(exp, "code", "?")


def detect_plan_final_contract_conflicts(plan: Any, final: FinalAnalyticalContract) -> List[Dict[str, Any]]:
    """Compare the compiler PROPOSAL (plan) against the FINAL contract.

    The final contract is always authoritative.  Each conflict is tagged:
      BLOCKING   -- proposal and final make different, non-empty assertions about
                    problem class / target / predictors: execution must stop.
      OVERRIDDEN -- the proposal asserts something the final contract does not
                    have; the final contract wins and the difference is recorded.
    Unlike MethodSelectionEngine.detect_canonical_plan_role_disagreement, this also
    catches proposal-vs-empty cases and problem-class drift.
    """
    out: List[Dict[str, Any]] = []
    sem = getattr(plan, "semantics", None)

    def add(field_name: str, proposal: Any, final_value: Any, severity: str) -> None:
        out.append({"field": field_name, "proposal": proposal, "final": final_value, "severity": severity})

    plan_task = getattr(plan, "task", None)
    if plan_task and plan_task != final.canonical_task:
        add("problem_class", plan_task, final.canonical_task, "BLOCKING")

    plan_target = getattr(sem, "target_column", None)
    if plan_target and final.target_column and plan_target != final.target_column:
        add("target_column", plan_target, final.target_column, "BLOCKING")
    elif plan_target and not final.target_column:
        add("target_column", plan_target, None, "OVERRIDDEN")

    plan_preds = sorted(dict.fromkeys(getattr(sem, "explanatory_columns", None) or []))
    final_preds = sorted(final.predictor_columns)
    if plan_preds and final_preds and not set(final_preds).issubset(plan_preds):
        # The final contract holds a predictor the proposal never made: canonical
        # resolution may only narrow a proposal, never introduce one.
        add("predictor_columns", plan_preds, final_preds, "BLOCKING")
    elif plan_preds and final_preds and plan_preds != final_preds:
        # Canonical validation dropped some proposed predictors (fail-closed
        # filtering, e.g. a categorical column).  Recorded, not blocking.
        add("predictor_columns", plan_preds, final_preds, "OVERRIDDEN")
    elif plan_preds and not final_preds:
        add("predictor_columns", plan_preds, [], "OVERRIDDEN")

    plan_time = getattr(sem, "time_column", None)
    if plan_time and final.time_column and plan_time != final.time_column:
        add("time_column", plan_time, final.time_column, "OVERRIDDEN")
    plan_group = list(getattr(sem, "grouping_columns", None) or [])
    if plan_group and final.comparison_dimension and final.comparison_dimension not in plan_group:
        add("comparison_dimension", plan_group, final.comparison_dimension, "OVERRIDDEN")
    return out


def _family_for_capability(capability: MethodCapability) -> Optional[MethodFamily]:
    """Map a MethodCapability back to its MethodFamily via executor_id
    namespace, since MethodCapability itself does not store MethodFamily
    directly (see MethodSelectionDecision.method_family for the analogous
    single-family accessor at decision time).
    """
    code = capability.code
    if code in {"association_numeric", "association_categorical_binary"}:
        return MethodFamily.CORRELATION
    if code == "forecast_rolling_origin":
        return MethodFamily.FORECAST
    if code == "binary_risk_prediction":
        # Churn/risk prediction is scored under the CHURN family when the
        # contract's problem class is SURVIVAL_CHURN; otherwise it behaves
        # like a diagnostic-battery style predictive method.
        return MethodFamily.CHURN
    # generic_diagnostic_battery, comparison_group_effect, stable_segmentation,
    # row_reconciliation, and any future non-{CORRELATION,FORECAST,CHURN}
    # method all fall back to DIAGNOSTIC_BATTERY -- MethodFamily currently has
    # no dedicated slot for them (see MethodFamily in engines/method_selection.py).
    return MethodFamily.DIAGNOSTIC_BATTERY
