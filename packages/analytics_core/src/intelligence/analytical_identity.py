"""v20-C4.2.3 -- deterministic analytical identity and the FINAL analytical contract.

This module deliberately introduces NO new semantic model.  It is a strict,
deterministic *projection* of objects that already exist:

    MethodSelectionDecision (problem_class, objective, estimand, claim ceiling,
                             selected_method_code)
    + SemanticBindingSet     (role/column bindings)
    + dataset fingerprints   (dataset identity)
    + METHOD_REGISTRY        (method configuration, verification regime)

``FinalAnalyticalContract`` is the value object that is persisted on
``InvestigationContract`` at finalization and consumed by every downstream
stage (validation, hypotheses, experiments, evidence, provenance, replay).
It is *what the database stores*, not a competing decision-maker: it can only
be built from a reconciled MethodSelectionDecision (``from_method_decision``)
and it re-derives its own identity on load, failing closed on any mismatch.

Why a strict serializer instead of ``evidence_identity.canonical_json``:
that helper uses ``json.dumps(default=str)``, which silently stringifies
unknown objects (including ones whose repr embeds a memory address).  An
analytical identity must never depend on that, so every value here must be a
plain JSON type or the call raises ``AnalyticalIdentityError``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

IDENTITY_SCHEMA_VERSION = "aid-1"

# Compiler task vocabulary is not restated here; the canonical task string is
# copied verbatim from the reconciled decision (CANONICAL_TASK_AUTHORITY).


class AnalyticalIdentityError(ValueError):
    """Raised whenever an analytical identity cannot be established.

    Callers must treat this as fail-closed: no identity -> no execution.
    """


# --------------------------------------------------------------------------
# Strict canonical serialization
# --------------------------------------------------------------------------
def _canonicalize(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalyticalIdentityError(f"non-finite float at {path}")
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise AnalyticalIdentityError(f"non-string key {k!r} at {path}")
            out[k] = _canonicalize(v, f"{path}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(v, path) for v in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int)):
        # Enum members: identity depends on the value, never the repr.
        return _canonicalize(value.value, path)
    raise AnalyticalIdentityError(
        f"unsupported type {type(value).__name__} at {path}; refusing to stringify"
    )


def canonical_serialize(value: Any) -> str:
    """Deterministic, key-order-independent, address-independent serialization."""
    return json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _digest(domain: str, payload: Any) -> str:
    body = f"{IDENTITY_SCHEMA_VERSION}\n{domain}\n{canonical_serialize(payload)}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Component helpers
# --------------------------------------------------------------------------
def canonical_predictors(predictors: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Canonical predictor ordering: de-duplicated, sorted.

    This mirrors ``EstimandSpec.deterministic_predictor_columns`` so the
    identity layer and the estimand can never disagree about ordering.
    """
    out: List[str] = []
    for p in predictors or ():
        if not isinstance(p, str) or not p.strip():
            raise AnalyticalIdentityError(f"invalid predictor column {p!r}")
        out.append(p)
    return tuple(sorted(dict.fromkeys(out)))


def binding_triples(bindings: Any) -> Tuple[Tuple[str, str, str], ...]:
    """Identity-relevant projection of a SemanticBindingSet: (role, table, column).

    Confidence, provenance and physical-type details are deliberately excluded:
    they are diagnostics, not identity.
    """
    raw = getattr(bindings, "bindings", bindings) or []
    triples = set()
    for b in raw:
        role = getattr(b, "role", None)
        role_v = getattr(role, "value", role)
        col = getattr(b, "column", None)
        table = getattr(b, "table", None)
        if not role_v or not col:
            raise AnalyticalIdentityError(f"malformed semantic binding {b!r}")
        triples.add((str(role_v), str(table or ""), str(col)))
    return tuple(sorted(triples))


def method_config(method_code: str) -> Dict[str, Any]:
    """Method configuration relevant to computation, from the method registry.

    The registry has no explicit version field, so the configuration itself is
    the version: any change to these fields changes the analytical identity.
    """
    from packages.analytics_core.src.engines.method_selection import METHOD_REGISTRY

    cap = METHOD_REGISTRY.get(method_code)
    if cap is None:
        raise AnalyticalIdentityError(f"unknown method code {method_code!r}")
    return {
        "code": cap.code,
        "estimand": cap.estimand,
        "uncertainty_method": cap.uncertainty_method,
        "verification_method": cap.verification_method,
        "executor_id": cap.executor_id,
        "min_sample": int(cap.min_sample),
    }


def method_identifier(method_code: str) -> str:
    """Deterministic method identifier: ``<code>@<config-digest-12>``."""
    return f"{method_code}@{_digest('method-config', method_config(method_code))[:12]}"


def verification_regime_for(method_code: str) -> str:
    regime = method_config(method_code)["verification_method"]
    if not regime:
        raise AnalyticalIdentityError(f"method {method_code!r} declares no verification regime")
    return regime


# --------------------------------------------------------------------------
# The final analytical contract
# --------------------------------------------------------------------------
_ESTIMAND_FIELDS = (
    "target_column", "comparison_dimension", "time_column",
    "numerator_column", "denominator_column", "weight_column",
)


@dataclass(frozen=True)
class FinalAnalyticalContract:
    """The reconciled, final analytical decision.  Immutable by construction."""

    dataset_identity: str
    problem_class: str
    canonical_task: str
    objective: str
    target_column: Optional[str]
    predictor_columns: Tuple[str, ...]
    # ``None`` is an EXPLICIT, identity-bearing state: "no registry method was
    # selected" (e.g. the DIAGNOSTIC generic-battery / specialist route, which
    # select_for_plan deliberately leaves without a registry method_code).  It is
    # part of the analytical identity, so a later replan that adds a method is a
    # material change.  It is not a silent absence and not a failure.
    selected_method_code: Optional[str]
    claim_ceiling: Optional[str]
    verification_regime: Optional[str]
    method_family: str
    bindings: Tuple[Tuple[str, str, str], ...]
    # A decision produced by MethodSelectionEngine.fallback() (method selection
    # crashed) is a different claim from a properly reconciled one.
    fallback_used: bool = False
    comparison_dimension: Optional[str] = None
    time_column: Optional[str] = None
    numerator_column: Optional[str] = None
    denominator_column: Optional[str] = None
    weight_column: Optional[str] = None
    schema_version: str = IDENTITY_SCHEMA_VERSION

    # ----- construction -------------------------------------------------
    def __post_init__(self) -> None:
        if not self.dataset_identity:
            raise AnalyticalIdentityError("dataset identity is empty")
        for name in ("problem_class", "canonical_task", "objective", "method_family"):
            if not getattr(self, name):
                raise AnalyticalIdentityError(f"final contract field {name!r} is empty")
        if self.selected_method_code:
            if not self.verification_regime:
                raise AnalyticalIdentityError("a selected method must declare a verification regime")
            method_config(self.selected_method_code)  # raises for unknown methods
        elif self.verification_regime:
            raise AnalyticalIdentityError("a verification regime requires a selected method")
        if self.predictor_columns != canonical_predictors(self.predictor_columns):
            raise AnalyticalIdentityError("predictor_columns must be in canonical (sorted, unique) order")
        if self.schema_version != IDENTITY_SCHEMA_VERSION:
            raise AnalyticalIdentityError(f"unsupported identity schema {self.schema_version!r}")

    @classmethod
    def from_method_decision(
        cls,
        decision: Any,
        *,
        dataset_fingerprints: Optional[Mapping[str, str]],
        semantic_bindings: Any,
    ) -> "FinalAnalyticalContract":
        """Build the final contract from a *reconciled* MethodSelectionDecision.

        Fails closed when the dataset cannot be identified or no method was
        selected (``canonical_dataset_identity({})`` would otherwise happily
        hash the empty dict into a valid-looking digest).
        """
        if not dataset_fingerprints:
            raise AnalyticalIdentityError("no dataset fingerprints: dataset identity cannot be established")
        code = getattr(decision, "selected_method_code", None) or None
        est = decision.estimand
        # A CORRELATIONAL (pairwise association) claim is defined by target +
        # predictor only.  Grouping/time/ratio components that leaked into the
        # estimand from a plan proposal (e.g. the column canonical resolution
        # rejected as a predictor) are not part of the claim and must not change
        # its identity -- otherwise an invalid predictor would alter the valid pair.
        pairwise = decision.problem_class.value == "CORRELATIONAL"

        def _component(name: str) -> Optional[str]:
            return None if pairwise else getattr(est, name, None)

        return cls(
            dataset_identity=_digest("dataset", dict(dataset_fingerprints)),
            problem_class=decision.problem_class.value,
            canonical_task=str(decision.canonical_task or ""),
            objective=decision.objective.value,
            target_column=getattr(est, "target_column", None),
            predictor_columns=canonical_predictors(getattr(est, "predictor_columns", None)),
            selected_method_code=code,
            claim_ceiling=decision.max_verdict_tier,
            verification_regime=(verification_regime_for(code) if code else None),
            method_family=decision.method_family.value,
            fallback_used=bool(getattr(decision, "fallback_used", False)),
            bindings=binding_triples(semantic_bindings),
            comparison_dimension=_component("comparison_dimension"),
            time_column=_component("time_column"),
            numerator_column=_component("numerator_column"),
            denominator_column=_component("denominator_column"),
            weight_column=_component("weight_column"),
        )

    # ----- identity -----------------------------------------------------
    def estimand_components(self, predictor: Optional[str] = None) -> Dict[str, Any]:
        comps = {f: getattr(self, f) for f in _ESTIMAND_FIELDS}
        comps["predictor_columns"] = list(self.predictor_columns if predictor is None else (predictor,))
        return comps

    @property
    def semantic_binding_identity(self) -> str:
        return _digest("bindings", [list(t) for t in self.bindings])

    @property
    def estimand_identity(self) -> str:
        return _digest("estimand", self.estimand_components())

    def _method_payload(self) -> Dict[str, Any]:
        return {
            "method": (method_identifier(self.selected_method_code) if self.selected_method_code else None),
            "verification_regime": self.verification_regime,
            "claim_ceiling": self.claim_ceiling,
            "fallback_used": self.fallback_used,
        }

    @property
    def analytical_identity(self) -> str:
        """Identity of the whole analytical claim (all predictors, canonical order)."""
        return _digest("analytical-claim", {
            "dataset": self.dataset_identity,
            "bindings": self.semantic_binding_identity,
            "problem_class": self.problem_class,
            "canonical_task": self.canonical_task,
            "objective": self.objective,
            "estimand": self.estimand_components(),
            **self._method_payload(),
        })

    def pair_binding_identity(self, predictor: str) -> str:
        """Bindings restricted to this pair's own columns, so a pair's identity
        does not change when an unrelated predictor is added or removed."""
        keep = {c for c in (self.target_column, predictor, self.comparison_dimension, self.time_column) if c}
        return _digest("pair-bindings", [list(t) for t in self.bindings if t[2] in keep])

    def pair_identity(self, predictor: str) -> str:
        """Identity of ONE analytical pair (target, predictor).  Independent of
        predictor order and of the other predictors in the contract."""
        if predictor not in self.predictor_columns:
            raise AnalyticalIdentityError(
                f"predictor {predictor!r} is not part of this contract {list(self.predictor_columns)!r}"
            )
        return _digest("analytical-pair", {
            "dataset": self.dataset_identity,
            "bindings": self.pair_binding_identity(predictor),
            "problem_class": self.problem_class,
            "canonical_task": self.canonical_task,
            "objective": self.objective,
            "estimand": self.estimand_components(predictor),
            **self._method_payload(),
        })

    def pair_identities(self) -> Dict[str, str]:
        return {p: self.pair_identity(p) for p in self.predictor_columns}

    # ----- persistence --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_identity": self.dataset_identity,
            "problem_class": self.problem_class,
            "canonical_task": self.canonical_task,
            "objective": self.objective,
            "target_column": self.target_column,
            "predictor_columns": list(self.predictor_columns),
            "selected_method_code": self.selected_method_code,
            "method_identifier": (method_identifier(self.selected_method_code) if self.selected_method_code else None),
            "fallback_used": self.fallback_used,
            "claim_ceiling": self.claim_ceiling,
            "verification_regime": self.verification_regime,
            "method_family": self.method_family,
            "bindings": [list(t) for t in self.bindings],
            "comparison_dimension": self.comparison_dimension,
            "time_column": self.time_column,
            "numerator_column": self.numerator_column,
            "denominator_column": self.denominator_column,
            "weight_column": self.weight_column,
            "estimand_identity": self.estimand_identity,
            "analytical_identity": self.analytical_identity,
            "pair_identities": self.pair_identities(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalAnalyticalContract":
        """Rebuild and VERIFY.  Any stored identity that does not match the
        re-derived one (tampering, registry drift, corruption) fails closed."""
        try:
            obj = cls(
                dataset_identity=data["dataset_identity"],
                problem_class=data["problem_class"],
                canonical_task=data["canonical_task"],
                objective=data["objective"],
                target_column=data.get("target_column"),
                predictor_columns=tuple(data.get("predictor_columns") or ()),
                selected_method_code=data.get("selected_method_code"),
                claim_ceiling=data.get("claim_ceiling"),
                verification_regime=data.get("verification_regime"),
                method_family=data["method_family"],
                fallback_used=bool(data.get("fallback_used", False)),
                bindings=tuple(tuple(t) for t in data.get("bindings") or ()),
                comparison_dimension=data.get("comparison_dimension"),
                time_column=data.get("time_column"),
                numerator_column=data.get("numerator_column"),
                denominator_column=data.get("denominator_column"),
                weight_column=data.get("weight_column"),
                schema_version=data.get("schema_version", ""),
            )
        except KeyError as exc:
            raise AnalyticalIdentityError(f"final contract missing field {exc}") from exc
        stored = data.get("analytical_identity")
        if stored != obj.analytical_identity:
            raise AnalyticalIdentityError(
                "stored analytical_identity does not match the identity re-derived from the "
                "stored components (corruption, tampering, or method-registry drift)"
            )
        if data.get("pair_identities") != obj.pair_identities():
            raise AnalyticalIdentityError("stored pair_identities do not match re-derived pair identities")
        return obj

    def material_diff(self, other: "FinalAnalyticalContract") -> List[str]:
        """Names of the identity-relevant fields that differ (empty == same claim)."""
        return [] if self.analytical_identity == other.analytical_identity else [
            f for f in (
                "dataset_identity", "problem_class", "canonical_task", "objective",
                "target_column", "predictor_columns", "selected_method_code",
                "claim_ceiling", "verification_regime", "bindings", "fallback_used",
                *_ESTIMAND_FIELDS,
            ) if getattr(self, f) != getattr(other, f)
        ]


# --------------------------------------------------------------------------
# Hypothesis / experiment linkage
# --------------------------------------------------------------------------
def hypothesis_predictor(hypothesis: Any) -> str:
    """The predictor a hypothesis is about, from STRUCTURED fields only:
    ``secondary_metric`` (pairwise correlation hypotheses) else ``target_dimension``
    (grouped / churn hypotheses).  Empty when it declares neither."""
    return (getattr(hypothesis, "secondary_metric", "") or getattr(hypothesis, "target_dimension", "") or "")


def pair_identity_for_hypothesis(final: FinalAnalyticalContract, hypothesis: Any) -> Optional[str]:
    """Pair identity of a hypothesis of this contract, from its structured fields
    (``target_metric`` and its predictor) -- never from narrative text.

    Returns None when the hypothesis is not about a (target, predictor) pair of
    this contract -- e.g. a confounder counter-hypothesis on a dimension the user
    did not ask about.  Such hypotheses stay unlinked; they are never guessed
    onto a pair.
    """
    predictor = hypothesis_predictor(hypothesis)
    if predictor in final.predictor_columns and getattr(hypothesis, "target_metric", None) == final.target_column:
        return final.pair_identity(predictor)
    return None


def experiment_outcome(experiment: Any) -> Optional[str]:
    """Declared outcome of an experiment: explicit ``outcome``, else ``target_metric``.

    Single implementation, shared with experiment_contract_validation (which
    delegates here) so validation and identity stamping can never disagree.
    Structured fields only -- never narrative text.
    """
    explicit = getattr(experiment, "outcome", None)
    if explicit:
        return explicit
    return getattr(experiment, "target_metric", None)


def experiment_predictors(experiment: Any) -> List[str]:
    """Declared predictors: explicit ``predictors``, else the non-outcome
    ``metrics`` then ``dimensions`` (de-duplicated, order preserved)."""
    explicit = getattr(experiment, "predictors", None)
    if explicit:
        return list(explicit)
    outcome = experiment_outcome(experiment)
    metrics = list(getattr(experiment, "metrics", None) or [])
    dims = list(getattr(experiment, "dimensions", None) or [])
    pool = [m for m in metrics if m and m != outcome] + [d for d in dims if d and d != outcome]
    seen = set()
    out: List[str] = []
    for c in pool:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def pair_identity_for_experiment(final: FinalAnalyticalContract, experiment: Any) -> Optional[str]:
    """Pair identity of an experiment from its declared structured predictors.

    Returns None unless the experiment declares exactly one contract predictor and
    (when it names an outcome) that outcome is the contract target.
    """
    outcome = experiment_outcome(experiment)
    if outcome is not None and outcome != final.target_column:
        return None
    preds = [p for p in experiment_predictors(experiment) if p != outcome]
    if len(preds) == 1 and preds[0] in final.predictor_columns:
        return final.pair_identity(preds[0])
    return None


def hypothesis_identity_for(final: FinalAnalyticalContract, hypothesis: Any) -> str:
    """Analytical identity a hypothesis belongs to, or ``""`` when it is not linked.

    Joint multivariable association hypotheses are one estimand/claim over the
    complete predictor set, so H1/H0 share the canonical contract identity.
    Pairwise hypotheses retain their predictor-specific identities.
    """
    if final.predictor_columns:
        joint_predictors = list((getattr(hypothesis, "provenance", None) or {}).get("joint_predictors", []) or [])
        if len(joint_predictors) > 1 and set(joint_predictors) == set(final.predictor_columns):
            return final.analytical_identity
        return pair_identity_for_hypothesis(final, hypothesis) or ""
    return final.analytical_identity


def stamp_experiment_identity(
    final: FinalAnalyticalContract,
    experiment: Any,
    hypothesis_identity_by_code: Mapping[str, str],
) -> None:
    """Stamp an experiment's analytical identity and its target hypothesis' identity.

    The experiment identity comes from its STRUCTURED predictors, and the
    hypothesis identity from the hypothesis it targets -- so an experiment whose
    predictor disagrees with its hypothesis ends up with two different identities
    and is rejected by validate_experiment_against_contract().  Nothing is ever
    re-attributed.
    """
    agg = str(getattr(experiment, "aggregation_type", "") or "").upper()
    preds = [p for p in experiment_predictors(experiment) if p != experiment_outcome(experiment)]
    is_joint = agg == "MULTIVARIATE_REGRESSION" and len(preds) > 1
    if is_joint and set(preds) == set(final.predictor_columns):
        ident = final.analytical_identity
    else:
        ident = pair_identity_for_experiment(final, experiment) or ""
        if not ident and not final.predictor_columns and getattr(experiment, "experiment_role", "") == "PRIMARY":
            ident = final.analytical_identity
    experiment.analytical_identity = ident
    experiment.hypothesis_analytical_identity = hypothesis_identity_by_code.get(
        getattr(experiment, "target_hypothesis_code", None) or "", ""
    )
