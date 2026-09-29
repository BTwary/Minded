"""PredictiveHypothesis: Formal predictive hypothesis objects with testable observables and falsification criteria."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.hypothesis_identity import stamp_semantic_identity
from packages.analytics_core.src.engines.method_selection import ProblemClass

# v20-C2: predictive_hypothesis.py's semantic authority is now
# CanonicalSemanticResolution (built once per investigation at the
# synthesis entry boundary -- see _build_canonical_semantics below), not
# the raw SemanticResolution fields it wraps. build_canonical_semantic_resolution
# and CanonicalSemanticResolution are imported here rather than through any
# deprecated runtime module, per v20-C2 section 16.
from packages.analytics_core.src.intelligence.semantic_resolution_builder import (
    build_canonical_semantic_resolution,
)
from packages.schemas.src.semantic_resolution_contract import (
    CanonicalSemanticResolution,
    QuestionRoleProposal,
)

# v20-C2 section 4: the only place the compiler task / decision maps onto
# TARGET-vs-OUTCOME role classification for the canonical outcome field.
# This does not change *which* column resolves -- only how it is labeled --
# so it carries zero risk to existing hypothesis content. ProblemClass
# values that already coincide with the compiler task string used by
# semantic_builder_vocabulary.TARGET_STYLE_TASKS (e.g. ProblemClass.PREDICTION
# == "PREDICTION") need no entry here; only ProblemClass.FORECASTING's value
# ("FORECASTING") does not textually match the compiler's "FORECAST" task
# string and needs translating.
_PROBLEM_CLASS_TASK_STRINGS = {
    ProblemClass.FORECASTING: "FORECAST",
}


@dataclass
class PredictiveHypothesis:
    """Rigorous scientific hypothesis node with empirical predictions and falsification boundaries."""
    id: str
    hypothesis_code: str
    claim: str
    mechanism: str
    predicted_observables_if_true: List[str]
    predicted_observables_if_false: List[str]
    falsification_criteria: str
    required_assumptions: List[str]
    prior_probability: float
    posterior_probability: float
    belief_state: str = "PROPOSED"  # PROPOSED, ACTIVE, SUPPORTED, WEAKENED, REFUTED, INCONCLUSIVE, RETIRED
    target_metric: str = ""
    target_dimension: str = ""
    is_counter_hypothesis: bool = False

    # --- Structured extension fields (prediction lifecycle & emergent synthesis) ---
    target_value: Optional[Union[float, str]] = None
    mechanism_detail: str = ""
    source_evidence: List[str] = field(default_factory=list)
    parent_hypotheses: List[str] = field(default_factory=list)
    generated_reason: str = ""
    prediction_ids: List[str] = field(default_factory=list)
    supporting_prediction_count: int = 0
    refuted_prediction_count: int = 0
    unresolved_prediction_count: int = 0
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    # DEFECT-005: for a CORRELATION hypothesis, the SECOND numeric variable
    # being tested for association with target_metric (target_dimension is
    # reserved for categorical GROUP BY dimensions elsewhere and must not be
    # repurposed for this). Empty for every non-correlation hypothesis.
    secondary_metric: str = ""
    # DEFECT-005: marks this hypothesis's claim_type for downstream verdict/
    # narrative formulation so the epistemic type (ASSOCIATION vs
    # PREDICTION vs OBSERVATION) is preserved end-to-end instead of being
    # inferred (or defaulted to root-cause-style DIAGNOSED language)
    # downstream. "" means the generic/root-cause default.
    claim_type: str = ""

    # --- Canonical identity & consolidation bookkeeping (Item 1) ---
    # canonical_identity is the ONLY signal used to decide two hypotheses
    # are the same claim; it is computed via
    # hypothesis_identity.compute_semantic_identity and must never be
    # recomputed with different logic elsewhere.
    canonical_identity: str = ""
    # v20-C4.2.3: deterministic analytical-pair identity from the FINAL contract
    # (structured target/predictor fields only -- never narrative text).  The
    # counter-hypothesis (H0) of a pair shares its pair's identity.
    analytical_identity: str = ""
    temporal_scope: str = ""
    direction: str = ""
    experiment_refs: List[str] = field(default_factory=list)
    posterior_history: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    # One entry per hypothesis (batch-duplicate or cross-round match) that
    # was folded into this one, PLUS one entry for the surviving hypothesis
    # itself -- so len(consolidation_log) is the total number of evidence
    # contributions behind this canonical hypothesis.
    consolidation_log: List[Dict[str, Any]] = field(default_factory=list)
    # P0-A (defect019 audit): explicit dimension-resolution provenance.
    # "RESOLVED" means target_dimension names a real, unambiguously chosen
    # column. "UNRESOLVED" means zero candidate grouping columns existed.
    # "AMBIGUOUS" means multiple equally-plausible candidates existed and
    # none was silently chosen. target_dimension is "" whenever this is not
    # "RESOLVED" -- never a fabricated placeholder like "segment".
    dimension_resolution_status: str = "RESOLVED"
    dimension_candidates: List[str] = field(default_factory=list)


def resolve_group_dimension(semantic: "SemanticResolution") -> "Tuple[str, str, List[str]]":
    """Single authoritative rule for turning candidate categorical columns
    into a grouping dimension, without ever fabricating one.

    Returns (dim, status, candidates):
      - explicit semantic.group_dimension_col already resolved upstream
        -> (col, "RESOLVED", [col])
      - exactly one candidate categorical column
        -> (col, "RESOLVED", [col])
      - zero candidates
        -> ("", "UNRESOLVED", [])
      - two or more candidates
        -> ("", "AMBIGUOUS", candidates)

    Never returns a literal fallback such as "segment", "data_table", or
    the first column by position. Callers must not silently pick one of
    several ambiguous candidates.
    """
    if semantic.group_dimension_col:
        return semantic.group_dimension_col, "RESOLVED", [semantic.group_dimension_col]
    candidates = list(semantic.available_categorical_cols or [])
    if len(candidates) == 1:
        return candidates[0], "RESOLVED", candidates
    if len(candidates) == 0:
        return "", "UNRESOLVED", []
    return "", "AMBIGUOUS", candidates


def _build_canonical_semantics(
    semantic: "SemanticResolution",
    intent: Any = None,
    decision: Any = None,
    question_roles: Optional[QuestionRoleProposal] = None,
) -> CanonicalSemanticResolution:
    """Builds the v20-C1 CanonicalSemanticResolution exactly once for this
    investigation, at the hypothesis-synthesis entry boundary (v20-C2
    section 3). Every hypothesis-family method downstream receives the
    resulting object rather than independently rebuilding it or reading
    semantic.* fields directly.

    The task string derived here ONLY affects whether the canonical
    outcome field is labeled TARGET vs OUTCOME (see
    semantic_builder_vocabulary.TARGET_STYLE_TASKS) -- it never changes
    *which* column is resolved, so an imprecise/missing task string
    (e.g. no decision or intent supplied) cannot alter hypothesis content.
    """
    task = ""
    if decision is not None:
        pc = getattr(decision, "problem_class", None)
        task = _PROBLEM_CLASS_TASK_STRINGS.get(pc, getattr(pc, "value", "") or str(pc or ""))
    elif intent is not None:
        task = str(getattr(intent, "intent_type", "") or "")
    q_roles = question_roles
    if q_roles is None and decision is not None:
        est = getattr(decision, "estimand", None)
        if est is not None and getattr(est, "predictor_columns", None):
            q_roles = QuestionRoleProposal(
                target_column=getattr(est, "target_column", None),
                requested_explanatory=list(getattr(est, "predictor_columns", None) or []),
            )
    return build_canonical_semantic_resolution(semantic, task, question_roles=q_roles)


def _dimension_from_canonical(canonical: CanonicalSemanticResolution) -> "Tuple[str, str, List[str]]":
    """Reads the (dim, status, candidates) tuple resolve_group_dimension()
    used to return, from the canonical dimension resolution instead of
    recomputing it (v20-C2 section 5/14). canonical.dimension is built by
    the exact same 0/1/>1 fail-closed rule as resolve_group_dimension()
    (parity-verified by test_v20c1_canonical_semantic_resolution.py), so
    this is a read of the same fact, not a second resolution authority.
    resolve_group_dimension() itself is left in place, unchanged, since
    controller.py and existing unit tests call it directly and migrating
    those call sites is out of scope for C2 (section 22).
    """
    d = canonical.dimension
    status = d.status.value if hasattr(d.status, "value") else str(d.status)
    return d.value or "", status, list(d.candidates)


def _target_metric_via_canonical(
    semantic: "SemanticResolution", canonical: CanonicalSemanticResolution
) -> Optional[str]:
    """Reads target_metric_col through the canonical outcome resolution
    for the common case (every non-churn hypothesis family), while
    guarding against a pre-existing routing edge case: a dataset can have
    churn_event_col resolved (making the canonical outcome field
    churn-sourced) yet still fall through synthesize_competing_hypotheses'
    intent-based routing into a non-churn hypothesis family, if neither
    the intent nor the question text signals churn (see
    docs/AAOS_V20C2_BEHAVIORAL_EQUIVALENCE.md). In that edge case the
    canonical outcome would report the churn column, not target_metric_col
    -- so this only trusts the canonical value when its own provenance
    confirms it was actually built from target_metric_col, and falls back
    to the raw field otherwise. This keeps C2 exactly
    behavior-preserving; it is not a second permanent resolution path for
    the common case, which always takes the canonical branch.
    """
    if canonical.outcome.provenance.get("source") == "semantic.target_metric_col":
        return canonical.outcome_column()
    return semantic.target_metric_col


class HypothesisSynthesizer:
    """Synthesizes competing, predictive hypotheses directly from the Semantic World Model and discovered evidence."""

    @staticmethod
    def synthesize_competing_hypotheses(
        semantic: SemanticResolution,
        question: str,
        intent: Any = None,
        decision: Any = None,
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
    ) -> List[PredictiveHypothesis]:
        # BUGFIX (DEFECT-005): this method used to ALWAYS generate the same
        # fixed root-cause-style hypothesis set (dimensional concentration /
        # systemic drift / secondary confounding) regardless of what kind of
        # question was actually asked -- `intent` was computed upstream in
        # the controller but never passed in here at all. A CORRELATION
        # question ("is X associated with Y?") and a FORECAST question
        # ("what will X be next month?") therefore got byte-identical
        # hypotheses to a ROOT_CAUSE question ("why did X change?"), which
        # then made every downstream stage (experiment selection, evidence
        # interpretation, verdict claim type) generate root-cause-shaped
        # output for questions that were never asking a root-cause question.
        # Branch on intent.intent_type now; ROOT_CAUSE/PERFORMANCE/GENERAL
        # (and any unset/legacy intent) keep exactly the previous behavior.
        # v20-C2: build the canonical semantic resolution once, at this
        # entry boundary, and thread it through every hypothesis-family
        # branch below instead of letting each branch (or a later call)
        # rebuild/re-derive it independently (section 3).
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic, intent=intent, decision=decision)

        # Canonical decision-aware routing. When a decision is supplied, it is the
        # single family authority and the family implementations themselves remain unchanged.
        #
        # secondary_metric_col/time_col truthiness below is read through the
        # canonical resolution (section 6/7) -- both are resolved by
        # semantic_resolution_builder with candidate_pool=None, so
        # canonical.secondary_metric_column() / canonical.time_variable() are
        # RESOLVED-or-UNRESOLVED-only and equal the raw field's truthiness
        # exactly; there is no AMBIGUOUS state to lose information about
        # here, so this is a pure consolidation with no behavior change.
        #
        # The churn_event_col / churn_outcome_available check just below is
        # deliberately NOT migrated to a canonical accessor: canonical's own
        # churn-relevance test additionally considers churn_event_ambiguity
        # and churn_event_resolution_status, which is a broader (and
        # differently-scoped) condition than this routing check's plain
        # "is churn_event_col set, or is outcome unavailable" test. Widening
        # this specific check to match canonical would change which
        # investigations enter the churn family -- exactly the scientific
        # behavior change v20-C2 section 13 forbids without a test
        # demonstrating the old check depended on an unsafe fallback. See
        # docs/AAOS_V20C2_PREDICTIVE_SEMANTIC_MIGRATION.md for the full
        # per-site rationale.
        if decision is not None:
            pc = getattr(decision, "problem_class", None)
            if pc == ProblemClass.SURVIVAL_CHURN:
                return HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, question, canonical_semantics=canonical)
            if pc == ProblemClass.CORRELATIONAL and (canonical.secondary_metric_column() or (getattr(decision, "estimand", None) and getattr(decision.estimand, "predictor_columns", None))):
                # v20-C4.2.1c: decision.estimand.predictor_columns is the
                # single authoritative predictor MethodSelectionEngine.decide()
                # already resolved (question-explicit when valid, discovered
                # otherwise -- see method_selection.py's C4.2.1b fix). Pass
                # it straight through rather than letting this hypothesis
                # re-derive its own answer from `canonical`.
                est = getattr(decision, "estimand", None)
                predictor_cols = getattr(est, "predictor_columns", None) or []
                predictor_column = predictor_cols[0] if len(predictor_cols) == 1 else None
                joint = bool(getattr(getattr(decision, "estimand", None), "joint_predictors", False))
                return HypothesisSynthesizer._synthesize_correlation_hypotheses(
                    semantic, question, canonical_semantics=canonical, predictor_column=predictor_column, predictor_columns=predictor_cols,
                    joint_predictors=joint,
                )
            if pc == ProblemClass.FORECASTING and canonical.time_variable():
                return HypothesisSynthesizer._synthesize_forecast_hypotheses(semantic, question, canonical_semantics=canonical)
            if pc == ProblemClass.COMPARATIVE:
                return HypothesisSynthesizer._synthesize_segmentation_hypotheses(semantic, question, canonical_semantics=canonical)
        else:
            intent_type = getattr(intent, "intent_type", None)
            if intent_type == "CHURN" or (hasattr(semantic, "churn_event_col") and semantic.churn_event_col is not None) or not getattr(semantic, "churn_outcome_available", True):
                if intent_type == "CHURN" or any(w in question.lower() for w in ["churn", "cancellation", "attrition", "dropoff"]):
                    return HypothesisSynthesizer._synthesize_churn_hypotheses(semantic, question, canonical_semantics=canonical)
            if intent_type == "CORRELATION" and (canonical.secondary_metric_column() or canonical.requested_explanatory_columns()):
                return HypothesisSynthesizer._synthesize_correlation_hypotheses(
                    semantic, question, canonical_semantics=canonical,
                    predictor_columns=canonical.requested_explanatory_columns(),
                )
            if intent_type == "FORECAST" and canonical.time_variable():
                return HypothesisSynthesizer._synthesize_forecast_hypotheses(semantic, question, canonical_semantics=canonical)
            if intent_type in ("SEGMENTATION", "PERFORMANCE"):
                return HypothesisSynthesizer._synthesize_segmentation_hypotheses(semantic, question, canonical_semantics=canonical)

        # Default/legacy family (ROOT_CAUSE/GENERAL/unset intent). metric
        # reads through the canonical outcome resolution (guarded against
        # the churn-fallthrough edge case, see _target_metric_via_canonical);
        # dim/available_dims read the canonical dimension resolution and the
        # full categorical-candidate pool respectively (section 5/12).
        metric = _target_metric_via_canonical(semantic, canonical)
        available_dims = canonical.available_categorical_candidates
        dim = canonical.dimension.value or (available_dims[0] if len(available_dims) == 1 else "")
        dim_claim = f"across {dim}" if dim else f"in {metric}"

        hyps: List[PredictiveHypothesis] = []

        # H1: Localized Dimensional Concentration
        h1 = PredictiveHypothesis(
            id=f"HYP-01",
            hypothesis_code="HYP-01",
            claim=f"Change in {metric} is concentrated within specific high-volume partitions." if not dim else f"Change in {metric} is concentrated within specific high-volume categories of {dim}.",
            mechanism=f"Unequal shift across partitions with significant variance explained." if not dim else f"Unequal shift across {dim} partitions with significant variance explained (ANOVA eta^2 >= 15%).",
            predicted_observables_if_true=[
                f"High variance across {dim} levels for {metric}." if dim else f"High variation in {metric}.",
                f"Top 20% categories account for > 60% of absolute variance." if dim else f"Top partitions dominate net delta.",
            ],
            predicted_observables_if_false=[
                f"Uniform distribution shift across all categories of {dim}." if dim else f"Uniform distribution across all observations.",
            ],
            falsification_criteria=f"Empirical ANOVA eta-squared across {dim} is less than 5.0%." if dim else f"Variance explained is less than 5.0%.",
            required_assumptions=[f"Data completeness for dimension {dim}" if dim else "Data completeness", "Consistent reporting definitions"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=metric,
            target_dimension=dim,
            is_counter_hypothesis=False,
        )
        hyps.append(h1)

        # H2: Uniform Systemic Macro Drift (Counter-Hypothesis)
        h2 = PredictiveHypothesis(
            id=f"HYP-02",
            hypothesis_code="HYP-02",
            claim=f"Systemic macro-level shift across all partitions with uniform proportional impact." if not dim else f"Systemic macro-level shift across all categories of {dim} with uniform proportional impact.",
            mechanism=f"Baseline macro drift across the entire population without localized category variance.",
            predicted_observables_if_true=[
                f"Categorical variance is statistically negligible.",
                f"Mean percentage shift is uniform across all categories.",
            ],
            predicted_observables_if_false=[
                f"Heavy concentration of delta within top categories.",
            ],
            falsification_criteria=f"At least one category accounts for > 40% of the entire net change.",
            required_assumptions=["Macro uniformity across partitions"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=metric,
            target_dimension=dim,
            is_counter_hypothesis=True,
        )
        hyps.append(h2)

        # H3: Secondary Dimensional Confounding (if additional dimensions exist)
        sec_dim = next((d for d in available_dims if d != dim), None)
        if sec_dim:
            h3 = PredictiveHypothesis(
                id=f"HYP-03",
                hypothesis_code="HYP-03",
                claim=f"Apparent variance across {dim} is confounded or mediated by {sec_dim}.",
                mechanism=f"Simpson's paradox or composition shift in {sec_dim} masquerading as {dim} shift.",
                predicted_observables_if_true=[
                    f"Controlling for {sec_dim} reduces between-group variance in {dim}.",
                ],
                predicted_observables_if_false=[
                    f"Effect of {dim} remains robust after conditioning on {sec_dim}.",
                ],
                falsification_criteria=f"Partial correlation or two-way ANOVA shows {dim} effect remains orthogonal.",
                required_assumptions=[f"Orthogonality between {dim} and {sec_dim}"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=metric,
                target_dimension=sec_dim,
                is_counter_hypothesis=True,
            )
            hyps.append(h3)
        # No domain prior is justified here. Every generated hypothesis starts
        # with the same structural weight and is normalized to 1/N below.
        # Evidence, not hypothesis position or label, determines posterior belief.

        # Normalize neutral structural priors to 1/N. If a future caller supplies
        # an explicit domain prior, it must be handled by the authoritative prior
        # policy rather than by this hypothesis synthesizer.
        total_priors = sum(h.prior_probability for h in hyps)
        for h in hyps:
            h.prior_probability /= total_priors
            h.posterior_probability = h.prior_probability

        for h in hyps:
            h.direction = semantic.direction_hint
            stamp_semantic_identity(h)
            h.consolidation_log.append({
                "event": "created",
                "prior_source": "UNIFORM_STRUCTURAL_PRIOR",
                "hypothesis_code": h.hypothesis_code,
                "claim": h.claim,
            })

        return hyps

    @staticmethod
    def _synthesize_correlation_hypotheses(
        semantic: SemanticResolution,
        question: str,
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
        predictor_column: Optional[str] = None,
        predictor_columns: Optional[Sequence[str]] = None,
        joint_predictors: bool = False,
    ) -> List[PredictiveHypothesis]:
        """DEFECT-005 (section 4): real CORRELATION hypothesis pair.

        H1: metric and secondary_metric are materially associated.
        H0: metric and secondary_metric are approximately independent.

        Final claim type is deliberately ASSOCIATION, never CAUSATION --
        this is a purely observational bivariate test with no causal
        identification strategy attached.

        v20-C4.2.1c: `predictor_column`, when supplied by the caller (the
        production call site passes `decision.estimand.predictor_columns[0]`
        -- see synthesize_competing_hypotheses above), is authoritative over
        `canonical.secondary_metric_column()`. Falls back to the discovered
        secondary metric only when no predictor_column is given (legacy
        callers with no MethodSelectionDecision at all), preserving prior
        behavior for those call sites exactly.

        v20-C4.2.2e: `predictor_columns`, when multiple predictors are supplied,
        synthesizes 1 pairwise hypothesis per predictor (HYP-01, HYP-02, ...).
        When 1 predictor is supplied, preserves the classic H1/H0 pair.
        """
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic)
        metric = _target_metric_via_canonical(semantic, canonical)

        preds = []
        if predictor_columns:
            preds = list(dict.fromkeys(predictor_columns))
        elif predictor_column:
            preds = [predictor_column]
        elif canonical is not None and canonical.requested_explanatory_columns():
            preds = list(dict.fromkeys(canonical.requested_explanatory_columns()))
        elif canonical is not None and canonical.secondary_metric_column():
            preds = [canonical.secondary_metric_column()]
        elif getattr(semantic, "secondary_metric_col", None):
            preds = [semantic.secondary_metric_col]

        if joint_predictors and len(preds) > 1:
            joined = ", ".join(preds[:-1]) + (f" and {preds[-1]}" if len(preds) > 1 else preds[0])
            h1 = PredictiveHypothesis(
                id="HYP-01", hypothesis_code="HYP-01",
                claim=f"{metric} has conditional associations with {joined}, holding the included predictors constant.",
                mechanism=f"The joint linear model for {metric} contains non-zero partial effects for the included predictors.",
                predicted_observables_if_true=[f"The fitted multivariable model explains meaningful variation in {metric} beyond an intercept-only model."],
                predicted_observables_if_false=[f"Jointly modeling {joined} adds little explanatory information beyond an intercept-only model."],
                falsification_criteria="Joint model explanatory power is negligible and no predictor has a stable partial effect.",
                required_assumptions=[f"Data completeness for {metric} and all included predictors", "Linear conditional mean specification", "No exact multicollinearity"],
                prior_probability=1.0, posterior_probability=1.0, target_metric=metric, target_dimension="",
                secondary_metric=", ".join(preds), is_counter_hypothesis=False, claim_type="ASSOCIATION",
                provenance={"joint_predictors": list(preds)},
            )
            h0 = PredictiveHypothesis(
                id="HYP-02", hypothesis_code="HYP-02",
                claim=f"{metric} has no material conditional association with the included predictor set ({joined}).",
                mechanism="Any apparent joint explanatory power is attributable to sampling variation or shared structure not supported by the observed data.",
                predicted_observables_if_true=["The joint model adds no material explanatory power and partial effects are unstable or negligible."],
                predicted_observables_if_false=["The joint model shows stable explanatory power and/or meaningful partial effects."],
                falsification_criteria="The joint model explains material variation or at least one partial effect is stable and practically meaningful.",
                required_assumptions=["Independent observations conditional on the modeled structure"],
                prior_probability=1.0, posterior_probability=1.0, target_metric=metric, target_dimension="",
                secondary_metric=", ".join(preds), is_counter_hypothesis=True, claim_type="ASSOCIATION",
                provenance={"joint_predictors": list(preds)},
            )
            hyps = [h1, h0]
        elif len(preds) > 1:
            hyps: List[PredictiveHypothesis] = []
            for idx, other in enumerate(preds, start=1):
                h_code = f"HYP-{idx:02d}"
                h = PredictiveHypothesis(
                    id=h_code,
                    hypothesis_code=h_code,
                    claim=f"{metric} and {other} are materially associated.",
                    mechanism=f"A non-trivial monotonic or linear relationship exists between {metric} and {other} in the observed data.",
                    predicted_observables_if_true=[
                        f"Pearson (and/or Spearman) correlation coefficient between {metric} and {other} is substantially different from zero.",
                        f"The relationship is statistically significant (p < 0.05) given the observed sample size.",
                    ],
                    predicted_observables_if_false=[
                        f"Correlation coefficient between {metric} and {other} is close to zero.",
                    ],
                    falsification_criteria=f"Observed |r| < 0.10 or p-value >= 0.05 for the association between {metric} and {other}.",
                    required_assumptions=[f"Data completeness for {metric} and {other}", "Consistent reporting definitions"],
                    prior_probability=1.0,
                    posterior_probability=1.0,
                    target_metric=metric,
                    target_dimension="",
                    secondary_metric=other,
                    is_counter_hypothesis=False,
                    claim_type="ASSOCIATION",
                )
                hyps.append(h)
        else:
            other = preds[0] if preds else ""
            h1 = PredictiveHypothesis(
                id="HYP-01", hypothesis_code="HYP-01",
                claim=f"{metric} and {other} are materially associated.",
                mechanism=f"A non-trivial monotonic or linear relationship exists between {metric} and {other} in the observed data.",
                predicted_observables_if_true=[
                    f"Pearson (and/or Spearman) correlation coefficient between {metric} and {other} is substantially different from zero.",
                    f"The relationship is statistically significant (p < 0.05) given the observed sample size.",
                ],
                predicted_observables_if_false=[
                    f"Correlation coefficient between {metric} and {other} is close to zero.",
                ],
                falsification_criteria=f"Observed |r| < 0.10 or p-value >= 0.05 for the association between {metric} and {other}.",
                required_assumptions=[f"Data completeness for {metric} and {other}", "Consistent reporting definitions"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=metric,
                target_dimension="",
                secondary_metric=other,
                is_counter_hypothesis=False,
                claim_type="ASSOCIATION",
            )
            h0 = PredictiveHypothesis(
                id="HYP-02", hypothesis_code="HYP-02",
                claim=f"{metric} and {other} are approximately independent (no material association).",
                mechanism=f"Any apparent relationship between {metric} and {other} is attributable to sampling noise.",
                predicted_observables_if_true=[
                    f"Correlation coefficient between {metric} and {other} is close to zero.",
                    f"p-value for the association is not statistically significant (p >= 0.05).",
                ],
                predicted_observables_if_false=[
                    f"Correlation coefficient is substantially different from zero and statistically significant.",
                ],
                falsification_criteria=f"Observed |r| >= 0.30 with p-value < 0.05 for the association between {metric} and {other}.",
                required_assumptions=["No material association exists absent evidence otherwise"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=metric,
                target_dimension="",
                secondary_metric=other,
                is_counter_hypothesis=True,
                claim_type="ASSOCIATION",
            )
            hyps = [h1, h0]
        for h in hyps:
            h.direction = semantic.direction_hint
            stamp_semantic_identity(h)
            h.consolidation_log.append({"event": "created", "prior_source": "UNIFORM_STRUCTURAL_PRIOR", "hypothesis_code": h.hypothesis_code, "claim": h.claim})
        return hyps

    @staticmethod
    def _synthesize_forecast_hypotheses(
        semantic: SemanticResolution,
        question: str,
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
    ) -> List[PredictiveHypothesis]:
        """DEFECT-005 (section 5): real FORECAST hypothesis pair.

        H1: observed temporal structure contains predictive signal.
        H0: series is approximately stationary/noise-dominated.

        Final claim type is PREDICTION, never presented as settled fact.
        """
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic, decision=None)
        metric = _target_metric_via_canonical(semantic, canonical)
        time_col = canonical.time_variable()

        h1 = PredictiveHypothesis(
            id="HYP-01", hypothesis_code="HYP-01",
            claim=f"Observed temporal structure in {metric} over {time_col} contains predictive signal.",
            mechanism=f"A statistically significant trend/slope in {metric} over {time_col} exists and out-of-sample backtest error improves materially on a naive baseline.",
            predicted_observables_if_true=[
                f"Regression of {metric} on {time_col} has a statistically significant non-zero slope (or strong R^2).",
                f"A chronological holdout backtest shows lower forecast error than a naive (last-value or mean) baseline.",
            ],
            predicted_observables_if_false=[
                f"Trend/slope of {metric} over {time_col} is statistically indistinguishable from zero.",
            ],
            falsification_criteria=f"Backtest forecast error is not better than the naive baseline, or the trend slope is not statistically significant.",
            required_assumptions=[f"Chronological ordering of {time_col} is reliable", "No unrecorded structural breaks contaminate the holdout window"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=metric,
            target_dimension=time_col,
            is_counter_hypothesis=False,
            claim_type="PREDICTION",
        )
        h0 = PredictiveHypothesis(
            id="HYP-02", hypothesis_code="HYP-02",
            claim=f"{metric} is approximately stationary/noise-dominated over {time_col} (no exploitable predictive structure).",
            mechanism=f"Period-to-period variation in {metric} is not distinguishable from noise; a naive baseline forecasts as well as any fitted trend.",
            predicted_observables_if_true=[
                f"Regression slope of {metric} on {time_col} is not statistically significant.",
                f"A naive baseline backtest performs comparably to or better than a fitted trend.",
            ],
            predicted_observables_if_false=[
                f"A fitted trend/model materially outperforms the naive baseline on held-out data.",
            ],
            falsification_criteria=f"A fitted trend materially and significantly outperforms the naive baseline on the chronological holdout.",
            required_assumptions=["No exploitable temporal structure exists absent evidence otherwise"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=metric,
            target_dimension=time_col,
            is_counter_hypothesis=True,
            claim_type="PREDICTION",
        )
        hyps = [h1, h0]
        for h in hyps:
            h.direction = semantic.direction_hint
            stamp_semantic_identity(h)
            h.consolidation_log.append({"event": "created", "prior_source": "UNIFORM_STRUCTURAL_PRIOR", "hypothesis_code": h.hypothesis_code, "claim": h.claim})
        return hyps

    @staticmethod
    def _synthesize_segmentation_hypotheses(
        semantic: SemanticResolution,
        question: str,
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
    ) -> List[PredictiveHypothesis]:
        """DEFECT-005 (section 7): real SEGMENTATION hypothesis pair,
        phrased as heterogeneity-across-segments (not raw dominant-share)
        so ANOVA-style evidence maps onto it directly.

        Final claim type is OBSERVATION/ASSOCIATION (heterogeneity is
        described, not a causal mechanism).
        """
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic)
        metric = _target_metric_via_canonical(semantic, canonical)
        dim, dim_status, dim_candidates = _dimension_from_canonical(canonical)
        if dim_status == "AMBIGUOUS":
            dim_clause = f"across an AMBIGUOUS grouping dimension (candidates: {', '.join(dim_candidates)}; none selected)"
        elif dim_status == "UNRESOLVED":
            dim_clause = "across an UNRESOLVED grouping dimension (no candidate categorical column identified)"
        else:
            dim_clause = f"across {dim}"

        h1 = PredictiveHypothesis(
            id="HYP-01", hypothesis_code="HYP-01",
            claim=f"Outcomes in {metric} are materially heterogeneous {dim_clause}.",
            mechanism=f"Between-segment variance in {metric} {dim_clause} is large relative to within-segment variance.",
            predicted_observables_if_true=[
                f"ANOVA eta-squared {dim_clause} for {metric} is >= 15%.",
                f"At least one segment's mean {metric} differs materially from the overall mean, with adequate subgroup sample size.",
            ],
            predicted_observables_if_false=[
                f"Segment means for {metric} {dim_clause} are statistically indistinguishable.",
            ],
            falsification_criteria=f"Empirical ANOVA eta-squared {dim_clause} is less than 5.0%, or no subgroup has adequate sample size.",
            required_assumptions=[f"Data completeness for {dim}" if dim else "Data completeness", "Consistent segment definitions"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=metric,
            target_dimension=dim,
            is_counter_hypothesis=False,
            claim_type="OBSERVATION",
        )
        h0 = PredictiveHypothesis(
            id="HYP-02", hypothesis_code="HYP-02",
            claim=f"Segment differences in {metric} {dim_clause} are small relative to within-segment variation.",
            mechanism=f"Any apparent difference {dim_clause} is within the range expected from within-segment noise.",
            predicted_observables_if_true=[
                f"ANOVA eta-squared {dim_clause} for {metric} is < 5%.",
            ],
            predicted_observables_if_false=[
                f"At least one segment shows a materially and significantly different mean {metric}.",
            ],
            falsification_criteria=f"Empirical ANOVA eta-squared {dim_clause} is >= 15% with adequate subgroup sample sizes.",
            required_assumptions=["No material heterogeneity exists absent evidence otherwise"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=metric,
            target_dimension=dim,
            is_counter_hypothesis=True,
            claim_type="OBSERVATION",
        )
        hyps = [h1, h0]
        for h in hyps:
            h.direction = semantic.direction_hint
            h.dimension_resolution_status = dim_status
            h.dimension_candidates = dim_candidates
            stamp_semantic_identity(h)
            h.consolidation_log.append({"event": "created", "prior_source": "UNIFORM_STRUCTURAL_PRIOR", "hypothesis_code": h.hypothesis_code, "claim": h.claim})
        return hyps

    @staticmethod
    def _synthesize_churn_hypotheses(
        semantic: SemanticResolution,
        question: str,
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
    ) -> List[PredictiveHypothesis]:
        """DEFECT-015: Churn identifiability hypotheses.

        If churn outcome is absent from the dataset (e.g. customers.csv),
        generates an explicit unidentifiable hypothesis pair so the controller
        fails closed honestly instead of fabricating predictions.

        If churn outcome is present, generates competing hypotheses:
        - HYP-01: Segment association (observed difference across segments).
        - HYP-02: Systemic baseline uniformity (counter-hypothesis / sampling noise).
        - HYP-03: Confounding by cohort/tenure (Simpson's paradox counter-hypothesis).
        - HYP-04: Exposure/observation duration imbalance (if exposure column present).
        """
        # outcome_available/churn_event_status are NOT part of the v20-C2
        # migration's 9-field scope (churn_outcome_available and
        # churn_event_resolution_status are internal SemanticResolution
        # flags, not one of the fields CanonicalSemanticResolution
        # canonicalizes) -- left as direct reads deliberately.
        outcome_available = getattr(semantic, "churn_outcome_available", True) and getattr(semantic, "churn_event_col", None) is not None
        churn_event_status = getattr(semantic, "churn_event_resolution_status", "RESOLVED" if outcome_available else "UNRESOLVED")
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic)
        dim, dim_status, dim_candidates = _dimension_from_canonical(canonical)
        if dim_status == "AMBIGUOUS":
            dim_clause = f"across an AMBIGUOUS grouping dimension (candidates: {', '.join(dim_candidates)}; none selected)"
        elif dim_status == "UNRESOLVED":
            dim_clause = "across an UNRESOLVED grouping dimension (no candidate categorical column identified)"
        else:
            dim_clause = f"across {dim}"

        if not outcome_available:
            # target_metric here is deliberately still a direct
            # semantic.target_metric_col read, not canonical.outcome_column():
            # we are already inside the churn-outcome-unavailable branch, so
            # canonical.outcome represents the (unresolved) churn
            # resolution, not target_metric_col -- there is no canonical
            # field for "the raw target metric during a churn
            # investigation" (v20-C1 deliberately did not add one; see its
            # "no giant metadata bag" principle), so this stays a direct
            # read rather than misreading canonical.outcome_column() (which
            # would return None here) or canonical.outcome.value (also None).
            target_metric = semantic.target_metric_col or "churn_event"
            ambiguous_candidates = list(canonical.churn_event_resolution().candidates)
            if churn_event_status == "AMBIGUOUS" and ambiguous_candidates:
                unidentifiable_claim = (
                    f"Multiple equally-plausible churn/cancellation outcome columns are present in dataset "
                    f"'{semantic.primary_dataset_name}' (candidates: {', '.join(ambiguous_candidates)}); "
                    f"no single column can be selected without an explicit definition, so churn rate and "
                    f"risk differences are AMBIGUOUS rather than computable."
                )
                unidentifiable_mechanism = (
                    "The dataset records more than one candidate churn-like outcome column and no rule "
                    "distinguishes which one is authoritative; choosing one arbitrarily would fabricate "
                    "the outcome definition."
                )
            else:
                unidentifiable_claim = f"No churn/cancellation outcome column is recorded in dataset '{semantic.primary_dataset_name}' (churn rate and risk differences are unidentifiable)."
                unidentifiable_mechanism = "The dataset does not record whether or when individual entities churned; analytical estimands cannot be computed from absent outcome data."
            h1 = PredictiveHypothesis(
                id="HYP-01", hypothesis_code="HYP-01",
                claim=unidentifiable_claim,
                mechanism=unidentifiable_mechanism,
                predicted_observables_if_true=["Dataset schema lacks a single unambiguous identifiable churn, cancellation, or attrition outcome column."],
                predicted_observables_if_false=["A single, unambiguous, identifiable churn event column exists in the schema."],
                falsification_criteria="Exactly one identifiable churn event column is present and contains valid outcome data.",
                required_assumptions=["Accurate schema metadata"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=target_metric,
                target_dimension=dim,
                is_counter_hypothesis=False,
                claim_type="OBSERVATION",
            )
            h0 = PredictiveHypothesis(
                id="HYP-02", hypothesis_code="HYP-02",
                claim=f"Hypothetical churn variation could exist if an unambiguous outcome column were established.",
                mechanism="Unobserved outcome distribution.",
                predicted_observables_if_true=["Unobserved churn events."],
                predicted_observables_if_false=["Outcome data is absent or ambiguous."],
                falsification_criteria="Outcome data remains unrecorded or ambiguous.",
                required_assumptions=["Unobserved potential outcomes"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=target_metric,
                target_dimension=dim,
                is_counter_hypothesis=True,
                claim_type="OBSERVATION",
            )
            hyps = [h1, h0]
            for h in hyps:
                h.direction = semantic.direction_hint
                h.dimension_resolution_status = dim_status
                h.dimension_candidates = dim_candidates
                stamp_semantic_identity(h)
                h.consolidation_log.append({"event": "created", "prior_source": "UNIFORM_STRUCTURAL_PRIOR", "hypothesis_code": h.hypothesis_code, "claim": h.claim, "churn_event_resolution_status": churn_event_status})
            return hyps

        # event_col/confounders/exposure below are read through the
        # canonical churn/confounder/exposure resolutions (sections 8/10/11).
        # We are inside the outcome_available branch here, so
        # canonical.churn_event_resolution() is guaranteed RESOLVED and
        # churn-sourced -- there is no fallthrough ambiguity risk the way
        # there is for target_metric_col in the non-churn families.
        event_col = canonical.churn_event_resolution().value
        confounders = canonical.confounder_set().columns
        exposure = canonical.exposure_column()

        hyps: List[PredictiveHypothesis] = []

        # H1: Segment Association (Observation, never causal claim)
        h1 = PredictiveHypothesis(
            id="HYP-01", hypothesis_code="HYP-01",
            claim=f"Observed churn rate differs materially {dim_clause}.",
            mechanism=f"Differences in customer characteristics or behaviors across {dim} are associated with differential churn rates.",
            predicted_observables_if_true=[
                f"Crude churn rate differs statistically significantly {dim_clause} (chi-square p < 0.05).",
                f"Difference persists after controlling for available confounders.",
            ],
            predicted_observables_if_false=[
                f"Churn rate is approximately uniform {dim_clause}.",
            ],
            falsification_criteria=f"No statistically significant difference in churn {dim_clause} (p >= 0.05), or effect collapses under stratification.",
            required_assumptions=[f"Outcome data completeness for {event_col}", f"Consistent definition of {dim}"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=event_col,
            target_dimension=dim,
            is_counter_hypothesis=False,
            claim_type="OBSERVATION",
        )
        hyps.append(h1)

        # H2: Systemic Baseline Uniformity
        h2 = PredictiveHypothesis(
            id="HYP-02", hypothesis_code="HYP-02",
            claim=f"Churn occurs approximately uniformly {dim_clause} (any observed difference is attributable to sampling noise).",
            mechanism=f"Baseline population hazard without genuine category-level divergence {dim_clause}.",
            predicted_observables_if_true=[
                f"Crude churn rates {dim_clause} do not differ significantly (p >= 0.05).",
            ],
            predicted_observables_if_false=[
                f"Statistically significant difference in churn rates {dim_clause}.",
            ],
            falsification_criteria=f"Statistically significant association between {dim} and {event_col} (p < 0.05).",
            required_assumptions=["Baseline uniformity absent evidence of divergence"],
            prior_probability=1.0,
            posterior_probability=1.0,
            target_metric=event_col,
            target_dimension=dim,
            is_counter_hypothesis=True,
            claim_type="OBSERVATION",
        )
        hyps.append(h2)

        # H3+: Confounding / Composition Shift (Simpson's paradox).  Preserve
        # each resolved confounder as a distinct competing hypothesis so that
        # multiple explanations cannot disappear from the evidence graph.
        for idx, conf_col in enumerate(confounders, start=3):
            hcode = f"HYP-{idx:02d}"
            h_conf = PredictiveHypothesis(
                id=hcode, hypothesis_code=hcode,
                claim=f"Apparent churn differences {dim_clause} are confounded or explained by {conf_col} composition (Simpson's paradox).",
                mechanism=f"Subgroup composition shifts in {conf_col} mediate or explain away the aggregate difference in churn {dim_clause}.",
                predicted_observables_if_true=[
                    f"Stratifying by {conf_col} eliminates or substantially reduces the difference between {dim} groups.",
                ],
                predicted_observables_if_false=[
                    f"Differences {dim_clause} persist robustly within strata of {conf_col}.",
                ],
                falsification_criteria=f"Within-stratum churn differences {dim_clause} remain consistent with the aggregate difference.",
                required_assumptions=[f"Observational coverage of {conf_col}"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=event_col,
                target_dimension=conf_col,
                is_counter_hypothesis=True,
                claim_type="OBSERVATION",
            )
            hyps.append(h_conf)

        # H4: Exposure Duration Imbalance (if exposure column present)
        if exposure:
            h_exp = PredictiveHypothesis(
                id=f"HYP-0{len(hyps)+1}", hypothesis_code=f"HYP-0{len(hyps)+1}",
                claim=f"Differences in crude churn rate {dim_clause} are driven by unequal exposure/observation duration ({exposure}).",
                mechanism=f"Groups with longer observation windows accumulate more churn events; person-time adjusted rates converge.",
                predicted_observables_if_true=[
                    f"Person-time adjusted hazard rates are approximately equal {dim_clause} despite crude rate divergence.",
                ],
                predicted_observables_if_false=[
                    f"Person-time adjusted hazard rates remain materially different {dim_clause}.",
                ],
                falsification_criteria=f"Exposure-adjusted rates maintain statistically significant divergence {dim_clause}.",
                required_assumptions=[f"Accurate recording of observation time in {exposure}"],
                prior_probability=1.0,
                posterior_probability=1.0,
                target_metric=event_col,
                target_dimension=exposure,
                is_counter_hypothesis=True,
                claim_type="OBSERVATION",
            )
            hyps.append(h_exp)

        # Normalize priors
        tot_prior = sum(h.prior_probability for h in hyps)
        for h in hyps:
            h.prior_probability /= tot_prior
            h.posterior_probability = h.prior_probability
            h.direction = semantic.direction_hint
            # h1/h2 use the primary grouping dimension resolved above; h3+
            # (confounders) and h4 (exposure) each carry their own,
            # independently-resolved real column as target_dimension and
            # must not be overwritten with the primary dimension's status.
            if h.hypothesis_code in ("HYP-01", "HYP-02"):
                h.dimension_resolution_status = dim_status
                h.dimension_candidates = dim_candidates
            stamp_semantic_identity(h)
            h.consolidation_log.append({"event": "created", "prior_source": "UNIFORM_STRUCTURAL_PRIOR", "hypothesis_code": h.hypothesis_code, "claim": h.claim})

        return hyps

    @staticmethod
    def synthesize_emergent_hypotheses(
        discovered_patterns: List[Dict[str, Any]],
        semantic: SemanticResolution,
        existing_hyps: List[PredictiveHypothesis],
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
    ) -> List[PredictiveHypothesis]:
        """Synthesizes genuine emergent explanatory hypotheses from observed evidence patterns."""
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic)
        available_categorical_candidates = canonical.available_categorical_candidates
        emergent: List[PredictiveHypothesis] = []
        existing_codes = {h.hypothesis_code for h in existing_hyps}
        next_idx = len(existing_hyps) + 1

        for pat in discovered_patterns:
            pat_type = str(pat.get("type", "concentration")).lower()
            seg_val = pat.get("segment_value") or pat.get("top_value") or pat.get("dominant_value")
            raw_dim = pat.get("dimension") or pat.get("dimension_col")
            if raw_dim and raw_dim in available_categorical_candidates:
                dim = raw_dim
            else:
                dim = canonical.dimension.value or (available_categorical_candidates[0] if len(available_categorical_candidates) == 1 else "")
            metric = pat.get("metric") or _target_metric_via_canonical(semantic, canonical)
            share = pat.get("observed_value") or pat.get("top_share_pct", 0.0)
            src_exp = pat.get("source_experiment") or pat.get("source_experiment_id", "")

            code = f"HYP-{next_idx:02d}"
            while code in existing_codes:
                next_idx += 1
                code = f"HYP-{next_idx:02d}"

            if pat_type in ("concentration", "concentration_pattern"):
                claim = (
                    f"Concentration in segment '{seg_val}' of {dim} is the primary driver of {metric} variance ({share:.1f}% share)."
                    if seg_val else
                    f"Concentration across {dim} is disproportionately associated with {metric} variance."
                )
                mechanism = f"Specific localized concentration in '{seg_val}' ({dim}) driving aggregate {metric}." if seg_val else f"High categorical concentration in {dim}."
                pred_true = [f"Subgroup '{seg_val}' exhibits > 50% contribution to net change." if seg_val else f"Top categories account for > 60% of {metric}."]
                pred_false = [f"Variance in '{seg_val}' is proportional to baseline volume." if seg_val else f"Uniform spread across {dim}."]
                falsify = f"Segment '{seg_val}' contribution drops below 35% in controlled tests." if seg_val else f"Top share < 35%."
            elif pat_type in ("segment_difference", "segment_difference_pattern"):
                claim = f"Significant variance disparity exists across partitions of {dim} for {metric} (CV = {share/100:.2f})."
                mechanism = f"Categorical partitioning across {dim} reveals structural heterogeneity in {metric}."
                pred_true = [f"ANOVA eta-squared across {dim} is statistically significant (> 15%).", f"Subgroup differences in {metric} persist across partitions."]
                pred_false = [f"Between-group variance across {dim} is indistinguishable from noise (eta^2 < 5%)."]
                falsify = f"Between-group sum of squares eta-squared drops below 5%."
            elif pat_type in ("temporal_change", "temporal_change_pattern"):
                claim = f"A temporal shift or regime transition in {metric} occurred at '{seg_val}' ({dim})."
                mechanism = f"Time-series regime acceleration at {seg_val} driving aggregate shift in {metric}."
                pred_true = [f"Pre/post mean difference across temporal boundary '{seg_val}' is significant.", f"Post-transition variance is sustained."]
                pred_false = [f"Temporal step delta reverts to historical mean within baseline noise window."]
                falsify = f"Pre-vs-post period difference in {metric} is statistically insignificant (p > 0.05)."
            elif pat_type in ("anomaly", "anomaly_pattern"):
                claim = f"Anomalous localized outlier spike observed in {metric} at '{seg_val}' ({dim})."
                mechanism = f"Extreme statistical outlier in {seg_val} accounts for transient movement in {metric}."
                pred_true = [f"Trimming {seg_val} removes majority of variance in {metric}.", f"Z-score of {seg_val} exceeds 3.0 standard deviations."]
                pred_false = [f"Distribution spread is identical with and without {seg_val}."]
                falsify = f"Z-score of {seg_val} is less than 2.0 standard deviations."
            else:
                claim = f"Discovered empirical pattern ({pat_type}) in {dim} is associated with {metric} variance."
                mechanism = f"Empirical regularity in {dim} explains variation in {metric}."
                pred_true = [f"Observed pattern in {dim} replicates in targeted validation."]
                pred_false = [f"Effect attenuates completely in controlled secondary partition."]
                falsify = f"Validation test rejects association between {dim} and {metric}."

            h = PredictiveHypothesis(
                id=code,
                hypothesis_code=code,
                claim=claim,
                mechanism=mechanism,
                predicted_observables_if_true=pred_true,
                predicted_observables_if_false=pred_false,
                falsification_criteria=falsify,
                required_assumptions=[f"Reliability of reporting for {dim}"],
                prior_probability=1.0,
                posterior_probability=1.0,
                belief_state="PROPOSED",
                target_metric=metric,
                target_dimension=dim,
                target_value=seg_val,
                source_evidence=[src_exp] if src_exp else [],
                generated_reason=f"Emergent hypothesis discovered from {pat_type} pattern on '{seg_val}' ({share:.1f}%).",
            )
            h.direction = semantic.direction_hint
            stamp_semantic_identity(h)
            h.consolidation_log.append({
                "event": "created",
                "prior_source": "UNIFORM_STRUCTURAL_PRIOR",
                "hypothesis_code": h.hypothesis_code,
                "claim": h.claim,
                "pattern_type": pat_type,
                "source_experiment_id": src_exp,
            })
            emergent.append(h)
            existing_codes.add(code)
            next_idx += 1

        # Same-batch consolidation: several detectors firing in the same
        # round can independently propose hypotheses that are the same
        # underlying claim (e.g. a concentration detector and a
        # segment-difference detector both pointing at Region B / revenue /
        # March). Collapse those into one canonical hypothesis with merged
        # evidence contributions before this batch ever reaches the state
        # manager, per the single authoritative consolidation path.
        from packages.analytics_core.src.intelligence.hypothesis_consolidation import HypothesisConsolidator
        return HypothesisConsolidator.consolidate_batch(emergent)

    @staticmethod
    def synthesize_alternative_hypotheses(
        refuted_hypothesis: PredictiveHypothesis,
        semantic: SemanticResolution,
        existing_hyps: List[PredictiveHypothesis],
        canonical_semantics: Optional[CanonicalSemanticResolution] = None,
    ) -> List[PredictiveHypothesis]:
        """Synthesizes alternative hypotheses when an existing hypothesis is refuted."""
        canonical = canonical_semantics if canonical_semantics is not None else _build_canonical_semantics(semantic)
        alternatives: List[PredictiveHypothesis] = []
        dim = refuted_hypothesis.target_dimension
        metric = refuted_hypothesis.target_metric or _target_metric_via_canonical(semantic, canonical)
        available_dims = [d for d in canonical.available_categorical_candidates if d != dim]

        existing_codes = {h.hypothesis_code for h in existing_hyps}
        next_idx = len(existing_hyps) + 1

        for alt_dim in available_dims:
            code = f"HYP-{next_idx:02d}"
            while code in existing_codes:
                next_idx += 1
                code = f"HYP-{next_idx:02d}"

            alt_h = PredictiveHypothesis(
                id=code,
                hypothesis_code=code,
                claim=f"Alternative dimension {alt_dim} explains {metric} variance following refutation of {refuted_hypothesis.hypothesis_code}.",
                mechanism=f"Primary shift originates in {alt_dim} rather than {dim}.",
                predicted_observables_if_true=[
                    f"Significant ANOVA eta^2 across {alt_dim}.",
                ],
                predicted_observables_if_false=[
                    f"Uniform distribution across {alt_dim}.",
                ],
                falsification_criteria=f"ANOVA eta^2 for {alt_dim} < 5%.",
                required_assumptions=[f"Data validity for {alt_dim}"],
                prior_probability=1.0,
                posterior_probability=1.0,
                belief_state="PROPOSED",
                target_metric=metric,
                target_dimension=alt_dim,
                parent_hypotheses=[refuted_hypothesis.hypothesis_code],
                generated_reason=f"Synthesized alternative explanation after refutation of {refuted_hypothesis.hypothesis_code}.",
            )
            alt_h.direction = semantic.direction_hint
            stamp_semantic_identity(alt_h)
            alt_h.consolidation_log.append({
                "event": "created",
                "hypothesis_code": alt_h.hypothesis_code,
                "claim": alt_h.claim,
            })
            alternatives.append(alt_h)
            existing_codes.add(code)
            next_idx += 1

        from packages.analytics_core.src.intelligence.hypothesis_consolidation import HypothesisConsolidator
        return HypothesisConsolidator.consolidate_batch(alternatives)
