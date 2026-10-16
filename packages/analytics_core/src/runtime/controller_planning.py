"""Pure candidate-planning helpers used by InvestigationController."""
from typing import Any, Dict, List, Optional, Tuple

from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment
from packages.analytics_core.src.intelligence.predictive_hypothesis import resolve_group_dimension
from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver
from packages.schemas.src.semantic_role import ExperimentRole


def compute_task_compatible_candidates(
    candidates: List["CandidateExperiment"],
    task: str,
    semantic: "SemanticResolution",
) -> Tuple[List["CandidateExperiment"], Optional[Dict[str, Any]]]:
    """P0-B (controller hardening): the single authoritative rule for
    filtering candidate experiments down to the ones compatible with the
    selected analytical task.

    Returns (filtered_candidates, diagnostic). filtered_candidates is NEVER
    silently widened back to the full input when filtering eliminates
    everything -- an analytically incompatible candidate (e.g. a
    correlation candidate for a FORECAST task) must never be restored and
    executed merely because nothing matched. When filtering does eliminate
    every candidate, diagnostic is a NO_COMPATIBLE_EXPERIMENT_CANDIDATE
    payload the caller should record to provenance before replanning or
    stopping; otherwise diagnostic is None.

    Extracted as a standalone, independently-testable function (rather than
    the previous in-loop closure) so this invariant can be unit tested
    without running a full investigation.
    """
    if not candidates:
        return [], None
    keep_prefixes = None
    if task == "ASSOCIATION":
        # For a churn association question, do not spend the first
        # experiment on baseline uniformity/censoring diagnostics.
        # EXP-COND-* (the Simpson's-paradox conditional adversarial
        # candidate built by build_simpsons_conditional_candidate) is always
        # allowed once it exists -- its own construction is already gated on
        # a genuinely resolved primary dimension and a detected paradox, so
        # it must not be silently stripped out by this task-family filter
        # the way "filtered or candidates" used to accidentally preserve it.
        # EXP-CORR is the general (non-churn-specific) Pearson/Spearman
        # correlation candidate ExperimentSynthesizer.synthesize_correlation_experiment
        # emits for any ASSOCIATION hypothesis -- it is a legitimate member
        # of this task family and was previously only reachable through the
        # same "filtered or candidates" fallback bug fixed here; excluding
        # it would silently break non-churn association questions once that
        # bug was fixed, so it is included explicitly instead.
        keep_prefixes = ("EXP-CHURN-CRUDE", "EXP-CHURN-STRAT", "EXP-CHURN-EXPOSURE", "EXP-COND", "EXP-CORR")
        if not getattr(semantic, "churn_confounder_cols", None):
            keep_prefixes = ("EXP-CHURN-CRUDE", "EXP-CHURN-EXPOSURE", "EXP-COND", "EXP-CORR")
        if not getattr(semantic, "churn_exposure_col", None):
            keep_prefixes = tuple(x for x in keep_prefixes if x != "EXP-CHURN-EXPOSURE")
    elif task == "FORECAST":
        keep_prefixes = ("EXP-FORECAST",)
    elif task == "CAUSAL":
        keep_prefixes = ("EXP-CAUSAL", "EXP-IV", "EXP-DID", "EXP-RDD")
    if keep_prefixes is None:
        return candidates, None
    filtered = [c for c in candidates if any(str(c.code).startswith(prefix) for prefix in keep_prefixes)]
    if filtered:
        return filtered, None
    diagnostic = {
        "diagnostic": "NO_COMPATIBLE_EXPERIMENT_CANDIDATE",
        "analytical_task": task,
        "allowed_prefixes": list(keep_prefixes),
        "candidate_codes": [str(c.code) for c in candidates],
        "rejected_candidate_codes": [str(c.code) for c in candidates],
        "reason": (
            f"No candidate experiment code matched the compatible method-family "
            f"prefixes {list(keep_prefixes)} for task {task}; incompatible candidates "
            f"were not restored/executed."
        ),
        "replanning_status": "CALLER_MUST_REPLAN_OR_STOP",
    }
    return [], diagnostic

def build_simpsons_conditional_candidate(
    semantic: "SemanticResolution",
    conf_dim: str,
    leading_hypothesis_code: str,
    target_metric_col_label: Optional[str] = None,
) -> Tuple[Optional["CandidateExperiment"], Optional[Dict[str, Any]]]:
    """P0-B (controller hardening): construct the conditional adversarial
    (Simpson's-paradox) experiment ONLY from a genuinely resolved primary
    grouping dimension and resolved metric aggregation semantics.

    Returns (candidate_or_None, diagnostic_or_None). Exactly one of the two
    is non-None:
      - metric aggregation unresolved -> (None, {"status": "UNRESOLVED_METRIC_SEMANTICS", ...})
      - primary dimension UNRESOLVED  -> (None, {"status": "UNRESOLVED_PRIMARY_DIMENSION", ...})
      - primary dimension AMBIGUOUS   -> (None, {"status": "AMBIGUOUS_PRIMARY_DIMENSION", ...})
      - both resolved                 -> (CandidateExperiment(...), None)

    Never falls back to a literal placeholder dimension (e.g. "segment")
    and never silently picks one of several ambiguous candidates.
    Extracted as a standalone function so this invariant is independently
    unit-testable without running a full investigation loop.
    """
    primary_dim, primary_dim_status, primary_dim_candidates = resolve_group_dimension(semantic)
    _metric_def = semantic.metric_definition
    if _metric_def is None or getattr(_metric_def, "aggregation_type", None) is None:
        return None, {
            "confounding_dimension": conf_dim,
            "type": "simpsons_paradox",
            "status": "UNRESOLVED_METRIC_SEMANTICS",
            "reason": "Conditional adversarial test could not be constructed without a resolved metric aggregation.",
        }
    if primary_dim_status != "RESOLVED":
        return None, {
            "confounding_dimension": conf_dim,
            "type": "simpsons_paradox",
            "status": f"{primary_dim_status}_PRIMARY_DIMENSION",
            "candidate_dimensions": primary_dim_candidates,
            "reason": (
                "Conditional adversarial test could not be constructed: the primary "
                f"grouping dimension is {primary_dim_status} (candidates: "
                f"{primary_dim_candidates}); the controller does not choose one "
                "implicitly and did not execute a conditional grouping query."
            ),
        }
    _metric_expr = MetricSemanticsResolver.sql_aggregation_expression(_metric_def, "total_metric")
    _agg_str = _metric_def.aggregation_type.value.upper()
    candidate = CandidateExperiment(
        code=f"EXP-COND-{conf_dim.upper()}",
        target_hypothesis_code=leading_hypothesis_code,
        hypothesis_ids=[leading_hypothesis_code],
        tool_name="duckdb_sql",
        # Session 8 (DEFECT-023): the multi-row-result guard in execution_provider
        # rejects any grouped result without an ORDER BY, so this experiment
        # failed on every run (invisibly to the analyst). Same deterministic
        # ordering the synthesizer's EXP-COND variant already uses.
        query_sql=f"SELECT {primary_dim}, {conf_dim}, {_metric_expr} FROM data_table GROUP BY {primary_dim}, {conf_dim} ORDER BY total_metric DESC, {primary_dim}, {conf_dim}",
        description=f"Conditional test on {conf_dim} to address adversarial Simpson's paradox ({target_metric_col_label or semantic.target_metric_col} aggregated as {_agg_str}).",
        aggregation_type=_agg_str,
        discriminating_power=0.99,
        discrimination_value=0.99,
        estimated_cost=1.4,
        reliability_weight=0.96,
        decision_relevance=0.95,
        target_dimension=f"{primary_dim}_{conf_dim}",
        target_metric=semantic.target_metric_col,
        metrics=[semantic.target_metric_col],
        dimensions=[primary_dim, conf_dim],
        adversarial_value=0.95,
        # v20-B1: this is a Simpson's-paradox counter-hypothesis challenge by
        # construction -- always ADVERSARIAL in intent, regardless of
        # prediction linkage. Tag it explicitly here rather than relying on
        # the gate-time UNASSIGNED backfill (which had no way to distinguish
        # this from an ordinary PRIMARY/SUPPORTING candidate and would
        # mis-tag it as PRIMARY).
        experiment_role=ExperimentRole.ADVERSARIAL.value,
        target_uncertainty=f"Simpson's paradox challenge on {conf_dim}",
        provenance={"attack_type": "simpsons_paradox", "confounding_dimension": conf_dim},
    )
    return candidate, None

def _multiverse_pinned_aggregations(semantic: Any) -> Optional[List[str]]:
    """Aggregation the question's metric is defined with, so the robustness grid varies only
    choices that keep the estimand fixed (outlier trimming), not the estimand itself."""
    md = getattr(semantic, "metric_definition", None)
    raw_agg = getattr(md, "aggregation_type", "") or ""
    agg = str(getattr(raw_agg, "value", raw_agg)).upper()
    # Rates, ratios and proportions are means of per-row values; summing or taking the
    # median of a 0/1 flag answers a different question than the one asked.
    agg = {"RATE": "MEAN", "RATIO": "MEAN", "PROPORTION": "MEAN", "WEIGHTED_MEAN": "MEAN",
           "AVG": "MEAN", "AVERAGE": "MEAN"}.get(agg, agg)
    return [agg] if agg in ("SUM", "MEAN", "MEDIAN") else None
