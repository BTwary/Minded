"""
IntentEngine: Parses natural language analytical questions into formal investigation intent.
Extracts target outcomes, candidate dimensions, comparison operations, and uncertainty bounds.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Q6.1 (v20-C3.1a follow-up): the CORRELATION keyword check below used to be
# a literal-substring list (`w in q_lower for w in [...]`). Substring
# matching against a fixed word list means every inflection of a root word
# has to be spelled out individually, and it is easy to miss one -- the
# audited gap was "association" being listed but not its "-ed" inflection
# "associated", so "Is price associated with sales?" / "Are sales and price
# associated?" fell through every branch to GENERAL and never reached the
# CORRELATION method family at all (see
# tests/independent_release/test_c3_1_q6_real_compiler_symmetry.py,
# TestQ6RealCompilerPathDocumentedLimitation, C3.1's documented limitation).
#
# Fixed by switching the single-root words (correlat-, relationship-,
# coupl-, depend-, associat-, impact-, affect-, link-, influenc-, connect-)
# to `\b<stem>\w*\b` regex matching -- the same inflection-robust approach
# already used elsewhere in this codebase for the identical problem (see
# `_ASSOC` in intelligence/universal_question_planner.py and the
# `associat\w*` pattern in intelligence/question_intelligence.py and
# intelligence/nl_semantic_interpreter.py). This one regex subsumes what
# used to be separate literal entries for "affect"/"affects"/"affected",
# "linked"/"linkage", and "connection"/"connected" -- those are now all
# matched by their shared stem instead of needing to be listed individually
# (and along with them, other inflections that were never listed at all,
# e.g. "correlates", "depends", "impacted", "influencing"). Multi-word
# phrases ("tied to", "move(s) together", "effect of") are not single-root
# words and are kept as literal alternatives.
#
# v20-C4.2.2c.1: widened `relationship\w*` to the bare `relat\w*` stem.
# C4.2.2c made UniversalQuestionCompiler._ASSOC a direct alias of this
# regex (see universal_question_planner.py) so that both consumers read
# from exactly one place. That swap was scope-checked beforehand and found
# one real, intentional narrowing: the compiler's OLD, separately
# maintained `_ASSOC` matched a bare `relat\w*` stem (covering "relate" /
# "related" without requiring the "-ship" suffix), which this regex did
# not. That gap was flagged explicitly in
# AAOS_V20C4_2_2C_SHARED_VOCABULARY_AND_ADMISSIBILITY_INTEGRITY.md as a
# deliberate, checked trade-off rather than an oversight, with a
# regression-restore left as the very next step (C4.2.2c.1) rather than
# being silently reintroduced inside the same pass as the alias swap.
# `relat\w*` matching is a strict superset of `relationship\w*` (it also
# covers "relate", "relates", "related", "relation", "relational", etc.,
# in addition to every "relationship*" inflection already covered), so
# this is a pure widening: nothing that matched before stops matching.
_CORRELATION_KEYWORD_RE = re.compile(
    r"\b(correlat\w*|relat\w*|coupl\w*|depend\w*|associat\w*|impact\w*|"
    r"affect\w*|link\w*|influenc\w*|connect\w*)\b"
    r"|\b(tied to|moves? together|effect of)\b",
    re.IGNORECASE,
)


@dataclass
class InvestigationIntent:
    """Formal structured intent representation extracted from a business question."""
    raw_question: str
    intent_type: str  # ROOT_CAUSE, CORRELATION, FORECAST, SEGMENTATION, PERFORMANCE, GENERAL
    comparison_type: str = "GENERAL_INVESTIGATION"  # PERIOD_OVER_PERIOD, SEGMENT_CONTRAST, ANOMALY_ROOT_CAUSE, CORRELATION_SEARCH, DISTRIBUTION_SHIFT
    target_metric_hint: Optional[str] = None
    dimension_hint: Optional[str] = None
    comparison_period_hint: Optional[str] = None
    requested_operations: List[str] = field(default_factory=list)
    candidate_dimensions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    business_objective: str = ""
    uncertainty_score: float = 0.0  # 0.0 (certain) to 1.0 (ambiguous)
    assumptions: List[str] = field(default_factory=list)
    # Direction of the change the question is actually asking about (e.g.
    # "why did revenue FALL" -> decrease). This is resolved ONCE here, from
    # the question itself, and threaded through SemanticResolution to every
    # hypothesis in the investigation -- it is deliberately not re-derived
    # per-hypothesis from each hypothesis's own generated claim text, since
    # hypothesis claim templates don't consistently restate it and doing so
    # would make "direction" an unreliable, wording-dependent identity axis.
    direction_hint: str = "unspecified"


class IntentEngine:
    """Extracts analytical objectives and target candidates without hardcoded business assumptions."""

    @staticmethod
    def parse_intent(question: str, available_columns: Optional[List[str]] = None) -> InvestigationIntent:
        q_lower = question.lower()
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower)
        
        # 1. Intent Classification
        #
        # BUGFIX (DEFECT-005): CORRELATION and FORECAST are checked BEFORE
        # ROOT_CAUSE now. Previously ROOT_CAUSE was checked first, and its
        # keyword list includes generic words ("driver", "cause") that
        # legitimately co-occur in correlation/forecast phrasing too --
        # e.g. "What are the correlation drivers of revenue?" contains both
        # "correlation" (unambiguous) and "driver" (ambiguous, ROOT_CAUSE
        # list), and used to be misclassified as ROOT_CAUSE purely because
        # ROOT_CAUSE's elif branch ran first. CORRELATION's and FORECAST's
        # own keyword lists ("correlation", "correlated", "forecast",
        # "predict", "trajectory", etc.) are far more specific and rarely if
        # ever legitimately mean root-cause, so checking them first resolves
        # the ambiguity in favor of the more specific, less ambiguous signal.
        if _CORRELATION_KEYWORD_RE.search(q_lower):
            intent_type = "CORRELATION"
            comparison_type = "CORRELATION_SEARCH"
            ops = ["BIVARIATE_CORRELATION", "PARTIAL_CORRELATION", "CRAMERS_V"]
        elif any(w in q_lower for w in [
            "forecast", "predict", "trajectory", "future", "next month", "trend", "growth",
            # Expanded: other common ways of asking for a forward-looking
            # projection that the original list didn't recognize.
            "outlook", "projection", "going forward", "upcoming",
            "next quarter", "next year", "look like next", "headed",
        ]):
            intent_type = "FORECAST"
            comparison_type = "PERIOD_OVER_PERIOD"
            ops = ["TIME_SERIES_TREND", "GROWTH_RATE", "CUSUM_CHANGE_POINT"]
        elif any(w in q_lower for w in [
            "why", "drop", "fell", "fall", "decrease", "decline", "spike", "surge", "cause", "driver", "root cause", "down",
            # Expanded: "cause" as a bare word never matches its own common
            # inflections ("causing", "caused", "causes" don't contain the
            # substring "cause" -- e.g. "causing" is missing the trailing
            # "e"). Also added other everyday diagnostic phrasings
            # ("what happened to X", "what changed", "reason behind",
            # "account for", "dip", "jump", "miss(ed)") that real users
            # reach for at least as often as "why"/"drop"/"decline".
            "causing", "caused", "causes", "reason", "explain", "diagnose",
            "dip", "jump", "miss", "missed", "what happened", "what changed",
            "account for", "behind the",
        ]):
            intent_type = "ROOT_CAUSE"
            comparison_type = "ANOMALY_ROOT_CAUSE"
            ops = ["CONCENTRATION", "VARIANCE_DECOMPOSITION", "SEGMENT_CONTRAST", "ADVERSARIAL_SIMPSON"]
        elif any(w in q_lower for w in [
            "churn", "churned", "cancellation", "cancelling", "canceled", "cancelled", "attrition", "dropoff", "retention", "retained",
            # Expanded: "at risk of leaving" / "at risk" / "churn risk" are
            # everyday phrasings for the same retention question that never
            # mention the word "churn" itself.
            "at risk of leaving", "at risk", "churn risk",
        ]):
            intent_type = "CHURN"
            comparison_type = "SEGMENT_CONTRAST"
            ops = ["CRUDE_CHURN_RATE", "EXPOSURE_ADJUSTED_RATE", "STRATIFIED_CHURN_CHECK"]
        elif any(w in q_lower for w in [
            "segment", "rfm", "cluster", "cohort", "groups",
        ]):
            intent_type = "SEGMENTATION"
            comparison_type = "SEGMENT_CONTRAST"
            ops = ["SEGMENT_DECOMPOSITION", "PARETO_80_20", "ANOVA_ETA_SQUARED"]
        elif any(w in q_lower for w in [
            "performing", "performance", "top", "bottom", "ranking", "worst", "best", "share", "contribution",
            # Expanded: explicit ranking/comparison phrasings ("rank the
            # reps by...", "how does Q1 compare to Q2", "underperforms")
            # that name the operation the user wants without using any of
            # the words above.
            "rank", "ranked", "compare", "comparison", "compared to",
            "most", "least", "underperform", "outperform",
        ]):
            intent_type = "PERFORMANCE"
            comparison_type = "SEGMENT_CONTRAST"
            ops = ["RANKING", "SHARE_OF_TOTAL", "GINI_CONCENTRATION"]
        else:
            intent_type = "GENERAL"
            comparison_type = "GENERAL_INVESTIGATION"
            ops = ["DESCRIPTIVE_SUMMARY", "DISTRIBUTION_SCAN"]

        # 2. Extract column matches dynamically from available columns if provided.
        # Relation grammar has an explicit subject/object direction ONLY for
        # "depend(s) on", where the grammatical SUBJECT genuinely is the
        # outcome being explained ("Does revenue depend on price and
        # marketing_spend?" -- revenue, the subject, is the target; price
        # and marketing_spend, the object, are predictors). This must not
        # mistakenly choose the first predictor as the target.
        #
        # "affect(s)"/"influence(s)" are grammatically the OPPOSITE polarity
        # -- in "X affects/influences Y", the SUBJECT (X) is the cause/
        # predictor and the OBJECT (Y) is the outcome/target ("Does price
        # influence annual_sales?" -- price, the subject, is the predictor).
        # Treating them the same as "depend on" previously forced the
        # predictor into the target role for this verb group.
        #
        # Symmetric associative verbs ("associated with", "correlated
        # with", "related to", "relate to") carry no direction at all: "Is
        # price associated with annual_sales?" does not grammatically
        # privilege either side as the outcome.
        #
        # All three misclassifications collapsed to the same downstream
        # symptom (confirmed via live regression): forcing an unrelated
        # subject into target_metric_hint short-circuited SemanticEngine's
        # own alphabetically-stable bivariate tie-break in
        # resolve_schema_static, silently flipping which column was
        # treated as the requested predictor. Only "depend on"/"depends
        # on" sets target_metric_hint from the subject here; "affect(s)"/
        # "influence(s)" set it from the OBJECT instead; symmetric verbs
        # set neither and fall through to the generic scoring/tie-break
        # logic below.
        target_metric_hint = None
        dimension_hint = None
        candidate_dims: List[str] = []

        if available_columns:
            depend_match = re.search(
                r"\b(?:is|are|do|does|can|whether)\s+(.+?)\s+"
                r"(?:depend on|depends on)\s+"
                r"(.+?)(?=,|;|\bcontrolling for\b|\bgiven\b|\bafter accounting for\b|\bholding\b|\?|$)",
                q_lower,
                flags=re.IGNORECASE,
            )
            if depend_match:
                subject_text, object_text = depend_match.groups()
                subject_cols = [c for c in available_columns if c.lower() in subject_text or c.lower().replace("_", " ") in subject_text]
                if len(subject_cols) == 1:
                    target_metric_hint = subject_cols[0]
            else:
                affect_match = re.search(
                    r"\b(?:is|are|do|does|can|whether)\s+(.+?)\s+"
                    r"(?:influence|influences|affect|affects)\s+"
                    r"(.+?)(?=,|;|\bcontrolling for\b|\bgiven\b|\bafter accounting for\b|\bholding\b|\?|$)",
                    q_lower,
                    flags=re.IGNORECASE,
                )
                if affect_match:
                    subject_text, object_text = affect_match.groups()
                    object_cols = [c for c in available_columns if c.lower() in object_text or c.lower().replace("_", " ") in object_text]
                    if len(object_cols) == 1:
                        target_metric_hint = object_cols[0]

            metric_matches: List[str] = []
            for col in available_columns:
                c_clean = col.lower().replace("_", " ")
                if c_clean in q_lower or col.lower() in words:
                    if any(k in col.lower() for k in ["id", "code", "region", "country", "type", "category", "segment", "channel", "tier"]):
                        if not dimension_hint:
                            dimension_hint = col
                        candidate_dims.append(col)
                    else:
                        metric_matches.append(col)
            if not target_metric_hint:
                if intent_type == "CORRELATION" and len(metric_matches) >= 2:
                    # Two numeric metrics both explicitly named in a
                    # CORRELATION question ("Is price associated with
                    # annual_sales?") is the expected bivariate case, not
                    # one where this loop's incidental match order should
                    # crown a "target". Picking metric_matches[0] here
                    # silently forced whichever column this loop happened
                    # to see first into the target role (regression:
                    # target_metric_hint="price" for that exact question,
                    # which then short-circuited SemanticEngine's own
                    # alphabetically-stable bivariate tie-break in
                    # resolve_schema_static). Leave target_metric_hint
                    # unset here and let that tie-break decide.
                    pass
                elif metric_matches:
                    target_metric_hint = metric_matches[0]

        # 3. Uncertainty Scoring
        uncertainty = 0.1 if (target_metric_hint and dimension_hint) else (0.4 if target_metric_hint else 0.7)

        # 3b. Direction of the change being asked about. Resolved once from
        # the raw question text, deterministically, so every hypothesis
        # synthesized downstream can share the SAME direction rather than
        # each guessing independently from its own claim wording.
        decrease_kw = (
            "drop", "fell", "fall", "decreas", "declin", "down", "shrink", "lower", "reduc", "worsen",
            # Expanded to match the ROOT_CAUSE keyword additions above.
            "dip", "miss",
        )
        increase_kw = (
            "increas", "rise", "rose", "grow", "surge", "spike", "gain", "up", "higher", "improv",
            # Expanded to match the ROOT_CAUSE keyword additions above.
            "jump",
        )
        if any(w in q_lower for w in decrease_kw):
            direction_hint = "decrease"
        elif any(w in q_lower for w in increase_kw):
            direction_hint = "increase"
        else:
            direction_hint = "unspecified"

        # 4. Formulate Business Objective
        obj = f"Investigate {intent_type.lower().replace('_', ' ')} for {target_metric_hint or 'primary metrics'} across {dimension_hint or 'key business partitions'}."

        return InvestigationIntent(
            raw_question=question,
            intent_type=intent_type,
            comparison_type=comparison_type,
            target_metric_hint=target_metric_hint,
            dimension_hint=dimension_hint,
            direction_hint=direction_hint,
            requested_operations=ops,
            candidate_dimensions=candidate_dims,
            keywords=words,
            business_objective=obj,
            uncertainty_score=uncertainty,
            assumptions=[
                "Deterministic execution across available dataset columns.",
                "Hypothesis falsification prioritized over answer guessing.",
            ],
        )
