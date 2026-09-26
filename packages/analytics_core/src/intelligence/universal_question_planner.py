"""Universal question -> analysis compiler for AA-OS.

This module is intentionally deterministic and auditable. It does not attempt to
"understand" a business problem by inventing context. Instead it separates:
- what the user explicitly asked,
- what can be resolved from the data/schema,
- what remains unknown,
- what claim class is admissible,
- what evidence is minimally sufficient,
- and what specialist engine should run first.

The planner is a policy compiler, not a language model. An LLM may propose a
richer interpretation elsewhere, but this contract is the local source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd

# v20-C4.2.2c: reuse IntentEngine's association-vocabulary regex directly,
# rather than maintaining a second, independently-drifting copy of it here.
# This module previously had its own `_ASSOC` pattern that was missing
# several inflections IntentEngine already recognized (`influenc\w*`,
# `impact\w*`, `coupl\w*`, and the multi-word phrases "tied to"/"effect
# of") -- meaning a question like "Does price influence annual_sales?"
# was classified as CORRELATION by IntentEngine (which feeds
# MethodSelectionEngine.decide()) but NOT as an ASSOCIATION task by this
# compiler (which feeds select_for_plan()), producing an UNRECOGNIZED_TASK
# result for a question decide() handled correctly (see
# AAOS_V20C4_2_2B_COMPILER_SEMANTIC_AUTHORITY_AUDIT.md, Finding 2).
# `intent.py` has no imports of its own (stdlib only), so importing from it
# here carries no circular-import risk.
from packages.analytics_core.src.engines.intent import _CORRELATION_KEYWORD_RE


TASKS = {
    "DATA_QUALITY",
    "DESCRIPTIVE",
    "COMPARISON",
    "ASSOCIATION",
    "DIAGNOSTIC",
    "FORECAST",
    "PREDICTION",
    "CAUSAL",
    "SEGMENTATION",
    "RECONCILIATION",
    "PRESCRIPTIVE",
    "GOVERNANCE",
    "GENERAL_EXPLORATION",
}

CLAIM_BY_TASK = {
    "DATA_QUALITY": "OBSERVATION",
    "DESCRIPTIVE": "OBSERVATION",
    "COMPARISON": "ASSOCIATION",
    "ASSOCIATION": "ASSOCIATION",
    "DIAGNOSTIC": "ASSOCIATION",
    "FORECAST": "PREDICTION",
    "PREDICTION": "PREDICTION",
    "CAUSAL": "CAUSAL_INFERENCE",
    "SEGMENTATION": "OBSERVATION",
    "RECONCILIATION": "OBSERVATION",
    "PRESCRIPTIVE": "SIMULATION",
    "GOVERNANCE": "OBSERVATION",
    "GENERAL_EXPLORATION": "OBSERVATION",
}


@dataclass
class BusinessContextContract:
    core_problem: Optional[str]
    audience: str
    success_definition: Optional[str]
    decision_context_known: bool
    unknown_context: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_problem": self.core_problem,
            "audience": self.audience,
            "success_definition": self.success_definition,
            "decision_context_known": self.decision_context_known,
            "unknown_context": list(self.unknown_context),
        }


@dataclass
class SemanticContract:
    referenced_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    explanatory_columns: List[str] = field(default_factory=list)
    grouping_columns: List[str] = field(default_factory=list)
    time_column: Optional[str] = None
    grain: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    resolution_confidence: float = 0.0
    semantic_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "referenced_columns": list(self.referenced_columns),
            "target_column": self.target_column,
            "explanatory_columns": list(self.explanatory_columns),
            "grouping_columns": list(self.grouping_columns),
            "time_column": self.time_column,
            "grain": self.grain,
            "notes": list(self.notes),
            "resolution_confidence": float(self.resolution_confidence),
            "semantic_evidence": dict(self.semantic_evidence),
        }


@dataclass
class EvidenceRequirement:
    code: str
    requirement: str
    mandatory: bool = True
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "requirement": self.requirement,
            "mandatory": self.mandatory,
            "rationale": self.rationale,
        }


@dataclass
class ExperimentPlan:
    code: str
    purpose: str
    method: str
    target_hypothesis: Optional[str] = None
    estimated_cost: float = 1.0
    expected_information_gain: Optional[float] = None
    prerequisites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "purpose": self.purpose,
            "method": self.method,
            "target_hypothesis": self.target_hypothesis,
            "estimated_cost": self.estimated_cost,
            "expected_information_gain": self.expected_information_gain,
            "prerequisites": list(self.prerequisites),
        }


@dataclass
class UniversalAnalysisPlan:
    task: str
    claim_type: str
    business_context: BusinessContextContract
    semantics: SemanticContract
    estimand: Dict[str, Any]
    required_evidence: List[EvidenceRequirement]
    hypotheses: List[Dict[str, Any]]
    experiments: List[ExperimentPlan]
    stopping_rule: str
    decision_status: str
    unresolved_questions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    visualization_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "claim_type": self.claim_type,
            "business_context": self.business_context.to_dict(),
            "semantics": self.semantics.to_dict(),
            "estimand": dict(self.estimand),
            "required_evidence": [e.to_dict() for e in self.required_evidence],
            "hypotheses": list(self.hypotheses),
            "experiments": [e.to_dict() for e in self.experiments],
            "stopping_rule": self.stopping_rule,
            "decision_status": self.decision_status,
            "unresolved_questions": list(self.unresolved_questions),
            "limitations": list(self.limitations),
            "visualization_candidates": list(self.visualization_candidates),
        }


class UniversalQuestionCompiler:
    """Compile a natural-language question into an auditable analytical contract."""

    _SEMANTIC_ALIASES = {
        "churn": "churn",
        "churned": "churn",
        "attrition": "churn",
        "retention": "churn",
        "customer churn": "churn",
        "plan": "plan_tier",
        "plan tier": "plan_tier",
        "subscription plan": "plan_tier",
        "subscription plans": "plan_tier",
        "sales": "sales",
        "revenue": "revenue",
        "orders": "orders",
        "aov": "aov",
        "average order value": "aov",
    }

    _CAUSAL = re.compile(r"\b(caus(e|ed|es|al)|treatment effect|intervention|effect of)\b", re.I)
    _RECON = re.compile(r"\b(discrep\w*|reconcil\w*|math(?:ematical)?|formula\w*|consistent\w*|mismatch\w*|ties?|add\s+up)\b", re.I)
    _DATA_QUALITY = re.compile(r"\b(clean|quality|missing|duplicate|invalid|outlier|anomal|inconsisten|corrupt|bias)\b", re.I)
    _SEGMENT = re.compile(r"\b(cluster|clustering|hidden(?:\s+\w+){0,3}\s+segments?|segments?\s+exist|group customers|discover groups|persona)\b", re.I)
    _PRESCRIPTIVE = re.compile(r"\b(what should we do|recommend|recommendation|action|optimi[sz]|best intervention|how should)\b", re.I)
    _PREDICT_RISK = re.compile(r"\b(which|who|identify|rank|score|probability).{0,60}(customer|user|account).{0,60}(churn|attrition|cancel)|\b(highest|top).{0,30}(churn|attrition|cancel).{0,30}(risk|probability)\b", re.I)
    # Grouped-rate comparison phrasing ("which customer segments have higher
    # churn rates") shares "which"/"customer"/"churn" vocabulary with
    # entity-level risk prediction ("which customers are likely to churn")
    # and previously matched _PREDICT_RISK first. It asks for a rate
    # compared across named groups, not a per-customer risk ranking, so it
    # must be routed as COMPARISON. Scoped narrowly to an explicit "rate"
    # noun plus a group noun plus comparison grammar to avoid reclassifying
    # genuine per-entity risk-ranking questions such as "rank customers by
    # churn rate" (no group noun) or "which customers have the highest
    # churn probability" (no "rate").
    _GROUP_NOUNS = re.compile(r"\b(segments?|tiers?|groups?|cohorts?|categories|category|regions?|plans?)\b", re.I)
    _FORECAST = re.compile(r"\b(forecast|project|future|next quarter|next year|will .* increase|will .* decrease|trend)\b", re.I)
    _COMPARE = re.compile(r"\b(compare|difference|higher|lower|highest|lowest|best|worst|top|bottom|more|less|versus|vs\.?|between)\b", re.I)
    # NOTE: each alternative below is a genuine word-stem prefix (e.g.
    # "correlat" is meant to catch "correlation"/"correlated"/"correlates"),
    # v20-C4.2.2c: was a separately-maintained pattern; now a direct alias
    # to IntentEngine's _CORRELATION_KEYWORD_RE (see the module-level
    # import comment above) so the two classifiers cannot drift apart
    # again. Any vocabulary change should be made once, in intent.py.
    _ASSOC = _CORRELATION_KEYWORD_RE
    _DIAGNOSTIC = re.compile(r"\b(why|driver|root cause|explain|what caused|contribut)\b", re.I)
    _DESCRIPTIVE = re.compile(r"\b(what is|how many|how much|average|median|mean|typical|usual|distribution|summary|summarize|baseline|describe)\b", re.I)
    _GOVERNANCE = re.compile(r"\b(privacy|pii|gdpr|ccpa|ethical|ethics|legal|compliance|fairness|disparate|harmful|harm|data protection)\b", re.I)

    @classmethod
    def compile(
        cls,
        question: str,
        *,
        semantic: Any,
        df: pd.DataFrame,
        quality_assessment: Any = None,
        ai_provider: Any = None,
    ) -> UniversalAnalysisPlan:
        q = " ".join((question or "").split())
        ql = q.lower()
        from packages.analytics_core.src.intelligence.question_intelligence import interpret_question
        from packages.analytics_core.src.intelligence.nl_semantic_interpreter import interpret_with_schema
        interpreted = interpret_question(q)
        semantic_proposal = interpret_with_schema(q, [str(c) for c in df.columns], ai_provider=ai_provider)
        task = cls._classify(ql)
        if task == "GENERAL_EXPLORATION" and semantic_proposal.task != "GENERAL_EXPLORATION":
            task = semantic_proposal.task
        # Broad language interpretation is a proposal layer only. The deterministic
        # classifier remains authoritative; use the interpreter to enrich unresolved
        # context and compound-question traceability without allowing it to execute SQL.
        business = cls._business_context(q, task)
        semantics = cls._semantic_contract(semantic, df, ql, semantic_proposal)
        hypotheses = cls._hypotheses(task, semantics, q)
        evidence = cls._evidence(task, semantics)
        experiments = cls._experiments(task, semantics)
        estimand = cls._estimand(task, semantics, quality_assessment)
        unresolved = cls._unresolved(task, semantics, business, quality_assessment)
        semantics.notes.extend([f"Universal NL interpreter: {n}" for n in interpreted.notes])
        semantics.notes.extend([f"Schema-grounded semantic proposal: {n}" for n in semantic_proposal.limitations])
        semantics.semantic_evidence["nl_proposal"] = {"source": semantic_proposal.source, "confidence": semantic_proposal.confidence, "task": semantic_proposal.task}
        if interpreted.compound_clauses:
            semantics.semantic_evidence["compound_clauses"] = list(interpreted.compound_clauses)
            if len(interpreted.compound_clauses) > 1 and "compound_question_requires_clause_preserving_analysis" not in unresolved:
                unresolved.append("compound_question_requires_clause_preserving_analysis")
        limitations = cls._limitations(task, semantics, quality_assessment)
        viz = cls._visualizations(task)
        decision_status = "RESOLVED" if not unresolved else "PARTIALLY_RESOLVED"
        stopping = cls._stopping_rule(task)
        return UniversalAnalysisPlan(
            task=task,
            claim_type=CLAIM_BY_TASK[task],
            business_context=business,
            semantics=semantics,
            estimand=estimand,
            required_evidence=evidence,
            hypotheses=hypotheses,
            experiments=experiments,
            stopping_rule=stopping,
            decision_status=decision_status,
            unresolved_questions=unresolved,
            limitations=limitations,
            visualization_candidates=viz,
        )

    @classmethod
    def _classify(cls, q: str) -> str:
        """Classify by analytical semantics, not single-token matches.

        Specific intent patterns win over generic words such as ``segment`` or
        ``trend``.  A comparison mentioning *customer segments* is still a
        COMPARISON question; a SEGMENTATION question asks to discover/construct
        groups.  Likewise, customer-level future risk is prediction, while a
        future aggregate trajectory is forecasting.
        """
        if cls._GOVERNANCE.search(q):
            return "GOVERNANCE"
        if cls._RECON.search(q):
            return "RECONCILIATION"
        if cls._DATA_QUALITY.search(q) and re.search(
            r"\b(clean|cleaning|repair|fix|remediat|data quality|bad data|invalid|malformed|missing|duplicate|anomal)", q, re.I
        ):
            return "DATA_QUALITY"
        if cls._CAUSAL.search(q):
            return "CAUSAL"
        if cls._PREDICT_RISK.search(q):
            grouped_rate_comparison = bool(
                re.search(r"\brates?\b", q, re.I)
                and cls._GROUP_NOUNS.search(q)
                and cls._COMPARE.search(q)
            )
            if not grouped_rate_comparison:
                return "PREDICTION"
            # else: fall through -- comparison grammar below takes this.
        if cls._FORECAST.search(q):
            return "FORECAST"
        if cls._PRESCRIPTIVE.search(q):
            return "PRESCRIPTIVE"
        # Performance/ranking language is comparative when the user asks which
        # observed entities or groups are doing better or worse. This prevents
        # questions such as "Which products are performing badly?" from falling
        # into generic exploration, whose preflight has no declared contrast
        # estimand.
        if re.search(r"\b(perform(?:ing|ance)|ranking|ranked|best|worst|top|bottom)\b", q, re.I):
            return "COMPARISON"
        # Segmentation requires an explicit discovery/group-construction
        # intent, not merely the word ``segment`` used as a grouping dimension.
        if (
            cls._SEGMENT.search(q)
            and re.search(r"\b(discover|find|identify|create|build|cluster|clustering|hidden|persona|segments?\s+exist)", q, re.I)
            and not re.search(r"\b(highest|lowest|best|worst|compare|difference|between|versus|vs\.?|average|rate|aov)", q, re.I)
        ):
            return "SEGMENTATION"
        if cls._DIAGNOSTIC.search(q):
            return "DIAGNOSTIC"
        # Retention/churn phrasing such as "different retention by plan" is
        # a group comparison, while "is retention associated with plan tier"
        # is association. Let explicit comparison grammar win.
        # "between" is ambiguous on its own: "difference between A and B" is
        # a group comparison, but "correlation between A and B" / "the
        # relationship between A and B" is a bivariate association -- it is
        # not itself comparison grammar, it is the standard preposition used
        # to introduce the *arguments* of a correlation/relationship phrase.
        # Treating any "between" as COMPARISON (its previous behavior via
        # `_COMPARE`) meant every correlation question phrased this way --
        # arguably the single most common phrasing for one -- was routed to
        # group-comparison analysis instead of association analysis, which
        # then failed closed for lack of a (nonexistent, not needed)
        # grouping column. An explicit *stronger* comparison signal (differs
        # from bare "between") still wins immediately, as before. Failing
        # that, an explicit association/correlation keyword now wins over a
        # bare "between". A "between" with neither signal still falls back
        # to COMPARISON, unchanged from before.
        strong_compare = re.search(
            r"\b(compare|difference|different|higher|lower|highest|lowest|best|worst|top|bottom|more|less|versus|vs\.?)\b",
            q, re.I,
        )
        if strong_compare:
            return "COMPARISON"
        if cls._ASSOC.search(q):
            return "ASSOCIATION"
        if re.search(r"\bbetween\b", q, re.I):
            return "COMPARISON"
        if cls._DESCRIPTIVE.search(q):
            return "DESCRIPTIVE"
        return "GENERAL_EXPLORATION"

    @staticmethod
    def _business_context(question: str, task: str) -> BusinessContextContract:
        audience = "UNSPECIFIED_DEFAULT_DUAL"
        q = question.lower()
        if re.search(r"\b(ceo|cfo|board|executive|leadership|management)\b", q):
            audience = "EXECUTIVE"
        elif re.search(r"\b(data scientist|statistician|engineer|analyst|technical|model)\b", q):
            audience = "TECHNICAL"
        success = None
        if task == "COMPARISON":
            success = "Identify the requested ranking/difference with uncertainty and sample support."
        elif task == "ASSOCIATION":
            success = "Estimate the requested association with effect size and uncertainty without causal overclaiming."
        elif task == "FORECAST":
            success = "Produce an out-of-sample validated forecast with uncertainty."
        elif task == "CAUSAL":
            success = "Establish the requested causal estimand only if identification assumptions are supported."
        elif task == "RECONCILIATION":
            success = "Quantify violations of the inferred mathematical identity and identify affected records."
        elif task == "SEGMENTATION":
            success = "Discover stable, interpretable segments and report cluster quality/stability."
        elif task == "DATA_QUALITY":
            success = "Identify material quality defects without silently altering the source."
        elif task == "GOVERNANCE":
            success = "Identify privacy, governance and foreseeable harm risks without claiming legal certification."
        return BusinessContextContract(
            core_problem=question.strip() or None,
            audience=audience,
            success_definition=success,
            decision_context_known=False,
            unknown_context=["Operational constraints", "Business cost of action", "Risk tolerance"] if task == "PRESCRIPTIVE" else [],
        )

    @staticmethod
    def _semantic_contract(semantic: Any, df: pd.DataFrame, ql: str, proposal: Any = None) -> SemanticContract:
        """Resolve question roles with explicit evidence and calibrated ambiguity.

        Resolution order:
        1. exact physical column reference
        2. normalized-name match
        3. high-similarity name match
        4. existing semantic binding

        We never silently choose among materially competing candidates.
        """
        cols = [str(c) for c in df.columns]
        referenced: List[str] = []
        exact_hits: Dict[str, str] = {}

        def normalize(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

        norm_cols = {c: normalize(c) for c in cols}
        token_text = normalize(ql)
        for c in cols:
            nc = norm_cols[c]
            if normalize(c) and (re.search(rf"(?<!\w){re.escape(nc)}(?!\w)", token_text) or nc in token_text):
                referenced.append(c)
                exact_hits[nc] = c

        # Resolve column phrases mentioned before/after analytical relation words.
        def resolve_phrase(phrase: str) -> Tuple[Optional[str], List[Tuple[str, float, str]]]:
            phrase_n = normalize(phrase)
            candidates: List[Tuple[str, float, str]] = []
            for c in cols:
                cn = norm_cols[c]
                if cn == phrase_n:
                    candidates.append((c, 1.0, "exact"))
                    continue
                if phrase_n and (phrase_n in cn or cn in phrase_n):
                    candidates.append((c, 0.93, "contains"))
                    continue
                sim = SequenceMatcher(None, phrase_n, cn).ratio() if phrase_n else 0.0
                if sim >= 0.78:
                    candidates.append((c, sim, "fuzzy_name"))
            candidates.sort(key=lambda x: (-x[1], x[0]))
            if not candidates:
                # Fall back to a small semantic alias vocabulary, but only
                # accept it when it maps uniquely to a physical column.
                alias = UniversalQuestionCompiler._SEMANTIC_ALIASES.get(phrase_n)
                if alias:
                    alias_norm = normalize(alias)
                    alias_candidates = [c for c in cols if normalize(c) == alias_norm]
                    if len(alias_candidates) == 1:
                        return alias_candidates[0], [(alias_candidates[0], 0.82, "semantic_alias")]
                return None, []
            top = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            # Exact physical naming is authoritative when the question names
            # the column exactly. Otherwise require a clear margin before a
            # fuzzy/substring binding is accepted.
            if top[2] == "exact":
                return top[0], candidates[:5]
            if second is None or top[1] - second[1] >= 0.08:
                return top[0], candidates[:5]
            return None, candidates[:5]

        def resolve_column_list(phrase: str) -> Tuple[List[str], List[Tuple[str, float, str]], List[str]]:
            """Resolve a phrase that may contain one or more columns separated by commas / 'and'."""
            phrase_clean = phrase.strip(" .?!")
            single_col, single_cands = resolve_phrase(phrase_clean)
            if single_col and single_cands and single_cands[0][2] == "exact":
                return [single_col], single_cands, []

            parts = [p.strip(" ,") for p in re.split(r",|\band\b", phrase_clean, flags=re.I) if p.strip(" ,")]
            if len(parts) > 1:
                res_cols: List[str] = []
                cands_acc: List[Tuple[str, float, str]] = []
                unresolved: List[str] = []
                for p in parts:
                    c, cands = resolve_phrase(p)
                    if c:
                        if c not in res_cols:
                            res_cols.append(c)
                        cands_acc.extend(cands)
                    else:
                        unresolved.append(p)
                if res_cols or unresolved:
                    return res_cols, cands_acc, unresolved

            if single_col:
                return [single_col], single_cands, []
            return [], single_cands, [phrase_clean] if phrase_clean else []

        target = getattr(semantic, "target_metric_col", None)
        group = getattr(semantic, "group_dimension_col", None)
        time_col = getattr(semantic, "time_col", None)
        expl: List[str] = []
        sec = getattr(semantic, "secondary_metric_col", None)
        evidence: Dict[str, Any] = {"column_references": referenced, "role_resolution": {}}
        if getattr(semantic, "churn_event_col", None):
            evidence["resolved_churn_event"] = getattr(semantic, "churn_event_col")
        unresolved_pairs: List[str] = []

        # Explicit relation: X affects/is associated with Y.
        relation_patterns = [
            # Inflection-tolerant on purpose: "is ad spend related to revenue?" was
            # classified CORRELATION by IntentEngine but "related to" did not match
            # the literal "relate to" here, so the compiler fell back to an unrelated
            # categorical column and the final-contract guard (correctly) refused to
            # run -- a plain two-variable question ended INCONCLUSIVE.
            (
                r"(?:does|do|did|is|are|was|were|can|could|will|would|has|have)\s+(.+?)\s+"
                r"(?:affect\w*|influenc\w*|impact\w*|relat\w*\s+to|associated\s+with|correlat\w*\s+with|"
                r"linked\s+to|tied\s+to|connected\s+to|depend\w*\s+on|coupled\s+(?:to|with))\s+(.+?)(?:\?|$)",
                True,
            ),
            (
                r"(?:relationship|association|correlation|correlate[ds]?|link|connection)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
                False,
            ),
            (
                r"(?:effect|impact)\s+of\s+(.+?)\s+on\s+(.+?)(?:\?|$)",
                True,
            ),
        ]
        explicit_pair = None
        for pat, right_side_is_target in relation_patterns:
            m = re.search(pat, ql, re.I)
            if not m:
                continue
            left, right = m.group(1).strip(" .?!"), m.group(2).strip(" .?!")
            left_cols, left_candidates, left_unresolved = resolve_column_list(left)
            right_cols, right_candidates, right_unresolved = resolve_column_list(right)

            # Semantic aliases are only used after physical-name matching.
            if not left_cols and normalize(left) in {"plan", "plan tier", "subscription plan", "subscription plans"}:
                if group and normalize(group) in {"plan tier", "plan", "subscription plan", "subscription plans"}:
                    left_cols = [group]
                    left_candidates = [(group, 0.82, "semantic_binding")]
            if not right_cols and normalize(right) in {"churn", "churned", "attrition", "retention"}:
                churn_col = getattr(semantic, "churn_event_col", None) or (target if target and "churn" in str(target).lower() else None)
                if churn_col:
                    right_cols = [churn_col]
                    right_candidates = [(churn_col, 0.82, "semantic_binding")]

            evidence["role_resolution"]["exposure"] = {
                "phrase": left,
                "selected": left_cols[0] if len(left_cols) == 1 else left_cols,
                "candidates": left_candidates,
                "unresolved": left_unresolved,
            }
            evidence["role_resolution"]["target"] = {
                "phrase": right,
                "selected": right_cols[0] if len(right_cols) == 1 else right_cols,
                "candidates": right_candidates,
                "unresolved": right_unresolved,
            }

            if left_cols and right_cols:
                known_outcome = getattr(semantic, "churn_event_col", None) or getattr(semantic, "target_metric_col", None)
                known_group = getattr(semantic, "group_dimension_col", None)

                if known_outcome and known_outcome in right_cols and known_outcome not in left_cols:
                    resolved_target = known_outcome
                    resolved_expl = [c for c in left_cols if c != resolved_target]
                elif known_outcome and known_outcome in left_cols and len(left_cols) == 1 and (not known_group or right_cols[0] == known_group):
                    resolved_target = known_outcome
                    resolved_expl = [c for c in right_cols if c != resolved_target]
                elif len(right_cols) == 1 and len(left_cols) >= 1:
                    resolved_target = right_cols[0]
                    resolved_expl = [c for c in left_cols if c != resolved_target]
                elif len(left_cols) == 1 and len(right_cols) >= 1:
                    resolved_target = left_cols[0]
                    resolved_expl = [c for c in right_cols if c != resolved_target]
                else:
                    resolved_target = right_cols[0]
                    resolved_expl = [c for c in left_cols if c != resolved_target]

                if resolved_target and resolved_expl:
                    explicit_pair = (resolved_expl, resolved_target)
                    break

            # Preserve any role that is unambiguously established by the side
            # of the grammar that carries it, even when the other side is not
            # resolvable.  For "does X affect Y?" and "effect of X on Y",
            # the right-hand side is the target by construction.  The previous
            # all-or-nothing gate discarded Y merely because X could not be
            # matched (e.g. "discount percentage" -> "discount_pct"), which
            # turned a partially resolvable question into two missing roles.
            if right_side_is_target and len(right_cols) == 1 and not left_cols:
                target = right_cols[0]
                if target not in referenced:
                    referenced.append(target)
                evidence["role_resolution"]["partial_target_resolution"] = {
                    "selected": target,
                    "reason": "target side resolved although explanatory side remained unresolved",
                }

            if left_unresolved or right_unresolved:
                unresolved_pairs.append(
                    f"Could not uniquely resolve all sides of requested relationship "
                    f"(unresolved: {left_unresolved + right_unresolved})."
                )
            else:
                unresolved_pairs.append("Could not uniquely resolve both sides of the requested relationship.")

        if explicit_pair:
            expl = list(explicit_pair[0])
            target = explicit_pair[1]
            for c in expl:
                if c not in referenced:
                    referenced.append(c)
            if target not in referenced:
                referenced.append(target)
        else:
            # Role hints from common business-question grammar. For churn/retention
            # comparisons, the semantic resolver may already know the exact
            # outcome and grouping columns even when the user paraphrases them.
            if re.search(r"\b(churn|churned|attrition|retention)\b", ql, re.I):
                known_outcome = getattr(semantic, "churn_event_col", None) or getattr(semantic, "target_metric_col", None)
                if known_outcome:
                    target = known_outcome
                    evidence["role_resolution"]["semantic_outcome"] = {"selected": known_outcome, "reason": "known churn/retention outcome"}
                known_group = getattr(semantic, "group_dimension_col", None)
                if known_group and re.search(r"\b(plan|tier|segment|group|region|channel|by|between|across)\b", ql, re.I):
                    group = known_group
                    if group not in referenced:
                        referenced.append(group)
                    evidence["role_resolution"]["semantic_group"] = {"selected": known_group, "reason": "known analytical grouping"}
            # Role hints from common question grammar.
            # Resolve multi-word semantic aliases before the generic metric
            # grammar consumes their trigger word (e.g. "average" in
            # "average order value"). The old regex captured only
            # "order value", so the exact "average order value" alias was
            # never consulted. Only a physically resolvable alias is accepted;
            # this path never invents a derived metric from unrelated columns.
            for alias_phrase, alias_column in sorted(
                UniversalQuestionCompiler._SEMANTIC_ALIASES.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                if " " not in alias_phrase or not re.search(rf"(?<!\w){re.escape(alias_phrase)}(?!\w)", ql):
                    continue
                col, cands = resolve_phrase(alias_phrase)
                evidence["role_resolution"]["target_from_semantic_alias"] = {
                    "phrase": alias_phrase,
                    "alias": alias_column,
                    "selected": col,
                    "candidates": cands,
                }
                if col:
                    target = col
                    break

            metric_match = re.search(r"(?:calculate|what is|show|plot|trend of|revenue by|sales by|average order value|average|avg|aov of|highest|lowest)\s+([a-zA-Z0-9_ ]+?)(?:\s+by\s+|\s+for\s+|\s+after\s+|\s+before\s+|\?|$)", ql)
            by_match = re.search(r"\bby\s+([a-zA-Z0-9_ ]+?)(?:\s+after\s+|\s+before\s+|\?|$)", ql)
            if metric_match:
                metric_phrase = metric_match.group(1).strip()
                col, cands = resolve_phrase(metric_phrase)
                evidence["role_resolution"]["target_from_metric_phrase"] = {"phrase": metric_phrase, "selected": col, "candidates": cands}
                if col:
                    target = col
            if by_match:
                group_phrase = by_match.group(1).strip()
                col, cands = resolve_phrase(group_phrase)
                evidence["role_resolution"]["group_from_by_phrase"] = {"phrase": group_phrase, "selected": col, "candidates": cands}
                if col:
                    group = col

            # Common ranking grammar: "which X has the highest Y?" carries a
            # grouping dimension even without an explicit "by X" phrase. When
            # X resolves to a physical column, promote it to the grouping role.
            which_group_match = re.search(
                r"\bwhich\s+([a-zA-Z0-9_ ]+?)\s+(?:has|have|shows?|with)\s+"
                r"(?:the\s+)?(?:highest|lowest|largest|smallest|best|worst|most|least)\b",
                ql,
            )
            if which_group_match and not group:
                group_phrase = which_group_match.group(1).strip()
                col, cands = resolve_phrase(group_phrase)
                evidence["role_resolution"]["group_from_which_phrase"] = {
                    "phrase": group_phrase,
                    "selected": col,
                    "candidates": cands,
                }
                if col and col != target:
                    group = col
                    if col not in referenced:
                        referenced.append(col)

        # Fall back to semantic bindings only after explicit question evidence.
        if not expl and UniversalQuestionCompiler._ASSOC.search(ql):
            if group and group != target:
                expl.append(group)
            elif sec and sec != target:
                expl.append(sec)
        if group and group not in referenced and re.search(r"\b(by|between|across|within|for each|per)\b", ql):
            if normalize(group) in token_text or normalize(group).split()[0] in token_text:
                referenced.append(group)

        # Detect time role explicitly and horizon semantics.
        horizon = None
        for label, pattern in [("next_month", r"next\s+month"), ("next_quarter", r"next\s+quarter"), ("next_year", r"next\s+year"), ("next_week", r"next\s+week")]:
            if re.search(pattern, ql):
                horizon = label
                break
        if horizon:
            evidence["time_horizon"] = horizon

        # A validated schema-grounded NL proposal may fill only roles that map
        # directly to physical columns. It can enrich interpretation but cannot
        # invent or execute a column. Deterministic question evidence remains
        # authoritative whenever it already resolved a role.
        if proposal is not None:
            proposal_target = getattr(proposal, "target", None)
            proposal_group = getattr(proposal, "group", None)
            proposal_time = getattr(proposal, "time", None)
            proposal_secondary = getattr(proposal, "secondary_target", None)
            if proposal_target in cols and not target:
                target = proposal_target
                referenced.append(proposal_target)
            if proposal_group in cols and not group:
                group = proposal_group
                referenced.append(proposal_group)
            if proposal_time in cols and not time_col:
                time_col = proposal_time
                referenced.append(proposal_time)
            if proposal_secondary in cols and proposal_secondary != target and proposal_secondary not in expl:
                expl.append(proposal_secondary)
                referenced.append(proposal_secondary)
            if getattr(proposal, "limitations", None):
                unresolved_pairs.extend(list(proposal.limitations))

        grain = getattr(semantic, "table_grain", None)
        notes: List[str] = []
        if explicit_pair:
            expl_repr = ", ".join(explicit_pair[0]) if isinstance(explicit_pair[0], list) else str(explicit_pair[0])
            notes.append(f"Explicit analytical relationship resolved from question: {expl_repr} → {explicit_pair[1]}.")
        elif unresolved_pairs:
            notes.extend(unresolved_pairs)
        if referenced:
            notes.append("Physical columns explicitly referenced or confidently resolved from the question were prioritized.")
        else:
            notes.append("No physical column was uniquely referenced; semantic bindings remain provisional.")
        if getattr(semantic, "metric_definition", None) is not None:
            notes.append("Metric aggregation semantics were provided by the metric-semantics layer.")

        # Conservative confidence: explicit exact bindings score highest;
        # fuzzy/provisional fallback scores lower.
        confidence = 0.92 if explicit_pair else (0.78 if referenced else 0.45)
        if unresolved_pairs:
            confidence = min(confidence, 0.35)
        semantic_evidence = evidence
        return SemanticContract(
            referenced_columns=sorted(set(referenced)),
            target_column=target,
            explanatory_columns=expl,
            grouping_columns=[group] if group else [],
            time_column=time_col,
            grain=grain,
            notes=notes,
            resolution_confidence=confidence,
            semantic_evidence=semantic_evidence,
        )

    @staticmethod
    def _estimand(task: str, s: SemanticContract, quality: Any) -> Dict[str, Any]:
        base = {"task": task, "target": s.target_column, "grouping": s.grouping_columns, "time": s.time_column, "unit_of_analysis": s.grain}
        if task == "ASSOCIATION":
            base.update({"estimand": "population association between explanatory and target variables", "claim_ceiling": "association"})
        elif task == "COMPARISON":
            base.update({"estimand": "group contrast in target metric", "claim_ceiling": "association"})
        elif task == "FORECAST":
            base.update({"estimand": "future target trajectory conditional on observed history", "claim_ceiling": "prediction"})
        elif task == "CAUSAL":
            base.update({"estimand": "causal effect of treatment/exposure on outcome", "claim_ceiling": "causal only after identification"})
        elif task == "SEGMENTATION":
            base.update({"estimand": "latent grouping of observed entities", "claim_ceiling": "observation"})
        elif task == "RECONCILIATION":
            base.update({"estimand": "row-level mathematical identity residual", "claim_ceiling": "observation"})
        else:
            base.update({"estimand": "question-specific descriptive quantity", "claim_ceiling": CLAIM_BY_TASK[task]})
        if quality is not None:
            base["data_quality"] = {
                "score": getattr(quality, "overall_quality_score", None),
                "fitness_verdict": getattr(quality, "fitness_verdict", None),
            }
        return base

    @staticmethod
    def _hypotheses(task: str, s: SemanticContract, question: str) -> List[Dict[str, Any]]:
        target = s.target_column or "target"
        group = (s.grouping_columns or ["observed groups"])[0]
        if task == "ASSOCIATION":
            if len(s.explanatory_columns) > 1:
                hyps = []
                for idx, exp_col in enumerate(s.explanatory_columns, start=1):
                    hyps.append({
                        "id": f"HYP-{idx:02d}",
                        "statement": f"{exp_col} is associated with {target}.",
                        "role": "primary",
                    })
                hyps.append({
                    "id": f"HYP-{len(s.explanatory_columns)+1:02d}",
                    "statement": f"The observed associations with {target} are absent or materially smaller after uncertainty is considered.",
                    "role": "counter",
                })
                return hyps
            return [
                {"id": "HYP-01", "statement": f"{s.explanatory_columns[0] if s.explanatory_columns else 'the explanatory variable'} is associated with {target}.", "role": "primary"},
                {"id": "HYP-02", "statement": f"The observed association with {target} is absent or materially smaller after uncertainty is considered.", "role": "counter"},
            ]
        if task == "CAUSAL":
            return [
                {"id": "HYP-01", "statement": f"The requested causal effect from {s.explanatory_columns[0] if s.explanatory_columns else 'treatment'} to {target} is identifiable under the available assumptions.", "role": "primary"},
                {"id": "HYP-02", "statement": "Unmeasured confounding, selection or design limitations prevent a causal conclusion.", "role": "counter"},
            ]
        if task == "FORECAST":
            return [
                {"id": "HYP-01", "statement": "Historical signal contains out-of-sample predictive information for the requested horizon.", "role": "primary"},
                {"id": "HYP-02", "statement": "A simple baseline forecast performs as well as or better than complex alternatives.", "role": "counter"},
            ]
        if task == "SEGMENTATION":
            return [
                {"id": "HYP-01", "statement": "The observed entities contain stable, interpretable latent segments.", "role": "primary"},
                {"id": "HYP-02", "statement": "Apparent segments are unstable or mostly an artifact of scaling/noise.", "role": "counter"},
            ]
        if task == "RECONCILIATION":
            return [
                {"id": "HYP-01", "statement": "The relevant columns satisfy a coherent mathematical identity within tolerance.", "role": "primary"},
                {"id": "HYP-02", "statement": "Material row-level discrepancies exist relative to the inferred identity.", "role": "counter"},
            ]
        if task == "COMPARISON":
            return [
                {"id": "HYP-01", "statement": f"Observed {target} differs materially across {group}.", "role": "primary"},
                {"id": "HYP-02", "statement": f"Observed differences in {target} across {group} are small relative to uncertainty.", "role": "counter"},
            ]
        if task == "DATA_QUALITY":
            return [{"id": "HYP-01", "statement": "Material data-quality defects are present and can be characterized deterministically.", "role": "primary"}]
        return [{"id": "HYP-01", "statement": f"The requested pattern in {target} is supported by the observed data.", "role": "primary"}, {"id": "HYP-02", "statement": "The observed pattern is weak, absent or explained by an alternative structure.", "role": "counter"}]

    @staticmethod
    def _evidence(task: str, s: SemanticContract) -> List[EvidenceRequirement]:
        common = [EvidenceRequirement("SOURCE_SCOPE", "Record exact dataset/version and row scope."), EvidenceRequirement("REPRODUCIBLE_CALC", "Persist formula/query/parameters used to compute the result."), EvidenceRequirement("INDEPENDENT_CHECK", "Independently recompute or validate the conclusion where applicable.")]
        if task == "ASSOCIATION":
            return common + [EvidenceRequirement("EFFECT", "Report effect size and uncertainty."), EvidenceRequirement("CONFOUNDING", "Screen material confounding or composition differences.", mandatory=False)]
        if task == "CAUSAL":
            return common + [EvidenceRequirement("IDENTIFICATION", "Prove a valid identification strategy; do not invent a DAG."), EvidenceRequirement("POSITIVITY", "Check treatment/exposure support and overlap."), EvidenceRequirement("SENSITIVITY", "Quantify sensitivity to unmeasured/confounder assumptions.", mandatory=False)]
        if task == "FORECAST":
            return common + [EvidenceRequirement("BACKTEST", "Use rolling-origin out-of-sample evaluation."), EvidenceRequirement("INTERVAL", "Report uncertainty intervals based on forecast errors.")]
        if task == "SEGMENTATION":
            return common + [EvidenceRequirement("CLUSTER_QUALITY", "Report silhouette/fit quality."), EvidenceRequirement("STABILITY", "Report stability across seeds/resamples."), EvidenceRequirement("INTERPRETABILITY", "Describe cluster profiles in source variables.")]
        if task == "RECONCILIATION":
            return common + [EvidenceRequirement("FORMULA", "Show the exact identity tested."), EvidenceRequirement("RESIDUALS", "Report discrepancy counts, magnitudes and affected records.")]
        if task == "GOVERNANCE":
            return common + [EvidenceRequirement("PII_SCAN", "Scan direct identifiers and PII-like fields."), EvidenceRequirement("HARM_SCREEN", "Identify foreseeable misuse/proxy/harm risks."), EvidenceRequirement("LEGAL_BOUNDARY", "Explicitly state that dataset inspection alone cannot certify legal compliance.")]
        return common

    @staticmethod
    def _experiments(task: str, s: SemanticContract) -> List[ExperimentPlan]:
        if task == "ASSOCIATION":
            if s.target_column and getattr(s, "semantic_evidence", {}).get("resolved_churn_event") is not None:
                return [ExperimentPlan("EXP-ASSOC-PRIMARY", "Estimate the requested churn-rate association directly by the requested grouping variable.", "churn_rate_association"), ExperimentPlan("EXP-ASSOC-CONFOUND", "Check only material composition/confounding that could overturn the primary association.", "targeted_stratified_churn_check", estimated_cost=1.2)]
            if len(s.explanatory_columns) > 1:
                plans = []
                for idx, exp_col in enumerate(s.explanatory_columns, start=1):
                    plans.append(ExperimentPlan(
                        f"EXP-ASSOC-{idx:02d}",
                        f"Estimate the requested association between {exp_col} and {s.target_column or 'target'}.",
                        "correlation_or_regression",
                        target_hypothesis=f"HYP-{idx:02d}",
                    ))
                return plans
            return [ExperimentPlan("EXP-ASSOC-PRIMARY", "Estimate the requested association with an appropriate method.", "correlation_or_regression"), ExperimentPlan("EXP-ASSOC-CONFOUND", "Check whether composition/confounding could explain the signal.", "stratified_or_partial_association", estimated_cost=1.3)]
        if task == "CAUSAL":
            return [ExperimentPlan("EXP-CAUSAL-ID", "Evaluate identification conditions before estimating an effect.", "causal_identification_gate"), ExperimentPlan("EXP-CAUSAL-EFFECT", "Estimate the identified effect with cross-fitting/robust uncertainty.", "aipw_or_g_computation", estimated_cost=2.0, prerequisites=["EXP-CAUSAL-ID"])]
        if task == "FORECAST":
            return [ExperimentPlan("EXP-FORECAST-BACKTEST", "Compare candidate forecasting baselines with rolling-origin validation.", "forecast_backtest")]
        if task == "SEGMENTATION":
            return [ExperimentPlan("EXP-SEGMENT-AUTO", "Discover candidate latent groups and evaluate stability.", "automatic_clustering", estimated_cost=1.5)]
        if task == "RECONCILIATION":
            return [ExperimentPlan("EXP-RECONCILE", "Infer and test candidate accounting/math identities row by row.", "mathematical_reconciliation", estimated_cost=0.8)]
        if task == "DATA_QUALITY":
            return [ExperimentPlan("EXP-DATA-QUALITY", "Profile missingness, duplicates, invalids, outliers and structural anomalies.", "data_quality_profile", estimated_cost=0.5)]
        if task == "GOVERNANCE":
            return [ExperimentPlan("EXP-GOVERNANCE", "Run privacy, proxy-risk and harm/governance scan.", "governance_audit", estimated_cost=0.5)]
        if task == "PRESCRIPTIVE":
            return [ExperimentPlan("EXP-DECISION-CONTEXT", "Establish decision constraints and eligible intervention candidates.", "decision_context_gate"), ExperimentPlan("EXP-IMPACT", "Simulate or compare supported interventions only after evidence exists.", "decision_impact_analysis", estimated_cost=1.8, prerequisites=["EXP-DECISION-CONTEXT"])]
        return [ExperimentPlan("EXP-DESCRIPTIVE", "Answer the requested descriptive quantity directly.", "descriptive_query")]

    @staticmethod
    def experiment_budget(task: str, *, semantics: Optional[SemanticContract] = None) -> int:
        """Return the intended number of experiments for the question class.

        This is a scientific planning budget, not a performance timeout. It is
        deliberately small for direct questions and can be expanded by the
        runtime when a new material adversarial issue actually requires a
        targeted follow-up.
        """
        if task in {"DATA_QUALITY", "SEGMENTATION", "RECONCILIATION", "GOVERNANCE", "FORECAST"}:
            return 1
        if task in {"ASSOCIATION", "CAUSAL", "COMPARISON", "PREDICTION"}:
            if task == "ASSOCIATION" and semantics and len(semantics.explanatory_columns) > 1:
                return max(2, len(semantics.explanatory_columns))
            return 2
        if task == "DIAGNOSTIC":
            return 3
        if task == "PRESCRIPTIVE":
            return 2
        return 3

    @staticmethod
    def _stopping_rule(task: str) -> str:
        if task == "ASSOCIATION":
            return "Stop after the requested association is estimated with uncertainty, independently checked, and no material unresolved confounding issue remains for the requested claim."
        if task == "CAUSAL":
            return "Stop immediately with a non-causal verdict if identification fails; otherwise stop when the causal estimand is estimated, sensitivity assessed as required, and independently verified."
        if task == "FORECAST":
            return "Stop after candidate models are backtested and the best defensible model/interval is selected; do not keep experimenting after the requested predictive objective is resolved."
        if task == "SEGMENTATION":
            return "Stop when a stable, interpretable clustering solution dominates alternatives; otherwise report no stable segmentation."
        if task == "RECONCILIATION":
            return "Stop after candidate identities are tested and all material residual discrepancies are enumerated."
        if task == "DATA_QUALITY":
            return "Stop after material quality defects and their impact have been characterized; transformations require an explicit recipe."
        return "Stop when the required evidence for the requested claim is complete and independently reproducible."

    @staticmethod
    def _unresolved(task: str, s: SemanticContract, b: BusinessContextContract, quality: Any) -> List[str]:
        u: List[str] = []
        if task in {"ASSOCIATION", "CAUSAL", "FORECAST", "PREDICTION"} and not s.target_column:
            u.append("Target variable could not be resolved confidently.")
        if task == "ASSOCIATION" and not s.explanatory_columns:
            u.append("Explanatory variable could not be resolved confidently.")
        if task in {"COMPARISON", "SEGMENTATION"} and not s.grouping_columns and task == "COMPARISON":
            u.append("Grouping dimension could not be resolved confidently.")
        if task == "FORECAST" and not s.time_column:
            u.append("Time column could not be resolved confidently.")
        if task == "PRESCRIPTIVE" and not b.decision_context_known:
            u.append("Operational decision constraints are not known; recommendations must remain conditional.")
        if quality is not None and getattr(quality, "fitness_verdict", None) == "UNFIT":
            u.append("Data quality gate marked the primary dataset UNFIT.")
        return u

    @staticmethod
    def _limitations(task: str, s: SemanticContract, quality: Any) -> List[str]:
        out: List[str] = []
        if task == "CAUSAL":
            out.append("Causal conclusions require an identification strategy; observational association is not sufficient.")
        if task == "GOVERNANCE":
            out.append("Legal compliance cannot be certified from dataset inspection alone.")
        if task == "SEGMENTATION":
            out.append("Clusters are descriptive partitions; they are not proof of natural or causal customer types.")
        if task == "PRESCRIPTIVE":
            out.append("No financial or operational impact magnitude is invented without user-provided cost/benefit assumptions.")
        if quality is not None and getattr(quality, "warnings", None):
            out.append("Quality warnings are carried into the analytical contract.")
        return out

    @staticmethod
    def _visualizations(task: str) -> List[str]:
        return {
            "DATA_QUALITY": ["quality_summary", "missingness_heatmap", "anomaly_table"],
            "DESCRIPTIVE": ["metric_card", "distribution", "table"],
            "COMPARISON": ["grouped_bar", "box_or_interval_plot", "table"],
            "ASSOCIATION": ["scatter_or_effect_plot", "partial_effect_plot", "table"],
            "DIAGNOSTIC": ["contribution_bar", "conditional_comparison", "table"],
            "FORECAST": ["forecast_with_interval", "backtest_comparison"],
            "PREDICTION": ["calibration_plot", "lift_or_risk_rank", "table"],
            "CAUSAL": ["effect_interval_plot", "overlap_plot", "sensitivity_curve"],
            "SEGMENTATION": ["cluster_profile_plot", "dimensional_projection", "cluster_size_bar"],
            "RECONCILIATION": ["residual_histogram", "discrepancy_scatter", "exception_table"],
            "PRESCRIPTIVE": ["scenario_comparison", "impact_tradeoff_plot"],
            "GOVERNANCE": ["risk_summary", "pii_inventory", "governance_table"],
            "GENERAL_EXPLORATION": ["overview_dashboard"],
        }.get(task, ["table"])


class AutomaticClusteringEngine:
    """Deterministic, bounded clustering with quality and stability evidence."""

    @staticmethod
    def analyze(df: pd.DataFrame, *, requested_columns: Optional[Sequence[str]] = None, max_k: int = 6, seed: int = 1729) -> Dict[str, Any]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        from sklearn.preprocessing import StandardScaler

        work = df.copy()
        candidate = list(requested_columns or [])
        if not candidate:
            for c in work.select_dtypes(include=[np.number]).columns:
                cl = str(c).lower()
                if any(x in cl for x in ["_id", "id", "key", "code", "zip"]):
                    continue
                if work[c].nunique(dropna=True) > 1:
                    candidate.append(str(c))
        candidate = candidate[:20]
        if len(candidate) < 2:
            return {"status": "INSUFFICIENT_FEATURES", "features": candidate, "clusters": []}

        # Support mixed numeric + low-cardinality categorical features while
        # excluding obvious identifiers/high-cardinality text. This makes the
        # segmentation question genuinely data-driven instead of requiring
        # the user to pre-encode their dataset.
        numeric_cols = [c for c in candidate if pd.api.types.is_numeric_dtype(work[c])]
        categorical_cols = [
            c for c in candidate
            if c not in numeric_cols and 1 < work[c].nunique(dropna=True) <= 20
        ]
        usable = numeric_cols + categorical_cols
        if len(usable) < 2:
            return {"status": "INSUFFICIENT_FEATURES", "features": usable, "clusters": []}

        subset = work[usable].copy()
        for c in numeric_cols:
            subset[c] = pd.to_numeric(subset[c], errors="coerce")
            subset[c] = subset[c].fillna(subset[c].median())
        for c in categorical_cols:
            subset[c] = subset[c].astype("string").fillna("<MISSING>")

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        transformers = []
        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
        pre = ColumnTransformer(transformers, remainder="drop")
        X = pre.fit_transform(subset)
        if len(X) < 20:
            return {"status": "INSUFFICIENT_ROWS", "rows_used": int(len(X)), "features": usable, "clusters": []}
        candidate = usable
        upper_k = min(int(max_k), max(2, min(10, len(X) // 10)))
        scores: Dict[int, float] = {}
        models: Dict[int, Any] = {}
        for k in range(2, upper_k + 1):
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            labels = model.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            scores[k] = float(silhouette_score(X, labels))
            models[k] = model
        if not scores:
            return {"status": "NO_CLUSTERING_SOLUTION", "features": candidate, "clusters": []}
        best_k = max(scores, key=scores.get)
        model = models[best_k]
        labels = model.labels_
        stability_models = [KMeans(n_clusters=best_k, n_init=20, random_state=s) for s in (seed + 1, seed + 2)]
        stability_labels = [m.fit_predict(X) for m in stability_models]
        aris = [float(adjusted_rand_score(labels, other)) for other in stability_labels]
        profile = []
        data_with_labels = subset.copy()
        data_with_labels["__cluster"] = labels
        for cluster_id, part in data_with_labels.groupby("__cluster"):
            means = {str(c): round(float(part[c].mean()), 6) for c in numeric_cols}
            modes = {}
            for c in categorical_cols:
                mode = part[c].mode(dropna=False)
                if not mode.empty:
                    modes[str(c)] = str(mode.iloc[0])
            profile.append({
                "cluster": int(cluster_id),
                "n": int(len(part)),
                "share_pct": round(float(len(part) / len(subset) * 100), 2),
                "numeric_means": means,
                "categorical_modes": modes,
            })
        return {
            "status": "COMPLETED",
            "features": candidate,
            "rows_available": int(len(work)),
            "rows_used": int(len(subset)),
            "rows_excluded_missing": 0,
            "k_scores": {str(k): round(v, 6) for k, v in scores.items()},
            "selected_k": int(best_k),
            "silhouette_score": round(float(scores[best_k]), 6),
            "stability_ari": round(float(np.mean(aris)), 6),
            "stability_ari_runs": [round(v, 6) for v in aris],
            "cluster_profiles": profile,
            "parameters": {"seed": seed, "max_k": max_k, "standardization": "z_score", "missing_policy": "numeric_median_and_categorical_missing_token"},
        }


class MathematicalReconciliationEngine:
    """Infer and test row-level arithmetic identities without silent assumptions."""

    _ALIASES = {
        "quantity": ["quantity", "qty", "units", "unit_count"],
        "unit_price": ["unit_price", "unitprice", "price_per_unit", "price"],
        "discount": ["discount", "discount_rate", "discount_pct", "discount_percent"],
        "total": ["total_amount", "totalamount", "net_amount", "order_total", "amount", "total", "revenue", "sales"],
        "tax": ["tax", "tax_amount", "vat", "gst"],
        "gross": ["gross_amount", "gross_total", "gross_value", "subtotal", "sub_total"],
    }

    @classmethod
    def _find(cls, cols: Iterable[str], aliases: List[str]) -> Optional[str]:
        normalized = {str(c).lower().replace(" ", "_"): c for c in cols}
        for a in aliases:
            if a in normalized:
                return normalized[a]
        for c in cols:
            cl = str(c).lower().replace(" ", "_")
            if any(a in cl for a in aliases):
                return c
        return None

    @classmethod
    def analyze(cls, df: pd.DataFrame, *, tolerance_abs: float = 0.01, tolerance_rel: float = 1e-6) -> Dict[str, Any]:
        bindings = {k: cls._find(df.columns, aliases) for k, aliases in cls._ALIASES.items()}
        q, p, d, total = bindings["quantity"], bindings["unit_price"], bindings["discount"], bindings["total"]
        if not all([q, p, total]):
            return {"status": "INSUFFICIENT_COLUMNS", "bindings": bindings, "identities": []}
        work = pd.DataFrame(index=df.index)
        for c in {q, p, d, total} - {None}:
            work[c] = pd.to_numeric(df[c], errors="coerce")
        discount = work[d] if d else pd.Series(0.0, index=work.index)
        # Interpret discount as a rate only when its scale supports that. A raw
        # currency discount amount is handled by a separate identity below.
        identities = []
        if d and discount.dropna().between(0, 1).mean() >= 0.8:
            expected = work[q] * work[p] * (1.0 - discount)
            name = "quantity * unit_price * (1 - discount_rate)"
            identities.append((name, expected))
        expected_gross = work[q] * work[p]
        identities.append(("quantity * unit_price", expected_gross))
        if d:
            discount_amount = work[d]
            identities.append(("quantity * unit_price - discount_amount", expected_gross - discount_amount))
        results = []
        for formula, expected in identities:
            actual = work[total]
            mask = expected.notna() & actual.notna()
            residual = (actual - expected).loc[mask]
            tol = tolerance_abs + tolerance_rel * expected.abs().loc[mask]
            bad = residual.abs() > tol
            results.append({
                "formula": formula,
                "rows_checked": int(mask.sum()),
                "discrepancy_rows": int(bad.sum()),
                "discrepancy_pct": round(float(bad.mean() * 100), 4) if len(bad) else None,
                "max_abs_error": round(float(residual.abs().max()), 8) if len(residual) else None,
                "mean_abs_error": round(float(residual.abs().mean()), 8) if len(residual) else None,
                "affected_row_indices": [str(i) for i in residual.index[bad][:100]],
                "tolerance": {"absolute": tolerance_abs, "relative": tolerance_rel},
            })
        best = min(results, key=lambda r: r["discrepancy_pct"] if r["discrepancy_pct"] is not None else float("inf"))
        return {"status": "COMPLETED", "bindings": bindings, "best_identity": best, "identities": results}


class GovernanceRiskEngine:
    """Dataset-level privacy, harm, and legal-boundary assessment."""

    _SENSITIVE_TOKENS = ("income", "salary", "health", "medical", "religion", "race", "ethnicity", "sex", "gender", "age", "disability", "political")
    _PII_TOKENS = ("email", "phone", "ssn", "address", "name", "dob", "birth", "passport", "national_id")

    @classmethod
    def assess(cls, df: pd.DataFrame) -> Dict[str, Any]:
        pii = []
        sensitive = []
        proxy_candidates = []
        for c in df.columns:
            cl = str(c).lower()
            if any(t in cl for t in cls._PII_TOKENS):
                pii.append(c)
            if any(t in cl for t in cls._SENSITIVE_TOKENS):
                sensitive.append(c)
        if sensitive and pii:
            proxy_candidates = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in pii and df[c].nunique(dropna=True) <= max(2, min(50, len(df) // 10))]
        risks = []
        if pii:
            risks.append("Direct identifiers or PII-like columns are present; minimize exposure and mask where possible.")
        if sensitive:
            risks.append("Sensitive-attribute-like columns are present; avoid using them for eligibility/pricing decisions without governance review.")
        if proxy_candidates:
            risks.append("Low-cardinality categorical variables may act as proxies for sensitive attributes; inspect before decision use.")
        risks.append("Legal compliance is not certified by this dataset scan; purpose, lawful basis, jurisdiction and organizational controls remain unresolved.")
        return {
            "status": "ASSESSED",
            "pii_columns": pii,
            "sensitive_columns": sensitive,
            "proxy_candidate_columns": proxy_candidates[:50],
            "risks": risks,
            "legal_boundary": "Dataset inspection alone cannot establish GDPR/CCPA or other legal compliance; legal basis, purpose, jurisdiction and organizational controls are required.",
            "recommended_controls": ["purpose limitation", "least-privilege access", "retention policy", "pseudonymization/masking", "human review for consequential decisions"],
        }


class DecisionImpactEngine:
    """Generate conditional decision options without inventing financial impact."""

    @staticmethod
    def build(*, finding: str, evidence_verified: bool, causal_identified: bool, decision_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = decision_context or {}
        actions = []
        if not evidence_verified:
            return {"status": "HOLD", "reason": "Evidence is not independently verified; no consequential action should be promoted.", "actions": []}
        if causal_identified:
            actions.append({"type": "INTERVENTION_CANDIDATE", "action": f"Evaluate an intervention directly targeting the supported factor: {finding}", "claim_strength": "conditional causal"})
        else:
            actions.append({"type": "DIAGNOSTIC_FOLLOWUP", "action": f"Run a targeted validation/pilot for: {finding}", "claim_strength": "association only"})
        if context.get("budget") is not None:
            actions.append({"type": "CONSTRAINED_OPTION", "action": "Rank interventions against the user-provided budget constraint.", "claim_strength": "scenario/optimization"})
        return {"status": "CONDITIONAL", "actions": actions, "disclosed_unknowns": ["implementation cost", "operational feasibility", "strategic priorities"]}
