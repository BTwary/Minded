"""Deterministic local conversational front-end for AA-OS.

This layer understands common analytical language without an LLM. It does not
calculate, invent, or alter analytical results. It converts natural-language
questions into a compact routing hint that the canonical InvestigationController
still owns and executes.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass(frozen=True)
class ConversationPlan:
    problem_class: str
    user_intent: str
    target_horizon: str | None
    requested_metric_hint: str | None
    explanation: str
    follow_ups: List[str]

    def to_dict(self) -> dict:
        return {
            "problem_class": self.problem_class,
            "user_intent": self.user_intent,
            "target_horizon": self.target_horizon,
            "requested_metric_hint": self.requested_metric_hint,
            "explanation": self.explanation,
            "follow_ups": list(self.follow_ups),
        }


class LocalConversationalAnalyst:
    """Deterministic question interpretation for offline chat."""

    _FORECAST = re.compile(
        r"\b(forecast|predict|prediction|likely|next\s+(?:month|quarter|week|year)|future|expected|projection|projected)\b",
        re.I,
    )
    _CAUSAL = re.compile(
        r"\b(cause|caused|causing|impact|effect|responsible|because\s+of|due\s+to|does\s+.*\s+cause|did\s+.*\s+cause)\b",
        re.I,
    )
    _DIAGNOSTIC = re.compile(
        r"\b(why|driver|drivers|reason|reasons|explain|explaining|what\s+is\s+behind|contributing)\b",
        re.I,
    )
    _COMPARATIVE = re.compile(
        r"\b(compare|comparison|versus|vs\.?|difference|higher|lower|better|worse|between)\b",
        re.I,
    )
    _CORRELATIONAL = re.compile(
        r"\b(correlat|association|relationship|related|co-move|moves\s+with)\b",
        re.I,
    )
    _SURVIVAL = re.compile(
        r"\b(churn|retention|survival|time\s+to|time-to-event|hazard|attrition)\b",
        re.I,
    )
    _DESCRIPTIVE = re.compile(
        r"\b(how\s+much|how\s+many|what\s+is|what\s+was|show|summarize|summary|average|total|count|trend)\b",
        re.I,
    )
    _METRIC = re.compile(
        r"\b(revenue|sales|profit|margin|cost|orders|customers?|conversion|retention|churn|price|units?|quantity|volume)\b",
        re.I,
    )
    _HORIZON = re.compile(
        r"\b(next\s+(?P<horizon>month|quarter|week|year)|(?P<horizon2>\d+)\s+(?P<unit>months?|quarters?|weeks?|years?))\b",
        re.I,
    )


    @classmethod
    def resolve_followup(cls, question: str, prior_user_questions: List[str]) -> tuple[str, bool]:
        """Resolve short conversational follow-ups without inventing new facts.

        A follow-up is expanded only when it is clearly referential/fragmentary
        (e.g. "by region", "what about last quarter?", "and churn?"). A normal
        standalone question is returned unchanged.
        """
        current = " ".join((question or "").split())
        if not current or not prior_user_questions:
            return current, False
        ql = current.lower()
        standalone_signals = re.search(
            r"\b(what|why|how|which|compare|show|calculate|does|do|is|are|will|predict|forecast|find|check|explain)\b",
            ql,
        )
        referential = bool(re.search(
            r"^(?:and|also|what about|how about|by|per|for each|across|within|there|that|those|same|instead|then|now)\b|\b(it|that|those|same|above|previous)\b",
            ql,
        ))
        if standalone_signals and not referential and len(current.split()) >= 4:
            return current, False
        previous = next((p.strip() for p in reversed(prior_user_questions) if p and p.strip()), None)
        if not previous:
            return current, False
        # Preserve the original wording and make the relationship explicit for
        # the deterministic compiler. It is still the original user question
        # plus context, not an LLM-generated replacement.
        expanded = f"{previous.rstrip('?')} ; follow-up: {current}"
        return expanded, True

    @classmethod
    def plan(cls, question: str) -> ConversationPlan:
        q = " ".join((question or "").split())
        if not q:
            return ConversationPlan(
                problem_class="general",
                user_intent="empty_question",
                target_horizon=None,
                requested_metric_hint=None,
                explanation="Ask a question about the data, such as a trend, driver, comparison, forecast, or causal effect.",
                follow_ups=[],
            )

        # Causal wording must take precedence over correlation/diagnostic words.
        if cls._CAUSAL.search(q):
            problem_class = "causal"
            user_intent = "estimate_or_assess_causal_effect"
            explanation = "AA-OS will test whether the available data can identify and support the requested causal claim; it will not equate association with causation."
        elif cls._FORECAST.search(q):
            problem_class = "forecasting"
            user_intent = "forecast_future_value_or_direction"
            explanation = "AA-OS will evaluate historical signal, competing forecasting approaches, backtest them where supported, and report uncertainty rather than a guaranteed outcome."
        elif cls._SURVIVAL.search(q):
            problem_class = "survival_churn"
            user_intent = "analyze_retention_or_time_to_event"
            explanation = "AA-OS will determine the appropriate retention/churn estimand and assess whether the available observation window supports it."
        elif cls._DIAGNOSTIC.search(q):
            problem_class = "diagnostic"
            user_intent = "identify_drivers_or_explanations"
            explanation = "AA-OS will investigate competing explanations and test which factors are empirically supported as contributors to the observed outcome."
        elif cls._COMPARATIVE.search(q):
            problem_class = "comparative"
            user_intent = "compare_groups_or_periods"
            explanation = "AA-OS will define the comparison estimand and account for relevant data quality, grain, and exposure differences before comparing groups or periods."
        elif cls._CORRELATIONAL.search(q):
            problem_class = "correlational"
            user_intent = "measure_association"
            explanation = "AA-OS will measure association and uncertainty while keeping the conclusion distinct from a causal claim."
        elif cls._DESCRIPTIVE.search(q):
            problem_class = "descriptive"
            user_intent = "describe_or_summarize_data"
            explanation = "AA-OS will compute the requested descriptive metric from the actual dataset and preserve the full calculation trace."
        else:
            problem_class = "general"
            user_intent = "general_analytical_question"
            explanation = "AA-OS will inspect the question and available data, then route it through the canonical analytical investigation engine."

        horizon = None
        match = cls._HORIZON.search(q)
        if match:
            horizon = match.group("horizon") or f"{match.group('horizon2')} {match.group('unit')}"

        metric_matches = [m.group(1).lower() for m in cls._METRIC.finditer(q)]
        metric_priority = ["churn", "retention", "revenue", "sales", "profit", "margin", "cost", "orders", "customers", "conversion", "price", "units", "quantity", "volume"]
        metric = next((candidate for candidate in metric_priority if candidate in metric_matches), None)

        follow_ups = cls._followups(problem_class, metric, horizon)
        return ConversationPlan(problem_class, user_intent, horizon, metric, explanation, follow_ups)

    @staticmethod
    def _followups(problem_class: str, metric: str | None, horizon: str | None) -> List[str]:
        if problem_class == "forecasting":
            return [
                "Show the forecast calculation and uncertainty",
                "Compare the competing forecasting methods",
                "Test whether the forecast is sensitive to recent changes",
            ]
        if problem_class == "diagnostic":
            return [
                "Show the strongest competing explanations",
                "Show the calculations behind the main driver",
                "Test an alternative explanation",
            ]
        if problem_class == "causal":
            return [
                "Show the causal assumptions and identification limits",
                "Show which confounders were considered",
                "Try a non-causal association analysis",
            ]
        if problem_class == "comparative":
            return [
                "Show the comparison formula and groups",
                "Check whether group exposure differs",
                "Test the comparison for robustness",
            ]
        if problem_class == "correlational":
            return [
                "Show the correlation calculation",
                "Check for influential observations",
                "Test whether the association is robust",
            ]
        if problem_class == "survival_churn":
            return [
                "Show the churn/retention estimand",
                "Check the observation window",
                "Test whether exposure or cohort mix changes the result",
            ]
        return [
            "Show the exact calculations",
            "Show the evidence and verification",
            "Challenge the conclusion",
        ]
