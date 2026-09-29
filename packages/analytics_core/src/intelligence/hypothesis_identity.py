"""Canonical semantic hypothesis identity.

This module is the ONLY place that decides whether two hypotheses are the
same substantive claim. Every other component (synthesis, state management,
the controller loop, and persistence/reconstruction) must call
``compute_semantic_identity`` rather than re-deriving equivalence rules of
its own -- that is what previously allowed the synthesizer and the state
manager to silently disagree about what counted as "the same hypothesis".

Identity is computed from the SUBSTANCE of the claim, not its wording:

    target metric
    + mechanism/claim type   (a coarse taxonomy, not the free-text sentence)
    + target dimension/entity
    + target value
    + material temporal scope
    + direction

Two hypotheses with different wording but the same values along every one
of those axes are the same hypothesis. A difference on ANY axis (metric,
dimension, value, period, mechanism type, or direction) makes them distinct
claims that must never be silently merged.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Mechanism / claim-type taxonomy
# ---------------------------------------------------------------------------
# Coarse, deterministic classification of *what kind of explanatory claim*
# a hypothesis is making. This is intentionally small and keyword-driven so
# that it stays deterministic and auditable -- it is not meant to capture
# nuance, only to separate hypotheses that are making fundamentally
# different kinds of claims (e.g. "the change is localized to one segment"
# vs. "uniform macro drift" vs. "confounding by a second dimension" vs.
# "a temporal regime shift").
#
# Deliberately NOT split further than this: concentration, dispersion, and
# anomaly detectors are different STATISTICAL TECHNIQUES for finding the
# same underlying claim ("this target_dimension/target_value segment is
# where the target_metric change is concentrated"). Splitting them into
# separate mechanism types would defeat same-batch consolidation -- three
# detectors firing on the same segment in one round must land in the same
# "localized_effect" bucket so target_metric+dimension+value+period+direction
# alone (not which detector found it) decide identity.
_MECHANISM_TAXONOMY: Dict[str, tuple] = {
    "confounding": ("confound", "simpson", "mediat", "masquerad", "orthogonal"),
    "temporal_shift": ("temporal shift", "regime", "time-series", "pre/post", "pre-vs-post", "transition"),
    "macro_uniform": ("uniform", "macro-level", "macro drift", "systemic macro", "baseline macro"),
    "localized_effect": (
        "concentrat", "localized", "high-volume partitions", "disproportionate",
        "dispersion", "heterogeneity", "partitioning across", "variance disparity",
        "anomal", "outlier", "z-score", "spike",
        "alternative dimension", "following refutation",
    ),
}

_DIRECTION_KEYWORDS = {
    "decrease": ("decline", "decreas", "drop", "fall", "fell", "reduc", "shrink", "lower", "down"),
    "increase": ("increas", "rise", "rose", "grow", "surge", "gain", "up ", "higher", "elevat", "exceed", "above"),
}

# Matches common month names / numeric year-months / bare years, used as a
# best-effort extraction of the "material temporal scope" of a claim when
# the hypothesis object itself does not carry an explicit temporal_scope.
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[\s\-]*'?\s*(\d{2,4})?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ISO_PERIOD_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")


def classify_mechanism_type(hypothesis: Any) -> str:
    """Deterministically classify the coarse mechanism/claim type of a hypothesis.

    A counter/null hypothesis (is_counter_hypothesis=True) is always its own
    "counter_null" bucket, checked before any keyword matching: a counter
    hypothesis asserts the ABSENCE of the specific effect being claimed
    elsewhere, which is a fundamentally different kind of claim from any
    positive explanatory hypothesis -- even when both happen to have sparse
    or generic claim/mechanism text that would otherwise classify the same
    way. Without this, a thinly-described counter-hypothesis could
    spuriously collide in identity with an unrelated leading hypothesis
    purely because neither's text matched any taxonomy keyword.

    Otherwise looks at ``mechanism_detail`` first (structured field, when
    populated), then falls back to ``mechanism`` and finally ``claim``.
    """
    if getattr(hypothesis, "is_counter_hypothesis", False):
        return "counter_null"

    text = " ".join(
        str(getattr(hypothesis, field_name, "") or "")
        for field_name in ("mechanism_detail", "mechanism", "claim")
    ).lower()
    for category, keywords in _MECHANISM_TAXONOMY.items():
        if any(kw in text for kw in keywords):
            return category
    return "unclassified"


def extract_direction(hypothesis: Any) -> str:
    """Best-effort deterministic direction of the claimed effect.

    Prefers an explicit ``direction`` attribute if the hypothesis object
    already carries one (e.g. stamped from a linked prediction's
    ``expected_direction``); otherwise scans claim/mechanism text.
    """
    explicit = str(getattr(hypothesis, "direction", "") or "").strip().lower()
    if explicit and explicit != "unspecified":
        return explicit

    text = " ".join(
        str(getattr(hypothesis, field_name, "") or "")
        for field_name in ("claim", "mechanism", "mechanism_detail")
    ).lower()
    for direction, keywords in _DIRECTION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return direction
    return "unspecified"


def extract_temporal_scope(hypothesis: Any) -> str:
    """Best-effort deterministic material temporal scope of a claim.

    Prefers an explicit ``temporal_scope`` attribute (should be populated
    by the semantic world model / temporal-scope resolver upstream when
    available); otherwise scans claim/mechanism text for an ISO period,
    a bare year, or a month (+ optional year).
    """
    explicit = str(getattr(hypothesis, "temporal_scope", "") or "").strip()
    if explicit:
        return explicit.lower()

    # NOTE: deliberately excludes ``generated_reason``. That field is
    # provenance/explanation text -- how or when the synthesizer happened to
    # notice the pattern -- not part of the substantive claim. Two
    # synthesis passes can produce the same hypothesis ("revenue lower in
    # region B") with differently worded (or absent) generated_reason text;
    # if that text leaked into temporal-scope extraction, an incidental
    # month mention in one pass's explanation but not the other's would
    # hash the identical claim to two different canonical identities and
    # split evidence/posterior mass across what is substantively one
    # hypothesis. Only the claim/mechanism text -- the actual proposition --
    # is scanned here.
    text = " ".join(
        str(getattr(hypothesis, field_name, "") or "")
        for field_name in ("claim", "mechanism")
    )

    iso_match = _ISO_PERIOD_RE.search(text)
    if iso_match:
        return iso_match.group(0)

    month_match = _MONTH_RE.search(text)
    if month_match:
        month_token = month_match.group(1).lower()
        year_token = month_match.group(2) or ""
        return f"{month_token}-{year_token}" if year_token else month_token

    year_match = _YEAR_RE.search(text)
    if year_match:
        return year_match.group(0)

    return "unscoped"


def _normalize(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


@dataclass(frozen=True)
class SemanticIdentityComponents:
    """Human-readable normalized components behind a canonical identity hash.

    Kept separate from the hash itself so tests and debugging tools can
    assert on individual axes ("did these differ on target_value?") without
    reverse-engineering a hash.
    """
    target_metric: str
    mechanism_type: str
    target_dimension: str
    target_value: str
    temporal_scope: str
    direction: str
    fallback_claim: str = ""
    # v20-C4.2.3: the predictor a pairwise hypothesis is about (structured
    # ``secondary_metric``).  Without this axis "price -> annual_sales" and
    # "marketing_spend -> annual_sales" hash to the SAME identity and persistence
    # consolidates the second pair into the first, silently dropping a requested
    # predictor.  It contributes to the key ONLY when non-empty, so every
    # hypothesis that has no predictor keeps its existing identity byte-for-byte.
    predictor: str = ""

    def as_key(self) -> str:
        key = "|".join(
            [
                self.target_metric,
                self.mechanism_type,
                self.target_dimension,
                self.target_value,
                self.temporal_scope,
                self.direction,
                self.fallback_claim,
            ]
        )
        return f"{key}|predictor={self.predictor}" if self.predictor else key


def compute_semantic_identity_components(hypothesis: Any) -> SemanticIdentityComponents:
    """Derive the normalized identity axes for a hypothesis. Pure function, no I/O."""
    mechanism = classify_mechanism_type(hypothesis)
    target_metric = _normalize(getattr(hypothesis, "target_metric", ""))
    target_dimension = _normalize(getattr(hypothesis, "target_dimension", ""))
    target_value = _normalize(getattr(hypothesis, "target_value", None))
    temporal_scope = extract_temporal_scope(hypothesis)
    direction = extract_direction(hypothesis)

    # Safety rule for legacy/free-text hypotheses: when the structured axes are
    # substantially absent, do not allow every unclassified statement to hash to
    # the same identity. A normalized claim fallback prevents false consolidation.
    # Structured hypotheses retain wording-independent identity; the fallback is
    # only activated when there is not enough structured substance to distinguish
    # propositions reliably.
    structured_missing = not any([target_metric, target_dimension, target_value])
    fallback = ""
    if structured_missing:
        claim = " ".join(str(getattr(hypothesis, f, "") or "") for f in ("claim", "statement", "mechanism", "rationale")).lower()
        claim = re.sub(r"[^a-z0-9]+", " ", claim).strip()
        # Legacy hypotheses often have no structured target fields. Canonicalize
        # common business-language paraphrases while retaining directional meaning
        # so semantically equivalent free-text propositions can consolidate without
        # collapsing unrelated claims.
        aliases = {
            "accounts": "customer", "account": "customer", "customers": "customer",
            "elevated": "increase", "elevatedly": "increase", "higher": "increase",
            "greater": "increase", "exceeds": "increase", "above": "increase",
            "lower": "decrease", "less": "decrease", "decline": "decrease",
            "declines": "decrease", "fell": "decrease",
        }
        stop = {"the", "a", "an", "have", "has", "is", "are", "was", "were", "for", "of", "in", "on", "to", "and", "that", "with", "by", "from"}
        tokens = []
        for token in claim.split():
            token = aliases.get(token, token)
            if token in stop:
                continue
            tokens.append(token)
        fallback = " ".join(sorted(set(tokens)))
    return SemanticIdentityComponents(
        target_metric=target_metric,
        mechanism_type=mechanism,
        target_dimension=target_dimension,
        target_value=target_value,
        temporal_scope=temporal_scope,
        direction=direction,
        fallback_claim=fallback,
        predictor=_normalize(getattr(hypothesis, "secondary_metric", None)),
    )


def compute_semantic_identity(hypothesis: Any) -> str:
    """THE authoritative canonical identity for a hypothesis.

    Returns a stable sha256 hex digest over the normalized identity
    components. Two hypotheses (regardless of wording, hypothesis_code, or
    generation round) that produce the same digest are the same substantive
    claim and MUST be consolidated. Two hypotheses that differ on ANY axis
    produce different digests and MUST remain separate.

    This function must be called from every site that creates or persists
    a hypothesis (HypothesisSynthesizer, InvestigationStateManager, the
    controller loop, and state reconstruction on restart) -- never
    reimplemented locally.
    """
    components = compute_semantic_identity_components(hypothesis)
    return hashlib.sha256(components.as_key().encode("utf-8")).hexdigest()


def stamp_semantic_identity(hypothesis: Any) -> Any:
    """Compute and attach canonical_identity (+ derived temporal_scope/direction
    when not already set) onto a hypothesis object in place. Returns the same
    object for convenient chaining at synthesis call sites.
    """
    components = compute_semantic_identity_components(hypothesis)
    if hasattr(hypothesis, "temporal_scope") and not getattr(hypothesis, "temporal_scope", ""):
        hypothesis.temporal_scope = components.temporal_scope
    if hasattr(hypothesis, "direction") and not getattr(hypothesis, "direction", ""):
        hypothesis.direction = components.direction
    if hasattr(hypothesis, "canonical_identity"):
        hypothesis.canonical_identity = compute_semantic_identity(hypothesis)
    return hypothesis
