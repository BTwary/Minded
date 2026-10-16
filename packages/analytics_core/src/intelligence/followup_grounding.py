"""Conservative grounding of entity references in compound follow-up questions.

The helper intentionally accepts only explicitly quoted, short labels from a
small set of recognizable result templates. Free-form answer text is untrusted
context and must not be converted directly into a query term or instruction.
"""
from __future__ import annotations

import re

_LABEL_BODY = r"(?P<label>[A-Za-z0-9][A-Za-z0-9 _./&+-]{0,79})"
_QUOTED_LABEL = rf"(?P<quote>['\"]){_LABEL_BODY}(?P=quote)"
_DANGEROUS_WORDS = re.compile(
    r"\b(?:ignore|system|developer|assistant|instruction|instructions|prompt|execute|select|drop|delete|insert|update|alter|union|override|secret|password|token)\b",
    re.I,
)

# A small, intentionally conservative set of syntactic answer shapes. The label
# must be quoted; unquoted natural-language spans are too easy to parse wrongly.
_ENTITY_PATTERNS = (
    re.compile(rf"{_QUOTED_LABEL}\s+accounts for\b", re.I),
    re.compile(rf"\bconcentrated in\s+{_QUOTED_LABEL}(?=[.,;:]|$)", re.I),
    re.compile(rf"{_QUOTED_LABEL}\s+has the highest\b", re.I),
    re.compile(rf"{_QUOTED_LABEL}\s+vs\s+runner-up\b", re.I),
    re.compile(rf"\b(?:highest|largest|greatest|most)\b.{{0,60}}\b(?:in|for)\s+{_QUOTED_LABEL}", re.I),
)

_ENTITY_TYPES = (
    "customers?", "clients?", "accounts?", "users?", "cohorts?", "segments?",
    "groups?", "products?", "orders?", "records?", "regions?", "categories?",
    "campaigns?", "channels?", "suppliers?", "employees?", "teams?", "departments?",
)
_ENTITY_NOUNS = "|".join(_ENTITY_TYPES)


def _safe_label(value: str) -> str | None:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;!?\"'")
    if not value or len(value) > 80 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _./&+-]{0,79}", value):
        return None
    if _DANGEROUS_WORDS.search(value):
        return None
    if len(value.split()) > 8:
        return None
    if value.casefold() in {"the", "this", "that", "these", "those", "it", "all", "none", "unknown"}:
        return None
    return value


def extract_compound_entity(answer: str) -> str | None:
    """Extract an entity only from a narrowly recognized, quoted result label."""
    text = " ".join(str(answer or "").split())
    for pattern in _ENTITY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _safe_label(match.group("label"))
        if candidate:
            return candidate
    return None


def ground_entity_reference(statement: str, entity_label: str) -> str:
    """Replace an explicit entity anaphora only when both pieces are safe.

    Entity-type nouns are preserved. Broad forms such as ``for those`` are never
    rewritten because the referent type is missing.
    """
    label = _safe_label(entity_label)
    if not label:
        return statement
    result = statement
    pattern = re.compile(
        rf"(?P<prep>\b(?:for|among|in|of)\s+)(?:those|these|such|the same)\s+(?P<noun>{_ENTITY_NOUNS})\b",
        re.I,
    )
    result = pattern.sub(lambda match: match.group("prep") + _with_type(label, match.group("noun")), result)
    pattern = re.compile(rf"\b(?:those|these|such|the same)\s+(?P<noun>{_ENTITY_NOUNS})\b", re.I)
    result = pattern.sub(lambda match: _with_type(label, match.group("noun")), result)
    return result


def _with_type(label: str, noun: str) -> str:
    singular = re.sub(r"s$", "", noun.casefold())
    label_last = label.casefold().split()[-1]
    if label_last == singular or label_last == noun.casefold():
        return label
    return f"{label} {noun}"
