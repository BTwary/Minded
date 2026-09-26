"""HypothesisConsolidator: the single authoritative merge path for hypotheses.

Two situations need consolidation, and both funnel through the same
``_merge_pair`` logic so there is exactly one definition of "what merging
means":

1. Same-batch duplication -- several detectors or synthesis passes in a
   single round independently propose hypotheses that turn out to share a
   canonical_identity (see ``hypothesis_identity.py``). ``consolidate_batch``
   collapses these before they ever reach the state manager.

2. Cross-round duplication -- a hypothesis proposed in round N has the same
   canonical_identity as one already registered from an earlier round (or
   with different wording). ``register_or_merge`` is the entry point the
   state manager uses for this case.

Merging is always additive: the surviving hypothesis keeps its original id,
hypothesis_code, canonical_identity, and current belief (prior/posterior
probability are NEVER touched here -- that is the Bayesian update engine's
job, not consolidation's). What gets merged in is provenance: source
evidence, supporting/contradicting evidence ids, experiment references,
prediction ids, posterior history, and the consolidation log. The absorbed
hypothesis is never silently discarded -- its own consolidation_log
(including its own "created" entry) is folded into the survivor's, so the
survivor's consolidation_log length is exactly the count of evidence
contributions behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity


def _union(a: List[Any], b: List[Any]) -> List[Any]:
    out = list(a)
    for item in b:
        if item not in out:
            out.append(item)
    return out


def _getattr_list(obj: Any, name: str) -> List[Any]:
    val = getattr(obj, name, None)
    return list(val) if val else []


def _merge_pair(survivor: Any, absorbed: Any) -> Any:
    """Fold ``absorbed`` into ``survivor`` in place. Returns survivor.

    Never overwrites survivor.id / hypothesis_code / canonical_identity /
    prior_probability / posterior_probability -- those belong to the
    survivor and are left exactly as they were before the merge.

    Uses defensive getattr/setattr throughout so it tolerates lightweight
    duck-typed hypothesis objects (e.g. legacy test/registration helpers)
    that don't carry the full PredictiveHypothesis provenance fields --
    those fields are simply treated as empty rather than raising.
    """
    for list_field in (
        "source_evidence", "supporting_evidence_ids", "contradicting_evidence_ids",
        "experiment_refs", "parent_hypotheses", "prediction_ids",
    ):
        if hasattr(survivor, list_field):
            setattr(survivor, list_field, _union(_getattr_list(survivor, list_field), _getattr_list(absorbed, list_field)))

    if hasattr(survivor, "posterior_history"):
        survivor.posterior_history = _getattr_list(survivor, "posterior_history")
        survivor.posterior_history.append({
            "merged_from_hypothesis_code": getattr(absorbed, "hypothesis_code", ""),
            "merged_from_id": getattr(absorbed, "id", ""),
            "prior_probability": getattr(absorbed, "prior_probability", None),
            "posterior_probability": getattr(absorbed, "posterior_probability", None),
        })

    if hasattr(survivor, "provenance"):
        prov = getattr(survivor, "provenance", None) or {}
        prov.setdefault("consolidated_from", [])
        prov["consolidated_from"].append({
            "hypothesis_code": getattr(absorbed, "hypothesis_code", ""),
            "id": getattr(absorbed, "id", ""),
            "canonical_identity": getattr(absorbed, "canonical_identity", ""),
            "claim": getattr(absorbed, "claim", ""),
        })
        survivor.provenance = prov

    # Fold the absorbed hypothesis's own contribution history in wholesale
    # (it may itself already represent multiple prior merges) so the
    # survivor's consolidation_log length is exactly the number of original
    # evidence contributions behind it -- one "created" entry per
    # originating pattern/hypothesis, full stop. The merge lineage itself
    # (who got merged into whom) is recorded separately in `provenance`
    # above, so it isn't double-counted here as a second log entry.
    if hasattr(survivor, "consolidation_log"):
        survivor.consolidation_log = _getattr_list(survivor, "consolidation_log")
        absorbed_log = _getattr_list(absorbed, "consolidation_log")
        if absorbed_log:
            survivor.consolidation_log.extend(absorbed_log)
        else:
            survivor.consolidation_log.append({
                "event": "created",
                "hypothesis_code": getattr(absorbed, "hypothesis_code", ""),
                "claim": getattr(absorbed, "claim", ""),
            })

    absorbed_claim = getattr(absorbed, "claim", "") or ""
    survivor_claim = getattr(survivor, "claim", "") or ""
    if absorbed_claim and absorbed_claim != survivor_claim and hasattr(survivor, "generated_reason"):
        note = f"Consolidated duplicate claim ({getattr(absorbed, 'hypothesis_code', '')}): {absorbed_claim}"
        existing_reason = getattr(survivor, "generated_reason", "") or ""
        survivor.generated_reason = f"{existing_reason} | {note}" if existing_reason else note

    return survivor


@dataclass
class ConsolidationResult:
    """Outcome of registering a batch of incoming hypotheses against a registry."""
    registry: Dict[str, Any]
    created: List[str] = field(default_factory=list)
    merges: List[Dict[str, Any]] = field(default_factory=list)


class HypothesisConsolidator:
    """The single authoritative path for merging duplicate hypotheses."""

    @staticmethod
    def consolidate_batch(hypotheses: List[Any]) -> List[Any]:
        """Collapse same-batch duplicates (matching canonical_identity) into
        one canonical hypothesis each. Order-preserving: the first
        occurrence of an identity is the survivor.
        """
        survivors_by_identity: Dict[str, Any] = {}
        ordered: List[Any] = []
        for h in hypotheses:
            identity = h.canonical_identity or compute_semantic_identity(h)
            h.canonical_identity = identity
            existing = survivors_by_identity.get(identity)
            if existing is None:
                survivors_by_identity[identity] = h
                ordered.append(h)
            else:
                _merge_pair(existing, h)
        return ordered

    @staticmethod
    def register_or_merge(
        registry: Dict[str, Any],
        incoming: Any,
    ) -> Tuple[str, str, Any]:
        """Register ``incoming`` into ``registry`` (keyed by hypothesis_code),
        merging into an existing entry with the same canonical_identity if
        one exists.

        Returns (action, canonical_code, resulting_object) where action is
        "created" or "merged". The registry dict is mutated in place;
        callers should not also insert ``incoming`` under its own code
        after calling this.
        """
        identity = getattr(incoming, "canonical_identity", None) or compute_semantic_identity(incoming)
        setattr(incoming, "canonical_identity", identity)

        for existing_code, existing_h in registry.items():
            existing_identity = getattr(existing_h, "canonical_identity", None) or compute_semantic_identity(existing_h)
            setattr(existing_h, "canonical_identity", existing_identity)
            if existing_identity == identity and existing_code != incoming.hypothesis_code:
                merged = _merge_pair(existing_h, incoming)
                return "merged", existing_code, merged
            if existing_code == incoming.hypothesis_code and existing_identity == identity:
                # Re-registration of the exact same object/code (e.g. a
                # reconstruction replay) -- treat as idempotent update, not
                # a fresh creation and not a merge-into-a-different-code.
                registry[existing_code] = incoming
                return "created", existing_code, incoming

        registry[incoming.hypothesis_code] = incoming
        return "created", incoming.hypothesis_code, incoming
