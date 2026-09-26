"""Analyst-facing wording for an adversarially refuted headline comparison.

When ``detect_simpsons_reversal`` (materiality / min-cell-size / same-sign
gated) finds that a group comparison reverses once a secondary dimension is
controlled for, the human analyst needs to be told *what* was found, *why* the
headline number should not be trusted, and *what to do next* -- not a generic
"inconclusive" and not an internal posterior.  This module is the single source
of that wording so the verdict engine and the direct-answer composer cannot
drift apart.
"""
from typing import Any, Dict, List, Mapping, Tuple


def _dir(x: float) -> str:
    return "higher" if x > 0 else "lower"


def describe_simpsons_refutation(refutation: Mapping[str, Any]) -> Tuple[str, str, List[str]]:
    """Return ``(headline, finding, next_steps)`` for a detected reversal.

    ``headline`` is one sentence (it is rendered as the page headline),
    ``finding`` carries the concrete numbers and the plain-language reading, and
    ``next_steps`` are concrete things the analyst can do.
    """
    r: Dict[str, Any] = dict(refutation)
    a, b = (list(r.get("groups") or ["group A", "group B"]) + ["group B"])[:2]
    dim = r.get("primary_dimension") or "the compared dimension"
    sec = r.get("secondary_dimension") or "a secondary dimension"
    metric = r.get("metric") or "the metric"
    marg = float(r.get("marginal_difference", 0.0))
    adj = float(r.get("adjusted_difference", 0.0))
    n = int(r.get("strata_used", 0))

    headline = (
        f"The apparent difference in {metric} across {dim} is not supported: "
        f"it reverses once results are compared within {sec}."
    )
    finding = (
        f"On the pooled data, mean {metric} for '{a}' is {abs(marg):.4g} {_dir(marg)} than for '{b}'. "
        f"But within {sec} (across {n} {sec} groups with enough rows in both), it is {abs(adj):.4g} "
        f"{_dir(adj)} -- the opposite direction (Simpson's paradox). "
        f"The pooled gap most likely reflects how {sec} is distributed across {dim} groups rather than "
        f"an effect of {dim} itself. Compare {dim} within {sec} instead of on the pooled total. "
        f"This result does not by itself show that the reversed direction is a real effect, and other "
        f"confounders may remain."
    )
    next_steps = [
        f"Report the {dim} comparison stratified by {sec}, not the pooled difference.",
        f"Check why {sec} is unevenly distributed across {dim} groups (assignment, eligibility or sampling).",
        f"Adjust for {sec} (and other plausible confounders) before drawing conclusions about {dim}.",
    ]
    return headline, finding, next_steps
