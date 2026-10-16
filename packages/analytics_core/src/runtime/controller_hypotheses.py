"""Hypothesis relationship helpers used by the investigation controller."""
from typing import Any


def _is_binary_mirror_hypothesis(counter_hyp: Any, tested_hyp: Any) -> bool:
    """True only when counter_hyp is the direct binary mirror of tested_hyp
    (same target metric/dimension/value) -- e.g. "no material association"
    vs "revenue and cost are correlated". Structurally distinct alternative
    hypotheses (a different mechanism, a different confound entirely, such
    as churn's stratification-confound or exposure-imbalance hypotheses)
    must NOT be auto-credited with evidence they were never actually tested
    by -- each needs its own dedicated experiment. Without this scoping, a
    single verified experiment against one hypothesis could prematurely
    satisfy the "every counter hypothesis has direct evidence" stopping
    gate for unrelated counter hypotheses, skipping real confound-check
    experiments the investigation still needed to run.
    """
    if counter_hyp is None or tested_hyp is None:
        return False
    return (
        getattr(counter_hyp, "target_metric", None) == getattr(tested_hyp, "target_metric", None)
        and getattr(counter_hyp, "target_dimension", None) == getattr(tested_hyp, "target_dimension", None)
    )
