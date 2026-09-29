"""DEPRECATED legacy Investigation Graph Engine.

AAOS_FORENSIC_AUDIT (2026-09-10, item #10 "Legacy Bayesian graph"): every
prior/posterior this module used to compute (0.40, 0.60, 0.75, 0.80, 0.85,
0.90, the 0.10/0.50/0.90 likelihood constants, a `p_value=0.0001` fallback
for a missing computed p-value, and the `primary_share`/`max_r`-derived
likelihoods in the old `update_beliefs`) was an arbitrary fixed confidence
number, not a value derived from actual evidence. That is scientifically
incompatible with AA-OS's evidence-derived-belief requirement. Canonical
Bayesian updates go through the explicit likelihood/evidence model in
`packages.analytics_core.src.runtime.controller` /
`packages.analytics_core.src.intelligence.belief_engine` instead.

A repo-wide search confirms `InvestigationGraphEngine` is not imported by
the canonical controller, runtime, or any currently-passing test -- it was
dead code left over from an earlier design. Per audit remediation option 1
("fully deprecate and prevent from participating in canonical
investigations"), instantiation fails loudly rather than staying silently
importable and inviting reuse.

2026-09-11 hardening-round-6 follow-up: the fabricated-constant method
bodies (`initialize_graph`, `update_beliefs`, `add_counter_hypothesis`,
`_rank_candidate_tests`, ~450 lines) have been removed outright rather
than kept "as a reference for a future rewrite onto the canonical
likelihood/evidence model" (audit remediation option 2). That rewrite was
never scheduled or attempted across multiple hardening rounds, and in the
meantime the unreachable fabricated numbers were mistaken for live
production logic in at least one subsequent audit pass -- unreachable
code that reads as active analytical logic is a liability, not a useful
reference, when nobody is actually going to finish option 2 with it. If
this engine is ever rebuilt on the canonical model, it should be written
fresh against `belief_engine`/`controller`, not resurrected from this
history. The guard below is what actually keeps it out of any canonical
investigation; it is unaffected by this trim and still required.
"""
from typing import Any


class InvestigationGraphEngine:
    """DEPRECATED -- do not use in canonical investigations.

    This class is permanently blocked from instantiation to prevent it
    from ever contaminating canonical AA-OS belief state with hard-coded
    priors/likelihoods. See the module docstring for the full audit
    finding.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "InvestigationGraphEngine is deprecated and disabled (AAOS_FORENSIC_AUDIT "
            "2026-09-10, item #10): its hypothesis priors and evidence likelihoods were "
            "hard-coded constants (e.g. 0.90/0.10/0.50), not evidence-derived values, so "
            "it must not participate in canonical investigations. Use the canonical "
            "controller/belief_engine pipeline instead."
        )
