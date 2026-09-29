"""Regression test: Dunn's post-hoc pairwise test (the Kruskal-Wallis path of
`pairwise_posthoc`) must compute each group's mean rank against its own
position in the *full family* ranking, not against an arbitrary positional
slice of a re-ranked array.

The original bug: `_dunn_pair(a, b, all_values)` re-ranked `all_values` (the
concatenation of every group in the family, in dict-insertion order) and then
took `pooled_ranks[:n1]` / `pooled_ranks[n1:n1+n2]` as "group a's" and
"group b's" mean ranks -- a purely positional slice. For any family with
more than two equal-sized groups, this silently attributes the WRONG mean
rank to whichever group doesn't happen to sit at the start of that
concatenation, and (as this test demonstrates) can make every pairwise
comparison in an equal-sized family collapse to the identical z-statistic
regardless of which two groups are actually being compared.

Uses three well-separated, non-overlapping, equal-sized groups A < B < C
(so ties/variance are identical across all three pairs) as a case where the
correct answer is analytically obvious: the A-vs-C gap in mean rank is
exactly double the A-vs-B and B-vs-C gaps, so |z_AC| should be about double
|z_AB| and |z_BC|, and the three z-statistics must NOT all be identical.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.statistics.multiple_comparisons import pairwise_posthoc


def test_dunn_posthoc_mean_ranks_reflect_full_family_not_positional_slice():
    groups = {
        "A": [1.0, 2.0, 3.0],
        "B": [10.0, 11.0, 12.0],
        "C": [20.0, 21.0, 22.0],
    }
    result = pairwise_posthoc(
        groups,
        omnibus_method="Kruskal-Wallis",
        require_significant_omnibus=False,
        correction="bonferroni",
    )
    assert result["status"] == "completed", result

    by_pair = {(c["group_a"], c["group_b"]): c for c in result["comparisons"]}
    z_ab = by_pair[("A", "B")]["statistic"]
    z_ac = by_pair[("A", "C")]["statistic"]
    z_bc = by_pair[("B", "C")]["statistic"]

    # The core regression check: with the positional-slice bug, every pair in
    # an equal-sized 3-group family collapses to the SAME z-statistic. That
    # must not happen -- A vs C (rank gap 6) is a materially different
    # comparison from A vs B or B vs C (rank gap 3 each).
    assert not (abs(z_ab - z_ac) < 1e-9 and abs(z_ab - z_bc) < 1e-9), (
        f"z_AB={z_ab}, z_AC={z_ac}, z_BC={z_bc} are all identical -- "
        "Dunn mean ranks are being sliced positionally instead of per-group"
    )

    # A vs C spans twice the rank distance of A vs B / B vs C (2, 5, 8 mean
    # ranks respectively), and all three pairs share identical n1, n2, and
    # tie structure, so |z_AC| should be ~2x |z_AB| and |z_BC|, and A-vs-B
    # should equal B-vs-C in magnitude.
    assert abs(z_ac) > abs(z_ab) * 1.5, f"expected |z_AC| >> |z_AB|, got {z_ac} vs {z_ab}"
    assert abs(z_ac) > abs(z_bc) * 1.5, f"expected |z_AC| >> |z_BC|, got {z_ac} vs {z_bc}"
    assert abs(abs(z_ab) - abs(z_bc)) < 1e-6, f"expected |z_AB| == |z_BC| by symmetry, got {z_ab} vs {z_bc}"
    assert abs(abs(z_ac) - 2 * abs(z_ab)) < 1e-6, f"expected |z_AC| ~= 2*|z_AB|, got {z_ac} vs {2 * z_ab}"

    # Sanity: A ranks lowest, C ranks highest -> A-vs-C statistic sign should
    # be negative (mean rank of A minus mean rank of C).
    assert z_ac < 0


if __name__ == "__main__":
    test_dunn_posthoc_mean_ranks_reflect_full_family_not_positional_slice()
    print("PASS: Dunn post-hoc mean ranks reflect each group's real family-wide rank.")
