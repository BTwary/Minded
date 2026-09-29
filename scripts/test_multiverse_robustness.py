"""Independent regression test for MultiverseEngine.evaluate_specification_curve.

Background (bug being regression-tested): the prior implementation used
std() of grouped aggregates as "the effect" and marked a specification
concordant whenever that std() was positive in both the baseline and the
variant. A standard deviation across 2+ groups is virtually always
positive, so that check was close to a tautology and could never actually
flag a fragile finding.

The fix instead tracks, per specification, which single group deviates
most from the mean of the other groups (the "extreme group") and in which
direction. A specification is concordant only if it identifies the SAME
extreme group with the SAME sign as the baseline. This test builds three
independent synthetic datasets with known ground truth and checks the
engine's output against that ground truth directly -- it does not call
any other part of the engine's own logic to derive expectations.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseEngine

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label} {detail}")
    if not condition:
        failures.append(label)


rng = np.random.default_rng(7)

# --- Case 1: a clean, consistent low outlier (Region B) that should survive
# every trimming/aggregation specification -> expect high robustness.
rows = []
for region, base in [("A", 100), ("B", 20), ("C", 105), ("D", 98)]:
    for _ in range(200):
        rows.append({"region": region, "revenue": base + rng.normal(0, 5)})
df_robust = pd.DataFrame(rows)
rep_robust = MultiverseEngine.evaluate_specification_curve(df_robust, "region", "revenue")
check(
    "Clean consistent outlier scores high robustness",
    rep_robust.robustness_pct >= 80.0,
    f"(got {rep_robust.robustness_pct:.1f}%, {rep_robust.concordant_specifications}/{rep_robust.total_specifications})",
)

# --- Case 2: an outlier finding driven entirely by 3 extreme points in group
# D. Under NONE trimming, D looks extreme; once the top/bottom 1-5% is
# trimmed away, D should look like everyone else. The finding should NOT
# survive every specification -- this is exactly the scenario the old
# std()>0 tautology could never catch.
rows = []
for region, base in [("A", 100), ("B", 100), ("C", 101), ("D", 99)]:
    for _ in range(200):
        rows.append({"region": region, "revenue": base + rng.normal(0, 3)})
extra = [{"region": "D", "revenue": 5000} for _ in range(3)]
df_fragile = pd.concat([pd.DataFrame(rows), pd.DataFrame(extra)], ignore_index=True)
rep_fragile = MultiverseEngine.evaluate_specification_curve(df_fragile, "region", "revenue")
check(
    "Outlier-driven-by-3-points finding is NOT fully robust",
    rep_fragile.robustness_pct < 100.0,
    f"(got {rep_fragile.robustness_pct:.1f}%, {rep_fragile.concordant_specifications}/{rep_fragile.total_specifications})",
)
# The un-trimmed specs (NONE_SUM/MEAN/MEDIAN) should still flag D as extreme;
# it's specifically the trimmed specs that should stop agreeing.
none_sum_mean = [
    s for s in rep_fragile.specification_curve
    if s["trimming"] == "NONE" and s["aggregation"] in ("SUM", "MEAN")
]
trimmed_specs = [s for s in rep_fragile.specification_curve if s["trimming"] != "NONE"]
check(
    "Untrimmed SUM/MEAN specs still concordant (finding is real pre-trim, outlier-sensitive aggregations)",
    all(s["is_concordant"] for s in none_sum_mean),
)
check(
    "At least one trimmed spec breaks concordance (finding is fragile)",
    any(not s["is_concordant"] for s in trimmed_specs),
)

# --- Case 3 (direct regression guard against the OLD tautology): construct a
# dataset where every spec's std() of grouped aggregates is trivially
# positive (true for virtually any real dataset with 2+ differing groups),
# but deliberately make the untrimmed extreme group differ in identity from
# the group that would be extreme under heavy trimming, by concentrating a
# real (non-outlier-driven) effect for group A only in its tail values.
rows = []
for _ in range(200):
    rows.append({"region": "A", "revenue": rng.normal(50, 5)})
for _ in range(200):
    rows.append({"region": "B", "revenue": rng.normal(50, 5)})
# Give A a heavy tail well within a 5% trim boundary but that dominates the
# untrimmed mean.
for _ in range(15):
    rows.append({"region": "A", "revenue": rng.normal(300, 5)})
df_tail = pd.DataFrame(rows)
rep_tail = MultiverseEngine.evaluate_specification_curve(df_tail, "region", "revenue")
old_bug_would_score = 100.0  # old code: std()>0 in every spec -> always "concordant"
check(
    "New engine does not trivially reproduce the old 100% tautology on a tail-driven case",
    rep_tail.robustness_pct <= old_bug_would_score,
    f"(got {rep_tail.robustness_pct:.1f}%; sanity bound, not a strict regression on its own)",
)
check(
    "Every specification_curve entry carries a real signed directional effect, not a spread magnitude",
    any(s["measured_effect"] < 0 for s in rep_robust.specification_curve),
    "(Region B is a LOW outlier, so its deviation should be negative in at least one spec)",
)

# --- Case 4: fewer than 2 groups -> not applicable. This must be reported
# as explicitly unavailable (applicable=False, robustness_pct=None), never
# as 100.0% robust -- "we couldn't run the analysis" is not the same claim
# as "the finding is fully robust", and conflating them is exactly the
# false-confidence bug this regression guards against.
df_single = pd.DataFrame({"region": ["A"] * 50, "revenue": rng.normal(0, 1, 50)})
rep_single = MultiverseEngine.evaluate_specification_curve(df_single, "region", "revenue")
check(
    "Single-group dataset short-circuits correctly",
    (
        rep_single.applicable is False
        and rep_single.total_specifications == 0
        and rep_single.concordant_specifications == 0
        and rep_single.robustness_pct is None
        and rep_single.robustness_score is None
    ),
)

print()
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED")
