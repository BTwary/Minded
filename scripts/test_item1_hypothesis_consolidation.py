"""Item 1 regression suite: canonical hypothesis identity + consolidation.

Covers, per the required test list:
  - same batch -> merge
  - different rounds -> merge
  - different wording -> merge
  - same dimension / different value -> don't merge
  - same value / different metric -> don't merge
  - same metric / different period -> don't merge
  - different mechanism -> don't merge
  - duplicate evidence -> don't duplicate

State-reconstruction-after-restart and full-controller-loop coverage live
in test_item1_state_reconstruction.py and are exercised against a real
sqlite file (a fresh OS process re-opening the file), not an in-memory
simulation.

Run: python3 scripts/test_item1_hypothesis_consolidation.py
"""
import sys
from dataclasses import replace

sys.path.insert(0, ".")

from packages.analytics_core.src.intelligence.hypothesis_identity import (
    compute_semantic_identity,
)
from packages.analytics_core.src.intelligence.hypothesis_consolidation import (
    HypothesisConsolidator,
)
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    PredictiveHypothesis,
)

FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        FAILURES.append(name)


def make_hyp(code, claim, mechanism, metric="revenue", dim="region", value="Region B",
             mechanism_detail="", generated_reason="", direction="decrease"):
    h = PredictiveHypothesis(
        id=code,
        hypothesis_code=code,
        claim=claim,
        mechanism=mechanism,
        predicted_observables_if_true=["x"],
        predicted_observables_if_false=["y"],
        falsification_criteria="z",
        required_assumptions=[],
        prior_probability=0.4,
        posterior_probability=0.4,
        target_metric=metric,
        target_dimension=dim,
        target_value=value,
        mechanism_detail=mechanism_detail,
        generated_reason=generated_reason,
        source_evidence=[f"EXP-{code}"],
    )
    # In production this is stamped from SemanticResolution.direction_hint
    # (resolved once from the investigation's question) -- explicit here so
    # these unit tests exercise the same reliable signal rather than
    # depending on whichever words happen to appear in a claim sentence.
    h.direction = direction
    h.canonical_identity = compute_semantic_identity(h)
    h.consolidation_log.append({"event": "created", "hypothesis_code": code, "claim": claim})
    return h


# ---------------------------------------------------------------------------
# 1. Same batch -> merge (3 patterns pointing at Region B collapse to 1 + 3
#    evidence contributions)
# ---------------------------------------------------------------------------
batch = [
    make_hyp("HYP-10", "Region B caused the March revenue decline in concentration terms.",
             "Unequal shift concentrated in Region B partitions.", generated_reason="March 2026 decline"),
    make_hyp("HYP-11", "Revenue decline in March was driven by Region B (segment dispersion).",
             "Categorical partitioning across region reveals heterogeneity.", generated_reason="March 2026 decline"),
    make_hyp("HYP-12", "Concentration in Region B is the primary driver of the March revenue variance.",
             "Specific localized concentration in Region B driving aggregate revenue.", generated_reason="March 2026 decline"),
]
consolidated = HypothesisConsolidator.consolidate_batch(batch)
check("same batch -> merges to exactly 1 canonical hypothesis", len(consolidated) == 1)
check(
    "same batch -> merged hypothesis records 3 evidence contributions",
    len(consolidated[0].consolidation_log) == 3 if consolidated else False,
)
check(
    "same batch -> survivor keeps its own original hypothesis_code",
    consolidated[0].hypothesis_code == "HYP-10" if consolidated else False,
)
check(
    "same batch -> source_evidence union preserved for all three",
    consolidated[0].source_evidence == ["EXP-HYP-10", "EXP-HYP-11", "EXP-HYP-12"] if consolidated else False,
)

# ---------------------------------------------------------------------------
# 2. Different rounds -> merge (cross-round via register_or_merge)
# ---------------------------------------------------------------------------
registry = {}
round1 = make_hyp("HYP-20", "Region B caused the March revenue decline.",
                   "Unequal shift concentrated in Region B partitions.", generated_reason="March 2026 decline")
action1, code1, obj1 = HypothesisConsolidator.register_or_merge(registry, round1)
check("round 1 registration is a fresh 'created'", action1 == "created")

round2 = make_hyp("HYP-21", "Region B is again implicated in the March revenue decline (round 2).",
                   "Unequal shift concentrated in Region B partitions.", generated_reason="March 2026 decline, round 2")
action2, code2, obj2 = HypothesisConsolidator.register_or_merge(registry, round2)
check("different rounds -> second round merges into the first", action2 == "merged")
check("different rounds -> canonical code is the original round-1 code", code2 == "HYP-20")
check("different rounds -> registry has exactly one live entry", len(registry) == 1)
check(
    "different rounds -> original hypothesis id/code preserved on survivor",
    registry["HYP-20"].hypothesis_code == "HYP-20" and registry["HYP-20"].id == "HYP-20",
)
check(
    "different rounds -> posterior_history captures the absorbed round",
    any(e.get("merged_from_hypothesis_code") == "HYP-21" for e in registry["HYP-20"].posterior_history),
)
check(
    "different rounds -> provenance.consolidated_from records the merge",
    any(e.get("hypothesis_code") == "HYP-21" for e in registry["HYP-20"].provenance.get("consolidated_from", [])),
)

# ---------------------------------------------------------------------------
# 3. Different wording -> merge
# ---------------------------------------------------------------------------
h_a = make_hyp("HYP-30", "Region B caused the March revenue decline.",
               "Unequal shift concentrated in Region B partitions.")
h_b = make_hyp("HYP-31", "Revenue decline in March was driven by Region B.",
               "Unequal shift concentrated in Region B partitions.")
check(
    "different wording, same substance -> identical canonical identity",
    compute_semantic_identity(h_a) == compute_semantic_identity(h_b),
)

# ---------------------------------------------------------------------------
# 4. Non-merge cases -- every one of these must remain a DISTINCT identity
# ---------------------------------------------------------------------------
base = make_hyp("HYP-40", "Region B drove the March revenue decline.",
                "Unequal shift concentrated in Region B partitions.",
                metric="revenue", dim="region", value="Region B")

same_dim_diff_value = make_hyp("HYP-41", "Region C drove the March revenue decline.",
                                "Unequal shift concentrated in Region C partitions.",
                                metric="revenue", dim="region", value="Region C")
check(
    "same dimension / different value -> distinct identity",
    compute_semantic_identity(base) != compute_semantic_identity(same_dim_diff_value),
)

same_value_diff_metric = make_hyp("HYP-42", "Region B drove the March profit decline.",
                                   "Unequal shift concentrated in Region B partitions.",
                                   metric="profit", dim="region", value="Region B")
check(
    "same value / different metric -> distinct identity",
    compute_semantic_identity(base) != compute_semantic_identity(same_value_diff_metric),
)

same_metric_diff_period = make_hyp("HYP-43", "Region B drove the April revenue decline.",
                                    "Unequal shift concentrated in Region B partitions.",
                                    metric="revenue", dim="region", value="Region B")
check(
    "same metric / different period -> distinct identity",
    compute_semantic_identity(base) != compute_semantic_identity(same_metric_diff_period),
)

different_mechanism = make_hyp("HYP-44", "Region B's apparent effect on March revenue is confounded by channel mix.",
                                "Simpson's paradox mediated by channel mix masquerading as a Region B effect.",
                                metric="revenue", dim="region", value="Region B")
check(
    "different mechanism (concentration vs confounding) -> distinct identity",
    compute_semantic_identity(base) != compute_semantic_identity(different_mechanism),
)

different_direction = make_hyp("HYP-45", "Region B drove a jump in March revenue.",
                                "Unequal shift concentrated in Region B partitions.",
                                metric="revenue", dim="region", value="Region B", direction="increase")
check(
    "same everything / different direction -> distinct identity",
    compute_semantic_identity(base) != compute_semantic_identity(different_direction),
)

# ---------------------------------------------------------------------------
# 5. Duplicate evidence -> don't duplicate
# ---------------------------------------------------------------------------
survivor = make_hyp("HYP-50", "Region B drove the March revenue decline.",
                     "Unequal shift concentrated in Region B partitions.")
survivor.supporting_evidence_ids = ["EV-001", "EV-002"]
dup = make_hyp("HYP-51", "Revenue decline in March traces back to Region B.",
               "Unequal shift concentrated in Region B partitions.")
dup.supporting_evidence_ids = ["EV-002", "EV-003"]  # EV-002 overlaps
reg2 = {"HYP-50": survivor}
action3, code3, merged_obj = HypothesisConsolidator.register_or_merge(reg2, dup)
check("duplicate evidence -> merge occurs", action3 == "merged")
check(
    "duplicate evidence -> supporting_evidence_ids deduplicated, not repeated",
    merged_obj.supporting_evidence_ids == ["EV-001", "EV-002", "EV-003"],
)

# ---------------------------------------------------------------------------
# 6. Synthesizer-level integration: three same-round emergent patterns on the
#    same segment collapse via synthesize_emergent_hypotheses itself.
# ---------------------------------------------------------------------------
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
from packages.analytics_core.src.engines.semantic import SemanticResolution

semantic = SemanticResolution(
    primary_dataset_name="orders",
    target_metric_col="revenue",
    group_dimension_col="region",
    time_col="order_date",
    table_grain="row",
    available_numeric_cols=["revenue"],
    available_categorical_cols=["region", "channel"],
    direction_hint="decrease",
)
patterns = [
    {"type": "concentration", "dimension": "region", "metric": "revenue",
     "segment_value": "Region B", "top_share_pct": 62.0, "source_experiment_id": "EXP-A"},
    {"type": "segment_difference", "dimension": "region", "metric": "revenue",
     "segment_value": "Region B", "top_share_pct": 40.0, "source_experiment_id": "EXP-B"},
    {"type": "anomaly", "dimension": "region", "metric": "revenue",
     "segment_value": "Region B", "top_share_pct": 12.0, "source_experiment_id": "EXP-C"},
]
emergent = HypothesisSynthesizer.synthesize_emergent_hypotheses(patterns, semantic, existing_hyps=[])
check(
    "synthesizer: 3 same-segment patterns in one call -> 1 canonical hypothesis",
    len(emergent) == 1,
)

unrelated_patterns = patterns + [
    {"type": "concentration", "dimension": "channel", "metric": "revenue",
     "segment_value": "Online", "top_share_pct": 55.0, "source_experiment_id": "EXP-D"},
]
emergent2 = HypothesisSynthesizer.synthesize_emergent_hypotheses(unrelated_patterns, semantic, existing_hyps=[])
check(
    "synthesizer: an unrelated dimension pattern is NOT folded in",
    len(emergent2) == 2,
)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL ITEM 1 CONSOLIDATION TESTS PASSED")
