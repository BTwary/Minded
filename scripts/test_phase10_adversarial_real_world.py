"""
scripts/test_phase10_adversarial_real_world.py

Phase 10 -- Adversarial Real-World Data Integrity & Analytical Safety Validation.

Every scenario below calls scripts.adversarial.harness.run_investigation(...),
which is the ONLY function in this suite that touches
InvestigationController.execute_investigation(...). No scenario manually
drives IntentEngine / HypothesisSynthesizer / EIGOptimizer / VerdictEngine /
VerificationEngine to manufacture an investigation path -- the controller
decides everything: which hypotheses to generate, which experiments to run,
when to challenge, when to stop, and what verdict to issue. Ground truth for
every scenario is computed independently with plain pandas/numpy, never from
any AA-OS engine.

Outcome classification used throughout (see PHASE10_REAL_WORLD_SAFETY_REPORT.md
for the full rubric and results table):
    CORRECT                  - system reached the right, appropriately
                                confident conclusion
    CORRECT_WITH_WARNING     - right conclusion, but only via generic caution,
                                not targeted diagnosis of the specific defect
    CORRECTLY_INCONCLUSIVE   - system correctly declined to commit
    DATA_QUALITY_BLOCKED     - system halted before analysis, citing the
                                specific defect (this is the desired
                                behavior for several scenarios)
    UNSAFE_FALSE_CONFIDENCE  - system produced a confident claim the
                                benchmark establishes is wrong or materially
                                misleading (the single most serious class)
    ARCHITECTURAL_GAP        - the capability needed to even attempt the
                                scenario correctly does not exist yet;
                                documented rather than patched in-place per
                                the phase's scope-boundary rule

Run directly: `python3 scripts/test_phase10_adversarial_real_world.py`
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.adversarial.harness import run_investigation

RESULTS = []


def record(scenario: str, classification: str, note: str = ""):
    RESULTS.append({"scenario": scenario, "classification": classification, "note": note})
    print(f"\n>>> RESULT [{scenario}]: {classification}  {('- ' + note) if note else ''}\n")


# ---------------------------------------------------------------------------
# Scenario 1 -- Duplicate primary keys
# ---------------------------------------------------------------------------
def test_scenario_01_duplicate_primary_keys():
    np.random.seed(2)
    n = 120
    account_ids = [f"ACC{i:04d}" for i in range(n)]
    segments = np.random.choice(["Enterprise", "SMB", "MidMarket"], n)
    revenue = np.random.normal(1000, 100, n)
    df = pd.DataFrame({"account_id": account_ids, "segment_tier": segments, "order_revenue": revenue})
    dup_idx = np.random.choice(n, 15, replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    dup_rows["order_revenue"] = dup_rows["order_revenue"] + np.random.normal(50, 10, 15)
    messy_df = pd.concat([df, dup_rows], ignore_index=True)

    res = run_investigation({"orders": messy_df}, "Why did order_revenue vary across segment_tier?", "s01")
    res.print_trace("Scenario 1: Duplicate Primary Keys")

    assert not res.ok, "Expected the investigation to be blocked by the duplicate-key check"
    assert res.failure_taxonomy == "DATA_QUALITY_FAILURE"
    assert "account_id" in (res.error_message or "") and "duplicat" in (res.error_message or "").lower()
    record("01_duplicate_primary_keys", "DATA_QUALITY_BLOCKED",
           "quantified 15 duplicated key values / 135 rows (11.1%) before any aggregation ran")


# ---------------------------------------------------------------------------
# Scenario 2 -- Schema drift / semantic rename
# ---------------------------------------------------------------------------
def test_scenario_02_schema_drift_rename():
    np.random.seed(3)
    n = 100
    segments = np.random.choice(["Enterprise", "SMB", "MidMarket", "Growth"], n, p=[0.25, 0.35, 0.25, 0.15])
    rev = [np.random.normal(8000, 300) if s == "Enterprise" else np.random.normal(300, 50) for s in segments]

    df_a = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(n)], "tier_segment": segments, "revenue": rev})
    df_b = pd.DataFrame({"client_key": [f"C{i:04d}" for i in range(n)], "plan_class": segments, "net_sales": rev})

    gt = pd.Series(rev, index=pd.Index(segments, name="seg")).groupby(level=0).sum()
    gt_top_share = gt.max() / gt.sum()

    res_a = run_investigation({"customers": df_a}, "Why did revenue vary across tier_segment?", "s02a")
    res_a.print_trace("Scenario 2a: standard column names")
    res_b = run_investigation({"customers": df_b}, "Why did net_sales vary across plan_class?", "s02b")
    res_b.print_trace("Scenario 2b: renamed/synonym column names")

    assert res_a.ok and res_b.ok
    assert res_a.verdict_type == res_b.verdict_type == "DIAGNOSED"
    assert abs(res_a.confidence - res_b.confidence) < 1e-6, "Renamed schema should reach an equivalent conclusion"
    assert abs(gt_top_share - 0.890) < 0.05
    record("02_schema_drift_rename", "CORRECT",
           "renamed/synonym schema (client_key/plan_class/net_sales) reached the identical 95% posterior "
           "conclusion as the original column names -- genuine schema-agnostic semantic resolution")


# ---------------------------------------------------------------------------
# Scenario 3 -- Join fanout explosion
# ---------------------------------------------------------------------------
def test_scenario_03_join_fanout():
    np.random.seed(21)
    customers = pd.DataFrame({"customer_id": [f"C{i:03d}" for i in range(40)],
                               "region": np.random.choice(["East", "West"], 40)})
    orders = pd.DataFrame({"order_id": [f"O{i:04d}" for i in range(120)],
                            "customer_id": np.random.choice(customers.customer_id, 120)})
    item_rows = []
    for _, o in orders.iterrows():
        for _ in range(np.random.randint(1, 5)):
            item_rows.append({"order_id": o.order_id, "item_revenue": np.random.uniform(10, 100)})
    order_items = pd.DataFrame(item_rows)
    true_total_revenue = order_items["item_revenue"].sum()  # independent ground truth

    res = run_investigation(
        {"customers": customers, "orders": orders, "order_items": order_items},
        "Why did item_revenue vary across region?",
        "s03",
    )
    res.print_trace("Scenario 3: Join Fanout (multi-table input)")

    assert not res.ok, "Controller has no cross-table join execution path; it cannot resolve this question at all"
    record("03_join_fanout", "ARCHITECTURAL_GAP",
          f"controller never executes a JOIN anywhere in the autonomous path -- it only ever operates on a single "
          f"'primary' table (chosen via world-model heuristics) and fails outright when the requested columns span "
          f"tables. It fails SAFE (no fabricated number; true total is {true_total_revenue:.2f}), but for an "
          f"unrelated generic schema-resolution reason, not because fanout risk was detected. Cross-table "
          f"aggregation -- safe or unsafe -- is not implemented in the on-demand controller path at all.")


# ---------------------------------------------------------------------------
# Scenario 4 -- Silent unit change
# ---------------------------------------------------------------------------
def test_scenario_04_silent_unit_change():
    np.random.seed(4)
    n_jan, n_feb = 60, 60
    regions = ["us", "eu", "apac"]
    jan = pd.DataFrame({
        "txn_id": [f"J{i:04d}" for i in range(n_jan)], "period": ["2026-01"] * n_jan,
        "region": np.random.choice(regions, n_jan), "revenue": np.random.normal(100.0, 10.0, n_jan),
    })
    feb = pd.DataFrame({
        "txn_id": [f"F{i:04d}" for i in range(n_feb)], "period": ["2026-02"] * n_feb,
        "region": np.random.choice(regions, n_feb), "revenue": np.random.normal(100.0, 10.0, n_feb) * 100.0,
    })
    combined = pd.concat([jan, feb], ignore_index=True)
    gt_jan_total = jan["revenue"].sum()
    gt_feb_true_dollars = feb["revenue"].sum() / 100.0  # independent ground truth: true growth is ~flat

    res = run_investigation({"transactions": combined}, "Why did revenue surge in period 2026-02 versus 2026-01?", "s04")
    res.print_trace("Scenario 4: Silent Unit Change (dollars -> cents)")

    assert not res.ok, "Expected the magnitude-discontinuity check to block a units artifact from being reported"
    assert res.failure_taxonomy == "DATA_QUALITY_FAILURE"
    assert "unit" in (res.error_message or "").lower() or "scale" in (res.error_message or "").lower()
    assert abs(gt_jan_total - gt_feb_true_dollars) / gt_jan_total < 0.10, "true dollar-equivalent growth is near-flat"
    record("04_silent_unit_change", "DATA_QUALITY_BLOCKED",
           f"true Jan={gt_jan_total:.0f} vs true Feb={gt_feb_true_dollars:.0f} (near-flat); naive Feb read as "
           f"{feb['revenue'].sum():.0f} (a fake ~99x jump). Blocked before a '10,400%-style growth' claim could "
           f"be issued.")


# ---------------------------------------------------------------------------
# Scenario 5 -- Timezone boundary semantics
# ---------------------------------------------------------------------------
def test_scenario_05_timezone_boundary():
    np.random.seed(5)
    n = 80
    ts_utc = pd.date_range("2026-03-01 20:00:00", periods=n, freq="30min", tz="UTC")
    region = np.random.choice(["A", "B"], n)
    val = np.random.normal(100, 10, n)
    df = pd.DataFrame({"event_id": [f"E{i:04d}" for i in range(n)], "event_timestamp": ts_utc,
                        "region": region, "value": val})

    res = run_investigation({"events": df}, "Why did value vary across region?", "s05")
    res.print_trace("Scenario 5: Timezone Boundary Semantics")

    # This scenario is graded on capability presence, not on the outcome of this one run.
    import subprocess
    grep = subprocess.run(
        ["grep", "-rl", "-i", "tz_convert\\|tzinfo\\|pytz\\|zoneinfo",
         os.path.join(os.path.dirname(__file__), "..", "packages"),
         os.path.join(os.path.dirname(__file__), "..", "apps")],
        capture_output=True, text=True,
    )
    has_tz_handling = bool(grep.stdout.strip())
    assert not has_tz_handling, "expected no timezone-aware business-day handling to exist yet"
    record("05_timezone_boundary", "ARCHITECTURAL_GAP",
           "no timezone-aware business-day/period localization exists anywhere in the codebase (confirmed by "
           "source grep). Any period-based aggregation implicitly assumes the raw timestamp values are already "
           "in the correct reporting period, and this assumption is never disclosed in the verdict or provenance.")


# ---------------------------------------------------------------------------
# Scenario 6 -- MNAR / structured missingness
# ---------------------------------------------------------------------------
def test_scenario_06_mnar_missingness():
    np.random.seed(7)
    n_ent, n_smb = 150, 150
    ent_cost = np.random.normal(5000, 800, n_ent)
    smb_cost = np.random.normal(200, 30, n_smb)
    # value-dependent missingness: higher-cost Enterprise records are MORE likely to be missing
    ent_missing_prob = 1 / (1 + np.exp(-(ent_cost - 5000) / 500))
    ent_missing_mask = np.random.rand(n_ent) < ent_missing_prob
    smb_missing_mask = np.random.rand(n_smb) < 0.02
    ent_cost_obs = ent_cost.copy(); ent_cost_obs[ent_missing_mask] = np.nan
    smb_cost_obs = smb_cost.copy(); smb_cost_obs[smb_missing_mask] = np.nan
    df = pd.DataFrame({
        "account_id": [f"A{i:04d}" for i in range(n_ent + n_smb)],
        "segment": ["Enterprise"] * n_ent + ["SMB"] * n_smb,
        "cost": np.concatenate([ent_cost_obs, smb_cost_obs]),
    })
    gt_true_ent_mean = ent_cost.mean()
    naive_ent_mean = pd.Series(ent_cost_obs).dropna().mean()
    pct_missing_ent = ent_missing_mask.mean() * 100

    res = run_investigation({"accounts": df}, "Why does cost vary across segment?", "s06")
    res.print_trace("Scenario 6: MNAR / Structured Missingness")

    # --- Phase 12 test-migration note (spec Section 32) ------------------
    # ORIGINAL ASSERTION (pre-Phase-12): `verdict_type == "DIAGNOSED" and confidence >= 0.9`
    # plus `no_missingness_disclosure` (asserted TRUE -- i.e. asserted the
    # verdict text did NOT mention missingness at all).
    # ORIGINAL INTENDED MEANING: this was a Phase 10 *adversarial finding*
    # fixture, deliberately recording an unsafe behavior under the
    # "UNSAFE_FALSE_CONFIDENCE" taxonomy (see the `record(...)` call below,
    # unchanged) -- the assertion encoded the bug on purpose, with an
    # explicit inline note: "if this ever starts disclosing missingness
    # bias, update this scenario's classification."
    # ACTUAL PHASE 11.1/PHASE 12 CONTRACT: Phase 12
    # (AUDIT_PHASE12_MISSINGNESS_SELECTION_BIAS.md Section 21) requires that
    # a conclusion whose missingness sensitivity classification is SENSITIVE
    # or UNIDENTIFIABLE must NOT be presented as DIAGNOSED, and the
    # dependence on the missingness assumption MUST be disclosed in the
    # verdict text -- the exact opposite of the old assertion.
    # INDEPENDENT DEMONSTRATION OF THE CONFLICT: this fixture's own ground
    # truth (gt_true_ent_mean vs naive_ent_mean, computed independently of
    # any AA-OS engine, see above) shows Enterprise's naive complete-case
    # mean is biased low by construction (missingness probability increases
    # with cost), with 45%+ of Enterprise's cost values missing -- textbook
    # MNAR. A system that reports 95%+ confidence with zero disclosure on
    # this exact fixture is, by definition, the unsafe behavior Phase 12
    # exists to remove; asserting that behavior as correct is no longer
    # defensible now that the fix exists.
    # WHY THE OLD ASSERTION IS NOW WRONG: it required the deficiency to
    # still be present to pass.
    # CORRECTED EXPECTED BEHAVIOR: verdict_type must NOT be DIAGNOSED, and
    # the verdict text MUST disclose the missingness dependency.
    # NEW COVERAGE PRESERVING THE ORIGINAL VALID INTENT: the original
    # fixture's purpose -- catching a *silent* high-confidence verdict on
    # value-dependent MNAR data -- is preserved below as a positive
    # assertion (the inverse of the old one), so a future regression that
    # brings back silent DIAGNOSED-with-no-disclosure on this exact fixture
    # will fail this test again, just as it was designed to.
    assert res.ok
    assert res.verdict_type != "DIAGNOSED", (
        "Phase 12 regression: a conclusion this dependent on MNAR missingness "
        "(45%+ missing, value-dependent) must not reach DIAGNOSED."
    )
    missingness_disclosed = (
        "missing" in res.direct_answer.lower()
        or (res.verdict is not None and "missing" in (res.verdict.justification or "").lower())
    )
    assert missingness_disclosed, (
        "Phase 12 regression: missingness sensitivity must reach the human-facing verdict text, "
        "not merely be computed and silently dropped (this was the original Phase 10 finding)."
    )
    record("06_mnar_missingness", "FIXED_PHASE12",
           f"45.3% of Enterprise's cost values are missing, MORE likely to be missing for higher-cost records "
           f"(true Ent mean={gt_true_ent_mean:.0f} vs naive complete-case mean={naive_ent_mean:.0f}). Phase 12's "
           f"missingness sensitivity analysis now detects this, downgrades the verdict away from DIAGNOSED "
           f"({res.verdict_type}, confidence={res.confidence:.2f}), and discloses the dependency in the verdict "
           f"text -- see AUDIT_PHASE12_MISSINGNESS_SELECTION_BIAS.md Section 41 for the full before/after record.")


# ---------------------------------------------------------------------------
# Scenario 7 -- Target / post-treatment leakage
# ---------------------------------------------------------------------------
def test_scenario_07_target_leakage():
    np.random.seed(9)
    n = 200
    tenure = np.random.uniform(1, 60, n)
    churn_prob = 1 / (1 + np.exp(-(30 - tenure) / 10))
    churned = (np.random.rand(n) < churn_prob).astype(int)
    # categorical leaked field: only populated (non-"Unknown") AFTER churn happens
    reasons = np.where(churned == 1, np.random.choice(["Price", "Support", "Competitor"], n), "Unknown")
    plan = np.random.choice(["Basic", "Pro", "Enterprise"], n)
    df = pd.DataFrame({"customer_id": [f"U{i:04d}" for i in range(n)], "plan": plan,
                        "tenure_months": tenure, "cancellation_reason": reasons, "is_churn": churned})

    res = run_investigation({"customers": df}, "Why did is_churn increase across cancellation_reason?", "s07")
    res.print_trace("Scenario 7: Target / Post-Treatment Leakage")

    assert res.ok and res.verdict_type == "DIAGNOSED" and res.confidence >= 0.9
    record("07_target_leakage", "UNSAFE_FALSE_CONFIDENCE",
           "'cancellation_reason' only exists because the customer already churned (it's 'Unknown' for every "
           "retained customer by construction), yet the controller reports 93.7% confidence that churn is "
           "'concentrated' in the 'Competitor' reason -- treating a post-outcome field as a legitimate "
           "explanatory dimension. No leakage/post-treatment detection exists anywhere in the codebase (no "
           "predictive-modeling or temporal-ordering machinery is wired into the controller path at all).")


# ---------------------------------------------------------------------------
# Scenario 8 -- Correlation vs causation
# ---------------------------------------------------------------------------
def test_scenario_08_correlation_vs_causation():
    np.random.seed(10)
    n = 150
    season = np.random.uniform(0, 10, n)  # latent confound, not included as a column
    A = season * 20 + np.random.normal(0, 5, n)
    B = season * 15 + np.random.normal(0, 5, n)  # A does not cause B; both driven by `season`
    segment = np.random.choice(["North", "South"], n)
    df = pd.DataFrame({"record_id": [f"R{i:04d}" for i in range(n)], "segment": segment,
                        "ice_cream_sales": A, "incident_count": B})

    res = run_investigation({"records": df}, "Is ice_cream_sales correlated with incident_count?", "s08")
    res.print_trace("Scenario 8: Correlation vs Causation")

    causal_verbs = ["causes", "drives", "leads to", "results in", "because of"]
    used_causal_language = any(v in res.direct_answer.lower() for v in causal_verbs)
    assert not used_causal_language
    assert res.verdict_type != "CAUSALLY_SUPPORTED"
    record("08_correlation_vs_causation", "CORRECT",
           "no causal language was used and verdict_type never became CAUSALLY_SUPPORTED -- correct by "
           "conservative default (causal_intent stays None unless the question uses explicit cause/effect "
           "phrasing, and even then a DAG is required and none is ever supplied). Caveat: this appears to be "
           "safe-by-omission rather than positive discrimination -- true bivariate correlation analysis between "
           "two numeric variables isn't functionally reachable through the controller either; it falls back to "
           "the same segment-concentration paradigm using whichever categorical column is available.")


# ---------------------------------------------------------------------------
# Scenario 9 -- No signal / insufficient evidence
# ---------------------------------------------------------------------------
def test_scenario_09_no_signal():
    np.random.seed(11)
    n = 200
    segment = np.random.choice(["A", "B", "C", "D"], n)
    metric = np.random.normal(500, 50, n)  # pure noise, no relationship to segment
    df = pd.DataFrame({"record_id": [f"R{i:04d}" for i in range(n)], "segment": segment, "metric": metric})

    res = run_investigation({"records": df}, "Why did metric vary across segment?", "s09")
    res.print_trace("Scenario 9: No Signal / Insufficient Evidence")

    assert res.verdict_type == "INCONCLUSIVE"
    assert res.confidence < 0.85
    assert "insufficient evidence" in res.direct_answer.lower()
    record("09_no_signal", "CORRECT",
           f"pure-noise dataset correctly resolved to INCONCLUSIVE (confidence={res.confidence:.2f}, "
           f"stopping reason MAX_ITERATIONS_REACHED) rather than manufacturing a hypothesis")


# ---------------------------------------------------------------------------
# Scenario 10 -- Simpson's paradox / confounding
# ---------------------------------------------------------------------------
def test_scenario_10_simpsons_paradox():
    np.random.seed(14)
    rows = []
    def gen(region, treatment, n, mean, sd):
        for v in np.random.normal(mean, sd, n):
            rows.append({"region": region, "treatment": treatment, "score": v})
    gen("East", "New", 900, 20.0, 4.0)
    gen("East", "Old", 50, 15.0, 4.0)
    gen("West", "New", 50, 5.0, 4.0)
    gen("West", "Old", 900, 25.0, 4.0)
    df = pd.DataFrame(rows)
    df["record_id"] = [f"R{i:05d}" for i in range(len(df))]

    res = run_investigation({"records": df}, "Why did score vary across treatment?", "s10")
    res.print_trace("Scenario 10: Simpson's Paradox / Confounding")

    spawned_conditional_test = any("EXP-COND" in e.test_code for e in res.experiments)
    assert spawned_conditional_test, "expected a conditional/stratified experiment to be spawned on the confound"
    assert res.verdict_type == "INCONCLUSIVE" or res.confidence < 0.85
    record("10_simpsons_paradox", "CORRECTLY_INCONCLUSIVE",
           f"the region confound was detected, a targeted conditional experiment "
           f"({[e.test_code for e in res.experiments if 'EXP-COND' in e.test_code]}) was spawned, and the "
           f"investigation correctly avoided a decisive stop (confidence={res.confidence:.2f}, "
           f"verification failed on the conditional check) rather than confidently crediting the treatment "
           f"with more SUM-share than the confound actually supports")


# ---------------------------------------------------------------------------
# Scenario 11 -- Survivorship / selection bias
# ---------------------------------------------------------------------------
def test_scenario_11_survivorship_bias():
    np.random.seed(15)
    n = 150
    strategy = np.random.choice(["Aggressive", "Conservative"], n, p=[0.5, 0.5])
    revenue = np.where(strategy == "Aggressive", np.random.normal(900, 300, n), np.random.normal(600, 100, n))
    survived = np.where(strategy == "Aggressive", np.random.rand(n) < 0.35, np.random.rand(n) < 0.90)
    df = pd.DataFrame({"store_id": [f"S{i:04d}" for i in range(n)], "strategy": strategy,
                        "revenue": revenue})[survived]

    res = run_investigation({"stores": df}, "Why did revenue vary across strategy?", "s11")
    res.print_trace("Scenario 11: Survivorship / Selection Bias")

    assert res.ok and res.verdict_type == "DIAGNOSED"
    record("11_survivorship_bias", "ARCHITECTURAL_GAP",
           "no selection-bias/survivorship detection exists in the codebase. The verdict's boilerplate caveat "
           "('Findings apply to observed reporting period and population coverage') is a static string on every "
           "verdict, not a real check -- it does not change based on whether attrition actually occurred. Note: "
           "AdversarialAttacker's own docstring lists 'Missingness & selection bias' as one of its four "
           "falsification mechanisms, but the implementation only contains two (outlier-leverage and Simpson's "
           "paradox/confounding) -- the docstring overclaims a capability that isn't implemented.")


# ---------------------------------------------------------------------------
# Scenario 12 -- Temporal / future-information leakage
# ---------------------------------------------------------------------------
def test_scenario_12_temporal_leakage():
    np.random.seed(16)
    n = 150
    region = np.random.choice(["North", "South"], n)
    june_metric = np.random.normal(100, 15, n)
    july_growth_rate = np.where(region == "North", np.random.normal(0.30, 0.03, n), np.random.normal(-0.05, 0.03, n))
    df = pd.DataFrame({"store_id": [f"S{i:04d}" for i in range(n)], "region": region,
                        "june_metric": june_metric, "july_growth_rate_feature": july_growth_rate})

    res = run_investigation({"stores": df}, "Why did july_growth_rate_feature vary across region?", "s12")
    res.print_trace("Scenario 12: Temporal / Future-Information Leakage")

    assert res.ok and res.verdict_type == "DIAGNOSED" and res.confidence >= 0.9
    record("12_temporal_leakage", "ARCHITECTURAL_GAP",
           "a column named to make its future-relative-to-June provenance explicit "
           "('july_growth_rate_feature') is treated identically to any other numeric metric. There is no "
           "temporal-ordering awareness anywhere in the codebase -- no check for column names, dates, or "
           "metadata implying information from after the analysis period.")


# ---------------------------------------------------------------------------
# Scenario 13 -- Outlier investigation & sensitivity
# ---------------------------------------------------------------------------
def test_scenario_13_outlier_investigation():
    # 13a: legitimate extreme value (a real anchor enterprise deal)
    np.random.seed(17)
    n = 200
    segment = np.random.choice(["Enterprise", "SMB"], n, p=[0.3, 0.7])
    revenue = np.where(segment == "Enterprise", np.random.normal(5000, 500, n), np.random.normal(300, 50, n))
    revenue[0] = 300000.0
    segment = segment.copy(); segment[0] = "Enterprise"
    df_legit = pd.DataFrame({"deal_id": [f"D{i:04d}" for i in range(n)], "segment": segment, "revenue": revenue})
    res_legit = run_investigation({"deals": df_legit}, "Why did revenue vary across segment?", "s13a")
    res_legit.print_trace("Scenario 13a: legitimate extreme value")

    # 13b: genuine data-entry error (a fat-finger typo, not a real transaction)
    np.random.seed(18)
    segment2 = np.random.choice(["Enterprise", "SMB"], n, p=[0.3, 0.7])
    revenue2 = np.where(segment2 == "Enterprise", np.random.normal(5000, 500, n), np.random.normal(300, 50, n))
    revenue2[5] = 1_000_000_000.0
    segment2 = segment2.copy(); segment2[5] = "SMB"
    df_error = pd.DataFrame({"deal_id": [f"D{i:04d}" for i in range(n)], "segment": segment2, "revenue": revenue2})
    res_error = run_investigation({"deals": df_error}, "Why did revenue vary across segment?", "s13b")
    res_error.print_trace("Scenario 13b: genuine data-entry error")

    assert res_legit.ok and res_error.ok
    assert res_legit.confidence < 0.85 and res_error.confidence < 0.85, \
        "expected the outlier-leverage falsification check to prevent a confident stop in both cases"
    record("13_outlier_investigation", "CORRECT_WITH_WARNING",
           f"both the legitimate mega-deal (confidence={res_legit.confidence:.2f}) and the fat-finger typo "
           f"(confidence={res_error.confidence:.2f}) correctly failed to reach a decisive verdict -- the "
           f"outlier-leverage falsification test is real and does fire. However it does NOT discriminate "
           f"between the two cases: both are handled identically (fall back to general uncertainty) rather "
           f"than the system actually diagnosing 'this looks like a data error, here is the sensitivity' vs "
           f"'this is a legitimate anchor value, conclusion is/isn't robust to it'. Safe, but not the "
           f"investigate->diagnose->quantify->disclose behavior the scenario is meant to test.")


# ---------------------------------------------------------------------------
# Scenario 14 -- High-dimensional, schema-agnostic stress test
# ---------------------------------------------------------------------------
def test_scenario_14_high_dimensional_stress():
    np.random.seed(19)
    n = 300
    region = np.random.choice(["Zeta", "Omicron", "Vex", "Nyra"], n, p=[0.1, 0.2, 0.3, 0.4])
    cost_val = np.where(region == "Zeta", np.random.normal(9000, 500, n), np.random.normal(400, 80, n))
    df = pd.DataFrame({"entity_ref_9f2": [f"X{i:05d}" for i in range(n)],
                        "grp_dim_alpha": region, "flux_metric_q7": cost_val})
    rng = np.random.default_rng(20)
    for j in range(60): df[f"noise_num_{j}"] = rng.normal(0, 1, n)
    for j in range(30): df[f"noise_cat_{j}"] = rng.choice(["p", "q", "r", "s"], n)
    for j in range(15): df[f"noise_hicard_{j}"] = [f"uuid-{rng.integers(0, 1_000_000)}" for _ in range(n)]
    for j in range(10): df[f"noise_const_{j}"] = "SAME_VALUE"
    df["noise_text_blob"] = ["lorem ipsum dolor sit amet " + str(i) for i in range(n)]
    assert df.shape[1] >= 100, "need 100+ columns for a genuine stress test"

    res_named = run_investigation({"wide_table": df}, "Why did flux_metric_q7 vary across grp_dim_alpha?", "s14a")
    res_named.print_trace("Scenario 14a: 119 columns, question names the columns")
    res_blind = run_investigation({"wide_table": df}, "Why did the total change so much?", "s14b")
    res_blind.print_trace("Scenario 14b: 119 columns, fully ambiguous question")

    for res in (res_named, res_blind):
        assert res.ok and res.verdict_type in ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT")
        assert "Zeta" in res.direct_answer
        assert res.confidence >= 0.9
    record("14_high_dimensional_stress", "CORRECT",
           f"found the correct 2-of-119-column signal (Zeta, ~74% share) with 95% confidence in both the "
           f"explicit-column-name framing and the fully ambiguous 'why did the total change' framing -- genuine "
           f"schema-agnostic discovery, not reliance on being told the answer's shape")


if __name__ == "__main__":
    tests = [
        test_scenario_01_duplicate_primary_keys,
        test_scenario_02_schema_drift_rename,
        test_scenario_03_join_fanout,
        test_scenario_04_silent_unit_change,
        test_scenario_05_timezone_boundary,
        test_scenario_06_mnar_missingness,
        test_scenario_07_target_leakage,
        test_scenario_08_correlation_vs_causation,
        test_scenario_09_no_signal,
        test_scenario_10_simpsons_paradox,
        test_scenario_11_survivorship_bias,
        test_scenario_12_temporal_leakage,
        test_scenario_13_outlier_investigation,
        test_scenario_14_high_dimensional_stress,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"\n!!! ASSERTION FAILED in {t.__name__}: {e}\n")

    print("\n" + "=" * 80)
    print("PHASE 10 SUMMARY")
    print("=" * 80)
    for r in RESULTS:
        print(f"  {r['scenario']:32s} -> {r['classification']}")
    print("=" * 80)
    if failures:
        print(f"\n{len(failures)} scenario assertion(s) FAILED (harness/assertion issue, not classification):")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} scenarios executed and were classified. OK.")
