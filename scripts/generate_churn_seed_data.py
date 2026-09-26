"""
generate_churn_seed_data.py

DEFECT-005 (churn/segmentation identifiability) — synthetic seed data.

The existing benchmark seed data (`data/seed/customers.csv`) has NO
churn/cancellation/attrition signal at all (documented in
DEFECT_005_ROOT_CAUSE.md Sec 5.2). None of the identifiability logic this
defect is about (exposure adjustment, censoring, cohort/tenure confounding,
Simpson's-paradox-style composition shifts) can be exercised without a
dataset that actually has churn events, tenure, and observation windows.

This script generates ONE denormalized table, `data/seed/churn.csv`
(the architecture is single-table only — see ROOT_CAUSE Sec 5.2 — so all
fields needed for identifiability analysis live on one row per customer,
not split across tables requiring a JOIN).

Columns:
    customer_id        unique id
    segment             the variable under test ("Enterprise"/"SMB"/"Consumer")
    cohort               signup cohort, "2025-H1" / "2025-H2" (confounder)
    tenure_days           days between signup and observation_end OR churn
    observation_days     the customer's total exposure / person-time at risk
    signup_date
    observation_end_date  end of the observation window for this customer
    churn_event           1 = confirmed churn (event observed), 0 = did not churn
    censored              1 = observation window ended before a churn event
                              could be confirmed (right-censored), 0 = fully observed

`churned_event=1` and `censored=1` are mutually exclusive. A row with
`churn_event=0, censored=0` is a CONFIRMED non-churner (observed for the
full window and did not churn). A row with `churn_event=0, censored=1` is
NOT a confirmed non-churner — it is "not observed long enough to establish
non-churn" (Sec 6 of the defect brief). This distinction is the entire
point of the censoring fix and must never be collapsed by downstream code.

Five independent, self-contained scenario generators are provided for the
known-answer matrix (Sec 12 of the defect brief). Each returns a DataFrame
plus a dict of independently-computed ground-truth statistics so tests can
assert against arithmetic derived separately from the data-generating
process, not from AA-OS's own code.
"""
import os
import random

import numpy as np
import pandas as pd

SEED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "seed")
os.makedirs(SEED_DIR, exist_ok=True)

RNG_SEED = 42


def _make_customer(cid, segment, cohort, tenure_days, observation_days,
                    churn_event, censored):
    assert not (churn_event and censored), "event and censoring are mutually exclusive"
    return {
        "customer_id": cid,
        "segment": segment,
        "cohort": cohort,
        "tenure_days": tenure_days,
        "observation_days": observation_days,
        "signup_date": "2025-01-01",
        "observation_end_date": "2026-01-01",
        "churn_event": int(churn_event),
        "censored": int(censored),
    }


def scenario_raw_count_trap(rng):
    """Case 1: raw event count says A > B; correct RATE says B > A.

    Segment A: 900 customers, 90 churn events  -> rate 10.0%
    Segment B: 100 customers, 15 churn events  -> rate 15.0%
    Raw counts: A=90 > B=15 (naive "A is worse"). Rate: B > A (B is worse).
    Equal exposure (observation_days=365 for everyone) so this isolates the
    denominator bug specifically, not an exposure confound.
    """
    rows = []
    cid = 1
    for _ in range(900):
        churn = rng.random() < 0.10
        rows.append(_make_customer(f"CUST-{cid:05d}", "A", "2025-H1", 365, 365, churn, False))
        cid += 1
    for _ in range(100):
        churn = rng.random() < 0.15
        rows.append(_make_customer(f"CUST-{cid:05d}", "B", "2025-H1", 365, 365, churn, False))
        cid += 1
    df = pd.DataFrame(rows)
    truth = {
        "A_customers": 900, "A_events": int(df[df.segment == "A"].churn_event.sum()),
        "B_customers": 100, "B_events": int(df[df.segment == "B"].churn_event.sum()),
        "expected_pattern": "raw_count_favors_A_but_rate_favors_B_as_worse",
    }
    truth["A_rate"] = truth["A_events"] / truth["A_customers"]
    truth["B_rate"] = truth["B_events"] / truth["B_customers"]
    truth["raw_count_says_A_worse"] = truth["A_events"] > truth["B_events"]
    truth["rate_says_B_worse"] = truth["B_rate"] > truth["A_rate"]
    return df, truth


def scenario_exposure_trap(rng):
    """Case 2: identical underlying hazard, different exposure duration.

    Both segments share the SAME per-day churn hazard. Segment C is
    observed for 365 days (full year); Segment D is observed for only 90
    days (newer cohort, less exposure). D will show a lower crude event
    COUNT and a lower crude RATE purely because it had less time to churn,
    even though the person-time-adjusted hazard is identical. A correct
    system must not conclude Segment D "retains better."
    """
    daily_hazard = 0.0006  # ~19.8%/yr equivalent
    rows = []
    cid = 1
    for _ in range(500):
        exposure = 365
        p_churn = 1 - (1 - daily_hazard) ** exposure
        churn = rng.random() < p_churn
        rows.append(_make_customer(f"CUST-{cid:05d}", "C", "2025-H1", exposure, exposure, churn, not churn))
        cid += 1
    for _ in range(500):
        exposure = 90
        p_churn = 1 - (1 - daily_hazard) ** exposure
        churn = rng.random() < p_churn
        rows.append(_make_customer(f"CUST-{cid:05d}", "D", "2025-H2", exposure, exposure, churn, not churn))
        cid += 1
    df = pd.DataFrame(rows)
    c_events = int(df[df.segment == "C"].churn_event.sum())
    d_events = int(df[df.segment == "D"].churn_event.sum())
    c_persontime = int(df[df.segment == "C"].observation_days.sum())
    d_persontime = int(df[df.segment == "D"].observation_days.sum())
    truth = {
        "daily_hazard": daily_hazard,
        "C_crude_rate": c_events / 500,
        "D_crude_rate": d_events / 500,
        "C_persontime_rate": c_events / c_persontime,
        "D_persontime_rate": d_events / d_persontime,
        "expected_pattern": "crude_rates_differ_but_persontime_rates_converge",
    }
    return df, truth


def scenario_cohort_trap(rng):
    """Case 3: apparent segment difference fully explained by cohort mix.

    Underlying churn behavior depends on COHORT, not segment:
      - "2025-H1" cohort (mature): 5% churn hazard
      - "2025-H2" cohort (new):    25% churn hazard
    Segment E is 90% H1 / 10% H2 (mostly mature) -> low observed churn.
    Segment F is 10% H1 / 90% H2 (mostly new)    -> high observed churn.
    Within each cohort, E and F have IDENTICAL churn rates (no real
    segment effect) -- the aggregate difference is confounded by cohort.
    """
    def make_segment(seg_name, n, frac_h1, cid_start):
        rows = []
        cid = cid_start
        for _ in range(n):
            is_h1 = rng.random() < frac_h1
            cohort = "2025-H1" if is_h1 else "2025-H2"
            hazard = 0.05 if is_h1 else 0.25
            churn = rng.random() < hazard
            rows.append(_make_customer(f"CUST-{cid:05d}", seg_name, cohort, 365, 365, churn, not churn))
            cid += 1
        return rows, cid

    rows_e, nxt = make_segment("E", 600, 0.90, 1)
    rows_f, _ = make_segment("F", 600, 0.10, nxt)
    df = pd.DataFrame(rows_e + rows_f)

    def rate(seg, cohort=None):
        sub = df[df.segment == seg]
        if cohort:
            sub = sub[sub.cohort == cohort]
        return sub.churn_event.mean() if len(sub) else None

    truth = {
        "E_aggregate_rate": rate("E"),
        "F_aggregate_rate": rate("F"),
        "E_H1_rate": rate("E", "2025-H1"),
        "F_H1_rate": rate("F", "2025-H1"),
        "E_H2_rate": rate("E", "2025-H2"),
        "F_H2_rate": rate("F", "2025-H2"),
        "expected_pattern": "aggregate_differs_but_within_cohort_rates_converge",
    }
    return df, truth


def scenario_tenure_trap(rng):
    """Case 4: segment membership correlates strongly with tenure; tenure
    (not segment) drives churn. Short-tenure customers churn more
    regardless of segment; Segment G happens to skew short-tenure,
    Segment H happens to skew long-tenure.
    """
    rows = []
    cid = 1
    for _ in range(600):
        tenure = int(rng.uniform(10, 90))     # short tenure, high risk window
        hazard = 0.30
        churn = rng.random() < hazard
        rows.append(_make_customer(f"CUST-{cid:05d}", "G", "2025-H2", tenure, tenure, churn, not churn))
        cid += 1
    for _ in range(600):
        tenure = int(rng.uniform(300, 365))   # long tenure, low risk window
        hazard = 0.05
        churn = rng.random() < hazard
        rows.append(_make_customer(f"CUST-{cid:05d}", "H", "2025-H1", tenure, tenure, churn, not churn))
        cid += 1
    df = pd.DataFrame(rows)
    truth = {
        "G_rate": df[df.segment == "G"].churn_event.mean(),
        "H_rate": df[df.segment == "H"].churn_event.mean(),
        "G_median_tenure": df[df.segment == "G"].tenure_days.median(),
        "H_median_tenure": df[df.segment == "H"].tenure_days.median(),
        "expected_pattern": "segment_and_tenure_are_collinear_confound_not_separable_without_stratification",
    }
    return df, truth


def scenario_genuine_effect(rng):
    """Case 5: a real segment effect that SURVIVES stratification by the
    known confounders (cohort, tenure-band). Segment I has a higher
    per-day hazard than Segment J at every matched tenure band and within
    every cohort, so the difference is not explained away.
    """
    rows = []
    cid = 1
    for _ in range(600):
        is_h1 = rng.random() < 0.5
        cohort = "2025-H1" if is_h1 else "2025-H2"
        tenure = int(rng.uniform(100, 300))
        hazard = 0.22  # consistently higher
        churn = rng.random() < hazard
        rows.append(_make_customer(f"CUST-{cid:05d}", "I", cohort, tenure, tenure, churn, not churn))
        cid += 1
    for _ in range(600):
        is_h1 = rng.random() < 0.5
        cohort = "2025-H1" if is_h1 else "2025-H2"
        tenure = int(rng.uniform(100, 300))
        hazard = 0.08  # consistently lower, matched cohort/tenure distribution
        churn = rng.random() < hazard
        rows.append(_make_customer(f"CUST-{cid:05d}", "J", cohort, tenure, tenure, churn, not churn))
        cid += 1
    df = pd.DataFrame(rows)
    truth = {
        "I_rate": df[df.segment == "I"].churn_event.mean(),
        "J_rate": df[df.segment == "J"].churn_event.mean(),
        "expected_pattern": "segment_effect_survives_cohort_and_tenure_stratification",
    }
    return df, truth


def scenario_insufficient_information(rng):
    """Case 6: strip out cohort/tenure/exposure fields entirely (only
    customer_id, segment, churn_event remain meaningful) so no
    confounder can be checked or ruled out. The correct behavior is
    INSUFFICIENT_EVIDENCE / an explicit "identifiability limited" verdict,
    never a confident segment claim.
    """
    rows = []
    cid = 1
    for seg, n, hazard in [("K", 40, 0.20), ("L", 40, 0.10)]:
        for _ in range(n):
            churn = rng.random() < hazard
            # tenure/observation/cohort deliberately left as NaN/unknown
            rows.append({
                "customer_id": f"CUST-{cid:05d}",
                "segment": seg,
                "cohort": None,
                "tenure_days": None,
                "observation_days": None,
                "signup_date": None,
                "observation_end_date": None,
                "churn_event": int(churn),
                "censored": None,
            })
            cid += 1
    df = pd.DataFrame(rows)
    truth = {
        "K_rate": df[df.segment == "K"].churn_event.mean(),
        "L_rate": df[df.segment == "L"].churn_event.mean(),
        "expected_pattern": "no_confounder_fields_available_must_return_insufficient_evidence",
    }
    return df, truth


SCENARIOS = {
    "raw_count_trap": scenario_raw_count_trap,
    "exposure_trap": scenario_exposure_trap,
    "cohort_trap": scenario_cohort_trap,
    "tenure_trap": scenario_tenure_trap,
    "genuine_effect": scenario_genuine_effect,
    "insufficient_information": scenario_insufficient_information,
}


def generate_primary_churn_csv():
    """The main `data/seed/churn.csv` wired into the benchmark project —
    uses the cohort_trap-shaped population (a real confound present) so
    that a naive implementation is exercised against a non-trivial case
    by default, while the other five scenarios stay available as isolated
    fixtures for the known-answer test matrix."""
    rng = random.Random(RNG_SEED)
    df, truth = scenario_cohort_trap(rng)
    out_path = os.path.join(SEED_DIR, "churn.csv")
    df.to_csv(out_path, index=False)
    print(f"-> Saved {len(df)} churn rows to {out_path}")
    print(f"   ground truth: {truth}")
    return df, truth


if __name__ == "__main__":
    generate_primary_churn_csv()
