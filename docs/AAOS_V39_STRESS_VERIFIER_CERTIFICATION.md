# AA-OS v39 — Stress-Verifier Certification Harness (measurement pass)

Status: **measurement only. NOT RELEASE READY.** No engine code and no ROOT_CAUSE behavior was changed.

## What changed
`scripts/stress_verifier.py` (new) replaces the inline `verify_ground_truth` in
`scripts/session23_real_data_stress.py`. Invariant: `truth_ok=True` iff the answer's reported
result **and** the persisted canonical contract agree with ground truth computed here from the
dataframe (pandas / scipy / statsmodels), never from engine output.

- No generic-success path; unknown kinds, blank answers, missing contracts, unparseable required results all fail.
- Rounding-aware numeric matching; group values must sit adjacent to their label.
- CORR: printed `r` (sign + value) and `n` vs pandas. Words like "correlated" prove nothing.
- RANK: winner named first, winner value, order of listed entries vs truth.
- DESC / ANOVA-style: every group reported and correct; eta-squared checked.
- Pairwise / root-cause-worded: Yes/No polarity vs Welch t-test, both means, and the difference.
- MULTI: both r values and which |r| is larger; interaction: LR test verdict and all sex x class cells.
- TREND: direction vs OLS slope significance, per-year slope, Kendall tau, period, aggregation label (average vs total).
- Contract: target, question variables present, no spurious grouping/predictor/time, canonical estimand, requested aggregation, requested/supported claim, claim ceiling never above ASSOCIATION.
- Unhedged causal wording flag (FALSE_CAUSAL_CLAIM).

Classes: CORRECT, UNSUPPORTED_REFUSED, CONTRACT_WRONG, AGGREGATION_WRONG, NUMERICALLY_WRONG,
INCOMPLETE_OUTPUT (added: right numbers but required items omitted), FALSE_CAUSAL_CLAIM.
`row["all_classes"]` keeps every failure; `result_class` is the primary one.

## Result on the 32-question corpus (both dataset variants identical)
Old verifier: 32/32. **Independent verifier: 23/32 CORRECT.**

| # | Question | Class(es) | Finding |
|---|---|---|---|
| 6 | tip ~ total_bill | AGGREGATION_WRONG | correlation contract requests `SUM` (r itself is correct) |
| 15 | mpg ~ weight | CONTRACT_WRONG | spurious grouping `origin` |
| 17 | fuel efficiency over model years | CONTRACT_WRONG | spurious grouping `origin` (numbers correct) |
| 19 | horsepower vs weight | CONTRACT_WRONG | spurious grouping `origin` (numbers correct) |
| 22 | avg price by clarity | INCOMPLETE_OUTPUT | 8 groups asked; IF and VVS1 truncated behind "..." |
| 23 | Why Fair > Ideal | CONTRACT_WRONG | routed `group_difference`, requested_claim ASSOCIATION; the other two "Why" questions are `root_cause`/CAUSAL |
| 26 | orbital period over years | CONTRACT_WRONG + **NUMERICALLY_WRONG** | spurious grouping `method`; printed slope +144.6/yr vs true OLS +139.3/yr |
| 28 | borough with highest fare | CONTRACT_WRONG | spurious time field `pickup` |
| 30 | passengers over time | CONTRACT_WRONG | spurious grouping `month` (numbers correct) |

Real numeric engine bug (Q26): the trend slope is regressed on the ordinal position of years
that have data (0..n-1), not on the year. Years 1990, 1991, 1993 are missing, so "per year" is
wrong (index-based slope reproduces 144.6 exactly). It affects any trend with gaps.

Observations not scored as failures: `target_json.aggregation_type` is `sum` on every contract
(requested_aggregation is the field that is honoured); Q25 (count of rows) has target=`method`
(accepted under an explicit row-count convention); the three "Why" answers are still group
comparisons — that is the ROOT_CAUSE gap, intentionally untouched here.

## Tests
`tests/test_stress_verifier.py` (fixture `tests/fixtures/stress_answers_v39.json`): generic
"substantive finding" text fails for all 32 specs; blank/failed/no-contract fails; a 13% numeric
drift, wrong winner, flipped verdict, wrong trend direction/aggregation label, wrong r, dropped
group, unhedged causal wording, and target/estimand/time/aggregation/claim/spurious-variable
contract mutations are each rejected. Needs seaborn datasets (skips offline).

## Next (not done)
1. Fix the engine defects above; re-run to reach 32/32 under this verifier.
2. Then change ROOT_CAUSE, measured against Q4/Q18/Q23.

## Amendments (post-review)
- Failure counting corrected: 9 failing rows = **7 contract** (Q15, 17, 19, 23, 26, 28, 30; Q26 also numeric) + 1 aggregation (Q6) + 1 incomplete output (Q22).
- `chk_pairwise` hardened: each group's mean must be bound to its own label (`Label averages N`, `Label (N`, `N for Label`). Swapped-mean answers now fail (test added). Baseline unchanged: **23/32**.
- Verified-in-environment note: the 8 verifier tests need the seaborn datasets; offline they skip (1 pass / rest skipped). Baseline must be re-established on a machine with dataset access.
- Frozen Release-1 grammar: "Why ...?" => ROOT_CAUSE, requested_claim CAUSAL, supported/ceiling <= ASSOCIATION.
- Agreed engine order: canonical-role cutover (explicit absence clears grouping/time/predictors) -> trend coordinate fix (regress on actual year) -> complete-grouped-output rule (no truncation unless top-N asked) -> correlation aggregation fix -> reach 32/32 -> ROOT_CAUSE.
