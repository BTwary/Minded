"""Independent certification verifier for the 32-question real-data stress corpus.

Invariant:  truth_ok == True  <=>  the answer's reported analytical result AND the
persisted canonical contract both agree with ground truth computed here, directly
from the dataframe with pandas/scipy/statsmodels (never from engine output).

Design rules
  * No generic-success path.  Every check that cannot positively confirm a value
    fails.  A missing / unparseable required result is a failure, not a pass.
  * Numeric matching is rounding-aware: a token printed with d decimals matches an
    expected value iff |token - expected| <= 0.5 * 10**-d (plus float epsilon).
    Group values must be adjacent to their group label ("Label: value"), so an
    unrelated number elsewhere in the text cannot satisfy a check.
  * Contract checks read the persisted InvestigationContract (target, predictors,
    grouping, time, requested aggregation, estimand, claim fields).

Failure classes (primary class = first that applies, in this order)
  UNSUPPORTED_REFUSED, CONTRACT_WRONG, AGGREGATION_WRONG, NUMERICALLY_WRONG,
  INCOMPLETE_OUTPUT, FALSE_CAUSAL_CLAIM.  Success class: CORRECT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

CORRECT = "CORRECT"
UNSUPPORTED = "UNSUPPORTED_REFUSED"
CONTRACT = "CONTRACT_WRONG"
AGG = "AGGREGATION_WRONG"
NUMERIC = "NUMERICALLY_WRONG"
INCOMPLETE = "INCOMPLETE_OUTPUT"
CAUSAL_FLAG = "FALSE_CAUSAL_CLAIM"
_ORDER = [UNSUPPORTED, CONTRACT, AGG, NUMERIC, INCOMPLETE, CAUSAL_FLAG]

# --------------------------------------------------------------------------- numbers
_TOK = re.compile(r"(?<![\w.])([+-]?)(\d[\d,]*)(?:\.(\d+))?(\s*%)?")


def tokens(text: str) -> list[tuple[float, int, bool, str]]:
    """(value_with_sign, decimals, is_percent, sign_char) for every number in text."""
    out = []
    for m in _TOK.finditer(text):
        sign, whole, frac, pct = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            v = float(whole.replace(",", "") + ("." + frac if frac else ""))
        except ValueError:
            continue
        out.append((-v if sign == "-" else v, len(frac) if frac else 0, bool(pct), sign))
    return out


def _close(tok: float, dec: int, exp: float) -> bool:
    return abs(tok - exp) <= 0.5 * 10 ** (-dec) + 1e-9


def has_number(text: str, expected: float, *, pct_scale: bool = False, sign: bool = False) -> bool:
    """True iff some printed number in `text` equals `expected` at its own precision.

    pct_scale: expected is a fraction that may be printed as a percentage (x100).
    sign:      the sign of the printed number must agree with `expected`.
    """
    for v, dec, is_pct, s in tokens(text):
        cands = [expected]
        if pct_scale and is_pct:
            cands = [expected * 100.0]
        for c in cands:
            if sign:
                if _close(v, dec, c) and (v == 0 or c == 0 or (v < 0) == (c < 0)):
                    return True
            elif _close(abs(v), dec, abs(c)):
                return True
    return False


def value_after_label(text: str, label: str) -> list[tuple[float, int, bool]]:
    """All 'label: number[%]' occurrences (label matched on word boundary)."""
    pat = re.compile(
        r"(?<![\w])" + re.escape(label) + r"\s*:\s*([+-]?\d[\d,]*(?:\.\d+)?)(\s*%)?", re.I
    )
    res = []
    for m in pat.finditer(text):
        num = m.group(1).replace(",", "")
        dec = len(num.split(".")[1]) if "." in num else 0
        res.append((float(num), dec, bool(m.group(2))))
    return res


def label_value_ok(text: str, label: str, expected: float, *, rate: bool) -> bool:
    for v, dec, is_pct in value_after_label(text, label):
        exp = expected * 100.0 if (rate and is_pct) else expected
        if _close(v, dec, exp):
            return True
    return False


def first_position(text: str, label: str) -> int:
    m = re.search(r"(?<![\w])" + re.escape(label) + r"(?![\w])", text, re.I)
    return m.start() if m else -1


def key_label(k: Any) -> str:
    return " ".join(map(str, k)) if isinstance(k, tuple) else str(k)


# --------------------------------------------------------------------------- results
@dataclass
class Verdict:
    classes: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def fail(self, cls: str, msg: str) -> None:
        if cls not in self.classes:
            self.classes.append(cls)
        self.details.append(f"[{cls}] {msg}")

    @property
    def primary(self) -> str:
        for c in _ORDER:
            if c in self.classes:
                return c
        return CORRECT

    @property
    def ok(self) -> bool:
        return not self.classes


# --------------------------------------------------------------------------- contract
def summarize_contract(contract_row: Any, investigation_row: Any = None) -> dict:
    """Normalise a persisted InvestigationContract row into a plain dict."""
    if contract_row is None:
        return {}
    est = contract_row.estimand_json or {}
    fin = contract_row.final_contract_json or {}
    tgt = contract_row.target_json or {}
    bindings = fin.get("bindings") or []
    bound = {}
    for b in bindings:
        if len(b) == 3:
            bound.setdefault(b[0], []).append(b[2])
    grouping = list(est.get("grouping") or [])
    if fin.get("comparison_dimension"):
        grouping.append(fin["comparison_dimension"])
    predictors = list(fin.get("predictor_columns") or []) + list(
        contract_row.explanatory_variables_json or []
    )
    times = [t for t in (est.get("time"), fin.get("time_column"),
                         (contract_row.time_window_json or {}).get("time_column")) if t]
    return dict(
        target=fin.get("target_column") or est.get("target") or tgt.get("target"),
        grouping=sorted(set(grouping)),
        predictors=sorted(set(predictors)),
        time=sorted(set(times)),
        bound=bound,
        estimand=est.get("canonical_estimand"),
        task=est.get("task"),
        requested_aggregation=est.get("requested_aggregation"),
        requested_claim=est.get("requested_claim"),
        supported_claim=est.get("supported_claim"),
        claim_ceiling=est.get("claim_ceiling"),
        problem_class=contract_row.problem_class,
        claim_type=contract_row.claim_type,
        verdict_type=getattr(investigation_row, "verdict_type", None),
    )


@dataclass
class ContractSpec:
    target: str | None                   # None => row-count convention (see below)
    factors: tuple[str, ...] = ()        # grouping / predictor variables the question names
    time: str | None = None
    estimand: tuple[str, ...] = ()       # acceptable canonical_estimand values
    aggregation: tuple[str | None, ...] = ("MEAN",)   # acceptable requested_aggregation
    requested_claim: str | None = None   # None => not asserted
    count_target_group: bool = False     # row-count question: target may be None or the group column


_CAUSAL_CEILINGS = {"OBSERVATION", "ASSOCIATION"}


def check_contract(c: dict, spec: ContractSpec, v: Verdict) -> None:
    if not c:
        v.fail(CONTRACT, "no persisted contract found")
        return
    # target
    tgt = c.get("target")
    if spec.count_target_group:
        if tgt not in (None, *spec.factors):
            v.fail(CONTRACT, f"target={tgt!r}; row-count question expects none/group column")
    elif tgt != spec.target:
        v.fail(CONTRACT, f"target={tgt!r}, expected {spec.target!r}")
    # factor coverage: every named factor must appear as grouping/predictor/bound column
    seen = set(c.get("grouping", [])) | set(c.get("predictors", []))
    for role in ("EXPLANATORY_VARIABLE", "GROUPING_DIMENSION", "COMPARISON_DIMENSION"):
        seen |= set(c.get("bound", {}).get(role, []))
    if not spec.count_target_group:
        seen.discard(spec.target)
    missing = [f for f in spec.factors if f not in seen]
    if missing:
        v.fail(CONTRACT, f"question variable(s) {missing} absent from contract "
                         f"(grouping={c.get('grouping')}, predictors={c.get('predictors')})")
    # spurious variables the question never mentions
    allowed = set(spec.factors) | ({spec.target} if spec.target else set())
    extra = sorted((set(c.get("grouping", [])) | set(c.get("predictors", []))) - allowed)
    if extra:
        v.fail(CONTRACT, f"contract carries variable(s) not in the question: {extra}")
    # time
    times = set(c.get("time", []))
    exp_t = {spec.time} if spec.time else set()
    if times != exp_t:
        v.fail(CONTRACT, f"time field {sorted(times) or None}, expected {spec.time!r}")
    # estimand
    if spec.estimand and c.get("estimand") not in spec.estimand:
        v.fail(CONTRACT, f"canonical_estimand={c.get('estimand')!r}, expected one of {list(spec.estimand)}")
    # aggregation
    ra = c.get("requested_aggregation")
    if ra not in spec.aggregation:
        v.fail(AGG, f"requested_aggregation={ra!r}, expected one of {list(spec.aggregation)}")
    # claims
    if spec.requested_claim and c.get("requested_claim") != spec.requested_claim:
        v.fail(CONTRACT, f"requested_claim={c.get('requested_claim')!r}, expected {spec.requested_claim!r}")
    for k in ("supported_claim", "claim_ceiling"):
        if c.get(k) not in _CAUSAL_CEILINGS:
            v.fail(CONTRACT, f"{k}={c.get(k)!r}: observational data cannot support above ASSOCIATION")
    if c.get("verdict_type") == "CAUSALLY_SUPPORTED":
        v.fail(CONTRACT, "verdict_type CAUSALLY_SUPPORTED on observational data")


# --------------------------------------------------------------------------- causal language
_CAUSAL = re.compile(r"\b(caused|causes|causing|because of|due to|driven by|drives|leads? to|"
                     r"results? in|the reason|proves?)\b", re.I)
_SAFE = re.compile(r"\b(not|no|cannot|can't|doesn't|does not|association|associated|rather than)\b", re.I)


def causal_language(answer: str) -> list[str]:
    hits = []
    for s in re.split(r"(?<=[.!?;])\s+", answer):
        if _CAUSAL.search(s) and not _SAFE.search(s):
            hits.append(s.strip()[:160])
    return hits


# --------------------------------------------------------------------------- truth helpers
def agg_series(df: pd.DataFrame, by: str, col: str, agg: str) -> pd.Series:
    if agg == "count":
        return df.groupby(by, observed=True).size().astype(float)
    s = df.groupby(by, observed=True)[col].agg(agg)
    return s.dropna().astype(float)


def is_rate(df: pd.DataFrame, col: str) -> bool:
    u = pd.Series(df[col].dropna().unique())
    return set(u.astype(float)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- checkers
def _texts(ans: str, fnd: str) -> tuple[str, str]:
    return (ans or "") + " " + (fnd or ""), (ans or "")[:600]


def chk_groups(df, ans, fnd, v, *, by, col, agg, mode, need_all=False):
    """mode: 'desc' (all groups + values), 'rank' (winner, values, order prefix)."""
    full, _ = _texts(ans, fnd)
    truth = agg_series(df, by, col, agg)
    rate = agg == "mean" and is_rate(df, col)
    order = list(truth.sort_values(ascending=False).index)
    labels = {k: key_label(k) for k in order}
    # values adjacent to labels
    missing, wrong = [], []
    for k in order:
        vals = value_after_label(ans or "", labels[k])
        if not vals:
            missing.append(labels[k])
        elif not label_value_ok(ans or "", labels[k], float(truth[k]), rate=rate):
            wrong.append(f"{labels[k]}: expected {truth[k]:.4g}, printed {[x[0] for x in vals]}")
    if wrong:
        v.fail(NUMERIC, "group value mismatch: " + "; ".join(wrong))
    if mode == "desc" and missing:
        v.fail(INCOMPLETE, f"groups not reported: {missing}")
    if mode == "rank":
        # winner must be the first group named, and its value must be printed
        pos = {labels[k]: first_position(ans or "", labels[k]) for k in order}
        named = sorted((p, l) for l, p in pos.items() if p >= 0)
        top = labels[order[0]]
        if not named or named[0][1] != top:
            v.fail(NUMERIC, f"winner should be {top!r}; first group named is "
                            f"{named[0][1] if named else None!r}")
        if labels[order[0]] in missing:
            v.fail(NUMERIC, f"winner value for {top!r} not printed")
        # ranking order of the listed entries must follow truth
        listed = [(m.start(), m.group(1)) for k in order
                  for m in [re.search(r"(?<![\w])(" + re.escape(labels[k]) + r")\s*:\s*[+-]?\d", ans or "", re.I)]
                  if m]
        got = [l for _, l in sorted(listed)]
        want = [labels[k] for k in order if labels[k].lower() in {g.lower() for g in got}]
        if [g.lower() for g in got] != [w.lower() for w in want]:
            v.fail(NUMERIC, f"ranking order {got} != truth order {want}")


def chk_corr(df, ans, fnd, v, *, x, y):
    full, _ = _texts(ans, fnd)
    r = float(df[x].corr(df[y]))
    m = re.search(r"\br\s*=\s*([+-]?\d*\.\d+)", full)
    if not m:
        v.fail(NUMERIC, "no 'r=' statistic reported for correlation question")
        return
    val = m.group(1)
    dec = len(val.split(".")[1])
    if not _close(float(val), dec, r) or (abs(r) > 0.005 and (float(val) < 0) != (r < 0)):
        v.fail(NUMERIC, f"r printed {val}, pandas r={r:.4f}")
    n = int(df[[x, y]].dropna().shape[0])
    mn = re.search(r"\bn\s*=\s*([\d,]+)", full)
    if mn and int(mn.group(1).replace(",", "")) != n:
        v.fail(NUMERIC, f"n printed {mn.group(1)}, complete cases {n}")
    for c in (x, y):
        if first_position(full, c) < 0:
            v.fail(CONTRACT, f"variable {c!r} not named in the answer")


def chk_multi_corr(df, ans, fnd, v, *, y, xs):
    full, head = _texts(ans, fnd)
    rs = {x: float(df[x].corr(df[y])) for x in xs}
    for x in xs:
        m = re.search(re.escape(x) + r"\W+r\s*=\s*([+-]?\d*\.\d+)", full, re.I)
        if not m:
            v.fail(NUMERIC, f"no r reported for {x!r}")
            continue
        dec = len(m.group(1).split(".")[1])
        if not _close(float(m.group(1)), dec, rs[x]):
            v.fail(NUMERIC, f"{x}: printed r={m.group(1)}, pandas r={rs[x]:.4f}")
    winner = max(xs, key=lambda k: abs(rs[k]))
    pos = sorted((first_position(head, x), x) for x in xs if first_position(head, x) >= 0)
    if not pos or pos[0][1] != winner or "more strongly" not in head.lower():
        v.fail(NUMERIC, f"answer must lead with {winner!r} as more strongly related")



_NUM = r"([+-]?\d[\d,]*(?:\.\d+)?)(\s*%)?"


def pair_value_ok(text: str, label: str, expected: float, *, rate: bool) -> bool:
    """Group value must be bound to ITS label: 'Label averages N', 'Label (N', or 'N for Label'."""
    lab = r"(?<![\w])" + re.escape(label) + r"(?![\w])"
    pats = [lab + r"\s*(?:averages\s+|avg\s+|mean\s+|:\s*|\(\s*)" + _NUM,
            _NUM + r"\s*(?:[A-Za-z_]+\s+){0,2}?(?:for|in)\s+" + lab]
    for pat in pats:
        for m in re.finditer(pat, text, re.I):
            num = m.group(1).replace(",", "")
            dec = len(num.split(".")[1]) if "." in num else 0
            exp = expected * 100.0 if (rate and m.group(2)) else expected
            if _close(float(num), dec, exp):
                return True
    return False


def chk_pairwise(df, ans, fnd, v, *, by, col, a, b, direction="higher"):
    """Two-group comparison: means, difference, and Yes/No polarity from Welch t."""
    full, head = _texts(ans, fnd)
    ga = df.loc[df[by].astype(str) == a, col].dropna().astype(float)
    gb = df.loc[df[by].astype(str) == b, col].dropna().astype(float)
    ma, mb = ga.mean(), gb.mean()
    t = stats.ttest_ind(ga, gb, equal_var=False)
    sig = t.pvalue < 0.05
    expect_yes = bool(sig and ((ma > mb) if direction == "higher" else (ma < mb)))
    first = re.match(r"\s*(yes|no)\b", ans or "", re.I)
    if not first:
        v.fail(NUMERIC, "answer does not open with a Yes/No verdict")
    elif (first.group(1).lower() == "yes") != expect_yes:
        v.fail(NUMERIC, f"verdict {first.group(1)!r} but Welch p={t.pvalue:.3g}, "
                        f"mean({a})={ma:.4g}, mean({b})={mb:.4g} -> expected "
                        f"{'Yes' if expect_yes else 'No'}")
    rate = is_rate(df, col)
    for lab, mval in ((a, ma), (b, mb)):
        if first_position(head, lab) < 0:
            v.fail(NUMERIC, f"group {lab!r} not named in the answer opening")
        if not pair_value_ok(head, lab, mval, rate=rate):
            v.fail(NUMERIC, f"mean for {lab!r} ({mval:.4g}) not reported next to its own label")
    d = ma - mb
    if not (has_number(head, d, pct_scale=False) or has_number(head, d * 100 if rate else d)):
        v.fail(NUMERIC, f"difference {d:.4g} not reported")


def chk_anova_groups(df, ans, fnd, v, *, by, col):
    full, _ = _texts(ans, fnd)
    groups = [g.dropna().astype(float).values for _, g in df.groupby(by, observed=True)[col]]
    f, p = stats.f_oneway(*groups)
    grand = df[col].dropna().astype(float)
    ssb = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    eta2 = ssb / ((grand - grand.mean()) ** 2).sum()
    if p < 0.05 and "differs" not in full.lower():
        v.fail(NUMERIC, f"ANOVA p={p:.3g} (significant) but answer does not say the groups differ")
    if not has_number(full, eta2 * 100, sign=False):
        v.fail(NUMERIC, f"variance explained {eta2:.1%} not reported")
    chk_groups(df, ans, fnd, v, by=by, col=col, agg="mean", mode="desc")


def chk_stratified(df, ans, fnd, v, *, a, b, col):
    """Interaction: every a x b cell reported, and 'differs' agrees with an LR test."""
    import statsmodels.formula.api as smf
    from scipy.stats import chi2
    full, head = _texts(ans, fnd)
    d = df[[a, b, col]].dropna().copy()
    d[a] = d[a].astype(str)
    d[b] = d[b].astype(str)
    full_m = smf.logit(f"{col} ~ C({a}) * C({b})", d).fit(disp=0)
    add_m = smf.logit(f"{col} ~ C({a}) + C({b})", d).fit(disp=0)
    lr = 2 * (full_m.llf - add_m.llf)
    p = chi2.sf(lr, full_m.df_model - add_m.df_model)
    says_yes = bool(re.match(r"\s*yes\b", ans or "", re.I))
    if says_yes != (p < 0.05):
        v.fail(NUMERIC, f"LR interaction p={p:.3g}; answer verdict "
                        f"{'Yes' if says_yes else 'No'} disagrees")
    means = d.groupby([a, b])[col].mean()
    missing, wrong = [], []
    for k, val in means.items():
        lab = key_label(k)
        vals = value_after_label(ans or "", lab)
        if not vals:
            missing.append(lab)
        elif not label_value_ok(ans or "", lab, float(val), rate=True):
            wrong.append(f"{lab}: expected {val:.4f}, printed {[x[0] for x in vals]}")
    if wrong:
        v.fail(NUMERIC, "cell mismatch: " + "; ".join(wrong))
    if missing:
        v.fail(INCOMPLETE, f"stratified cells not reported: {missing}")


def chk_trend(df, ans, fnd, v, *, time, col, agg, label):
    full, head = _texts(ans, fnd)
    y = agg_series(df, time, col, agg).sort_index()
    x = np.array([float(i) for i in y.index])
    lr = stats.linregress(x, y.values)
    tau, tau_p = stats.kendalltau(x, y.values)
    sig = lr.pvalue < 0.05
    exp_dir = "up" if (sig and lr.slope > 0) else "down" if (sig and lr.slope < 0) else "none"
    low = full.lower()
    got_dir = ("up" if "trending up" in low else "down" if "trending down" in low
               else "none" if "no statistically detectable trend" in low else None)
    if got_dir is None:
        v.fail(NUMERIC, "answer states no trend direction")
    elif got_dir != exp_dir:
        v.fail(NUMERIC, f"trend direction {got_dir!r}, OLS slope={lr.slope:.4g} p={lr.pvalue:.3g} "
                        f"-> expected {exp_dir!r}")
    if f"({label} per year)" not in low:
        v.fail(AGG, f"answer aggregation label should be '({label} per year)'")
    m = re.search(r"slope\s*([+-]?\d[\d,]*\.?\d*)|([+-]\d[\d,]*\.?\d*)\s*per year", full)
    sval = (m.group(1) or m.group(2)) if m else None
    if sval is None:
        v.fail(NUMERIC, "no per-year slope reported")
    else:
        dec = len(sval.split(".")[1]) if "." in sval else 0
        if not _close(float(sval.replace(",", "")), dec, lr.slope):
            v.fail(NUMERIC, f"slope printed {sval}, OLS slope={lr.slope:.5g}")
    mt = re.search(r"tau\s*=\s*([+-]?\d*\.\d+)", full)
    if mt:
        dec = len(mt.group(1).split(".")[1])
        if not _close(float(mt.group(1)), dec, tau):
            v.fail(NUMERIC, f"Kendall tau printed {mt.group(1)}, scipy tau={tau:.4f}")
    mn = re.search(r"across (\d+) years \((\d{4}) to (\d{4})\)", full)
    if mn:
        def _yr(v):  # 2-digit source years (e.g. mpg model_year 70..82) may print as 19xx
            return {str(int(v)), str(int(v) + 1900)} if v < 100 else {str(int(v))}
        if not (int(mn.group(1)) == len(y) and mn.group(2) in _yr(x[0]) and mn.group(3) in _yr(x[-1])):
            v.fail(NUMERIC, f"period printed {mn.groups()}, data has {len(y)} years "
                            f"{int(x[0])}-{int(x[-1])}")
    else:
        v.fail(NUMERIC, "period/number of years not reported")


# --------------------------------------------------------------------------- spec table
@dataclass
class Q:
    question: str
    dataset: str
    kind: str
    contract: ContractSpec
    params: dict = field(default_factory=dict)
    causal_ask: bool = False   # question asks a causal/why question


_MEAN = ("MEAN",)
_RATE = ("RATE", "MEAN")
_NOAGG = (None, "CORRELATION", "MEAN")   # row-level statistic: a SUM/COUNT request is wrong


def _corr(question, ds, x, y, **kw):
    return Q(question, ds, "corr",
             ContractSpec(target=y, factors=(x,), estimand=("correlation",), aggregation=_NOAGG,
                          requested_claim=kw.pop("claim", "ASSOCIATION")),
             dict(x=x, y=y), causal_ask=kw.pop("causal", False))


SPECS: list[Q] = [
    Q("What is the survival rate by passenger class?", "titanic", "desc",
      ContractSpec("survived", ("passenger_class",), estimand=("rate", "mean"), aggregation=_RATE),
      dict(by="passenger_class", col="survived", agg="mean")),
    _corr("Is survival correlated with fare?", "titanic", "fare", "survived"),
    Q("Which sex had the highest survival rate?", "titanic", "rank",
      ContractSpec("survived", ("sex",), estimand=("ranking",), aggregation=_RATE),
      dict(by="sex", col="survived", agg="mean")),
    _corr("Did age affect survival?", "titanic", "age", "survived", claim="CAUSAL", causal=True),
    Q("Why did third class passengers have lower survival than first class?", "titanic", "pair",
      ContractSpec("survived", ("passenger_class",), estimand=("root_cause",), aggregation=_NOAGG + ("RATE",),
                   requested_claim="CAUSAL"),
      dict(by="passenger_class", col="survived", a="Third", b="First", direction="lower"), causal_ask=True),
    Q("Does the effect of class on survival differ between men and women?", "titanic", "strat",
      ContractSpec("survived", ("passenger_class", "sex"), estimand=("interaction",), aggregation=_RATE),
      dict(a="sex", b="passenger_class", col="survived")),
    _corr("Is tip correlated with total bill?", "tips", "total_bill", "tip"),
    Q("Which day has the highest average tip?", "tips", "rank",
      ContractSpec("tip", ("day",), estimand=("ranking",), aggregation=_MEAN),
      dict(by="day", col="tip", agg="mean")),
    Q("Do smokers tip more than non-smokers?", "tips", "pair",
      ContractSpec("tip", ("smoker",), estimand=("group_difference",), aggregation=_MEAN),
      dict(by="smoker", col="tip", a="Yes", b="No")),
    Q("What is the average tip by day?", "tips", "desc",
      ContractSpec("tip", ("day",), estimand=("mean",), aggregation=_MEAN),
      dict(by="day", col="tip", agg="mean")),
    _corr("Does party size drive the tip amount?", "tips", "size", "tip", claim="CAUSAL", causal=True),
    _corr("Is body mass correlated with flipper length?", "penguins", "flipper_length_mm", "body_mass_g"),
    Q("Which species has the highest average body mass?", "penguins", "rank",
      ContractSpec("body_mass_g", ("species",), estimand=("ranking",), aggregation=_MEAN),
      dict(by="species", col="body_mass_g", agg="mean")),
    Q("How does bill length differ between species?", "penguins", "anova",
      ContractSpec("bill_length_mm", ("species",), estimand=("group_difference",), aggregation=_MEAN),
      dict(by="species", col="bill_length_mm")),
    Q("Do male penguins weigh more than female penguins?", "penguins", "pair",
      ContractSpec("body_mass_g", ("sex",), estimand=("group_difference",), aggregation=_MEAN),
      dict(by="sex", col="body_mass_g", a="Male", b="Female")),
    _corr("Is mpg correlated with weight?", "mpg", "weight", "mpg"),
    Q("Which origin has the highest average mpg?", "mpg", "rank",
      ContractSpec("mpg", ("origin",), estimand=("ranking",), aggregation=_MEAN),
      dict(by="origin", col="mpg", agg="mean")),
    Q("Has fuel efficiency improved over model years?", "mpg", "trend",
      ContractSpec("mpg", (), time="model_year", estimand=("trend",), aggregation=_MEAN),
      dict(time="model_year", col="mpg", agg="mean", label="average")),
    Q("Why do cars from the usa have lower mpg?", "mpg", "anova_why",
      ContractSpec("mpg", ("origin",), estimand=("root_cause",), aggregation=_MEAN, requested_claim="CAUSAL"),
      dict(by="origin", col="mpg"), causal_ask=True),
    Q("Is horsepower or weight more strongly related to mpg?", "mpg", "multi",
      ContractSpec("mpg", ("horsepower", "weight"), estimand=("multi_correlation",),
                   aggregation=_NOAGG),
      dict(y="mpg", xs=["horsepower", "weight"])),
    _corr("Is price correlated with carat?", "diamonds", "carat", "price"),
    Q("Which cut has the highest average price?", "diamonds", "rank",
      ContractSpec("price", ("cut",), estimand=("ranking",), aggregation=_MEAN),
      dict(by="cut", col="price", agg="mean")),
    Q("What is the average price by clarity?", "diamonds", "desc",
      ContractSpec("price", ("clarity",), estimand=("mean",), aggregation=_MEAN),
      dict(by="clarity", col="price", agg="mean")),
    Q("Why are Fair cut diamonds priced higher than Ideal cut on average?", "diamonds", "pair",
      ContractSpec("price", ("cut",), estimand=("root_cause",), aggregation=_MEAN, requested_claim="CAUSAL"),
      dict(by="cut", col="price", a="Fair", b="Ideal"), causal_ask=True),
    Q("What is the average mass by discovery method?", "planets", "desc",
      ContractSpec("mass", ("method",), estimand=("mean",), aggregation=_MEAN),
      dict(by="method", col="mass", agg="mean")),
    Q("Which method discovered the most planets?", "planets", "rank",
      ContractSpec(None, ("method",), estimand=("ranking",), aggregation=("COUNT",), count_target_group=True),
      dict(by="method", col=None, agg="count")),
    Q("Has the orbital period of discovered planets changed over the years?", "planets", "trend",
      ContractSpec("orbital_period", (), time="year", estimand=("trend",), aggregation=_MEAN),
      dict(time="year", col="orbital_period", agg="mean", label="average")),
    _corr("Is tip correlated with fare?", "taxis", "fare", "tip"),
    Q("Which pickup borough has the highest average fare?", "taxis", "rank",
      ContractSpec("fare", ("pickup_borough",), estimand=("ranking",), aggregation=_MEAN),
      dict(by="pickup_borough", col="fare", agg="mean")),
    Q("Do credit card payers tip more than cash payers?", "taxis", "pair",
      ContractSpec("tip", ("payment",), estimand=("group_difference",), aggregation=_MEAN),
      dict(by="payment", col="tip", a="credit card", b="cash")),
    Q("Has the number of passengers grown over time?", "flights", "trend",
      ContractSpec("passengers", (), time="year", estimand=("trend",), aggregation=("SUM",)),
      dict(time="year", col="passengers", agg="sum", label="total")),
    Q("Which month has the highest average passengers?", "flights", "rank",
      ContractSpec("passengers", ("month",), estimand=("ranking",), aggregation=_MEAN),
      dict(by="month", col="passengers", agg="mean")),
]
SPEC_BY_QUESTION = {s.question: s for s in SPECS}


# --------------------------------------------------------------------------- entry point
_REFUSAL = re.compile(r"\b(cannot answer|can't answer|unable to|could not (?:resolve|determine)|"
                      r"unresolved|need(?:s)? clarification)\b", re.I)


def verify(question: str, df: pd.DataFrame, *, status: str, verdict_type: str | None,
           answer: str, finding: str, contract: dict) -> Verdict:
    v = Verdict()
    spec = SPEC_BY_QUESTION.get(question)
    if spec is None:
        v.fail(UNSUPPORTED, "question has no ground-truth spec (cannot certify)")
        return v
    if status != "COMPLETED":
        v.fail(UNSUPPORTED, f"status={status}")
        return v
    if not (answer or "").strip():
        v.fail(UNSUPPORTED, "blank answer")
        return v
    if _REFUSAL.search((answer or "")[:250]):
        v.fail(UNSUPPORTED, "answer reads as a refusal / unresolved")
        return v
    check_contract(contract, spec.contract, v)
    p, k = spec.params, spec.kind
    if k == "desc":
        chk_groups(df, answer, finding, v, mode="desc", **p)
    elif k == "rank":
        chk_groups(df, answer, finding, v, mode="rank", **p)
    elif k == "corr":
        chk_corr(df, answer, finding, v, **p)
    elif k == "multi":
        chk_multi_corr(df, answer, finding, v, **p)
    elif k == "pair":
        chk_pairwise(df, answer, finding, v, **p)
    elif k in ("anova", "anova_why"):
        chk_anova_groups(df, answer, finding, v, **p)
    elif k == "strat":
        chk_stratified(df, answer, finding, v, **p)
    elif k == "trend":
        chk_trend(df, answer, finding, v, **p)
    else:  # unknown kind must never pass
        v.fail(UNSUPPORTED, f"no checker for kind {k!r}")
    hits = causal_language(answer)
    if hits:
        v.fail(CAUSAL_FLAG, f"unhedged causal wording: {hits}")
    return v
