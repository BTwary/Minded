"""Post-hoc pairwise inference and multiplicity control.

All procedures are deterministic and return an audit-ready family definition,
adjusted p-values, effect sizes and method assumptions.  The module deliberately
separates the omnibus question ("is any group different?") from the post-hoc
question ("which groups differ?").
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy import stats

from .analytical_math import multiple_testing_correction


@dataclass(frozen=True)
class PairwiseComparison:
    group_a: str
    group_b: str
    statistic: float
    raw_p_value: float
    adjusted_p_value: float
    effect_size: float
    effect_size_name: str
    estimate: float | None
    confidence_interval: tuple[float, float] | None
    method: str
    correction: str
    reject: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "statistic": float(self.statistic),
            "raw_p_value": float(self.raw_p_value),
            "adjusted_p_value": float(self.adjusted_p_value),
            "effect_size": float(self.effect_size),
            "effect_size_name": self.effect_size_name,
            "estimate": None if self.estimate is None else float(self.estimate),
            "confidence_interval": None if self.confidence_interval is None else [float(self.confidence_interval[0]), float(self.confidence_interval[1])],
            "method": self.method,
            "correction": self.correction,
            "reject": bool(self.reject),
        }


from .preflight import _to_clean_float_array


def _clean(values: Sequence[float]) -> np.ndarray:
    return _to_clean_float_array(values)


def _welch_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    diff = float(np.mean(a) - np.mean(b))
    v1 = float(np.var(a, ddof=1) / len(a))
    v2 = float(np.var(b, ddof=1) / len(b))
    se = float(np.sqrt(v1 + v2))
    if se <= 0:
        return (diff, diff)
    df = float((v1 + v2) ** 2 / ((v1 ** 2) / (len(a) - 1) + (v2 ** 2) / (len(b) - 1)))
    margin = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return (diff - margin, diff + margin)


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    return float(2.0 * u / (n1 * n2) - 1.0)


def _dunn_pair(a: np.ndarray, b: np.ndarray, a_ranks: np.ndarray, b_ranks: np.ndarray, all_values: np.ndarray) -> tuple[float, float, float]:
    """Dunn pairwise z, two-sided p, rank-biserial effect.

    `a_ranks`/`b_ranks` must be the family-wide (all groups) rank values
    that correspond to `a`'s and `b`'s own original observations -- NOT
    ranks recomputed over just the concatenation of `a` and `b`. Dunn's
    test is only valid when each group's mean rank reflects its position
    against the *entire* family (all k groups from the Kruskal-Wallis
    omnibus), not just the current pair; using a pair-local ranking (or,
    worse, an arbitrary positional slice of the family ranks) silently
    produces the wrong mean rank whenever the family has more than two
    groups. The tie correction and rank variance are likewise computed
    across the complete family of observations via `all_values`.
    """
    n1, n2 = len(a), len(b)
    n = len(all_values)
    mean1 = float(np.mean(a_ranks))
    mean2 = float(np.mean(b_ranks))
    _, counts = np.unique(all_values, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    base_var = n * (n + 1) / 12.0
    correction = 1.0 - tie_term / (n * (n - 1) * (n + 1)) if n > 1 else 1.0
    var_r = base_var * correction
    se = float(np.sqrt(var_r * (1.0 / n1 + 1.0 / n2)))
    z = float((mean1 - mean2) / se) if se > 0 else 0.0
    p = float(2.0 * stats.norm.sf(abs(z)))
    u = float(np.sum(a_ranks) - n1 * (n1 + 1) / 2.0)
    return z, p, _rank_biserial(u, n1, n2)


def _adjust(values: Sequence[float], correction: str, alpha: float) -> tuple[np.ndarray, str]:
    key = correction.strip().lower().replace("-", "_")
    if key not in {"holm", "bonferroni", "fdr_bh"}:
        raise ValueError("Unsupported correction. Use holm, bonferroni, or fdr_bh.")
    from statsmodels.stats.multitest import multipletests
    method = {"holm": "holm", "bonferroni": "bonferroni", "fdr_bh": "fdr_bh"}[key]
    _, p_adj, _, _ = multipletests(np.asarray(values, dtype=float), alpha=alpha, method=method)
    return np.asarray(p_adj, dtype=float), key


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def _run_tukey(clean: Mapping[str, np.ndarray], alpha: float) -> list[Dict[str, Any]]:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    names = list(clean.keys())
    values = np.concatenate([clean[k] for k in names])
    labels = np.concatenate([np.full(len(clean[k]), k, dtype=object) for k in names])
    tuk = pairwise_tukeyhsd(values, labels, alpha=alpha)
    out: list[Dict[str, Any]] = []
    g = tuk._multicomp.groupsunique
    for row, (i, j) in enumerate(zip(tuk._multicomp.pairindices[0], tuk._multicomp.pairindices[1])):
        ga, gb = str(g[i]), str(g[j])
        a, b = clean[ga], clean[gb]
        estimate = float(tuk.meandiffs[row])
        out.append({
            "group_a": ga, "group_b": gb,
            "statistic": float(tuk.meandiffs[row] / tuk.std_pairs[row]) if float(tuk.std_pairs[row]) > 0 else 0.0,
            "raw_p": float(tuk.pvalues[row]),
            "estimate": estimate,
            "ci": (float(tuk.confint[row][0]), float(tuk.confint[row][1])),
            "effect": _cohens_d(a, b),
            "effect_name": "Cohen's d",
            "method": "Tukey HSD",
            "reject": bool(tuk.reject[row]),
        })
    return out


def pairwise_posthoc(
    groups: Mapping[str, Sequence[float]],
    *,
    omnibus_method: str,
    alpha: float = 0.05,
    correction: str = "holm",
    require_significant_omnibus: bool = True,
    omnibus_p_value: float | None = None,
) -> Dict[str, Any]:
    """Run an auditable multiplicity-controlled post-hoc family.

    One-way ANOVA defaults to Holm-adjusted pairwise pooled t-tests. Tukey HSD
    is available explicitly via ``correction="tukey-hsd"``. Welch ANOVA uses
    pairwise Welch tests; Kruskal-Wallis uses Dunn tests. Holm is the default
    family-wise correction for all non-Tukey paths.
    """
    clean = {str(k): _clean(v) for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if len(v) >= 2}
    if len(clean) < 2:
        return {"status": "not_run", "reason": "At least two groups with >=2 observations are required.", "comparisons": []}

    omnibus_p_value_underflowed = False
    if omnibus_p_value is None:
        if omnibus_method == "One-way ANOVA":
            omnibus_p_value = float(stats.f_oneway(*clean.values()).pvalue)
            if omnibus_p_value == 0.0:
                omnibus_p_value = float(np.nextafter(0.0, 1.0))
                omnibus_p_value_underflowed = True
        elif omnibus_method == "Welch ANOVA":
            n = np.array([len(v) for v in clean.values()], dtype=float)
            means = np.array([np.mean(v) for v in clean.values()], dtype=float)
            vars_ = np.array([np.var(v, ddof=1) for v in clean.values()], dtype=float)
            if np.any(vars_ <= 0):
                return {
                    "status": "not_run",
                    "reason": "Welch ANOVA requires positive within-group variance for every group; the omnibus p-value is undefined.",
                    "omnibus_p_value": None,
                    "alpha": float(alpha),
                    "comparisons": [],
                }
            else:
                w = n / vars_
                ws = float(np.sum(w))
                mw = float(np.sum(w * means) / ws)
                k = len(clean)
                num = float(np.sum(w * (means - mw) ** 2) / (k - 1))
                corr = float((2 * (k - 2) / (k**2 - 1)) * np.sum(((1 - w / ws) ** 2) / (n - 1)))
                f = num / (1 + corr)
                df1 = k - 1
                df2 = float((k**2 - 1) / (3 * np.sum(((1 - w / ws) ** 2) / (n - 1))))
                omnibus_p_value = float(stats.f.sf(f, df1, df2))
                if omnibus_p_value == 0.0:
                    omnibus_p_value = float(np.nextafter(0.0, 1.0))
                    omnibus_p_value_underflowed = True
        elif omnibus_method == "Kruskal-Wallis":
            omnibus_p_value = float(stats.kruskal(*clean.values()).pvalue)
            if omnibus_p_value == 0.0:
                omnibus_p_value = float(np.nextafter(0.0, 1.0))
                omnibus_p_value_underflowed = True

    if require_significant_omnibus and (omnibus_p_value is None or float(omnibus_p_value) >= alpha):
        return {
            "status": "not_run",
            "reason": "Omnibus gate not significant; pairwise family was not searched.",
            "omnibus_p_value": None if omnibus_p_value is None else float(omnibus_p_value),
            "omnibus_p_value_underflowed": bool(omnibus_p_value_underflowed),
            "alpha": float(alpha),
            "comparisons": [],
        }

    if omnibus_method == "One-way ANOVA" and correction.strip().lower().replace("-", "_") == "tukey_hsd":
        try:
            # One global Tukey fit: p-values are familywise-adjusted across ALL pairs.
            tukey_rows = _run_tukey(clean, alpha)
            comparisons = [PairwiseComparison(
                group_a=x["group_a"], group_b=x["group_b"], statistic=x["statistic"],
                raw_p_value=x["raw_p"], adjusted_p_value=x["raw_p"], effect_size=x["effect"],
                effect_size_name=x["effect_name"], estimate=x["estimate"], confidence_interval=x["ci"],
                method=x["method"], correction="tukey-hsd", reject=x["reject"]
            ).to_dict() for x in tukey_rows]
            return {
                "status": "completed", "omnibus_method": omnibus_method,
                "omnibus_p_value": float(omnibus_p_value), "omnibus_p_value_underflowed": bool(omnibus_p_value_underflowed), "alpha": float(alpha),
                "family_size": len(comparisons), "correction": "tukey-hsd",
                "familywise_search_policy": "omnibus-gated" if require_significant_omnibus else "ungated",
                "comparisons": comparisons,
                "note": "Tukey HSD p-values are familywise adjusted; no second correction was applied."
            }
        except Exception:
            # Fail closed to explicit pairwise pooled t-tests + requested multiplicity correction.
            pairs = list(combinations(clean.keys(), 2))
            intermediate = []
            raw = []
            for ga, gb in pairs:
                a, b = clean[ga], clean[gb]
                res = stats.ttest_ind(a, b, equal_var=True)
                p = float(res.pvalue); raw.append(p)
                diff = float(np.mean(a) - np.mean(b))
                intermediate.append({"group_a":ga,"group_b":gb,"statistic":float(res.statistic),"raw_p":p,"estimate":diff,"ci":_welch_ci(a,b,alpha),"effect":_cohens_d(a,b),"effect_name":"Cohen's d","method":"pooled t pairwise (Tukey unavailable)"})
            adjusted, corr_key = _adjust(raw, correction, alpha)
            comparisons=[PairwiseComparison(group_a=x["group_a"],group_b=x["group_b"],statistic=x["statistic"],raw_p_value=x["raw_p"],adjusted_p_value=float(a),effect_size=x["effect"],effect_size_name=x["effect_name"],estimate=x["estimate"],confidence_interval=x["ci"],method=x["method"],correction=corr_key,reject=bool(a<alpha)).to_dict() for x,a in zip(intermediate, adjusted)]
            return {"status":"completed","omnibus_method":omnibus_method,"omnibus_p_value":float(omnibus_p_value),"omnibus_p_value_underflowed":bool(omnibus_p_value_underflowed),"alpha":float(alpha),"family_size":len(comparisons),"correction":corr_key,"familywise_search_policy":"omnibus-gated" if require_significant_omnibus else "ungated","comparisons":comparisons,"note":"Tukey HSD unavailable; explicit multiplicity correction applied to pairwise pooled t-tests."}

    pairs = list(combinations(clean.keys(), 2))
    raw: list[float] = []
    intermediate: list[Dict[str, Any]] = []
    family_rank_by_group: Dict[str, np.ndarray] = {}
    all_values: np.ndarray | None = None
    if omnibus_method == "Kruskal-Wallis":
        # Compute the family-wide rank of every observation exactly once,
        # then slice each group's own rank values back out by its original
        # position. Every pairwise Dunn comparison below reuses these same
        # per-group rank vectors, so a group's mean rank always reflects its
        # position against the whole family (all k groups), not just
        # whichever other group it happens to be paired against.
        group_order = list(clean.keys())
        all_values = np.concatenate([clean[g] for g in group_order])
        family_ranks = stats.rankdata(all_values, method="average")
        _pos = 0
        for g in group_order:
            _ln = len(clean[g])
            family_rank_by_group[g] = family_ranks[_pos:_pos + _ln]
            _pos += _ln
    for ga, gb in pairs:
        a, b = clean[ga], clean[gb]
        if omnibus_method == "One-way ANOVA":
            res = stats.ttest_ind(a, b, equal_var=True)
            p = float(res.pvalue)
            intermediate.append({"group_a":ga,"group_b":gb,"statistic":float(res.statistic),"raw_p":p,"estimate":float(np.mean(a)-np.mean(b)),"ci":_welch_ci(a,b,alpha),"effect":_cohens_d(a,b),"effect_name":"Cohen's d","method":"pooled pairwise t-test"})
        elif omnibus_method == "Welch ANOVA":
            res = stats.ttest_ind(a, b, equal_var=False)
            p = float(res.pvalue)
            intermediate.append({"group_a":ga,"group_b":gb,"statistic":float(res.statistic),"raw_p":p,"estimate":float(np.mean(a)-np.mean(b)),"ci":_welch_ci(a,b,alpha),"effect":_cohens_d(a,b),"effect_name":"standardized mean difference","method":"Welch pairwise t-test"})
        elif omnibus_method == "Kruskal-Wallis":
            stat, p, effect = _dunn_pair(a, b, family_rank_by_group[ga], family_rank_by_group[gb], all_values)
            intermediate.append({"group_a":ga,"group_b":gb,"statistic":stat,"raw_p":p,"estimate":float(np.median(a)-np.median(b)),"ci":None,"effect":effect,"effect_name":"rank-biserial correlation","method":"Dunn pairwise rank test"})
        else:
            raise ValueError(f"Unsupported omnibus method: {omnibus_method}")
        raw.append(p)
    adjusted, corr_key = _adjust(raw, correction, alpha)
    comparisons=[]
    for x,a_p in zip(intermediate, adjusted):
        comparisons.append(PairwiseComparison(group_a=x["group_a"],group_b=x["group_b"],statistic=x["statistic"],raw_p_value=x["raw_p"],adjusted_p_value=float(a_p),effect_size=x["effect"],effect_size_name=x["effect_name"],estimate=x["estimate"],confidence_interval=x["ci"],method=x["method"],correction=corr_key,reject=bool(a_p<alpha)).to_dict())
    return {"status":"completed","omnibus_method":omnibus_method,"omnibus_p_value":None if omnibus_p_value is None else float(omnibus_p_value),"omnibus_p_value_underflowed":bool(omnibus_p_value_underflowed),"alpha":float(alpha),"family_size":len(comparisons),"correction":corr_key,"familywise_search_policy":"omnibus-gated" if require_significant_omnibus else "ungated","comparisons":comparisons}
