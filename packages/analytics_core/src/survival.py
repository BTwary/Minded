"""Deterministic survival-analysis primitives for AA-OS."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

@dataclass
class SurvivalResult:
    estimate: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    formulas: Dict[str, str] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    events: int = 0
    censored: int = 0


def _validate(df: pd.DataFrame, duration_col: str, event_col: str) -> pd.DataFrame:
    if duration_col not in df.columns or event_col not in df.columns:
        raise ValueError("duration and event columns are required")
    w = df[[duration_col, event_col]].copy()
    w[duration_col] = pd.to_numeric(w[duration_col], errors="coerce")
    w[event_col] = pd.to_numeric(w[event_col], errors="coerce")
    if w[[duration_col, event_col]].isna().any().any():
        raise ValueError("duration and event values must be non-missing")
    if (w[duration_col] < 0).any():
        raise ValueError("duration must be non-negative")
    if not w[event_col].isin([0, 1]).all():
        raise ValueError("event must be binary 0/1")
    return w


def kaplan_meier(df: pd.DataFrame, *, duration_col: str, event_col: str, group_col: str | None = None) -> SurvivalResult:
    w = _validate(df, duration_col, event_col)
    order = np.argsort(w[duration_col].to_numpy())
    t = w[duration_col].to_numpy()[order]
    e = w[event_col].to_numpy(dtype=int)[order]
    unique = np.unique(t)
    at_risk = len(t)
    surv = 1.0
    rows=[]
    var_log=-0.0
    for tm in unique:
        d = int(((t == tm) & (e == 1)).sum())
        c = int(((t == tm) & (e == 0)).sum())
        if d:
            surv *= (1.0 - d / at_risk)
            if at_risk - d > 0:
                var_log += d / (at_risk * (at_risk - d))
        rows.append({"time": float(tm), "at_risk": int(at_risk), "events": d, "censored": c, "survival": float(surv)})
        at_risk -= d + c
    median = None
    for r in rows:
        if r["survival"] <= 0.5:
            median = r["time"]
            break
    trace={"duration_column": duration_col, "event_column": event_col, "n": len(w), "estimator": "Kaplan-Meier"}
    return SurvivalResult(
        estimate={"curve": rows, "median_survival": median},
        diagnostics={"greenwood_log_variance_at_end": float(var_log)},
        formulas={"survival": "S(t) = product_{j:t_j<=t} (1 - d_j/n_j)", "greenwood": "Var(log S) = sum d_j/[n_j(n_j-d_j)]"},
        trace=trace, events=int(e.sum()), censored=int((1-e).sum())
    )


def logrank_test(df: pd.DataFrame, *, duration_col: str, event_col: str, group_col: str) -> SurvivalResult:
    if group_col not in df.columns:
        raise ValueError("group column is required")
    w = _validate(df, duration_col, event_col)
    groups = df[group_col].astype(str).reset_index(drop=True)
    if groups.nunique() != 2:
        raise ValueError("log-rank test currently requires exactly two groups")
    a,b=sorted(groups.unique())
    raw=df[[duration_col,event_col]].copy().reset_index(drop=True)
    raw["group"]=groups.values
    times=np.sort(raw.loc[raw[event_col]==1,duration_col].unique())
    O1=E1=V=0.0
    for tm in times:
        at=raw[raw[duration_col] >= tm]
        ev=raw[(raw[duration_col] == tm) & (raw[event_col] == 1)]
        n= len(at); d=len(ev); n1=int((at.group==a).sum()); d1=int((ev.group==a).sum())
        if n <= 1 or d == 0: continue
        e1=d*n1/n
        v1=(n1/n)*(1-n1/n)*d*(n-d)/(n-1)
        O1 += d1; E1 += e1; V += v1
    stat=float((O1-E1)**2/V) if V>0 else 0.0
    p=float(1-chi2.cdf(stat,1))
    return SurvivalResult(
        estimate={"groups":[a,b],"observed_group1":O1,"expected_group1":E1,"chi_square":stat,"p_value":p},
        diagnostics={"variance":V},
        formulas={"logrank":"sum_t (O_1t-E_1t)^2/V_1t"},
        trace={"duration_column":duration_col,"event_column":event_col,"group_column":group_col},
        events=int(raw[event_col].sum()), censored=int((1-raw[event_col]).sum())
    )


def cox_proportional_hazards(df: pd.DataFrame, *, duration_col: str, event_col: str, covariates: Sequence[str]) -> SurvivalResult:
    if any(c not in df.columns for c in covariates):
        raise ValueError("all covariates must be present")
    w=_validate(df,duration_col,event_col)
    X=df[list(covariates)].copy()
    for c in X.columns:
        X[c]=pd.to_numeric(X[c],errors="coerce")
    if X.isna().any().any(): raise ValueError("covariates must be numeric and non-missing for Cox")
    model=PHReg(w[duration_col].to_numpy(), X.to_numpy(), status=w[event_col].to_numpy())
    fit=model.fit()
    params=np.asarray(fit.params); se=np.asarray(fit.bse)
    rows=[]
    for i,c in enumerate(covariates):
        rows.append({"covariate":c,"log_hazard_ratio":float(params[i]),"hazard_ratio":float(np.exp(params[i])),"ci_lower":float(np.exp(params[i]-1.96*se[i])),"ci_upper":float(np.exp(params[i]+1.96*se[i]))})
    return SurvivalResult(
        estimate={"coefficients":rows},
        diagnostics={"ph_assumption_status":"REVIEW_REQUIRED","n":len(w),"events":int(w[event_col].sum())},
        formulas={"cox":"h(t|X)=h_0(t) exp(X beta)","hazard_ratio":"exp(beta)"},
        trace={"duration_column":duration_col,"event_column":event_col,"covariates":list(covariates)},
        events=int(w[event_col].sum()), censored=int((1-w[event_col]).sum())
    )
