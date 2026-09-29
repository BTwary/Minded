"""Deterministic exploratory reconnaissance for the AA-OS analytical engine.

This module is intentionally descriptive/screening-only. It computes reproducible
statistics that help the autonomous planner discover structure worth investigating,
but it never upgrades an exploratory signal into a causal or inferential verdict.
"""
from __future__ import annotations

from itertools import combinations
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class ExploratoryAnalysisEngine:
    """Deterministic, dependency-light analytical reconnaissance engine."""

    VERSION = "EAE-1.1"
    DEFAULT_MAX_NUMERIC_PAIRS = 24
    DEFAULT_MAX_CATEGORICAL_PAIRS = 12
    DEFAULT_MAX_GROUP_SCREENS = 24
    DEFAULT_MAX_TREND_SCREENS = 12

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        *,
        max_numeric_pairs: int = DEFAULT_MAX_NUMERIC_PAIRS,
        max_categorical_pairs: int = DEFAULT_MAX_CATEGORICAL_PAIRS,
        max_group_screens: int = DEFAULT_MAX_GROUP_SCREENS,
        max_trend_screens: int = DEFAULT_MAX_TREND_SCREENS,
    ) -> Dict[str, Any]:
        if df is None:
            raise ValueError("Exploratory analysis requires a dataframe.")

        frame = df.copy()
        numeric = sorted([str(c) for c in frame.select_dtypes(include=[np.number]).columns])
        categorical = sorted([
            str(c)
            for c in frame.columns
            if str(c) not in numeric
            and (
                pd.api.types.is_object_dtype(frame[c])
                or pd.api.types.is_string_dtype(frame[c])
                or isinstance(frame[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(frame[c])
            )
        ])
        datetimes = cls._datetime_columns(frame)

        column_summary: Dict[str, Any] = {}
        for col in frame.columns:
            s = frame[col]
            valid = int(s.notna().sum())
            missing = int(s.isna().sum())
            entry: Dict[str, Any] = {
                "dtype": str(s.dtype),
                "non_null": valid,
                "missing": missing,
                "missing_rate": round(float(missing / len(frame)), 6) if len(frame) else 0.0,
                "unique": int(s.nunique(dropna=True)),
            }
            if str(col) in numeric:
                vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals):
                    q1, median, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
                    iqr = float(q3 - q1)
                    mad = float(np.median(np.abs(vals - median)))
                    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
                    iqr_outliers = int(np.sum((vals < lower) | (vals > upper)))
                    robust_z = np.abs(0.67448975 * (vals - median) / mad) if mad > 0 else np.zeros_like(vals)
                    mad_outliers = int(np.sum(robust_z > 3.5)) if mad > 0 else 0
                    entry.update(
                        {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "min": float(np.min(vals)),
                            "q1": float(q1),
                            "median": float(median),
                            "q3": float(q3),
                            "max": float(np.max(vals)),
                            "iqr": iqr,
                            "mad": mad,
                            "iqr_outlier_count": iqr_outliers,
                            "mad_outlier_count": mad_outliers,
                        }
                    )
            else:
                top = s.astype("string").value_counts(dropna=True).head(10)
                entry["top_categories"] = {str(k): int(v) for k, v in top.items()}
            column_summary[str(col)] = entry

        correlations = cls._numeric_relationships(frame, numeric, max_numeric_pairs)
        categorical_associations = cls._categorical_relationships(frame, categorical, max_categorical_pairs)
        group_effects = cls._group_effect_screens(frame, numeric, categorical, max_group_screens)
        trend_screens = cls._trend_screens(frame, numeric, datetimes, max_trend_screens)
        missingness = cls._missingness_patterns(frame)

        all_screened = [*correlations, *categorical_associations, *group_effects, *trend_screens]
        cls._attach_family_qvalues(all_screened)

        strongest = sorted(
            all_screened,
            key=lambda x: (-float(x.get("screening_strength", 0.0)), str(x.get("key", ""))),
        )[:20]

        return {
            "engine_version": cls.VERSION,
            "scope": {"rows": int(len(frame)), "columns": int(len(frame.columns)), "numeric_columns": numeric, "categorical_columns": categorical, "datetime_columns": datetimes},
            "columns": column_summary,
            "correlations": correlations,
            "categorical_associations": categorical_associations,
            "group_effects": group_effects,
            "temporal_trends": trend_screens,
            "missingness": missingness,
            "top_screening_signals": strongest,
            "interpretation": "Exploratory screening only; signals require method-specific validation before any inferential or causal claim.",
        }

    @staticmethod
    def _datetime_columns(df: pd.DataFrame) -> List[str]:
        out: List[str] = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                out.append(str(col))
                continue
            sample = df[col].dropna().head(50)
            if len(sample) and ("date" in str(col).lower() or "time" in str(col).lower() or "timestamp" in str(col).lower()):
                parsed = pd.to_datetime(sample, errors="coerce", utc=True)
                if parsed.notna().mean() >= 0.9:
                    out.append(str(col))
        return sorted(out)[:12]

    @classmethod
    def _numeric_relationships(cls, df: pd.DataFrame, numeric: Sequence[str], max_pairs: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        candidates = list(combinations(numeric, 2))[: max(0, int(max_pairs))]
        raw_p: List[Tuple[int, float]] = []
        temp: List[Dict[str, Any]] = []
        for x, y in candidates:
            x, y = sorted((str(x), str(y)))
            pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 5 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
                continue
            pearson = stats.pearsonr(pair[x], pair[y])
            spearman = stats.spearmanr(pair[x], pair[y])
            p_values = [
                float(p)
                for p in (pearson.pvalue, spearman.pvalue)
                if p is not None and np.isfinite(float(p))
            ]
            if not p_values:
                continue
            # Two related tests (Pearson and Spearman) are being used as
            # alternative screens for the same relationship.  Taking min(p)
            # without adjustment inflates false positives.  Use Simes'
            # combination for a valid omnibus pair-level screen, then apply
            # one BH correction across the complete exploratory family.
            p_sorted = sorted(p_values)
            p = float(min(min(1.0, len(p_sorted) * p_sorted[0]), max(p_sorted)))
            temp.append({
                "key": f"CORR::{x}::{y}",
                "x": x, "y": y,
                "n": int(len(pair)),
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
                "p_value": float(p),
                "p_value_method": "simes_two_test",
                "screening_strength": float(max(abs(pearson.statistic), abs(spearman.statistic))),
            })
            raw_p.append((len(temp) - 1, p))
        cls._attach_bh_qvalues(temp, raw_p)
        return sorted(temp, key=lambda r: (-float(r["screening_strength"]), r["key"]))

    @classmethod
    def _categorical_relationships(cls, df: pd.DataFrame, categorical: Sequence[str], max_pairs: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for a, b in list(combinations(categorical, 2))[: max(0, int(max_pairs))]:
            ca, cb = df[a], df[b]
            if ca.nunique(dropna=True) < 2 or cb.nunique(dropna=True) < 2:
                continue
            table = pd.crosstab(ca, cb)
            if table.size == 0:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
            n = float(table.to_numpy().sum())
            k = min(table.shape[0] - 1, table.shape[1] - 1)
            cramers_v = float(np.sqrt(max(0.0, chi2 / (n * k)))) if n > 0 and k > 0 else 0.0
            out.append({
                "key": f"CAT_ASSOC::{a}::{b}",
                "x": a, "y": b,
                "n": int(n), "chi2": float(chi2), "degrees_of_freedom": int(dof), "p_value": float(p),
                "cramers_v": cramers_v, "screening_strength": cramers_v,
            })
        return sorted(out, key=lambda r: (-float(r["screening_strength"]), r["key"]))

    @classmethod
    def _group_effect_screens(cls, df: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str], max_pairs: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for group in categorical:
            levels = df[group].dropna().nunique()
            if levels < 2 or levels > 30:
                continue
            for metric in numeric:
                frame = df[[group, metric]].copy()
                frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
                frame = frame.dropna()
                grouped = [g[metric].to_numpy(dtype=float) for _, g in frame.groupby(group, sort=True) if len(g) >= 2]
                if len(grouped) < 2:
                    continue
                try:
                    anova = stats.f_oneway(*grouped)
                    allv = np.concatenate(grouped)
                    grand = float(np.mean(allv))
                    ss_between = float(sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in grouped))
                    ss_total = float(np.sum((allv - grand) ** 2))
                    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
                except Exception:
                    continue
                out.append({
                    "key": f"GROUP_EFFECT::{group}::{metric}",
                    "group": group, "metric": metric, "groups": int(len(grouped)), "n": int(len(allv)),
                    "f_statistic": float(anova.statistic), "p_value": float(anova.pvalue),
                    "eta_squared_screen": float(max(0.0, min(1.0, eta2))),
                    "screening_strength": float(max(0.0, min(1.0, eta2))),
                })
                if len(out) >= max_pairs:
                    return sorted(out, key=lambda r: (-float(r["screening_strength"]), r["key"]))
        return sorted(out, key=lambda r: (-float(r["screening_strength"]), r["key"]))

    @classmethod
    def _trend_screens(cls, df: pd.DataFrame, numeric: Sequence[str], datetimes: Sequence[str], max_screens: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for time_col in datetimes:
            dt = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            time_num = (dt.astype("int64") / 86_400_000_000_000).astype(float)
            for metric in numeric:
                vals = pd.to_numeric(df[metric], errors="coerce")
                mask = time_num.notna() & vals.notna()
                if int(mask.sum()) < 5 or vals[mask].nunique() < 2:
                    continue
                x = time_num[mask].to_numpy(dtype=float)
                y = vals[mask].to_numpy(dtype=float)
                slope, intercept, r, p, stderr = stats.linregress(x, y)
                rho, sp = stats.spearmanr(x, y)
                out.append({
                    "key": f"TREND::{time_col}::{metric}",
                    "time_column": time_col, "metric": metric, "n": int(len(y)),
                    "linear_slope_per_day": float(slope), "linear_r": float(r), "linear_p": float(p),
                    "spearman_rho": float(rho), "spearman_p": float(sp), "p_value": float(min(p, sp)), "slope_stderr": float(stderr),
                    "screening_strength": float(abs(r)),
                    "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
                })
                if len(out) >= max_screens:
                    return sorted(out, key=lambda r: (-float(r["screening_strength"]), r["key"]))
        return sorted(out, key=lambda r: (-float(r["screening_strength"]), r["key"]))

    @staticmethod
    def _missingness_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        cols = sorted(map(str, df.columns))
        rates = {c: round(float(df[c].isna().mean()), 6) if len(df) else 0.0 for c in cols}
        pairs: List[Dict[str, Any]] = []
        for a, b in list(combinations(cols, 2))[:60]:
            ma = df[a].isna().astype(int)
            mb = df[b].isna().astype(int)
            if ma.nunique() < 2 or mb.nunique() < 2:
                continue
            r, p = stats.phi(ma, mb) if hasattr(stats, "phi") else stats.pearsonr(ma, mb)
            pairs.append({"x": a, "y": b, "phi_like": float(r), "p_value": float(p)})
        pairs.sort(key=lambda x: (-abs(float(x["phi_like"])), x["x"], x["y"]))
        return {"rates": rates, "strongest_missingness_co_movements": pairs[:15]}

    @staticmethod
    def _attach_bh_qvalues(rows: List[Dict[str, Any]], indexed_pvalues: List[Tuple[int, float]]) -> None:
        if not indexed_pvalues:
            return
        ordered = sorted(indexed_pvalues, key=lambda t: (t[1], t[0]))
        m = len(ordered)
        qvals: Dict[int, float] = {}
        running = 1.0
        for rank in range(m, 0, -1):
            idx, p = ordered[rank - 1]
            running = min(running, float(p) * m / rank)
            qvals[idx] = min(1.0, running)
        for idx, q in qvals.items():
            rows[idx]["q_value_bh"] = float(q)

    @staticmethod
    def _attach_family_qvalues(items: List[Dict[str, Any]]) -> None:
        valid = [(i, float(item["p_value"])) for i, item in enumerate(items)
                 if item.get("p_value") is not None and np.isfinite(float(item["p_value"]))]
        m = len(valid)
        if not m:
            return
        ordered = sorted(valid, key=lambda t: (t[1], str(items[t[0]].get("key", ""))))
        qvals: Dict[int, float] = {}
        running = 1.0
        for rank in range(m, 0, -1):
            idx, p = ordered[rank - 1]
            running = min(running, p * m / rank)
            qvals[idx] = min(1.0, running)
        for idx, q in qvals.items():
            items[idx]["q_value_bh_family"] = float(q)
            items[idx]["fdr_adjustment"] = "benjamini_hochberg_family_wide"
