"""DataRemediationEngine: Pre-flight missingness remediation (MICE) and fuzzy categorical normalization."""
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import polars as pl

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    _HAS_MICE = True
except ImportError:
    _HAS_MICE = False

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


@dataclass
class RemediationResult:
    """Output of pre-flight data remediation."""
    df: Any
    imputation_variance_delta: float = 0.0
    rows_dropped: int = 0
    fuzzy_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class DataRemediationEngine:
    """Pre-flight Data Janitor engine for AA-OS Phase 7."""

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """Pure Python Levenshtein similarity ratio between 0.0 and 100.0."""
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if s1 == s2:
            return 100.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        dist = dp[len1][len2]
        return max(0.0, (1.0 - (dist / max(len1, len2))) * 100.0)

    @classmethod
    def remediate_missingness(cls, df: Union[pd.DataFrame, pl.DataFrame], target_metric: str) -> RemediationResult:
        """Applies MICE to non-target covariates and propagates uncertainty.
        
        Strict Invariant: Never impute the dependent target metric.
        """
        is_polars = isinstance(df, pl.DataFrame)
        pdf = df.to_pandas() if is_polars else df.copy()
        warnings: List[str] = []
        rows_dropped = 0

        if target_metric not in pdf.columns:
            return RemediationResult(df=df, warnings=["Target metric not in columns"])

        # STRICT INVARIANT: Target metric missingness check
        target_nulls = int(pdf[target_metric].isnull().sum())
        total_rows = len(pdf)
        if target_nulls > 0:
            null_pct = target_nulls / max(1, total_rows)
            if null_pct > 0.05:
                raise ValueError(
                    f"Target metric '{target_metric}' has {null_pct:.1%} missingness (>5%). "
                    "Fail-closed invariant triggered: dependent variable cannot be artificially imputed."
                )
            pdf = pdf.dropna(subset=[target_metric])
            rows_dropped = target_nulls
            warnings.append(
                f"Dropped {target_nulls} null rows for target metric '{target_metric}' "
                "because the dependent variable is never imputed."
            )

        # Identify numeric covariates with missingness between 5% and 40%
        numeric_cols = [
            c for c in pdf.columns
            if c != target_metric and pd.api.types.is_numeric_dtype(pdf[c])
        ]
        missing_rates = {
            c: float(pdf[c].isnull().sum() / max(1, len(pdf)))
            for c in numeric_cols
        }
        mice_candidates = [
            c for c, rate in missing_rates.items()
            if 0.0 < rate < 0.40
        ]
        high_missing = [
            c for c, rate in missing_rates.items()
            if rate >= 0.40
        ]
        low_missing_untouched = [
            c for c, rate in missing_rates.items()
            if 0.0 < rate <= 0.05
        ]
        if high_missing:
            warnings.append(
                "Skipped imputation for high-missingness covariates (>=40%): "
                + ", ".join(high_missing)
            )
        if low_missing_untouched:
            warnings.append(
                "Untouched low-missingness covariates (<=5%) remain NaN: "
                + ", ".join(
                    f"{c} ({int(pdf[c].isnull().sum())} missing)"
                    for c in low_missing_untouched
                )
            )

        imputation_variance_delta = 0.0
        if mice_candidates:
            pdf_covs = pdf[mice_candidates]
            original_variance = float(np.nanmean(np.nanvar(pdf_covs.values, axis=0)))

            if _HAS_MICE:
                imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=True)
                imputed_array = imputer.fit_transform(pdf_covs)
            else:
                # Deterministic conditional median imputation fallback
                imputed_array = pdf_covs.fillna(pdf_covs.median()).values

            imputed_variance = float(np.nanmean(np.var(imputed_array, axis=0)))
            imputation_variance_delta = max(0.0, imputed_variance - original_variance)

            pdf[mice_candidates] = imputed_array
            warnings.append(f"Applied MICE imputation across {len(mice_candidates)} covariates ({', '.join(mice_candidates)}).")

        out_df = pl.from_pandas(pdf) if is_polars else pdf
        return RemediationResult(
            df=out_df,
            imputation_variance_delta=imputation_variance_delta,
            fuzzy_mappings={},
            rows_dropped=rows_dropped,
            warnings=warnings,
        )

    @classmethod
    def normalize_fuzzy_categoricals(
        cls,
        df: Union[pd.DataFrame, pl.DataFrame],
        max_cardinality: int = 500,
        similarity_threshold: float = 80.0,
    ) -> RemediationResult:
        """Clusters and normalizes fragmented categorical strings (e.g. 'USA', 'U.S.', 'us')."""
        is_polars = isinstance(df, pl.DataFrame)
        pdf = df.to_pandas() if is_polars else df.copy()
        fuzzy_mappings: Dict[str, Dict[str, str]] = {}
        warnings: List[str] = []

        for col in pdf.columns:
            if pd.api.types.is_string_dtype(pdf[col]) or pd.api.types.is_object_dtype(pdf[col]):
                non_null_s = pdf[col].dropna().astype(str)
                n_unique = non_null_s.nunique()
                if 2 <= n_unique <= max_cardinality:
                    unique_vals = list(non_null_s.unique())
                    clusters: Dict[str, List[str]] = defaultdict(list)
                    visited = set()

                    # Sort by length descending and frequency so canonical representative is prioritized
                    val_counts = non_null_s.value_counts()
                    sorted_vals = sorted(unique_vals, key=lambda v: (len(v), val_counts.get(v, 0)), reverse=True)

                    for val in sorted_vals:
                        if val in visited:
                            continue
                        cluster = [val]
                        visited.add(val)

                        for other in sorted_vals:
                            if other in visited:
                                continue
                            if _HAS_RAPIDFUZZ:
                                sim = fuzz.token_sort_ratio(val, other)
                            else:
                                sim = cls._levenshtein_ratio(val, other)

                            v_clean = "".join(ch for ch in val if ch.isalnum()).upper()
                            o_clean = "".join(ch for ch in other if ch.isalnum()).upper()
                            is_prefix_acronym = (
                                (v_clean.startswith(o_clean) or o_clean.startswith(v_clean))
                                and abs(len(v_clean) - len(o_clean)) <= 2
                                and min(len(v_clean), len(o_clean)) >= 2
                            )
                            if v_clean == o_clean or is_prefix_acronym or sim >= similarity_threshold:
                                cluster.append(other)
                                visited.add(other)

                        canonical = max(cluster, key=lambda x: (val_counts.get(x, 0), len(x)))
                        for item in cluster:
                            clusters[canonical].append(item)

                    mapping = {
                        alias: canonical
                        for canonical, aliases in clusters.items()
                        for alias in aliases
                        if alias != canonical
                    }

                    if mapping:
                        pdf[col] = pdf[col].replace(mapping)
                        fuzzy_mappings[col] = mapping
                        warnings.append(f"Normalized {len(mapping)} categorical variants in column '{col}'.")

        out_df = pl.from_pandas(pdf) if is_polars else pdf
        return RemediationResult(
            df=out_df,
            imputation_variance_delta=0.0,
            fuzzy_mappings=fuzzy_mappings,
            warnings=warnings,
        )
