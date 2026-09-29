"""
segmentation_engine.py
======================
Unsupervised clustering pipeline for AA-OS.

Pipeline
--------
1. Feature eligibility filtering
2. Median imputation + StandardScaler
3. Silhouette-guided k search (k = 2 ... max_k)
4. Bootstrap stability check (ARI across 5 subsamples)
5. Silhouette gate (>= 0.15)
6. Cluster profiling with effect-size-ranked discriminating features
7. Usefulness gate (at least one discriminating feature with |effect| >= 0.30)

claim_ceiling is always 'OBSERVATION' -- clusters are descriptive, never causal.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterProfile:
    """Summary statistics for a single cluster."""
    cluster_id: int
    n_members: int
    member_fraction: float          # fraction of total rows in this cluster
    centroid: Dict[str, float]      # feature -> mean value (original scale)
    discriminating_features: List[Dict]  # [{feature, cluster_mean, global_mean, effect_size}]
    label: str                      # auto-generated interpretive label


@dataclass
class SegmentationResult:
    """Full output of the segmentation pipeline."""
    status: str                     # 'SEGMENTED' | 'REJECTED' | 'FAILED' | 'INSUFFICIENT_DATA'
    rejection_reason: Optional[str] # populated when status != 'SEGMENTED'
    n_clusters: int                 # 0 if rejected / failed
    selected_k: Optional[int]
    features_used: List[str]
    n_rows: int
    silhouette_score: Optional[float]
    stability_score: Optional[float]
    cluster_profiles: List[ClusterProfile]
    claim_ceiling: str = "OBSERVATION"  # clusters are never causal
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ID_KEYWORDS = {"_id", "id", "key", "pk", "uuid", "code"}
_HIGH_CARDINALITY_RATIO = 0.80  # unique values / n_rows > this -> treat as identifier


def _has_id_keyword(col: str) -> bool:
    """Return True if the column name contains any identifier keyword."""
    lower = col.lower()
    for kw in _ID_KEYWORDS:
        if kw in lower:
            return True
    return False


def _is_identifier_column(series: pd.Series) -> bool:
    """True when column name looks like an ID AND cardinality is very high."""
    if not _has_id_keyword(series.name):
        return False
    n_total = len(series.dropna())
    if n_total == 0:
        return False
    cardinality_ratio = series.nunique() / n_total
    return cardinality_ratio > _HIGH_CARDINALITY_RATIO


def _select_eligible_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    missingness_threshold: float = 0.60,
) -> List[str]:
    """Return list of eligible numeric column names."""
    if feature_cols is not None:
        candidates = [c for c in feature_cols if c in df.columns]
    else:
        candidates = df.select_dtypes(include=[np.number]).columns.tolist()

    eligible = []
    for col in candidates:
        series = df[col]
        # Missingness gate
        miss_frac = series.isna().mean()
        if miss_frac >= missingness_threshold:
            logger.debug("Dropping %s: missingness=%.2f", col, miss_frac)
            continue
        # Zero-variance gate (checked on non-null values)
        if series.dropna().std(ddof=0) == 0:
            logger.debug("Dropping %s: zero variance", col)
            continue
        # Identifier gate
        if _is_identifier_column(series):
            logger.debug("Dropping %s: identifier-style column", col)
            continue
        eligible.append(col)

    return eligible


def _impute_and_scale(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, StandardScaler]:
    """Median-impute then StandardScale. Returns (X_scaled, fitted_scaler)."""
    X = df[feature_cols].copy()
    for col in feature_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))
    return X_scaled, scaler


def _fit_kmeans(X: np.ndarray, k: int, seed: int) -> KMeans:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(X)
    return km


def _k_search(
    X: np.ndarray,
    max_k: int,
    n_rows: int,
    seed: int,
) -> Tuple[int, float, Dict[int, float]]:
    """
    Search k = 2 ... min(max_k, n_rows // 10).
    Returns (best_k, best_silhouette, {k: silhouette}).
    """
    k_upper = min(max_k, n_rows // 10)
    k_upper = max(k_upper, 2)  # always test at least k=2

    scores: Dict[int, float] = {}
    for k in range(2, k_upper + 1):
        try:
            km = _fit_kmeans(X, k, seed)
            labels = km.labels_
            # silhouette requires at least 2 distinct labels
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(X, labels)
            scores[k] = sil
            logger.debug("k=%d silhouette=%.4f", k, sil)
        except Exception as exc:
            logger.warning("k=%d failed: %s", k, exc)

    if not scores:
        return 2, -1.0, scores

    best_k = max(scores, key=scores.__getitem__)
    return best_k, scores[best_k], scores


def _bootstrap_stability(
    X: np.ndarray,
    k: int,
    seed: int,
    n_bootstraps: int = 5,
    subsample_fraction: float = 0.80,
) -> float:
    """
    Fit KMeans on bootstrap subsamples and measure cluster-assignment consistency
    via mean Adjusted Rand Index across all pairs of bootstrap runs.
    """
    rng = np.random.RandomState(seed + 1000)
    n = X.shape[0]
    sub_n = max(k + 1, int(n * subsample_fraction))

    all_labels: List[np.ndarray] = []
    for i in range(n_bootstraps):
        idx = rng.choice(n, size=sub_n, replace=False)
        X_sub = X[idx]
        try:
            km = _fit_kmeans(X_sub, k, seed=seed + i)
            # Predict cluster for ALL rows (not just the subsample)
            labels = km.predict(X)
            all_labels.append(labels)
        except Exception as exc:
            logger.warning("Bootstrap %d failed: %s", i, exc)

    if len(all_labels) < 2:
        return 0.0

    aris = [
        adjusted_rand_score(a, b)
        for a, b in itertools.combinations(all_labels, 2)
    ]
    return float(np.mean(aris))


def _profile_cluster(
    cluster_id: int,
    mask: np.ndarray,
    X_original: pd.DataFrame,
    feature_cols: List[str],
    global_means: np.ndarray,
    global_stds: np.ndarray,
    effect_threshold: float = 0.30,
) -> ClusterProfile:
    """Build a ClusterProfile for a single cluster given original-scale data."""
    cluster_data = X_original[mask]
    n_members = int(mask.sum())
    member_fraction = n_members / len(X_original)

    centroid: Dict[str, float] = {}
    discriminating: List[Dict] = []

    for i, feat in enumerate(feature_cols):
        cluster_mean = float(cluster_data[feat].mean())
        global_mean = float(global_means[i])
        global_std = float(global_stds[i])
        centroid[feat] = cluster_mean

        if global_std > 0:
            effect_size = (cluster_mean - global_mean) / global_std
        else:
            effect_size = 0.0

        if abs(effect_size) >= effect_threshold:
            discriminating.append(
                {
                    "feature": feat,
                    "cluster_mean": cluster_mean,
                    "global_mean": global_mean,
                    "effect_size": effect_size,
                }
            )

    # Sort by absolute effect size descending
    discriminating.sort(key=lambda d: abs(d["effect_size"]), reverse=True)

    # Auto-label from top-2 discriminating features
    label = _auto_label(discriminating[:2])

    return ClusterProfile(
        cluster_id=cluster_id,
        n_members=n_members,
        member_fraction=member_fraction,
        centroid=centroid,
        discriminating_features=discriminating,
        label=label,
    )


def _auto_label(top_features: List[Dict]) -> str:
    """
    Generate a human-readable label from up to 2 discriminating features.
    e.g. 'High revenue, Low churn rate'
    """
    if not top_features:
        return "Undifferentiated cluster"

    parts = []
    for fd in top_features:
        direction = "High" if fd["effect_size"] > 0 else "Low"
        name = fd["feature"].replace("_", " ")
        parts.append(f"{direction} {name}")

    return ", ".join(parts)


def _make_insufficient(reason: str, n_rows: int = 0) -> SegmentationResult:
    return SegmentationResult(
        status="INSUFFICIENT_DATA",
        rejection_reason=reason,
        n_clusters=0,
        selected_k=None,
        features_used=[],
        n_rows=n_rows,
        silhouette_score=None,
        stability_score=None,
        cluster_profiles=[],
        summary=reason,
    )


def _make_rejected(
    reason: str,
    features_used: List[str],
    n_rows: int,
    silhouette: Optional[float],
    stability: Optional[float],
) -> SegmentationResult:
    return SegmentationResult(
        status="REJECTED",
        rejection_reason=reason,
        n_clusters=0,
        selected_k=None,
        features_used=features_used,
        n_rows=n_rows,
        silhouette_score=silhouette,
        stability_score=stability,
        cluster_profiles=[],
        summary=f"Segmentation rejected: {reason}",
    )


def _make_failed(reason: str, n_rows: int) -> SegmentationResult:
    return SegmentationResult(
        status="FAILED",
        rejection_reason=reason,
        n_clusters=0,
        selected_k=None,
        features_used=[],
        n_rows=n_rows,
        silhouette_score=None,
        stability_score=None,
        cluster_profiles=[],
        summary=f"Segmentation failed: {reason}",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SegmentationEngine:
    """
    Unsupervised clustering pipeline.

    Public API
    ----------
    result = SegmentationEngine().segment(df, feature_cols=None, max_k=6, seed=42)
    """

    def segment(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        max_k: int = 6,
        seed: int = 42,
    ) -> SegmentationResult:
        """
        Run the full segmentation pipeline on *df*.

        Parameters
        ----------
        df           : Input DataFrame (any columns).
        feature_cols : Explicit list of feature columns to use; if None, all
                       numeric columns are considered.
        max_k        : Maximum number of clusters to test (inclusive).
        seed         : Random seed for reproducibility.

        Returns
        -------
        SegmentationResult with status in {'SEGMENTED', 'REJECTED', 'FAILED',
        'INSUFFICIENT_DATA'}.
        """
        # guard: empty dataframe
        if df is None or df.empty:
            return _make_insufficient("empty dataframe")

        n_rows = len(df)

        # guard: too few rows
        if n_rows < 20:
            return _make_insufficient(
                "fewer than 20 rows — clustering unreliable", n_rows=n_rows
            )

        # Step 1: feature eligibility
        eligible = _select_eligible_features(df, feature_cols)

        if len(eligible) < 2:
            return _make_insufficient(
                "fewer than 2 eligible numeric features", n_rows=n_rows
            )

        # Edge case: verify non-null variance survives after imputation
        X_check = df[eligible].copy()
        for col in eligible:
            X_check[col] = X_check[col].fillna(X_check[col].median())
        surviving = [c for c in eligible if X_check[c].std(ddof=0) > 0]
        if len(surviving) < 2:
            return _make_insufficient(
                "fewer than 2 eligible numeric features after imputation",
                n_rows=n_rows,
            )
        eligible = surviving

        # Step 2: preprocessing
        try:
            X_scaled, scaler = _impute_and_scale(df, eligible)
        except Exception as exc:
            return _make_failed(f"preprocessing error: {exc}", n_rows)

        # Original-scale data (after imputation) for profiling
        X_orig = df[eligible].copy()
        for col in eligible:
            X_orig[col] = X_orig[col].fillna(X_orig[col].median())

        global_means = X_orig.mean().values
        global_stds = X_orig.std(ddof=0).values

        # Step 3: k search
        try:
            best_k, best_sil, _all_scores = _k_search(
                X_scaled, max_k=max_k, n_rows=n_rows, seed=seed
            )
        except Exception as exc:
            return _make_failed(f"k-search error: {exc}", n_rows)

        # Step 4: stability check
        try:
            stability = _bootstrap_stability(X_scaled, k=best_k, seed=seed)
        except Exception as exc:
            return _make_failed(f"stability check error: {exc}", n_rows)

        if stability < 0.60:
            return _make_rejected(
                f"cluster structure unstable across bootstrap subsamples (ARI={stability:.2f})",
                features_used=eligible,
                n_rows=n_rows,
                silhouette=best_sil,
                stability=stability,
            )

        # Step 5: silhouette gate
        if best_sil < 0.15:
            return _make_rejected(
                f"insufficient cluster separation (silhouette={best_sil:.3f} < 0.15)",
                features_used=eligible,
                n_rows=n_rows,
                silhouette=best_sil,
                stability=stability,
            )

        # Step 6: final fit and profiling
        try:
            final_km = _fit_kmeans(X_scaled, best_k, seed)
            labels = final_km.labels_
        except Exception as exc:
            return _make_failed(f"final KMeans fit failed: {exc}", n_rows)

        profiles: List[ClusterProfile] = []
        for cid in range(best_k):
            mask = labels == cid
            profile = _profile_cluster(
                cluster_id=cid,
                mask=mask,
                X_original=X_orig,
                feature_cols=eligible,
                global_means=global_means,
                global_stds=global_stds,
            )
            profiles.append(profile)

        # Step 7: usefulness gate
        any_discriminating = any(
            len(p.discriminating_features) > 0 for p in profiles
        )
        if not any_discriminating:
            return _make_rejected(
                "clusters do not differ meaningfully on any measured feature",
                features_used=eligible,
                n_rows=n_rows,
                silhouette=best_sil,
                stability=stability,
            )

        # Build summary
        summary_lines = [
            f"Segmented {n_rows} rows into {best_k} clusters "
            f"(silhouette={best_sil:.3f}, stability={stability:.3f}).",
        ]
        for p in profiles:
            summary_lines.append(
                f"  Cluster {p.cluster_id}: n={p.n_members} ({p.member_fraction:.1%}) — {p.label}"
            )
        summary = "\n".join(summary_lines)

        return SegmentationResult(
            status="SEGMENTED",
            rejection_reason=None,
            n_clusters=best_k,
            selected_k=best_k,
            features_used=eligible,
            n_rows=n_rows,
            silhouette_score=best_sil,
            stability_score=stability,
            cluster_profiles=profiles,
            claim_ceiling="OBSERVATION",
            summary=summary,
        )
