"""UpliftEngine: T-Learner and Inverse Probability Weighting (IPW) for CATE estimation and 4-quadrant segmentation."""
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import polars as pl

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@dataclass
class UpliftSegmentation:
    """4-Quadrant Uplift Segmentation outcome."""
    persuadables_mask: np.ndarray
    sure_things_mask: np.ndarray
    lost_causes_mask: np.ndarray
    sleeping_dogs_mask: np.ndarray
    cate_scores: np.ndarray


class UpliftEngine:
    """Computes Conditional Average Treatment Effect (CATE) using T-Learner and IPW."""

    @staticmethod
    def compute_t_learner_uplift(
        df: Union[pd.DataFrame, pl.DataFrame],
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        propensity_scores: Optional[np.ndarray] = None,
    ) -> UpliftSegmentation:
        """Computes Uplift using T-Learner approach with optional Inverse Probability Weighting (IPW)."""
        pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df

        required_cols = list(dict.fromkeys([*feature_cols, outcome_col, treatment_col]))
        missing_cols = [c for c in required_cols if c not in pdf.columns]
        if missing_cols:
            raise ValueError(f"UpliftEngine missing required columns: {missing_cols}")

        if pdf[feature_cols].isna().any().any():
            raise ValueError("UpliftEngine requires imputed features; NaN detected.")
        if pdf[treatment_col].isna().any():
            raise ValueError("NaN in treatment column.")
        if pdf[outcome_col].isna().any():
            raise ValueError("NaN in outcome column.")

        X = pdf[feature_cols].to_numpy(dtype=float)
        y = pdf[outcome_col].to_numpy(dtype=float)
        t = pdf[treatment_col].to_numpy(dtype=int)

        if not set(np.unique(t)).issubset({0, 1}):
            raise ValueError("Uplift treatment must be binary 0/1.")
        if len(np.unique(t)) < 2:
            raise ValueError("Uplift requires both treatment and control units.")
        if len(np.unique(y)) < 2:
            raise ValueError("Uplift requires a non-constant outcome.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("UpliftEngine requires finite feature and outcome values.")
        # The current T-Learner implementation uses LogisticRegression and
        # predict_proba, so its estimand is binary-outcome uplift only.
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("Uplift T-Learner currently supports binary outcomes 0/1 only.")

        n = len(X)
        if not _HAS_SKLEARN:
            # Fallback simple deterministic conditional difference
            t_mask = t == 1
            c_mask = t == 0
            mean_t = float(np.mean(y[t_mask])) if np.sum(t_mask) > 0 else 0.5
            mean_c = float(np.mean(y[c_mask])) if np.sum(c_mask) > 0 else 0.5
            diff = mean_t - mean_c
            cate = np.full(n, diff)
            p_y1 = np.full(n, mean_t)
            p_y0 = np.full(n, mean_c)
        else:
            # 1. Propensity score estimation if not provided
            if propensity_scores is None:
                ps_model = LogisticRegression().fit(X, t)
                propensity_scores = ps_model.predict_proba(X)[:, 1]
            else:
                propensity_scores = np.asarray(propensity_scores, dtype=float)
                if propensity_scores.shape != (n,):
                    raise ValueError("Propensity scores must have one value per row.")
                if not np.all(np.isfinite(propensity_scores)):
                    raise ValueError("Propensity scores must be finite.")
                if np.any((propensity_scores <= 0.0) | (propensity_scores >= 1.0)):
                    raise ValueError("Propensity scores must lie strictly between 0 and 1.")

            propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
            weights = np.where(t == 1, 1.0 / propensity_scores, 1.0 / (1.0 - propensity_scores))

            # 2. Train Treatment model (T=1)
            t_idx = t == 1
            if np.sum(t_idx) > 2 and len(np.unique(y[t_idx])) > 1:
                model_t = LogisticRegression().fit(X[t_idx], y[t_idx], sample_weight=weights[t_idx])
                p_y1 = model_t.predict_proba(X)[:, 1]
            else:
                p_y1 = np.full(n, float(np.mean(y[t_idx])) if np.sum(t_idx) > 0 else 0.5)

            # 3. Train Control model (T=0)
            c_idx = t == 0
            if np.sum(c_idx) > 2 and len(np.unique(y[c_idx])) > 1:
                model_c = LogisticRegression().fit(X[c_idx], y[c_idx], sample_weight=weights[c_idx])
                p_y0 = model_c.predict_proba(X)[:, 1]
            else:
                p_y0 = np.full(n, float(np.mean(y[c_idx])) if np.sum(c_idx) > 0 else 0.5)

            cate = p_y1 - p_y0

        # 4. Segment into the 4 Quadrants
        persuadables = (p_y1 > 0.5) & (p_y0 <= 0.5)
        sure_things = (p_y1 > 0.5) & (p_y0 > 0.5)
        lost_causes = (p_y1 <= 0.5) & (p_y0 <= 0.5)
        sleeping_dogs = (p_y1 <= 0.5) & (p_y0 > 0.5)

        return UpliftSegmentation(
            persuadables_mask=persuadables,
            sure_things_mask=sure_things,
            lost_causes_mask=lost_causes,
            sleeping_dogs_mask=sleeping_dogs,
            cate_scores=cate,
        )
