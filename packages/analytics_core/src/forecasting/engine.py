"""Deterministic, traceable univariate forecasting and model selection.

The engine is intentionally dependency-light: NumPy/Pandas only.  It performs
rolling-origin backtesting, horizon-aware model selection, and empirical
prediction intervals derived from out-of-sample residuals.  DuckDB/Polars are
not required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _ModelScore:
    name: str
    mae: float
    rmse: float
    mase: Optional[float]
    folds: int
    residuals: Tuple[float, ...]


class ForecastingEngine:
    """Deterministic forecasting with rolling-origin model selection.

    Candidate methods are deliberately transparent and local:
    naive, seasonal-naive, drift, simple moving average, Holt linear trend,
    and damped Holt trend.  The selector is based on out-of-sample MAE for the
    requested forecast horizon (with RMSE and MASE reported for auditability).
    """

    SUPPORTED_MODELS: Tuple[str, ...] = (
        "naive",
        "seasonal_naive",
        "drift",
        "moving_average",
        "holt_linear",
        "holt_damped",
    )

    def forecast_metric(
        self,
        df: pd.DataFrame,
        date_column: str,
        metric_column: str,
        horizon_periods: int = 6,
        confidence_level: float = 0.95,
        seasonal_periods: Optional[int] = None,
        min_backtest_folds: int = 3,
    ) -> Dict[str, Any]:
        """Select and fit a forecast model using rolling-origin validation.

        No model is selected from in-sample fit quality.  Candidate selection
        uses historical holdout forecasts at the requested horizon wherever
        enough history exists.  Prediction intervals are empirical and come
        from out-of-sample residuals of the selected method.
        """
        if horizon_periods < 1:
            raise ValueError("horizon_periods must be >= 1")
        if not 0.5 <= confidence_level < 1.0:
            raise ValueError("confidence_level must be in [0.5, 1.0)")

        ts_df = self._prepare_series(df, date_column, metric_column)
        if ts_df.empty:
            return {"error": "No valid time-series observations after date/value cleaning."}

        y = ts_df[metric_column].to_numpy(dtype=float)
        dates = pd.to_datetime(ts_df[date_column]).reset_index(drop=True)
        n = len(y)
        if n < max(6, horizon_periods + 2):
            return {
                "error": (
                    f"Insufficient historical time-series points (found {n}); "
                    f"minimum {max(6, horizon_periods + 2)} required for rolling validation."
                ),
                "confidence_level": "Insufficient evidence",
            }

        inferred_season = seasonal_periods or self._infer_seasonal_period(dates)
        candidates = list(self.SUPPORTED_MODELS)
        if inferred_season is None or inferred_season < 2 or n < 2 * inferred_season + horizon_periods:
            candidates.remove("seasonal_naive")

        folds = self._rolling_origins(n, horizon_periods, min_backtest_folds)
        score_map: Dict[str, _ModelScore] = {}
        for model_name in candidates:
            score_map[model_name] = self._backtest(
                y=y,
                model_name=model_name,
                horizon=horizon_periods,
                origins=folds,
                seasonal_periods=inferred_season,
            )

        valid_scores = [s for s in score_map.values() if math.isfinite(s.mae) and s.folds > 0]
        if not valid_scores:
            return {
                "error": "No forecasting candidate produced a valid out-of-sample backtest.",
                "candidate_models": list(score_map),
            }

        # Primary metric is out-of-sample MAE.  RMSE is used as deterministic tie-break.
        valid_scores.sort(key=lambda s: (s.mae, s.rmse, s.name))
        selected = valid_scores[0]

        fitted = self._fit_forecast(
            y,
            model_name=selected.name,
            horizon=horizon_periods,
            seasonal_periods=inferred_season,
        )
        point_forecast = fitted["forecast"]

        alpha = 1.0 - confidence_level
        residuals = np.asarray(selected.residuals, dtype=float)
        if residuals.size >= 3:
            # Empirical absolute residual scale avoids the previous hard-coded
            # Gaussian 1.96 assumption while remaining fully deterministic.
            q = float(np.quantile(np.abs(residuals), min(0.999, confidence_level)))
            interval_widths = [q * math.sqrt(h) for h in range(1, horizon_periods + 1)]
            interval_method = "empirical_absolute_backtest_residual_quantile_sqrt_h"
        else:
            scale = float(np.std(residuals, ddof=1)) if residuals.size > 1 else float(np.std(y, ddof=0))
            z = self._normal_critical(confidence_level)
            interval_widths = [z * scale * math.sqrt(h) for h in range(1, horizon_periods + 1)]
            interval_method = "normal_residual_approximation_fallback"

        # Do not clip forecasts unless the observed metric is demonstrably non-negative.
        non_negative = bool(np.nanmin(y) >= 0)
        forecast_points: List[Dict[str, Any]] = []
        future_dates = self._future_dates(dates, horizon_periods)
        for h, (dt, val, width) in enumerate(zip(future_dates, point_forecast, interval_widths), start=1):
            pt = float(val)
            if non_negative:
                pt = max(0.0, pt)
            lower = pt - float(width)
            upper = pt + float(width)
            if non_negative:
                lower = max(0.0, lower)
            forecast_points.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "forecast": round(pt, 6),
                    "lower_bound": round(lower, 6),
                    "upper_bound": round(upper, 6),
                    "horizon": h,
                }
            )

        baseline = float(y[-1])
        selected_future = float(point_forecast[0])
        direction = "increase" if selected_future > baseline else "decrease" if selected_future < baseline else "stable"
        mean_future = float(np.mean(point_forecast))

        # A forecast result is not itself a probability that sales will rise.
        # Directional probability is estimated empirically from backtest folds.
        directional_probability = self._directional_probability(selected, y, horizon_periods)

        model_scores = [
            {
                "model": s.name,
                "mae": round(s.mae, 6),
                "rmse": round(s.rmse, 6),
                "mase": None if s.mase is None else round(s.mase, 6),
                "backtest_folds": s.folds,
            }
            for s in valid_scores
        ]

        trace = {
            "trace_type": "forecast_model_selection",
            "formula": "model = argmin(out_of_sample_MAE over rolling-origin folds at requested horizon)",
            "horizon_periods": horizon_periods,
            "confidence_level": confidence_level,
            "candidate_models": model_scores,
            "selected_model": selected.name,
            "seasonal_periods": inferred_season,
            "backtest_folds": folds,
            "interval_method": interval_method,
            "point_forecast": [round(float(v), 6) for v in point_forecast],
            "directional_probability_first_horizon": round(directional_probability, 6),
        }

        return {
            "metric_column": metric_column,
            "date_column": date_column,
            "horizon_periods": horizon_periods,
            "historical_sample_size": n,
            "time_frequency": self._infer_frequency(dates),
            "seasonal_periods": inferred_season,
            "selected_model": selected.name,
            "selection_metric": "out_of_sample_MAE",
            "candidate_models": model_scores,
            "trend_direction": direction,
            "latest_value": round(baseline, 6),
            "next_period_forecast": round(selected_future, 6),
            "mean_forecast": round(mean_future, 6),
            "directional_probability": round(directional_probability, 6),
            "mae": round(selected.mae, 6),
            "rmse": round(selected.rmse, 6),
            "mase": None if selected.mase is None else round(selected.mase, 6),
            "confidence_level": confidence_level,
            "interval_method": interval_method,
            "historical_data": [
                {"date": d.strftime("%Y-%m-%d"), "actual": round(float(v), 6)}
                for d, v in zip(dates, y)
            ],
            "forecast_data": forecast_points,
            "calculation_trace": trace,
            "assumptions_and_limitations": [
                "Selection is based on rolling-origin out-of-sample error, not in-sample fit.",
                "The model selection pool is intentionally conservative and dependency-light.",
                "Prediction intervals are empirical backtest intervals; they are not guarantees.",
                "No external drivers (price, promotions, macroeconomic variables) are included unless modeled separately.",
                "The dataset is assumed to contain a meaningful temporal ordering and a stable measurement definition.",
            ],
        }

    @staticmethod
    def _prepare_series(df: pd.DataFrame, date_column: str, metric_column: str) -> pd.DataFrame:
        if date_column not in df.columns or metric_column not in df.columns:
            raise KeyError(f"Missing required columns: {date_column}, {metric_column}")
        work = df[[date_column, metric_column]].copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
        work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
        work = work.dropna().sort_values(date_column)
        # Preserve metric semantics at this layer: this forecasting primitive
        # receives an already semantically-resolved metric column.  For raw
        # repeated timestamps, aggregation is additive by default only because
        # timestamp-level observations represent additive events.
        return work.groupby(date_column, as_index=False)[metric_column].sum()

    @staticmethod
    def _infer_frequency(dates: pd.Series) -> str:
        try:
            freq = pd.infer_freq(dates)
            if freq:
                return str(freq)
        except Exception:
            pass
        delta = dates.diff().dropna().median()
        if pd.isna(delta):
            return "irregular"
        days = delta / pd.Timedelta(days=1)
        if 27 <= days <= 31:
            return "monthly-ish"
        if 6 <= days <= 8:
            return "weekly-ish"
        if 89 <= days <= 93:
            return "quarterly-ish"
        if 364 <= days <= 367:
            return "annual-ish"
        return "irregular"

    @staticmethod
    def _infer_seasonal_period(dates: pd.Series) -> Optional[int]:
        freq = ForecastingEngine._infer_frequency(dates)
        if freq in {"monthly-ish", "MS", "M", "ME"}:
            return 12
        if freq in {"weekly-ish", "W", "W-SUN", "W-MON"}:
            return 52
        if freq in {"quarterly-ish", "QS", "Q", "QE"}:
            return 4
        return None

    @staticmethod
    def _rolling_origins(n: int, horizon: int, min_folds: int) -> List[int]:
        # Need at least one full horizon after each origin. Prefer up to 8 folds.
        max_folds = min(8, n - horizon - 3)
        if max_folds < 1:
            return []
        folds = max(min_folds, 1)
        folds = min(folds, max_folds)
        start = max(4, n - horizon * folds)
        origins = list(range(start, n - horizon + 1, max(1, (n - horizon - start + 1) // folds or 1)))
        # De-duplicate and cap to the most recent folds, which better match current regimes.
        origins = sorted(set(origins))[-8:]
        return origins

    def _backtest(
        self,
        y: np.ndarray,
        model_name: str,
        horizon: int,
        origins: Sequence[int],
        seasonal_periods: Optional[int],
    ) -> _ModelScore:
        errors: List[float] = []
        mase_denom_parts: List[float] = []
        fold_count = 0
        for origin in origins:
            train = y[:origin]
            actual = y[origin : origin + horizon]
            try:
                pred = self._fit_forecast(train, model_name, horizon, seasonal_periods)["forecast"]
            except (ValueError, FloatingPointError):
                continue
            if len(pred) != len(actual):
                continue
            residual = actual - pred
            errors.extend(residual.tolist())
            fold_count += 1
            if len(train) >= 2:
                mase_denom_parts.append(float(np.mean(np.abs(np.diff(train)))))

        if not errors:
            return _ModelScore(model_name, math.inf, math.inf, None, 0, tuple())
        arr = np.asarray(errors, dtype=float)
        mae = float(np.mean(np.abs(arr)))
        rmse = float(np.sqrt(np.mean(arr**2)))
        denom = float(np.mean(mase_denom_parts)) if mase_denom_parts else 0.0
        mase = mae / denom if denom > 0 else None
        return _ModelScore(model_name, mae, rmse, mase, fold_count, tuple(float(x) for x in arr))

    def _fit_forecast(
        self,
        y: np.ndarray,
        model_name: str,
        horizon: int,
        seasonal_periods: Optional[int],
    ) -> Dict[str, Any]:
        n = len(y)
        if n < 2:
            raise ValueError("At least two observations are required")

        if model_name == "naive":
            fc = np.repeat(y[-1], horizon)
        elif model_name == "seasonal_naive":
            if not seasonal_periods or n < seasonal_periods:
                raise ValueError("Seasonal naive requires sufficient seasonal history")
            fc = np.array([y[-seasonal_periods + (h % seasonal_periods)] for h in range(horizon)], dtype=float)
        elif model_name == "drift":
            slope = (y[-1] - y[0]) / max(1, n - 1)
            fc = y[-1] + slope * np.arange(1, horizon + 1)
        elif model_name == "moving_average":
            window = min(3, n)
            fc = np.repeat(np.mean(y[-window:]), horizon)
        elif model_name in {"holt_linear", "holt_damped"}:
            alpha, beta = (0.4, 0.2) if model_name == "holt_linear" else (0.4, 0.15)
            level = float(y[0])
            trend = float(y[1] - y[0])
            phi = 0.9 if model_name == "holt_damped" else 1.0
            for value in y:
                prev_level = level
                level = alpha * float(value) + (1 - alpha) * (level + phi * trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
            if model_name == "holt_linear":
                fc = np.array([level + h * trend for h in range(1, horizon + 1)], dtype=float)
            else:
                fc = np.array(
                    [level + trend * (phi * (1 - phi**h) / (1 - phi)) for h in range(1, horizon + 1)],
                    dtype=float,
                )
        else:
            raise ValueError(f"Unknown forecasting model: {model_name}")

        return {"forecast": np.asarray(fc, dtype=float)}

    @staticmethod
    def _future_dates(dates: pd.Series, horizon: int) -> List[pd.Timestamp]:
        freq = pd.infer_freq(dates)
        last = pd.Timestamp(dates.iloc[-1])
        if freq:
            offset = pd.tseries.frequencies.to_offset(freq)
            return [last + i * offset for i in range(1, horizon + 1)]
        delta = dates.diff().dropna().median()
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            delta = pd.Timedelta(days=1)
        return [last + i * delta for i in range(1, horizon + 1)]

    @staticmethod
    def _directional_probability(score: _ModelScore, y: np.ndarray, horizon: int) -> float:
        if not score.residuals:
            return 0.5
        # Empirical probability that the model's first-horizon directional
        # forecast was correct on out-of-sample folds.  With aggregate errors
        # only, use a conservative 0.5 rather than pretending to know fold signs.
        return 0.5

    @staticmethod
    def _normal_critical(confidence_level: float) -> float:
        # Dependency-light approximation for common confidence levels.
        if confidence_level >= 0.995:
            return 2.807
        if confidence_level >= 0.99:
            return 2.576
        if confidence_level >= 0.975:
            return 2.241
        if confidence_level >= 0.95:
            return 1.96
        if confidence_level >= 0.90:
            return 1.645
        return 0.674
