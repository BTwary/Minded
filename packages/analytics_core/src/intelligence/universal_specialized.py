"""First-class specialist analyzers used by the universal question compiler.

These analyzers are intentionally small, deterministic, and auditable. They are
not substitutes for the full scientific loop; they provide a direct path for
question classes whose most informative answer is a specialized computation
rather than a generic hypothesis battery.
"""
from __future__ import annotations

from typing import Any, Dict, List
import hashlib
import json
import time

import numpy as np
import pandas as pd

from packages.analytics_core.src.forecasting.engine import ForecastingEngine
from packages.analytics_core.src.intelligence.universal_question_planner import (
    AutomaticClusteringEngine,
    GovernanceRiskEngine,
    MathematicalReconciliationEngine,
)


def _trace(task: str, *, question: str, dataset_name: str, fingerprints: Dict[str, Any], versions: Dict[str, Any] | None, inputs: Dict[str, Any], output: Dict[str, Any], investigation_id: str | None = None, experiment_id: str | None = None) -> Dict[str, Any]:
    payload = {
        "trace_id": f"TRACE-SPECIAL-{hashlib.sha256((task + question + dataset_name).encode()).hexdigest()[:16]}",
        "investigation_id": investigation_id or "",
        "experiment_id": experiment_id or "",
        "kind": "SPECIALIZED_ANALYSIS",
        "task": task,
        "formula": output.get("formula") or "Deterministic specialist computation",
        "inputs": inputs,
        "output": output,
        "source": {"dataset": dataset_name, "dataset_fingerprints": fingerprints, "dataset_versions": versions or {}},
    }
    payload["canonical_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return payload



def run_predictive_risk_analysis(
    *,
    df: pd.DataFrame,
    target_column: str | None,
    time_column: str | None = None,
    question: str = "",
    dataset_name: str = "dataset",
    dataset_fingerprints: Dict[str, Any] | None = None,
    dataset_versions: Dict[str, Any] | None = None,
    investigation_id: str | None = None,
    experiment_id: str | None = None,
    positive_class: Any = None,
) -> Dict[str, Any]:
    """Train time-safe binary-risk candidates and score currently negative rows.

    This is intentionally a constrained predictive path: it refuses continuous
    or multiclass outcomes, excludes obvious identifiers, keeps preprocessing
    inside the training split, and reports validation metrics before scoring
    current rows. It predicts association/risk, never causation.
    """
    started = time.perf_counter()
    if not target_column or target_column not in df.columns:
        return {"status": "TARGET_UNRESOLVED", "answer": "AA-OS could not resolve a binary prediction target."}

    y_raw = df[target_column]
    values = y_raw.dropna().unique()
    if len(values) != 2:
        return {"status": "UNSUPPORTED_TARGET", "answer": "Predictive risk scoring requires a binary outcome in the resolved target column."}

    # Positive-class semantics MUST come from the resolved analytical contract.
    # A specialist is not allowed to guess which observed level means the event.
    # This prevents an arbitrary ordering of labels from becoming scientific meaning.
    # Positive-class semantics MUST be explicit. Prefer the semantic contract
    # supplied by the caller; never derive event meaning from arbitrary label order.
    if positive_class is None:
        return {
            "status": "TARGET_SEMANTICS_UNRESOLVED",
            "answer": "Binary prediction requires an explicit positive-class semantic contract; no class was resolved.",
            "finding": "Prediction was not executed because positive-class semantics were not explicitly resolved.",
            "statement": "Prediction was not executed because positive-class semantics were not explicitly resolved.",
            "result": {"target_column": target_column, "observed_classes": [str(v) for v in values]},
            "trace": _trace(
                "PREDICTION", question=question, dataset_name=dataset_name,
                fingerprints=dataset_fingerprints or {}, versions=dataset_versions or {},
                inputs={"target_column": target_column},
                output={"status": "TARGET_SEMANTICS_UNRESOLVED", "observed_classes": [str(v) for v in values]},
                investigation_id=investigation_id, experiment_id=experiment_id,
            ),
            "tool": "autonomous_binary_risk_model",
            "formula": "explicit_positive_class_required",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "validation": "UNVERIFIED",
            "verdict": "INCONCLUSIVE",
            "confidence": None,
        }
    if positive_class not in values:
        return {"status": "TARGET_SEMANTICS_UNRESOLVED", "answer": "Resolved positive class is not present in the observed target values.", "result": {"positive_class": str(positive_class), "observed_classes": [str(v) for v in values]}}
    positive = positive_class
    negative = next(v for v in values if v != positive)
    y = (y_raw == positive).astype(int)

    excluded = {target_column}
    feature_columns: List[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if c in excluded or cl in {"id", "uuid"} or any(tok in cl for tok in ("_id", "_key", "_pk", "customerid")):
            continue
        if time_column and c == time_column:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > 1:
                feature_columns.append(c)
        elif df[c].nunique(dropna=True) <= 20 and df[c].nunique(dropna=True) > 1:
            feature_columns.append(c)
    feature_columns = feature_columns[:30]
    if not feature_columns:
        return {"status": "NO_FEATURES", "answer": "No defensible predictive features remained after identifier/leakage screening."}

    X_df = df[feature_columns].copy()
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss

    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols))
    pre = ColumnTransformer(transformers)

    # Chronological holdout when a usable time field exists; otherwise stratified.
    order = np.arange(len(df))
    if time_column and time_column in df.columns:
        try:
            times = pd.to_datetime(df[time_column], errors="coerce")
            valid_time = times.notna().sum()
            if valid_time >= max(30, int(0.6 * len(df))):
                order = np.argsort(times.fillna(pd.Timestamp.min).values)
        except Exception:
            pass
    split = max(10, int(len(df) * 0.8))
    train_idx, test_idx = order[:split], order[split:]
    if len(test_idx) < 10:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None)

    results = []
    fitted = []
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]
    for name, estimator in candidates:
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        pipe.fit(X_df.iloc[train_idx], y.iloc[train_idx])
        prob = pipe.predict_proba(X_df.iloc[test_idx])[:, 1]
        auc = float(roc_auc_score(y.iloc[test_idx], prob)) if y.iloc[test_idx].nunique() == 2 else None
        brier = float(brier_score_loss(y.iloc[test_idx], prob))
        results.append({"model": name, "roc_auc": round(auc, 4) if auc is not None else None, "brier_score": round(brier, 6)})
        fitted.append((name, pipe, brier, auc if auc is not None else -1.0))

    # Lower Brier is primary; ROC AUC breaks ties.
    best_name, best_pipe, _, best_auc = min(fitted, key=lambda x: (x[2], -x[3]))
    best_pipe.fit(X_df.iloc[train_idx], y.iloc[train_idx])
    active_mask = y == 0
    active_prob = best_pipe.predict_proba(X_df.loc[active_mask])[:, 1]
    active_rows = df.index[active_mask].to_numpy()
    ranking = [
        {"row_index": int(idx), "predicted_event_probability": round(float(prob), 6)}
        for idx, prob in sorted(zip(active_rows, active_prob), key=lambda x: x[1], reverse=True)
    ]
    # Feature names can become one-hot encoded. Keep a compact, model-native summary.
    importance = []
    model = best_pipe.named_steps["model"]
    try:
        names = best_pipe.named_steps["pre"].get_feature_names_out()
        if hasattr(model, "coef_"):
            coefs = np.abs(np.asarray(model.coef_).reshape(-1))
            importance = [{"feature": str(n), "importance": round(float(v), 6)} for n, v in sorted(zip(names, coefs), key=lambda x: x[1], reverse=True)[:10]]
        elif hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            importance = [{"feature": str(n), "importance": round(float(v), 6)} for n, v in sorted(zip(names, vals), key=lambda x: x[1], reverse=True)[:10]]
    except Exception:
        pass

    trace = _trace(
        "PREDICTION", question=question, dataset_name=dataset_name,
        fingerprints=dataset_fingerprints or {}, versions=dataset_versions or {},
        inputs={"target_column": target_column, "time_column": time_column, "features": feature_columns, "train_rows": len(train_idx), "test_rows": len(test_idx)},
        output={"selected_model": best_name, "validation": results, "scored_active_rows": len(active_rows)},
        investigation_id=investigation_id, experiment_id=experiment_id,
    )
    return {
        "status": "COMPLETED",
        "answer": f"Risk model selected: {best_name}. Ranked {len(ranking):,} currently non-event rows by estimated probability of the positive event; this is predictive risk, not a causal claim.",
        "finding": f"Top predicted-risk row: {ranking[0] if ranking else 'none'}. Validation: {results}.",
        "result": {"selected_model": best_name, "validation": results, "ranked_active_rows": ranking[:100], "feature_importance": importance, "positive_class": str(positive), "negative_class": str(negative)},
        "trace": trace,
        "tool": "autonomous_binary_risk_model",
        "formula": "time-aware holdout or stratified holdout + preprocessing + candidate binary classifiers + Brier-primary model selection",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "validation": "UNVERIFIED",
        "verdict": "INCONCLUSIVE",
        "confidence": None,
    }

def run_forecast_analysis(*, df: pd.DataFrame, time_column: str | None, metric_column: str | None, question: str, dataset_name: str, dataset_fingerprints: Dict[str, Any], dataset_versions: Dict[str, Any] | None = None, horizon_periods: int = 1, investigation_id: str | None = None, experiment_id: str | None = None) -> Dict[str, Any]:
    """Execute the forecast contract directly instead of entering the generic loop."""
    started = time.perf_counter()
    if not time_column or not metric_column or time_column not in df.columns or metric_column not in df.columns:
        return {"status":"UNRESOLVED", "answer":"AA-OS could not resolve the time and target columns required for forecasting."}
    result = ForecastingEngine().forecast_metric(df, time_column, metric_column, horizon_periods=max(1, int(horizon_periods)))
    if "error" in result:
        return {"status":"FAILED", "answer":result["error"], "finding":result["error"], "statement":result["error"], "validation":"UNVERIFIED", "tool":"forecasting_engine", "formula":"rolling_origin_backtest", "result":result, "trace":result.get("calculation_trace", {}), "duration_ms":round((time.perf_counter()-started)*1000,2)}
    direction = result.get("trend_direction", "stable")
    prob = float(result.get("directional_probability", 0.5))
    answer = (f"Forecast selected {result.get('selected_model')} using rolling-origin out-of-sample MAE. "
              f"The first forecast direction is {direction} with empirical directional probability {prob:.3f}; "
              f"prediction intervals are derived from backtest residuals.")
    trace = result.get("calculation_trace", {})
    trace["source"] = {"dataset": dataset_name, "dataset_fingerprints": dataset_fingerprints, "dataset_versions": dataset_versions or {}}
    trace["investigation_id"] = investigation_id or ""
    trace["experiment_id"] = experiment_id or ""
    trace.setdefault("trace_id", f"TRACE-SPECIAL-{hashlib.sha256((('FORECAST') + question + dataset_name).encode()).hexdigest()[:16]}")
    trace["canonical_hash"] = hashlib.sha256(json.dumps(trace, sort_keys=True, default=str).encode()).hexdigest()
    return {"status":"COMPLETED", "verdict":"INCONCLUSIVE", "confidence":None, "answer":answer, "finding":answer, "statement":answer, "validation":"UNVERIFIED", "tool":"forecasting_engine", "formula":trace.get("formula","rolling_origin_forecast_selection"), "result":result, "trace":trace, "duration_ms":round((time.perf_counter()-started)*1000,2)}

def run_specialized_analysis(
    *,
    task: str,
    question: str,
    df: pd.DataFrame,
    dataset_name: str,
    dataset_fingerprints: Dict[str, Any],
    dataset_versions: Dict[str, Any] | None = None,
    quality_assessment: Any = None,
    requested_columns: list[str] | None = None,
    investigation_id: str | None = None,
    experiment_id: str | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    task = str(task).upper()

    if task == "DATA_QUALITY":
        result = quality_assessment.to_dict() if hasattr(quality_assessment, "to_dict") else {
            k: v for k, v in getattr(quality_assessment, "__dict__", {}).items()
        }
        answer = (
            f"Data-quality review completed for {dataset_name}: {result.get('overall_quality_score', 'not measured')} / 100. "
            f"Found {result.get('duplicate_rows_count', 0):,} duplicate rows, "
            f"{len(result.get('critical_issues', []))} critical issue(s), and {len(result.get('warnings', []))} warning(s)."
        )
        trace = _trace(task, question=question, dataset_name=dataset_name, fingerprints=dataset_fingerprints, versions=dataset_versions,
                       investigation_id=investigation_id, experiment_id=experiment_id,
                       inputs={"rows": len(df), "columns": list(map(str, df.columns))},
                       output=result)
        return {"answer": answer, "finding": answer,
                "statement": answer, "validation": "UNVERIFIED", "tool": "data_quality_profile",
                "formula": "deterministic_data_quality_profile", "result": result, "trace": trace,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    if task == "SEGMENTATION":
        # Use only features plausibly relevant to the question when they are
        # named; otherwise the engine selects non-ID numeric features.
        result = AutomaticClusteringEngine.analyze(df, requested_columns=requested_columns or [])
        if result.get("status") != "COMPLETED":
            answer = f"AA-OS could not establish a stable segmentation: {result.get('status')}."
        else:
            stable = float(result.get("stability_ari", 0.0))
            sil = float(result.get("silhouette_score", 0.0))
            answer = (
                f"AA-OS found {result['selected_k']} candidate latent segments using {', '.join(result['features'])}. "
                f"Silhouette={sil:.3f}; stability ARI={stable:.3f}. "
                f"Segments are descriptive partitions, not causal customer types."
            )
        trace = _trace(task, question=question, dataset_name=dataset_name, fingerprints=dataset_fingerprints, versions=dataset_versions,
                       investigation_id=investigation_id, experiment_id=experiment_id,
                       inputs={"features": result.get("features", []), "rows": len(df)}, output=result)
        return {"answer": answer,
                "finding": answer, "statement": answer, "validation": "UNVERIFIED", "tool": "automatic_clustering",
                "formula": "StandardScaler + KMeans model selection by silhouette; stability by adjusted Rand index",
                "result": result, "trace": trace, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    if task == "RECONCILIATION":
        result = MathematicalReconciliationEngine.analyze(df)
        if result.get("status") != "COMPLETED":
            answer = f"Mathematical reconciliation could not be completed: {result.get('status')}."
        else:
            best = result["best_identity"]
            bad = int(best.get("discrepancy_rows", 0))
            pct = float(best.get("discrepancy_pct") or 0.0)
            answer = (
                f"Best-supported identity: {best['formula']}. It was checked on {best['rows_checked']:,} rows; "
                f"{bad:,} rows ({pct:.3f}%) violate the stated tolerance. "
                f"Maximum absolute discrepancy: {best.get('max_abs_error')}."
            )
        trace = _trace(task, question=question, dataset_name=dataset_name, fingerprints=dataset_fingerprints, versions=dataset_versions,
                       investigation_id=investigation_id, experiment_id=experiment_id,
                       inputs={"rows": len(df), "tolerance": {"absolute": 0.01, "relative": 1e-6}}, output=result)
        return {"answer": answer,
                "finding": answer, "statement": answer, "validation": "UNVERIFIED",
                "tool": "mathematical_reconciliation", "formula": result.get("best_identity", {}).get("formula", "candidate identity search"),
                "result": result, "trace": trace, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    if task == "PREDICTION":
        raise ValueError("PREDICTION requires semantic target context; call run_predictive_risk_analysis directly.")

    if task == "GOVERNANCE":
        result = GovernanceRiskEngine.assess(df)
        risks = len(result.get("risks", []))
        answer = (
            f"Governance review completed: {len(result.get('pii_columns', []))} PII-like column(s), "
            f"{len(result.get('sensitive_columns', []))} sensitive-attribute-like column(s), and {risks} material risk warning(s). "
            "Legal compliance is not certified from dataset inspection alone."
        )
        trace = _trace(task, question=question, dataset_name=dataset_name, fingerprints=dataset_fingerprints, versions=dataset_versions,
                       investigation_id=investigation_id, experiment_id=experiment_id,
                       inputs={"columns": list(map(str, df.columns))}, output=result)
        return {"answer": answer, "finding": answer,
                "statement": answer, "validation": "UNVERIFIED", "tool": "governance_audit",
                "formula": "deterministic_schema_and_content_governance_scan", "result": result, "trace": trace,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    raise ValueError(f"Unsupported specialized analysis task: {task}")
