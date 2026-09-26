from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MachineLearningEngine:
    """Production-grade ML engine with automated feature preprocessing and validation."""

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        problem_type: Optional[str] = None,  # classification, regression, auto
    ) -> Dict[str, Any]:
        """Train candidate models with cross-validation and evaluate on a holdout test set."""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn is not installed in the environment."}
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in dataframe."}

        # Select feature columns
        if not feature_columns:
            feature_columns = [
                c for c in df.columns
                if c != target_column
                and not c.lower().endswith("_id")
                and c.lower() != "id"
            ]

        if not feature_columns:
            return {"error": "No valid feature columns available."}

        # Filter clean rows
        df_clean = df[[target_column] + feature_columns].dropna(subset=[target_column]).copy()
        if len(df_clean) < 10:
            return {"error": f"Insufficient clean rows for ML training (found {len(df_clean)}, minimum 10 required)."}

        # Infer problem type if auto
        target_series = df_clean[target_column]
        if problem_type in (None, "auto"):
            if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
                problem_type = "regression"
            else:
                problem_type = "classification"

        # Preprocess features
        X_df, feature_names = self._preprocess_features(df_clean[feature_columns])
        y = target_series.values

        if len(feature_names) == 0:
            return {"error": "No valid features could be encoded."}

        # Train/test split (80/20)
        is_stratified = problem_type == "classification" and len(np.unique(y)) > 1
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y if is_stratified else None,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if problem_type == "classification":
            return self._run_classification(X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_column)
        else:
            return self._run_regression(X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_column)

    def _preprocess_features(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values, encode categoricals, and return clean numeric matrix."""
        processed_parts = []
        feature_names = []

        for col in df_features.columns:
            s = df_features[col]
            if pd.api.types.is_numeric_dtype(s):
                median_val = float(s.median()) if len(s.dropna()) > 0 else 0.0
                filled = s.fillna(median_val)
                processed_parts.append(filled.values.reshape(-1, 1))
                feature_names.append(col)
            else:
                # One-hot encode low cardinality
                n_unique = s.nunique(dropna=True)
                if 1 < n_unique <= 10:
                    dummies = pd.get_dummies(s.astype(str), prefix=col, drop_first=True)
                    processed_parts.append(dummies.values)
                    feature_names.extend(dummies.columns.tolist())

        if not processed_parts:
            return pd.DataFrame(), []

        X_mat = np.hstack(processed_parts)
        return pd.DataFrame(X_mat, columns=feature_names), feature_names

    def _run_classification(
        self, X_train, X_test, y_train, y_test, feature_names: List[str], target_column: str
    ) -> Dict[str, Any]:
        """Train and compare classification models."""
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
        }

        results = []
        best_model = None
        best_name = ""
        best_f1 = -1.0
        is_binary = len(np.unique(y_train)) == 2

        for name, model in models.items():
            cv = StratifiedKFold(n_splits=min(5, max(2, len(y_train) // 4)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="weighted"))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

            roc_auc = None
            if is_binary and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
                except Exception:
                    pass

            res = {
                "model_name": name,
                "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
                "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
                "test_accuracy": round(acc, 4),
                "test_f1_score": round(f1, 4),
                "test_precision": round(prec, 4),
                "test_recall": round(rec, 4),
                "test_roc_auc": roc_auc,
            }
            results.append(res)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name

        # Feature importances
        feature_importance = self._extract_feature_importance(best_model, feature_names)

        return {
            "problem_type": "classification",
            "target_column": target_column,
            "sample_size": len(X_train) + len(X_test),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "candidate_models_evaluated": results,
            "selected_best_model": best_name,
            "feature_importance": feature_importance,
            "evaluation_verdict": (
                f"Model '{best_name}' selected with Test Accuracy: {results[0]['test_accuracy']:.2%} and F1: {best_f1:.4f}."
            ),
        }

    def _run_regression(
        self, X_train, X_test, y_train, y_test, feature_names: List[str], target_column: str
    ) -> Dict[str, Any]:
        """Train and compare regression models."""
        models = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
        }

        results = []
        best_model = None
        best_name = ""
        best_r2 = -float("inf")

        for name, model in models.items():
            cv = KFold(n_splits=min(5, max(2, len(y_train) // 4)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            res = {
                "model_name": name,
                "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
                "test_r2_score": round(r2, 4),
                "test_mae": round(mae, 4),
                "test_rmse": round(rmse, 4),
            }
            results.append(res)

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

        feature_importance = self._extract_feature_importance(best_model, feature_names)

        return {
            "problem_type": "regression",
            "target_column": target_column,
            "sample_size": len(X_train) + len(X_test),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "candidate_models_evaluated": results,
            "selected_best_model": best_name,
            "feature_importance": feature_importance,
            "evaluation_verdict": (
                f"Model '{best_name}' selected with Test R2: {best_r2:.4f} and RMSE: {results[0]['test_rmse']}."
            ),
        }

    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extract sorted feature importances or coefficients from model."""
        importances = []
        if hasattr(model, "feature_importances_"):
            for name, score in zip(feature_names, model.feature_importances_):
                importances.append({"feature": name, "importance": round(float(score), 4)})
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs)
            total = np.sum(coefs) or 1.0
            for name, score in zip(feature_names, coefs):
                importances.append({"feature": name, "importance": round(float(score / total), 4)})

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:10]
