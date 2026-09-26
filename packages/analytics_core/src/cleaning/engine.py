"""Non-destructive Data Cleaning and Transformation Engine."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TransformationLog:
    def __init__(
        self,
        transformation_type: str,
        target_column: Optional[str],
        affected_rows: int,
        parameters: Dict[str, Any],
        transformation_reliability: float = 0.95,
        description: Optional[str] = None,
    ):
        self.transformation_type = transformation_type
        self.target_column = target_column
        self.affected_rows = affected_rows
        self.parameters = parameters
        self.transformation_reliability = transformation_reliability
        self.description = description
        self.applied_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformation_type": self.transformation_type,
            "target_column": self.target_column,
            "affected_rows": self.affected_rows,
            "parameters": self.parameters,
            "transformation_reliability": self.transformation_reliability,
            "description": self.description,
            "applied_at": self.applied_at,
        }


class DataCleaningEngine:
    """Non-destructive data cleaning engine that creates audited dataset versions."""

    def __init__(self):
        self.transformation_history: List[TransformationLog] = []

    def clean_dataset(
        self,
        df: pd.DataFrame,
        cleaning_recipe: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply a series of non-destructive cleaning transformations."""
        # Always operate on a copy to never mutate original
        cleaned_df = df.copy()
        applied_logs: List[TransformationLog] = []

        for step in cleaning_recipe:
            action = step.get("action")
            column = step.get("column")
            params = step.get("params", {})

            if action == "standardize_dates" and column in cleaned_df.columns:
                cleaned_df, log = self._standardize_dates(cleaned_df, column, params)
                applied_logs.append(log)
            elif action == "normalize_categories" and column in cleaned_df.columns:
                cleaned_df, log = self._normalize_categories(cleaned_df, column, params)
                applied_logs.append(log)
            elif action == "handle_missing_values" and column in cleaned_df.columns:
                cleaned_df, log = self._handle_missing_values(cleaned_df, column, params)
                applied_logs.append(log)
            elif action == "deduplicate_rows":
                cleaned_df, log = self._deduplicate_rows(cleaned_df, params)
                applied_logs.append(log)
            elif action == "filter_outliers" and column in cleaned_df.columns:
                cleaned_df, log = self._filter_outliers(cleaned_df, column, params)
                applied_logs.append(log)
            elif action == "cast_type" and column in cleaned_df.columns:
                cleaned_df, log = self._cast_type(cleaned_df, column, params)
                applied_logs.append(log)

        log_dicts = [l.to_dict() for l in applied_logs]
        self.transformation_history.extend(applied_logs)
        return cleaned_df, log_dicts

    def _standardize_dates(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, TransformationLog]:
        """Standardize date strings into ISO format."""
        target_format = params.get("format", "%Y-%m-%d")
        original_series = df[column].copy()
        
        parsed = pd.to_datetime(df[column], errors="coerce")
        affected = int((original_series.notnull() & parsed.notnull()).sum())
        df[column] = parsed.dt.strftime(target_format)

        log = TransformationLog(
            transformation_type="standardize_dates",
            target_column=column,
            affected_rows=affected,
            parameters={"target_format": target_format},
            transformation_reliability=0.99,
            description=f"Standardized date column '{column}' to ISO format.",
        )
        return df, log

    def _normalize_categories(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, TransformationLog]:
        """Normalize casing, strip whitespace, and apply category mappings."""
        mapping = params.get("mapping", {})
        casing = params.get("casing", "title")  # upper, lower, title, strip_only

        series = df[column].astype(str).str.strip()
        if casing == "upper":
            series = series.str.upper()
        elif casing == "lower":
            series = series.str.lower()
        elif casing == "title":
            series = series.str.title()

        if mapping:
            series = series.replace(mapping)

        affected = int((df[column] != series).sum())
        df[column] = series

        log = TransformationLog(
            transformation_type="normalize_categories",
            target_column=column,
            affected_rows=affected,
            parameters={"casing": casing, "mapping": mapping},
            transformation_reliability=0.98,
            description=f"Normalized categorical values in '{column}' ({affected} values modified).",
        )
        return df, log

    def _handle_missing_values(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, TransformationLog]:
        """Fill or drop missing values with a deterministic strategy."""
        strategy = params.get("strategy", "mean")  # mean, median, mode, constant, drop
        fill_value = params.get("fill_value")
        missing_count = int(df[column].isnull().sum())

        if missing_count == 0:
            return df, TransformationLog(
                transformation_type="handle_missing_values",
                target_column=column,
                affected_rows=0,
                parameters={"strategy": strategy},
                transformation_reliability=1.0,
                description=f"No missing values in '{column}'.",
            )

        if strategy == "drop":
            df = df.dropna(subset=[column])
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
            calc_mean = float(df[column].mean())
            df[column] = df[column].fillna(calc_mean)
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
            calc_median = float(df[column].median())
            df[column] = df[column].fillna(calc_median)
        elif strategy == "mode":
            mode_val = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None
            if mode_val is not None:
                df[column] = df[column].fillna(mode_val)
        elif strategy == "constant" and fill_value is not None:
            df[column] = df[column].fillna(fill_value)

        log = TransformationLog(
            transformation_type="handle_missing_values",
            target_column=column,
            affected_rows=missing_count,
            parameters={"strategy": strategy, "fill_value": fill_value},
            transformation_reliability=0.95,
            description=f"Imputed {missing_count} missing values in '{column}' using '{strategy}' strategy.",
        )
        return df, log

    def _deduplicate_rows(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, TransformationLog]:
        """Remove duplicate rows."""
        subset = params.get("subset")
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset)
        removed_count = initial_len - len(df)

        log = TransformationLog(
            transformation_type="deduplicate_rows",
            target_column=None,
            affected_rows=removed_count,
            parameters={"subset": subset},
            transformation_reliability=1.0,
            description=f"Removed {removed_count} duplicate rows.",
        )
        return df, log

    def _filter_outliers(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, TransformationLog]:
        """Clamp or remove outliers based on IQR."""
        action = params.get("action", "clamp")  # clamp, drop
        q25 = df[column].quantile(0.25)
        q75 = df[column].quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        outliers_mask = (df[column] < lower) | (df[column] > upper)
        affected = int(outliers_mask.sum())

        if action == "clamp":
            df[column] = df[column].clip(lower=lower, upper=upper)
        elif action == "drop":
            df = df[~outliers_mask]

        log = TransformationLog(
            transformation_type="filter_outliers",
            target_column=column,
            affected_rows=affected,
            parameters={"method": "iqr", "action": action, "bounds": [float(lower), float(upper)]},
            transformation_reliability=0.92,
            description=f"Handled {affected} outliers in '{column}' via IQR {action}.",
        )
        return df, log

    def _cast_type(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, TransformationLog]:
        """Safely cast column to target type."""
        target_type = params.get("target_type", "float")
        initial_series = df[column].copy()

        rows_failed_conversion = 0
        if target_type == "int":
            numeric = pd.to_numeric(df[column], errors="coerce")
            rows_failed_conversion = int((df[column].notna() & numeric.isna()).sum())
            # Preserve failed conversions as missing values. Never manufacture a
            # numeric zero from an unparseable source value unless the recipe
            # explicitly supplies such a business rule.
            df[column] = numeric.astype("Int64")
        elif target_type == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif target_type == "str":
            df[column] = df[column].astype(str)

        affected = int((initial_series != df[column]).sum())
        log = TransformationLog(
            transformation_type="cast_type",
            target_column=column,
            affected_rows=affected,
            parameters={
                "target_type": target_type,
                "rows_failed_conversion": rows_failed_conversion,
            },
            transformation_reliability=0.97,
            description=f"Casted '{column}' to {target_type}; {rows_failed_conversion} non-null values failed conversion."
            if target_type == "int" else f"Casted '{column}' to {target_type}.",
        )
        return df, log
