"""Automated Data Discovery Engine for Schema, Grain, Keys, Metrics, and Worthy Inquiries."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DiscoveryEngine:
    """Automated Data Understanding and Discovery Engine."""

    def discover_dataset(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive automated understanding of a dataset without prior instructions."""
        row_count = len(df)
        col_count = len(df.columns)

        if row_count == 0:
            return {
                "table_name": table_name,
                "row_count": 0,
                "grain": "empty",
                "metrics": [],
                "dimensions": [],
                "time_columns": [],
                "candidate_keys": [],
                "worthy_inquiries": [],
            }

        # 1. Classify Column Semantics & Cardinality
        metrics: List[Dict[str, Any]] = []
        dimensions: List[Dict[str, Any]] = []
        time_columns: List[Dict[str, Any]] = []
        candidate_keys: List[str] = []

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            unique_count = series.nunique()
            col_lower = str(col).lower()

            # Check if Primary Key Candidate
            if unique_count == row_count and ("id" in col_lower or "key" in col_lower or "code" in col_lower):
                candidate_keys.append(col)

            # Check for Date / Time
            is_date = False
            if pd.api.types.is_datetime64_any_dtype(series) or any(
                term in col_lower for term in ["date", "time", "timestamp", "month", "year", "day", "created_at", "updated_at"]
            ):
                try:
                    # Attempt parse sample
                    pd.to_datetime(series.head(20))
                    is_date = True
                    min_val = str(series.min())
                    max_val = str(series.max())
                    time_columns.append({"column": col, "min": min_val, "max": max_val, "distinct_periods": unique_count})
                except Exception:
                    pass

            if is_date:
                continue

            # Check for Numeric Metrics
            if pd.api.types.is_numeric_dtype(series) and not (unique_count == row_count and "id" in col_lower):
                if unique_count > 5:
                    mu = float(series.mean())
                    sd = float(series.std()) if len(series) > 1 else 0.0
                    metrics.append({
                        "column": col,
                        "mean": round(mu, 2),
                        "std": round(sd, 2),
                        "min": round(float(series.min()), 2),
                        "max": round(float(series.max()), 2),
                        "sum": round(float(series.sum()), 2),
                        "is_currency_or_amount": any(k in col_lower for k in ["revenue", "price", "amount", "cost", "sales", "spend", "margin", "profit"]),
                    })
                else:
                    dimensions.append({"column": col, "cardinality": unique_count, "top_values": series.value_counts().head(5).to_dict()})

            # Categorical / String Dimensions
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
                if unique_count < row_count:
                    dimensions.append({
                        "column": col,
                        "cardinality": unique_count,
                        "top_values": series.value_counts().head(5).to_dict(),
                        "is_geographic": any(k in col_lower for k in ["region", "country", "state", "city", "territory"]),
                        "is_segment": any(k in col_lower for k in ["segment", "category", "channel", "tier", "type", "cohort"]),
                    })

        # 2. Determine Dataset Grain
        # Never let dataframe column order silently decide the semantic grain.
        # A single candidate is safe; multiple equally plausible candidates are
        # intentionally left unresolved for the semantic layer.
        grain = "transaction_level"
        if len(candidate_keys) == 1:
            grain = f"record_keyed_by_{candidate_keys[0]}"
        elif len(candidate_keys) > 1:
            grain = "record_keyed_by_ambiguous_candidate_key"
        elif len(time_columns) == 1 and len(dimensions) > 0:
            grain = f"periodic_time_series_by_{time_columns[0]['column']}"
        elif len(time_columns) > 1 and len(dimensions) > 0:
            grain = "periodic_time_series_by_ambiguous_time_column"
        elif len(dimensions) == 1:
            grain = f"segmented_by_{dimensions[0]['column']}"
        elif len(dimensions) > 1:
            grain = "segmented_by_ambiguous_dimension"

        # 3. Generate Proactive "Worthy Business Inquiries"
        worthy_inquiries = self._generate_worthy_inquiries(table_name, metrics, dimensions, time_columns)

        return {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": col_count,
            "grain": grain,
            "candidate_keys": candidate_keys,
            "primary_metrics": sorted(metrics, key=lambda m: m.get("is_currency_or_amount", False), reverse=True),
            "dimensions": dimensions,
            "time_columns": time_columns,
            "worthy_inquiries": worthy_inquiries,
        }

    def _generate_worthy_inquiries(
        self,
        table_name: str,
        metrics: List[Dict[str, Any]],
        dimensions: List[Dict[str, Any]],
        time_columns: List[Dict[str, Any]],
    ) -> List[str]:
        """Proactively infer what an executive or analyst should investigate in this data."""
        inquiries = []

        ranked_metrics = sorted(
            metrics,
            key=lambda m: (
                bool(m.get("is_currency_or_amount")),
                float(m.get("std", 0.0) or 0.0) > 0,
                float(m.get("max", 0.0) or 0.0),
                str(m.get("column")),
            ),
            reverse=True,
        )
        top_metric = ranked_metrics[0]["column"] if ranked_metrics else "volume"
        ranked_dims = sorted(dimensions, key=lambda d: (bool(d.get("is_segment")), bool(d.get("is_geographic")), -int(d.get("cardinality", 10**9)), str(d.get("column"))))
        top_dim = ranked_dims[0]["column"] if ranked_dims else None
        geo_dim = next((d["column"] for d in dimensions if d.get("is_geographic")), None)
        seg_dim = next((d["column"] for d in dimensions if d.get("is_segment")), None)
        # A proactive time-series inquiry is only emitted when the time role is
        # uniquely established. Do not infer a time axis from dataframe order.
        time_col = time_columns[0]["column"] if len(time_columns) == 1 else None

        if time_col and top_metric:
            inquiries.append(f"What is the historical trajectory and monthly growth rate of '{top_metric}' over time?")
            inquiries.append(f"Forecast next period demand and trajectory for '{top_metric}' with 95% confidence intervals.")

        if geo_dim and top_metric:
            inquiries.append(f"Which territory in '{geo_dim}' is driving the highest concentration of '{top_metric}'?")

        if seg_dim and top_metric:
            inquiries.append(f"How does '{top_metric}' vary across different '{seg_dim}' cohorts, and is the variance statistically significant?")

        if len(ranked_metrics) >= 2:
            m1 = ranked_metrics[0]["column"]
            m2 = ranked_metrics[1]["column"]
            inquiries.append(f"What is the statistical correlation and sensitivity between '{m1}' and '{m2}'?")

        if not inquiries:
            inquiries.append(f"Perform comprehensive exploratory data analysis (EDA) and distribution profiling on '{table_name}'.")

        return inquiries[:5]
