"""Deterministic Data Profiling and Data Quality Engine."""
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from packages.schemas.src.dataset import (
    ColumnDistributionSchema,
    ColumnProfileSchema,
    DataQualityBreakdownSchema,
    DatasetProfileSchema,
    RelationshipSchema,
)
from packages.shared.src.enums import ColumnDataType, SemanticType


class DataProfiler:
    """Comprehensive, deterministic dataset profiling and quality scoring engine."""

    def __init__(self, sample_size: Optional[int] = 100000):
        self.sample_size = sample_size

    def profile_dataframe(self, df: pd.DataFrame, dataset_name: str, version: int = 1) -> DatasetProfileSchema:
        """Profile all columns in a DataFrame and generate composite quality metrics."""
        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0:
            return DatasetProfileSchema(
                dataset_name=dataset_name,
                version=version,
                row_count=0,
                column_count=total_cols,
                columns=[],
                potential_primary_keys=[],
                potential_foreign_keys=[],
                duplicate_row_count=0,
                data_quality=DataQualityBreakdownSchema(
                    overall_score=0.0,
                    missing_values_score=0.0,
                    duplicate_rows_score=100.0,
                    type_consistency_score=100.0,
                    outlier_risk_score=100.0,
                    date_validity_score=100.0,
                    details=["Dataset contains 0 rows."],
                ),
                summary_text="Empty dataset.",
            )

        # Duplicate row count
        duplicate_rows = int(df.duplicated().sum())

        column_profiles: List[ColumnProfileSchema] = []
        potential_pks: List[str] = []

        total_null_cells = 0
        total_cells = total_rows * total_cols
        total_outliers = 0

        for col in df.columns:
            col_series = df[col]
            prof = self._profile_column(col_series, col, total_rows)
            column_profiles.append(prof)
            total_null_cells += prof.null_count
            total_outliers += prof.outlier_count

            # PK detection: 0 nulls and 100% unique
            if prof.null_count == 0 and prof.unique_count == total_rows and total_rows > 1:
                potential_pks.append(col)

        # Calculate composite data quality score
        quality_breakdown = self._calculate_data_quality(
            total_rows=total_rows,
            total_cells=total_cells,
            total_null_cells=total_null_cells,
            duplicate_rows=duplicate_rows,
            column_profiles=column_profiles,
        )

        summary_text = (
            f"Dataset '{dataset_name}' (v{version}) has {total_rows:,} rows and {total_cols} columns. "
            f"Overall Data Quality: {quality_breakdown.overall_score:.1f}/100. "
            f"Identified {len(potential_pks)} potential primary key(s)."
        )

        return DatasetProfileSchema(
            dataset_name=dataset_name,
            version=version,
            row_count=total_rows,
            column_count=total_cols,
            columns=column_profiles,
            potential_primary_keys=potential_pks,
            potential_foreign_keys=[],
            duplicate_row_count=duplicate_rows,
            data_quality=quality_breakdown,
            summary_text=summary_text,
        )

    def _profile_column(self, series: pd.Series, col_name: str, total_rows: int) -> ColumnProfileSchema:
        """Profile an individual column deterministically."""
        null_count = int(series.isnull().sum())
        null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        cardinality_ratio = unique_count / total_rows if total_rows > 0 else 0.0

        # Non-null values for analysis
        valid_series = series.dropna()

        data_type, semantic_type = self._infer_types(valid_series, col_name, unique_count, total_rows)

        min_val, max_val = None, None
        mean_val, median_val, std_val, variance_val = None, None, None, None
        mode_val, mode_frequency = None, None
        quantiles: Dict[str, float] = {}
        outlier_count = 0
        dist = ColumnDistributionSchema()

        if len(valid_series) > 0:
            if pd.api.types.is_numeric_dtype(valid_series) and data_type != ColumnDataType.BOOLEAN:
                numeric_series = pd.to_numeric(valid_series, errors="coerce").dropna()
                if len(numeric_series) > 0:
                    min_val = float(numeric_series.min())
                    max_val = float(numeric_series.max())
                    mean_val = float(numeric_series.mean())
                    median_val = float(numeric_series.median())
                    # Preserve the existing sample standard deviation semantics.
                    # The new variance is explicitly population variance for the
                    # observed dataset itself.
                    std_val = float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0
                    variance_val = float(numeric_series.var(ddof=0)) if len(numeric_series) > 0 else 0.0

                    value_counts = numeric_series.value_counts(dropna=True)
                    if not value_counts.empty:
                        max_frequency = int(value_counts.max())
                        if max_frequency > 1:
                            # Deterministic tie-break: choose the smallest observed
                            # value among equally frequent modes. A column where every
                            # value occurs once has no statistical mode.
                            tied_modes = value_counts[value_counts == max_frequency].index.tolist()
                            mode_val = float(min(tied_modes))
                            mode_frequency = max_frequency

                    q25 = float(numeric_series.quantile(0.25))
                    q50 = float(numeric_series.quantile(0.50))
                    q75 = float(numeric_series.quantile(0.75))
                    q95 = float(numeric_series.quantile(0.95))
                    quantiles = {"0.25": q25, "0.50": q50, "0.75": q75, "0.95": q95}

                    # Outlier calculation via IQR
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
                    outlier_count = int(len(outliers))

                    # Histogram bins (10 bins)
                    counts, bin_edges = np.histogram(numeric_series, bins=min(10, len(numeric_series.unique()) or 1))
                    dist.histogram_bins = [float(b) for b in bin_edges]
                    dist.histogram_counts = [int(c) for c in counts]
            elif data_type in (ColumnDataType.DATETIME, ColumnDataType.DATE):
                try:
                    dt_series = pd.to_datetime(valid_series, errors="coerce").dropna()
                    if len(dt_series) > 0:
                        min_val = str(dt_series.min())
                        max_val = str(dt_series.max())
                except Exception:
                    pass
            else:
                # Categorical or String distribution
                str_series = valid_series.astype(str)
                min_val = str(str_series.min()) if len(str_series) > 0 else None
                max_val = str(str_series.max()) if len(str_series) > 0 else None
                top_vals = str_series.value_counts().head(10).to_dict()
                dist.top_categories = {str(k): int(v) for k, v in top_vals.items()}

        return ColumnProfileSchema(
            name=col_name,
            data_type=data_type,
            semantic_type=semantic_type,
            null_count=null_count,
            null_percentage=round(null_percentage, 2),
            unique_count=unique_count,
            cardinality_ratio=round(cardinality_ratio, 4),
            min_value=min_val,
            max_value=max_val,
            mean_value=round(mean_val, 4) if mean_val is not None and not math.isnan(mean_val) else None,
            median_value=round(median_val, 4) if median_val is not None and not math.isnan(median_val) else None,
            std_dev=round(std_val, 4) if std_val is not None and not math.isnan(std_val) else None,
            variance=round(variance_val, 4) if variance_val is not None and not math.isnan(variance_val) else None,
            mode_value=mode_val,
            mode_frequency=mode_frequency,
            quantiles=quantiles,
            outlier_count=outlier_count,
            distribution=dist,
        )

    def _infer_types(
        self, series: pd.Series, col_name: str, unique_count: int, total_rows: int
    ) -> Tuple[ColumnDataType, SemanticType]:
        """Infer raw data type and business semantic type."""
        lower_name = col_name.lower()

        # Check boolean
        if pd.api.types.is_bool_dtype(series) or (unique_count <= 2 and set(series.astype(str).str.lower().unique()).issubset({"true", "false", "0", "1", "yes", "no"})):
            return ColumnDataType.BOOLEAN, SemanticType.DIMENSION

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series) or "date" in lower_name or "time" in lower_name or "timestamp" in lower_name:
            try:
                pd.to_datetime(series.iloc[:min(50, len(series))], errors="raise")
                return ColumnDataType.DATETIME, SemanticType.TIMESTAMP
            except Exception:
                pass

        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            is_int = pd.api.types.is_integer_dtype(series) or (series.dropna() % 1 == 0).all()
            data_type = ColumnDataType.INTEGER if is_int else ColumnDataType.FLOAT

            # Semantic inference
            if lower_name.endswith("_id") or lower_name == "id":
                return data_type, SemanticType.IDENTIFIER
            if any(term in lower_name for term in ["revenue", "sales", "price", "cost", "amount", "spend", "budget", "profit", "discount", "margin", "salary"]):
                return data_type, SemanticType.CURRENCY if "discount" not in lower_name else SemanticType.PERCENTAGE
            if any(term in lower_name for term in ["rate", "pct", "percent", "percentage", "ratio", "score"]):
                return data_type, SemanticType.PERCENTAGE
            if any(term in lower_name for term in ["qty", "quantity", "count", "units", "items", "views", "clicks"]):
                return data_type, SemanticType.METRIC

            return data_type, SemanticType.METRIC

        # String / Categorical / Geographic
        if any(term in lower_name for term in ["country", "state", "city", "region", "postal", "zip", "latitude", "longitude"]):
            return ColumnDataType.STRING, SemanticType.GEOGRAPHIC
        if lower_name.endswith("_id") or lower_name == "id" or "code" in lower_name:
            return ColumnDataType.STRING, SemanticType.IDENTIFIER

        if unique_count <= 50 or (total_rows > 0 and (unique_count / total_rows) < 0.15):
            return ColumnDataType.CATEGORICAL, SemanticType.CATEGORICAL

        return ColumnDataType.STRING, SemanticType.TEXT

    def _calculate_data_quality(
        self,
        total_rows: int,
        total_cells: int,
        total_null_cells: int,
        duplicate_rows: int,
        column_profiles: List[ColumnProfileSchema],
    ) -> DataQualityBreakdownSchema:
        """Calculate composite quality score (0-100) with explainable components."""
        details: List[str] = []

        # 1. Missing values score (30% weight)
        missing_rate = (total_null_cells / total_cells) if total_cells > 0 else 0.0
        missing_score = max(0.0, 100.0 - (missing_rate * 100.0 * 2.0))  # 5% missing drops 10 pts
        if missing_rate > 0:
            details.append(f"Missing values: {missing_rate * 100:.1f}% across all cells.")

        # 2. Duplicate rows score (20% weight)
        dup_rate = (duplicate_rows / total_rows) if total_rows > 0 else 0.0
        dup_score = max(0.0, 100.0 - (dup_rate * 100.0 * 5.0))  # 2% dups drops 10 pts
        if duplicate_rows > 0:
            details.append(f"Duplicates: {duplicate_rows} duplicate rows detected ({dup_rate * 100:.1f}%).")

        # 3. Outlier risk score (20% weight)
        total_outliers = sum(p.outlier_count for p in column_profiles)
        outlier_rate = (total_outliers / total_cells) if total_cells > 0 else 0.0
        outlier_score = max(0.0, 100.0 - (outlier_rate * 100.0 * 10.0))
        if total_outliers > 0:
            details.append(f"Outliers: {total_outliers} statistical outliers detected.")

        # 4. Type consistency score (15% weight)
        type_score = 100.0  # Perfect by default unless irregular parsing errors detected

        # 5. Date validity score (15% weight)
        date_score = 100.0

        overall_score = round(
            (missing_score * 0.30)
            + (dup_score * 0.20)
            + (outlier_score * 0.20)
            + (type_score * 0.15)
            + (date_score * 0.15),
            1,
        )

        return DataQualityBreakdownSchema(
            overall_score=overall_score,
            missing_values_score=round(missing_score, 1),
            duplicate_rows_score=round(dup_score, 1),
            type_consistency_score=round(type_score, 1),
            outlier_risk_score=round(outlier_score, 1),
            date_validity_score=round(date_score, 1),
            details=details,
        )

    def detect_cross_table_relationships(
        self, tables: Dict[str, pd.DataFrame]
    ) -> List[RelationshipSchema]:
        """Detect potential PK-FK relationships across multiple tables."""
        relationships: List[RelationshipSchema] = []
        table_names = list(tables.keys())

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                t1, t2 = table_names[i], table_names[j]
                df1, df2 = tables[t1], tables[t2]

                for col1 in df1.columns:
                    for col2 in df2.columns:
                        # Check name similarity
                        if col1.lower() == col2.lower() or col1.lower() == f"{t2.lower()}_id" or col2.lower() == f"{t1.lower()}_id":
                            s1 = set(df1[col1].dropna().unique())
                            s2 = set(df2[col2].dropna().unique())
                            if not s1 or not s2:
                                continue

                            # If s1 is subset of s2 (df1 foreign key -> df2 primary key)
                            if s1.issubset(s2) and len(s1) > 5:
                                match_ratio = len(s1.intersection(s2)) / len(s1)
                                relationships.append(
                                    RelationshipSchema(
                                        source_table=t1,
                                        source_column=col1,
                                        target_table=t2,
                                        target_column=col2,
                                        relationship_type="many_to_one",
                                        match_ratio=match_ratio,
                                        detection_method="name_and_subset",
                                    )
                                )
                            elif s2.issubset(s1) and len(s2) > 5:
                                match_ratio = len(s2.intersection(s1)) / len(s2)
                                relationships.append(
                                    RelationshipSchema(
                                        source_table=t2,
                                        source_column=col2,
                                        target_table=t1,
                                        target_column=col1,
                                        relationship_type="many_to_one",
                                        match_ratio=match_ratio,
                                        detection_method="name_and_subset",
                                    )
                                )

        return relationships
