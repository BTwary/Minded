"""Secure Universal Tool Registry for Autonomous AI Data Analyst."""
import math
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from packages.analytics_core.src.anomaly.engine import AnomalyDetectionEngine
from packages.analytics_core.src.forecasting.engine import ForecastingEngine
from packages.analytics_core.src.ml.engine import MachineLearningEngine
from packages.analytics_core.src.profiling.profiler import DataProfiler
from packages.analytics_core.src.scenarios.engine import ScenarioSimulator
from packages.analytics_core.src.sql.engine import DuckDBSQLEngine
from packages.analytics_core.src.statistics.engine import StatisticalEngine
from packages.analytics_core.src.validation.validator import IndependentValidator


class ToolRegistry:
    """Central managed execution registry for deterministic analytical tools."""

    def __init__(
        self,
        sql_engine: Optional[DuckDBSQLEngine] = None,
        datasets: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.sql_engine = sql_engine or DuckDBSQLEngine()
        self.datasets = datasets or {}
        self.profiler = DataProfiler()
        self.stats_engine = StatisticalEngine()
        self.forecast_engine = ForecastingEngine()
        self.ml_engine = MachineLearningEngine()
        self.anomaly_engine = AnomalyDetectionEngine()
        self.scenario_engine = ScenarioSimulator()
        self.validator = IndependentValidator()

        # Register datasets in DuckDB
        for name, df in self.datasets.items():
            self.sql_engine.register_dataframe(name, df)

    def register_dataset(self, name: str, df: pd.DataFrame) -> None:
        """Register a dataset DataFrame in memory and in DuckDB."""
        self.datasets[name] = df
        self.sql_engine.register_dataframe(name, df)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform execution protocol for any action proposed by the Autonomous Planner."""
        tool_clean = str(tool_name).strip().lower()

        if tool_clean in ["run_sql", "query_sql", "sql"]:
            return self.run_sql(arguments.get("sql", ""))

        elif tool_clean in ["profile_dataset", "profile_data", "profile"]:
            dataset_name = arguments.get("dataset_name") or (list(self.datasets.keys())[0] if self.datasets else "sales")
            return self.profile_dataset(dataset_name)

        elif tool_clean in ["derive_metric", "create_calculated_column"]:
            return self.derive_metric(
                table_name=arguments.get("table_name") or (list(self.datasets.keys())[0] if self.datasets else "sales"),
                new_metric_name=arguments.get("new_metric_name", "derived_metric"),
                formula_expression=arguments.get("formula_expression", ""),
            )

        elif tool_clean in ["calculate_statistics", "stats", "correlation", "t_test", "regression"]:
            test_type = arguments.get("test_type", tool_clean)
            return self.calculate_statistics(test_type, arguments)

        elif tool_clean in ["detect_anomalies", "anomaly_detection"]:
            method = arguments.get("method", "outliers")
            return self.detect_anomalies(method, arguments)

        elif tool_clean in ["run_forecast", "forecast", "time_series_forecast"]:
            return self.run_forecast(arguments)

        elif tool_clean in ["train_model", "train_ml", "machine_learning"]:
            return self.train_model(arguments)

        elif tool_clean in ["decompose_variance", "variance_decomposition"]:
            return self.decompose_variance(
                table_name=arguments.get("table_name") or (list(self.datasets.keys())[0] if self.datasets else "sales"),
                metric_column=arguments.get("metric_column", "revenue"),
                dimension_column=arguments.get("dimension_column", "region"),
                time_column=arguments.get("time_column", "order_date"),
                period_a=arguments.get("period_a", ""),
                period_b=arguments.get("period_b", ""),
            )

        elif tool_clean == "rfm_segmentation":
            return self.run_rfm_segmentation(
                table_name=arguments.get("table_name") or (list(self.datasets.keys())[0] if self.datasets else "sales"),
                customer_column=arguments.get("customer_column", "customer_id"),
                time_column=arguments.get("time_column", "order_date"),
                metric_column=arguments.get("metric_column", "revenue"),
            )


        elif tool_clean == "customer_churn_analysis":
            return {
                "error": (
                    "Standalone customer_churn_analysis is not an admissible analytical operation. "
                    "Churn analysis requires an explicit observed churn outcome/event in the semantic contract "
                    "and must run through the canonical investigation controller."
                )
            }

        elif tool_clean in ["validate_result", "validate_claim"]:
            return self.validate_result(arguments)

        elif tool_clean in ["simulate_scenario", "scenario_simulation"]:
            return self.simulate_scenario(arguments)

        return {"error": f"Unknown tool: '{tool_name}'"}

    def derive_metric(self, table_name: str, new_metric_name: str, formula_expression: str) -> Dict[str, Any]:
        """Dynamically compute a derived metric/column and update in-memory DuckDB relation."""
        df = self.datasets.get(table_name)
        if df is None:
            return {"error": f"Table '{table_name}' not found."}

        try:
            sql = f"SELECT *, ({formula_expression}) AS {new_metric_name} FROM {table_name}"
            res = self.sql_engine.execute_query(sql, max_rows=1000000)
            rows = res.get("data", [])
            new_df = pd.DataFrame(rows)
            self.register_dataset(table_name, new_df)
            
            series = new_df[new_metric_name].dropna()
            return {
                "status": "success",
                "derived_metric": new_metric_name,
                "formula": formula_expression,
                "row_count": len(new_df),
                "mean": round(float(series.mean()), 4) if len(series) > 0 else 0.0,
                "min": round(float(series.min()), 4) if len(series) > 0 else 0.0,
                "max": round(float(series.max()), 4) if len(series) > 0 else 0.0,
            }
        except Exception as e:
            return {"error": f"Failed to derive metric '{new_metric_name}': {str(e)}"}

    def decompose_variance(
        self, table_name: str, metric_column: str, dimension_column: str, time_column: str, period_a: str, period_b: str
    ) -> Dict[str, Any]:
        """Decompose variance of a metric across dimensions between two periods in DuckDB."""
        try:
            sql = f"""
            SELECT 
                {dimension_column},
                ROUND(SUM(CASE WHEN SUBSTRING(CAST({time_column} AS VARCHAR), 1, 7) = '{period_a}' THEN {metric_column} ELSE 0 END), 2) as period_a_val,
                ROUND(SUM(CASE WHEN SUBSTRING(CAST({time_column} AS VARCHAR), 1, 7) = '{period_b}' THEN {metric_column} ELSE 0 END), 2) as period_b_val,
                ROUND(SUM(CASE WHEN SUBSTRING(CAST({time_column} AS VARCHAR), 1, 7) = '{period_b}' THEN {metric_column} ELSE 0 END) - 
                      SUM(CASE WHEN SUBSTRING(CAST({time_column} AS VARCHAR), 1, 7) = '{period_a}' THEN {metric_column} ELSE 0 END), 2) as net_change
            FROM {table_name}
            GROUP BY 1
            ORDER BY net_change ASC
            """
            res = self.sql_engine.execute_query(sql)
            rows = res.get("data", [])
            
            total_loss = sum(abs(r["net_change"]) for r in rows if r.get("net_change", 0) < 0)
            contributors = []
            for r in rows:
                chg = r.get("net_change", 0)
                contrib_pct = round((abs(chg) / total_loss * 100.0), 1) if (total_loss > 0 and chg < 0) else 0.0
                contributors.append({
                    "dimension_value": str(r[dimension_column]),
                    "period_a": r["period_a_val"],
                    "period_b": r["period_b_val"],
                    "net_change": chg,
                    "contribution_pct": contrib_pct,
                })

            worst = contributors[0] if contributors else None
            return {
                "status": "success",
                "table": table_name,
                "metric": metric_column,
                "dimension": dimension_column,
                "period_a": period_a,
                "period_b": period_b,
                "total_negative_variance": round(total_loss, 2),
                "primary_contributor": worst,
                "breakdown": contributors,
                "sql": sql,
            }
        except Exception as e:
            return {"error": f"Variance decomposition failed: {str(e)}"}

    def run_rfm_segmentation(self, table_name: str, customer_column: str, time_column: str, metric_column: str) -> Dict[str, Any]:
        """Perform genuine RFM customer segmentation and churn vulnerability analysis."""
        try:
            sql = f"""
            SELECT 
                {customer_column},
                COUNT(*) as frequency,
                ROUND(SUM({metric_column}), 2) as monetary,
                MAX({time_column}) as last_order_date
            FROM {table_name}
            GROUP BY 1
            """
            res = self.sql_engine.execute_query(sql)
            rows = res.get("data", [])
            total_customers = len(rows)

            if total_customers == 0:
                return {"error": "No customer records found."}

            freqs = [r.get("frequency", 1) for r in rows]
            med_freq = float(np.median(freqs)) if freqs else 1.0
            below_median = sum(1 for f in freqs if f < med_freq)
            single_orders = sum(1 for f in freqs if f == 1)

            return {
                "status": "success",
                "total_unique_customers": total_customers,
                "median_order_frequency": round(med_freq, 1),
                "below_median_frequency_count": below_median,
                "below_median_pct": round((below_median / total_customers * 100.0), 1),
                "single_order_count": single_orders,
                "single_order_pct": round((single_orders / total_customers * 100.0), 1),
                "sample_customers": rows[:5],
                "sql": sql,
            }
        except Exception as e:
            return {"error": f"RFM segmentation failed: {str(e)}"}

    def run_sql(self, sql: str) -> Dict[str, Any]:
        """Execute a read-only DuckDB SQL query deterministically."""
        return self.sql_engine.execute_query(sql)

    def profile_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Profile a registered dataset."""
        df = self.datasets.get(dataset_name)
        if df is None:
            return {"error": f"Dataset '{dataset_name}' not found."}
        profile = self.profiler.profile_dataframe(df, dataset_name)
        return profile.model_dump()

    def calculate_statistics(self, test_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run deterministic statistical tests."""
        dataset_name = params.get("dataset_name")
        df = self.datasets.get(dataset_name) if dataset_name else list(self.datasets.values())[0] if self.datasets else None

        if df is None:
            return {"error": "No valid dataset available for statistics."}

        if test_type in ["descriptive", "summary"]:
            col = params.get("column")
            if col not in df.columns:
                return {"error": f"Column '{col}' not found."}
            return self.stats_engine.descriptive_summary(df[col])

        elif test_type in ["t_test", "two_sample_t_test"]:
            col = params.get("metric_column")
            split_col = params.get("group_column")
            group_a = params.get("group_a")
            group_b = params.get("group_b")

            if col not in df.columns or split_col not in df.columns:
                return {"error": "Specified columns not found for t-test."}

            s_a = df[df[split_col] == group_a][col].dropna()
            s_b = df[df[split_col] == group_b][col].dropna()
            return self.stats_engine.two_sample_t_test(s_a, s_b)

        elif test_type in ["correlation", "pearson"]:
            cols = params.get("columns", [])
            if not cols:
                cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:5]
            method = params.get("method", "pearson")
            return self.stats_engine.correlation_analysis(df, cols, method=method)

        elif test_type in ["regression", "ols"]:
            x_cols = params.get("x_columns", [])
            y_col = params.get("y_column")
            return self.stats_engine.infer_regression(
                df, x_cols, y_col,
                alpha=float(params.get("alpha", 0.05)),
                cluster_column=params.get("cluster_column"),
                time_column=params.get("time_column"),
            )

        return {"error": f"Unsupported statistical test type: '{test_type}'."}

    def detect_anomalies(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run outlier or time-series anomaly detection."""
        dataset_name = params.get("dataset_name")
        df = self.datasets.get(dataset_name) if dataset_name else list(self.datasets.values())[0] if self.datasets else None

        if df is None:
            return {"error": "No valid dataset available for anomaly detection."}

        if method in ["outliers", "iqr"]:
            col = params.get("column")
            threshold = params.get("threshold", 1.5)
            return self.anomaly_engine.detect_numerical_outliers(df, col, method="iqr", threshold=threshold)

        elif method in ["time_series", "rolling"]:
            date_col = params.get("date_column")
            metric_col = params.get("metric_column")
            window = params.get("window", 5)
            return self.anomaly_engine.detect_time_series_anomalies(df, date_col, metric_col, window=window)

        return {"error": f"Unsupported anomaly detection method: '{method}'."}

    def run_forecast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run time-series forecasting."""
        dataset_name = params.get("dataset_name")
        df = self.datasets.get(dataset_name) if dataset_name else list(self.datasets.values())[0] if self.datasets else None

        if df is None:
            return {"error": "No valid dataset available for forecasting."}

        date_col = params.get("date_column")
        metric_col = params.get("metric_column")
        horizon = params.get("horizon_periods", 3)

        return self.forecast_engine.forecast_metric(df, date_col, metric_col, horizon_periods=horizon)

    def train_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models."""
        dataset_name = params.get("dataset_name")
        df = self.datasets.get(dataset_name) if dataset_name else list(self.datasets.values())[0] if self.datasets else None

        if df is None:
            return {"error": "No valid dataset available for ML."}

        target_col = params.get("target_column")
        feature_cols = params.get("feature_columns")
        problem_type = params.get("problem_type", "auto")

        return self.ml_engine.train_and_evaluate(df, target_col, feature_cols, problem_type=problem_type)

    def validate_result(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Independently validate a numerical calculation."""
        claim_name = params.get("claim_name", "Validation Check")
        expected_val = float(params.get("expected_value", 0.0))
        actual_val = float(params.get("actual_value", 0.0))
        tolerance = float(params.get("tolerance", 0.01))

        res = self.validator.validate_numerical_claim(claim_name, expected_val, actual_val, relative_tolerance=tolerance)
        return res.model_dump()

    def simulate_scenario(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run deterministic counterfactual scenario."""
        dataset_name = params.get("dataset_name")
        df = self.datasets.get(dataset_name) if dataset_name else list(self.datasets.values())[0] if self.datasets else None
        if df is None:
            return {"error": "No dataset available for scenario simulation."}

        sc_type = params.get("type", "discount_cap")
        if sc_type == "discount_cap":
            cap = float(params.get("cap", 0.10))
            return self.scenario_engine.simulate_discount_cap(df, target_discount_cap=cap)
        elif sc_type == "volume_recovery":
            dim = params.get("dimension_column", "region")
            val = params.get("lagging_value", "Region B")
            growth = float(params.get("growth_pct", 15.0))
            return self.scenario_engine.simulate_volume_recovery(df, dimension_col=dim, lagging_value=val, benchmark_target_growth_pct=growth)
        return {"error": f"Unknown scenario type: '{sc_type}'"}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return descriptions of all registered tools for AI agent planning."""
        return [
            {"name": "query_sql", "description": "Execute read-only DuckDB SQL on analytical datasets", "params": ["sql"]},
            {"name": "profile_dataset", "description": "Profile dataset nulls, distributions, and cardinality", "params": ["dataset_name"]},
            {"name": "derive_metric", "description": "Dynamically create calculated metrics (e.g. margin, profit, conversion)", "params": ["table_name", "new_metric_name", "formula_expression"]},
            {"name": "decompose_variance", "description": "Decompose metric changes across dimensions between two periods", "params": ["table_name", "metric_column", "dimension_column", "time_column", "period_a", "period_b"]},
            {"name": "calculate_statistics", "description": "Compute Pearson correlation, t-tests, ANOVA, regression, descriptive stats", "params": ["test_type", "columns", "metric_column", "group_column"]},
            {"name": "detect_anomalies", "description": "Detect numerical IQR outliers or rolling time-series anomalies", "params": ["method", "column", "date_column", "metric_column"]},
            {"name": "run_forecast", "description": "Fit Holt's linear exponential smoothing model with prediction intervals", "params": ["dataset_name", "date_column", "metric_column", "horizon_periods"]},
            {"name": "train_model", "description": "Train and evaluate ML classification/regression models with feature importance", "params": ["target_column", "feature_columns", "problem_type"]},
            {"name": "rfm_segmentation", "description": "Compute customer frequency, monetary, and retention cohorts", "params": ["table_name", "customer_column", "time_column", "metric_column"]},
            {"name": "simulate_scenario", "description": "Run deterministic counterfactual business scenario simulations", "params": ["type", "cap", "dimension_column", "lagging_value", "growth_pct"]},
            {"name": "validate_result", "description": "Independently validate numerical calculation against tolerance", "params": ["claim_name", "expected_value", "actual_value", "tolerance"]},
        ]
