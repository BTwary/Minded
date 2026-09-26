"""Visualization Agent for Dynamic Chart Selection and Plotly Specs."""
from typing import Any, Dict, List, Optional
from apps.api.src.ai.providers.base import BaseAIProvider
from packages.schemas.src.dashboard import ChartSpecSchema
from packages.shared.src.enums import ChartType


class VisualizationAgent:
    """Agent that maps analytical findings and query data into clean Plotly visualization specs."""

    def __init__(self, provider: BaseAIProvider):
        self.provider = provider

    def select_and_generate_chart(
        self,
        question: str,
        query_result: Dict[str, Any],
        finding_summary: Optional[str] = None,
    ) -> ChartSpecSchema:
        """Select best visualization and construct interactive Plotly figure specification."""
        columns = query_result.get("columns", [])
        data_rows = query_result.get("data", [])

        if not columns or not data_rows:
            return ChartSpecSchema(
                chart_type=ChartType.TABLE,
                title="Query Results",
                data_rows=data_rows,
                explanation="No data available for graphical plotting.",
            )

        # Infer best chart type deterministically
        col_lower = [c.lower() for c in columns]
        
        # 1. Time-series trend (e.g. Month / Date in first col)
        if any("date" in c or "month" in c or "year" in c or "day" in c for c in col_lower):
            date_col = next(c for c in columns if any(t in c.lower() for t in ["date", "month", "year", "day"]))
            metric_cols = [c for c in columns if c != date_col]
            y_col = metric_cols[0] if metric_cols else date_col

            fig_json = {
                "data": [
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "x": [r.get(date_col) for r in data_rows],
                        "y": [r.get(y_col) for r in data_rows],
                        "name": y_col,
                        "line": {"color": "#3b82f6", "width": 3},
                        "marker": {"size": 6},
                    }
                ],
                "layout": {
                    "title": f"{y_col.replace('_', ' ').title()} Over Time",
                    "xaxis": {"title": date_col.replace('_', ' ').title()},
                    "yaxis": {"title": y_col.replace('_', ' ').title()},
                    "template": "plotly_white",
                    "margin": {"l": 40, "r": 40, "t": 40, "b": 40},
                },
            }

            return ChartSpecSchema(
                chart_type=ChartType.LINE,
                title=f"{y_col.replace('_', ' ').title()} Trend",
                x_column=date_col,
                y_column=y_col,
                plotly_figure_json=fig_json,
                data_rows=data_rows[:50],
                explanation="Line chart highlights temporal fluctuations and trajectory.",
            )

        # 2. Categorical breakdown (e.g. Region / Category in first col)
        elif len(columns) >= 2:
            cat_col = columns[0]
            metric_col = columns[1]

            fig_json = {
                "data": [
                    {
                        "type": "bar",
                        "x": [str(r.get(cat_col)) for r in data_rows],
                        "y": [r.get(metric_col) for r in data_rows],
                        "marker": {"color": "#6366f1"},
                    }
                ],
                "layout": {
                    "title": f"{metric_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                    "xaxis": {"title": cat_col.replace('_', ' ').title()},
                    "yaxis": {"title": metric_col.replace('_', ' ').title()},
                    "template": "plotly_white",
                    "margin": {"l": 40, "r": 40, "t": 40, "b": 40},
                },
            }

            return ChartSpecSchema(
                chart_type=ChartType.BAR,
                title=f"{metric_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                x_column=cat_col,
                y_column=metric_col,
                plotly_figure_json=fig_json,
                data_rows=data_rows[:50],
                explanation="Bar chart provides clear comparative ranking between groups.",
            )

        # Default fallback to Table
        return ChartSpecSchema(
            chart_type=ChartType.TABLE,
            title="Data Table View",
            data_rows=data_rows[:50],
            explanation="Table presentation of query output.",
        )
