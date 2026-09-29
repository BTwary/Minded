"""AutonomousDriftDaemon: Proactive background monitoring daemon enqueuing investigation jobs on detected drift."""
from typing import Any, List, Optional
import numpy as np
from packages.analytics_core.src.execution.queue import DatabaseQueueProvider
from packages.analytics_core.src.monitoring.drift_engine import DriftMonitorEngine, DriftAlert


class AutonomousDriftDaemon:
    """Proactive background monitor detecting statistical distribution shifts and triggering autonomous investigations."""

    def __init__(self, queue_provider: DatabaseQueueProvider):
        self.queue = queue_provider

    def fetch_baseline_sample(self, table: Any) -> Any:
        """Fetches historical baseline sample for registered table."""
        return None

    def fetch_recent_window(self, table: Any) -> Any:
        """Fetches recent observation window (e.g. last 24-48h)."""
        return None

    def run_hourly_cycle(self, registered_tables: List[Any]):
        """Executes one cycle across registered tables and triggers autonomous investigations on detected drift."""
        for table in registered_tables:
            baseline_df = self.fetch_baseline_sample(table)
            recent_df = self.fetch_recent_window(table)

            if baseline_df is None or recent_df is None:
                continue

            for col in getattr(table, "monitored_columns", []):
                alert: Optional[DriftAlert] = None
                col_name = getattr(col, "name", str(col))

                if getattr(col, "is_continuous", False):
                    b_arr = baseline_df[col_name].to_numpy() if hasattr(baseline_df[col_name], "to_numpy") else np.array(baseline_df[col_name])
                    r_arr = recent_df[col_name].to_numpy() if hasattr(recent_df[col_name], "to_numpy") else np.array(recent_df[col_name])
                    alert = DriftMonitorEngine.evaluate_continuous_drift(
                        b_arr, r_arr, col_name=col_name, table_name=getattr(table, "name", "table")
                    )
                elif getattr(col, "is_categorical", False):
                    b_counts = dict(baseline_df[col_name].value_counts())
                    r_counts = dict(recent_df[col_name].value_counts())
                    alert = DriftMonitorEngine.evaluate_categorical_drift(
                        b_counts, r_counts, col_name=col_name, table_name=getattr(table, "name", "table")
                    )

                if alert:
                    self.queue.enqueue_investigation(
                        project_id=getattr(table, "project_id", "default_proj"),
                        question=alert.auto_investigation_question,
                        priority="HIGH",
                        trigger_source="AUTONOMOUS_DAEMON",
                    )
