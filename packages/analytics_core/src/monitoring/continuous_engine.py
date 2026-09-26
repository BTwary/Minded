"""Continuous Autonomous Monitoring Engine: Detects Anomaly Deviations & Triggers Autonomous Investigation Loops."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from apps.api.src.core.config import settings
from apps.api.src.models.entities import AlertEvent, AlertRule, Dataset, Investigation, gen_uuid
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.execution.queue import DatabaseQueueProvider
from packages.analytics_core.src.execution.state_machine import InvestigationState
from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver
from packages.schemas.src.analysis import AggregationType


class ContinuousMonitorEngine:
    """Evaluates active alert rules, detects distribution anomalies, and autonomously dispatches root-cause investigations."""

    def __init__(self, db: Session):
        self.db = db
        self.dataset_service = DatasetService(db)
        self.controller = InvestigationController(session_factory=lambda: db)
        self.queue = DatabaseQueueProvider(session_factory=lambda: db)

    def evaluate_project_rules(self, project_id: str) -> List[Dict[str, Any]]:
        """Evaluate all active monitoring rules for a given project."""
        rules = (
            self.db.query(AlertRule)
            .filter(AlertRule.project_id == project_id, AlertRule.is_active == True)
            .all()
        )
        triggered_events = []

        for rule in rules:
            try:
                # Load dataset
                dataset = self.db.query(Dataset).filter(Dataset.id == rule.dataset_id).first()
                if not dataset:
                    continue

                df = self.dataset_service.get_dataset_dataframe(dataset.id, dataset.current_version)
                if df.empty or rule.metric_name not in df.columns:
                    continue

                # Compute metric deviation (period-over-period or outlier threshold)
                metric_col = rule.metric_name
                time_col = next((c for c in df.columns if any(k in c.lower() for k in ["date", "time", "order_date"])), None)

                if time_col and time_col in df.columns:
                    df["_period"] = df[time_col].astype(str).str.slice(0, 7)
                    periods = sorted(df["_period"].unique())
                    if len(periods) >= 2:
                        p_prev = periods[-2]
                        p_curr = periods[-1]
                        metric_def = MetricSemanticsResolver.resolve(
                            metric_col,
                            df,
                            table_name=dataset.name,
                            group_dimension_col=None,
                            grain="row",
                        )

                        def _aggregate_period(frame: pd.DataFrame) -> Optional[float]:
                            series = frame[metric_col]
                            agg = metric_def.aggregation_type
                            if agg == AggregationType.SUM:
                                return float(series.sum())
                            if agg == AggregationType.MEAN:
                                return float(series.mean()) if len(series.dropna()) else None
                            if agg == AggregationType.COUNT:
                                return float(series.count())
                            if agg == AggregationType.COUNT_DISTINCT:
                                return float(series.nunique(dropna=True))
                            if agg in (AggregationType.RATE, AggregationType.RATIO, AggregationType.PROPORTION):
                                num = metric_def.numerator_column
                                den = metric_def.denominator_column
                                if num and den and den in frame.columns:
                                    den_sum = float(frame[den].sum())
                                    return float(frame[num].sum()) / den_sum if den_sum else None
                                return float(series.mean()) if len(series.dropna()) else None
                            if agg == AggregationType.WEIGHTED_MEAN and metric_def.weight_column in frame.columns:
                                vals = frame[metric_col]
                                weights = frame[metric_def.weight_column]
                                mask = vals.notna() & weights.notna()
                                weight_sum = float(weights[mask].sum())
                                return float((vals[mask] * weights[mask]).sum()) / weight_sum if weight_sum else None
                            return float(series.mean()) if len(series.dropna()) else None

                        val_prev = _aggregate_period(df[df["_period"] == p_prev])
                        val_curr = _aggregate_period(df[df["_period"] == p_curr])
                        # A missing/undefined period aggregate or zero baseline does not
                        # imply a zero change. Do not manufacture an analytical percentage.
                        pct_change = None
                        if val_prev is not None and val_curr is not None and val_prev != 0:
                            pct_change = (val_curr - val_prev) / val_prev * 100.0

                        # Check threshold only when the deviation is defined.
                        if pct_change is not None and abs(pct_change) >= rule.threshold_pct_change:
                            # Autonomous Trigger
                            analysis_id = None
                            evidence_summary = None

                            if rule.auto_investigate:
                                inv_id = f"INV-ALERT-{gen_uuid()[:8]}"
                                question = f"Why did {metric_col} change by {pct_change:.1f}% between {p_prev} and {p_curr}?"
                                inv_entity = Investigation(
                                    id=inv_id,
                                    project_id=project_id,
                                    question=question,
                                    status=InvestigationState.QUEUED,
                                )
                                self.db.add(inv_entity)
                                self.db.commit()
                                self.queue.enqueue(investigation_id=inv_id, priority=50)
                                analysis_id = inv_id
                                evidence_summary = f"Autonomous investigation enqueued for {metric_col} ({pct_change:+.1f}% shift)."

                            event = AlertEvent(
                                rule_id=rule.id,
                                title=f"Anomaly: {metric_col} {pct_change:+.1f}% Shift",
                                severity=rule.severity,
                                description=(
                                    f"Observed {pct_change:+.1f}% deviation in {metric_col} between {p_prev} ({val_prev:,.4g}) "
                                    f"and {p_curr} ({val_curr:,.4g}). Metric semantics: {metric_def.aggregation_type.value}."
                                ),
                                current_value=val_curr,
                                expected_value=val_prev,
                                deviation_percentage=round(pct_change, 2),
                                affected_dimensions_json={
                                    "period_a": p_prev,
                                    "period_b": p_curr,
                                    "metric_aggregation": metric_def.aggregation_type.value,
                                    "metric_semantics": metric_def.to_provenance_dict(),
                                },
                                investigation_analysis_id=analysis_id,
                                evidence_summary=evidence_summary,
                                is_resolved=False,
                                triggered_at=datetime.now(timezone.utc),
                            )
                            self.db.add(event)
                            self.db.commit()
                            self.db.refresh(event)

                            rule.last_evaluated_at = datetime.now(timezone.utc)
                            self.db.commit()

                            triggered_events.append({
                                "event_id": event.id,
                                "rule_name": rule.name,
                                "metric": metric_col,
                                "deviation_pct": pct_change,
                                "investigation_id": analysis_id,
                                "evidence_summary": evidence_summary,
                            })
            except Exception as e:
                continue

        return triggered_events
