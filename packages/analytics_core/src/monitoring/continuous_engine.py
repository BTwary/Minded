"""Persisted, version-to-version evaluation for local alert rules."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from apps.api.src.models.entities import AlertEvent, AlertRule, Dataset, DatasetVersion
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver
from packages.schemas.src.analysis import AggregationType


class ContinuousMonitorEngine:
    """Evaluate configured rules against compatible adjacent dataset versions."""

    def __init__(self, db: Session):
        self.db = db
        self.dataset_service = DatasetService(db)

    def evaluate_project_rules(
        self,
        project_id: str,
        *,
        trigger: str = "manual",
        dataset_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = self.db.query(AlertRule).filter(
            AlertRule.project_id == project_id,
            AlertRule.is_active.is_(True),
        )
        if dataset_id:
            query = query.filter(AlertRule.dataset_id == dataset_id)

        outcomes: list[dict[str, Any]] = []
        for rule in query.order_by(AlertRule.created_at.asc()).all():
            outcomes.append(self._evaluate_rule(rule, trigger))
        return outcomes

    @staticmethod
    def _aggregate(frame: pd.DataFrame, column: str, semantics: Any) -> Optional[float]:
        aggregation = semantics.aggregation_type
        if aggregation in {AggregationType.RATE, AggregationType.RATIO, AggregationType.PROPORTION}:
            numerator = semantics.numerator_column
            denominator = semantics.denominator_column
            if numerator and denominator and numerator in frame and denominator in frame:
                numerator_sum = pd.to_numeric(frame[numerator], errors="coerce").sum(min_count=1)
                denominator_sum = pd.to_numeric(frame[denominator], errors="coerce").sum(min_count=1)
                if pd.isna(numerator_sum) or pd.isna(denominator_sum) or denominator_sum == 0:
                    return None
                return float(numerator_sum / denominator_sum)
            aggregation = (
                AggregationType.WEIGHTED_MEAN
                if semantics.weight_column
                else AggregationType.MEAN
            )

        if aggregation == AggregationType.WEIGHTED_MEAN and semantics.weight_column:
            weight_column = semantics.weight_column
            if weight_column not in frame:
                return None
            pairs = pd.DataFrame({
                "metric": pd.to_numeric(frame[column], errors="coerce"),
                "weight": pd.to_numeric(frame[weight_column], errors="coerce"),
            }).dropna()
            weight_sum = pairs["weight"].sum()
            if pairs.empty or weight_sum == 0:
                return None
            return float((pairs["metric"] * pairs["weight"]).sum() / weight_sum)

        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return None
        if aggregation == AggregationType.SUM:
            return float(values.sum(min_count=1))
        if aggregation == AggregationType.COUNT:
            return float(values.count())
        if aggregation == AggregationType.COUNT_DISTINCT:
            return float(values.nunique())
        if aggregation == AggregationType.MEAN:
            return float(values.mean())
        if aggregation == AggregationType.MEDIAN:
            return float(values.median())
        if aggregation == AggregationType.MIN:
            return float(values.min())
        if aggregation == AggregationType.MAX:
            return float(values.max())
        raise ValueError(f"Unsupported monitoring aggregation: {aggregation.value}")

    @staticmethod
    def _schema(profile: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
        if not profile or not isinstance(profile.get("columns"), list):
            return None
        return {
            str(column.get("name")): str(column.get("data_type", "")).upper()
            for column in profile["columns"]
            if isinstance(column, dict) and column.get("name") is not None
        }

    def _record_rule_outcome(self, rule: AlertRule, trigger: str, status: str) -> None:
        rule.last_evaluated_at = datetime.now(timezone.utc)
        rule.last_evaluation_status = status
        rule.last_evaluation_trigger = trigger
        self.db.commit()

    def _evaluate_rule(self, rule: AlertRule, trigger: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        try:
            dataset = self.db.query(Dataset).filter(
                Dataset.id == rule.dataset_id,
                Dataset.project_id == rule.project_id,
            ).first()
            if dataset is None:
                self._record_rule_outcome(rule, trigger, "SOURCE_UNAVAILABLE")
                return {"rule_id": rule.id, "status": "SOURCE_UNAVAILABLE"}

            current = self.db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset.id,
                DatasetVersion.version_number == dataset.current_version,
            ).first()
            previous = self.db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset.id,
                DatasetVersion.version_number == dataset.current_version - 1,
            ).first()
            if current is None or previous is None:
                self._record_rule_outcome(rule, trigger, "BASELINE_UNAVAILABLE")
                return {
                    "rule_id": rule.id,
                    "dataset_id": dataset.id,
                    "dataset_version": dataset.current_version,
                    "status": "BASELINE_UNAVAILABLE",
                }

            current_schema = self._schema(current.profile_json)
            previous_schema = self._schema(previous.profile_json)
            if current_schema is None or previous_schema is None:
                self._record_rule_outcome(rule, trigger, "PROFILE_UNAVAILABLE")
                return {"rule_id": rule.id, "dataset_id": dataset.id, "status": "PROFILE_UNAVAILABLE"}
            if current_schema != previous_schema:
                self._record_rule_outcome(rule, trigger, "INCOMPATIBLE_SCHEMA")
                return {"rule_id": rule.id, "dataset_id": dataset.id, "status": "INCOMPATIBLE_SCHEMA"}
            if rule.metric_name not in current_schema:
                self._record_rule_outcome(rule, trigger, "METRIC_UNAVAILABLE")
                return {"rule_id": rule.id, "dataset_id": dataset.id, "status": "METRIC_UNAVAILABLE"}
            if rule.threshold_pct_change is None:
                self._record_rule_outcome(rule, trigger, "THRESHOLD_UNAVAILABLE")
                return {"rule_id": rule.id, "dataset_id": dataset.id, "status": "THRESHOLD_UNAVAILABLE"}

            baseline_df = self.dataset_service.get_dataset_dataframe(dataset.id, previous.version_number)
            current_df = self.dataset_service.get_dataset_dataframe(dataset.id, current.version_number)
            if rule.metric_name not in baseline_df or rule.metric_name not in current_df:
                self._record_rule_outcome(rule, trigger, "METRIC_UNAVAILABLE")
                return {"rule_id": rule.id, "dataset_id": dataset.id, "status": "METRIC_UNAVAILABLE"}

            metric = rule.metric_name
            semantics = MetricSemanticsResolver.resolve(
                metric, current_df, table_name=dataset.name, group_dimension_col=None, grain="row"
            )
            aggregation = semantics.aggregation_type
            baseline_value = self._aggregate(baseline_df, metric, semantics)
            current_value = self._aggregate(current_df, metric, semantics)
            if baseline_value is None or current_value is None or baseline_value == 0:
                self._record_rule_outcome(rule, trigger, "BASELINE_UNDEFINED")
                return {"rule_id": rule.id, "dataset_id": dataset.id, "status": "BASELINE_UNDEFINED"}

            change_pct = (current_value - baseline_value) / baseline_value * 100.0
            if abs(change_pct) < rule.threshold_pct_change:
                self._record_rule_outcome(rule, trigger, "NO_CHANGE")
                return {
                    "rule_id": rule.id,
                    "dataset_id": dataset.id,
                    "dataset_version": current.version_number,
                    "status": "NO_CHANGE",
                    "change_pct": change_pct,
                }

            identity_payload = {
                "rule_id": rule.id,
                "dataset_version_id": current.id,
                "dataset_fingerprint": current.content_hash,
                "metric": metric,
                "baseline_value": baseline_value,
                "current_value": current_value,
                "threshold": rule.threshold_pct_change,
            }
            event_identity = hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
            existing = self.db.query(AlertEvent).filter(
                AlertEvent.event_identity == event_identity
            ).first()
            if existing:
                self._record_rule_outcome(rule, trigger, "DUPLICATE")
                return {
                    "rule_id": rule.id,
                    "event_id": existing.id,
                    "dataset_id": dataset.id,
                    "dataset_version": current.version_number,
                    "status": "DUPLICATE",
                }

            dimensions = {
                "comparison": "adjacent_dataset_versions",
                "metric_aggregation": aggregation.value,
                "metric_semantics": semantics.to_provenance_dict(),
                "metric_unit": semantics.unit,
                "threshold_pct_change": rule.threshold_pct_change,
                "baseline_rows": previous.row_count,
                "comparison_rows": current.row_count,
                "trigger": trigger,
            }
            event = AlertEvent(
                project_id=rule.project_id,
                rule_id=rule.id,
                event_identity=event_identity,
                dataset_id=dataset.id,
                dataset_version=current.version_number,
                dataset_version_id=current.id,
                dataset_fingerprint=current.content_hash,
                metric_name=metric,
                baseline_period=f"v{previous.version_number}",
                comparison_period=f"v{current.version_number}",
                title=f"{metric} changed {change_pct:+.1f}% between dataset versions",
                severity=rule.severity,
                description=(
                    f"{metric} {aggregation.value.lower()} changed from {baseline_value:,.6g} "
                    f"in version {previous.version_number} to {current_value:,.6g} "
                    f"in version {current.version_number} ({change_pct:+.2f}%)."
                ),
                current_value=current_value,
                expected_value=baseline_value,
                deviation_percentage=round(change_pct, 4),
                affected_dimensions_json=dimensions,
                evidence_summary=(
                    f"Compared profile-compatible snapshots {previous.id} and {current.id}; "
                    f"content hashes {previous.content_hash or 'not recorded'} and {current.content_hash or 'not recorded'}."
                ),
                is_resolved=False,
                lifecycle_status="active",
                triggered_at=now,
            )
            self.db.add(event)
            self._record_rule_outcome(rule, trigger, "TRIGGERED")
            self.db.refresh(event)
            return {
                "rule_id": rule.id,
                "event_id": event.id,
                "dataset_id": dataset.id,
                "dataset_version": current.version_number,
                "status": "TRIGGERED",
                "change_pct": change_pct,
            }
        except Exception as exc:
            self.db.rollback()
            try:
                failed_rule = self.db.query(AlertRule).filter(AlertRule.id == rule.id).first()
                if failed_rule:
                    self._record_rule_outcome(failed_rule, trigger, "FAILED")
            except Exception:
                self.db.rollback()
            return {"rule_id": rule.id, "status": "FAILED", "error": str(exc)}
