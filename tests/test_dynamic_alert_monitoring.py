import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _db():
    from apps.api.src.core.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def _workspace(db, *, versions=2):
    from apps.api.src.models.entities import AlertRule, Dataset, DatasetVersion, Project, User

    user = User(
        id="monitor-user",
        email="monitor@example.com",
        hashed_password="unused",
        full_name="Monitor User",
        role="analyst",
        is_active=True,
    )
    project = Project(
        id="monitor-project",
        owner_id=user.id,
        org_id="monitor-org",
        name="Monitoring Workspace",
    )
    dataset = Dataset(
        id="monitor-dataset",
        project_id=project.id,
        name="Revenue",
        current_version=versions,
        row_count=2,
        column_count=1,
        profile_json={"columns": [{"name": "revenue", "data_type": "FLOAT"}]},
    )
    db.add_all([user, project, dataset])
    for version_number in range(1, versions + 1):
        db.add(DatasetVersion(
            id=f"monitor-version-{version_number}",
            dataset_id=dataset.id,
            version_number=version_number,
            file_path=f"dataset-v{version_number}.parquet",
            row_count=2,
            column_count=1,
            content_hash=f"fingerprint-{version_number}",
            profile_json={"columns": [{"name": "revenue", "data_type": "FLOAT"}]},
        ))
    rule = AlertRule(
        id="monitor-rule",
        project_id=project.id,
        name="Revenue increase",
        metric_name="revenue",
        dataset_id=dataset.id,
        threshold_pct_change=20.0,
        frequency="daily",
        severity="high",
        auto_investigate=False,
        is_active=True,
    )
    db.add(rule)
    db.commit()
    return user, project, dataset, rule


def test_version_evaluation_persists_evidence_deduplicates_and_lifecycle(monkeypatch):
    from apps.api.src.api.v1.alerts import acknowledge_alert_event, list_alert_events
    from apps.api.src.models.entities import AlertEvent
    from apps.api.src.services.dataset_service import DatasetService
    from packages.analytics_core.src.monitoring.continuous_engine import ContinuousMonitorEngine

    engine, session_factory = _db()
    db = session_factory()
    user, project, dataset, _ = _workspace(db)
    frames = {
        1: pd.DataFrame({"revenue": [10.0, 10.0]}),
        2: pd.DataFrame({"revenue": [20.0, 20.0]}),
    }
    monkeypatch.setattr(
        DatasetService,
        "get_dataset_dataframe",
        lambda _service, _dataset_id, version=None: frames[version].copy(),
    )

    first = ContinuousMonitorEngine(db).evaluate_project_rules(project.id, trigger="manual")
    assert first[0]["status"] == "TRIGGERED"
    event = db.query(AlertEvent).one()
    assert event.current_value == 40.0
    assert event.expected_value == 20.0
    assert event.deviation_percentage == 100.0
    assert event.dataset_version_id == "monitor-version-2"
    assert event.dataset_fingerprint == "fingerprint-2"
    assert event.affected_dimensions_json["metric_aggregation"] == "sum"
    assert event.affected_dimensions_json["threshold_pct_change"] == 20.0

    second = ContinuousMonitorEngine(db).evaluate_project_rules(project.id, trigger="manual")
    assert second[0]["status"] == "DUPLICATE"
    assert db.query(AlertEvent).count() == 1

    response = list_alert_events(project_id=project.id, current_user=user, db=db)
    assert len(response) == 1
    assert response[0].source_available is True
    assert response[0].dataset_fingerprint == "fingerprint-2"
    assert response[0].threshold_pct_change == 20.0

    acknowledged = acknowledge_alert_event(event.id, current_user=user, db=db)
    assert acknowledged.lifecycle_status == "acknowledged"
    event_id = event.id
    db.close()

    reopened = session_factory()
    persisted = reopened.get(AlertEvent, event_id)
    assert persisted.lifecycle_status == "acknowledged"
    assert persisted.acknowledged_at is not None
    reopened.close()
    engine.dispose()


def test_rule_create_endpoint_returns_persisted_rule_without_fabricated_evaluation():
    from apps.api.src.api.v1.alerts import create_alert_rule
    from apps.api.src.models.entities import AlertRule
    from packages.schemas.src.alert import AlertRuleCreate

    engine, session_factory = _db()
    db = session_factory()
    user, project, dataset, _ = _workspace(db, versions=1)

    response = create_alert_rule(
        AlertRuleCreate(
            name="Quantity alert",
            project_id=project.id,
            metric_name="quantity",
            dataset_id=dataset.id,
        ),
        current_user=user,
        db=db,
    )

    persisted = db.get(AlertRule, response.id)
    assert persisted is not None
    assert response.threshold_pct_change == 15.0
    assert response.last_evaluated_at is None
    assert response.last_evaluation_status is None
    db.close()
    engine.dispose()


def test_missing_baseline_is_persisted_without_manufacturing_an_event():
    from apps.api.src.api.v1.alerts import _evaluate
    from apps.api.src.models.entities import AlertRule
    from packages.analytics_core.src.monitoring.continuous_engine import ContinuousMonitorEngine

    engine, session_factory = _db()
    db = session_factory()
    _, project, _, rule = _workspace(db, versions=1)

    outcomes = ContinuousMonitorEngine(db).evaluate_project_rules(project.id, trigger="import")

    db.refresh(rule)
    assert outcomes == [{
        "rule_id": rule.id,
        "dataset_id": "monitor-dataset",
        "dataset_version": 1,
        "status": "BASELINE_UNAVAILABLE",
    }]
    assert rule.last_evaluation_status == "BASELINE_UNAVAILABLE"
    assert rule.last_evaluation_trigger == "import"
    assert db.query(AlertRule).count() == 1
    from apps.api.src.models.entities import AlertEvent
    assert db.query(AlertEvent).count() == 0
    qualified = _evaluate(project.id, "import", db)
    assert qualified["status"] == "qualified"
    assert qualified["outcomes"][0]["status"] == "BASELINE_UNAVAILABLE"
    db.close()
    engine.dispose()


def test_monitoring_aggregations_preserve_ratio_and_weight_semantics():
    from packages.analytics_core.src.monitoring.continuous_engine import ContinuousMonitorEngine
    from packages.schemas.src.analysis import AggregationType

    weighted = pd.DataFrame({"rate": [0.1, 0.5], "volume": [90, 10]})
    weighted_semantics = type("Metric", (), {
        "aggregation_type": AggregationType.WEIGHTED_MEAN,
        "weight_column": "volume",
    })()
    assert ContinuousMonitorEngine._aggregate(weighted, "rate", weighted_semantics) == 0.14

    ratio = pd.DataFrame({"rate": [0.1, 0.5], "conversions": [1, 5], "visits": [10, 10]})
    ratio_semantics = type("Metric", (), {
        "aggregation_type": AggregationType.RATE,
        "numerator_column": "conversions",
        "denominator_column": "visits",
        "weight_column": None,
    })()
    assert ContinuousMonitorEngine._aggregate(ratio, "rate", ratio_semantics) == 0.3


def test_cleaning_profiles_new_version_and_runs_import_triggered_rules(tmp_path, monkeypatch):
    from apps.api.src.api.v1 import datasets as datasets_api
    from apps.api.src.api.v1.datasets import CleanDatasetPayload
    from apps.api.src.models.entities import AlertEvent, AlertRule
    from apps.api.src.services.dataset_service import DatasetService
    from apps.api.src.services.storage_service import StorageService
    from packages.analytics_core.src.providers.storage import LocalStorageProvider

    engine, session_factory = _db()
    db = session_factory()
    user, project, _, _ = _workspace(db, versions=1)
    service = DatasetService(
        db,
        storage=StorageService(
            base_dir=str(tmp_path),
            provider=LocalStorageProvider(str(tmp_path)),
        ),
    )
    created = service.ingest_dataset_file(
        project_id=project.id,
        filename="sales.csv",
        file_bytes=b"revenue\n10\n10\n",
    )
    rule = AlertRule(
        id="import-trigger-rule",
        project_id=project.id,
        name="Revenue change",
        metric_name="revenue",
        dataset_id=created.id,
        threshold_pct_change=20.0,
        frequency="daily",
        severity="high",
        auto_investigate=False,
        is_active=True,
    )
    db.add(rule)
    db.commit()
    monkeypatch.setattr(datasets_api, "DatasetService", lambda _db: service)

    result = datasets_api.clean_dataset(
        dataset_id=created.id,
        payload=CleanDatasetPayload(recipe=[{"action": "deduplicate_rows", "params": {}}]),
        current_user=user,
        db=db,
    )

    assert result["new_version"] == 2
    assert result["monitoring_evaluation"]["trigger"] == "import"
    assert result["monitoring_evaluation"]["status"] == "success"
    assert result["monitoring_evaluation"]["triggered_count"] == 1
    assert db.query(AlertEvent).count() == 1
    assert db.query(AlertEvent).one().dataset_version == 2
    db.close()
    engine.dispose()
