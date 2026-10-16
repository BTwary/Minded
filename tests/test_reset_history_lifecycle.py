from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.core.config import settings


def _database(path=":memory:"):
    from apps.api.src.core.database import Base

    engine = create_engine(f"sqlite:///{path}" if path != ":memory:" else "sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def _workspace(db, investigation_count=1):
    from apps.api.src.models.entities import Investigation, Project, User

    user = User(
        id="history-user",
        email="history@example.com",
        hashed_password="unused",
        full_name="History User",
        role="analyst",
        is_active=True,
    )
    project = Project(id="history-project", owner_id=user.id, org_id="history-org", name="History Project")
    db.add_all([user, project])
    db.commit()

    investigations = []
    for index in range(investigation_count):
        created_at = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
        investigation = Investigation(
            id=f"INV-{index:04d}",
            project_id=project.id,
            user_id=user.id,
            question=f"Question {index}",
            status="COMPLETED" if index % 2 == 0 else "FAILED",
            verdict_type="OBSERVED" if index % 2 == 0 else "INCONCLUSIVE",
            direct_answer=f"Saved answer {index}",
            requested_dataset_ids_json=["removed-dataset"],
            created_at=created_at,
        )
        investigations.append(investigation)
    db.add_all(investigations)
    db.commit()
    return user, project, investigations


def test_reset_removes_dataset_profile_versions_and_transformations_but_preserves_history(tmp_path, monkeypatch):
    from apps.api.src.api.v1.investigations import list_analysis_history
    from apps.api.src.api.v1.projects import reset_project
    from apps.api.src.models.entities import (
        AlertEvent,
        AlertRule,
        Dataset,
        DatasetTransformation,
        DatasetVersion,
        Investigation,
    )

    engine, session_factory = _database()
    db = session_factory()
    user, project, _ = _workspace(db)
    storage_root = tmp_path / "data_store"
    storage_root.mkdir()
    parquet_file = storage_root / "dataset.parquet"
    parquet_file.write_bytes(b"local dataset snapshot")
    dataset = Dataset(
        id="dataset-1",
        project_id=project.id,
        name="Revenue",
        current_version=1,
        row_count=3,
        column_count=2,
        data_quality_score=91.0,
        profile_json={"profiled_at": "2025-01-01T00:00:00Z", "columns": [{"name": "revenue", "mean_value": 25}]},
    )
    version = DatasetVersion(
        id="version-1",
        dataset_id=dataset.id,
        version_number=1,
        file_path=str(parquet_file),
        row_count=3,
        column_count=2,
        transformation_applied={"source_sha256": "snapshot-hash"},
    )
    transformation = DatasetTransformation(
        id="transform-1",
        dataset_id=dataset.id,
        output_version_id=version.id,
        operation="INGEST",
        record_hash="transformation-hash",
    )
    rule = AlertRule(
        id="reset-preserved-rule",
        project_id=project.id,
        name="Revenue drift",
        metric_name="revenue",
        dataset_id=dataset.id,
        threshold_pct_change=15.0,
        severity="high",
        is_active=True,
    )
    alert_event = AlertEvent(
        id="reset-preserved-alert",
        project_id=project.id,
        rule_id=rule.id,
        event_identity="reset-event-identity",
        dataset_id=dataset.id,
        dataset_version=1,
        dataset_version_id=version.id,
        dataset_fingerprint="snapshot-hash",
        metric_name="revenue",
        title="Revenue drift detected",
        severity="high",
        description="Recorded evidence from the removed source snapshot.",
        current_value=120.0,
        expected_value=100.0,
        deviation_percentage=20.0,
        is_resolved=False,
        lifecycle_status="active",
    )
    db.add_all([dataset, version, transformation, rule, alert_event])
    db.commit()
    dataset_id = dataset.id
    version_id = version.id
    transformation_id = transformation.id
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "DATA_STORAGE_DIR", str(storage_root))

    result = reset_project(project_id=project.id, current_user=user, db=db)

    assert result["status"] == "success"
    assert result["history_preserved"] is True
    assert db.get(Dataset, dataset_id) is None
    assert db.get(DatasetVersion, version_id) is None
    assert db.get(DatasetTransformation, transformation_id) is None
    assert db.get(AlertEvent, alert_event.id) is not None
    assert db.get(AlertRule, rule.id).is_active is False
    assert not parquet_file.exists()
    assert db.get(Investigation, "INV-0000").direct_answer == "Saved answer 0"
    history = list_analysis_history(current_user=user, db=db, limit=100)
    assert history["records"][0]["direct_answer"] == "Saved answer 0"
    assert history["records"][0]["source_datasets"][0]["available"] is False
    from apps.api.src.api.v1.alerts import list_alert_events
    historical_alerts = list_alert_events(project_id=project.id, current_user=user, db=db)
    assert len(historical_alerts) == 1
    assert historical_alerts[0].source_available is False
    assert historical_alerts[0].dataset_fingerprint == "snapshot-hash"
    db.close()
    engine.dispose()


def test_reset_failure_is_reported_and_does_not_remove_other_project_data(tmp_path, monkeypatch):
    from apps.api.src.api.v1.projects import reset_project
    from apps.api.src.models.entities import Dataset, Project, User

    _, session_factory = _database()
    db = session_factory()
    user, project, _ = _workspace(db)
    other_project = Project(id="other-project", owner_id="other-user", org_id="other-org", name="Other")
    other_dataset = Dataset(id="other-dataset", project_id=other_project.id, name="Other data")
    db.add_all([other_project, other_dataset])
    db.commit()

    outside_file = tmp_path / "outside.parquet"
    outside_file.write_bytes(b"must not be removed")
    dataset = Dataset(
        id="bad-path-dataset",
        project_id=project.id,
        name="Protected",
        profile_json={"columns": [{"name": "value", "mean_value": 100}]},
    )
    from apps.api.src.models.entities import DatasetVersion
    db.add(dataset)
    db.commit()
    db.add(DatasetVersion(
        id="bad-path-version",
        dataset_id=dataset.id,
        version_number=1,
        file_path=str(outside_file),
    ))
    db.commit()
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "DATA_STORAGE_DIR", str(tmp_path / "managed"))

    with pytest.raises(HTTPException) as raised:
        reset_project(project_id=project.id, current_user=user, db=db)

    assert raised.value.status_code == 503
    assert "incomplete" in raised.value.detail.lower()
    assert outside_file.read_bytes() == b"must not be removed"
    assert db.get(Project, project.id) is not None
    assert db.get(Dataset, other_dataset.id) is not None
    with pytest.raises(HTTPException) as retry:
        reset_project(project_id=project.id, current_user=user, db=db)
    assert retry.value.status_code == 503
    assert outside_file.read_bytes() == b"must not be removed"
    assert db.get(Dataset, other_dataset.id) is not None
    db.close()


def test_analysis_history_survives_database_reopen_and_keeps_latest_ten(tmp_path, monkeypatch):
    from apps.api.src.api.v1.investigations import list_analysis_history
    from apps.api.src.models.entities import Evidence, EvidenceVerification, InvestigationEvent

    database_path = tmp_path / "history.sqlite"
    engine, session_factory = _database(str(database_path))
    db = session_factory()
    user, _, _ = _workspace(db, investigation_count=12)
    db.add_all([
        InvestigationEvent(
            id="event-analyst-result",
            sequence=1,
            investigation_id="INV-0011",
            event_type="investigation.analyst_result",
            event_payload_json={"headline": "Original headline", "metric": {"value": 42, "unit": "rows"}},
        ),
        Evidence(
            id="evidence-11",
            investigation_id="INV-0011",
            statement="Persisted evidence statement",
            validation_status="verified",
            calculation_summary="Original calculation summary",
        ),
    ])
    db.commit()
    db.add(EvidenceVerification(
        id="verification-11",
        evidence_id="evidence-11",
        primary_tool="DuckDB",
        secondary_tool="Python",
        status="PASSED",
    ))
    db.commit()
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    db.commit()
    db.close()
    engine.dispose()

    reopened_engine, reopened_factory = _database(str(database_path))
    reopened_db = reopened_factory()
    from apps.api.src.models.entities import User
    reopened_user = reopened_db.get(User, "history-user")
    result = list_analysis_history(current_user=reopened_user, db=reopened_db, limit=100)

    assert len(result["records"]) == 12
    assert {row["question"] for row in result["records"]} >= {f"Question {i}" for i in range(10)}
    assert result["records"][0]["direct_answer"] == "Saved answer 11"
    assert result["records"][0]["status"] == "FAILED"
    assert result["records"][0]["headline"] == "Original headline"
    assert result["records"][0]["verification_status"] == "VERIFIED"
    assert result["records"][0]["evidence"][0]["statement"] == "Persisted evidence statement"
    assert result["records"][0]["analyst_result"]["metric"]["value"] == 42
    reopened_db.close()
    reopened_engine.dispose()


def test_clear_history_removes_analysis_records_only(tmp_path, monkeypatch):
    from apps.api.src.api.v1.investigations import clear_analysis_history, list_analysis_history
    from apps.api.src.models.entities import Dataset, Investigation, Project

    _, session_factory = _database()
    db = session_factory()
    user, project, _ = _workspace(db)
    dataset = Dataset(id="preserved-dataset", project_id=project.id, name="Still here")
    db.add(dataset)
    db.commit()
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    result = clear_analysis_history(current_user=user, db=db)

    assert result["status"] == "success"
    assert db.get(Investigation, "INV-0000") is None
    assert db.get(Project, project.id) is not None
    assert db.get(Dataset, dataset.id) is not None
    assert list_analysis_history(current_user=user, db=db, limit=100)["records"] == []
    db.close()


def test_new_import_returns_fresh_profile_and_preview_from_same_version(tmp_path, monkeypatch):
    from apps.api.src.api.v1 import datasets as datasets_api
    from apps.api.src.services.dataset_service import DatasetService
    from apps.api.src.services.storage_service import StorageService
    from packages.analytics_core.src.providers.storage import LocalStorageProvider

    _, session_factory = _database()
    db = session_factory()
    user, project, _ = _workspace(db)
    storage = StorageService(
        base_dir=str(tmp_path),
        provider=LocalStorageProvider(str(tmp_path)),
    )
    service = DatasetService(db, storage=storage)
    dataset = service.ingest_dataset_file(
        project_id=project.id,
        filename="quantities.csv",
        file_bytes=b"discount_pct,quantity\n0.1,2\n0.2,4\n,6\n",
    )
    monkeypatch.setattr(datasets_api, "DatasetService", lambda _db: service)
    details = datasets_api.get_dataset(dataset.id, current_user=user, db=db)
    preview = datasets_api.get_dataset_data(
        dataset_id=dataset.id,
        limit=10,
        version=details["current_version"],
        current_user=user,
        db=db,
    )

    assert details["profile"]["columns"]
    assert details["current_version"] == 1
    assert preview["dataset_id"] == dataset.id
    assert preview["columns"] == ["discount_pct", "quantity"]
    assert preview["row_count"] == preview["total_rows"] == 3
    assert preview["rows"][0] == {"discount_pct": 0.1, "quantity": 2}
    assert preview["rows"][2]["discount_pct"] is None
    db.close()
