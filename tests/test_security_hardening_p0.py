import os
from pathlib import Path

from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from apps.api.src.core.config import settings
from apps.api.src.core.security import get_password_hash, verify_password, needs_password_rehash
from packages.analytics_core.src.sandbox.runner import PythonSandboxRunner, SandboxSecurityError


def test_bcrypt_hash_roundtrip():
    hashed = get_password_hash("correct horse battery staple")
    assert hashed.startswith("$2")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)
    assert not needs_password_rehash(hashed)


def test_arbitrary_python_execution_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "CODE_EXECUTION_ENABLED", False)
    runner = PythonSandboxRunner()
    with pytest.raises(SandboxSecurityError):
        runner.execute_safe_python("import numpy as np; result = np.memmap('secret.txt', mode='r')")


def test_known_filesystem_numerical_escape_is_blocked(monkeypatch):
    monkeypatch.setattr(settings, "CODE_EXECUTION_ENABLED", True)
    runner = PythonSandboxRunner()
    payloads = [
        "import numpy as np; result=np.memmap('secret.txt', mode='r')",
        "import numpy as np; result=np.fromfile('secret.txt', dtype='uint8')",
        "import numpy as np; result=np.load('secret.npy')",
    ]
    for payload in payloads:
        with pytest.raises(SandboxSecurityError):
            runner.execute_safe_python(payload, timeout_seconds=1)


def test_repo_default_does_not_enable_code_execution():
    assert settings.CODE_EXECUTION_ENABLED is False


def test_project_reset_is_project_scoped(tmp_path, monkeypatch):
    from apps.api.src.core.database import Base
    from apps.api.src.models.entities import Project, Dataset, DatasetVersion, AlertRule, AlertEvent, User
    from apps.api.src.api.v1.projects import reset_project

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "DATA_STORAGE_DIR", str(tmp_path / "data_store"))

    user = User(id="u1", email="u@example.com", hashed_password="x", full_name="User", role="analyst", is_active=True)
    p1 = Project(id="p1", owner_id="u1", org_id="o1", name="P1")
    p2 = Project(id="p2", owner_id="u2", org_id="o2", name="P2")
    db.add_all([user, p1, p2])
    db.commit()

    d1 = Dataset(id="d1", project_id="p1", name="d1")
    d2 = Dataset(id="d2", project_id="p2", name="d2")
    db.add_all([d1, d2]); db.commit()
    v1 = DatasetVersion(id="v1", dataset_id="d1", version_number=1, file_path="p1.csv")
    v2 = DatasetVersion(id="v2", dataset_id="d2", version_number=1, file_path="p2.csv")
    r1 = AlertRule(id="r1", project_id="p1", name="r1", metric_name="m", dataset_id="d1")
    r2 = AlertRule(id="r2", project_id="p2", name="r2", metric_name="m", dataset_id="d2")
    db.add_all([v1, v2, r1, r2]); db.commit()
    e1 = AlertEvent(id="e1", project_id="p1", rule_id="r1", title="e1", description="e1", current_value=1, expected_value=1, deviation_percentage=0)
    e2 = AlertEvent(id="e2", project_id="p2", rule_id="r2", title="e2", description="e2", current_value=1, expected_value=1, deviation_percentage=0)
    db.add_all([e1, e2]); db.commit()

    reset_project(project_id="p1", current_user=user, db=db)

    assert db.get(Dataset, "d1") is None
    assert db.get(DatasetVersion, "v1") is None
    assert db.get(AlertRule, "r1") is None
    assert db.get(AlertEvent, "e1") is None
    assert db.get(Dataset, "d2") is not None
    assert db.get(DatasetVersion, "v2") is not None
    assert db.get(AlertRule, "r2") is not None
    assert db.get(AlertEvent, "e2") is not None
    db.close()


def test_demo_token_policy_requires_local_development(monkeypatch):
    from apps.api.src.api.v1.auth import get_demo_token
    from starlette.requests import Request

    def make_request(host):
        scope={"type":"http","method":"POST","path":"/auth/demo-token","headers":[],"client":(host,1234),"scheme":"http","server":(host,8000)}
        return Request(scope)

    monkeypatch.setattr(settings, "REQUIRE_LOCAL_DEMO_TOKEN", True)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    db = None
    with pytest.raises(Exception) as exc:
        get_demo_token(make_request("10.0.0.7"), db)
    assert getattr(exc.value, "status_code", None) == 404
