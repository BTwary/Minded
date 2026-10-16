"""Registration must authenticate only the newly persisted account."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from apps.api.src.api.v1 import auth
from apps.api.src.core.database import get_db
from apps.api.src.core.security import decode_access_token, verify_password
from apps.api.src.models.entities import Organization, User


PAYLOAD = {
    "email": "new@example.com",
    "password": "registration password",
    "full_name": "New Analyst",
    "role": "superadmin",
}


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'registration.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Organization.__table__.create(engine)
    User.__table__.create(engine)
    with Session(engine) as db:
        db.add(Organization(id="default-org", name="Test Organization", slug="test"))
        db.commit()
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine):
    with Session(engine) as db:
        yield db


def _app(session_factory):
    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")

    def database():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    return app


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        yield client


@pytest.fixture
def token_issuer(monkeypatch):
    issuer = Mock(wraps=auth.create_access_token)
    monkeypatch.setattr(auth, "create_access_token", issuer)
    return issuer


def _add_user(db, user_id, email):
    db.add(User(
        id=user_id, email=email, full_name="Existing User",
        hashed_password=auth.get_password_hash("existing password"), role="admin",
    ))
    db.commit()


def test_successful_registration_preserves_token_password_and_login_contract(client, db, token_issuer):
    response = client.post("/api/v1/auth/register", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"access_token", "token_type", "user_id", "email", "full_name", "role"}
    assert body["token_type"] == "bearer"
    assert body["email"] == PAYLOAD["email"]
    assert body["full_name"] == PAYLOAD["full_name"]
    assert body["role"] == "analyst"
    user = db.query(User).filter_by(email=PAYLOAD["email"]).one()
    assert body["user_id"] == decode_access_token(body["access_token"]) == user.id
    assert user.hashed_password.startswith("$2")
    assert verify_password(PAYLOAD["password"], user.hashed_password)
    assert not verify_password("wrong password", user.hashed_password)
    token_issuer.assert_called_once_with(subject=user.id)

    profile = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert profile.status_code == 200
    assert profile.json()["id"] == user.id
    login = client.post("/api/v1/auth/login", json={"email": PAYLOAD["email"], "password": PAYLOAD["password"]})
    assert login.status_code == 200
    assert decode_access_token(login.json()["access_token"]) == user.id


@pytest.mark.parametrize("user_id", ["existing-user", "default-user"])
def test_existing_email_returns_conflict_without_issuing_a_token(client, db, token_issuer, monkeypatch, user_id):
    _add_user(db, user_id, PAYLOAD["email"])
    commit = Mock(wraps=db.commit)
    monkeypatch.setattr(db, "commit", commit)
    response = client.post("/api/v1/auth/register", json=PAYLOAD)
    assert response.status_code == 409
    assert response.json() == {"detail": "User with this email already exists."}
    commit.assert_not_called()  # Exercise the pre-check, not the unique-violation handler.
    token_issuer.assert_not_called()
    user = db.query(User).filter_by(email=PAYLOAD["email"]).one()
    assert user.id == user_id
    assert user.role == "admin"
    assert verify_password("existing password", user.hashed_password)


@pytest.mark.parametrize("default_present", [False, True])
@pytest.mark.parametrize("constraint", ["foreign_key", "not_null", "primary_key"])
def test_unrelated_integrity_failure_never_authenticates_an_existing_user(
    client, db, token_issuer, monkeypatch, default_present, constraint,
):
    _add_user(db, "other-user", "other@example.com")
    if default_present:
        _add_user(db, "default-user", "default@example.com")
    rollback = Mock(wraps=db.rollback)
    monkeypatch.setattr(db, "rollback", rollback)

    @event.listens_for(db, "before_flush")
    def inject_invalid_insert(session, _context, _instances):
        for user in session.new:
            if isinstance(user, User):
                if constraint == "foreign_key":
                    user.organization_id = "missing-organization"
                elif constraint == "not_null":
                    user.full_name = None
                else:
                    user.id = "other-user"

    response = client.post("/api/v1/auth/register", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to register user."}
    token_issuer.assert_not_called()
    rollback.assert_called_once_with()
    assert db.query(User).filter_by(email=PAYLOAD["email"]).count() == 0
    assert db.get(User, "other-user").email == "other@example.com"
    if default_present:
        assert db.get(User, "default-user").email == "default@example.com"


@pytest.mark.parametrize("stage", ["query", "commit", "refresh"])
def test_persistence_errors_are_sanitized_and_never_issue_tokens(client, db, token_issuer, monkeypatch, stage):
    _add_user(db, "default-user", "default@example.com")
    rollback = Mock(wraps=db.rollback)
    monkeypatch.setattr(db, "rollback", rollback)
    error = OperationalError("private SQL", {"private": "parameters"}, sqlite3.OperationalError("private database details"))
    with monkeypatch.context() as patch:
        patch.setattr(db, stage, Mock(side_effect=error))
        response = client.post("/api/v1/auth/register", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to register user."}
    token_issuer.assert_not_called()
    rollback.assert_called_once_with()
    # A failed refresh can follow a committed insert, but must still issue no token.
    assert db.query(User).filter_by(email=PAYLOAD["email"]).count() == (stage == "refresh")


@pytest.mark.parametrize("code_attribute", ["pgcode", "sqlstate"])
@pytest.mark.parametrize("code,constraint,expected_status", [
    ("23505", "ix_users_email", 409),
    ("23505", "users_pkey", 500),
    ("23503", "ix_users_email", 500),
    ("23502", None, 500),
])
def test_postgresql_constraint_diagnostics_are_scoped_to_unique_email(
    client, db, token_issuer, monkeypatch, code_attribute, code, constraint, expected_status,
):
    _add_user(db, "default-user", "default@example.com")
    original = Exception("private PostgreSQL error")
    setattr(original, code_attribute, code)
    original.diag = SimpleNamespace(constraint_name=constraint)
    error = IntegrityError("private SQL", {}, original)
    monkeypatch.setattr(db, "commit", Mock(side_effect=error))
    response = client.post("/api/v1/auth/register", json=PAYLOAD)
    assert response.status_code == expected_status
    expected_detail = "User with this email already exists." if expected_status == 409 else "Unable to register user."
    assert response.json() == {"detail": expected_detail}
    token_issuer.assert_not_called()


def test_concurrent_duplicate_registration_issues_only_the_winning_identity_token(engine, token_issuer):
    barrier = Barrier(2)
    unique_failures = []

    class RacingSession(Session):
        def commit(self):
            # Both requests must finish their existence check before either inserts.
            barrier.wait(timeout=10)
            try:
                return super().commit()
            except IntegrityError as error:
                unique_failures.append(error)
                raise

    factory = sessionmaker(bind=engine, class_=RacingSession)
    with Session(engine) as db:
        _add_user(db, "default-user", "default@example.com")
    app = _app(factory)

    def register(index):
        payload = dict(PAYLOAD, full_name=f"Registrant {index}", password=f"password for registrant {index}")
        with TestClient(app) as client:
            return payload, client.post("/api/v1/auth/register", json=payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(register, range(2)))
    assert sorted(response.status_code for _, response in results) == [200, 409]
    assert len(unique_failures) == 1  # Exercise the real database race, not the pre-check.
    payload, success = next(item for item in results if item[1].status_code == 200)
    failure = next(response for _, response in results if response.status_code == 409)
    assert failure.json() == {"detail": "User with this email already exists."}
    with Session(engine) as db:
        user = db.query(User).filter_by(email=PAYLOAD["email"]).one()
        assert user.full_name == payload["full_name"]
        assert verify_password(payload["password"], user.hashed_password)
        assert decode_access_token(success.json()["access_token"]) == user.id != "default-user"
        token_issuer.assert_called_once_with(subject=user.id)
        assert db.get(User, "default-user").email == "default@example.com"
