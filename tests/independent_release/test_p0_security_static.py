from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_registration_cannot_persist_payload_role():
    text = (ROOT / "apps/api/src/api/v1/auth.py").read_text()
    assert 'role=payload.role' not in text
    assert 'role="analyst"' in text


def test_compose_has_no_usable_default_passwords():
    text = (ROOT / "docker-compose.yml").read_text()
    assert 'POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-' not in text
    assert 'MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-' not in text
    assert '${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}' in text
    assert '${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD must be set}' in text


def test_sql_source_contains_registered_relation_and_path_blocks():
    text = (ROOT / "packages/analytics_core/src/sql/engine.py").read_text()
    assert "path-backed relations are forbidden" in text
    assert "Relation '{base}' is not registered" in text
