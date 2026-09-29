from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_local_compose_is_zero_cost_and_ai_off():
    c = _load("docker-compose.dev.yml")
    env = c["services"]["api"]["environment"]
    assert env["DATABASE_URL"].startswith("sqlite:")
    assert env["AI_ENABLED"] == "false"
    assert env["AI_PROVIDER"] == "none"
    assert set(c["services"]) == {"api", "web"}


def test_prod_compose_requires_all_secret_interpolations():
    raw = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")
    required = [
        "AAOS_POSTGRES_USER", "AAOS_POSTGRES_PASSWORD", "AAOS_REDIS_PASSWORD",
        "AAOS_MINIO_ROOT_USER", "AAOS_MINIO_ROOT_PASSWORD", "AAOS_JWT_SECRET",
        "AAOS_ADMIN_BOOTSTRAP_TOKEN",
    ]
    for name in required:
        assert re.search(r"\$\{" + re.escape(name) + r":\?[^}]+\}", raw)


def test_prod_api_receives_entrypoint_secret_variables():
    c = _load("docker-compose.prod.yml")
    env = c["services"]["api"]["environment"]
    for name in [
        "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "DATABASE_URL",
        "SECRET_KEY", "REDIS_URL", "AAOS_REDIS_PASSWORD",
        "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "AAOS_ADMIN_BOOTSTRAP_TOKEN",
    ]:
        assert name in env, name


def test_infrastructure_not_on_public_network():
    c = _load("docker-compose.prod.yml")
    for name in ["postgres", "redis", "minio"]:
        assert c["services"][name]["networks"] == ["internal"]
    assert c["networks"]["internal"]["internal"] is True


def test_pinned_images_and_hardened_containers():
    c = _load("docker-compose.prod.yml")
    assert c["services"]["postgres"]["image"] == "pgvector/pgvector:0.6.2-pg16"
    assert c["services"]["redis"]["image"] == "redis:7.2.4-alpine"
    assert c["services"]["minio"]["image"] == "minio/minio:RELEASE.2024-03-03T17-50-39Z"
    for name in ["api", "web"]:
        s = c["services"][name]
        assert "no-new-privileges:true" in s["security_opt"]
        assert "ALL" in s["cap_drop"]


def test_prod_compose_does_not_put_redis_password_in_healthcheck_cli():
    c = _load("docker-compose.prod.yml")
    raw = str(c["services"]["redis"]["healthcheck"]["test"])
    assert "AAOS_REDIS_PASSWORD" not in raw


def test_entrypoint_is_fail_closed():
    raw = (ROOT / "apps/api/docker-entrypoint.sh").read_text(encoding="utf-8")
    for name in ["POSTGRES_USER", "POSTGRES_PASSWORD", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "AAOS_REDIS_PASSWORD", "SECRET_KEY", "AAOS_ADMIN_BOOTSTRAP_TOKEN"]:
        assert f'check_secret "{name}"' in raw
    assert "alembic upgrade head" in raw


def test_web_runtime_uses_lockfile_and_pruned_dependencies():
    raw = (ROOT / "apps/web/Dockerfile").read_text(encoding="utf-8")
    assert "COPY package.json package-lock.json ./" in raw
    assert "npm prune --omit=dev" in raw
    assert "node:20.11.1-alpine" in raw


def test_api_runtime_uses_certified_dependency_lock():
    raw = (ROOT / "apps/api/Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements-lock.txt /tmp/requirements-lock.txt" in raw
    assert "pip install --no-cache-dir -r /tmp/requirements-lock.txt" in raw

