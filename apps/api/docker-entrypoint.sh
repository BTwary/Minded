#!/bin/bash
set -euo pipefail

if [ "${ENVIRONMENT:-development}" = "production" ]; then
    echo "Validating production configuration..."

    check_secret() {
        local var_name="$1"
        local var_value="${2:-}"
        local min_len="${3:-32}"

        if [ -z "$var_value" ]; then
            echo "FATAL: $var_name is missing or empty."
            exit 1
        fi
        if [ "${#var_value}" -lt "$min_len" ]; then
            echo "FATAL: $var_name must be at least ${min_len} characters."
            exit 1
        fi
        case "$var_value" in
            analyst_secret_2026|minioadmin|minioadmin123|password|secret|changeme|replace-with-a-secure-random-secret-key|dev_secret_key_12345|super-secret-dev-*)
                echo "FATAL: $var_name contains a known default/weak value."
                exit 1
                ;;
        esac
    }

    check_secret "POSTGRES_USER" "${POSTGRES_USER:-}" 16
    check_secret "POSTGRES_PASSWORD" "${POSTGRES_PASSWORD:-}" 32
    check_secret "MINIO_ROOT_USER" "${MINIO_ROOT_USER:-}" 16
    check_secret "MINIO_ROOT_PASSWORD" "${MINIO_ROOT_PASSWORD:-}" 32
    check_secret "AAOS_REDIS_PASSWORD" "${AAOS_REDIS_PASSWORD:-}" 32
    check_secret "SECRET_KEY" "${SECRET_KEY:-}" 32
    check_secret "AAOS_ADMIN_BOOTSTRAP_TOKEN" "${AAOS_ADMIN_BOOTSTRAP_TOKEN:-}" 32

    case "${DATABASE_URL:-}" in
        *analyst_secret_2026*|*minioadmin*|*password*)
            echo "FATAL: DATABASE_URL contains an obvious default/weak credential."
            exit 1
            ;;
    esac

    echo "Production configuration validated."
fi

if command -v alembic >/dev/null 2>&1; then
    echo "Running Alembic migrations..."
    alembic upgrade head
elif [ -f "apps/api/src/db/migrate.py" ]; then
    echo "Running custom migrations..."
    python -m apps.api.src.db.migrate
else
    echo "No migration tool detected; relying on application database initialization."
fi

exec "$@"
