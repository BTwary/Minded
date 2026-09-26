from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")

# Config must default code execution off.
config = read("apps/api/src/core/config.py")
assert 'CODE_EXECUTION_ENABLED: bool = os.getenv("AAOS_CODE_EXECUTION_ENABLED", "false")' in config

# Demo token must be server-gated to local development, not just frontend gated.
auth = read("apps/api/src/api/v1/auth.py")
assert 'client_host = request.client.host if request.client else ""' in auth
assert 'settings.ENVIRONMENT not in {"development", "dev", "local"}' in auth

# Password hashing must use bcrypt and retain legacy verification only for migration.
security = read("apps/api/src/core/security.py")
assert 'CryptContext(schemes=["bcrypt"], deprecated="auto")' in security
assert 'return _pwd_context.hash(password)' in security

# Project reset must never contain unscoped destructive bulk deletes for these entities.
projects = read("apps/api/src/api/v1/projects.py")
assert 'db.query(DatasetVersion).delete()' not in projects
assert 'db.query(AlertEvent).delete()' not in projects

# Sandbox must default to disabled and explicitly reject arbitrary Python.
sandbox = read("packages/analytics_core/src/sandbox/runner.py")
assert 'if settings is not None and not settings.CODE_EXECUTION_ENABLED:' in sandbox
assert 'Arbitrary Python execution is disabled by default.' in sandbox
print("STATIC_SECURITY_REGRESSION_PASS")
