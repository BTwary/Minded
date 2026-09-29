"""System constants and defaults for Autonomous Data Analyst Platform."""

import os

API_V1_STR = "/api/v1"
PROJECT_NAME = "Autonomous AI Data Analyst"
DEFAULT_DATA_DIR = "./data"
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_QUERY_EXECUTION_TIME_SECONDS = int(os.getenv("AAOS_MAX_QUERY_EXECUTION_TIME_SECONDS", "30"))
MAX_INVESTIGATION_RUNTIME_SECONDS = int(os.getenv("AAOS_MAX_INVESTIGATION_RUNTIME_SECONDS", "300"))
MAX_INVESTIGATION_EXPERIMENTS = int(os.getenv("AAOS_MAX_INVESTIGATION_EXPERIMENTS", "8"))
MAX_QUERY_RESULT_ROWS = 50000
MAX_AUTONOMOUS_RELATIONAL_HOPS = int(os.getenv("AAOS_MAX_AUTONOMOUS_RELATIONAL_HOPS", "5"))  # up to 6 tables in automatic discovery
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
MAX_ORCHESTRATION_STEPS = 12
MAX_REPLAN_ATTEMPTS = 3

SQL_FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", 
    "CREATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "COPY", "ATTACH", "DETACH", "INSTALL", "LOAD", "EXPORT",
    "CALL", "PRAGMA", "VACUUM", "CHECKPOINT"
]

SAFE_PYTHON_MODULE_ALLOWLIST = [
    "math", "statistics", "numpy", "scipy", "pandas", "polars",
    "datetime", "json", "re", "collections", "itertools", "sklearn"
]
