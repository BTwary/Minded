"""Run the clean-room acceptance benchmark with explicit environment gating.

The engine-independent oracle suite must pass on any supported developer machine.
The real controller E2E gate is only reported as PASS when its required execution
engines are installed; missing engines are reported as BLOCKED, never as PASS.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV = os.environ.copy()
ENV["PYTHONPATH"] = os.pathsep.join([
    ROOT,
    os.path.join(ROOT, "apps", "api"),
    os.path.join(ROOT, "packages", "analytics_core", "src"),
    os.path.join(ROOT, "packages"),
])

cmd = [sys.executable, "-m", "unittest", "-v", "tests.independent_release.test_clean_room_acceptance"]
result = subprocess.run(cmd, cwd=ROOT, env=ENV, check=False)
if result.returncode != 0:
    raise SystemExit(result.returncode)

missing = [m for m in ("duckdb", "polars", "pyarrow") if importlib.util.find_spec(m) is None]
print("\nCLEAN-ROOM ACCEPTANCE RESULT")
print("engine-independent oracle: PASS")
if missing:
    print("controller E2E: BLOCKED (missing: " + ", ".join(missing) + ")")
    print("Install the project's cloud/runtime requirements before treating end-to-end controller acceptance as PASS.")
    raise SystemExit(2)

print("controller E2E prerequisite engines: PRESENT")
print("Run the real-controller benchmark scripts to complete E2E acceptance.")
