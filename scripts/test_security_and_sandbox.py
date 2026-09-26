"""Executable Verification of Process-Isolated Python Sandbox, OS-Level Termination, and Filesystem I/O Defense."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath("."))

from apps.api.src.core.config import settings
from packages.analytics_core.src.sandbox.runner import (
    PythonSandboxRunner,
    SandboxSecurityError,
    SandboxTimeoutError,
)

runner = PythonSandboxRunner()

if not settings.CODE_EXECUTION_ENABLED:
    pytest.skip("Arbitrary Python sandbox execution is disabled by default; enable an OS/container sandbox for this suite.", allow_module_level=True)

print("=" * 60)
print("RUNNING SECURITY & PROCESS ISOLATION TEST SUITE")
print("=" * 60)

# 1. Normal safe calculation in isolated child process
print("\n[TEST 1] Safe execution in child process...")
safe_code = """
import numpy as np
import pandas as pd
arr = np.array([10.0, 20.0, 30.0, 40.0])
result = float(np.mean(arr))
"""
res = runner.execute_safe_python(safe_code, timeout_seconds=5)
assert res["status"] == "success", f"Safe execution failed: {res}"
assert res["result"] == 25.0, f"Expected 25.0, got {res['result']}"
print(f" -> PASSED: result = {res['result']} (Time: {res['execution_time_ms']}ms)")

# 2. Block forbidden AST imports
print("\n[TEST 2] Static AST forbidden import interception...")
forbidden_scripts = [
    "import os; os.system('echo hack')",
    "import subprocess",
    "import sys; sys.exit(0)",
    "open('test.txt', 'w')",
    "x = ().__class__.__bases__[0].__subclasses__()",
    "import urllib.request",
    "import socket",
]

for script in forbidden_scripts:
    caught = False
    try:
        runner.execute_safe_python(script, timeout_seconds=2)
    except SandboxSecurityError:
        caught = True
    assert caught, f"Security sandbox failed to intercept: {script}"
print(f" -> PASSED: Intercepted {len(forbidden_scripts)} unauthorized payloads via AST scanner.")

# 3. Block Third-Party Package Filesystem I/O Escapes (Pandas/NumPy read_csv/to_csv)
print("\n[TEST 3] Filesystem escape attempt interception on Pandas/NumPy...")
io_escape_scripts = [
    "import pandas as pd; df = pd.read_csv('C:/Windows/win.ini')",
    "import pandas as pd; df = pd.read_parquet('data/seed/sales.csv')",
    "import pandas as pd; df = pd.DataFrame({'a': [1]}); df.to_csv('C:/Windows/leak.txt')",
    "import pandas as pd; df = pd.DataFrame({'a': [1]}); df.to_parquet('leak.parquet')",
]

for script in io_escape_scripts:
    caught = False
    try:
        runner.execute_safe_python(script, timeout_seconds=2)
    except SandboxSecurityError:
        caught = True
    assert caught, f"Sandbox failed to intercept filesystem escape: {script}"
print(f" -> PASSED: Intercepted {len(io_escape_scripts)} third-party filesystem I/O escape attacks.")

# 4. Hard OS-level process termination on runaway loop
print("\n[TEST 4] Hard OS-level process termination on infinite loop (1s timeout)...")
infinite_code = """
x = 0
while True:
    x += 1
"""
t0 = time.time()
timed_out = False
try:
    runner.execute_safe_python(infinite_code, timeout_seconds=1)
except SandboxTimeoutError:
    timed_out = True
elapsed = time.time() - t0

assert timed_out, "Runaway process was not terminated on timeout"
assert elapsed < 2.5, f"Timeout took too long: {elapsed:.2f}s"
print(f" -> PASSED: Child process forcefully killed by OS in {elapsed:.2f}s.")

# 5. Assert production security defaults
print("\n[TEST 5] Production security defaults...")
assert settings.DEMO_MODE is False, f"DEMO_MODE default must be False, got {settings.DEMO_MODE}"
assert "*" not in settings.CORS_ORIGINS, f"CORS_ORIGINS must not contain wildcard by default: {settings.CORS_ORIGINS}"
assert len(settings.CORS_ORIGINS) >= 1, "CORS_ORIGINS must contain explicit origin whitelist"
print(" -> PASSED: Strict production configuration verified.")

print("\n" + "=" * 60)
print("ALL 5 SECURITY & ISOLATION TESTS PASSED WITH 100% INTEGRITY")
print("=" * 60)
