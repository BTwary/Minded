"""Process-Isolated Python Sandbox Execution with OS-Level Subprocess Termination, AST Safety, and Filesystem/Network Disarming."""
import ast
import json
import subprocess
import sys
from typing import Any, Dict, Optional

try:
    from apps.api.src.core.config import settings
except Exception:
    settings = None


class SandboxSecurityError(Exception):
    pass


class SandboxTimeoutError(Exception):
    pass


class PythonSandboxRunner:
    """Secure process-isolated execution environment for AI-generated Python analytics scripts."""

    # Disallowed syntax tokens
    FORBIDDEN_KEYWORDS = [
        "import os", "import subprocess", "import socket",
        "import shutil", "import ctypes", "import urllib",
        "import requests", "import http", "open(",
        "exec(", "eval(", "getattr(", "setattr(",
        "globals()", "locals()", "os.", "subprocess.",
    ]

    def _verify_ast_safety(self, code_snippet: str) -> None:
        """Statically inspect Python AST for unsafe nodes, unauthorized module imports, and dangerous I/O methods."""
        try:
            tree = ast.parse(code_snippet)
        except SyntaxError as se:
            raise SandboxSecurityError(f"Syntax Error in analytical script: {str(se)}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ("math", "numpy", "np", "pandas", "pd", "scipy"):
                        raise SandboxSecurityError(f"Unauthorized module import: '{alias.name}'.")
            elif isinstance(node, ast.ImportFrom):
                if node.module not in ("math", "numpy", "pandas", "scipy"):
                    raise SandboxSecurityError(f"Unauthorized module import: '{node.module}'.")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise SandboxSecurityError(f"Dunder attribute access forbidden: '{node.attr}'.")
                # Block known filesystem escape methods on pandas/numpy
                if node.attr in (
                    "read_csv", "read_table", "read_excel", "read_parquet",
                    "read_json", "read_sql", "read_pickle", "read_clipboard",
                    "to_csv", "to_excel", "to_parquet", "to_json", "to_sql",
                    "to_pickle", "to_clipboard", "save", "tofile",
                    "memmap", "fromfile", "load", "loads", "savez", "savez_compressed",
                    "loadtxt", "savetxt", "genfromtxt", "fromfile", "open_memmap"
                ):
                    raise SandboxSecurityError(
                        f"Filesystem I/O method '{node.attr}' is forbidden inside the analytics sandbox."
                    )

    def execute_safe_python(
        self,
        code_snippet: str,
        input_data: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 5,
    ) -> Dict[str, Any]:
        """Execute optional generated Python only when explicitly enabled.

        This runner is not a complete security boundary on a general-purpose host.
        Therefore arbitrary Python execution is disabled by default. Production use
        must put this feature behind a genuine container/OS sandbox before enabling it.
        """
        if settings is not None and not settings.CODE_EXECUTION_ENABLED:
            raise SandboxSecurityError(
                "Arbitrary Python execution is disabled by default. "
                "Enable only behind an OS/container sandbox with AAOS_CODE_EXECUTION_ENABLED=true."
            )
        # 1. Textual security scan
        for forbidden in self.FORBIDDEN_KEYWORDS:
            if forbidden in code_snippet:
                raise SandboxSecurityError(f"Sandbox Violation: Code contains forbidden statement '{forbidden}'.")

        # 2. Static AST safety inspection
        self._verify_ast_safety(code_snippet)

        dangerous_calls = {
            "memmap", "fromfile", "load", "save", "savez", "savez_compressed",
            "loadtxt", "savetxt", "genfromtxt", "open_memmap", "loadmat", "savemat",
            "mmread", "mmwrite", "read_matrix", "write_matrix"
        }
        for node in ast.walk(ast.parse(code_snippet)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in dangerous_calls:
                raise SandboxSecurityError(f"Potential filesystem-capable numerical API '{node.func.attr}' is forbidden.")

        # 3. Formulate isolated worker wrapper script with monkeypatched/disarmed I/O
        worker_code = f"""
import sys, json, io, time
import numpy as np
import pandas as pd
import math, scipy

# Disarm filesystem and network methods in third-party numerical packages
def _forbidden_io(*args, **kwargs):
    raise PermissionError("Direct host filesystem and network I/O is strictly disabled in sandbox.")

for method_name in ("read_csv", "read_table", "read_excel", "read_parquet", "read_json", "read_sql", "read_pickle"):
    if hasattr(pd, method_name):
        setattr(pd, method_name, _forbidden_io)

for method_name in ("to_csv", "to_excel", "to_parquet", "to_json", "to_sql", "to_pickle"):
    if hasattr(pd.DataFrame, method_name):
        setattr(pd.DataFrame, method_name, _forbidden_io)

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in ("math", "numpy", "np", "pandas", "pd", "scipy"):
        raise ImportError(f"Unauthorized module import: '{{name}}'.")
    return __builtins__.__import__(name, globals, locals, fromlist, level) if hasattr(__builtins__, '__import__') else __import__(name, globals, locals, fromlist, level)

safe_builtins = {{
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "print": print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "__import__": _safe_import,
}}

sandbox_globals = {{
    "__builtins__": safe_builtins,
    "np": np, "numpy": np, "pd": pd, "pandas": pd, "math": math, "scipy": scipy
}}

code = {json.dumps(code_snippet)}
stdout_capture = io.StringIO()
old_stdout = sys.stdout

try:
    sys.stdout = stdout_capture
    t0 = time.time()
    exec(code, sandbox_globals)
    elapsed_ms = int((time.time() - t0) * 1000)
    sys.stdout = old_stdout
    res_val = sandbox_globals.get("result", None)
    if isinstance(res_val, pd.DataFrame):
        res_val = res_val.head(50).to_dict(orient="records")
    elif isinstance(res_val, np.ndarray):
        res_val = res_val.tolist()
    elif isinstance(res_val, (np.int64, np.int32)):
        res_val = int(res_val)
    elif isinstance(res_val, (np.float64, np.float32)):
        res_val = float(res_val)
    out = {{"status": "success", "result": res_val, "stdout": stdout_capture.getvalue(), "execution_time_ms": elapsed_ms}}
    print("__JSON_START__" + json.dumps(out) + "__JSON_END__")
except Exception as e:
    sys.stdout = old_stdout
    out = {{"status": "error", "error": str(e), "stdout": stdout_capture.getvalue(), "execution_time_ms": 0}}
    print("__JSON_START__" + json.dumps(out) + "__JSON_END__")
"""

        # 4. Launch isolated OS subprocess
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            # Forceful hard OS termination of child process
            proc.kill()
            proc.communicate()
            raise SandboxTimeoutError(
                f"Process Execution Timed Out: Computation exceeded {timeout_seconds}s limit and was terminated by OS."
            )

        if "__JSON_START__" in stdout_data and "__JSON_END__" in stdout_data:
            json_str = stdout_data.split("__JSON_START__")[1].split("__JSON_END__")[0]
            try:
                res_obj = json.loads(json_str)
                if res_obj.get("status") == "error":
                    raise RuntimeError(f"Sandbox runtime error: {res_obj.get('error')}")
                return res_obj
            except json.JSONDecodeError:
                raise RuntimeError(f"Failed to parse sandbox worker output: {stdout_data}")
        else:
            raise RuntimeError(f"Sandbox process failed or terminated abnormally: {stderr_data or stdout_data}")
