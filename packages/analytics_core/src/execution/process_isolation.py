"""Process-level execution isolation for optional untrusted/heavy work.

Security Hardening:
- ZERO Python pickle (.pkl) files.
- Uses Apache Arrow IPC (.arrow) for DataFrames.
- Uses JSON (.json) for function routing, scalars, and metadata.
- Isolated IPC storage placed in application-scoped data root, avoiding global %TEMP%
  and eliminating Windows Defender / EDR dropper heuristics and accidental sample submissions.
"""
from __future__ import annotations

import base64
import datetime
from decimal import Decimal
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ProcessExecutionResult:
    ok: bool
    result: Any = None
    error: str | None = None
    timed_out: bool = False


_WORKER_MODULE = "packages.analytics_core.src.execution._isolated_worker"


def serialize_ipc(val: Any, dir_path: Path, prefix: str) -> Any:
    """Serialize values safely without Python pickle.
    
    Tabular structures (DataFrames) are saved as Apache Arrow (.arrow) files.
    Primitives and collections are returned as JSON-compatible primitives.
    """
    if isinstance(val, pd.DataFrame):
        fname = f"{prefix}.arrow"
        val.to_feather(dir_path / fname)
        return {"__type__": "dataframe_arrow", "__file__": fname}
    if isinstance(val, pd.Series):
        fname = f"{prefix}.arrow"
        val.to_frame().to_feather(dir_path / fname)
        return {"__type__": "series_arrow", "__file__": fname, "__name__": val.name}
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, (np.integer, np.floating, np.bool_)):
        return val.item()
    if isinstance(val, (datetime.datetime, datetime.date)):
        return {
            "__type__": "datetime" if isinstance(val, datetime.datetime) else "date",
            "__iso__": val.isoformat(),
        }
    if isinstance(val, Decimal):
        return {"__type__": "decimal", "__val__": str(val)}
    if isinstance(val, uuid.UUID):
        return {"__type__": "uuid", "__val__": str(val)}
    if isinstance(val, bytes):
        return {"__type__": "bytes", "__b64__": base64.b64encode(val).decode("ascii")}
    if isinstance(val, tuple):
        return {
            "__type__": "tuple",
            "__items__": [serialize_ipc(x, dir_path, f"{prefix}_{i}") for i, x in enumerate(val)],
        }
    if isinstance(val, set):
        return {
            "__type__": "set",
            "__items__": [serialize_ipc(x, dir_path, f"{prefix}_{i}") for i, x in enumerate(sorted(val, key=str))],
        }
    if isinstance(val, list):
        return [serialize_ipc(x, dir_path, f"{prefix}_{i}") for i, x in enumerate(val)]
    if isinstance(val, dict):
        return {k: serialize_ipc(v, dir_path, f"{prefix}_{k}") for k, v in val.items()}
    raise TypeError(f"Unsupported type for secure IPC serialization: {type(val)!r}")


def deserialize_ipc(val: Any, dir_path: Path) -> Any:
    """Deserialize values safely from Arrow files and JSON-compatible trees."""
    if isinstance(val, dict):
        t = val.get("__type__")
        if t == "dataframe_arrow":
            return pd.read_feather(dir_path / val["__file__"])
        if t == "series_arrow":
            df = pd.read_feather(dir_path / val["__file__"])
            series = df.iloc[:, 0]
            series.name = val.get("__name__")
            return series
        if t == "tuple":
            return tuple(deserialize_ipc(x, dir_path) for x in val["__items__"])
        if t == "set":
            return set(deserialize_ipc(x, dir_path) for x in val["__items__"])
        if t == "datetime":
            return datetime.datetime.fromisoformat(val["__iso__"])
        if t == "date":
            return datetime.date.fromisoformat(val["__iso__"])
        if t == "decimal":
            return Decimal(val["__val__"])
        if t == "uuid":
            return uuid.UUID(val["__val__"])
        if t == "bytes":
            return base64.b64decode(val["__b64__"].encode("ascii"))
        return {k: deserialize_ipc(v, dir_path) for k, v in val.items()}
    if isinstance(val, list):
        return [deserialize_ipc(x, dir_path) for x in val]
    return val


def resolve_callable(module_name: str, qualname: str) -> Callable[..., Any]:
    """Dynamically resolve an existing function in the codebase without code deserialization."""
    mod = importlib.import_module(module_name)
    target: Any = mod
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def _get_ipc_directory() -> Path | None:
    try:
        from packages.analytics_core.src.platform_local_first import ensure_local_layout
        ipc_dir = ensure_local_layout()["root"] / "ipc"
        ipc_dir.mkdir(parents=True, exist_ok=True)
        return ipc_dir
    except Exception:
        return None


def run_isolated(fn: Callable[..., Any], *args, timeout_seconds: int = 60, **kwargs) -> ProcessExecutionResult:
    """Execute fn in a dedicated child process with hard timeout cancellation and secure IPC.
    
    Zero pickle files are created. DataFrames are transferred via Apache Arrow IPC,
    and metadata is transferred via JSON.
    """
    ipc_base = _get_ipc_directory()
    with tempfile.TemporaryDirectory(dir=str(ipc_base) if ipc_base else None, prefix="aaos_ipc_") as tmpdir:
        dir_path = Path(tmpdir)
        request_path = dir_path / "request.json"
        response_path = dir_path / "response.json"
        
        try:
            fn_module = getattr(fn, "__module__", None)
            fn_qualname = getattr(fn, "__qualname__", None)
            if not fn_module or not fn_qualname:
                return ProcessExecutionResult(False, error=f"Target callable {fn!r} lacks __module__ or __qualname__")
                
            serialized_args = serialize_ipc(args, dir_path, "arg")
            serialized_kwargs = serialize_ipc(kwargs, dir_path, "kwarg")
            
            payload = {
                "module": fn_module,
                "qualname": fn_qualname,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }
            request_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            return ProcessExecutionResult(False, error=f"Failed to serialize isolated call: {exc!r}")

        child_env = dict(os.environ)
        existing = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + ([existing] if existing else []))
        
        try:
            completed = subprocess.run(
                [sys.executable, "-m", _WORKER_MODULE, str(request_path), str(response_path)],
                timeout=timeout_seconds,
                capture_output=True,
                env=child_env,
            )
        except subprocess.TimeoutExpired:
            return ProcessExecutionResult(
                False,
                error=f"Process exceeded {timeout_seconds}s and was terminated.",
                timed_out=True,
            )
            
        if not response_path.exists():
            stderr = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""
            return ProcessExecutionResult(
                False,
                error=f"Worker exited with code {completed.returncode} without a result. stderr={stderr[-2000:]}",
            )
            
        try:
            resp_payload = json.loads(response_path.read_text(encoding="utf-8"))
            ok = resp_payload.get("ok", False)
            raw_result = resp_payload.get("result")
            error = resp_payload.get("error")
            result = deserialize_ipc(raw_result, dir_path) if ok else None
        except Exception as exc:
            return ProcessExecutionResult(False, error=f"Failed to deserialize isolated result: {exc!r}")
            
        return ProcessExecutionResult(bool(ok), result, error)

