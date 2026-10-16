"""Serialization helpers used by the investigation controller."""
from typing import Any
import numpy as np
import pandas as pd


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif isinstance(obj, (bool, np.bool_)):
        # Must precede the int check: bool is a subclass of int, so checking int first silently
        # turned every stored True/False into 1/0 (e.g. "is_additive": 0 instead of false).
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif pd.isna(obj):
        return None
    return str(obj) if type(obj).__module__ != "builtins" else obj
