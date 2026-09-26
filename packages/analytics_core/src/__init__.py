"""Analytics Core package.

Audit fix (Item 1 test-reproducibility defect): this package used to
eagerly `from .sql.engine import DuckDBSQLEngine` (and several other
heavy engines) at import time. Because Python always initializes a
parent package before any of its submodules, that meant importing
ANYTHING under `packages.analytics_core.src` -- including modules with
no relationship to SQL execution, such as
`intelligence.hypothesis_identity` or `runtime.state` -- transitively
required `duckdb` to be installed. In an environment without duckdb,
that made it impossible to even collect the Item-1 hypothesis-identity
and state-reconstruction tests, which don't touch SQL execution at
all: `ModuleNotFoundError: No module named 'duckdb'` was raised before
the tests could run.

Each top-level engine is now imported lazily, on first attribute
access, via PEP 562 module `__getattr__`. `from packages.analytics_core.src
import DuckDBSQLEngine` (or any other name below) still works exactly
as before -- it just defers the underlying import (and therefore the
`duckdb` dependency) until that name is actually touched, instead of
paying for it on every import of the package.
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for static type checkers / IDEs only
    from .sql.engine import DuckDBSQLEngine, SQLSecurityError
    from .profiling.profiler import DataProfiler
    from .cleaning.engine import DataCleaningEngine, TransformationLog
    from .statistics.engine import StatisticalEngine
    from .forecasting.engine import ForecastingEngine
    from .ml.engine import MachineLearningEngine
    from .anomaly.engine import AnomalyDetectionEngine
    from .validation.validator import IndependentValidator
    from .sandbox.runner import PythonSandboxRunner, SandboxSecurityError

__all__ = [
    "DuckDBSQLEngine",
    "SQLSecurityError",
    "DataProfiler",
    "DataCleaningEngine",
    "TransformationLog",
    "StatisticalEngine",
    "ForecastingEngine",
    "MachineLearningEngine",
    "AnomalyDetectionEngine",
    "IndependentValidator",
    "PythonSandboxRunner",
    "SandboxSecurityError",
]

# name -> (submodule path relative to this package, attribute name in that module)
_LAZY_ATTRS = {
    "DuckDBSQLEngine": (".sql.engine", "DuckDBSQLEngine"),
    "SQLSecurityError": (".sql.engine", "SQLSecurityError"),
    "DataProfiler": (".profiling.profiler", "DataProfiler"),
    "DataCleaningEngine": (".cleaning.engine", "DataCleaningEngine"),
    "TransformationLog": (".cleaning.engine", "TransformationLog"),
    "StatisticalEngine": (".statistics.engine", "StatisticalEngine"),
    "ForecastingEngine": (".forecasting.engine", "ForecastingEngine"),
    "MachineLearningEngine": (".ml.engine", "MachineLearningEngine"),
    "AnomalyDetectionEngine": (".anomaly.engine", "AnomalyDetectionEngine"),
    "IndependentValidator": (".validation.validator", "IndependentValidator"),
    "PythonSandboxRunner": (".sandbox.runner", "PythonSandboxRunner"),
    "SandboxSecurityError": (".sandbox.runner", "SandboxSecurityError"),
}


def __getattr__(name: str):
    try:
        module_path, attr_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache so repeated access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
