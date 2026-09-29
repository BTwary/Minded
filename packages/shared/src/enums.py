"""Shared Enums and Constants for Autonomous AI Data Analyst Platform."""
from enum import Enum


class UserRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class DatasetFormat(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    PARQUET = "parquet"


class ColumnDataType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class SemanticType(str, Enum):
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    METRIC = "metric"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    TEXT = "text"
    GEOGRAPHIC = "geographic"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"


class ConfidenceLevel(str, Enum):
    HIGH = "High confidence"
    MEDIUM = "Medium confidence"
    LOW = "Low confidence"
    INSUFFICIENT_EVIDENCE = "Insufficient evidence"


class ValidationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    # P0 epistemic-integrity fix (verification-status contract): reserved for
    # a genuinely unrecognized internal verification value. SKIPPED means
    # "verification was legitimately not performed"; UNKNOWN means "the
    # internal status string didn't match any known vocabulary," which is a
    # data-integrity signal and must never be silently collapsed into
    # SKIPPED. See packages.analytics_core.src.execution.state_machine
    # .to_public_validation_status, the single authoritative mapping.
    UNKNOWN = "UNKNOWN"


class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    DISCOVERING = "DISCOVERING"
    AUDITING = "AUDITING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REPLANNING = "REPLANNING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AnalyticalVerdict(str, Enum):
    OBSERVED = "OBSERVED"
    DIAGNOSED = "DIAGNOSED"
    STATISTICALLY_SIGNIFICANT = "STATISTICALLY_SIGNIFICANT"
    NO_DETECTABLE_EFFECT = "NO_DETECTABLE_EFFECT"
    PREDICTED = "PREDICTED"
    SIMULATED = "SIMULATED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    # A requested analytical variable could not be resolved to any physical
    # column in the selected dataset. Scientifically distinct from
    # INSUFFICIENT_DATA (the analysis was identified and attempted but the
    # sample can't support it) and from INCONCLUSIVE (the analysis ran but
    # didn't discriminate between hypotheses): here no experiment is run at
    # all, because the question named a variable that does not exist.
    VARIABLE_NOT_FOUND = "VARIABLE_NOT_FOUND"


class CausalEvidenceLevel(str, Enum):
    DIRECT = "DIRECT"
    PROXY = "PROXY"
    OBSERVATIONAL = "OBSERVATIONAL"
    UNSUPPORTED = "UNSUPPORTED"



class AgentType(str, Enum):
    PLANNER = "planner"
    DATA = "data"
    SQL = "sql"
    STATISTICS = "statistics"
    FORECASTING = "forecasting"
    ML = "ml"
    VISUALIZATION = "visualization"
    BUSINESS_ANALYST = "business_analyst"
    REPORT = "report"
    VALIDATION = "validation"


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    PIE = "pie"
    BOX = "box"
    METRIC_CARD = "metric_card"
    TABLE = "table"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScheduleFrequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
