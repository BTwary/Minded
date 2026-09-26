"""Real-world Data Ingestion Layer.

A deterministic analyst never gets clean data on the first try. This module
replaces naive `pd.read_csv(path)` ingestion with a robust loader that
survives the kinds of messiness real files actually contain:

- Unknown/mixed encodings (UTF-8 BOM, Windows-1252, Latin-1)
- Unknown delimiters (comma, semicolon, tab, pipe)
- Inconsistent NA tokens ("N/A", "NULL", "--", "unknown", "n.a.", empty string)
- Whitespace-padded or duplicated column headers
- Numbers stored as strings with thousands separators or currency symbols
  ("$1,234.56", "1.234,56" European style, "12%")

Every decision the loader makes is recorded in an `IngestionReport` so the
result is auditable rather than a silent guess -- consistent with the
project's "show your work" principle: a human analyst should be able to ask
"why does this column look the way it does?" and get a real answer.

This module does not invent or discard data. Every transformation is either
lossless (renaming, whitespace stripping) or additive-and-reversible
(a coerced numeric column keeps its original string form recoverable from
the raw file already stored by StorageService.save_raw_file). If a column
cannot be safely coerced, it is left as-is and flagged as a warning rather
than guessed at.
"""
from dataclasses import dataclass, field
from io import BytesIO, StringIO
import csv
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Encodings to try, in order of likelihood for real-world business exports.
_ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

# Delimiters to consider when sniffing fails or is ambiguous.
_DELIMITER_CANDIDATES = [",", ";", "\t", "|"]

# Tokens real-world exports use for missing data, beyond pandas' defaults.
_EXTRA_NA_TOKENS = [
    "n/a", "na", "n.a.", "n.a", "null", "none", "unknown", "unk",
    "--", "-", "?", "missing", "not available", "not applicable", "#n/a",
    "#value!", "#ref!", "#div/0!", "#num!", "#name?", "nil", "nan",
    "-999", "-9999", "99999", "n/d",
]

_CURRENCY_CHARS = re.compile(r"[$€£¥₹₩]")
# Comma-thousands style, e.g. "1,234.56" or a plain "987.10" with no separator at all.
_COMMA_STYLE_NUMBER = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$")
# European style, e.g. "1.234,56" or a plain "987,10" with no separator at all.
_EURO_STYLE_NUMBER = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$")
_HAS_COMMA_GROUP = re.compile(r",\d{3}")
_HAS_DOT_GROUP = re.compile(r"\.\d{3}")
_PERCENT_SUFFIX = re.compile(r"^-?\d+(\.\d+)?%$")


@dataclass
class IngestionReport:
    """Records every decision made while loading a raw file, for provenance."""
    source_filename: str
    file_format: str
    encoding_detected: Optional[str] = None
    delimiter_detected: Optional[str] = None
    sheet_used: Optional[str] = None
    other_sheets_ignored: List[str] = field(default_factory=list)
    columns_renamed: Dict[str, str] = field(default_factory=dict)
    duplicate_columns_deduped: Dict[str, str] = field(default_factory=dict)
    columns_coerced_numeric: List[str] = field(default_factory=list)
    na_tokens_normalized: List[str] = field(default_factory=list)
    rows_all_null_dropped: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "file_format": self.file_format,
            "encoding_detected": self.encoding_detected,
            "delimiter_detected": self.delimiter_detected,
            "sheet_used": self.sheet_used,
            "other_sheets_ignored": self.other_sheets_ignored,
            "columns_renamed": self.columns_renamed,
            "duplicate_columns_deduped": self.duplicate_columns_deduped,
            "columns_coerced_numeric": self.columns_coerced_numeric,
            "na_tokens_normalized": self.na_tokens_normalized,
            "rows_all_null_dropped": self.rows_all_null_dropped,
            "warnings": self.warnings,
        }


def _detect_encoding(raw_bytes: bytes) -> Tuple[str, bytes]:
    """Try known encodings in order; return the first that decodes cleanly."""
    for enc in _ENCODING_CANDIDATES:
        try:
            raw_bytes.decode(enc)
            return enc, raw_bytes
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort: decode with replacement so ingestion never hard-crashes,
    # but this is always surfaced as a warning by the caller.
    return "latin-1", raw_bytes


def _sniff_delimiter(sample_text: str) -> str:
    """Use csv.Sniffer with a fallback to counting candidate delimiters."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(_DELIMITER_CANDIDATES))
        if dialect.delimiter in _DELIMITER_CANDIDATES:
            return dialect.delimiter
    except csv.Error:
        pass
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    counts = {d: first_line.count(d) for d in _DELIMITER_CANDIDATES}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _normalize_columns(columns: List[str], report: IngestionReport) -> List[str]:
    """Strip whitespace and dedupe duplicate/blank headers, losslessly."""
    cleaned = []
    for c in columns:
        original = c
        new_name = str(c).strip()
        if new_name == "":
            new_name = "unnamed_column"
        if new_name != original:
            report.columns_renamed[str(original)] = new_name
        cleaned.append(new_name)

    seen: Dict[str, int] = {}
    final = []
    for name in cleaned:
        if name in seen:
            seen[name] += 1
            deduped = f"{name}_{seen[name]}"
            report.duplicate_columns_deduped[name] = report.duplicate_columns_deduped.get(name, name)
            final.append(deduped)
        else:
            seen[name] = 0
            final.append(name)
    return final


def _parse_enterprise_num_single(v: Any) -> Optional[float]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    is_neg = False
    if s.startswith("(") and s.endswith(")"):
        is_neg = True
        s = s[1:-1].strip()
    s = _CURRENCY_CHARS.sub("", s).strip()
    if s.startswith("-"):
        is_neg = True
        s = s[1:].strip()
    elif s.endswith("-"):
        is_neg = True
        s = s[:-1].strip()
    s = _CURRENCY_CHARS.sub("", s).strip()
    if not s:
        return None
    is_pct = False
    if s.endswith("%"):
        is_pct = True
        s = s[:-1].strip()
    if bool(_COMMA_STYLE_NUMBER.match(s)) or bool(re.match(r"^\d+(\.\d+)?$", s)):
        try:
            val = float(s.replace(",", ""))
        except (ValueError, TypeError):
            return None
    elif bool(_EURO_STYLE_NUMBER.match(s)):
        try:
            val = float(s.replace(".", "").replace(",", "."))
        except (ValueError, TypeError):
            return None
    else:
        return None
    if is_pct:
        val /= 100.0
    return -val if is_neg else val


def _try_coerce_numeric_series(series: pd.Series) -> Optional[pd.Series]:
    """Attempt to coerce a string column with currency/thousands formatting
    or accounting parentheses to numeric. Returns None (do not coerce) unless
    coercion is unambiguous and lossless for the non-null values actually present."""
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    # Verify that all non-null values unambiguously parse to numeric
    for val in non_null:
        if _parse_enterprise_num_single(val) is None:
            return None

    try:
        return series.apply(
            lambda v: _parse_enterprise_num_single(v) if pd.notna(v) else v
        )
    except (ValueError, TypeError):
        return None


class RobustFileLoader:
    """Deterministic, provenance-tracked loader for real-world CSV/Excel/JSON/Parquet files."""

    def __init__(self, sample_bytes_for_sniffing: int = 65536):
        self.sample_bytes_for_sniffing = sample_bytes_for_sniffing

    def load(self, file_path: Optional[str] = None, file_bytes: Optional[bytes] = None,
              filename: Optional[str] = None) -> Tuple[pd.DataFrame, IngestionReport]:
        """Load a file from disk (file_path) or from raw bytes already in memory
        (file_bytes + filename). Returns (dataframe, ingestion_report)."""
        if file_path is not None:
            with open(file_path, "rb") as f:
                raw = f.read()
            name = filename or file_path
        elif file_bytes is not None:
            raw = file_bytes
            name = filename or "uploaded_file"
        else:
            raise ValueError("Either file_path or file_bytes must be provided.")

        ext = (name.rsplit(".", 1)[-1].lower() if "." in name else "csv")
        report = IngestionReport(source_filename=name, file_format=ext)

        if ext == "parquet":
            df = pd.read_parquet(BytesIO(raw))
        elif ext in ("xlsx", "xls"):
            df, report = self._load_excel(raw, report)
        elif ext == "json":
            df = self._load_json(raw, report)
        else:
            df = self._load_delimited(raw, report)

        df.columns = _normalize_columns(list(df.columns), report)
        df = self._normalize_na_tokens(df, report)
        df = self._coerce_numeric_columns(df, report)

        all_null_mask = df.isna().all(axis=1)
        if all_null_mask.any():
            report.rows_all_null_dropped = int(all_null_mask.sum())
            df = df.loc[~all_null_mask].reset_index(drop=True)

        return df, report

    def _load_delimited(self, raw: bytes, report: IngestionReport) -> pd.DataFrame:
        encoding, raw = _detect_encoding(raw)
        report.encoding_detected = encoding
        if encoding != "utf-8-sig" and encoding != "utf-8":
            report.warnings.append(
                f"File was not valid UTF-8; decoded as '{encoding}'. Verify special characters."
            )
        text = raw.decode(encoding, errors="replace")

        sample = text[: self.sample_bytes_for_sniffing]
        delimiter = _sniff_delimiter(sample)
        report.delimiter_detected = delimiter

        return pd.read_csv(
            StringIO(text),
            sep=delimiter,
            engine="python",
            na_values=_EXTRA_NA_TOKENS,
            keep_default_na=True,
            skip_blank_lines=True,
        )

    def _load_excel(self, raw: bytes, report: IngestionReport) -> Tuple[pd.DataFrame, IngestionReport]:
        xls = pd.ExcelFile(BytesIO(raw))
        sheet_names = xls.sheet_names
        chosen = sheet_names[0]
        report.sheet_used = chosen
        if len(sheet_names) > 1:
            report.other_sheets_ignored = sheet_names[1:]
            report.warnings.append(
                f"Workbook has {len(sheet_names)} sheets; using '{chosen}'. "
                f"Ignored: {', '.join(sheet_names[1:])}."
            )
        df = xls.parse(chosen, na_values=_EXTRA_NA_TOKENS, keep_default_na=True)
        return df, report

    def _load_json(self, raw: bytes, report: IngestionReport) -> pd.DataFrame:
        import json

        encoding, raw = _detect_encoding(raw)
        report.encoding_detected = encoding
        text = raw.decode(encoding, errors="replace")
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return pd.json_normalize(parsed)

        if isinstance(parsed, dict):
            for key in ("data", "records", "rows", "results", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    report.warnings.append(f"Top-level JSON object; used nested key '{key}' as records.")
                    return pd.json_normalize(parsed[key])
            report.warnings.append("Top-level JSON object had no obvious records list; normalized whole object.")
            return pd.json_normalize(parsed)

        raise ValueError(f"Unsupported JSON top-level type: {type(parsed).__name__}")

    def _normalize_na_tokens(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        """Catch NA tokens with inconsistent casing/whitespace that na_values
        (exact match) missed, e.g. ' Unknown ', 'N/A '."""
        found_tokens = set()
        na_lower = {t.lower() for t in _EXTRA_NA_TOKENS}
        for col in df.columns[df.dtypes.map(lambda d: d == object or str(d) in ("object", "str"))]:
            mask = df[col].map(
                lambda v: isinstance(v, str) and v.strip().lower() in na_lower
            )
            if mask.any():
                found_tokens.update(df.loc[mask, col].astype(str).str.strip().unique().tolist())
                df.loc[mask, col] = None
        if found_tokens:
            report.na_tokens_normalized = sorted(found_tokens)
        return df

    def _coerce_numeric_columns(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        for col in df.columns[df.dtypes.map(lambda d: d == object or str(d) in ("object", "str"))]:
            coerced = _try_coerce_numeric_series(df[col])
            if coerced is not None:
                df[col] = coerced
                report.columns_coerced_numeric.append(col)
        return df
