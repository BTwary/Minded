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
from collections import Counter
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
# Identifier-like values ("00123", "007"): a leading zero followed by another
# digit is never a meaningful number, it is a code (ZIP, SKU, order id).
# Coercing it to a number silently destroys the identifier.
_LEADING_ZERO_ID = re.compile(r"^[+-]?0\d")
# Short alphabetic codes ("US", "IN", "NA"): a column made only of these is a
# code column, so a literal "NA" in it is a category (Namibia, North America),
# not a missing value.
_SHORT_ALPHA_CODE = re.compile(r"^[A-Za-z]{1,3}$")
# Numeric sentinels that read_csv nulls at parse time. Kept as missing (existing
# behaviour) but now reported, so a legitimate 99999 never disappears silently.
_NUMERIC_SENTINELS = ("-999", "-9999", "99999")
# Slash/dot/dash dates with a 4-digit year, e.g. 31/01/2025 or 1.2.2025.
# (ISO yyyy-mm-dd is already recognised downstream and is left alone.)
_SLASH_DATE = re.compile(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})$")


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
    preamble_rows_skipped: int = 0
    ragged_rows_repaired: int = 0
    columns_whitespace_trimmed: Dict[str, int] = field(default_factory=dict)
    columns_parsed_dates: Dict[str, str] = field(default_factory=dict)
    identifier_columns_preserved: List[str] = field(default_factory=list)
    literal_na_preserved: List[str] = field(default_factory=list)
    sentinel_values_nulled: Dict[str, Dict[str, int]] = field(default_factory=dict)
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
            "preamble_rows_skipped": self.preamble_rows_skipped,
            "ragged_rows_repaired": self.ragged_rows_repaired,
            "columns_whitespace_trimmed": self.columns_whitespace_trimmed,
            "columns_parsed_dates": self.columns_parsed_dates,
            "identifier_columns_preserved": self.identifier_columns_preserved,
            "literal_na_preserved": self.literal_na_preserved,
            "sentinel_values_nulled": self.sentinel_values_nulled,
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
            report.duplicate_columns_deduped[deduped] = name
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

    # Identifier guard: "00123"/"01234" are codes, not numbers.
    if non_null.str.match(_LEADING_ZERO_ID).any():
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


def _score_delimiter(lines: List[str], delimiter: str) -> Tuple[int, float]:
    """(modal field count, share of lines having it) for one candidate delimiter."""
    widths = []
    for row in csv.reader(lines, delimiter=delimiter):
        if any(c.strip() for c in row):
            widths.append(len(row))
    if not widths:
        return 0, 0.0
    width, count = Counter(widths).most_common(1)[0]
    return width, count / len(widths)


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

        if not raw or not raw.strip():
            raise ValueError(f"'{name}' is empty (no bytes or only whitespace); nothing to ingest.")

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
        if ext in ("parquet", "json"):
            # Cells are still literal strings here (no parser has nulled them yet).
            df = self._protect_literal_na_strings(df, report)
        df = self._trim_string_cells(df, report)
        df = self._normalize_na_tokens(df, report)
        df = self._coerce_numeric_columns(df, report)
        df = self._parse_date_columns(df, report)

        all_null_mask = df.isna().all(axis=1)
        if all_null_mask.any():
            report.rows_all_null_dropped = int(all_null_mask.sum())
            df = df.loc[~all_null_mask].reset_index(drop=True)

        return df, report

    # ------------------------------------------------------------------ CSV
    def _read_csv_strict(self, text: str, delimiter: str) -> pd.DataFrame:
        return pd.read_csv(
            StringIO(text),
            sep=delimiter,
            engine="python",
            na_values=_EXTRA_NA_TOKENS,
            keep_default_na=True,
            skip_blank_lines=True,
        )

    @staticmethod
    def _validate_csv_quoting(text: str, delimiter: str) -> None:
        """Reject unterminated quoted fields instead of silently merging records."""
        try:
            reader = csv.reader(StringIO(text), delimiter=delimiter, strict=True)
            for _row in reader:
                pass
        except csv.Error as exc:
            raise ValueError(
                "Malformed delimited file: invalid or unterminated quoted field; "
                "correct the source file before analysis."
            ) from exc

    def _read_csv_raw(self, text: str, delimiter: str) -> pd.DataFrame:
        """Same parse, but every cell kept as the literal string in the file."""
        return pd.read_csv(
            StringIO(text),
            sep=delimiter,
            engine="python",
            dtype=str,
            keep_default_na=False,
            na_values=[],
            skip_blank_lines=True,
        )

    def _repair_delimited_text(self, text: str, delimiter: str, report: IngestionReport) -> Tuple[str, str]:
        """Rebuild a CSV that a strict parser rejects.

        Handles the two failures real exports cause: title/banner lines above
        the header ("Sales Report Q1", "Generated by ERP") and rows whose field
        count differs from the header. Nothing is invented: short rows are
        padded with empty cells, over-long rows keep their first N fields, and
        the counts are written to the report (the raw file keeps the original).
        """
        lines = text.splitlines()
        sample = lines[:300]
        # Re-pick the delimiter by structural consistency, not by the sniffer,
        # because banner lines confuse csv.Sniffer.
        best = (0.0, delimiter)
        for cand in _DELIMITER_CANDIDATES:
            width, share = _score_delimiter(sample, cand)
            if width > 1 and share * width > best[0]:
                best = (share * width, cand)
        delimiter = best[1] if best[0] > 0 else delimiter
        report.delimiter_detected = delimiter

        rows = [r for r in csv.reader(lines, delimiter=delimiter) if any(c.strip() for c in r)]
        if not rows:
            raise ValueError(f"'{report.source_filename}' contains no data rows.")
        widths = Counter(len(r) for r in rows[:300])
        modal = max(widths.items(), key=lambda kv: (kv[1], kv[0]))[0]
        header_idx = next(i for i, r in enumerate(rows) if len(r) == modal)

        report.preamble_rows_skipped = header_idx
        if header_idx:
            report.warnings.append(
                f"Skipped {header_idx} banner/title line(s) above the header row "
                f"(first: {rows[0][0][:60]!r})."
            )
        header, body = rows[header_idx], rows[header_idx + 1:]
        repaired, truncated, padded = [], 0, 0
        for r in body:
            if len(r) > modal:
                r = r[:modal]
                truncated += 1
            elif len(r) < modal:
                r = r + [""] * (modal - len(r))
                padded += 1
            repaired.append(r)
        report.ragged_rows_repaired = truncated + padded
        if truncated or padded:
            report.warnings.append(
                f"Repaired {truncated + padded} row(s) whose field count differed from the "
                f"{modal}-column header ({padded} padded, {truncated} truncated; extra fields dropped)."
            )
        buf = StringIO()
        writer = csv.writer(buf, delimiter=",", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(repaired)
        return buf.getvalue(), ","

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
        # Validate quote structure before permissive parser/repair paths can absorb
        # multiple physical rows into one field and return a plausible but corrupted table.
        self._validate_csv_quoting(text, delimiter)

        try:
            df = self._read_csv_strict(text, delimiter)
        except pd.errors.EmptyDataError:
            raise ValueError(f"'{report.source_filename}' has no parseable columns (empty or header-less).")
        except pd.errors.ParserError:
            text, delimiter = self._repair_delimited_text(text, delimiter, report)
            df = self._read_csv_strict(text, delimiter)

        raw_df = self._read_csv_raw(text, delimiter)

        # pandas mangles duplicate/blank headers ("a.1", "Unnamed: 3") before our
        # normaliser can see them; recover the literal header cells instead.
        header_cells = next(csv.reader(StringIO(text), delimiter=delimiter), [])
        if len(header_cells) == len(df.columns):
            df.columns = header_cells
        raw_df.columns = df.columns

        return self._apply_raw_overrides(df, raw_df, report)

    # ------------------------------------------------------------- overrides
    def _apply_raw_overrides(self, df: pd.DataFrame, raw_df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        """Use the literal file text to undo three silent, lossy parse decisions."""
        if raw_df.shape != df.shape:
            return df
        for pos in range(df.shape[1]):
            col = df.columns[pos]
            parsed = df.iloc[:, pos]
            literal = raw_df.iloc[:, pos].astype(object)
            stripped = literal.map(lambda v: v.strip() if isinstance(v, str) else v)

            # 1) Identifier columns: numeric parse stripped leading zeros.
            if pd.api.types.is_numeric_dtype(parsed) and stripped.astype(str).str.match(_LEADING_ZERO_ID).any():
                df.isetitem(pos, stripped.where(parsed.notna() & (stripped != ""), None))
                report.identifier_columns_preserved.append(str(col))
                continue

            # 2) Sentinel values nulled by parse-time na_values: keep behaviour, expose it.
            counts = {}
            for tok in _NUMERIC_SENTINELS:
                n = int(((stripped == tok) & parsed.isna()).sum())
                if n:
                    counts[tok] = n
            if counts:
                report.sentinel_values_nulled[str(col)] = counts
                report.warnings.append(
                    f"Column '{col}': treated sentinel value(s) {counts} as missing (value -> count)."
                )

            # 3) Literal "NA" in a code column is a category, not a missing value.
            is_text = parsed.dtype == object or str(parsed.dtype) in ("str", "string")
            if is_text:
                na_mask = parsed.isna() & (stripped == "NA")
                if na_mask.any():
                    others = parsed[parsed.notna()].astype(str).str.strip()
                    if others.nunique() >= 2 and others.str.match(_SHORT_ALPHA_CODE).all():
                        fixed = parsed.astype(object).copy()
                        fixed[na_mask] = "NA"
                        df.isetitem(pos, fixed)
                        report.literal_na_preserved.append(str(col))
                        report.warnings.append(
                            f"Column '{col}': kept {int(na_mask.sum())} literal 'NA' value(s) as a "
                            f"category (column looks like a code field), not as missing data."
                        )
        return df

    # ----------------------------------------------------------------- Excel
    def _load_excel(self, raw: bytes, report: IngestionReport) -> Tuple[pd.DataFrame, IngestionReport]:
        try:
            xls = pd.ExcelFile(BytesIO(raw))
        except ImportError as exc:
            raise ValueError(
                f"Cannot read '{report.source_filename}': {exc}. "
                "Legacy .xls needs `pip install xlrd`; otherwise re-save as .xlsx or .csv."
            ) from exc
        sheet_names = xls.sheet_names
        frames = {n: self._parse_sheet(xls, n) for n in sheet_names}

        chosen = sheet_names[0]
        first = frames[chosen]
        # A cover/notes sheet is not a data table: <5 rows and <3 columns. Only then,
        # and only if another sheet is clearly bigger, prefer that sheet.
        if len(sheet_names) > 1 and len(first) < 5 and first.shape[1] < 3:
            first_cells = max(first.shape[0] * first.shape[1], 1)
            biggest = max(sheet_names, key=lambda n: frames[n].shape[0] * frames[n].shape[1])
            if frames[biggest].shape[0] * frames[biggest].shape[1] > 4 * first_cells:
                report.warnings.append(
                    f"First sheet '{chosen}' looks like a cover/notes sheet "
                    f"({first.shape[0]} rows x {first.shape[1]} cols); using larger data sheet '{biggest}'."
                )
                chosen = biggest

        report.sheet_used = chosen
        df_chosen = frames[chosen]
        scratch = df_chosen.attrs.pop("_scratch_report", None)
        preamble = df_chosen.attrs.pop("_preamble_rows", 0)
        for other in frames.values():
            other.attrs.pop("_scratch_report", None)
            other.attrs.pop("_preamble_rows", None)
        if scratch is not None:
            report.identifier_columns_preserved.extend(scratch.identifier_columns_preserved)
            report.literal_na_preserved.extend(scratch.literal_na_preserved)
            report.sentinel_values_nulled.update(scratch.sentinel_values_nulled)
            report.warnings.extend(scratch.warnings)
        if preamble:
            report.preamble_rows_skipped = int(preamble)
            report.warnings.append(f"Skipped {int(preamble)} banner/title row(s) above the header in sheet '{chosen}'.")
        if len(sheet_names) > 1:
            report.other_sheets_ignored = [n for n in sheet_names if n != chosen]
            report.warnings.append(
                f"Workbook has {len(sheet_names)} sheets; using '{chosen}'. "
                f"Ignored: {', '.join(report.other_sheets_ignored)}."
            )
        return df_chosen, report

    def _parse_sheet(self, xls: "pd.ExcelFile", name: str) -> pd.DataFrame:
        df = xls.parse(name, na_values=_EXTRA_NA_TOKENS, keep_default_na=True)
        raw_df = xls.parse(name, dtype=str, keep_default_na=False, na_values=[])
        raw_df.columns = df.columns
        # Banner rows above the header make most column labels "Unnamed: n".
        unnamed = sum(str(c).startswith("Unnamed:") for c in df.columns)
        if df.shape[1] > 1 and unnamed >= max(2, df.shape[1] // 2):
            probe = xls.parse(name, header=None, na_values=_EXTRA_NA_TOKENS, keep_default_na=True)
            head = probe.head(30)
            counts = head.notna().sum(axis=1)
            if len(counts) and counts.max() > 1:
                header_idx = int(counts[counts == counts.max()].index[0])
                if header_idx > 0:
                    df = xls.parse(name, header=header_idx, na_values=_EXTRA_NA_TOKENS, keep_default_na=True)
                    raw_df = xls.parse(name, header=header_idx, dtype=str, keep_default_na=False, na_values=[])
                    raw_df.columns = df.columns
                    df.attrs["_preamble_rows"] = header_idx
        scratch = IngestionReport(source_filename=str(name), file_format="xlsx")
        df = self._apply_raw_overrides(df, raw_df, scratch)
        df.attrs["_scratch_report"] = scratch
        return df

    # ------------------------------------------------------------------ JSON
    def _load_json(self, raw: bytes, report: IngestionReport) -> pd.DataFrame:
        import json

        encoding, raw = _detect_encoding(raw)
        report.encoding_detected = encoding
        text = raw.decode(encoding, errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            records = []
            for lineno, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"'{report.source_filename}' is neither valid JSON nor JSON Lines "
                        f"(line {lineno}: {exc.msg})."
                    ) from exc
            if not records or not all(isinstance(r, dict) for r in records):
                raise ValueError(f"'{report.source_filename}': JSON Lines must be one object per line.")
            report.warnings.append(f"Parsed as JSON Lines ({len(records)} records).")
            return pd.json_normalize(records)

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

    # ------------------------------------------------------- cell clean-up
    @staticmethod
    def _text_columns(df: pd.DataFrame):
        return df.columns[df.dtypes.map(lambda d: d == object or str(d) in ("object", "str", "string"))]

    def _protect_literal_na_strings(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        """Record code columns whose literal "NA" must survive NA-token normalisation."""
        for col in self._text_columns(df):
            s = df[col]
            is_na = s.map(lambda v: isinstance(v, str) and v.strip() == "NA")
            if not is_na.any():
                continue
            others = s[s.notna() & ~is_na].astype(str).str.strip()
            if others.nunique() >= 2 and others.str.match(_SHORT_ALPHA_CODE).all():
                report.literal_na_preserved.append(str(col))
                report.warnings.append(
                    f"Column '{col}': kept {int(is_na.sum())} literal 'NA' value(s) as a "
                    f"category (column looks like a code field), not as missing data."
                )
        return df

    def _trim_string_cells(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        """Strip leading/trailing whitespace from text cells; whitespace-only -> missing.

        Without this "  us-east  " and "us-east" are different categories and every
        group-by silently splits one region into several. Lossless: the raw file is
        retained, and every affected column is counted in the report.
        """
        for col in self._text_columns(df):
            s = df[col]
            trimmed = s.map(lambda v: (v.strip() or None) if isinstance(v, str) else v)
            changed = int((s.map(lambda v: isinstance(v, str)) & (s.fillna("").astype(str) != trimmed.fillna("").astype(str))).sum())
            if changed:
                df[col] = trimmed
                report.columns_whitespace_trimmed[str(col)] = changed
        return df

    def _normalize_na_tokens(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        """Catch NA tokens with inconsistent casing/whitespace that na_values
        (exact match) missed, e.g. ' Unknown ', 'N/A '."""
        found_tokens = set()
        na_lower = {t.lower() for t in _EXTRA_NA_TOKENS}
        for col in self._text_columns(df):
            keep_na = str(col) in report.literal_na_preserved
            mask = df[col].map(
                lambda v: isinstance(v, str)
                and v.strip().lower() in na_lower
                and not (keep_na and v.strip().lower() == "na")
            )
            if mask.any():
                found_tokens.update(df.loc[mask, col].astype(str).str.strip().unique().tolist())
                df.loc[mask, col] = None
        if found_tokens:
            report.na_tokens_normalized = sorted(found_tokens)
        return df

    def _parse_date_columns(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        """Parse dd/mm/yyyy-style text columns, but only when day/month order is provable.

        If any first component exceeds 12 the order is day-first; if any second
        component does it is month-first. If neither does (every value <= 12) the
        column is ambiguous: it is left as text and a warning says why, because a
        wrong guess silently shifts every value into the wrong month.
        """
        for col in self._text_columns(df):
            non_null = df[col].dropna()
            if len(non_null) < 3 or not non_null.map(lambda v: isinstance(v, str)).all():
                continue
            m = non_null.astype(str).str.strip().str.extract(_SLASH_DATE)
            if m.isna().any(axis=None) or m[1].nunique() != 1:
                continue
            first, second = m[0].astype(int), m[2].astype(int)
            if first.max() > 12 and second.max() <= 12:
                order = "%d{s}%m{s}%Y"
            elif second.max() > 12 and first.max() <= 12:
                order = "%m{s}%d{s}%Y"
            elif first.max() <= 12 and second.max() <= 12:
                report.warnings.append(
                    f"Column '{col}' looks like a date but day/month order is ambiguous "
                    f"(all values <= 12); left as text. Re-export as ISO yyyy-mm-dd to analyse it over time."
                )
                continue
            else:
                continue
            fmt = order.format(s=m[1].iloc[0])
            parsed = pd.to_datetime(df[col].astype(str).str.strip().where(df[col].notna()), format=fmt, errors="coerce")
            if parsed.isna().sum() != df[col].isna().sum():
                continue
            df[col] = parsed
            report.columns_parsed_dates[str(col)] = fmt
        return df

    def _coerce_numeric_columns(self, df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
        for col in self._text_columns(df):
            coerced = _try_coerce_numeric_series(df[col])
            if coerced is not None:
                df[col] = coerced
                report.columns_coerced_numeric.append(col)
        return df
