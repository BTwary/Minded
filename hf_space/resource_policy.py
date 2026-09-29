"""Hosted resource and input policy for the public Hugging Face Space."""
from pathlib import Path

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 500_000
MAX_COLUMNS = 250
MAX_CELL_CHARS = 10_000
MAX_DATASETS = 4
MAX_QUESTION_CHARS = 4_000


def validate_file(path: str) -> None:
    p = Path(path)
    if p.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Only CSV and Parquet files are supported.")
    if p.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError(f"Hosted upload limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")


def validate_dataframe(df) -> None:
    if len(df) > MAX_ROWS:
        raise ValueError(f"Hosted execution is limited to {MAX_ROWS:,} rows per dataset.")
    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"Hosted execution is limited to {MAX_COLUMNS} columns per dataset.")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if (df[col].dropna().astype(str).str.len().max() if not df[col].dropna().empty else 0) > MAX_CELL_CHARS:
            raise ValueError(f"Column '{col}' contains an oversized text cell.")
