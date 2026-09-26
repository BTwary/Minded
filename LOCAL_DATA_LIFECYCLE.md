# AA-OS Local-First Data Lifecycle

AA-OS performs canonical analysis locally by default. DuckDB/Polars execute deterministic computations, SQLite stores lightweight investigation state/history, and source dataset files are retained locally for 180 days by default.

## Analysis modes

- `DETERMINISTIC`: the complete analytical investigation and its explanation run locally. No AI provider or network access is required.
- `AI_AUGMENTED`: the deterministic investigation still runs first. A configured AI provider may rewrite/augment the verified explanation for presentation only. It cannot replace the verdict, evidence, calculations, or epistemic strength. If AI is unavailable, AA-OS falls back to the local explanation.

## Retention

Raw/source dataset and derived file retention defaults to 180 days. Lightweight investigation history, evidence metadata, verdicts, provenance hashes, and explanation metadata are retained separately from source-file TTL.

After source-file expiry, AA-OS must not claim full reproducibility unless the user archived the source dataset or equivalent reproducibility artifact.
