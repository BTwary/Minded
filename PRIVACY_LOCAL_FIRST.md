# Minded Local-First Privacy Contract

MindEd/AA-OS is designed to work fully on the user device without an account, cloud database, AI API, telemetry service, or paid subscription.

## Default behavior

- Analytical computations execute locally.
- SQLite and dataset storage default to a per-user application-data directory, not the source tree.
- AI is disabled (`AI_ENABLED=false`, provider `none`).
- Cloud storage is local (`STORAGE_PROVIDER=local`).
- Telemetry is disabled.
- Feedback is stored locally and is not uploaded automatically.
- Encrypted portable backups are local files by default.

## Optional services

AI providers, S3/GCS backup, and any future hosted service are explicit user choices. They are adapters, not scientific authorities. The deterministic analytical result remains authoritative.

## Recovery

The local backup/restore workflow is encrypted with AES-GCM using a user passphrase and PBKDF2-HMAC-SHA256 key derivation. Credentials and API keys are intentionally excluded from backup artifacts.

## Hosted demo

The Hugging Face Space is a showcase environment only. It should not be treated as the privacy-equivalent of the local application.
