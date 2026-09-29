# Optional services without friction

| Service | Default | Opt-in trigger | Scientific authority | Offline-safe |
|---|---|---|---|---|
| Deterministic engine | ON | None | **Authoritative** | Yes |
| Local Ollama | OFF | Choose local AI provider | Presentation/planning only | Yes (with model installed) |
| Cloud AI | OFF | Provider + user key | Presentation/planning only | No |
| Local encrypted backup | Available | Backup command/UI | N/A | Yes |
| S3/R2/MinIO backup | OFF | User bucket + credentials | N/A | No |
| GCS backup | OFF | User bucket + credentials | N/A | No |
| Telemetry | OFF | Explicit future opt-in | N/A | Yes |
| Feedback upload | OFF | Explicit opt-in | N/A | Yes |
| Hugging Face Space | Separate demo | Visit the demo | **Not authoritative** | No |

The core application must continue to ingest, calculate, verify, save, restore, and explain results when every optional service is disabled.

Cloud backup stores an encrypted `.aaosbackup`; credentials/API keys are excluded from the artifact. Failure to reach cloud storage must not delete the local copy or block deterministic analysis.
