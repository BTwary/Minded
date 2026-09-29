# AA-OS Optional Cloud Backup & Migration

AA-OS is local-first. Cloud is an explicit **Bring Your Own Infrastructure (BYOI)** option and is never required for analytical execution, AI, explanation, or local storage.

## What the cloud option does

A user can create an encrypted `.aaosbackup` continuity bundle containing:

- the logical AA-OS database state (current schema rows and investigation history),
- locally retained source/derived files when the user enables **Include source files**,
- calculation/evidence/provenance history required to understand previous results,
- a versioned manifest and integrity checksums.

The bundle is encrypted before upload using a user-supplied passphrase. AA-OS does not store the passphrase.

Cloud provider credentials are not placed in backups. Environment files, AI API keys, and cloud secrets are explicitly excluded from the portable bundle.

## Supported provider model

The initial adapters support:

- **S3 / S3-compatible**: AWS S3, Cloudflare R2, MinIO and compatible endpoints.
- **Google Cloud Storage** using the normal Google Application Default Credentials mechanism.

The application does not resell, proxy, or subsidize storage. The user owns the bucket/account and directly bears storage, transfer, and provider charges.

Optional dependencies live in `requirements-cloud.txt`; the default local installation does not need them.

## Reinstall and migration

After software deletion/reinstallation, the user can:

1. reconnect their cloud provider,
2. select an AA-OS backup,
3. supply the backup passphrase,
4. restore the database and retained source files,
5. continue using the migrated investigation history and calculation traces.

A downloaded `.aaosbackup` file can also be recovered locally without contacting a cloud provider. The first-run recovery route is restricted to loopback requests by default.

Restoration is logical rather than a blind database-file copy. Rows are mapped to the current schema by matching available columns. This allows additive schema changes to migrate forward, while current migrations should be applied before restore.

## Security invariants

Cloud backup must remain optional and user-controlled.

AA-OS must never:

- silently upload user data,
- require a cloud account to run analyses,
- store the backup passphrase,
- put provider secrets inside a backup,
- claim a cloud backup exists until its upload succeeds,
- restore a backup without integrity validation and passphrase verification.

The portable bundle is intentionally self-contained so it can be moved between devices and providers.

## Retention interaction

Local raw/derived source files still follow the configured six-month default retention policy. A backup preserves files that exist at the time the user creates the backup, but it does not retroactively preserve files that local retention has already deleted.

Analytical metadata and calculation/evidence history remain separate from raw-file retention. After a source expires locally, its historical investigation can remain visible but should identify that the original source is no longer locally reproducible unless a surviving backup/archive exists.

## Active storage migration

The dataset storage service now uses the same provider abstraction for new uploads and new dataset versions. Setting the explicit storage provider to `s3` or `gcs` makes those logical dataset objects cloud-backed while the analytical engines continue to obtain a local readable cache through the provider.

Existing database rows that contain absolute local paths remain readable, so enabling cloud storage does not invalidate previous local investigations.

### Existing local data migration

`Migrate local storage` copies existing files under the AA-OS managed local storage directory to the selected user-owned provider and rewrites existing `DatasetVersion` file references after successful cloud-object verification. **Local copies are retained**; AA-OS never treats migration as permission to erase the only copy.
