"""Dataset Ingestion, Profiling, and Versioning Service."""
import io
import os
import uuid
import hashlib
import json
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session
from apps.api.src.models.entities import Dataset, DatasetVersion, DatasetTransformation
from apps.api.src.services.storage_service import StorageService
from packages.analytics_core.src.cleaning.engine import DataCleaningEngine
from packages.analytics_core.src.profiling.profiler import DataProfiler
from packages.analytics_core.src.data.content_identity import compute_content_hash
from packages.analytics_core.src.data.dataset_diff import DatasetDiff, compare_datasets
from packages.analytics_core.src.sql.engine import sanitize_sql_identifier
from packages.schemas.src.dataset import DatasetProfileSchema, DatasetResponse


class DatasetService:
    """Service managing dataset lifecycle, automated profiling, and versioned cleaning."""

    def __init__(self, db: Session, storage: Optional[StorageService] = None):
        self.db = db
        self.storage = storage or StorageService()
        self.profiler = DataProfiler()
        self.cleaner = DataCleaningEngine()

    def _unique_dataset_name(self, project_id: str, candidate_name: str) -> str:
        """Return a dataset name guaranteed not to collide with an existing
        one in this project once both are reduced to a SQL table identity.

        The SQL engine registers datasets by a sanitized version of their
        display name (see ``packages.analytics_core.src.sql.engine.
        sanitize_sql_identifier``). Two differently-spelled names can
        sanitize to the same identifier (e.g. "Sales Data" and
        "Sales-Data" both -> "Sales_Data"). If that were allowed to happen,
        the second dataset registered for a query would silently overwrite
        the first in the engine's table registry. Rather than only
        detecting that downstream at query time, prevent it from ever being
        created: if the candidate name's sanitized identity already exists
        in this project, append a numeric suffix until it doesn't.
        """
        existing_sanitized = {
            sanitize_sql_identifier(existing_name)
            for (existing_name,) in self.db.query(Dataset.name).filter(
                Dataset.project_id == project_id
            ).all()
        }
        if sanitize_sql_identifier(candidate_name) not in existing_sanitized:
            return candidate_name

        suffix = 2
        while True:
            attempt = f"{candidate_name} ({suffix})"
            if sanitize_sql_identifier(attempt) not in existing_sanitized:
                return attempt
            suffix += 1

    def ingest_dataset_file(
        self,
        project_id: str,
        filename: str,
        file_bytes: bytes,
        description: Optional[str] = None,
    ) -> Dataset:
        """Ingest raw dataset file, compute profile, convert to Parquet, and store version 1."""
        dataset_id = str(uuid.uuid4())
        name = self._unique_dataset_name(project_id, os.path.splitext(filename)[0])
        ext = os.path.splitext(filename)[1].lstrip(".").lower() or "csv"

        # 1. Save raw file and content-address the exact bytes so every
        # downstream profile/metric can be traced to the immutable source.
        source_sha256 = hashlib.sha256(file_bytes).hexdigest()
        raw_path = self.storage.save_raw_file(dataset_id, 1, filename, file_bytes)

        # 2. Load into DataFrame (robustly: handles real-world encoding,
        #    delimiter, NA-token, and numeric-formatting issues)
        df, ingestion_report = self.storage.load_dataframe_with_report(raw_path)

        # 3. Save as compressed Parquet
        parquet_path = self.storage.save_dataframe_as_parquet(dataset_id, 1, df)

        # 4. Profile dataset deterministically
        profile: DatasetProfileSchema = self.profiler.profile_dataframe(df, name, version=1)

        # 5. Create Dataset Record
        dataset = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=name,
            description=description or f"Dataset uploaded from {filename}",
            format=ext,
            current_version=1,
            row_count=len(df),
            column_count=len(df.columns),
            data_quality_score=profile.data_quality.overall_score,
            profile_json=profile.model_dump(mode="json"),
        )
        self.db.add(dataset)

        # 6. Create Dataset Version Record
        canonical_hash = compute_content_hash(
            df,
            source_metadata={
                "filename": filename,
                "extension": ext,
                "source_sha256": source_sha256,
            },
            transformation_lineage=[],
        )
        version_record = DatasetVersion(
            dataset_id=dataset_id,
            version_number=1,
            file_path=parquet_path,
            file_size_bytes=len(file_bytes),
            row_count=len(df),
            column_count=len(df.columns),
            content_hash=canonical_hash,
            transformation_applied={
                "action": "initial_upload",
                "filename": filename,
                "source_sha256": source_sha256,
                "raw_file_path": raw_path,
                "ingestion_report": ingestion_report.to_dict() if ingestion_report else None,
            },
        )
        self.db.add(version_record)
        self.db.flush()
        lineage_record = DatasetTransformation(
            dataset_id=dataset_id,
            input_version_id=None,
            output_version_id=version_record.id,
            operation="initial_upload",
            input_content_hash=None,
            output_content_hash=canonical_hash,
            parameters_json={"filename": filename, "extension": ext},
            affected_rows=len(df),
            affected_columns_json=[str(c) for c in df.columns],
            reason=description or "Initial dataset ingestion",
            source_record_json=version_record.transformation_applied or {},
            record_hash=hashlib.sha256(json.dumps(version_record.transformation_applied or {}, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest(),
        )
        self.db.add(lineage_record)
        self.db.commit()
        self.db.refresh(dataset)

        return dataset

    def clean_and_create_version(
        self,
        dataset_id: str,
        cleaning_recipe: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> DatasetVersion:
        """Apply non-destructive cleaning transformations and save as a new immutable version."""
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        # Load current version Parquet
        current_version_rec = (
            self.db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.version_number == dataset.current_version)
            .first()
        )
        if not current_version_rec:
            raise ValueError("Current version record missing.")

        current_df = self.storage.load_dataframe(current_version_rec.file_path)

        # Apply transformations non-destructively
        input_hash = current_version_rec.content_hash or compute_content_hash(current_df)
        cleaned_df, transformation_logs = self.cleaner.clean_dataset(current_df, cleaning_recipe)
        previous_meta = dict(current_version_rec.transformation_applied or {})
        source_metadata = {
            "filename": previous_meta.get("filename"),
            "extension": previous_meta.get("extension") or dataset.format,
            "source_sha256": previous_meta.get("source_sha256"),
        }
        prior_lineage = list((previous_meta.get("lineage_chain") or []))
        transformation_lineage = prior_lineage + [{
            "input_content_hash": input_hash,
            "operation": "clean_and_create_version",
            "recipe": cleaning_recipe,
            "logs": transformation_logs,
            "input_version": int(current_version_rec.version_number),
        }]
        output_hash = compute_content_hash(
            cleaned_df,
            source_metadata=source_metadata,
            transformation_lineage=transformation_lineage,
        )

        new_version_num = dataset.current_version + 1

        # Save new parquet
        new_parquet_path = self.storage.save_dataframe_as_parquet(dataset_id, new_version_num, cleaned_df)

        # Re-profile new version
        new_profile = self.profiler.profile_dataframe(cleaned_df, dataset.name, version=new_version_num)

        # Update dataset record
        dataset.current_version = new_version_num
        dataset.row_count = len(cleaned_df)
        dataset.column_count = len(cleaned_df.columns)
        dataset.data_quality_score = new_profile.data_quality.overall_score
        dataset.profile_json = new_profile.model_dump(mode="json")

        # Add new version record
        new_version_rec = DatasetVersion(
            dataset_id=dataset_id,
            version_number=new_version_num,
            file_path=new_parquet_path,
            file_size_bytes=os.path.getsize(self.storage._resolve_read_path(new_parquet_path)),
            row_count=len(cleaned_df),
            column_count=len(cleaned_df.columns),
            content_hash=output_hash,
            transformation_applied={
                "operation": "clean_and_create_version",
                "recipe": cleaning_recipe,
                "logs": transformation_logs,
                "input_content_hash": input_hash,
                "output_content_hash": output_hash,
                "lineage_chain": transformation_lineage,
            "lineage_hash_context": transformation_lineage,
                "filename": previous_meta.get("filename"),
                "extension": previous_meta.get("extension") or dataset.format,
                "source_sha256": previous_meta.get("source_sha256"),
                "input_version": int(current_version_rec.version_number),
                "output_version": int(new_version_num),
            },
        )
        self.db.add(new_version_rec)
        self.db.flush()
        lineage_payload = {
            "input_content_hash": input_hash,
            "output_content_hash": output_hash,
            "operation": "clean_and_create_version",
            "recipe": cleaning_recipe,
            "logs": transformation_logs,
            "input_version": int(current_version_rec.version_number),
            "output_version": int(new_version_num),
        }
        lineage_record = DatasetTransformation(
            dataset_id=dataset_id,
            input_version_id=current_version_rec.id,
            output_version_id=new_version_rec.id,
            operation="clean_and_create_version",
            input_content_hash=input_hash,
            output_content_hash=output_hash,
            parameters_json={"recipe": cleaning_recipe},
            affected_rows=abs(len(cleaned_df)-len(current_df)),
            affected_columns_json=[str(c) for c in cleaned_df.columns],
            reason=description or "Deterministic cleaning transformation",
            source_record_json=lineage_payload,
            record_hash=hashlib.sha256(__import__("json").dumps(lineage_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest(),
        )
        self.db.add(lineage_record)
        self.db.commit()
        self.db.refresh(new_version_rec)

        return new_version_rec

    def compare_dataset_versions(
        self,
        dataset_id: str,
        before_version: int,
        after_version: int,
        key_columns: Optional[List[str]] = None,
    ) -> DatasetDiff:
        """Run a deterministic forensic diff between two immutable dataset versions.

        This deliberately delegates all comparison semantics to ``DatasetDiff`` and
        performs no persistence or mutation.  Both versions must belong to the
        same dataset; the caller remains responsible for authorization.
        """
        if before_version == after_version:
            raise ValueError("before_version and after_version must be different")
        if before_version < 1 or after_version < 1:
            raise ValueError("dataset versions must be positive integers")

        versions = (
            self.db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version_number.in_([before_version, after_version]),
            )
            .all()
        )
        found = {int(v.version_number): v for v in versions}
        missing = [v for v in (before_version, after_version) if v not in found]
        if missing:
            raise ValueError(f"Dataset version(s) not found: {missing}")

        before_df = self.storage.load_dataframe(found[before_version].file_path)
        after_df = self.storage.load_dataframe(found[after_version].file_path)
        return compare_datasets(before_df, after_df, key_columns=key_columns)

    def get_dataset_dataframe(self, dataset_id: str, version: Optional[int] = None) -> pd.DataFrame:
        """Load DataFrame for a specific dataset version."""
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        target_ver = version or dataset.current_version
        version_rec = (
            self.db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.version_number == target_ver)
            .first()
        )
        if not version_rec:
            raise ValueError(f"Dataset version {target_ver} not found.")

        return self.storage.load_dataframe(version_rec.file_path)
