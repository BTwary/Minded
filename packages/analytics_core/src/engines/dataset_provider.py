"""DatasetProvider: Decoupled dataset acquisition, snapshotting, and cryptographic fingerprinting."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from apps.api.src.models.entities import Dataset, DatasetVersion
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.data.content_identity import compute_content_hash


@dataclass(frozen=True)
class InvestigationDataContext:
    """Immutable snapshot of project datasets and their cryptographic fingerprints."""
    project_id: str
    datasets_map: Dict[str, pd.DataFrame]
    dataset_fingerprints: Dict[str, str]
    dataset_versions: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # DEFECT-003 (Section 22): explicit record of the scoping decision that
    # produced this context, so callers/provenance can see WHY a dataset is
    # or isn't present -- rather than silently returning "all project
    # datasets" and leaving no trace of whether a selection was requested.
    requested_dataset_ids: Optional[List[str]] = None
    unavailable_requested_ids: List[str] = field(default_factory=list)
    load_errors: Dict[str, str] = field(default_factory=dict)
    scope_errors: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.datasets_map) == 0


class BaseDatasetProvider(ABC):
    """Abstract dataset provider interface."""

    @abstractmethod
    def acquire_context(self, project_id: str, dataset_ids: Optional[List[str]] = None) -> InvestigationDataContext:
        """Acquire immutable dataset snapshot context for an investigation.

        ``dataset_ids``, when provided, restricts acquisition to exactly
        those datasets -- and ONLY those that also belong to ``project_id``;
        an id belonging to a different project (or that doesn't exist) is
        silently excluded rather than raised, per the fail-closed principle
        (never expand scope on a bad id), but is reported back via
        ``unavailable_requested_ids`` so the omission is visible in
        provenance rather than silent.
        """
        pass


class DatabaseDatasetProvider(BaseDatasetProvider):
    """Provides dataset snapshots from the database and dataset service."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def acquire_context(self, project_id: str, dataset_ids: Optional[List[str]] = None) -> InvestigationDataContext:
        db: Session = self.session_factory()
        datasets_map: Dict[str, pd.DataFrame] = {}
        dataset_fingerprints: Dict[str, str] = {}
        dataset_versions: Dict[str, Dict[str, object]] = {}
        unavailable: List[str] = []
        load_errors: Dict[str, str] = {}
        scope_errors: List[str] = []
        try:
            ds_service = DatasetService(db)
            query = db.query(Dataset).filter(Dataset.project_id == project_id)
            if dataset_ids:
                # Scope is the INTERSECTION of "belongs to this project" and
                # "was explicitly requested" -- this is what actually
                # prevents cross-project leakage (a dataset_id from another
                # project is excluded by the project_id filter regardless of
                # being named in dataset_ids) and what prevents silent
                # expansion back to the whole project (an unfiltered id from
                # THIS project that wasn't requested is excluded too).
                query = query.filter(Dataset.id.in_(dataset_ids))
            db_datasets = query.all()
            found_ids = {d.id for d in db_datasets}
            names_to_ids: Dict[str, List[str]] = {}
            for item in db_datasets:
                names_to_ids.setdefault(item.name, []).append(item.id)
            duplicate_names = {name: ids for name, ids in names_to_ids.items() if len(ids) > 1}
            if duplicate_names:
                scope_errors.append(
                    "Dataset names are not unique within the requested scope: "
                    + "; ".join(f"{name}={sorted(ids)}" for name, ids in sorted(duplicate_names.items()))
                    + ". Rename the datasets or select only one of the colliding datasets; AA-OS will not silently overwrite a table."
                )
            if dataset_ids:
                unavailable = [did for did in dataset_ids if did not in found_ids]
            for d in db_datasets:
                try:
                    df = ds_service.get_dataset_dataframe(d.id, d.current_version)
                    datasets_map[d.name] = df
                    version_record = (
                        db.query(DatasetVersion)
                        .filter(DatasetVersion.dataset_id == d.id, DatasetVersion.version_number == d.current_version)
                        .first()
                    )
                    # Persisted DatasetVersion.content_hash is the authoritative
                    # identity for database-backed datasets. Recomputing a logical
                    # content-only hash here would silently disagree with the
                    # provenance-bound ingestion identity.
                    fingerprint = (version_record.content_hash if version_record and version_record.content_hash else compute_content_hash(df))
                    dataset_fingerprints[d.name] = fingerprint
                    dataset_versions[d.name] = {
                        "dataset_id": d.id,
                        "dataset_version_id": version_record.id if version_record else None,
                        "dataset_version": int(d.current_version),
                        "row_count": int(len(df)),
                        "column_count": int(len(df.columns)),
                        "file_name": d.description or d.name,
                    }
                except Exception as exc:
                    load_errors[d.id] = f"{type(exc).__name__}: {exc}"
                    if d.id not in unavailable:
                        unavailable.append(d.id)
        finally:
            db.close()

        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=datasets_map,
            dataset_fingerprints=dataset_fingerprints,
            dataset_versions=dataset_versions,
            requested_dataset_ids=list(dataset_ids) if dataset_ids else None,
            unavailable_requested_ids=unavailable,
            load_errors=load_errors,
            scope_errors=scope_errors,
        )


class InMemoryDatasetProvider(BaseDatasetProvider):
    """Provides dataset snapshots from an in-memory dictionary of dataframes."""

    def __init__(self, datasets_map: Optional[Dict[str, pd.DataFrame]] = None):
        self.datasets_map = datasets_map or {}

    def register_dataset(self, name: str, df: pd.DataFrame) -> None:
        self.datasets_map[name] = df

    def acquire_context(self, project_id: str, dataset_ids: Optional[List[str]] = None) -> InvestigationDataContext:
        # In-memory callers (benchmarks/tests/runtime adapter) key datasets
        # by name, not a database id, so "dataset_ids" here is matched
        # against dataset NAME for the same scoping semantics.
        if dataset_ids:
            scoped_map = {name: df for name, df in self.datasets_map.items() if name in dataset_ids}
            unavailable = [did for did in dataset_ids if did not in self.datasets_map]
        else:
            scoped_map = dict(self.datasets_map)
            unavailable = []
        fingerprints = {}
        dataset_versions = {}
        for name, df in scoped_map.items():
            fingerprints[name] = compute_content_hash(df)
            dataset_versions[name] = {
                "dataset_id": name,
                "dataset_version": 1,
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "file_name": name,
            }
        scope_errors: List[str] = []
        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=scoped_map,
            dataset_fingerprints=fingerprints,
            dataset_versions=dataset_versions,
            requested_dataset_ids=list(dataset_ids) if dataset_ids else None,
            unavailable_requested_ids=unavailable,
            load_errors={},
            scope_errors=scope_errors,
        )


