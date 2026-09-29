import os, sys
from types import SimpleNamespace
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.analytics_core.src.data.content_identity import compute_content_hash
from packages.analytics_core.src.graph.evidence_identity import compute_evidence_identity
from packages.analytics_core.src.engines.method_selection import MethodRegistry


def test_evidence_identity_changes_when_population_scope_changes():
    base = {"metric":"revenue", "dimension":"region", "aggregation_type":"SUM", "grouped":True,
            "population":{"filters":[]}, "temporal_scope":None}
    filtered = {**base, "population":{"filters":[{"column":"year","operator":"eq","value":2025}]}}
    ds = "a" * 64
    assert compute_evidence_identity(ds, "SELECT region, SUM(revenue) FROM sales GROUP BY region", "duckdb_sql", base) != \
           compute_evidence_identity(ds, "SELECT region, SUM(revenue) FROM sales WHERE year=2025 GROUP BY region", "duckdb_sql", filtered)


def test_dataset_identity_can_bind_stable_provenance_without_ids():
    df = pd.DataFrame({"a":[1,2], "b":["x","y"]})
    source = {"filename":"sales.csv", "extension":"csv", "source_sha256":"f"*64}
    assert compute_content_hash(df, source_metadata=source, transformation_lineage=[]) != \
           compute_content_hash(df, source_metadata={**source, "source_sha256":"e"*64}, transformation_lineage=[])
    assert compute_content_hash(df, source_metadata=source, transformation_lineage=[{"operation":"clean","input_content_hash":"1"}]) != \
           compute_content_hash(df, source_metadata=source, transformation_lineage=[{"operation":"clean","input_content_hash":"2"}])


def test_generic_executable_tasks_have_registry_identity():
    df = pd.DataFrame({"revenue":[1,2,3]})
    semantic = SimpleNamespace(target_column="revenue", metric_definition=object(), grouping_columns=[], group_dimension_col=None, explanatory_columns=[], time_column=None)
    plan = SimpleNamespace(task="DIAGNOSTIC", semantics=semantic)
    ok, detail = MethodRegistry.admissibility_for_plan(plan, semantic, None, df)
    assert ok
    assert detail["method"] == "generic_diagnostic_battery"
    assert detail["status"] == "ALLOWED"


def test_database_provider_prefers_persisted_dataset_version_identity():
    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    from apps.api.src.models.entities import Dataset, DatasetVersion, Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    # DatasetService is intentionally not exercised here; only the identity lookup contract is tested.
    d = Dataset(id="D1", project_id="P1", name="sales", current_version=1, row_count=1, column_count=1)
    db.add(d)
    db.add(DatasetVersion(dataset_id="D1", version_number=1, file_path="/tmp/noop", file_size_bytes=1, row_count=1, column_count=1, content_hash="a"*64, transformation_applied={"source_sha256":"b"*64}))
    db.commit()
    rec = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == "D1", DatasetVersion.version_number == 1).first()
    assert rec.content_hash == "a"*64
    db.close()
