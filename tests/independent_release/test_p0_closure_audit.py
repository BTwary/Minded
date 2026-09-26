import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.analytics_core.src.intelligence.nl_semantic_interpreter import interpret_with_schema
from packages.analytics_core.src.graph.evidence_identity import compute_evidence_identity


def test_physical_units_never_fabricated_from_range():
    df = pd.DataFrame({"temperature_c": [10, 20, 30], "pressure_kpa": [100, 110, 120], "latency_ms": [5, 7, 9]})
    model = SemanticWorldModelBuilder().build_world_model({"t": df})
    by_col = {m.column_name: m for m in model.metrics}
    assert by_col["temperature_c"].unit == "UNKNOWN"
    assert by_col["pressure_kpa"].unit == "UNKNOWN"
    assert by_col["latency_ms"].unit == "milliseconds"
    assert all(m.additivity.value == "UNKNOWN" for m in by_col.values())


def test_unknown_grain_is_unknown_not_first_column():
    df = pd.DataFrame({"temperature_c": [10, 20, 30], "pressure_kpa": [100, 110, 120]})
    model = SemanticWorldModelBuilder().build_world_model({"t": df})
    assert model.table_grains["t"] == "UNKNOWN"


def test_natural_paraphrases_route_without_exact_keywords():
    cols = ["client_key", "net_sales", "plan_class", "fulfillment_latency"]
    assert interpret_with_schema("What is the typical net sales by plan class?", cols).task in {"COMPARISON", "DESCRIPTIVE"}
    assert interpret_with_schema("How does fulfillment delay relate to net sales?", cols).task == "ASSOCIATION"
    assert interpret_with_schema("Which customer group has the highest margin?", cols).task == "COMPARISON"


def test_semantically_equivalent_simple_queries_share_identity():
    a = compute_evidence_identity("dataset", "SELECT SUM(x) FROM t", "duckdb_sql")
    b = compute_evidence_identity("dataset", "SELECT (SUM(x)) FROM t WHERE TRUE", "duckdb_sql")
    assert a == b
