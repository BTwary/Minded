import pandas as pd

from packages.analytics_core.src.discovery.engine import DiscoveryEngine


def test_discovery_does_not_choose_first_time_column():
    df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "region": ["A", "B", "A"],
            "revenue": [10, 20, 30],
        }
    )
    out = DiscoveryEngine().discover_dataset("orders", df)

    assert out["grain"] == "periodic_time_series_by_ambiguous_time_column"
    assert not any("historical trajectory" in q for q in out["worthy_inquiries"])


def test_discovery_preserves_unique_time_column_behavior():
    df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "region": ["A", "B", "A"],
            "revenue": [10, 20, 30],
        }
    )
    out = DiscoveryEngine().discover_dataset("orders", df)

    assert out["grain"] == "periodic_time_series_by_order_date"
    assert any("historical trajectory" in q for q in out["worthy_inquiries"])


def test_discovery_does_not_choose_first_candidate_key():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "customer_key": ["a", "b", "c"],
            "region": ["A", "B", "A"],
            "revenue": [10, 20, 30],
        }
    )
    out = DiscoveryEngine().discover_dataset("customers", df)

    assert out["grain"] == "record_keyed_by_ambiguous_candidate_key"
