"""Regressions found by running the real 4-CSV workspace (customers, orders, products, sales_orders_flat_120k)."""
import uuid

import pandas as pd
from sqlalchemy.pool import StaticPool

from apps.api.src.core import database as dbmod
from apps.api.src.models.entities import Investigation
from packages.analytics_core.src.intelligence.canonical_question_contract import resolve_phrase_to_column
from packages.analytics_core.src.semantic.missingness_sensitivity import MissingnessSensitivityEngine


def test_file_database_isolates_connections_per_thread() -> None:
    """StaticPool on a file DB shared ONE connection across threads, so the UI polling status from the
    API thread rolled back the worker thread's uncommitted rows ("Contract <id> not found")."""
    if dbmod.db_url.startswith("sqlite") and ":memory:" not in dbmod.db_url:
        assert not isinstance(dbmod.engine.pool, StaticPool)


def test_other_thread_closing_a_session_cannot_roll_back_uncommitted_rows() -> None:
    import threading

    iid = f"INV-POOL-{uuid.uuid4().hex[:8]}"
    writer = dbmod.SessionLocal()
    try:
        writer.add(Investigation(id=iid, project_id="p-pool", question="q", status="PLANNED"))
        writer.flush()  # the state the controller is in between flush() and commit()

        def poll() -> None:  # the UI's status poll, on the API thread
            poller = dbmod.SessionLocal()
            try:
                poller.query(Investigation).filter(Investigation.id == iid).first()
            finally:
                poller.close()  # close() rolls back ITS connection

        t = threading.Thread(target=poll)
        t.start()
        t.join(30)
        assert not t.is_alive()
        writer.commit()
    finally:
        writer.close()
    check = dbmod.SessionLocal()
    try:
        assert check.query(Investigation).filter(Investigation.id == iid).first() is not None
    finally:
        check.close()


def test_sessions_on_one_thread_still_share_a_connection() -> None:
    """Controller/tests rely on same-thread sessions seeing each other's flushed state."""
    iid = f"INV-SAME-{uuid.uuid4().hex[:8]}"
    a = dbmod.SessionLocal()
    b = dbmod.SessionLocal()
    try:
        a.add(Investigation(id=iid, project_id="p-same", question="q", status="PLANNED"))
        a.commit()
        assert b.query(Investigation).filter(Investigation.id == iid).first() is not None
    finally:
        a.close()
        b.close()


_ORDER_COLS = ["order_id", "customer_id", "segment", "category", "revenue", "discount_pct", "quantity", "profit"]


def test_average_order_value_never_binds_an_identifier_column() -> None:
    for phrase in ("average order value", "order value", "aov"):
        assert resolve_phrase_to_column(phrase, _ORDER_COLS) == "revenue", phrase
    # With no monetary column at all it must stay unresolved rather than fall back to order_id.
    assert resolve_phrase_to_column("average order value", ["order_id", "segment", "qty"]) is None
    assert resolve_phrase_to_column("customer spend", ["customer_id", "segment"]) is None


def test_identifier_columns_still_resolve_exactly() -> None:
    assert resolve_phrase_to_column("customer id", _ORDER_COLS) == "customer_id"
    assert resolve_phrase_to_column("customer segment", _ORDER_COLS) == "segment"


def test_missingness_bound_on_non_numeric_target_is_undefined_not_a_crash() -> None:
    df = pd.DataFrame({"order_id": [f"O{i:07d}" for i in range(50)], "segment": ["a", "b"] * 25})
    observed, lower, upper, kind, _, ref = MissingnessSensitivityEngine._bound_for_metric(df, None, "order_id")
    assert (observed, lower, upper, ref) == (None, None, None, None)
    assert kind == "none"
