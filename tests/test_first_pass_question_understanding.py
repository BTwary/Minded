"""Questions a human types once must be understood on the first pass, and run end to end."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("_rr", pathlib.Path(__file__).with_name("test_release_regressions.py"))
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)


_rr._SCRIPT = _rr._SCRIPT.replace(
    'names = ["customers", "sales"]',
    'names = sorted(f[:-4] for f in os.listdir(tables_dir) if f.endswith(".csv"))',
)


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    rng = np.random.default_rng(11)
    n = 1200
    df = pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(n)],
        "customer_segment": rng.choice(["Consumer", "Small Business", "Corporate"], n),
        "sales_channel": rng.choice(["Website", "Mobile App", "Marketplace", "Retail Partner"], n),
        "order_status": rng.choice(["Delivered", "Returned", "Cancelled"], n, p=[.7, .15, .15]),
        "returned": rng.choice([0, 1], n, p=[.85, .15]),
        "gross_sales": np.round(rng.gamma(2.0, 150.0, n), 2),
        "discount_pct": np.round(rng.random(n) * .3, 3),
    })
    d = tmp_path_factory.mktemp("ns")
    df.to_csv(d / "orders.csv", index=False)
    return d, df


def _ask(tables, tmp_path, q):
    _rr._SCRIPT = _rr._SCRIPT
    return _rr.run(tables, tmp_path, q, "orders")


def test_among_population_question_runs_with_all_three_filters(tables, tmp_path):
    _, df = tables
    r = _ask(tables, tmp_path, "Among Corporate orders that were delivered and not returned, which sales channel had the highest average gross sales per order?")
    assert r["status"] == "COMPLETED", r
    x = df[(df.customer_segment == "Corporate") & (df.order_status == "Delivered") & (df.returned == 0)]
    top = x.groupby("sales_channel").gross_sales.mean().sort_values(ascending=False)
    print("ANSWER:", r["answer"])
    assert r["answer"].lstrip().startswith(top.index[0]) or top.index[0] in r["answer"][:80], (r, top)
    expected = "; ".join(f"{k}: {v:.1f}" for k, v in top.items())
    assert r["answer"].startswith(expected), (r["answer"], expected)


def test_not_delivered_is_a_negation(tables, tmp_path):
    _, df = tables
    r = _ask(tables, tmp_path, "What is the total gross sales for orders that were not delivered?")
    assert r["status"] == "COMPLETED", r
    want = df[df.order_status != "Delivered"].gross_sales.sum()
    assert f"{want:,.0f}" in r["answer"].replace(",", ",") or f"{want:,.2f}" in r["answer"], (r, want)
