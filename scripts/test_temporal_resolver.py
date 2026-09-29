"""Independent test of TemporalResolver against the exact expression list
required by the brief (DEFECT-002): today, yesterday, last month, this
month, March, March 2026, Q1, last quarter, year over year, previous month,
previous year, last 90 days, trailing 12 months, week over week,
before/after a date, between two dates.

Ground truth for each case is computed independently in this file, not by
calling any other part of the resolver's own logic.
"""
import sys
sys.path.insert(0, ".")
from datetime import date
import pandas as pd
from packages.analytics_core.src.engines.temporal import TemporalResolver

# Synthetic dataset spanning 2025-01-01 .. 2026-04-30, so "March" appears in
# two different years (2025 and 2026) -- deliberately exercises the
# ambiguous-bare-month path, not just a single clean example.
dates = pd.date_range("2025-01-01", "2026-04-30", freq="D")
df = pd.DataFrame({"order_date": dates, "revenue": range(len(dates))})

failures = []


def check(question, expect_start, expect_end, expect_baseline_start=None, label=""):
    r = TemporalResolver.resolve(question, df, "order_date")
    ok = r.found and r.period_start == expect_start and r.period_end == expect_end
    if expect_baseline_start is not None:
        ok = ok and r.baseline_start == expect_baseline_start
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(question)
    print(f"[{status}] '{question}' ({label}) -> start={r.period_start} end={r.period_end} "
          f"baseline_start={r.baseline_start} ambiguous={r.ambiguous} method={r.resolution_method}")


data_max = date(2026, 4, 30)

check("Why did revenue fall in March 2026?", date(2026, 3, 1), date(2026, 4, 1), date(2026, 2, 1), "explicit month+year")
check("How did we do this month?", date(2026, 4, 1), date(2026, 5, 1), date(2026, 3, 1), "this month (anchored to data max)")
check("What happened last month?", date(2026, 3, 1), date(2026, 4, 1), date(2026, 2, 1), "last month")
check("What happened previous month?", date(2026, 3, 1), date(2026, 4, 1), date(2026, 2, 1), "previous month alias")
check("Show me Q1 2026 performance", date(2026, 1, 1), date(2026, 4, 1), date(2025, 10, 1), "explicit quarter+year")
check("How did last quarter go?", date(2026, 1, 1), date(2026, 4, 1), date(2025, 10, 1), "last quarter (anchored)")
check("Compare year over year performance", date(2026, 4, 1), date(2026, 5, 1), date(2025, 4, 1), "yoy")
check("What's the last 90 days trend?", date(2026, 1, 31), date(2026, 4, 30) + pd.Timedelta(days=1), None, "last N days")
check("Show trailing 12 months revenue", date(2025, 5, 1), date(2026, 4, 30) + pd.Timedelta(days=1), None, "trailing 12mo")
check("How was week over week growth?", date(2026, 4, 24), date(2026, 4, 30) + pd.Timedelta(days=1), None, "wow")
check("What happened yesterday?", date(2026, 4, 29), date(2026, 4, 30), date(2026, 4, 28), "yesterday")
check("What happened today?", date(2026, 4, 30), date(2026, 5, 1), date(2026, 4, 29), "today")
check("Show revenue before 2026-03-01", date(2025, 1, 1), date(2026, 3, 1), None, "before")
check("Show revenue after 2026-03-01", date(2026, 3, 2), date(2026, 5, 1), None, "after")
check("Show revenue between 2026-01-01 and 2026-01-31", date(2026, 1, 1), date(2026, 2, 1), None, "between")
check("What happened previous year?", date(2025, 1, 1), date(2026, 1, 1), date(2024, 1, 1), "previous year")

# Bare "March" is genuinely ambiguous in this dataset (present in both 2025
# and 2026) -- verify it's flagged as such, not silently resolved.
r = TemporalResolver.resolve("Why did revenue fall in March?", df, "order_date")
ok = r.found and r.ambiguous and r.period_start == date(2026, 3, 1)
print(f"[{'PASS' if ok else 'FAIL'}] 'Why did revenue fall in March?' (bare month, genuinely ambiguous across 2 years) "
      f"-> start={r.period_start} ambiguous={r.ambiguous} note={r.ambiguity_note!r}")
if not ok:
    failures.append("bare March (ambiguous)")

# Bare month that only occurs in one year of the data should resolve
# cleanly, not be flagged ambiguous.
single_year_dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
df_single = pd.DataFrame({"order_date": single_year_dates, "revenue": range(len(single_year_dates))})
r = TemporalResolver.resolve("Why did revenue fall in March?", df_single, "order_date")
ok = r.found and not r.ambiguous and r.period_start == date(2026, 3, 1) and r.period_end == date(2026, 4, 1)
print(f"[{'PASS' if ok else 'FAIL'}] 'Why did revenue fall in March?' (single-year data, unambiguous) "
      f"-> start={r.period_start} end={r.period_end} ambiguous={r.ambiguous}")
if not ok:
    failures.append("bare March (single year)")

# A month that doesn't appear in the data at all.
r = TemporalResolver.resolve("Why did revenue fall in December?", df_single, "order_date")
ok = (not r.found) and r.ambiguous
print(f"[{'PASS' if ok else 'FAIL'}] 'Why did revenue fall in December?' (month absent from data) "
      f"-> found={r.found} ambiguous={r.ambiguous} note={r.ambiguity_note!r}")
if not ok:
    failures.append("December not in data")

# No temporal language at all.
r = TemporalResolver.resolve("What are the correlation drivers of revenue?", df, "order_date")
ok = not r.found
print(f"[{'PASS' if ok else 'FAIL'}] 'What are the correlation drivers of revenue?' (no temporal language) -> found={r.found}")
if not ok:
    failures.append("no temporal language")

# SQL filter generation sanity check.
r = TemporalResolver.resolve("Why did revenue fall in March 2026?", df, "order_date")
sql = r.sql_filter("order_date")
expected_sql = "CAST(order_date AS VARCHAR) >= '2026-03-01' AND CAST(order_date AS VARCHAR) < '2026-04-01'"
ok = sql == expected_sql
print(f"[{'PASS' if ok else 'FAIL'}] sql_filter() -> {sql!r}")
if not ok:
    failures.append("sql_filter generation")

print()
if failures:
    print(f"FAILED: {len(failures)} case(s): {failures}")
    sys.exit(1)
else:
    print("ALL TEMPORAL RESOLVER CASES PASSED")
