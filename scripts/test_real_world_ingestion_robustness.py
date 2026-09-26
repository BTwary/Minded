"""
GOLDEN ACCEPTANCE SUITE: Real-World Data Ingestion Robustness.

Addresses roadmap item 16 ("real-world data" gap): a deterministic core that
only works on clean data is not yet a complete autonomous analyst. This suite
feeds RobustFileLoader genuinely messy files -- the kind a real company
export actually produces -- and proves it (a) does not crash, (b) recovers
the correct values, and (c) truthfully reports every parsing decision it made
rather than silently guessing.

Run: python3 scripts/test_real_world_ingestion_robustness.py
"""
import io
import sys
import os
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.analytics_core.src.ingestion.robust_loader import RobustFileLoader


class TestRealWorldIngestionRobustness(unittest.TestCase):

    def test_01_semicolon_delimited_latin1_with_bom(self):
        """European-style export: semicolon delimiter, Latin-1 encoded region
        names with accents, no BOM. Must autodetect both encoding and
        delimiter correctly."""
        raw = (
            "id;region;revenue\n"
            "1;M\xe9tropole;1000\n"
            "2;Qu\xe9bec;2000\n"
        ).encode("latin-1")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="export.csv")
        self.assertEqual(report.delimiter_detected, ";")
        self.assertIn(report.encoding_detected, ("latin-1", "cp1252"))
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["id", "region", "revenue"])
        self.assertEqual(df["region"].iloc[0], "M\xe9tropole")

    def test_02_utf8_bom_and_inconsistent_na_tokens(self):
        """A UTF-8 file with a BOM and several different ways of expressing
        'missing' in the same column -- all must normalize to real NaN."""
        raw = (
            "id,customer,segment\n"
            "1,Acme Co,Enterprise\n"
            "2,Beta LLC,N/A\n"
            "3,Gamma Inc,unknown\n"
            "4,Delta Ltd,--\n"
            "5,Epsilon Co, Unknown \n"
        ).encode("utf-8-sig")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="customers.csv")
        self.assertEqual(report.encoding_detected, "utf-8-sig")
        self.assertEqual(df["segment"].isna().sum(), 4)
        self.assertTrue(len(report.na_tokens_normalized) > 0)

    def test_03_currency_and_thousands_separator_coercion(self):
        """Revenue stored as formatted currency strings must become real
        numeric values so downstream statistics (mean, ANOVA, etc.) work."""
        raw = (
            "order_id,amount\n"
            "1,\"$1,234.56\"\n"
            "2,\"$987.10\"\n"
            "3,\"$2,000.00\"\n"
        ).encode("utf-8")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="orders.csv")
        self.assertIn("amount", report.columns_coerced_numeric)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["amount"]))
        self.assertAlmostEqual(float(df["amount"].iloc[0]), 1234.56, places=2)
        self.assertAlmostEqual(float(df["amount"].sum()), 4221.66, places=2)

    def test_04_duplicate_and_whitespace_padded_headers(self):
        """Real exports frequently have padded or duplicated column names
        (e.g. from a spreadsheet merge). Must be deduped losslessly, not
        silently drop a column."""
        raw = (
            " id , amount,amount\n"
            "1,100,200\n"
            "2,150,250\n"
        ).encode("utf-8")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="dupes.csv")
        self.assertEqual(list(df.columns), ["id", "amount", "amount_1"])
        self.assertIn("id", report.columns_renamed.values())
        self.assertEqual(df["amount"].iloc[0], 100)
        self.assertEqual(df["amount_1"].iloc[0], 200)

    def test_05_percent_column_coercion(self):
        """Percent-formatted strings should coerce to their fractional
        numeric value (15% -> 0.15), not be left as uncomputable text."""
        raw = "segment,discount_rate\nEnterprise,15%\nSMB,5%\n".encode("utf-8")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="discounts.csv")
        self.assertIn("discount_rate", report.columns_coerced_numeric)
        self.assertAlmostEqual(float(df["discount_rate"].iloc[0]), 0.15, places=4)

    def test_06_tab_delimited_file_with_csv_extension(self):
        """A tab-separated file mislabeled with a .csv extension (common when
        exported from legacy systems) must still be parsed correctly rather
        than producing one giant unsplit column."""
        raw = "id\tregion\tcost\n1\tEU\t500\n2\tUS\t700\n".encode("utf-8")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="legacy_export.csv")
        self.assertEqual(report.delimiter_detected, "\t")
        self.assertEqual(list(df.columns), ["id", "region", "cost"])
        self.assertEqual(len(df), 2)

    def test_07_all_null_rows_are_dropped_and_counted(self):
        """Trailing blank rows (common in manually-edited spreadsheet
        exports) should be dropped, and the drop must be reported, not
        silent."""
        raw = "id,value\n1,10\n2,20\n,\n,\n".encode("utf-8")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="trailing_blanks.csv")
        self.assertEqual(len(df), 2)
        self.assertEqual(report.rows_all_null_dropped, 2)

    def test_08_json_records_and_nested_wrapper(self):
        """JSON exports sometimes wrap the actual records list inside a
        top-level object (e.g. {"data": [...]}) rather than being a bare
        array. Both forms must load correctly."""
        bare_array = b'[{"id": 1, "region": "EU"}, {"id": 2, "region": "US"}]'
        df1, _ = RobustFileLoader().load(file_bytes=bare_array, filename="a.json")
        self.assertEqual(len(df1), 2)

        wrapped = b'{"data": [{"id": 1, "region": "EU"}, {"id": 2, "region": "US"}]}'
        df2, report2 = RobustFileLoader().load(file_bytes=wrapped, filename="b.json")
        self.assertEqual(len(df2), 2)
        self.assertTrue(any("nested key" in w for w in report2.warnings))

    def test_09_excel_multi_sheet_uses_first_and_reports_rest(self):
        """A multi-sheet workbook must use the first sheet by default and
        truthfully report which other sheets were ignored, rather than
        silently discarding data with no trace."""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame({"id": [1, 2], "region": ["EU", "US"]}).to_excel(
                writer, sheet_name="Q1", index=False
            )
            pd.DataFrame({"id": [3, 4], "region": ["APAC", "LATAM"]}).to_excel(
                writer, sheet_name="Q2", index=False
            )
        df, report = RobustFileLoader().load(file_bytes=buf.getvalue(), filename="quarters.xlsx")
        self.assertEqual(report.sheet_used, "Q1")
        self.assertEqual(report.other_sheets_ignored, ["Q2"])
        self.assertEqual(len(df), 2)

    def test_10_end_to_end_messy_file_still_yields_correct_ground_truth(self):
        """Combine several real-world defects in one file (semicolon
        delimiter, currency formatting, inconsistent NA tokens, padded
        headers) and confirm the final numeric ground truth is still
        recoverable -- this is the actual bar: not 'does it parse' but
        'does the analyst still get the right number.'"""
        raw = (
            " id ; segment ;revenue\n"
            "1;Enterprise;\"$10,000.00\"\n"
            "2;Enterprise;\"$12,000.00\"\n"
            "3;SMB;unknown\n"
            "4;SMB;\"$1,000.00\"\n"
        ).encode("utf-8")
        df, report = RobustFileLoader().load(file_bytes=raw, filename="messy_quarterly.csv")
        self.assertEqual(report.delimiter_detected, ";")
        self.assertIn("revenue", report.columns_coerced_numeric)
        enterprise_total = df[df["segment"] == "Enterprise"]["revenue"].sum()
        self.assertAlmostEqual(float(enterprise_total), 22000.00, places=2)
        self.assertEqual(df["revenue"].isna().sum(), 1)


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING REAL-WORLD DATA INGESTION ROBUSTNESS ACCEPTANCE SUITE")
    print("=" * 80)
    unittest.main(verbosity=2)
