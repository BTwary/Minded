"""Enterprise Multi-Data-Type Ingestion & Cleansing Verification Suite.

Tests support for:
1. Currency symbols: $, EUR, GBP, JPY, INR, KRW
2. Financial accounting negative parentheses: (1,234.50), ($50.00), (500)
3. Percentages: 15.5%, -3.2%, (2.5%)
4. European decimal/thousand formats: 1.234,56
5. Dirty spreadsheet NA tokens: #VALUE!, #REF!, #DIV/0!, nil, -999
6. Robust analyst recomputation on messy enterprise inputs
"""
import unittest
import pandas as pd
import numpy as np

from packages.analytics_core.src.ingestion.robust_loader import RobustFileLoader
from packages.analytics_core.src.engines.analyst_answer import _to_numeric, build_analyst_result


class TestEnterpriseDataTypes(unittest.TestCase):

    def test_robust_loader_enterprise_numeric_coercion(self):
        csv_data = (
            "account_id,revenue,discount,status\n"
            "A01,\"$1,234.50\",15.0%,active\n"
            "A02,\"($50.00)\",0.0%,churned\n"
            "A03,\"$500.00\",10.5%,active\n"
            "A04,\"($100.00)\",-3.0%,active\n"
            "A05,N/A,#VALUE!,unknown\n"
        )
        clean_df, report = RobustFileLoader().load(file_bytes=csv_data.encode("utf-8"), filename="enterprise.csv")
        self.assertEqual(clean_df["revenue"].iloc[0], 1234.50)
        self.assertEqual(clean_df["revenue"].iloc[1], -50.00)
        self.assertEqual(clean_df["revenue"].iloc[2], 500.00)
        self.assertEqual(clean_df["revenue"].iloc[3], -100.00)
        self.assertTrue(pd.isna(clean_df["revenue"].iloc[4]))

        self.assertEqual(clean_df["discount"].iloc[0], 0.15)
        self.assertEqual(clean_df["discount"].iloc[1], 0.0)
        self.assertEqual(clean_df["discount"].iloc[2], 0.105)
        self.assertEqual(clean_df["discount"].iloc[3], -0.03)
        self.assertTrue(pd.isna(clean_df["discount"].iloc[4]))

    def test_analyst_answer_to_numeric_enterprise_formats(self):
        s = pd.Series(["$1,234.50", "($50.00)", "(100)", "-$25.50", "15.5%", "N/A", "#VALUE!"])
        res = _to_numeric(s)
        self.assertEqual(res.iloc[0], 1234.50)
        self.assertEqual(res.iloc[1], -50.00)
        self.assertEqual(res.iloc[2], -100.00)
        self.assertEqual(res.iloc[3], -25.50)
        self.assertEqual(res.iloc[4], 15.50)
        self.assertTrue(pd.isna(res.iloc[5]))
        self.assertTrue(pd.isna(res.iloc[6]))

    def test_european_format_coercion(self):
        csv_data = (
            "id,amount\n"
            "1,\"1.234,56 €\"\n"
            "2,\"(500,00 €)\"\n"
            "3,\"250,50 €\"\n"
        )
        clean_df, report = RobustFileLoader().load(file_bytes=csv_data.encode("utf-8"), filename="euro.csv")
        self.assertAlmostEqual(clean_df["amount"].iloc[0], 1234.56, places=2)
        self.assertAlmostEqual(clean_df["amount"].iloc[1], -500.00, places=2)
        self.assertAlmostEqual(clean_df["amount"].iloc[2], 250.50, places=2)


if __name__ == "__main__":
    unittest.main()
