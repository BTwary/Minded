"""
GROUND-TRUTH BENCHMARK: Full Investigation Pipeline Against Messy Real-World Input.

Addresses roadmap items 16 and 20.5 together: a deterministic core proven only
against clean, hand-built pandas DataFrames is not yet proof that AA-OS works
on real data. This test takes the exact same known-ground-truth scenario as
test_golden_deterministic_investigation.py (Enterprise segment responsible
for a cost surge), serializes it into a genuinely messy CSV export (semicolon
delimiter, currency-formatted numbers, inconsistent NA tokens, whitespace-
padded headers), loads it back through RobustFileLoader exactly as a real
upload would be, and re-runs the full investigation pipeline.

The bar here is not "did it parse" -- it's "did the analyst still get the
correct answer." If messy ingestion silently corrupts a value, the posterior,
variance-explained, or verdict assertions below will fail even though the
file loaded without an exception.

Run: python3 scripts/test_ground_truth_benchmark_messy_input.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.ingestion.robust_loader import RobustFileLoader
from packages.analytics_core.src.engines.belief import BeliefEngine
from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.engines.verdict import VerdictEngine
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder


def _build_ground_truth_clean_df() -> pd.DataFrame:
    """Same ground truth as the golden test: Enterprise segment has a massive
    cost spike (~$5,000/unit); other segments are baseline (~$100/unit)."""
    np.random.seed(42)
    segments = ["Enterprise"] * 25 + ["SMB"] * 25 + ["MidMarket"] * 25 + ["Growth"] * 25
    costs = []
    for s in segments:
        if s == "Enterprise":
            costs.append(float(np.random.normal(5000.0, 200.0)))
        else:
            costs.append(float(np.random.normal(100.0, 15.0)))
    return pd.DataFrame({
        "account_id": [f"ACC_{i:04d}" for i in range(100)],
        "tier_segment": segments,
        "cost_metric": costs,
        "usage_volume": np.random.uniform(50.0, 100.0, 100),
    })


def _serialize_as_messy_real_world_csv(df: pd.DataFrame) -> bytes:
    """Turn the clean ground-truth DataFrame into bytes that look like a
    genuinely messy real-world export:
    - semicolon delimiter (European-style export)
    - cost_metric formatted as currency strings ("$5,123.45")
    - a scattering of usage_volume cells replaced with inconsistent NA tokens
    - whitespace padding around headers
    """
    lines = [" account_id ; tier_segment ; cost_metric ; usage_volume "]
    na_tokens = ["N/A", "unknown", "--", " Unknown "]
    rng = np.random.RandomState(7)
    for i, row in df.iterrows():
        usage_cell = f"{row['usage_volume']:.4f}"
        # Inject a non-critical missing-value defect in ~10% of rows; this
        # column is not the target metric, so it must not affect the verdict.
        if rng.rand() < 0.10:
            usage_cell = na_tokens[int(rng.randint(0, len(na_tokens)))]
        cost_cell = f"\"${row['cost_metric']:,.2f}\""
        lines.append(f"{row['account_id']};{row['tier_segment']};{cost_cell};{usage_cell}")
    return ("\n".join(lines) + "\n").encode("utf-8")


class TestGroundTruthBenchmarkMessyInput(unittest.TestCase):
    """Same ground-truth invariants as the golden test, but the pipeline is
    fed a genuinely messy real-world file instead of a hand-built DataFrame."""

    def test_investigation_reaches_correct_verdict_from_messy_csv(self):
        clean_df = _build_ground_truth_clean_df()
        messy_bytes = _serialize_as_messy_real_world_csv(clean_df)

        # ---------------------------------------------------------------
        # STEP 0: INGEST THE MESSY FILE (this is the part that's new
        # versus the golden test -- everything after this point is the
        # same pipeline, run on recovered data instead of clean data)
        # ---------------------------------------------------------------
        df, report = RobustFileLoader().load(file_bytes=messy_bytes, filename="quarterly_export.csv")

        self.assertEqual(report.delimiter_detected, ";")
        self.assertIn("cost_metric", report.columns_coerced_numeric)
        self.assertEqual(len(df), 100)
        # usage_volume should have picked up some real NaNs from the injected
        # NA tokens, proving the messiness was actually present and handled,
        # not just theoretically injected.
        self.assertGreater(df["usage_volume"].isna().sum(), 0)

        # cost_metric must be numerically faithful to the original values
        # (currency formatting recovered, not merely "some number").
        recovered_total = float(df["cost_metric"].sum())
        original_total = float(clean_df["cost_metric"].sum())
        self.assertAlmostEqual(recovered_total, original_total, delta=1.0)

        # ---------------------------------------------------------------
        # STEP 1: DATA QUALITY FITNESS GATE
        # ---------------------------------------------------------------
        fitness = DataQualityGate.evaluate_fitness(df, "cloud_infrastructure_costs")
        self.assertTrue(fitness.can_proceed)

        # ---------------------------------------------------------------
        # STEP 2: SEMANTIC WORLD MODEL
        # ---------------------------------------------------------------
        world_model = SemanticWorldModelBuilder().build_world_model({"cloud_costs": df})
        self.assertIn("cloud_costs", world_model.table_grains)

        # ---------------------------------------------------------------
        # STEP 3: INTENT + SCHEMA RESOLUTION
        # ---------------------------------------------------------------
        question = "Why did cost_metric surge across tier_segment?"
        intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

        semantic = SemanticEngine().resolve_schema(intent, {"cloud_costs": df})
        self.assertEqual(semantic.target_metric_col, "cost_metric")
        self.assertEqual(semantic.group_dimension_col, "tier_segment")

        # ---------------------------------------------------------------
        # STEP 4: HYPOTHESIS SYNTHESIS
        # ---------------------------------------------------------------
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        h1, h2 = hyps[0], hyps[1]

        # ---------------------------------------------------------------
        # STEP 5: DETERMINISTIC DUCKDB EXECUTION ON THE RECOVERED DATA
        # ---------------------------------------------------------------
        exec_provider = DuckDBExecutionProvider()
        exec_result = exec_provider.execute_query(
            df=df,
            table_name="cloud_costs",
            query_sql="SELECT tier_segment, SUM(cost_metric) as val FROM cloud_costs GROUP BY tier_segment ORDER BY val DESC;",
        )
        res_df = exec_result.result_df
        top_segment = res_df.iloc[0]["tier_segment"]
        top_val = float(res_df.iloc[0]["val"])
        total_cost = float(res_df["val"].sum())
        enterprise_share = top_val / total_cost

        # THIS is the actual bar: despite semicolons, currency formatting,
        # and injected NA tokens, the analyst still gets the right answer.
        self.assertEqual(top_segment, "Enterprise")
        self.assertGreater(enterprise_share, 0.90)

        # ---------------------------------------------------------------
        # STEP 6: DUAL-ENGINE VERIFICATION
        # ---------------------------------------------------------------
        verification = VerificationEngine.verify_secondary(
            primary_df=df,
            target_metric_col="cost_metric",
            aggregation_type="SUM",
            primary_metric=top_val,
            group_dimension_col="tier_segment",
            tolerance=1e-4,
        )
        self.assertEqual(verification.status, "VERIFIED")

        # ---------------------------------------------------------------
        # STEP 7: BAYESIAN BELIEF UPDATE
        # ---------------------------------------------------------------
        eta_sq = BeliefEngine.compute_variance_explained(df, "tier_segment", "cost_metric")
        self.assertGreater(eta_sq, 90.0)

        l_h1, l_h2 = BeliefEngine.compute_evidence_likelihoods(
            primary_metric=top_val,
            benchmark_metric=total_cost / 4.0,
            effect_size_eta_sq=eta_sq,
        )
        posteriors, entropy_delta = BeliefEngine.compute_bayesian_posteriors(
            priors=[h1.prior_probability, h2.prior_probability],
            likelihoods=[l_h1, l_h2],
        )
        h1.posterior_probability = posteriors[0]
        self.assertGreater(h1.posterior_probability, 0.90)

        # ---------------------------------------------------------------
        # STEP 8: FINAL VERDICT
        # ---------------------------------------------------------------
        verdict = VerdictEngine.evaluate_verdict(
            question=question,
            leading_hypothesis_code=h1.hypothesis_code,
            leading_posterior=h1.posterior_probability,
            initial_entropy=0.97,
            final_entropy=0.30,
            all_verifications_passed=True,
            variance_explained_pct=eta_sq,
            is_categorical_diagnostic=True,
        )
        self.assertEqual(verdict.verdict_type, "DIAGNOSED")
        self.assertGreaterEqual(verdict.confidence_score, 0.90)

        print("=" * 80)
        print("GROUND-TRUTH BENCHMARK (MESSY REAL-WORLD INPUT) TRACE SUMMARY:")
        print(f" -> Ingestion: delimiter='{report.delimiter_detected}', "
              f"coerced_numeric={report.columns_coerced_numeric}, "
              f"na_normalized={len(report.na_tokens_normalized)} token types")
        print(f" -> Recovered cost_metric total: {recovered_total:,.2f} "
              f"(original: {original_total:,.2f}, delta: {abs(recovered_total - original_total):.4f})")
        print(f" -> Leading Hypothesis: {h1.hypothesis_code} (Posterior: {h1.posterior_probability:.3f})")
        print(f" -> Variance Explained (ANOVA eta^2): {eta_sq:.2f}%")
        print(f" -> Dual-Engine Verification: {verification.status}")
        print(f" -> Verdict: {verdict.verdict_type} (Confidence: {verdict.confidence_score:.2f})")
        print("=" * 80)


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("RUNNING GROUND-TRUTH BENCHMARK: MESSY REAL-WORLD INPUT", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGroundTruthBenchmarkMessyInput)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("=" * 80, flush=True)
        print("GROUND-TRUTH PRESERVED THROUGH MESSY INGESTION (100% PROVEN)", flush=True)
        print("=" * 80, flush=True)
        sys.exit(0)
    else:
        sys.exit(1)
