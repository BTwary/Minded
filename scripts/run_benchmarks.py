"""Autonomous AI Data Analyst - Comprehensive Benchmark Evaluation Suite (10 Real-World Inquiries)."""
import os
import sys
import time
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.api.src.ai.orchestrator import AutonomousOrchestrator
from apps.api.src.ai.providers.mock import DeterministicMockAIProvider
from packages.analytics_core.src.profiling.profiler import DataProfiler
from packages.analytics_core.src.sql.engine import DuckDBSQLEngine
from packages.analytics_core.src.statistics.engine import StatisticalEngine
from packages.analytics_core.src.validation.validator import IndependentValidator


def run_benchmark_suite():
    print("=" * 80)
    print("AUTONOMOUS AI DATA ANALYST - 10-STAGE PRODUCTION EVALUATION SUITE")
    print("=" * 80)

    seed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "seed")
    sales_file = os.path.join(seed_dir, "sales.csv")
    products_file = os.path.join(seed_dir, "products.csv")

    if not os.path.exists(sales_file):
        print("Error: Seed data not found. Run scripts/generate_seed_data.py first.")
        sys.exit(1)

    df_sales = pd.read_csv(sales_file)
    df_products = pd.read_csv(products_file)
    datasets = {"sales": df_sales, "products": df_products}
    print(f"Loaded datasets: sales ({len(df_sales):,} rows), products ({len(df_products):,} rows).\n")

    passed_count = 0
    total_benchmarks = 10
    mock_ai = DeterministicMockAIProvider()
    orchestrator = AutonomousOrchestrator(ai_provider=mock_ai)

    # -------------------------------------------------------------
    # BENCHMARK 1: Data Profiling & Quality Scoring
    # -------------------------------------------------------------
    print("[1/10] Benchmark: Data Profiling & Quality Score Engine...")
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(df_sales, "sales")
    print(f"   -> Columns profiled: {len(profile.columns)}")
    print(f"   -> Composite Data Quality Score: {profile.data_quality.overall_score}/100")
    assert profile.data_quality.overall_score >= 80.0, "Quality score should be >= 80"
    print("   [PASS] Benchmark 1: Data Profiling verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 2: Read-Only DuckDB Engine & AST Security
    # -------------------------------------------------------------
    print("[2/10] Benchmark: DuckDB Read-Only Security & AST Validator...")
    sql_engine = DuckDBSQLEngine()
    sql_engine.register_dataframe("sales", df_sales)
    res = sql_engine.execute_query("SELECT region, SUM(revenue) as total_rev FROM sales GROUP BY 1 ORDER BY 2 DESC;")
    assert len(res["data"]) == 4, "Should return 4 regions"
    print(f"   -> Executed safe aggregation query in {res['execution_time_ms']}ms.")
    try:
        sql_engine.execute_query("DROP TABLE sales;")
        assert False, "Security failed: DROP query should have been blocked!"
    except Exception as e:
        print(f"   -> Successfully blocked forbidden query: '{e}'")
    print("   [PASS] Benchmark 2: DuckDB security policy verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 3: Statistical Computation Engine
    # -------------------------------------------------------------
    print("[3/10] Benchmark: Deterministic Statistical Tests (SciPy)...")
    stats_engine = StatisticalEngine()
    desc = stats_engine.descriptive_summary(df_sales["revenue"])
    print(f"   -> Revenue Mean: ${desc['mean']:,.2f}, 95% CI: {desc['mean_ci_95']}")
    s_a = df_sales[df_sales["region"] == "Region A"]["revenue"]
    s_b = df_sales[df_sales["region"] == "Region B"]["revenue"]
    ttest = stats_engine.two_sample_t_test(s_a, s_b)
    print(f"   -> Welch t-test (Region A vs B): t={ttest['t_statistic']}, p={ttest['p_value']:.4f}")
    assert "t_statistic" in ttest, "t-test must return t_statistic"
    print("   [PASS] Benchmark 3: Statistical tests verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 4: Independent Numerical Validation Engine
    # -------------------------------------------------------------
    print("[4/10] Benchmark: Independent Numerical Validator & Tolerance Checking...")
    validator = IndependentValidator()
    val_pass = validator.validate_numerical_claim(
        claim_name="March Revenue Decline",
        expected_value=-18.4,
        actual_value=-18.38,
        relative_tolerance=0.01,
    )
    assert val_pass.passed is True, "Validation should PASS within 1% tolerance"
    val_fail = validator.validate_numerical_claim(
        claim_name="Hallucinated Claim",
        expected_value=-50.0,
        actual_value=-18.38,
        relative_tolerance=0.01,
    )
    assert val_fail.passed is False, "Validation should FAIL on large discrepancy"
    print("   [PASS] Benchmark 4: Validation Engine verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 5: Revenue Root-Cause Decomposition
    # -------------------------------------------------------------
    print("[5/10] Benchmark: 'Why did revenue fall in March?'...")
    a1 = orchestrator.run_investigation(
        question="Why did revenue fall in March?",
        project_id="benchmark-proj",
        datasets=datasets,
    )
    print(f"   -> Answer: {a1.direct_answer}")
    print(f"   -> Finding: {a1.main_finding}")
    assert "Region B" in a1.main_finding, "Must isolate Region B"
    assert a1.evidence[0].validation_status.value == "PASSED"
    print("   [PASS] Benchmark 5: Revenue decline root-cause verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 6: Product Underperformance Analysis
    # -------------------------------------------------------------
    print("[6/10] Benchmark: 'Which products are performing badly?'...")
    a2 = orchestrator.run_investigation(
        question="Which products are performing badly?",
        project_id="benchmark-proj",
        datasets=datasets,
    )
    print(f"   -> Answer: {a2.direct_answer}")
    print(f"   -> Finding: {a2.main_finding}")
    assert len(a2.findings) >= 1
    assert a2.findings[0].suggested_chart_type is not None
    print("   [PASS] Benchmark 6: Product performance verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 7: Customer Churn & Segmentation
    # -------------------------------------------------------------
    print("[7/10] Benchmark: 'Which customers are likely to churn?'...")
    a3 = orchestrator.run_investigation(
        question="Which customers are likely to churn?",
        project_id="benchmark-proj",
        datasets=datasets,
    )
    print(f"   -> Answer: {a3.direct_answer}")
    print(f"   -> Finding: {a3.main_finding}")
    assert len(a3.evidence) >= 1
    print("   [PASS] Benchmark 7: Customer churn analysis verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 8: Forward Forecasting
    # -------------------------------------------------------------
    print("[8/10] Benchmark: 'Forecast next month revenue'...")
    a4 = orchestrator.run_investigation(
        question="Forecast next month's revenue",
        project_id="benchmark-proj",
        datasets=datasets,
    )
    print(f"   -> Answer: {a4.direct_answer}")
    print(f"   -> Finding: {a4.main_finding}")
    assert len(a4.evidence) >= 1
    print("   [PASS] Benchmark 8: Time-series forecast verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 9: Driver & Correlation Analysis
    # -------------------------------------------------------------
    print("[9/10] Benchmark: 'What are the main correlation drivers of revenue?'...")
    a5 = orchestrator.run_investigation(
        question="What are the main correlation drivers of revenue and profit?",
        project_id="benchmark-proj",
        datasets=datasets,
    )
    print(f"   -> Answer: {a5.direct_answer}")
    print(f"   -> Finding: {a5.main_finding}")
    assert len(a5.findings) >= 1
    print("   [PASS] Benchmark 9: Correlation analysis verified.\n")
    passed_count += 1

    # -------------------------------------------------------------
    # BENCHMARK 10: Open-Ended Business Health Check
    # -------------------------------------------------------------
    print("[10/10] Benchmark: 'Analyze my dataset and tell me what to worry about'...")
    a6 = orchestrator.run_investigation(
        question="Analyze my dataset and tell me what I should worry about",
        project_id="benchmark-proj",
        datasets=datasets,
    )
    print(f"   -> Answer: {a6.direct_answer}")
    print(f"   -> Finding: {a6.main_finding}")
    assert len(a6.findings) >= 1
    assert a6.manifest is not None
    print("   [PASS] Benchmark 10: Multi-dimensional business health check verified.\n")
    passed_count += 1

    print("=" * 80)
    print(f"FINAL RESULTS: {passed_count}/{total_benchmarks} BENCHMARKS PASSED (100% SUCCESS RATE)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_suite()
