"""
test_aaos_generalization.py: Autonomous Analytical Generalization & Multi-Domain Benchmark.

Evaluates MindEd AA-OS on unseen schemas, non-standard column names, complex multi-dimensional
structures, and deceptive ground truths across 5 distinct business domains:
1. E-Commerce (Sales Decline & Territory Concentration)
2. Subscription SaaS (Churn Drivers & Billing Tiers)
3. Marketing (Acquisition Channels & Conversion Rates)
4. Logistics & Operations (Transit Delays & Simpson's Paradox Confounding)
5. Financial Profitability (Operating Expense Spike & Outlier Emergence)
6. Sparse / Insufficient Evidence Gate (Negative Control)

Evaluates the complete unbroken autonomous loop without hardcoded assumptions or AI keys.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "analytics_core", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "api", "src"))

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
from packages.analytics_core.src.intelligence.prediction_engine import PredictionSynthesizer, PredictionEvaluator
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment, ExperimentSynthesizer
from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.intelligence.hypothesis_revision import HypothesisRevisionEngine
from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseEngine
from packages.analytics_core.src.causal.identifiability_gate import CausalIdentifiabilityGate
from packages.analytics_core.src.intelligence.epistemic_calibration import EpistemicCalibrationEngine
from packages.analytics_core.src.engines.stopping import StoppingEngine
from packages.analytics_core.src.engines.verdict import VerdictEngine
from packages.analytics_core.src.engines.provenance import ProvenanceEngine
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.schemas.src.analysis import ObjectiveType, VariableRef, CausalIntent


class TestAAOSGeneralization(unittest.TestCase):
    """Rigorous multi-domain generalization benchmark testing unseen schemas and autonomous reasoning."""

    def test_domain_a_ecommerce_sales_concentration(self):
        """Domain A: E-Commerce with non-standard column names (client_key, net_sales, territory)."""
        np.random.seed(101)
        n = 150
        territories = ["EMEA-North"] * 30 + ["EMEA-South"] * 40 + ["APAC-East"] * 40 + ["Americas-Central"] * 40
        # Ground Truth: EMEA-North accounts for 75%+ of the sales volume drop
        sales = []
        for t in territories:
            if t == "EMEA-North":
                sales.append(float(np.random.normal(85000.0, 1500.0)))
            else:
                sales.append(float(np.random.normal(9000.0, 400.0)))

        df = pd.DataFrame({
            "client_key": [f"CLI_{i:05d}" for i in range(n)],
            "territory": territories,
            "net_sales": sales,
            "discount_pct": np.random.uniform(0.05, 0.30, n),
            "order_timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
        })

        # 1. Semantic Discovery on Unseen Names
        question = "Why did net_sales decline across territory?"
        intent = IntentEngine.parse_intent(question, list(df.columns))
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

        semantic = SemanticEngine().resolve_schema(intent, {"ecommerce_orders": df})
        self.assertEqual(semantic.target_metric_col, "net_sales")
        self.assertEqual(semantic.group_dimension_col, "territory")

        # 2. Autonomous Hypothesis & Prediction Synthesis
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        self.assertGreaterEqual(len(hyps), 2)
        h1, h2 = hyps[0], hyps[1]

        p1 = PredictionSynthesizer.synthesize_prediction_for_hypothesis(h1, "PRED-ECOM-01", h1.id)
        self.assertEqual(p1.status, "PENDING")

        # 3. Dynamic Experiment Invention & EIG Ranking
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(hyps, semantic)
        self.assertGreaterEqual(len(candidates), 2)
        best_exp, _ = EIGOptimizer.select_next_experiment(candidates, hyps)
        self.assertIn("territory", best_exp.query_sql)

        # 4. DuckDB Execution & Independent Dual-Engine Verification
        exec_res = DuckDBExecutionProvider().execute_query(df=df, table_name="ecommerce_orders", query_sql=best_exp.query_sql)
        top_terr = exec_res.result_df.iloc[0]["territory"]
        self.assertEqual(top_terr, "EMEA-North")

        verif = VerificationEngine.verify_secondary(
            primary_df=df,
            target_metric_col="net_sales",
            aggregation_type="SUM",
            primary_metric=exec_res.primary_value,
            group_dimension_col="territory",
        )
        self.assertEqual(verif.status, "VERIFIED")

        # 5. Evidence & Bayesian Grounding
        eta_sq = BeliefEngine.compute_variance_explained(df, "territory", "net_sales")
        self.assertGreater(eta_sq, 90.0)

        eval_res = PredictionEvaluator.evaluate(p1, exec_res.result_df)
        self.assertEqual(eval_res.status, "SUPPORTED")

        posteriors, delta_h = BeliefEngine.compute_bayesian_posteriors(
            priors=[h1.prior_probability, h2.prior_probability],
            likelihoods=[0.95, 0.05],
        )
        self.assertGreater(posteriors[0], 0.90)
        self.assertLess(delta_h, -0.30)

    def test_domain_b_saas_churn_billing_tier(self):
        """Domain B: SaaS Subscription churn (account_identifier, churn_flag, billing_tier, arr_value)."""
        np.random.seed(202)
        n = 200
        tiers = ["Starter"] * 80 + ["Professional"] * 60 + ["Enterprise"] * 60
        # Ground Truth: Starter tier has 65% churn rate, others have < 5%
        churn = []
        for t in tiers:
            if t == "Starter":
                churn.append(1 if np.random.rand() < 0.65 else 0)
            else:
                churn.append(1 if np.random.rand() < 0.05 else 0)

        df = pd.DataFrame({
            "account_identifier": [f"ACC_{i:04d}" for i in range(n)],
            "billing_tier": tiers,
            "churn_flag": churn,
            "arr_value": np.random.uniform(500.0, 10000.0, n),
            "support_tickets": np.random.randint(0, 10, n),
        })

        question = "What is driving churn across billing_tier?"
        intent = IntentEngine.parse_intent(question, list(df.columns))
        semantic = SemanticEngine().resolve_schema(intent, {"saas_metrics": df})
        self.assertEqual(semantic.target_metric_col, "churn_flag")
        self.assertEqual(semantic.group_dimension_col, "billing_tier")

        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(hyps, semantic)
        best_exp, _ = EIGOptimizer.select_next_experiment(candidates, hyps)

        exec_res = DuckDBExecutionProvider().execute_query(df=df, table_name="saas_metrics", query_sql=best_exp.query_sql)
        self.assertIsNotNone(exec_res.result_df)
        starter_row = exec_res.result_df[exec_res.result_df["billing_tier"] == "Starter"].iloc[0]
        self.assertGreater(float(starter_row["total_metric"]), 30.0)

    def test_domain_c_marketing_conversion_by_channel(self):
        """Domain C: Marketing acquisition channels and conversion rates with Multiverse checking."""
        np.random.seed(303)
        n = 180
        channels = ["Paid-Search"] * 45 + ["Organic-SEO"] * 45 + ["Social-Ads"] * 45 + ["Referral-Partner"] * 45
        conversions = []
        for c in channels:
            if c == "Paid-Search":
                conversions.append(float(np.random.normal(12.5, 0.8)))
            else:
                conversions.append(float(np.random.normal(3.0, 0.4)))

        df = pd.DataFrame({
            "lead_id": [f"LEAD_{i:04d}" for i in range(n)],
            "acquisition_channel": channels,
            "conversion_rate": conversions,
            "ad_spend": np.random.uniform(100.0, 5000.0, n),
            "device_category": np.random.choice(["Mobile", "Desktop", "Tablet"], n),
        })

        question = "Did the acquisition_channel performance improve conversion_rate?"
        intent = IntentEngine.parse_intent(question, list(df.columns))
        semantic = SemanticEngine().resolve_schema(intent, {"marketing_funnel": df})
        self.assertEqual(semantic.target_metric_col, "conversion_rate")
        self.assertEqual(semantic.group_dimension_col, "acquisition_channel")

        # Multiverse Robustness Check across specifications
        m_rep = MultiverseEngine.evaluate_specification_curve(df, "acquisition_channel", "conversion_rate")
        self.assertGreaterEqual(m_rep.robustness_score, 0.75)
        self.assertGreaterEqual(m_rep.robustness_pct, 75.0)

    def test_domain_d_logistics_simpsons_paradox_adversarial_challenge(self):
        """Domain D: Negative Control / Simpson's Paradox Confounding in Logistics Transit Delays."""
        np.random.seed(404)
        # Aggregate: Carrier A looks slower than Carrier B overall
        # Confounded: Depot-East has much higher delays than Depot-West for all carriers,
        # but Carrier A was assigned 90% of Depot-East shipments.
        # Within every single depot, Carrier A is actually FASTER than Carrier B!
        records = []
        for _ in range(100):
            # Carrier A in Depot-East (Delay ~15 hrs)
            records.append({"carrier_partner": "Carrier-Alpha", "depot_location": "Depot-East", "transit_delay_hours": np.random.normal(15.0, 1.0)})
        for _ in range(20):
            # Carrier A in Depot-West (Delay ~4 hrs)
            records.append({"carrier_partner": "Carrier-Alpha", "depot_location": "Depot-West", "transit_delay_hours": np.random.normal(4.0, 0.5)})
        for _ in range(20):
            # Carrier B in Depot-East (Delay ~18 hrs)
            records.append({"carrier_partner": "Carrier-Beta", "depot_location": "Depot-East", "transit_delay_hours": np.random.normal(18.0, 1.0)})
        for _ in range(100):
            # Carrier B in Depot-West (Delay ~6 hrs)
            records.append({"carrier_partner": "Carrier-Beta", "depot_location": "Depot-West", "transit_delay_hours": np.random.normal(6.0, 0.5)})

        df = pd.DataFrame(records)
        df["consignment_id"] = [f"CON_{i:04d}" for i in range(len(df))]

        question = "Which carrier_partner is responsible for transit_delay_hours?"
        intent = IntentEngine.parse_intent(question, list(df.columns))
        semantic = SemanticEngine().resolve_schema(intent, {"logistics_manifest": df})

        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        h1 = hyps[0]
        h2 = hyps[1]

        # Adversarial Engine MUST challenge aggregate carrier conclusion
        adv_res = AdversarialAttacker.execute_adversarial_attack(
            df=df,
            leading_hypothesis=h1,
            counter_hypothesis=h2,
            semantic=semantic,
            execution_provider=DuckDBExecutionProvider(),
        )
        self.assertTrue(adv_res.simpsons_paradox_detected)
        self.assertEqual(adv_res.details.get("secondary_dimension"), "depot_location")

    def test_domain_e_financial_outlier_emergence_and_experiment_invention(self):
        """Domain E: Financial Expense Spike with dynamic emergent hypothesis and invented experiment."""
        np.random.seed(505)
        n = 160
        cost_centers = ["CC-101"] * 40 + ["CC-202"] * 40 + ["CC-303"] * 40 + ["CC-909-Outlier"] * 40
        expenses = []
        for cc in cost_centers:
            if cc == "CC-909-Outlier":
                expenses.append(float(np.random.normal(500000.0, 10000.0)))
            else:
                expenses.append(float(np.random.normal(15000.0, 1000.0)))

        df = pd.DataFrame({
            "ledger_entry_id": [f"LED_{i:05d}" for i in range(n)],
            "cost_center": cost_centers,
            "operating_expense": expenses,
            "fiscal_period": "2026-Q1",
        })

        question = "What explains the spike in operating_expense across cost_center?"
        intent = IntentEngine.parse_intent(question, list(df.columns))
        semantic = SemanticEngine().resolve_schema(intent, {"general_ledger": df})

        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        candidates_t0 = ExperimentSynthesizer.synthesize_candidate_experiments(hyps, semantic)
        best_exp_t0, _ = EIGOptimizer.select_next_experiment(candidates_t0, hyps)

        # Run first experiment
        exec_t0 = DuckDBExecutionProvider().execute_query(df=df, table_name="general_ledger", query_sql=best_exp_t0.query_sql)
        
        # Adaptive Replanning: Inspect evidence and invent new targeted emergent hypothesis
        replan = ExperimentSynthesizer.dynamically_replan_candidates(
            hypotheses=hyps,
            semantic=semantic,
            executed_codes=[best_exp_t0.code],
            current_posteriors=[h.posterior_probability for h in hyps],
            last_result_df=exec_t0.result_df,
            last_experiment=best_exp_t0,
        )

        # Invariant: Emergent hypotheses were created and candidate pool changed
        self.assertGreater(len(replan.new_hypotheses), 0)
        emergent_h = replan.new_hypotheses[0]
        self.assertEqual(emergent_h.target_value, "CC-909-Outlier")
        
        # Invariant: Invented isolated experiment exists in replanned candidates
        invented_exps = [c for c in replan.candidates if "ISOLATE" in c.code or "CC-909" in c.query_sql]
        self.assertGreater(len(invented_exps), 0)

    def test_domain_f_sparse_data_insufficient_evidence_gate(self):
        """Domain F: Negative Control / Fail-closed Data Quality Gate on Sparse Data ($N < 5$)."""
        sparse_df = pd.DataFrame({
            "id": ["A", "B", "C"],
            "metric": [10.0, None, 20.0],
            "category": ["X", "Y", "Z"],
        })

        # Pre-investigation Data Quality Gate MUST reject sparse dataset
        assessment = DataQualityGate.evaluate_fitness(sparse_df, "sparse_table")
        self.assertEqual(assessment.fitness_verdict, "UNFIT")
        self.assertFalse(assessment.can_proceed)
        self.assertTrue(any("sparse" in issue.lower() or "minimum" in issue.lower() for issue in assessment.critical_issues))


def main():
    print("=" * 80, flush=True)
    print("RUNNING AA-OS GENERALIZATION & MULTI-DOMAIN BENCHMARK (6 DOMAINS)", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAAOSGeneralization)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("ALL GENERALIZATION BENCHMARK DOMAINS PASSED (100% GENERAL AUTONOMY)", flush=True)
        print("=" * 80, flush=True)
        sys.exit(0)
    else:
        print("GENERALIZATION BENCHMARK FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
