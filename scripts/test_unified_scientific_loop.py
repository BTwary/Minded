"""test_unified_scientific_loop.py: 7 Golden Tests for AA-OS Internally Coherent Scientific Loop.

Tests:
1. TEST 1 — PREDICTION -> EXPERIMENT LINK
2. TEST 2 — MULTIPLE EVIDENCE TYPES (Concentration, Temporal, Segment Difference, Anomaly)
3. TEST 3 — IMMEDIATE STATE FAILURE (Fail-Closed on Inconsistency)
4. TEST 4 — CANONICAL STATE AUTHORITY (Deduplication & Single Source of Truth)
5. TEST 5 — ADVERSARIAL CANDIDATE COMPETITION
6. TEST 6 — ROBUSTNESS CANDIDATE COMPETITION
7. TEST 7 — THREE-ITERATION AUTONOMOUS LOOP (Real Execution Trace Proof)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

# Adjust module paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.engines.semantic import SemanticEngine, SemanticResolution
from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    HypothesisSynthesizer,
    PredictiveHypothesis,
)
from packages.analytics_core.src.intelligence.prediction_engine import (
    PredictionSynthesizer,
    PredictionEvaluator,
    StructuredPrediction,
)
from packages.analytics_core.src.intelligence.evidence_patterns import (
    ConcentrationPatternDetector,
    SegmentDifferenceDetector,
    TemporalChangeDetector,
    AnomalyPatternDetector,
    ACTIVE_DETECTORS,
)
from packages.analytics_core.src.intelligence.experiment_synthesizer import (
    CandidateExperiment,
    ExperimentSynthesizer,
)
from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.intelligence.state_validation import validate_state
from packages.analytics_core.src.intelligence.transition import ScientificTransitionService
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.analytics_core.src.engines.stopping import StoppingEngine
from packages.schemas.src.analysis import (
    ObjectiveType,
    FirstClassExperiment,
    PredictionEvidenceRecord,
    RawObservationRecord,
)


class TestUnifiedScientificLoop(unittest.TestCase):
    """Validation of 7 Golden Invariants for Autonomous Scientific Coherence."""

    def setUp(self):
        # Create standard synthetic benchmark data
        np.random.seed(42)
        n = 1000
        segments = np.random.choice(["Electronics", "Apparel", "Home", "Beauty"], size=n, p=[0.60, 0.20, 0.10, 0.10])
        regions = np.random.choice(["North", "South", "East", "West"], size=n)
        revenue = []
        for s in segments:
            if s == "Electronics":
                revenue.append(np.random.normal(500, 50))
            elif s == "Apparel":
                revenue.append(np.random.normal(150, 30))
            else:
                revenue.append(np.random.normal(80, 20))

        self.df = pd.DataFrame({
            "segment": segments,
            "region": regions,
            "revenue": revenue,
            "time_period": np.random.choice(["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"], size=n),
        })

        self.semantic = SemanticResolution(
            primary_dataset_name="sales_table",
            target_metric_col="revenue",
            group_dimension_col="segment",
            time_col="time_period",
            table_grain="segment",
            available_categorical_cols=["segment", "region", "time_period"],
            available_numeric_cols=["revenue"],
        )

    # -------------------------------------------------------------------------
    # TEST 1: Prediction -> Experiment Linkage
    # -------------------------------------------------------------------------
    def test_01_prediction_experiment_link(self):
        """Verify candidate experiments carry explicit prediction_ids and evaluate them upon execution."""
        state_mgr = InvestigationStateManager("INV-TEST-01", "Why did revenue change?", ObjectiveType.DESCRIBE)

        h1 = PredictiveHypothesis(
            id="HYP-01",
            hypothesis_code="HYP-01",
            claim="Electronics segment is driving the revenue concentration.",
            mechanism="Dominant categorical share.",
            predicted_observables_if_true=["Top segment accounts for >= 50%"],
            predicted_observables_if_false=["Uniform distribution"],
            falsification_criteria="Top share < 35%",
            required_assumptions=["Accurate segment reporting"],
            prior_probability=0.70,
            posterior_probability=0.70,
            target_metric="revenue",
            target_dimension="segment",
            target_value="Electronics",
        )
        state_mgr.create_hypothesis(h1)

        pred1 = PredictionSynthesizer.synthesize_prediction_for_hypothesis(h1, "PRED-001", "INV-TEST-01_HYP-01")
        state_mgr.create_prediction(pred1)
        h1.prediction_ids.append(pred1.prediction_id)

        # Candidate generation must explicitly link prediction_ids
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            hypotheses=[h1],
            semantic=self.semantic,
            predictions_by_hyp={"HYP-01": "PRED-001"},
        )

        # Find the experiment targeting this hypothesis
        target_exp = next((c for c in candidates if c.target_hypothesis_code == "HYP-01"), None)
        self.assertIsNotNone(target_exp)
        self.assertIn("PRED-001", target_exp.prediction_ids)
        self.assertEqual(target_exp.target_prediction_id, "PRED-001")
        self.assertFalse(target_exp.is_exploratory)

        # Execute transition on the linked experiment
        result_df = self.df.groupby("segment")["revenue"].sum().reset_index()
        primary_val = float(result_df["revenue"].sum())

        transition = ScientificTransitionService.apply_post_execution_transition(
            state_mgr,
            experiment_id="EXP-001",
            experiment_code="EXP-CONC",
            target_prediction_ids=target_exp.prediction_ids,
            target_hypothesis_code="HYP-01",
            result_df=result_df,
            primary_value=primary_val,
            primary_df=self.df,
            target_metric_col="revenue",
            group_dimension_col="segment",
            aggregation_type="SUM",
            effective_sql="SELECT segment, SUM(revenue) FROM sales_table GROUP BY segment",
            is_grouped=True,
        )

        # Verify prediction was actually evaluated
        self.assertEqual(len(transition.evaluated_predictions), 1)
        eval_record = transition.evaluated_predictions[0]
        self.assertEqual(eval_record.prediction_id, "PRED-001")
        self.assertIn(eval_record.status, ["SUPPORTED", "REFUTED", "INCONCLUSIVE"])

        # Check canonical state updated
        persisted_pred = state_mgr.get_prediction("PRED-001")
        self.assertEqual(persisted_pred.status, eval_record.status)
        print(f"\n[TEST 1 PASSED] Prediction {eval_record.prediction_id} evaluated to {eval_record.status} via experiment {target_exp.code}")

    # -------------------------------------------------------------------------
    # TEST 2: Multiple Evidence Types (Concentration, Segment Diff, Temporal, Anomaly)
    # -------------------------------------------------------------------------
    def test_02_multiple_evidence_types(self):
        """Verify detectors produce distinct pattern-specific emergent hypotheses with preserved provenance."""
        # 1. Concentration Pattern
        conc_df = pd.DataFrame({
            "segment": ["Electronics", "Books"],
            "revenue": [9000.0, 1000.0],
        })
        conc_det = ConcentrationPatternDetector()
        conc_expl = conc_det.detect(conc_df, {})
        self.assertIsNotNone(conc_expl)
        self.assertEqual(conc_expl.pattern_name, "concentration")

        # 2. Segment Difference Pattern
        diff_df = pd.DataFrame({
            "segment": ["Electronics", "Apparel", "Home", "Beauty"],
            "revenue": [1000.0, 20.0, 15.0, 10.0],
        })
        diff_det = SegmentDifferenceDetector()
        diff_expl = diff_det.detect(diff_df, {})
        self.assertIsNotNone(diff_expl)
        self.assertEqual(diff_expl.pattern_name, "segment_difference")

        # 3. Temporal Change Pattern
        temp_df = pd.DataFrame({
            "time_period": ["2025-01", "2025-02", "2025-03", "2025-04"],
            "revenue": [100.0, 110.0, 850.0, 870.0],
        })
        temp_det = TemporalChangeDetector()
        temp_expl = temp_det.detect(temp_df, {})
        self.assertIsNotNone(temp_expl)
        self.assertEqual(temp_expl.pattern_name, "temporal_change")

        # 4. Anomaly Pattern
        anom_df = pd.DataFrame({
            "region": ["North", "South", "East", "West", "Outlier_Spike"],
            "revenue": [100.0, 105.0, 95.0, 102.0, 10000.0],
        })
        anom_det = AnomalyPatternDetector()
        anom_expl = anom_det.detect(anom_df, {})
        self.assertIsNotNone(anom_expl)
        self.assertEqual(anom_expl.pattern_name, "anomaly")

        # Synthesize hypotheses from all 4 patterns
        patterns = [
            {"type": "concentration", "dimension": "segment", "metric": "revenue", "top_value": "Electronics", "top_share_pct": 90.0, "source_experiment_id": "EXP-01"},
            {"type": "segment_difference", "dimension": "segment", "metric": "revenue", "top_value": "Electronics", "top_share_pct": 85.0, "source_experiment_id": "EXP-02"},
            {"type": "temporal_change", "dimension": "time_period", "metric": "revenue", "top_value": "2025-03", "top_share_pct": 75.0, "source_experiment_id": "EXP-03"},
            {"type": "anomaly", "dimension": "region", "metric": "revenue", "top_value": "Outlier_Spike", "top_share_pct": 98.0, "source_experiment_id": "EXP-04"},
        ]

        hyps = HypothesisSynthesizer.synthesize_emergent_hypotheses(patterns, self.semantic, [])
        # Item 1 (canonical hypothesis identity + consolidation): patterns 1
        # and 2 are both localized-effect claims about the SAME dimension
        # ('segment') and SAME value ('Electronics') for the SAME metric --
        # they are the same substantive claim discovered by two different
        # detectors, and must consolidate into one canonical hypothesis
        # rather than remaining duplicates. Patterns 3 (different dimension,
        # temporal_shift mechanism) and 4 (different dimension and value,
        # localized_effect on 'region') are genuinely distinct claims and
        # must remain separate. So: 4 patterns -> 3 canonical hypotheses.
        self.assertEqual(len(hyps), 3)

        # Check distinct non-collapsed claims
        claims = [h.claim for h in hyps]
        self.assertTrue(any("Concentration in segment" in c for c in claims))
        self.assertTrue(any("temporal shift" in c for c in claims))
        self.assertTrue(any("Anomalous localized outlier" in c for c in claims))

        # The segment_difference pattern's claim ("Significant variance
        # disparity...") no longer surfaces as its own top-level hypothesis
        # -- it was consolidated into the concentration hypothesis on the
        # same segment/value. Its provenance must still be recoverable
        # rather than silently dropped.
        merged = next(h for h in hyps if "Concentration in segment" in h.claim)
        self.assertEqual(len(merged.consolidation_log), 2)
        self.assertTrue(
            any("Significant variance disparity" in c.get("claim", "") for c in merged.consolidation_log)
        )
        self.assertEqual(sorted(merged.source_evidence), ["EXP-01", "EXP-02"])

        for h in hyps:
            self.assertTrue(len(h.source_evidence) > 0)
            self.assertTrue(len(h.generated_reason) > 0)

        print("\n[TEST 2 PASSED] Discovered 3 canonical pattern-derived hypotheses "
              "(same-segment concentration + segment_difference correctly consolidated) "
              "with preserved provenance.")

    # -------------------------------------------------------------------------
    # TEST 3: Immediate State Failure (Fail-Closed on Inconsistency)
    # -------------------------------------------------------------------------
    def test_03_immediate_state_failure(self):
        """Verify validate_state detects state corruption immediately and fails closed."""
        state_mgr = InvestigationStateManager("INV-TEST-03", "Why did revenue change?", ObjectiveType.DESCRIBE)

        h1 = PredictiveHypothesis(
            id="HYP-01",
            hypothesis_code="HYP-01",
            claim="Electronics concentration.",
            mechanism="High volume.",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=0.5,
            posterior_probability=0.5,
        )
        state_mgr.create_hypothesis(h1)

        pred1 = PredictionSynthesizer.synthesize_prediction_for_hypothesis(h1, "PRED-001", "INV-TEST-03_HYP-01")
        state_mgr.create_prediction(pred1)

        # Case A: Valid initial state
        initial_check = state_mgr.validate_consistency()
        self.assertTrue(initial_check.is_consistent)

        # Case B: Inject corruption (duplicate prediction ID)
        state_mgr.state.predictions.append(pred1)
        corrupted_check = state_mgr.validate_consistency()
        self.assertFalse(corrupted_check.is_consistent)
        self.assertTrue(any(e.error_code == "duplicate_prediction_id" for e in corrupted_check.errors))

        # Case C: Inject dangling prediction
        state_mgr.state.predictions = [pred1]  # restore clean
        pred_dangling = StructuredPrediction(
            prediction_id="PRED-DANGLING",
            hypothesis_id="HYP-NONEXISTENT",
            hypothesis_code="HYP-NONEXISTENT",
            statement="Nonexistent hypothesis prediction.",
            target_metric="revenue",
            target_dimension="segment",
        )
        state_mgr.state.predictions.append(pred_dangling)
        dangling_check = state_mgr.validate_consistency()
        self.assertFalse(dangling_check.is_consistent)
        self.assertTrue(any(e.error_code == "prediction_references_missing_hypothesis" for e in dangling_check.errors))

        print("\n[TEST 3 PASSED] State consistency validator detected corruption immediately and failed closed.")

    # -------------------------------------------------------------------------
    # TEST 4: Canonical State Authority (Deduplication & Execution State)
    # -------------------------------------------------------------------------
    def test_04_canonical_state_authority(self):
        """Verify experiment deduplication and state mutations rely purely on canonical state."""
        state_mgr = InvestigationStateManager("INV-TEST-04", "Test question", ObjectiveType.DESCRIBE)

        h1 = PredictiveHypothesis(
            id="HYP-01",
            hypothesis_code="HYP-01",
            claim="Electronics concentration.",
            mechanism="High volume.",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=0.5,
            posterior_probability=0.5,
            target_metric="revenue",
            target_dimension="segment",
        )
        state_mgr.create_hypothesis(h1)

        cand = CandidateExperiment(
            code="EXP-CONC",
            target_hypothesis_code="HYP-01",
            tool_name="duckdb_sql",
            query_sql="SELECT segment, SUM(revenue) FROM sales_table GROUP BY segment",
            description="Concentration test",
            aggregation_type="SUM",
            target_dimension="segment",
            target_metric="revenue",
        )

        # Before execution: not in canonical state
        self.assertFalse(state_mgr.has_experiment_been_executed("EXP-CONC", cand.fingerprint))

        # Record execution in canonical state
        state_mgr.record_experiment_executed("EXP-CONC", fingerprint=cand.fingerprint)

        # After execution: authoritative check must return True
        self.assertTrue(state_mgr.has_experiment_been_executed("EXP-CONC"))
        self.assertTrue(state_mgr.has_experiment_been_executed("OTHER_NAME", fingerprint=cand.fingerprint))
        self.assertIn("EXP-CONC", state_mgr.get_executed_experiments())
        self.assertIn(cand.fingerprint, state_mgr.get_executed_fingerprints())

        # EIG Optimizer deduplication must filter out this candidate
        scored = EIGOptimizer.score_candidate_experiments(
            candidates=[cand],
            hypotheses=[h1],
            executed_codes=state_mgr.get_executed_experiments(),
            executed_fingerprints=state_mgr.get_executed_fingerprints(),
        )
        self.assertEqual(len(scored), 0)
        print("\n[TEST 4 PASSED] Canonical state authority verified for experiment deduplication.")

    # -------------------------------------------------------------------------
    # TEST 5: Adversarial Candidate Competition
    # -------------------------------------------------------------------------
    def test_05_adversarial_candidate_competition(self):
        """Verify adversarial candidates compete in the unified candidate pool with traceable score breakdown."""
        h1 = PredictiveHypothesis(
            id="HYP-01",
            hypothesis_code="HYP-01",
            claim="Electronics concentration.",
            mechanism="Dominant volume.",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=0.75,
            posterior_probability=0.75,
            target_metric="revenue",
            target_dimension="segment",
        )

        standard_exp = CandidateExperiment(
            code="EXP-STANDARD",
            target_hypothesis_code="HYP-01",
            tool_name="duckdb_sql",
            query_sql="SELECT segment, SUM(revenue) FROM sales_table GROUP BY segment",
            description="Standard volume test",
            aggregation_type="SUM",
            discriminating_power=0.40,
            adversarial_value=0.0,
        )

        adversarial_exp = CandidateExperiment(
            code="EXP-COND-REGION",
            target_hypothesis_code="HYP-01",
            tool_name="duckdb_sql",
            query_sql="SELECT segment, region, SUM(revenue) FROM sales_table GROUP BY segment, region",
            description="Conditional test on region to address Simpson's paradox",
            aggregation_type="SUM",
            discriminating_power=0.95,
            adversarial_value=0.95,
            target_uncertainty="Simpson's paradox challenge",
            provenance={"attack_type": "simpsons_paradox", "confounding_dimension": "region"},
        )

        # Unified scoring
        scored = EIGOptimizer.score_candidate_experiments(
            candidates=[standard_exp, adversarial_exp],
            hypotheses=[h1],
        )

        self.assertEqual(len(scored), 2)
        # Adversarial candidate should rank #1 due to composite information gain + adversarial value
        top_exp = scored[0].experiment
        self.assertEqual(top_exp.code, "EXP-COND-REGION")
        self.assertIn("Adversarial Value=0.95", top_exp.selection_rationale)
        print(f"\n[TEST 5 PASSED] Adversarial candidate won selection with rationale: {top_exp.selection_rationale}")

    # -------------------------------------------------------------------------
    # TEST 6: Robustness Candidate Competition
    # -------------------------------------------------------------------------
    def test_06_robustness_candidate_competition(self):
        """Verify candidate scoring incorporates multiverse specification robustness values."""
        h1 = PredictiveHypothesis(
            id="HYP-01",
            hypothesis_code="HYP-01",
            claim="Electronics concentration.",
            mechanism="Dominant volume.",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=0.60,
            posterior_probability=0.60,
            target_metric="revenue",
            target_dimension="segment",
        )

        exp1 = CandidateExperiment(
            code="EXP-UNROBUST",
            target_hypothesis_code="HYP-01",
            tool_name="duckdb_sql",
            query_sql="SELECT segment, SUM(revenue) FROM sales_table GROUP BY segment",
            description="Test with low robustness",
            aggregation_type="SUM",
            discriminating_power=0.50,
            robustness_value=0.10,
        )

        exp2 = CandidateExperiment(
            code="EXP-ROBUST",
            target_hypothesis_code="HYP-01",
            tool_name="duckdb_sql",
            query_sql="SELECT segment, SUM(revenue) FROM sales_table GROUP BY segment",
            description="Test verified across specification curves",
            aggregation_type="SUM",
            discriminating_power=0.50,
            robustness_value=0.95,
        )

        scored = EIGOptimizer.score_candidate_experiments([exp1, exp2], [h1])
        self.assertEqual(scored[0].experiment.code, "EXP-ROBUST")
        self.assertIn("Robustness Value=0.95", scored[0].experiment.selection_rationale)
        print(f"\n[TEST 6 PASSED] Robustness value successfully ranked candidate with rationale: {scored[0].experiment.selection_rationale}")

    # -------------------------------------------------------------------------
    # TEST 7: Three-Iteration Autonomous Loop (Real Execution Trace Proof)
    # -------------------------------------------------------------------------
    def test_07_three_iteration_autonomous_loop(self):
        """Execute a full 3-iteration recursive investigation loop demonstrating real state transitions."""
        print("\n" + "=" * 80)
        print("TEST 7: THREE-ITERATION RECURSIVE AUTONOMOUS INVESTIGATION TRACE")
        print("=" * 80)

        investigation_id = "INV-AUTONOMOUS-PROOF-07"
        question = "What is driving the variance in revenue across customer segments?"
        state_mgr = InvestigationStateManager(investigation_id, question, ObjectiveType.DESCRIBE)

        # ---------------------------------------------------------------------
        # Turn 0: Prior Initialization & Objective Decomposition
        # ---------------------------------------------------------------------
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(self.semantic, question)
        state_mgr.sync_hypotheses(hyps)

        # Synthesize predictions for each hypothesis
        for idx, h in enumerate(hyps):
            pred = PredictionSynthesizer.synthesize_prediction_for_hypothesis(h, f"PRED-{idx+1:03d}", f"{investigation_id}_{h.hypothesis_code}")
            state_mgr.create_prediction(pred)
            h.prediction_ids.append(pred.prediction_id)

        initial_probs = [h.posterior_probability for h in state_mgr.get_active_hypotheses()]
        initial_entropy = compute_shannon_entropy(initial_probs)

        print(f"\n[INITIAL STATE] Hypotheses: {[h.hypothesis_code for h in hyps]}")
        print(f"                Initial Priors: {initial_probs} (Entropy: {initial_entropy:.3f} bits)")
        print(f"                Active Predictions: {[p.prediction_id for p in state_mgr.get_pending_predictions()]}")

        # ---------------------------------------------------------------------
        # Turn 1: Initial Discovery Experiment (EXP-CONC)
        # ---------------------------------------------------------------------
        import duckdb
        import pandas as pd
        conn = duckdb.connect()
        conn.register("sales_table", self.df)
        conn.register("data_table", self.df)

        print("\n--- [ITERATION 1: Primary Concentration Experiment] ---")
        candidates_t1 = ExperimentSynthesizer.synthesize_candidate_experiments(
            state_mgr.get_active_hypotheses(),
            self.semantic,
            predictions_by_hyp={h.hypothesis_code: h.prediction_ids[0] for h in state_mgr.get_active_hypotheses() if h.prediction_ids},
        )
        selected_t1, rationale_t1 = EIGOptimizer.select_next_experiment(
            candidates_t1,
            state_mgr.get_active_hypotheses(),
            executed_codes=state_mgr.get_executed_experiments(),
            executed_fingerprints=state_mgr.get_executed_fingerprints(),
        )
        print(f"Selected Experiment: {selected_t1.code}")
        print(f"Selection Rationale: {rationale_t1}")

        # Execute query dynamically via DuckDB
        res1_df = conn.execute(selected_t1.query_sql).df()
        num_cols_1 = [c for c in res1_df.columns if pd.api.types.is_numeric_dtype(res1_df[c])]
        # Grouped verification expects the primary scalar for the leading group,
        # not the sum across all groups in the grouped result relation.
        primary_val_1 = (
            float(res1_df.sort_values(num_cols_1[0], ascending=False).iloc[0][num_cols_1[0]])
            if num_cols_1 else 100.0
        )
        is_grouped_1 = "GROUP BY" in selected_t1.query_sql.upper()

        trans_1 = ScientificTransitionService.apply_post_execution_transition(
            state_mgr,
            experiment_id=f"{investigation_id}_{selected_t1.code}",
            experiment_code=selected_t1.code,
            target_prediction_ids=selected_t1.prediction_ids,
            target_hypothesis_code=selected_t1.target_hypothesis_code,
            result_df=res1_df,
            primary_value=primary_val_1,
            primary_df=self.df,
            target_metric_col="revenue",
            group_dimension_col="segment",
            aggregation_type=selected_t1.aggregation_type,
            effective_sql=selected_t1.query_sql,
            is_grouped=is_grouped_1,
        )

        state_mgr.record_experiment_executed(f"{investigation_id}_{selected_t1.code}", fingerprint=selected_t1.fingerprint)

        # In-Flight State Check
        val_t1 = state_mgr.validate_consistency()
        self.assertTrue(val_t1.is_consistent, f"State inconsistency in Iteration 1: {val_t1.errors}")

        probs_t1 = [h.posterior_probability for h in state_mgr.get_active_hypotheses()]
        ent_t1 = compute_shannon_entropy(probs_t1)
        print(f"Observation Result: {trans_1.raw_observation.structured_result}")
        print(f"Evaluated Predictions: {[(p.prediction_id, p.status) for p in trans_1.evaluated_predictions]}")
        print(f"Posteriors after Turn 1: {dict(zip([h.hypothesis_code for h in state_mgr.get_active_hypotheses()], [round(p, 3) for p in probs_t1]))} (Entropy: {ent_t1:.3f} bits)")

        # ---------------------------------------------------------------------
        # Turn 2: Dynamic Replanning & Emergent Isolated Validation
        # ---------------------------------------------------------------------
        print("\n--- [ITERATION 2: Evidence-Driven Adaptive Replanning] ---")
        replan_t2 = ExperimentSynthesizer.dynamically_replan_candidates(
            hypotheses=state_mgr.get_active_hypotheses(),
            semantic=self.semantic,
            executed_codes=state_mgr.get_executed_experiments(),
            current_posteriors=probs_t1,
            last_result_df=res1_df,
            last_experiment=selected_t1,
        )

        # Register emergent hypothesis if discovered
        self.assertTrue(len(replan_t2.new_hypotheses) > 0, "Expected emergent hypothesis from concentration pattern")
        new_h = replan_t2.new_hypotheses[0]
        state_mgr.create_hypothesis(new_h)
        new_pred = PredictionSynthesizer.synthesize_prediction_for_hypothesis(new_h, "PRED-EMERGENT-001", f"{investigation_id}_{new_h.hypothesis_code}")
        state_mgr.create_prediction(new_pred)
        new_h.prediction_ids.append(new_pred.prediction_id)

        # Re-normalize priors across all active hypotheses
        all_active = state_mgr.get_active_hypotheses()
        total_p = sum(h.posterior_probability for h in all_active)
        for h in all_active:
            h.posterior_probability /= total_p
        state_mgr.sync_hypotheses(all_active)

        print(f"Discovered Emergent Hypothesis: {new_h.hypothesis_code} -> '{new_h.claim}'")
        print(f"Trigger Reason: {replan_t2.trigger_reason}")

        # Select candidate for Turn 2
        candidates_t2 = [c for c in replan_t2.candidates if not state_mgr.has_experiment_been_executed(c.code, c.fingerprint)]
        for c in candidates_t2:
            if c.target_hypothesis_code == new_h.hypothesis_code:
                c.target_prediction_id = new_pred.prediction_id
                c.prediction_ids = [new_pred.prediction_id]

        selected_t2, rationale_t2 = EIGOptimizer.select_next_experiment(
            candidates_t2,
            state_mgr.get_active_hypotheses(),
            executed_codes=state_mgr.get_executed_experiments(),
            executed_fingerprints=state_mgr.get_executed_fingerprints(),
        )
        print(f"Selected Experiment: {selected_t2.code}")
        print(f"Selection Rationale: {rationale_t2}")

        # Execute query dynamically via DuckDB
        res2_df = conn.execute(selected_t2.query_sql).df()
        num_cols_2 = [c for c in res2_df.columns if pd.api.types.is_numeric_dtype(res2_df[c])]
        primary_val_2 = float(res2_df[num_cols_2[0]].sum()) if num_cols_2 else 100.0
        is_grouped_2 = "GROUP BY" in selected_t2.query_sql.upper()

        trans_2 = ScientificTransitionService.apply_post_execution_transition(
            state_mgr,
            experiment_id=f"{investigation_id}_{selected_t2.code}",
            experiment_code=selected_t2.code,
            target_prediction_ids=selected_t2.prediction_ids,
            target_hypothesis_code=selected_t2.target_hypothesis_code,
            result_df=res2_df,
            primary_value=primary_val_2,
            primary_df=self.df,
            target_metric_col="revenue",
            group_dimension_col="segment",
            aggregation_type=selected_t2.aggregation_type,
            effective_sql=selected_t2.query_sql,
            is_grouped=is_grouped_2,
        )
        state_mgr.record_experiment_executed(f"{investigation_id}_{selected_t2.code}", fingerprint=selected_t2.fingerprint)

        val_t2 = state_mgr.validate_consistency()
        self.assertTrue(val_t2.is_consistent, f"State inconsistency in Iteration 2: {val_t2.errors}")

        probs_t2 = [h.posterior_probability for h in state_mgr.get_active_hypotheses()]
        ent_t2 = compute_shannon_entropy(probs_t2)
        print(f"Evaluated Predictions: {[(p.prediction_id, p.status) for p in trans_2.evaluated_predictions]}")
        print(f"Posteriors after Turn 2: {dict(zip([h.hypothesis_code for h in state_mgr.get_active_hypotheses()], [round(p, 3) for p in probs_t2]))} (Entropy: {ent_t2:.3f} bits)")

        # ---------------------------------------------------------------------
        # Turn 3: Rigorous Statistical Verification
        # ---------------------------------------------------------------------
        print("\n--- [ITERATION 3: Rigorous Statistical Verification] ---")
        candidates_t3 = ExperimentSynthesizer.synthesize_candidate_experiments(
            state_mgr.get_active_hypotheses(),
            self.semantic,
            predictions_by_hyp={h.hypothesis_code: h.prediction_ids[0] for h in state_mgr.get_active_hypotheses() if h.prediction_ids},
        )
        candidates_t3 = [c for c in candidates_t3 if not state_mgr.has_experiment_been_executed(c.code, c.fingerprint)]

        selected_t3, rationale_t3 = EIGOptimizer.select_next_experiment(
            candidates_t3,
            state_mgr.get_active_hypotheses(),
            executed_codes=state_mgr.get_executed_experiments(),
            executed_fingerprints=state_mgr.get_executed_fingerprints(),
        )
        print(f"Selected Experiment: {selected_t3.code}")
        print(f"Selection Rationale: {rationale_t3}")

        # Execute query dynamically via DuckDB
        res3_df = conn.execute(selected_t3.query_sql).df()
        num_cols_3 = [c for c in res3_df.columns if pd.api.types.is_numeric_dtype(res3_df[c])]
        primary_val_3 = float(res3_df[num_cols_3[0]].mean()) if num_cols_3 else 100.0
        is_grouped_3 = "GROUP BY" in selected_t3.query_sql.upper()

        trans_3 = ScientificTransitionService.apply_post_execution_transition(
            state_mgr,
            experiment_id=f"{investigation_id}_{selected_t3.code}",
            experiment_code=selected_t3.code,
            target_prediction_ids=selected_t3.prediction_ids,
            target_hypothesis_code=selected_t3.target_hypothesis_code,
            result_df=res3_df,
            primary_value=primary_val_3,
            primary_df=self.df,
            target_metric_col="revenue",
            group_dimension_col="segment",
            aggregation_type=selected_t3.aggregation_type,
            effective_sql=selected_t3.query_sql,
            is_grouped=is_grouped_3,
        )
        state_mgr.record_experiment_executed(f"{investigation_id}_{selected_t3.code}", fingerprint=selected_t3.fingerprint)

        val_t3 = state_mgr.validate_consistency()
        self.assertTrue(val_t3.is_consistent, f"State inconsistency in Iteration 3: {val_t3.errors}")

        probs_t3 = [h.posterior_probability for h in state_mgr.get_active_hypotheses()]
        ent_t3 = compute_shannon_entropy(probs_t3)
        print(f"Posteriors after Turn 3: {dict(zip([h.hypothesis_code for h in state_mgr.get_active_hypotheses()], [round(p, 3) for p in probs_t3]))} (Entropy: {ent_t3:.3f} bits)")

        # Evaluate stopping condition for 3-iteration bound
        stop_eval = StoppingEngine.evaluate_stopping(
            current_entropy=ent_t3,
            initial_entropy=initial_entropy,
            iteration_count=len(state_mgr.get_executed_experiments()),
            max_iterations=3,
            current_posteriors=probs_t3,
        )

        print(f"\nStopping Decision: should_stop={stop_eval.should_stop}, reason={stop_eval.reason}")
        print(f"Stopping Rationale: {stop_eval.rationale}")

        self.assertTrue(stop_eval.should_stop)
        self.assertIn(stop_eval.reason, ["DECISIVE_SIGNAL_RESOLVED", "SUFFICIENTLY_RESOLVED", "MAX_ITERATIONS_REACHED"])

        # Final Verification of State Integrity
        final_report = state_mgr.validate_consistency()
        self.assertTrue(final_report.is_consistent)
        self.assertEqual(len(state_mgr.get_executed_experiments()), 3)
        self.assertTrue(len(state_mgr.state.raw_observations) == 3)

        print("\n" + "=" * 80)
        print("THREE-ITERATION AUTONOMOUS INVESTIGATION SUCCEEDED (100% INTERNAL COHERENCE)")
        print("=" * 80)


if __name__ == "__main__":
    unittest.main()
