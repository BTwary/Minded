"""Suite 15: Phase 9 Automated Causal DAG Discovery (PC Algorithm) & Backdoor Identifiability Test Suite."""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from packages.analytics_core.src.causal.pc_algorithm import PCAlgorithmEngine, DiscoveredCausalDAG
from packages.analytics_core.src.causal.identifiability_gate import CausalIdentifiabilityGate


class TestPhase9CausalDiscovery(unittest.TestCase):
    """Verifies Peter-Clark (PC) algorithm: skeleton discovery, v-structures, Meek rules, and Backdoor adjustment."""

    def test_01_skeleton_discovery_chain(self):
        """Invariant: In chain X -> Y -> Z, X _||_ Z | Y, so skeleton removes edge (X, Z)."""
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1, n)
        y = 2.5 * x + np.random.normal(0, 0.5, n)
        z = -1.8 * y + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})

        dag = PCAlgorithmEngine.discover_causal_dag(df, feature_columns=["X", "Y", "Z"], alpha=0.05)

        # Invariant: X and Z must NOT be adjacent
        self.assertFalse(dag.is_adjacent("X", "Z"))
        # Invariant: X-Y and Y-Z must be adjacent
        self.assertTrue(dag.is_adjacent("X", "Y"))
        self.assertTrue(dag.is_adjacent("Y", "Z"))
        # Invariant: SepSet(X, Z) contains Y
        pair_key = ("X", "Z")
        self.assertIn("Y", dag.separating_sets.get(pair_key, set()))

    def test_02_v_structure_collider_orientation(self):
        """Invariant: In collider X -> Z <- Y, X _||_ Y unconditionally, so Z is a collider: X -> Z and Y -> Z."""
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0, 1, n)
        z = 2.0 * x + 3.0 * y + np.random.normal(0, 0.2, n)
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})

        dag = PCAlgorithmEngine.discover_causal_dag(df, feature_columns=["X", "Y", "Z"], alpha=0.05)

        # Invariant: X and Y are independent unconditionally (not adjacent)
        self.assertFalse(dag.is_adjacent("X", "Y"))
        # Invariant: Z is identified as collider
        self.assertIn("Z", dag.colliders)
        # Invariant: Oriented as X -> Z and Y -> Z
        self.assertTrue(dag.is_directed("X", "Z"))
        self.assertTrue(dag.is_directed("Y", "Z"))

    def test_03_meek_rule1_propagation(self):
        """Invariant: In graph W -> X - Y where W and Y are unshielded, Meek Rule 1 orients X -> Y."""
        np.random.seed(42)
        n = 1000
        w = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, n)
        x = 2.0 * w + 2.0 * u + np.random.normal(0, 0.5, n)
        y = 1.5 * x + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"W": w, "U": u, "X": x, "Y": y})

        dag = PCAlgorithmEngine.discover_causal_dag(df, feature_columns=["W", "U", "X", "Y"], alpha=0.05)

        # Invariant: X -> Y is oriented by Meek Rule 1
        self.assertTrue(dag.is_directed("X", "Y"))

    def test_04_autonomous_causal_identifiability_gate(self):
        """Invariant: identification is claimed only when the discovered graph is fully oriented."""
        # (a) Orientable graph: S->T<-C is a v-structure, C->Y<-D another, so PC orients every edge
        # (Meek rule 1 orients T->Y). The confounder C is then found in the backdoor adjustment set.
        rng = np.random.RandomState(42)
        n = 3000
        s, c, d = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
        t = 1.0 * s + 1.0 * c + rng.normal(0, 0.5, n)
        y = 2.0 * t + 1.5 * c + 1.0 * d + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"S": s, "C": c, "D": d, "T": t, "Y": y})
        res = CausalIdentifiabilityGate().discover_and_evaluate(
            df=df, treatment="T", outcome="Y", feature_columns=["S", "C", "D", "T", "Y"],
        )
        self.assertTrue(res.is_identifiable)
        self.assertIn("C", res.backdoor_adjustment_set)
        self.assertGreater(res.sensitivity_e_value, 1.0)

        # (b) Fully connected triangle (confounder -> treatment -> outcome, confounder -> outcome) has
        # no v-structure, so every orientation is Markov-equivalent and observational data cannot
        # identify the effect. The gate must refuse rather than invent an adjustment set or e-value.
        np.random.seed(42)
        m = 1000
        c2 = np.random.normal(10, 2, m)
        t2 = 1.5 * c2 + np.random.normal(0, 0.5, m)
        y2 = 2.0 * t2 + 3.0 * c2 + np.random.normal(0, 0.5, m)
        tri = pd.DataFrame({"confounder": c2, "treatment": t2, "outcome": y2})
        res2 = CausalIdentifiabilityGate().discover_and_evaluate(
            df=tri, treatment="df.treatment", outcome="df.outcome",
            feature_columns=["confounder", "treatment", "outcome"],
        )
        self.assertFalse(res2.is_identifiable)
        self.assertEqual(res2.status.name, "OBSERVATIONAL_ONLY")
        self.assertEqual(list(res2.backdoor_adjustment_set), [])
        self.assertIsNone(res2.sensitivity_e_value)

    def test_05_universal_ai_fingerprint_miap_generation(self):
        """Invariant: UniversalFingerprintEngine emits valid machine-readable MIAP schema."""
        from packages.analytics_core.src.engines.universal_fingerprint import UniversalFingerprintEngine
        from packages.analytics_core.src.intelligence.prescriptive_optimizer import OptimizationResult

        manifest_hash = "sha256:abcd1234ef5678"
        vectors = {"evidence_strength": 0.95, "causal_certainty": 0.88, "multiverse_robustness": 0.90}
        prescriptions = OptimizationResult(
            optimal_allocation={"Search": 100.0, "Social": 800.0, "Email": 100.0},
            max_expected_return=230.0,
            status="OPTIMAL",
        )

        block = UniversalFingerprintEngine.compile_fingerprint(
            manifest_hash=manifest_hash,
            epistemic_grade="A",
            epistemic_vectors=vectors,
            causal_result=None,
            prescriptive_result=prescriptions,
            verdict_summary="Revenue optimization complete.",
        )

        self.assertIn("```aa-os-fingerprint-v1", block)
        self.assertIn("sha256:abcd1234ef5678", block)
        self.assertIn("epistemic_grade\": \"A\"", block)
        self.assertIn("optimal_allocation", block)
        self.assertIn("CONSTRAINT:", block)


def main():
    print("=" * 80, flush=True)
    print("RUNNING SUITE 15: PHASE 9 AUTOMATED CAUSAL DAG DISCOVERY (PC ALGORITHM) & MIAP", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase9CausalDiscovery)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("PHASE 9 CAUSAL DISCOVERY & MIAP: ALL 5 TESTS PASSED (100%)", flush=True)
        print("=" * 80, flush=True)
        return 0
    else:
        print("PHASE 9 CAUSAL DISCOVERY: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
