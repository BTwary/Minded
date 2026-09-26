import pandas as pd
from packages.analytics_core.src.intelligence.universal_specialized import run_specialized_analysis, run_predictive_risk_analysis
from packages.analytics_core.src.intelligence.universal_question_planner import AutomaticClusteringEngine, MathematicalReconciliationEngine, GovernanceRiskEngine
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate


def test_reconciliation_finds_exact_identity_and_discrepancy():
    df = pd.DataFrame({
        'Quantity': [2, 3, 1],
        'UnitPrice': [10.0, 5.0, 20.0],
        'Discount': [0.0, 0.2, 0.0],
        'TotalAmount': [20.0, 12.0, 19.0],
    })
    result = MathematicalReconciliationEngine.analyze(df)
    assert result['status'] == 'COMPLETED'
    assert result['best_identity']['discrepancy_rows'] == 1


def test_clustering_reports_stability():
    rows = []
    for i in range(30):
        rows.append({'spend': 10 + i * 0.1, 'tickets': 2 + (i % 2)})
    for i in range(30):
        rows.append({'spend': 100 + i * 0.1, 'tickets': 20 + (i % 2)})
    df = pd.DataFrame(rows)
    result = AutomaticClusteringEngine.analyze(df, requested_columns=['spend', 'tickets'])
    assert result['status'] == 'COMPLETED'
    assert result['selected_k'] >= 2
    assert 0.0 <= result['stability_ari'] <= 1.0
    assert 0.0 <= result['silhouette_score'] <= 1.0


def test_governance_scan_does_not_claim_legal_compliance():
    df = pd.DataFrame({'customer_id': [1,2], 'email': ['a@x.com','b@x.com'], 'income': [10,20]})
    result = GovernanceRiskEngine.assess(df)
    text = ' '.join(result.get('risks', [])) + ' ' + ' '.join(result.get('recommendations', []))
    assert result['pii_columns']
    assert 'legal' in text.lower() or 'compliance' in text.lower()


def test_specialized_trace_contains_source_identity():
    df = pd.DataFrame({'Quantity':[1], 'UnitPrice':[10.0], 'Discount':[0.0], 'TotalAmount':[10.0]})
    quality = DataQualityGate.evaluate_fitness(df, 'sales')
    result = run_specialized_analysis(
        task='RECONCILIATION', question='Check Quantity UnitPrice Discount TotalAmount discrepancies',
        df=df, dataset_name='sales', dataset_fingerprints={'sales':'abc123'},
        dataset_versions={'sales': {'dataset_id':'d1','dataset_version':1,'file_name':'sales.csv'}},
        quality_assessment=quality,
    )
    assert result['trace']['source']['dataset_versions']['sales']['dataset_version'] == 1
    assert result['trace']['canonical_hash']


def test_predictive_risk_scores_current_active_rows_without_using_ids():
    rows = []
    for i in range(120):
        plan = "Pro" if i % 2 else "Basic"
        tickets = (6 if i % 3 == 0 else 1)
        churn = int(tickets >= 5)
        rows.append({"customer_id": i + 1, "plan_tier": plan, "support_tickets": tickets, "churned": churn})
    df = pd.DataFrame(rows)
    result = run_predictive_risk_analysis(
        df=df, target_column="churned", time_column=None, question="Which active customers are at highest churn risk?",
        dataset_name="subscriptions", dataset_fingerprints={"subscriptions":"hash"},
        dataset_versions={"subscriptions": {"dataset_version": 1, "file_name": "subscriptions.csv"}},
        positive_class=1,
    )
    assert result["status"] == "COMPLETED"
    # A specialist is a computational engine, never an epistemic authority: it
    # never independently produces a scientific verdict outside the canonical
    # claim gate, so this (like run_forecast_analysis) always reports
    # verdict=INCONCLUSIVE / confidence=None regardless of how well it ranked.
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["confidence"] is None
    assert len(result["result"]["ranked_active_rows"]) > 0
    assert result["trace"]["source"]["dataset_versions"]["subscriptions"]["dataset_version"] == 1
