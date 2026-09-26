import numpy as np
import pandas as pd
from packages.analytics_core.src.engines.method_selection import MethodRegistry
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.intelligence.universal_specialized import run_forecast_analysis, run_predictive_risk_analysis

class Sem:
    primary_dataset_name = "subscriptions"
    target_metric_col = "churned"
    group_dimension_col = "plan_tier"
    time_col = "signup_date"
    table_grain = "customer_id"
    secondary_metric_col = None
    churn_event_col = "churned"
    metric_definition = None

def test_question_classes_are_semantic_and_distinct():
    df = pd.DataFrame({"customer_id":[1,2,3,4],"plan_tier":["Basic","Pro","Basic","Pro"],"churned":[0,1,0,1],"signup_date":["2025-01-01"]*4})
    sem=Sem()
    assert UniversalQuestionCompiler.compile("Which customer segments exist?", semantic=sem, df=df).task == "SEGMENTATION"
    assert UniversalQuestionCompiler.compile("Will sales likely increase next quarter?", semantic=sem, df=df).task == "FORECAST"
    assert UniversalQuestionCompiler.compile("Clean this dataset and show exactly what you changed.", semantic=sem, df=df).task == "DATA_QUALITY"

def test_method_registry_is_queryable_and_declares_constraints():
    caps=MethodRegistry.for_problem_class("PREDICTION")
    assert any(c.code=="binary_risk_prediction" for c in caps)
    ok, errors = MethodRegistry.validate("binary_risk_prediction", row_count=100, roles={"binary_target"})
    assert ok and not errors

def test_data_readiness_exposes_nonfatal_quality_evidence():
    rng=np.random.default_rng(1)
    df=pd.DataFrame({"id":range(100),"score":np.r_[rng.normal(size=99),20.0],"target":[0]*95+[1]*5})
    r=DataQualityGate.evaluate_fitness(df, "t", min_sample_size=5)
    assert r.row_count==100
    assert "score" in r.outlier_columns
    assert "target" in r.class_imbalance
    assert r.fitness_verdict in {"FIT","CAUTION","UNFIT"}

def test_forecast_specialist_uses_out_of_sample_backtesting():
    dates=pd.date_range("2023-01-01", periods=24, freq="MS")
    values=np.arange(24,dtype=float)+np.sin(np.arange(24)/2.0)
    df=pd.DataFrame({"date":dates,"sales":values})
    out=run_forecast_analysis(df=df,time_column="date",metric_column="sales",question="Will sales increase next quarter?",dataset_name="sales",dataset_fingerprints={"sha256":"x"})
    assert out["status"]=="COMPLETED"
    assert out["result"]["selection_metric"]=="out_of_sample_MAE"

def test_prediction_specialist_outputs_validation_before_ranking():
    rng=np.random.default_rng(4)
    n=100
    x=rng.normal(size=n)
    y=(x+rng.normal(scale=0.6,size=n)>0).astype(int)
    df=pd.DataFrame({"customer_id":range(n),"signal":x,"churned":y})
    out=run_predictive_risk_analysis(df=df,target_column="churned",question="Which customers are most likely to churn next month?",dataset_name="customers",positive_class=1)
    assert out["status"]=="COMPLETED"
    assert out["result"]["validation"]
    assert out["result"]["ranked_active_rows"]
